"""
Orchestrates the RAG pipeline for a single /api/rag/explain request:

    patient + prediction + SHAP factors
        -> query_builder.build_queries()
        -> hybrid_retriever.retrieve()   (BM25 + Vector -> RRF -> MMR)
        -> grounded prompt
        -> ollama_client.generate()
        -> structured, cited response

This module never touches the ML prediction pipeline (model_loader.py) --
it only ever *consumes* a prediction + SHAP factors that the caller already
computed. Any failure here (no documents ingested, Ollama down, etc.) is
caught and returned as a structured "unavailable" response; it never raises
in a way that could take down /api/predict.
"""
from typing import Dict, Any, List

from rag.config import RAG_ENABLED, RAG_TOP_K_FINAL, DISCLAIMER
from rag.query_builder import build_queries
from rag.hybrid_retriever import get_hybrid_retriever
from app.core.ollama_client import get_ollama_client

SYSTEM_PROMPT = """You are a medical information assistant for ThyroidAI.

The recurrence prediction is produced by a separate machine-learning model. Never change, recalculate, or override the prediction.

Use ONLY the retrieved evidence supplied below. Do not invent medical facts, statistics, sources, or citations.

Structure your answer into exactly these sections:
1. ML Model Prediction (restate it plainly, do not alter it)
2. SHAP Explanation (in plain language, what drove this prediction)
3. Medical Context (grounded strictly in the retrieved evidence; cite each claim with the [Source N] tags provided)
4. Limitations (state clearly if the retrieved evidence is incomplete or insufficient for any part of the picture)

Do not diagnose the patient. Do not prescribe treatment or recommend medication changes. Do not claim the model prediction is a clinical diagnosis. Use cautious, hedged medical language. Every medical claim must be traceable to the retrieved evidence."""


def _format_evidence_block(evidence: List[Dict[str, Any]]) -> str:
    lines = []
    for i, chunk in enumerate(evidence, start=1):
        meta = chunk.get("metadata", {})
        lines.append(
            f"[Source {i}] {meta.get('source', 'unknown')} "
            f"({meta.get('section', 'unknown section')}, p.{meta.get('page', '?')})\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(lines)


def _build_prompt(
    patient: Dict[str, Any],
    prediction: Dict[str, Any],
    shap_factors: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
) -> str:
    shap_lines = "\n".join(
        f"- {f.get('feature')} = {f.get('value', patient.get(f.get('feature'), 'N/A'))} "
        f"({f.get('direction')} risk, impact={f.get('impact')})"
        for f in shap_factors
    ) or "- No SHAP factors provided."

    evidence_block = _format_evidence_block(evidence) if evidence else "(no evidence retrieved)"

    return f"""{SYSTEM_PROMPT}

--- ML PREDICTION ---
Prediction: {prediction.get('prediction')}
Probability of recurrence: {prediction.get('probability')}

--- SHAP FACTORS ---
{shap_lines}

--- RETRIEVED EVIDENCE ---
{evidence_block}

--- TASK ---
Write the four-section explanation described above, grounded strictly in the retrieved evidence. Keep it concise (roughly 150-250 words)."""


def is_rag_configured() -> bool:
    if not RAG_ENABLED:
        return False
    try:
        return get_hybrid_retriever().is_ready
    except Exception:
        return False


def generate_rag_explanation(
    patient: Dict[str, Any],
    prediction: Dict[str, Any],
    shap_factors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Returns a dict matching RagExplainResponse. Never raises: every failure
    mode (RAG disabled, no ingested documents, Ollama unreachable, model
    missing, timeout, empty response) is caught and converted into a
    status field the frontend can render distinctly.
    """
    if not RAG_ENABLED:
        return {
            "status": "rag_unavailable",
            "evidence": [],
            "queries_used": [],
            "disclaimer": DISCLAIMER,
            "limitations": "RAG is disabled on this deployment (RAG_ENABLED=false).",
        }

    queries = build_queries(patient, [f if isinstance(f, dict) else f.dict() for f in shap_factors])

    try:
        retriever = get_hybrid_retriever()
        if not retriever.is_ready:
            return {
                "status": "no_evidence",
                "evidence": [],
                "queries_used": queries,
                "disclaimer": DISCLAIMER,
                "limitations": "No medical reference documents have been ingested yet. "
                                "Run `python -m rag.ingestion` after adding source PDFs "
                                "under backend/rag/documents/.",
            }
        evidence = retriever.retrieve(queries, top_k_final=RAG_TOP_K_FINAL)
    except Exception:
        return {
            "status": "rag_unavailable",
            "evidence": [],
            "queries_used": queries,
            "disclaimer": DISCLAIMER,
            "limitations": "Evidence retrieval is temporarily unavailable.",
        }

    if not evidence:
        return {
            "status": "no_evidence",
            "evidence": [],
            "queries_used": queries,
            "disclaimer": DISCLAIMER,
            "limitations": "No relevant evidence was found in the ingested documents for this patient profile.",
        }

    evidence_payload = [
        {
            "text": chunk["text"],
            "source": chunk["metadata"].get("source", "unknown"),
            "document": chunk["metadata"].get("document", "unknown"),
            "section": chunk["metadata"].get("section", "unknown"),
            "page": int(chunk["metadata"].get("page", 0)),
            "score": round(float(chunk.get("mmr_score", chunk.get("rrf_score", 0.0))), 4),
        }
        for chunk in evidence
    ]

    ollama = get_ollama_client()
    if not ollama.health_check():
        return {
            "status": "rag_unavailable",
            "evidence": evidence_payload,
            "queries_used": queries,
            "disclaimer": DISCLAIMER,
            "limitations": "AI generation is currently unavailable (Ollama not reachable or "
                            "model not pulled). Retrieved medical evidence is still shown above.",
        }

    prompt = _build_prompt(patient, prediction, [f if isinstance(f, dict) else f.dict() for f in shap_factors], evidence)
    result = ollama.generate(prompt)

    if not result.ok:
        return {
            "status": "rag_unavailable",
            "evidence": evidence_payload,
            "queries_used": queries,
            "disclaimer": DISCLAIMER,
            "limitations": "AI generation is currently unavailable. Retrieved medical evidence "
                            "is still shown above.",
        }

    return {
        "status": "completed",
        "summary": f"{prediction.get('prediction')} recurrence predicted "
                   f"({round(float(prediction.get('probability', 0)) * 100, 2)}% probability).",
        "clinical_context": result.text,
        "evidence": evidence_payload,
        "queries_used": queries,
        "disclaimer": DISCLAIMER,
        "limitations": None,
    }
