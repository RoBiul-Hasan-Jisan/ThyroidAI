from fastapi import APIRouter
import traceback

from app.schemas.rag_schemas import RagExplainRequest, RagExplainResponse
from app.core.rag_engine import generate_rag_explanation, is_rag_configured
from app.core.ollama_client import get_ollama_client
from rag.config import RAG_ENABLED, DISCLAIMER
from rag.hybrid_retriever import get_hybrid_retriever

router = APIRouter()


@router.post("/explain", response_model=RagExplainResponse)
def rag_explain(request: RagExplainRequest):
    """
    Generate a grounded, cited medical-context explanation for an existing
    ML prediction + SHAP result. This endpoint is fully additive: it never
    touches /api/predict, and any failure here (Ollama down, no documents
    ingested, etc.) is returned as a structured `status` field rather than
    a 500 -- a RAG failure must never look like a prediction failure.
    """
    try:
        result = generate_rag_explanation(
            patient=request.patient,
            prediction=request.prediction.dict(),
            shap_factors=request.shap_factors,
        )
        return result
    except Exception as e:
        # Absolute last-resort guard: even an unexpected bug in the RAG
        # pipeline degrades to "unavailable", never a hard error.
        print(f" RAG explain failed: {e}")
        print(traceback.format_exc())
        return {
            "status": "rag_unavailable",
            "evidence": [],
            "queries_used": [],
            "disclaimer": DISCLAIMER,
            "limitations": "AI medical context is temporarily unavailable.",
        }


@router.get("/status")
def rag_status():
    """Lightweight status check the frontend can poll before/instead of calling /explain."""
    ollama_ok = False
    try:
        ollama_ok = get_ollama_client().health_check()
    except Exception:
        ollama_ok = False

    try:
        stats = get_hybrid_retriever().vector_store.get_stats()
        num_chunks = stats.get("num_chunks", 0)
    except Exception:
        num_chunks = 0

    return {
        "rag_enabled": RAG_ENABLED,
        "documents_ingested": num_chunks > 0,
        "num_chunks": num_chunks,
        "ollama_available": ollama_ok,
        "ready": is_rag_configured() and ollama_ok,
    }
