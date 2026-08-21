"""
Evaluates RETRIEVAL quality only (not generation) against a small,
manually-labeled test set of (question -> expected source document(s)).

This intentionally does not grade the LLM's generated text -- grounded
generation quality is a separate, harder problem. This measures whether
the hybrid retriever surfaces evidence from the right document(s) for a
handful of representative thyroid-cancer questions.

Run as:

    python -m rag.evaluate

The test set below references *document names* (the filename stem you
gave a PDF under backend/rag/documents/), not fabricated document content.
Edit TEST_SET to match whatever you've actually ingested before treating
the numbers as meaningful -- out of the box, with no documents ingested,
this will correctly report zero precision/recall everywhere.
"""
from typing import List, Dict, Any

from rag.hybrid_retriever import get_hybrid_retriever

# Each entry: a natural-language question a clinician/patient might ask,
# and the set of source *document names* (filename stem, no extension)
# that should legitimately answer it. Fill in `expected_documents` to match
# the PDFs you've actually placed under rag/documents/ before relying on
# these numbers -- left as an empty list, an entry contributes 0 to every
# metric rather than silently being skipped.
TEST_SET: List[Dict[str, Any]] = [
    {"question": "What does an excellent response mean?", "expected_documents": []},
    {"question": "What factors influence recurrence risk?", "expected_documents": []},
    {"question": "What is TNM staging?", "expected_documents": []},
    {"question": "What is papillary thyroid cancer?", "expected_documents": []},
    {"question": "How is recurrence monitored?", "expected_documents": []},
]


def _precision_recall_mrr(
    retrieved_documents: List[str],
    expected_documents: List[str],
    k: int,
) -> Dict[str, float]:
    if not expected_documents:
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0}

    top_k = retrieved_documents[:k]
    expected_set = set(expected_documents)

    hits = [1 if doc in expected_set else 0 for doc in top_k]
    precision = sum(hits) / k if k else 0.0
    recall = len(set(top_k) & expected_set) / len(expected_set)

    mrr = 0.0
    for rank, doc in enumerate(retrieved_documents, start=1):
        if doc in expected_set:
            mrr = 1.0 / rank
            break

    return {"precision": precision, "recall": recall, "mrr": mrr}


def evaluate_retrieval(top_k: int = 5) -> Dict[str, Any]:
    retriever = get_hybrid_retriever()
    per_question = []

    for item in TEST_SET:
        hits = retriever.retrieve([item["question"]], top_k_final=top_k)
        retrieved_documents = [h["metadata"].get("document", "") for h in hits]

        metrics = _precision_recall_mrr(retrieved_documents, item["expected_documents"], top_k)
        per_question.append({
            "question": item["question"],
            "retrieved_documents": retrieved_documents,
            "expected_documents": item["expected_documents"],
            **metrics,
        })

    n = len(per_question) or 1
    summary = {
        "precision_at_k": sum(q["precision"] for q in per_question) / n,
        "recall_at_k": sum(q["recall"] for q in per_question) / n,
        "mrr": sum(q["mrr"] for q in per_question) / n,
        "k": top_k,
        "num_questions": len(per_question),
    }
    return {"summary": summary, "per_question": per_question}


def main():
    results = evaluate_retrieval()
    summary = results["summary"]

    print(f"Evaluated {summary['num_questions']} questions @ K={summary['k']}\n")
    for q in results["per_question"]:
        print(f"- {q['question']}")
        print(f"    expected: {q['expected_documents'] or '(not labeled)'}")
        print(f"    retrieved: {q['retrieved_documents']}")
        print(f"    precision={q['precision']:.2f}  recall={q['recall']:.2f}  mrr={q['mrr']:.2f}\n")

    print(f"Precision@{summary['k']}: {summary['precision_at_k']:.3f}")
    print(f"Recall@{summary['k']}:    {summary['recall_at_k']:.3f}")
    print(f"MRR:            {summary['mrr']:.3f}")


if __name__ == "__main__":
    main()
