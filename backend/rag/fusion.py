"""
Manual implementations of Reciprocal Rank Fusion (RRF) and Maximal Marginal
Relevance (MMR). No paid reranking service, no external library -- both are
~20 lines of numpy/plain Python each.
"""
from typing import List, Dict, Any
import numpy as np


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Combine multiple ranked result lists (e.g. BM25 hits, vector hits) into
    a single ranking using RRF:

        score(chunk) = sum over lists containing chunk of 1 / (k + rank)

    `ranked_lists` is a list of lists of hit dicts (each already sorted best
    -> worst, each hit must have a "chunk_id"). Returns a single list of
    hit dicts sorted by fused score descending, deduplicated by chunk_id,
    with the fused score stored under "rrf_score" and the original hit
    payload (text/metadata) preserved from the first list it appeared in.
    """
    fused_scores: Dict[str, float] = {}
    chunk_payloads: Dict[str, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, hit in enumerate(ranked_list, start=1):
            chunk_id = hit["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in chunk_payloads:
                chunk_payloads[chunk_id] = hit

    fused = []
    for chunk_id, score in fused_scores.items():
        payload = dict(chunk_payloads[chunk_id])
        payload["rrf_score"] = score
        fused.append(payload)

    fused.sort(key=lambda h: h["rrf_score"], reverse=True)
    return fused


def maximal_marginal_relevance(
    candidates: List[Dict[str, Any]],
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int = 5,
    lambda_mult: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Greedy MMR selection over `candidates` (already RRF-fused, ranked hit
    dicts) to trade off relevance to the query against redundancy with
    chunks already selected.

        MMR = argmax_i [ lambda * sim(q, d_i) - (1 - lambda) * max_j sim(d_i, d_j) ]

    `candidate_embeddings` must be a (len(candidates), dim) array aligned
    1:1 with `candidates`, and `query_embedding` a (dim,) array. Embeddings
    are assumed to already be L2-normalized (as produced by
    rag/embeddings.py), so dot product == cosine similarity.
    """
    if not candidates:
        return []

    n = len(candidates)
    top_k = min(top_k, n)

    query_sims = candidate_embeddings @ query_embedding  # (n,)
    selected_idx: List[int] = []
    remaining = set(range(n))

    while len(selected_idx) < top_k and remaining:
        best_idx = None
        best_score = -float("inf")

        for i in remaining:
            relevance = query_sims[i]
            if selected_idx:
                redundancy = max(
                    float(candidate_embeddings[i] @ candidate_embeddings[j])
                    for j in selected_idx
                )
            else:
                redundancy = 0.0

            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected_idx.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected_idx:
        item = dict(candidates[idx])
        item["mmr_score"] = float(query_sims[idx])
        results.append(item)
    return results
