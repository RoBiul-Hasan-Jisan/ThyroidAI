"""
Ties BM25 + dense vector search together via RRF, then diversifies the
fused candidate set with MMR.

Pipeline (per docs/README):
    BM25 results  ┐
                   ├─> RRF ─> MMR ─> top-K diverse evidence chunks
    Vector results ┘
"""
import os
from typing import List, Dict, Any
from functools import lru_cache

import numpy as np

from rag.config import (
    RAG_TOP_K_VECTOR,
    RAG_TOP_K_BM25,
    RAG_RRF_K,
    RAG_TOP_K_FINAL,
    RAG_MMR_LAMBDA,
    BM25_INDEX_PATH,
)
from rag.embeddings import get_embedder
from rag.vector_store import VectorStore
from rag.bm25_retriever import BM25Retriever
from rag.fusion import reciprocal_rank_fusion, maximal_marginal_relevance


class HybridRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever()
        self._bm25_loaded = False
        if os.path.exists(BM25_INDEX_PATH):
            try:
                self.bm25.load(BM25_INDEX_PATH)
                self._bm25_loaded = True
            except Exception:
                self._bm25_loaded = False

    @property
    def is_ready(self) -> bool:
        stats = self.vector_store.get_stats()
        return stats["num_chunks"] > 0 and self._bm25_loaded

    def retrieve_for_query(
        self,
        query: str,
        top_k_vector: int = RAG_TOP_K_VECTOR,
        top_k_bm25: int = RAG_TOP_K_BM25,
    ) -> List[List[Dict[str, Any]]]:
        """Run BM25 + vector search for a single query, return the two ranked lists."""
        embedder = get_embedder()
        query_vec = embedder.embed_query(query)

        vector_hits = self.vector_store.search(query_vec, top_k=top_k_vector)
        bm25_hits = self.bm25.search(query, top_k=top_k_bm25) if self._bm25_loaded else []
        return [vector_hits, bm25_hits]

    def retrieve(
        self,
        queries: List[str],
        top_k_final: int = RAG_TOP_K_FINAL,
        rrf_k: int = RAG_RRF_K,
        mmr_lambda: float = RAG_MMR_LAMBDA,
    ) -> List[Dict[str, Any]]:
        """
        Full hybrid pipeline across one or more retrieval queries (typically
        one per important SHAP factor):

          for each query: BM25 hits + vector hits
          -> RRF fuse ALL ranked lists from ALL queries into one ranking
          -> MMR down-select to `top_k_final` diverse chunks
        """
        if not queries:
            return []

        all_ranked_lists: List[List[Dict[str, Any]]] = []
        for query in queries:
            vector_hits, bm25_hits = self.retrieve_for_query(query)
            if vector_hits:
                all_ranked_lists.append(vector_hits)
            if bm25_hits:
                all_ranked_lists.append(bm25_hits)

        if not all_ranked_lists:
            return []

        fused = reciprocal_rank_fusion(all_ranked_lists, k=rrf_k)
        if not fused:
            return []

        # MMR needs embeddings for the fused candidates and for the combined
        # query. We embed the concatenation of all queries as a single
        # "intent" vector for the relevance term, and re-embed candidate
        # texts (cheap: this is capped at a couple dozen chunks).
        embedder = get_embedder()
        combined_query = " ".join(queries)
        query_embedding = embedder.embed_query(combined_query)

        # Cap how many fused candidates we bother re-embedding for MMR.
        candidate_pool = fused[: max(top_k_final * 4, 20)]
        candidate_texts = [c["text"] for c in candidate_pool]
        candidate_embeddings = embedder.embed_documents(candidate_texts)

        diversified = maximal_marginal_relevance(
            candidate_pool,
            query_embedding,
            candidate_embeddings,
            top_k=top_k_final,
            lambda_mult=mmr_lambda,
        )
        return diversified


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever()


def reset_hybrid_retriever_cache():
    """Call after re-ingestion so a running FastAPI process picks up the new index."""
    get_hybrid_retriever.cache_clear()
