"""
Local BM25 keyword retrieval over the same chunks indexed in ChromaDB.

Persisted as a pickle next to the vector store (rag/vectorstore/bm25_index.pkl)
so it survives restarts, same as the vector index.
"""
import pickle
import re
from typing import List, Dict, Any

from rag.config import BM25_INDEX_PATH

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self):
        self._bm25 = None
        self._chunks: List[Dict[str, Any]] = []

    def build(self, chunks: List[Dict[str, Any]]):
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        return self

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self._bm25 is None or not self._chunks:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        hits = []
        for idx in ranked_idx:
            if scores[idx] <= 0:
                continue
            chunk = self._chunks[idx]
            hits.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(scores[idx]),
                "metadata": {
                    "source": chunk.get("source", ""),
                    "document": chunk.get("document", ""),
                    "section": chunk.get("section", ""),
                    "page": chunk.get("page", 0),
                    "topic": chunk.get("topic", ""),
                },
            })
        return hits

    # ------------------------------------------------------------------
    def save(self, path: str = BM25_INDEX_PATH):
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, f)

    def load(self, path: str = BM25_INDEX_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]
        return self

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and bool(self._chunks)
