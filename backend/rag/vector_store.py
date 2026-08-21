"""
Local, persistent ChromaDB vector store.

Storage lives at backend/rag/vectorstore/ on disk (see rag/config.py) and
survives process restarts -- ingestion is a separate, explicit step
(`python -m rag.ingestion`), never something that happens implicitly on
FastAPI startup.
"""
from typing import List, Dict, Any, Optional
import numpy as np

from rag.config import VECTORSTORE_DIR, CHROMA_COLLECTION_NAME


class VectorStore:
    def __init__(
        self,
        persist_dir: str = VECTORSTORE_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        import chromadb

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = None

    # ------------------------------------------------------------------
    def create_collection(self, reset: bool = False):
        """Create (or fetch) the collection. If reset=True, drop it first."""
        if reset:
            self.delete_collection()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def _get_or_create(self):
        if self._collection is None:
            self.create_collection(reset=False)
        return self._collection

    # ------------------------------------------------------------------
    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        batch_size: int = 256,
    ):
        """
        chunks: list of chunk dicts (must include chunk_id/text/source/
                document/section/page/topic).
        embeddings: (n, dim) array aligned 1:1 with `chunks`.
        """
        collection = self._get_or_create()
        n = len(chunks)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = chunks[start:end]
            collection.add(
                ids=[c["chunk_id"] for c in batch],
                documents=[c["text"] for c in batch],
                embeddings=embeddings[start:end].tolist(),
                metadatas=[
                    {
                        "source": c.get("source", ""),
                        "document": c.get("document", ""),
                        "section": c.get("section", ""),
                        "page": int(c.get("page", 0)) if c.get("page") is not None else 0,
                        "topic": c.get("topic", ""),
                    }
                    for c in batch
                ],
            )

    # ------------------------------------------------------------------
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        collection = self._get_or_create()
        if collection.count() == 0:
            return []

        result = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, collection.count()),
        )

        hits = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            # Chroma returns cosine *distance*; convert to a similarity score.
            score = 1.0 - float(dist)
            hits.append({
                "chunk_id": chunk_id,
                "text": text,
                "score": score,
                "metadata": meta or {},
            })
        return hits

    # ------------------------------------------------------------------
    def delete_collection(self):
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None

    def get_stats(self) -> Dict[str, Any]:
        collection = self._get_or_create()
        return {
            "collection_name": self.collection_name,
            "num_chunks": collection.count(),
            "persist_dir": self.persist_dir,
        }
