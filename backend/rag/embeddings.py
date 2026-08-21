"""
Local embedding model wrapper.

Uses sentence-transformers running entirely on-device (CPU is fine). No
network call is made at inference time; the model weights are downloaded
once from the Hugging Face Hub the first time this process runs (standard
sentence-transformers caching behaviour) and then reused from local cache.
There is no API key and no per-call cost.
"""
from functools import lru_cache
from typing import List

import numpy as np

from rag.config import RAG_EMBEDDING_MODEL


class LocalEmbedder:
    """Thin, reusable wrapper around a local sentence-transformers model."""

    def __init__(self, model_name: str = RAG_EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of chunk texts. Returns (n, dim) float32 array."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns a (dim,) float32 vector."""
        embedding = self._model.encode(
            [text],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return embedding.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


@lru_cache(maxsize=1)
def get_embedder() -> LocalEmbedder:
    """Process-wide singleton so the model is loaded into memory once."""
    return LocalEmbedder()
