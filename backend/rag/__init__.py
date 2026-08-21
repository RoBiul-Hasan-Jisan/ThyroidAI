"""
ThyroidAI RAG subsystem.

Everything under this package runs 100% locally / offline:
- embeddings.py    -> sentence-transformers (BAAI/bge-small-en-v1.5)
- vector_store.py  -> ChromaDB (local persistent client)
- bm25_retriever.py-> rank-bm25 keyword search
- hybrid_retriever.py -> Reciprocal Rank Fusion (RRF) + Maximal Marginal Relevance (MMR)
- chunking.py      -> section-aware text chunking
- ingestion.py      -> PDF -> chunks -> embeddings -> ChromaDB + BM25 index
- query_builder.py -> turns a patient profile + SHAP factors into retrieval queries
- evaluate.py      -> Precision@K / Recall@K / MRR against a small hand-labeled test set

No file in this package talks to OpenAI, Gemini, Claude, OpenRouter, or any
paid API. The only outbound network call anywhere in the RAG pipeline is to
a locally-running Ollama daemon (see app/core/ollama_client.py), which is
itself a local process, not a cloud service.
"""
