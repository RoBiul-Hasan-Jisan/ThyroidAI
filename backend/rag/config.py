"""
Central configuration for the RAG subsystem. Everything is read from
environment variables (see backend/.env.example) with sane local-only
defaults, so the module works out of the box on a laptop with no keys.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
RAG_DIR = os.path.join(BASE_DIR, "rag")

# ----------------------------------------------------------------------
# Documents / storage
# ----------------------------------------------------------------------
DOCUMENTS_DIR = os.path.join(RAG_DIR, "documents")
VECTORSTORE_DIR = os.environ.get(
    "RAG_VECTORSTORE_DIR", os.path.join(RAG_DIR, "vectorstore")
)
CHROMA_COLLECTION_NAME = os.environ.get("RAG_COLLECTION_NAME", "thyroid_knowledge")
BM25_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "bm25_index.pkl")
CHUNK_METADATA_PATH = os.path.join(VECTORSTORE_DIR, "chunks_metadata.jsonl")

# ----------------------------------------------------------------------
# Feature flag
# ----------------------------------------------------------------------
RAG_ENABLED = os.environ.get("RAG_ENABLED", "true").lower() in ("1", "true", "yes")

# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------
CHUNK_TOKEN_SIZE = int(os.environ.get("RAG_CHUNK_TOKEN_SIZE", "650"))
CHUNK_TOKEN_OVERLAP = int(os.environ.get("RAG_CHUNK_TOKEN_OVERLAP", "80"))

# ----------------------------------------------------------------------
# Embeddings (local, sentence-transformers)
# ----------------------------------------------------------------------
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------
RAG_TOP_K_VECTOR = int(os.environ.get("RAG_TOP_K_VECTOR", "10"))
RAG_TOP_K_BM25 = int(os.environ.get("RAG_TOP_K_BM25", "10"))
RAG_RRF_K = int(os.environ.get("RAG_RRF_K", "60"))
RAG_TOP_K_FINAL = int(os.environ.get("RAG_TOP_K_FINAL", "5"))
RAG_MMR_LAMBDA = float(os.environ.get("RAG_MMR_LAMBDA", "0.5"))

# ----------------------------------------------------------------------
# Local LLM (Ollama - no API key, no cloud provider)
# ----------------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "60"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "600"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

DISCLAIMER = (
    "This system is for research and educational purposes only. "
    "It is not a certified clinical decision-support system and does not "
    "replace professional medical judgment."
)
