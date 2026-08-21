from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import api, rag
from app.core.model_loader import get_bundle

app = FastAPI(
    title="ThyroidAI API",
    description="Explainable ML/DL API for Differentiated Thyroid Cancer Recurrence prediction.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # public demo system, no auth — open CORS is intentional
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model_on_startup():
    # Warm the singleton model bundle (models, preprocessor, SHAP explainer)
    # once, at process start, instead of lazily on first request.
    get_bundle()


@app.get("/")
def root():
    return {"service": "ThyroidAI API", "status": "running", "docs": "/docs"}


app.include_router(api.router, prefix="/api", tags=["ThyroidAI"])

# RAG is fully additive: a separate router, separate prefix, and it never
# imports from or modifies anything the /api/predict path depends on. RAG
# is not loaded/warmed at startup (unlike the ML model bundle above) -- the
# embedding model and vector store are only touched lazily, on first use of
# /api/rag/explain, so a machine with no Ollama/RAG setup still boots and
# serves predictions normally.
app.include_router(rag.router, prefix="/api/rag", tags=["RAG"])
