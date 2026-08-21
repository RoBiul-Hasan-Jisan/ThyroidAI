from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class RagPredictionInput(BaseModel):
    prediction: str
    probability: float


class RagShapFactor(BaseModel):
    feature: str
    value: Optional[str] = None
    impact: float
    direction: str


class RagExplainRequest(BaseModel):
    patient: Dict[str, Any]
    prediction: RagPredictionInput
    shap_factors: List[RagShapFactor] = []


class EvidenceChunk(BaseModel):
    text: str
    source: str
    document: str
    section: str
    page: int
    score: float


class RagExplainResponse(BaseModel):
    status: str  # "completed" | "rag_unavailable" | "no_evidence"
    summary: Optional[str] = None
    clinical_context: Optional[str] = None
    evidence: List[EvidenceChunk] = []
    retrieval_method: str = "BM25 + Vector + RRF + MMR"
    limitations: Optional[str] = None
    disclaimer: str
    queries_used: List[str] = []
