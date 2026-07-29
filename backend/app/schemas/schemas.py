from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class PatientInput(BaseModel):
    Age: float = Field(..., ge=0, le=120)
    Gender: str
    Smoking: str
    Hx_Smoking: str = Field(..., alias="Hx Smoking")
    Hx_Radiothreapy: str = Field(..., alias="Hx Radiothreapy")
    Thyroid_Function: str = Field(..., alias="Thyroid Function")
    Physical_Examination: str = Field(..., alias="Physical Examination")
    Adenopathy: str
    Pathology: str
    Focality: str
    Risk: str
    T: str
    N: str
    M: str
    Stage: str
    Response: str

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Age": 40, "Gender": "F", "Smoking": "No", "Hx Smoking": "No",
                "Hx Radiothreapy": "No", "Thyroid Function": "Euthyroid",
                "Physical Examination": "Multinodular goiter", "Adenopathy": "No",
                "Pathology": "Papillary", "Focality": "Uni-Focal", "Risk": "Low",
                "T": "T1a", "N": "N0", "M": "M0", "Stage": "I", "Response": "Excellent"
            }
        }


class ExplanationFactor(BaseModel):
    feature: str
    value: str
    impact: float
    direction: str  # "increases" | "decreases"


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    confidence: str
    model_used: str
    explanation: List[str]
    shap_factors: List[ExplanationFactor]
    probabilities: Dict[str, float]


class ModelInfoResponse(BaseModel):
    best_model: str
    best_model_type: str
    selection_priority: List[str]
    all_models: List[Dict]
    best_model_metrics: Dict
    roc_curves: Dict


class FeaturesResponse(BaseModel):
    feature_cols: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    categorical_options: Dict[str, List[str]]
    target_classes: List[str]
