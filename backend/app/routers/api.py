from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.core.model_loader import get_bundle
from app.core.analytics import get_analytics
from app.schemas.schemas import PatientInput

router = APIRouter()


class PredictRequest(BaseModel):
    patient: Dict[str, Any]
    model_choice: Optional[str] = "best"


class PredictAllRequest(BaseModel):
    patient: Dict[str, Any]


# ============================================
# Features Endpoint (FIX: Add this)
# ============================================

@router.get("/features")
def get_features():
    """Get feature information for the frontend form."""
    try:
        bundle = get_bundle()
        return {
            "feature_cols": bundle.feature_cols,
            "numeric_features": bundle.numeric_features,
            "categorical_features": bundle.categorical_features,
            "categorical_options": bundle.categorical_options,
            "target_classes": bundle.target_classes,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading features: {str(e)}")


@router.get("/features/importance")
def feature_importance(model_key: str = "best"):
    """Get feature importance for a specific model."""
    try:
        bundle = get_bundle()
        
        # For tree-based models, try to get feature importance
        if model_key == "best":
            model_key = bundle.best_model_key
        
        model = bundle.models.get(model_key)
        if model is None:
            return {"features": []}
        
        # Try to get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Map to feature names
            return {"features": []}
        
        # Fallback: use correlation-based importance from metadata
        return {"features": bundle.metadata.get("feature_importance", [])}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Models Endpoints
# ============================================

@router.get("/models")
def list_models():
    """Get all available models with their metrics."""
    try:
        bundle = get_bundle()
        
        # Get available models from metadata
        available_models = bundle.metadata.get("available_models", {})
        
        # Format response
        models_list = []
        for key, info in available_models.items():
            # Check if model is loaded
            is_loaded = key in bundle.models and bundle.models[key] is not None
            
            if is_loaded:
                models_list.append({
                    "key": key,
                    "name": info.get("name", key),
                    "type": info.get("type", "ML"),
                    "is_keras": info.get("is_keras", False),
                    "is_best": info.get("name") == bundle.best_model_name,
                    "path": info.get("path", ""),
                    "metrics": info.get("metrics", {})
                })
        
        return {
            "available_models": models_list,
            "best_model": bundle.best_model_name,
            "model_comparison": bundle.metadata.get("all_model_metrics", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")


@router.get("/models/comparison")
def model_comparison():
    """Get model comparison metrics."""
    try:
        bundle = get_bundle()
        return bundle.metadata.get("all_model_metrics", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_key}/metrics")
def model_metrics(model_key: str):
    """Get metrics for a specific model."""
    try:
        bundle = get_bundle()
        available_models = bundle.metadata.get("available_models", {})
        
        if model_key not in available_models:
            raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
        
        return available_models[model_key].get("metrics", {})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/best")
def best_model():
    """Get the best model info."""
    try:
        bundle = get_bundle()
        
        # Find best model
        best_key = None
        best_info = None
        for key, info in bundle.metadata.get("available_models", {}).items():
            if info.get("name") == bundle.best_model_name:
                best_key = key
                best_info = info
                break
        
        return {
            "key": best_key,
            "name": bundle.best_model_name,
            "type": bundle.metadata.get("best_model_type", "ML"),
            "is_keras": bundle.is_keras,
            "metrics": bundle.metrics.get("best_model_metrics", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Prediction Endpoints
# ============================================

@router.post("/predict")
def predict(request: PredictRequest):
    """
    Make a prediction using a specific model.
    
    Args:
        request.patient: Patient clinical data
        request.model_choice: 'best', 'random_forest', 'xgboost', 'ann', etc.
    """
    try:
        bundle = get_bundle()
        patient = request.patient
        
        # Validate categorical values
        for col in bundle.categorical_features:
            if col in patient:
                valid_options = bundle.categorical_options.get(col, [])
                patient_value = str(patient[col])
                if valid_options and patient_value not in valid_options:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Invalid value '{patient_value}' for '{col}'. Valid: {valid_options}",
                    )
        
        # Make prediction with chosen model
        result = bundle.predict(patient, model_choice=request.model_choice)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-all")
def predict_all(request: PredictAllRequest):
    """
    Make predictions using all available models for comparison.
    """
    try:
        bundle = get_bundle()
        patient = request.patient
        
        predictions = {}
        available_models = bundle.metadata.get("available_models", {})
        
        for key in available_models.keys():
            try:
                result = bundle.predict(patient, model_choice=key)
                predictions[key] = result
            except Exception as e:
                predictions[key] = {"error": str(e)}
        
        # Check agreement
        pred_values = []
        for key, pred in predictions.items():
            if "prediction" in pred:
                pred_values.append(pred["prediction"])
        
        agreement = len(set(pred_values)) == 1 if pred_values else False
        
        return {
            "predictions": predictions,
            "agreement": agreement,
            "models_used": list(predictions.keys())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
def explain(request: PredictRequest):
    """
    Get SHAP explanation for a prediction.
    """
    try:
        bundle = get_bundle()
        patient = request.patient
        
        # Make prediction with chosen model
        result = bundle.predict(patient, model_choice=request.model_choice)
        
        return {
            "shap_factors": result.get("shap_factors", []),
            "explanation": result.get("explanation", [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Validation Endpoints
# ============================================

@router.post("/validate")
def validate(patient: Dict[str, Any]):
    """
    Validate patient data before prediction.
    """
    try:
        bundle = get_bundle()
        errors = []
        
        # Check required fields
        for col in bundle.feature_cols:
            if col not in patient:
                errors.append(f"Missing field: {col}")
        
        # Validate categorical values
        for col in bundle.categorical_features:
            if col in patient:
                valid_options = bundle.categorical_options.get(col, [])
                patient_value = str(patient[col])
                if valid_options and patient_value not in valid_options:
                    errors.append(f"Invalid value '{patient_value}' for '{col}'. Valid: {valid_options}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors if errors else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Model Info Endpoint (if needed)
# ============================================

@router.get("/model-info")
def model_info():
    """Get model performance metrics and information."""
    try:
        bundle = get_bundle()
        
        response = {
            "best_model": bundle.best_model_name,
            "best_model_type": bundle.metadata.get("best_model_type", "Unknown"),
            "selection_priority": bundle.metadata.get("selection_priority", ["roc_auc", "f1", "recall"]),
            "all_models": bundle.metadata.get("all_model_metrics", []),
            "best_model_metrics": bundle.metadata.get("best_model_metrics", {}),
            "roc_curves": bundle.metadata.get("roc_curves", {}),
            "confusion_matrix": bundle.metadata.get("confusion_matrix", []),
            "target_classes": bundle.target_classes,
            "dataset_info": bundle.metadata.get("dataset_info", {}),
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model info: {str(e)}")