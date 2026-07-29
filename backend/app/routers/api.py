from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.core.model_loader import get_bundle
from app.core.analytics import get_analytics
from app.schemas.schemas import PatientInput
import json
import traceback
import math

router = APIRouter()


class PredictRequest(BaseModel):
    patient: Dict[str, Any]
    model_choice: Optional[str] = "best"


class PredictAllRequest(BaseModel):
    patient: Dict[str, Any]


# Helper function to clean NaN/Inf values
def clean_for_json(obj):
    """Recursively clean objects for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    else:
        return obj


# ============================================
# Analytics Endpoint
# ============================================

@router.get("/analytics")
def analytics():
    """Get EDA analytics data for the frontend dashboard."""
    try:
        return get_analytics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading analytics: {str(e)}")


# ============================================
# Features Endpoint
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
        
        if model_key is None:
            return {"features": []}
        
        model = bundle.models.get(model_key)
        if model is None:
            return {"features": []}
        
        # Try to get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Get feature names from preprocessor
            feature_names = bundle.encoded_feature_names
            
            # Create sorted list of feature importance
            features = []
            for idx, importance in enumerate(importances):
                if idx < len(feature_names):
                    features.append({
                        "feature": feature_names[idx],
                        "importance": float(importance)
                    })
            
            # Sort by importance descending
            features = sorted(features, key=lambda x: x["importance"], reverse=True)
            return {"features": features[:20]}  # Top 20 features
        
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
        
        print("🔍 Getting available models from bundle...")
        
        # Get available models from the bundle's available_models
        models_list = bundle.get_available_models()
        
        print(f"📊 Got {len(models_list)} models")
        
        # Ensure all data is JSON serializable
        formatted_models = []
        for model in models_list:
            try:
                # Get metrics safely
                metrics = model.get("metrics", {})
                
                formatted_model = {
                    "key": str(model.get("key", "")),
                    "name": str(model.get("name", "")),
                    "type": str(model.get("type", "ML")),
                    "is_keras": bool(model.get("is_keras", False)),
                    "is_best": bool(model.get("is_best", False)),
                    "path": str(model.get("path", "")),
                    "metrics": {
                        "accuracy": float(metrics.get("accuracy", 0.0)) if metrics.get("accuracy") is not None else None,
                        "precision": float(metrics.get("precision", 0.0)) if metrics.get("precision") is not None else None,
                        "recall": float(metrics.get("recall", 0.0)) if metrics.get("recall") is not None else None,
                        "f1": float(metrics.get("f1", 0.0)) if metrics.get("f1") is not None else None,
                        "roc_auc": float(metrics.get("roc_auc", 0.0)) if metrics.get("roc_auc") is not None else None,
                        "cv_roc_auc_mean": float(metrics.get("cv_roc_auc_mean")) if metrics.get("cv_roc_auc_mean") is not None else None,
                        "cv_roc_auc_std": float(metrics.get("cv_roc_auc_std")) if metrics.get("cv_roc_auc_std") is not None else None,
                    }
                }
                # Clean any NaN/Inf values
                formatted_model = clean_for_json(formatted_model)
                formatted_models.append(formatted_model)
            except Exception as e:
                print(f"❌ Error formatting model {model.get('key', 'unknown')}: {e}")
                # Add a minimal fallback model entry
                formatted_models.append({
                    "key": str(model.get("key", "unknown")),
                    "name": str(model.get("name", "Unknown Model")),
                    "type": "ML",
                    "is_keras": False,
                    "is_best": False,
                    "path": "",
                    "metrics": {
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "roc_auc": 0.0,
                        "cv_roc_auc_mean": None,
                        "cv_roc_auc_std": None,
                    }
                })
        
        # Get model comparison from metadata
        model_comparison = bundle.metadata.get("all_models", [])
        # Clean comparison data
        model_comparison = clean_for_json(model_comparison)
        
        response = {
            "available_models": formatted_models,
            "best_model": str(bundle.best_model_name),
            "model_comparison": model_comparison if model_comparison else []
        }
        
        print(f"✅ Returning {len(formatted_models)} models to frontend")
        return response
        
    except Exception as e:
        print(f"❌ Error in /models endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")


@router.get("/models/comparison")
def model_comparison():
    """Get model comparison metrics."""
    try:
        bundle = get_bundle()
        # Get comparison from metadata
        comparison = bundle.metadata.get("all_models", [])
        # Clean for JSON
        return clean_for_json(comparison)
    except Exception as e:
        print(f"❌ Error in /models/comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_key}/metrics")
def model_metrics(model_key: str):
    """Get metrics for a specific model."""
    try:
        bundle = get_bundle()
        metrics = bundle.get_model_metrics(model_key)
        
        if not metrics:
            raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
        
        return clean_for_json(metrics)
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in /models/{model_key}/metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/best")
def best_model():
    """Get the best model info."""
    try:
        bundle = get_bundle()
        
        response = {
            "key": str(bundle.best_model_key) if bundle.best_model_key else None,
            "name": str(bundle.best_model_name),
            "type": str(getattr(bundle, "best_model_type", "ML")),
            "is_keras": bool(bundle.is_keras),
            "metrics": bundle.best_model_metrics
        }
        return clean_for_json(response)
    except Exception as e:
        print(f"❌ Error in /models/best: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Debug Endpoint
# ============================================

@router.get("/debug/models")
def debug_models():
    """Debug endpoint to see raw model data."""
    try:
        bundle = get_bundle()
        return {
            "available_models_raw": clean_for_json(bundle.available_models),
            "loaded_models": {k: str(v is not None) for k, v in bundle.models.items()},
            "best_model_key": bundle.best_model_key,
            "best_model_name": bundle.best_model_name,
            "metadata_keys": list(bundle.metadata.keys()),
            "all_models_from_metadata": clean_for_json(bundle.metadata.get("all_models", [])),
            "models_list_from_get_available": clean_for_json(bundle.get_available_models())
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


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
        return clean_for_json(result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error in /predict: {str(e)}")
        print(traceback.format_exc())
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
        available_models = bundle.get_available_models()
        
        for model_info in available_models:
            key = model_info["key"]
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
        
        response = {
            "predictions": predictions,
            "agreement": agreement,
            "models_used": list(predictions.keys())
        }
        return clean_for_json(response)
        
    except Exception as e:
        print(f"❌ Error in /predict-all: {str(e)}")
        print(traceback.format_exc())
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
        
        response = {
            "shap_factors": result.get("shap_factors", []),
            "explanation": result.get("explanation", [])
        }
        return clean_for_json(response)
        
    except Exception as e:
        print(f"❌ Error in /explain: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Validation Endpoint
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
        print(f"❌ Error in /validate: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Model Info Endpoint
# ============================================

@router.get("/model-info")
def model_info():
    """Get model performance metrics and information."""
    try:
        bundle = get_bundle()
        metadata = bundle.metadata
        
        dataset_info = metadata.get("dataset_info", {})
        
        response = {
            "best_model": str(bundle.best_model_name),
            "best_model_type": str(getattr(bundle, "best_model_type", "Unknown")),
            "selection_priority": metadata.get("selection_priority", ["roc_auc", "f1", "recall"]),
            "all_models": metadata.get("all_models", []),
            "best_model_metrics": bundle.best_model_metrics,
            "roc_curves": metadata.get("roc_curves", {}),
            "confusion_matrix": getattr(bundle, "confusion_matrix", []),
            "target_classes": bundle.target_classes,
            "dataset_info": dataset_info,
        }
        return clean_for_json(response)
    except Exception as e:
        print(f"❌ Error in /model-info: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error loading model info: {str(e)}")


# ============================================
# Health Endpoint
# ============================================

@router.get("/health")
def health():
    """Health check endpoint."""
    try:
        bundle = get_bundle()
        available = [str(k) for k, v in bundle.models.items() if v is not None]
        return {
            "status": "ok",
            "best_model": str(bundle.best_model_name),
            "model_type": "DL" if bundle.is_keras else "ML",
            "features": len(bundle.feature_cols),
            "is_keras": bool(bundle.is_keras),
            "available_models": available
        }
    except Exception as e:
        print(f"❌ Error in /health: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


# ============================================
# Root Endpoint
# ============================================

@router.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "ThyroidAI API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/features": "Get feature information",
            "/models": "List available models",
            "/predict": "Make a prediction",
            "/predict-all": "Predict with all models",
            "/explain": "Get SHAP explanation",
            "/model-info": "Get model performance metrics"
        }
    }