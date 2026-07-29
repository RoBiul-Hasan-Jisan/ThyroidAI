"""
Loads all trained ML/DL artifacts once at startup and exposes prediction +
SHAP explanation utilities used by the API routers.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")


class ModelBundle:
    def __init__(self):
        print(f"📦 Loading model bundle from: {MODEL_DIR}")
        
        # Load unified metadata
        try:
            with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
                self.metadata = json.load(f)
            print("✅ Metadata loaded")
        except FileNotFoundError:
            print("❌ metadata.json not found! Please train models first.")
            raise
        
        # Load shared components
        try:
            self.preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessing.pkl"))
            self.target_encoder = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
            self.shap_background = joblib.load(os.path.join(MODEL_DIR, "shap_background.pkl"))
            print("✅ Preprocessor and encoders loaded")
        except FileNotFoundError as e:
            print(f"❌ Error loading preprocessor: {e}")
            raise
        
        # Feature info
        self.feature_cols = self.metadata["feature_cols"]
        self.numeric_features = self.metadata["numeric_features"]
        self.categorical_features = self.metadata["categorical_features"]
        self.categorical_options = self.metadata["categorical_options"]
        self.target_classes = self.metadata["target_classes"]
        
        # Load encoded feature names for SHAP
        try:
            with open(os.path.join(MODEL_DIR, "encoded_feature_names.json")) as f:
                self.encoded_feature_names = json.load(f)
        except FileNotFoundError:
            print("⚠️ encoded_feature_names.json not found, using feature_cols")
            self.encoded_feature_names = self.feature_cols
        
        # Load all available models
        self.models = {}
        self.available_models = self.metadata.get("available_models", {})
        
        print("\n📊 Loading models:")
        for model_key, model_info in self.available_models.items():
            try:
                model_path = os.path.join(MODEL_DIR, model_info["path"])
                if model_info["is_keras"]:
                    from tensorflow import keras
                    try:
                        # Try loading with custom_objects to handle quantization_config
                        self.models[model_key] = keras.models.load_model(
                            model_path,
                            compile=False  # Load without compiling first
                        )
                        # Then compile with appropriate settings
                        self.models[model_key].compile(
                            optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy']
                        )
                        print(f"  ✅ Loaded Keras model: {model_info['name']}")
                    except Exception as e:
                        print(f"  ⚠️ Could not load Keras model {model_info['name']}: {e}")
                        self.models[model_key] = None
                else:
                    self.models[model_key] = joblib.load(model_path)
                    print(f"  ✅ Loaded ML model: {model_info['name']}")
            except Exception as e:
                print(f"  ⚠️ Could not load {model_info['name']}: {e}")
                self.models[model_key] = None
        
        # If best model failed to load, use fallback
        self.best_model_name = self.metadata["best_model_name"]
        self.is_keras = self.metadata["is_keras_model"]
        
        # Find best model key
        self.best_model_key = None
        for key, info in self.available_models.items():
            if info.get("name") == self.best_model_name:
                self.best_model_key = key
                break
        
        # Check if best model loaded successfully, if not use fallback
        if self.best_model_key is None or self.best_model_key not in self.models or self.models[self.best_model_key] is None:
            print(f"⚠️ Best model '{self.best_model_name}' failed to load. Using fallback...")
            # Find the next best model (Random Forest or XGBoost)
            fallback_models = ['random_forest', 'xgboost', 'logistic_regression', 'lightgbm', 'svm']
            for key in fallback_models:
                if key in self.models and self.models[key] is not None:
                    self.best_model_key = key
                    self.best_model_name = self.available_models[key].get('name', key)
                    self.is_keras = False
                    print(f"  ✅ Using fallback model: {self.best_model_name}")
                    break
        
        # Initialize SHAP explainer for the best model
        self.explainer = None
        try:
            best_model = self.models.get(self.best_model_key)
            if best_model is not None:
                # Create prediction function for SHAP
                def predict_fn(X):
                    if self.is_keras:
                        return best_model.predict(X, verbose=0).ravel()
                    else:
                        return best_model.predict_proba(X)[:, 1]
                
                # Use a smaller background for faster SHAP
                bg_sample = self.shap_background[:50] if len(self.shap_background) > 50 else self.shap_background
                self.explainer = shap.Explainer(
                    predict_fn,
                    bg_sample,
                    feature_names=self.encoded_feature_names,
                )
                print("✅ SHAP explainer initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize SHAP explainer: {e}")
            self.explainer = None
        
        print(f"\n✅ Best model: {self.best_model_name}")
        print(f"✅ Available models: {[k for k, v in self.models.items() if v is not None]}")

    def _row_to_dataframe(self, patient: dict) -> pd.DataFrame:
        """Convert patient dict to DataFrame with correct column order."""
        row = {}
        for c in self.feature_cols:
            # Handle missing keys gracefully
            if c in patient:
                row[c] = patient[c]
            else:
                # Use default values if available
                if c in self.categorical_features:
                    row[c] = self.categorical_options.get(c, [""])[0] if self.categorical_options.get(c) else ""
                else:
                    row[c] = 0
        return pd.DataFrame([row])[self.feature_cols]

    def _to_dense(self, X_t):
        """Convert sparse matrix to dense if needed."""
        return X_t.toarray() if hasattr(X_t, "toarray") else X_t

    def _get_shap_explanation(self, X_dense, model_key: str) -> dict:
        """Get SHAP explanation for a prediction."""
        shap_factors = []
        explanation_strings = []
        
        if self.explainer is None:
            return {"shap_factors": [], "explanation": ["SHAP explanation not available"]}
        
        try:
            # Get SHAP values
            shap_values = self.explainer(X_dense)
            values = shap_values.values[0]
            
            # Group one-hot encoded SHAP contributions back to original feature names
            grouped = {}
            for enc_name, val in zip(self.encoded_feature_names, values):
                if enc_name in self.numeric_features:
                    original = enc_name
                else:
                    # One-hot names look like "Risk_High" -> original col "Risk"
                    original = next(
                        (c for c in self.categorical_features if enc_name.startswith(c + "_")), 
                        enc_name
                    )
                grouped[original] = grouped.get(original, 0.0) + float(val)
            
            # Rank features by impact
            ranked = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)
            
            for feature, impact in ranked[:6]:
                direction = "increases" if impact > 0 else "decreases"
                shap_factors.append({
                    "feature": feature,
                    "impact": round(impact, 4),
                    "direction": direction,
                })
                verb = "Increases" if impact > 0 else "Reduces"
                explanation_strings.append(f"{verb} risk: {feature}")
            
        except Exception as e:
            print(f"⚠️ SHAP explanation failed: {e}")
            explanation_strings.append("SHAP explanation temporarily unavailable")
        
        return {
            "shap_factors": shap_factors,
            "explanation": explanation_strings
        }

    def predict(self, patient: dict, model_choice: str = "best") -> dict:
        """
        Make prediction using specified model.
        
        Args:
            patient: Patient features
            model_choice: 'best', 'random_forest', 'xgboost', 'ann', etc.
        
        Returns:
            Prediction result with explanation
        """
        # Determine which model to use
        if model_choice == "best":
            model_key = self.best_model_key
        else:
            model_key = model_choice
        
        # Check if model exists
        if model_key not in self.models or self.models[model_key] is None:
            available = [k for k, v in self.models.items() if v is not None]
            raise ValueError(f"Model '{model_key}' not available. Available: {available}")
        
        model = self.models[model_key]
        model_info = self.available_models.get(model_key, {})
        model_name = model_info.get("name", model_key)
        is_keras = model_info.get("is_keras", False)
        
        # Preprocess
        X_df = self._row_to_dataframe(patient)
        X_t = self.preprocessor.transform(X_df)
        X_dense = self._to_dense(X_t)
        
        # Get prediction
        try:
            if is_keras:
                prob_yes = float(model.predict(X_dense, verbose=0)[0][0])
            else:
                prob_yes = float(model.predict_proba(X_dense)[0][1])
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
        
        prob_no = 1.0 - prob_yes
        pred_idx = int(prob_yes >= 0.5)
        pred_label = self.target_encoder.inverse_transform([pred_idx])[0]
        
        # Confidence level
        max_prob = max(prob_yes, prob_no)
        if max_prob >= 0.85:
            confidence = "High"
        elif max_prob >= 0.65:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Get SHAP explanation
        shap_result = self._get_shap_explanation(X_dense, model_key)
        
        # Combine prediction with patient values for explanation
        explanation_with_values = []
        for item in shap_result["explanation"]:
            # Extract feature name from explanation
            for factor in shap_result["shap_factors"]:
                if factor["feature"] in item:
                    patient_value = patient.get(factor["feature"], "N/A")
                    explanation_with_values.append(f"{item} (value: {patient_value})")
                    break
            else:
                explanation_with_values.append(item)
        
        return {
            "prediction": str(pred_label),
            "probability": round(prob_yes, 4),
            "confidence": confidence,
            "model_used": model_name,
            "model_key": model_key,
            "explanation": explanation_with_values if explanation_with_values else ["No explanation available"],
            "shap_factors": shap_result["shap_factors"],
            "probabilities": {
                self.target_classes[0]: round(prob_no, 4),
                self.target_classes[1]: round(prob_yes, 4),
            },
        }

    def get_available_models(self):
        """Get list of available models with their info."""
        models_list = []
        for key, info in self.available_models.items():
            if self.models.get(key) is not None:
                models_list.append({
                    "key": key,
                    "name": info.get("name", key),
                    "type": info.get("type", "ML"),
                    "is_keras": info.get("is_keras", False),
                    "is_best": info.get("name") == self.best_model_name,
                    "path": info.get("path", ""),
                    "metrics": info.get("metrics", {})
                })
        return models_list

    def get_model_comparison(self):
        """Get comparison of all models."""
        return self.metadata.get("all_model_metrics", [])

    def get_model_metrics(self, model_key: str):
        """Get metrics for a specific model."""
        available_models = self.metadata.get("available_models", {})
        if model_key in available_models:
            return available_models[model_key].get("metrics", {})
        return {}

    def get_feature_importance(self, model_key: str = "best"):
        """Get feature importance for a specific model."""
        # For tree-based models, try to get feature importance
        if model_key == "best":
            model_key = self.best_model_key
        
        model = self.models.get(model_key)
        if model is None:
            return []
        
        # Try to get feature importance
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                # Map to feature names
                # This is approximate - better to use precomputed importance
                return []
        except:
            pass
        
        # Fallback: use correlation-based importance from metadata
        return self.metadata.get("feature_importance", [])


# Singleton
_bundle: ModelBundle | None = None


def get_bundle() -> ModelBundle:
    """Get or create the singleton ModelBundle instance."""
    global _bundle
    if _bundle is None:
        _bundle = ModelBundle()
    return _bundle


def reload_bundle():
    """Force reload of the model bundle (useful for development)."""
    global _bundle
    _bundle = None
    return get_bundle()