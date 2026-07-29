# backend/app/core/model_loader.py
"""
Loads all trained ML/DL artifacts once at startup and provides prediction +
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
        
        # Extract feature info from unified metadata structure
        dataset_info = self.metadata.get("dataset_info", {})
        best_model = self.metadata.get("best_model", {})
        all_models = self.metadata.get("all_models", [])
        model_files = self.metadata.get("model_files", {})
        
        # Extract feature info
        self.feature_cols = dataset_info.get("feature_cols", [])
        self.numeric_features = dataset_info.get("numeric_features", [])
        self.categorical_features = dataset_info.get("categorical_features", [])
        self.categorical_options = dataset_info.get("categorical_options", {})
        self.target_classes = dataset_info.get("target_classes", [])
        
        # If not found in dataset_info, try top-level (for backward compatibility)
        if not self.feature_cols:
            self.feature_cols = self.metadata.get("feature_cols", [])
        if not self.numeric_features:
            self.numeric_features = self.metadata.get("numeric_features", [])
        if not self.categorical_features:
            self.categorical_features = self.metadata.get("categorical_features", [])
        if not self.categorical_options:
            self.categorical_options = self.metadata.get("categorical_options", {})
        if not self.target_classes:
            self.target_classes = self.metadata.get("target_classes", [])
        
        # Load encoded feature names for SHAP
        try:
            with open(os.path.join(MODEL_DIR, "encoded_feature_names.json")) as f:
                self.encoded_feature_names = json.load(f)
        except FileNotFoundError:
            print("⚠️ encoded_feature_names.json not found, using feature_cols")
            self.encoded_feature_names = self.feature_cols
        
        # Extract best model info - safely handle None values
        self.best_model_name = "Unknown"
        if best_model and isinstance(best_model, dict):
            self.best_model_name = best_model.get("name", "Unknown")
        elif self.metadata:
            self.best_model_name = self.metadata.get("best_model_name", "Unknown")
        
        # Ensure best_model_name is a string
        if self.best_model_name is None:
            self.best_model_name = "Unknown"
        
        self.is_keras = False
        if best_model and isinstance(best_model, dict):
            self.is_keras = best_model.get("is_keras", False)
        elif self.metadata:
            self.is_keras = self.metadata.get("is_keras_model", False)
        
        self.best_model_metrics = {}
        if best_model and isinstance(best_model, dict):
            self.best_model_metrics = best_model.get("metrics", {})
        elif self.metadata:
            self.best_model_metrics = self.metadata.get("best_model_metrics", {})
        
        # Build available_models dict from all_models and model_files
        self.available_models = {}
        for model_entry in all_models:
            if not isinstance(model_entry, dict):
                continue
                
            model_name = model_entry.get("model_name")
            # Skip if model_name is None
            if model_name is None:
                continue
                
            model_file = model_entry.get("model_file", "")
            model_type = model_entry.get("model_type", "sklearn")
            is_keras = model_type == "keras"
            
            # Find matching file info
            file_info = model_files.get(model_name, {}) if model_files else {}
            model_path = file_info.get("path", os.path.join("all_models", model_file))
            
            # Determine key from name - safely handle None
            if model_name is None:
                continue
            
            # Create safe key
            model_key = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            
            # Map to simpler keys
            if "_ann" in model_key or "neural_network" in model_key:
                model_key = "ann"
            elif "random_forest" in model_key:
                model_key = "random_forest"
            elif "xgboost" in model_key:
                model_key = "xgboost"
            elif "logistic_regression" in model_key:
                model_key = "logistic_regression"
            elif "lightgbm" in model_key:
                model_key = "lightgbm"
            elif "svm" in model_key:
                model_key = "svm"
            elif "gradient_boosting" in model_key:
                model_key = "gradient_boosting"
            elif "knn" in model_key:
                model_key = "knn"
            
            # Check if this is the best model
            is_best = False
            if self.best_model_name and model_name:
                is_best = model_name == self.best_model_name
            elif self.best_model_name and model_key:
                is_best = model_key == self.best_model_name.lower().replace(" ", "_")
            
            self.available_models[model_key] = {
                "name": model_name,
                "type": model_type,
                "is_keras": is_keras,
                "path": model_path,
                "metrics": model_entry,
                "is_best": is_best
            }
        
        # Also check for models in metadata top-level
        if not self.available_models:
            # Try to get from top-level available_models
            avail_models = self.metadata.get("available_models", {})
            if avail_models:
                for model_key, model_info in avail_models.items():
                    if isinstance(model_info, dict):
                        model_name = model_info.get("name", model_key)
                        # Skip None values
                        if model_name is None:
                            continue
                        self.available_models[model_key] = {
                            "name": model_name,
                            "type": model_info.get("type", "ML"),
                            "is_keras": model_info.get("is_keras", False),
                            "path": model_info.get("path", ""),
                            "metrics": model_info.get("metrics", {}),
                            "is_best": model_key == self.metadata.get("best_model_key", "")
                        }
        
        # Load all available models
        self.models = {}
        print("\n📊 Loading models:")
        
        # Also check for individual model files
        individual_models = {
            "best_model.pkl": "best",
            "xgb_model.pkl": "xgboost",
            "random_forest_model.pkl": "random_forest",
            "lgbm_model.pkl": "lightgbm",
            "logistic_regression.pkl": "logistic_regression",
            "svm_model.pkl": "svm",
            "gradient_boosting.pkl": "gradient_boosting",
            "knn_model.pkl": "knn"
        }
        
        # First load individual model files
        for filename, key in individual_models.items():
            filepath = os.path.join(MODEL_DIR, filename)
            if os.path.exists(filepath):
                try:
                    self.models[key] = joblib.load(filepath)
                    if key not in self.available_models:
                        self.available_models[key] = {
                            "name": key.replace("_", " ").title(),
                            "type": "sklearn",
                            "is_keras": False,
                            "path": filename,
                            "metrics": {},
                            "is_best": key == "best"
                        }
                    print(f"   ✅ Loaded ML model: {key}")
                except Exception as e:
                    print(f"   ⚠️ Could not load {filename}: {e}")
                    self.models[key] = None
        
        # Then load from all_models directory
        all_models_dir = os.path.join(MODEL_DIR, "all_models")
        if os.path.exists(all_models_dir):
            for file in os.listdir(all_models_dir):
                if file.endswith(".pkl"):
                    model_key = file.replace(".pkl", "")
                    # Skip if already loaded
                    if model_key in self.models and self.models[model_key] is not None:
                        continue
                    try:
                        filepath = os.path.join(all_models_dir, file)
                        self.models[model_key] = joblib.load(filepath)
                        print(f"   ✅ Loaded ML model: {model_key}")
                    except Exception as e:
                        print(f"   ⚠️ Could not load {file}: {e}")
                        self.models[model_key] = None
        
        # Load from available_models
        for model_key, model_info in self.available_models.items():
            if model_key in self.models and self.models[model_key] is not None:
                continue
            try:
                model_path = os.path.join(MODEL_DIR, model_info["path"])
                if not os.path.exists(model_path):
                    # Try in all_models directory
                    alt_path = os.path.join(MODEL_DIR, "all_models", f"{model_key}.pkl")
                    if os.path.exists(alt_path):
                        model_path = alt_path
                    else:
                        print(f"   ⚠️ Model file not found: {model_info['path']}")
                        continue
                
                if model_info.get("is_keras", False):
                    try:
                        from tensorflow import keras
                        self.models[model_key] = keras.models.load_model(model_path, compile=False)
                        self.models[model_key].compile(
                            optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy']
                        )
                        print(f"   ✅ Loaded Keras model: {model_info['name']}")
                    except Exception as e:
                        print(f"   ❌ Could not load Keras model {model_info['name']}: {e}")
                        self.models[model_key] = None
                else:
                    self.models[model_key] = joblib.load(model_path)
                    print(f"   ✅ Loaded ML model: {model_info['name']}")
            except Exception as e:
                print(f"   ❌ Could not load {model_info.get('name', model_key)}: {e}")
                self.models[model_key] = None
        
        # Find best model key
        self.best_model_key = None
        
        # First try to find by best_model_name
        if self.best_model_name and self.best_model_name != "Unknown":
            # Try to find matching model
            for key, model in self.models.items():
                if model is not None:
                    # Check if key matches best model name
                    if key == self.best_model_name.lower().replace(" ", "_").replace("(", "").replace(")", ""):
                        self.best_model_key = key
                        break
                    # Check if model info has matching name
                    if key in self.available_models:
                        info = self.available_models[key]
                        if info.get("name") == self.best_model_name:
                            self.best_model_key = key
                            break
        
        # If best model not found, use first available
        if self.best_model_key is None:
            for key in self.models.keys():
                if self.models.get(key) is not None:
                    self.best_model_key = key
                    break
        
        # If still None, use 'best' if available
        if self.best_model_key is None and 'best' in self.models and self.models['best'] is not None:
            self.best_model_key = 'best'
        
        # Initialize SHAP explainer
        self.explainer = None
        try:
            if self.best_model_key and self.best_model_key in self.models:
                best_model = self.models[self.best_model_key]
                if best_model is not None:
                    def predict_fn(X):
                        if self.is_keras:
                            return best_model.predict(X, verbose=0).ravel()
                        else:
                            try:
                                return best_model.predict_proba(X)[:, 1]
                            except AttributeError:
                                # Some models might not have predict_proba
                                return best_model.predict(X)
                    
                    bg_sample = self.shap_background[:50] if len(self.shap_background) > 50 else self.shap_background
                    self.explainer = shap.Explainer(
                        predict_fn,
                        bg_sample,
                        feature_names=self.encoded_feature_names,
                    )
                    print("✅ SHAP explainer initialized")
                else:
                    print("⚠️ Best model is None, SHAP explainer not initialized")
            else:
                print("⚠️ No best model found, SHAP explainer not initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize SHAP explainer: {e}")
            self.explainer = None
        
        # Get best model name for display
        display_name = self.best_model_name
        if display_name == "Unknown" and self.best_model_key:
            if self.best_model_key in self.available_models:
                display_name = self.available_models[self.best_model_key].get("name", self.best_model_key)
            else:
                display_name = self.best_model_key
        
        print(f"\n✅ Best model: {display_name}")
        available = [k for k, v in self.models.items() if v is not None]
        print(f"✅ Available models: {available}")

    def _row_to_dataframe(self, patient: dict) -> pd.DataFrame:
        """Convert patient dict to DataFrame with correct column order."""
        row = {}
        for c in self.feature_cols:
            if c in patient:
                row[c] = patient[c]
            else:
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
            shap_values = self.explainer(X_dense)
            values = shap_values.values[0]
            
            grouped = {}
            for enc_name, val in zip(self.encoded_feature_names, values):
                if enc_name in self.numeric_features:
                    original = enc_name
                else:
                    original = next(
                        (c for c in self.categorical_features if enc_name.startswith(c + "_")), 
                        enc_name
                    )
                grouped[original] = grouped.get(original, 0.0) + float(val)
            
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
        """Make prediction using specified model."""
        if model_choice == "best":
            model_key = self.best_model_key
        else:
            model_key = model_choice
        
        if model_key is None or model_key not in self.models or self.models[model_key] is None:
            available = [k for k, v in self.models.items() if v is not None]
            raise ValueError(f"Model '{model_key}' not available. Available: {available}")
        
        model = self.models[model_key]
        model_info = self.available_models.get(model_key, {})
        model_name = model_info.get("name", model_key)
        is_keras = model_info.get("is_keras", False)
        
        X_df = self._row_to_dataframe(patient)
        X_t = self.preprocessor.transform(X_df)
        X_dense = self._to_dense(X_t)
        
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
        
        max_prob = max(prob_yes, prob_no)
        if max_prob >= 0.85:
            confidence = "High"
        elif max_prob >= 0.65:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        shap_result = self._get_shap_explanation(X_dense, model_key)
        
        explanation_with_values = []
        for item in shap_result["explanation"]:
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
                # Get metrics from the model entry
                metrics = info.get("metrics", {})
                
                # Ensure all required fields are present
                models_list.append({
                    "key": key,
                    "name": info.get("name", key),
                    "type": info.get("type", "ML"),
                    "is_keras": info.get("is_keras", False),
                    "is_best": info.get("is_best", False),
                    "path": info.get("path", ""),
                    "metrics": {
                        "accuracy": metrics.get("accuracy", 0),
                        "precision": metrics.get("precision", 0),
                        "recall": metrics.get("recall", 0),
                        "f1": metrics.get("f1", 0),
                        "roc_auc": metrics.get("roc_auc", 0),
                        "cv_roc_auc_mean": metrics.get("cv_roc_auc_mean", None),
                        "cv_roc_auc_std": metrics.get("cv_roc_auc_std", None),
                    }
                })
        return models_list

    def get_model_comparison(self):
        """Get comparison of all models."""
        return self.metadata.get("all_models", [])

    def get_model_metrics(self, model_key: str):
        """Get metrics for a specific model."""
        if model_key in self.available_models:
            return self.available_models[model_key].get("metrics", {})
        return {}


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