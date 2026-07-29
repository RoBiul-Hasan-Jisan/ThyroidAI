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
        print(f" Loading model bundle from: {MODEL_DIR}")

        # Load unified metadata
        try:
            with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
                self.metadata = json.load(f)
            print(" Metadata loaded")
        except FileNotFoundError:
            print(" metadata.json not found! Please train models first.")
            raise

        # Load shared components
        try:
            self.preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessing.pkl"))
            self.target_encoder = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
            self.shap_background = joblib.load(os.path.join(MODEL_DIR, "shap_background.pkl"))
            print(" Preprocessor and encoders loaded")
        except FileNotFoundError as e:
            print(f" Error loading preprocessor: {e}")
            raise

        # Load ROC curve data (stored in a sibling file, not inside metadata.json)
        try:
            with open(os.path.join(MODEL_DIR, "roc_curves.json")) as f:
                self.metadata["roc_curves"] = json.load(f)
            print(" ROC curves loaded")
        except FileNotFoundError:
            self.metadata.setdefault("roc_curves", {})
            print(" roc_curves.json not found")

        # ------------------------------------------------------------------
        # Extract feature info. The real metadata.json stores these at the
        # TOP LEVEL, not nested under "dataset_info" (dataset_info only has
        # n_samples/n_features/class_balance). We read top-level first and
        # fall back to dataset_info/nested keys for older metadata formats.
        # ------------------------------------------------------------------
        dataset_info = self.metadata.get("dataset_info", {})
        all_models = self.metadata.get("all_models", [])
        model_files = self.metadata.get("model_files", {})

        self.feature_cols = self.metadata.get("feature_cols") or dataset_info.get("feature_cols", [])
        self.numeric_features = self.metadata.get("numeric_features") or dataset_info.get("numeric_features", [])
        self.categorical_features = self.metadata.get("categorical_features") or dataset_info.get("categorical_features", [])
        self.categorical_options = self.metadata.get("categorical_options") or dataset_info.get("categorical_options", {})
        self.target_classes = self.metadata.get("target_classes") or dataset_info.get("target_classes", [])

        # Load encoded feature names for SHAP
        try:
            with open(os.path.join(MODEL_DIR, "encoded_feature_names.json")) as f:
                self.encoded_feature_names = json.load(f)
        except FileNotFoundError:
            print(" encoded_feature_names.json not found, using feature_cols")
            self.encoded_feature_names = self.feature_cols

        # ------------------------------------------------------------------
        # Extract best-model info. Real metadata.json stores these as flat
        # top-level keys (best_model_name, model_type, is_keras,
        # best_model_metrics, confusion_matrix). An older/alternate format
        # nests them under a "best_model" dict — support both.
        # ------------------------------------------------------------------
        legacy_best_model = self.metadata.get("best_model")
        legacy_best_model = legacy_best_model if isinstance(legacy_best_model, dict) else {}

        self.best_model_name = legacy_best_model.get("name") or self.metadata.get("best_model_name") or "Unknown"
        if self.best_model_name is None:
            self.best_model_name = "Unknown"

        self.is_keras = legacy_best_model.get(
            "is_keras",
            self.metadata.get("is_keras", self.metadata.get("is_keras_model", False)),
        )

        self.best_model_metrics = legacy_best_model.get("metrics") or self.metadata.get("best_model_metrics", {})
        self.confusion_matrix = legacy_best_model.get("confusion_matrix") or self.metadata.get("confusion_matrix", [])
        self.best_model_type = legacy_best_model.get("type") or self.metadata.get("model_type", "Unknown")

        # ------------------------------------------------------------------
        # Build available_models from all_models + model_files.
        # NOTE: each entry in all_models uses "model" (not "model_name") and
        # "type" (not "model_type") as its keys, e.g.:
        #   {"model": "Random Forest", "type": "ML", "accuracy": ..., ...}
        # ------------------------------------------------------------------
        self.available_models = {}
        for model_entry in all_models:
            if not isinstance(model_entry, dict):
                continue

            model_name = model_entry.get("model") or model_entry.get("model_name")
            if model_name is None:
                continue

            model_type = model_entry.get("type") or model_entry.get("model_type") or "ML"
            is_keras = model_type == "DL" or "keras" in model_name.lower() or "ann" in model_name.lower()

            # Derive a simple, stable key from the model name
            model_key = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            if "ann" in model_key or "neural_network" in model_key:
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

            # Resolve file path with proper None handling
            if is_keras:
                # Check for keras model file
                model_path = model_files.get("best_model_keras")
                if not model_path:
                    # Look for any .keras file in the models directory
                    keras_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
                    if keras_files:
                        model_path = keras_files[0]
                        print(f"   Found Keras model: {model_path}")
                    else:
                        # Try default name
                        default_keras = "best_model.keras"
                        if os.path.exists(os.path.join(MODEL_DIR, default_keras)):
                            model_path = default_keras
                        else:
                            print(f"    No Keras model file found for {model_name}, skipping")
                            continue  # Skip this model entirely
            else:
                # Check for ML model file
                model_path = model_files.get(model_key)
                if not model_path:
                    # Try different naming conventions
                    fallback_name = model_name.replace(" ", "_")
                    possible_paths = [
                        os.path.join("all_models", f"{fallback_name}.pkl"),
                        f"{fallback_name}.pkl",
                        os.path.join("all_models", f"{model_key}.pkl"),
                        f"{model_key}.pkl"
                    ]
                    for path in possible_paths:
                        if os.path.exists(os.path.join(MODEL_DIR, path)):
                            model_path = path
                            break
                    if not model_path:
                        print(f"    Could not find model file for {model_name}, skipping")
                        continue  # Skip this model entirely

            is_best = model_name == self.best_model_name

            self.available_models[model_key] = {
                "name": model_name,
                "type": model_type,
                "is_keras": is_keras,
                "path": model_path,
                "metrics": model_entry,
                "is_best": is_best,
            }

        # Backward compatibility: some older metadata formats keep a
        # top-level "available_models" dict instead of "all_models".
        if not self.available_models:
            avail_models = self.metadata.get("available_models", {})
            for model_key, model_info in avail_models.items():
                if not isinstance(model_info, dict):
                    continue
                model_name = model_info.get("name", model_key)
                if model_name is None:
                    continue
                # Skip if path is None
                model_path = model_info.get("path", "")
                if not model_path:
                    print(f"    No file path for {model_name}, skipping")
                    continue
                self.available_models[model_key] = {
                    "name": model_name,
                    "type": model_info.get("type", "ML"),
                    "is_keras": model_info.get("is_keras", False),
                    "path": model_path,
                    "metrics": model_info.get("metrics", {}),
                    "is_best": model_key == self.metadata.get("best_model_key", ""),
                }

        # ------------------------------------------------------------------
        # Load every model referenced in available_models.
        # ------------------------------------------------------------------
        self.models = {}
        print("\n Loading models:")
        for model_key, model_info in self.available_models.items():
            # Skip if path is None or empty
            if not model_info.get("path"):
                print(f"    No file path specified for {model_info.get('name', model_key)}")
                self.models[model_key] = None
                continue

            model_path = os.path.join(MODEL_DIR, model_info["path"])
            
            # If file doesn't exist at the specified path, try alternative locations
            if not os.path.exists(model_path):
                alt_paths = [
                    os.path.join(MODEL_DIR, "all_models", f"{model_key}.pkl"),
                    os.path.join(MODEL_DIR, f"{model_key}.pkl"),
                    os.path.join(MODEL_DIR, "all_models", f"{model_info['name'].replace(' ', '_')}.pkl"),
                ]
                found = False
                for alt in alt_paths:
                    if os.path.exists(alt):
                        model_path = alt
                        found = True
                        break
                if not found:
                    print(f"    Model file not found: {model_info['path']}")
                    self.models[model_key] = None
                    continue

            try:
                if model_info.get("is_keras", False):
                    from tensorflow import keras
                    self.models[model_key] = keras.models.load_model(model_path, compile=False)
                    self.models[model_key].compile(
                        optimizer="adam",
                        loss="binary_crossentropy",
                        metrics=["accuracy"],
                    )
                    print(f"    Loaded Keras model: {model_info['name']}")
                else:
                    self.models[model_key] = joblib.load(model_path)
                    print(f"    Loaded ML model: {model_info['name']}")
            except Exception as e:
                print(f"    Could not load {model_info.get('name', model_key)}: {e}")
                self.models[model_key] = None

        # ------------------------------------------------------------------
        # Determine the best model key: prefer the entry flagged is_best that
        # actually loaded successfully, otherwise fall back to the first
        # successfully loaded model.
        # ------------------------------------------------------------------
        self.best_model_key = None
        for key, info in self.available_models.items():
            if info.get("is_best") and self.models.get(key) is not None:
                self.best_model_key = key
                break

        if self.best_model_key is None:
            for key in self.models:
                if self.models.get(key) is not None:
                    self.best_model_key = key
                    break

        # Keep is_keras in sync with whichever model actually ended up "best"
        if self.best_model_key and self.best_model_key in self.available_models:
            self.is_keras = self.available_models[self.best_model_key].get("is_keras", self.is_keras)

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
                    print(" SHAP explainer initialized")
                else:
                    print(" Best model is None, SHAP explainer not initialized")
            else:
                print(" No best model found, SHAP explainer not initialized")
        except Exception as e:
            print(f" Could not initialize SHAP explainer: {e}")
            self.explainer = None

        # Get best model name for display
        display_name = self.best_model_name
        if display_name == "Unknown" and self.best_model_key:
            if self.best_model_key in self.available_models:
                display_name = self.available_models[self.best_model_key].get("name", self.best_model_key)
            else:
                display_name = self.best_model_key

        print(f"\n Best model: {display_name}")
        available = [k for k, v in self.models.items() if v is not None]
        print(f" Available models: {available}")

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
            print(f" SHAP explanation failed: {e}")
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

        # Build value-annotated explanation strings directly from shap_factors
        # (each factor already knows its own feature name and direction) rather
        # than re-matching against the rendered strings, since short feature
        # names like "T"/"N"/"M" would otherwise substring-match unrelated
        # lines (e.g. "T" matching inside "Thyroid Function").
        explanation_with_values = []
        for factor in shap_result["shap_factors"]:
            verb = "Increases" if factor["direction"] == "increases" else "Reduces"
            patient_value = patient.get(factor["feature"], "N/A")
            explanation_with_values.append(
                f"{verb} risk: {factor['feature']} (value: {patient_value})"
            )
        if not explanation_with_values:
            explanation_with_values = list(shap_result["explanation"])

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