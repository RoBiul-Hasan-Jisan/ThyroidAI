"""
ThyroidAI ML Pipeline
======================
Full training pipeline for the Differentiated Thyroid Cancer Recurrence dataset.

Steps:
 1. Data loading
 2. Data validation
 3. Missing value handling
 4. Encoding categorical features (OneHotEncoder via ColumnTransformer)
 5. Feature scaling (StandardScaler for numeric)
 6. Model training (7 ML models + 1 DL model)
 7. Model evaluation (Accuracy, F1, ROC-AUC, Confusion Matrix, ROC Curve)
 8. Model saving (best_model.pkl, preprocessing.pkl, metadata.json, metrics.json, shap_background.pkl)

Model selection priority: ROC-AUC -> F1 -> Recall
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix, classification_report)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

import shap

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Thyroid_Diff.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL = "Recurred"

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Data loading")
print("=" * 70)
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {DATA_PATH}")

# ---------------------------------------------------------------------------
# 2. DATA VALIDATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Data validation")
print("=" * 70)
assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found"
assert df.shape[0] > 0, "Dataset is empty"
duplicate_count = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_count}")
print(f"Target classes: {df[TARGET_COL].unique().tolist()}")
print(f"Class balance:\n{df[TARGET_COL].value_counts()}")

# ---------------------------------------------------------------------------
# 3. MISSING VALUE HANDLING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Missing value handling")
print("=" * 70)
missing_total = df.isnull().sum().sum()
print(f"Missing values found: {missing_total}")
if missing_total > 0:
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].fillna(df[c].mode()[0])
        else:
            df[c] = df[c].fillna(df[c].median())
    print("Missing values imputed (mode for categorical, median for numeric).")
else:
    print("No missing values — dataset is clean.")

feature_cols = [c for c in df.columns if c != TARGET_COL]
numeric_features = ["Age"]
categorical_features = [c for c in feature_cols if c not in numeric_features]

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df[TARGET_COL])  # No=0, Yes=1
X = df[feature_cols].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------------------
# 4 & 5. ENCODING + SCALING (ColumnTransformer inside a sklearn Pipeline)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4-5: Encoding categorical features + scaling numeric features")
print("=" * 70)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
preprocessor.fit(X_train)
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)
print(f"Encoded feature dimensionality: {X_train_t.shape[1]}")

# ---------------------------------------------------------------------------
# 6 & 7. MODEL TRAINING + EVALUATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6-7: Model training + evaluation")
print("=" * 70)

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

sk_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
    "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss",
                              random_state=RANDOM_STATE, scale_pos_weight=pos_weight),
    "LightGBM": LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                random_state=RANDOM_STATE, class_weight="balanced", verbosity=-1),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []
fitted_models = {}
roc_curves = {}

for name, model in sk_models.items():
    cv_scores = cross_val_score(model, X_train_t, y_train, cv=skf, scoring="roc_auc")
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    metrics = {
        "model": name,
        "type": "ML",
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    results.append(metrics)
    fitted_models[name] = model
    print(f"{name:22s} | Acc {metrics['accuracy']:.4f} | F1 {metrics['f1']:.4f} | ROC-AUC {metrics['roc_auc']:.4f}")

# ---- Deep Learning: TensorFlow/Keras ANN ----
print("\nTraining Deep Learning model: TensorFlow/Keras ANN")
n_features = X_train_t.shape[1]
class_weight_dict = {0: 1.0, 1: float(pos_weight)}

ann = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
ann.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")])

early_stop = callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=25,
                                      restore_best_weights=True)

X_train_dense = X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t
X_test_dense = X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t

history = ann.fit(
    X_train_dense, y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=0,
)

y_proba_ann = ann.predict(X_test_dense, verbose=0).ravel()
y_pred_ann = (y_proba_ann >= 0.5).astype(int)
fpr, tpr, _ = roc_curve(y_test, y_proba_ann)
roc_curves["Neural Network (Keras ANN)"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

ann_metrics = {
    "model": "Neural Network (Keras ANN)",
    "type": "DL",
    "cv_roc_auc_mean": None,
    "cv_roc_auc_std": None,
    "accuracy": float(accuracy_score(y_test, y_pred_ann)),
    "precision": float(precision_score(y_test, y_pred_ann)),
    "recall": float(recall_score(y_test, y_pred_ann)),
    "f1": float(f1_score(y_test, y_pred_ann)),
    "roc_auc": float(roc_auc_score(y_test, y_proba_ann)),
}
results.append(ann_metrics)
print(f"{'Neural Network (ANN)':22s} | Acc {ann_metrics['accuracy']:.4f} | F1 {ann_metrics['f1']:.4f} | ROC-AUC {ann_metrics['roc_auc']:.4f} | epochs run: {len(history.history['loss'])}")

# ---------------------------------------------------------------------------
# MODEL SELECTION: priority ROC-AUC -> F1 -> Recall
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values(
    by=["roc_auc", "f1", "recall"], ascending=False
).reset_index(drop=True)
results_df = results_df.where(pd.notnull(results_df), None)  # NaN -> None (JSON-safe null)
print("\n" + "=" * 70)
print("MODEL COMPARISON (sorted by ROC-AUC, then F1, then Recall)")
print("=" * 70)
print(results_df.to_string(index=False))

best_row = results_df.iloc[0]
best_name = best_row["model"]
print(f"\n>>> SELECTED BEST MODEL: {best_name}")

is_ann_best = best_name == "Neural Network (Keras ANN)"

# ---------------------------------------------------------------------------
# 8. MODEL SAVING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8: Model saving")
print("=" * 70)

joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessing.pkl"))
joblib.dump(target_encoder, os.path.join(MODEL_DIR, "target_encoder.pkl"))
results_df.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"), index=False)

with open(os.path.join(MODEL_DIR, "roc_curves.json"), "w") as f:
    json.dump(roc_curves, f)

if is_ann_best:
    ann.save(os.path.join(MODEL_DIR, "best_model_ann.keras"))
    joblib.dump(None, os.path.join(MODEL_DIR, "best_model.pkl"))  # placeholder marker
else:
    joblib.dump(fitted_models[best_name], os.path.join(MODEL_DIR, "best_model.pkl"))

# Confusion matrix for best model
if is_ann_best:
    y_pred_best = y_pred_ann
else:
    y_pred_best = fitted_models[best_name].predict(X_test_t)
cm = confusion_matrix(y_test, y_pred_best)

# categorical options for the frontend form
categorical_options = {c: sorted(df[c].astype(str).unique().tolist()) for c in categorical_features}

metadata = {
    "best_model_name": best_name,
    "best_model_type": best_row["type"],
    "is_keras_model": bool(is_ann_best),
    "feature_cols": feature_cols,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "categorical_options": categorical_options,
    "target_classes": target_encoder.classes_.tolist(),
    "confusion_matrix": cm.tolist(),
    "dataset_info": {
        "n_samples": int(df.shape[0]),
        "n_features": len(feature_cols),
        "class_balance": df[TARGET_COL].value_counts().to_dict(),
    },
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2, default=str)

metrics_out = {
    "best_model": best_name,
    "selection_priority": ["roc_auc", "f1", "recall"],
    "all_models": results_df.to_dict(orient="records"),
    "best_model_metrics": best_row.to_dict(),
}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_out, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# SHAP background + explainer setup
# ---------------------------------------------------------------------------
print("\nBuilding SHAP background dataset...")
# Use a small representative sample of the training set as the SHAP background
background_idx = np.random.RandomState(RANDOM_STATE).choice(
    X_train_t.shape[0], size=min(50, X_train_t.shape[0]), replace=False
)
X_background = X_train_dense[background_idx]
joblib.dump(X_background, os.path.join(MODEL_DIR, "shap_background.pkl"))

# Get one-hot feature names for SHAP-friendly display
ohe = preprocessor.named_transformers_["cat"]
ohe_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
all_encoded_feature_names = numeric_features + ohe_feature_names
with open(os.path.join(MODEL_DIR, "encoded_feature_names.json"), "w") as f:
    json.dump(all_encoded_feature_names, f)

print("\nAll artifacts saved to", MODEL_DIR)
print(os.listdir(MODEL_DIR))
