# ThyroidAI — Explainable Thyroid Cancer Recurrence Prediction System

A full-stack, explainable ML/DL healthcare application predicting recurrence of
differentiated thyroid cancer, built on the UCI **Differentiated Thyroid Cancer
Recurrence** dataset (Borzooei & Tarokhian, 2023 — 383 patients, 16
clinicopathologic features, CC BY 4.0). Public demo system — no auth, opens
directly into the AI dashboard.

## Project structure

```
thyroid-ai-platform/
├── frontend/                     # Next.js 16 + TypeScript + Tailwind + shadcn-style UI
│   ├── src/app/                  # landing, /predict, /explainability, /models, /analytics
│   ├── src/components/ui/        # hand-built shadcn-pattern components (Radix + CVA)
│   ├── src/lib/                  # api.ts (axios client), types.ts, utils.ts
│   └── Dockerfile
├── backend/                      # FastAPI service
│   ├── app/
│   │   ├── main.py               # app factory, CORS, startup model load
│   │   ├── routers/api.py        # /api/health, /features, /model-info, /predict, /analytics
│   │   ├── core/model_loader.py  # loads artifacts, runs inference + SHAP
│   │   ├── core/analytics.py     # EDA aggregation for the analytics dashboard
│   │   └── schemas/schemas.py    # Pydantic request/response models
│   ├── ml_pipeline/train.py      # full training pipeline (see below)
│   ├── models/                   # generated artifacts (see below)
│   ├── data/Thyroid_Diff.csv     # dataset
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml            # orchestrates both services
├── docker/                       # reserved for extra deployment config
└── README.md
```

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, TypeScript, Tailwind CSS, shadcn-pattern components (Radix + CVA), Framer Motion, Recharts |
| Backend | FastAPI (Python), Pydantic v2 |
| ML | scikit-learn, XGBoost, LightGBM |
| DL | TensorFlow / Keras |
| Explainability | SHAP |

## ML pipeline (`backend/ml_pipeline/train.py`)

1. **Data loading** — reads `data/Thyroid_Diff.csv`
2. **Data validation** — schema/target checks, duplicate + class-balance report
3. **Missing value handling** — mode/median imputation (dataset has 0 missing values, but the step is real and runs regardless)
4. **Encoding** — `OneHotEncoder` for the 15 categorical features
5. **Feature scaling** — `StandardScaler` for `Age`
6. **Model training** — steps 4–5 are composed in a single `ColumnTransformer` fit inside the pipeline; 7 ML models + 1 DL model trained on the transformed features
7. **Model evaluation** — 5-fold stratified CV (ROC-AUC) + held-out 20% test set (accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve)
8. **Model saving** — best model auto-selected by **ROC-AUC → F1 → Recall**, saved with all preprocessing/metadata needed to serve it

### Models trained

**Machine Learning:** Logistic Regression, Random Forest, SVM (RBF), KNN, Gradient Boosting, XGBoost, LightGBM
**Deep Learning:** Keras ANN — `Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.2) → Dense(32, relu) → Dense(1, sigmoid)`, trained with class weights, a validation split, and early stopping on validation AUC.

### Result (this run)

**Random Forest** was selected as the best model — **96.1% test accuracy**, **ROC-AUC ≈ 0.995**. Full comparison table is served live at `/api/model-info` and rendered in the frontend's Model Bench page. Numbers can shift slightly between reruns due to the DL model's stochastic training; the selection logic always re-picks the actual best performer.

### Saved artifacts (`backend/models/`)

| File | Contents |
|---|---|
| `best_model.pkl` | The winning scikit-learn/XGBoost/LightGBM model (or a placeholder if the ANN wins) |
| `best_model_ann.keras` | Saved Keras model, present only if the ANN was selected |
| `preprocessing.pkl` | Fitted `ColumnTransformer` (encoding + scaling) |
| `target_encoder.pkl` | Label encoder for `Recurred` |
| `metadata.json` | Feature list, categorical options (drives the frontend form), confusion matrix, dataset info |
| `metrics.json` | Full model comparison + best model metrics |
| `model_comparison.csv` | Same comparison, CSV form |
| `roc_curves.json` | FPR/TPR arrays per model, for the ROC chart |
| `shap_background.pkl` | Background sample used by the SHAP explainer |
| `encoded_feature_names.json` | Post-one-hot-encoding feature names, for mapping SHAP values back to original fields |

## Backend API

Base URL: `http://localhost:8000`

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Service + best-model status |
| GET | `/api/features` | Feature list + valid categorical options (drives the form) |
| GET | `/api/model-info` | Full model comparison table, ROC curves, confusion matrix |
| GET | `/api/analytics` | EDA aggregates for the analytics dashboard |
| POST | `/api/predict` | Run a prediction on a patient profile |

### `POST /api/predict`

Request:
```json
{
  "Age": 40, "Gender": "F", "Smoking": "No", "Hx Smoking": "No",
  "Hx Radiothreapy": "No", "Thyroid Function": "Euthyroid",
  "Physical Examination": "Multinodular goiter", "Adenopathy": "No",
  "Pathology": "Papillary", "Focality": "Uni-Focal", "Risk": "Low",
  "T": "T1a", "N": "N0", "M": "M0", "Stage": "I", "Response": "Excellent"
}
```

Response:
```json
{
  "prediction": "No",
  "probability": 0.0431,
  "confidence": "High",
  "model_used": "Random Forest",
  "explanation": ["Reduces risk: Response = Excellent", "..."],
  "shap_factors": [{"feature": "Response", "value": "Excellent", "impact": -0.1442, "direction": "decreases"}],
  "probabilities": {"No": 0.9569, "Yes": 0.0431}
}
```

Invalid categorical values return `422` with the list of valid options for that field.

## Frontend pages

- **`/`** — landing page: project intro, AI workflow, dataset info, live model performance summary
- **`/predict`** — clinical prediction dashboard: dynamic form (built from `/api/features`), animated result card, probability meter, risk badge
- **`/explainability`** — SHAP dashboard for the most recent prediction: horizontal bar chart of feature contributions, risk-increasing/decreasing factor lists, plain-language patient explanation
- **`/models`** — model performance dashboard: comparison table, metric bar chart, ROC curves, confusion matrix
- **`/analytics`** — dataset analytics: target distribution, age histogram, and recurrence-rate breakdowns by gender/risk/stage/response/TNM/smoking/adenopathy/focality/pathology, all as interactive Recharts

## Running locally (without Docker)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python ml_pipeline/train.py        # trains everything, writes models/
uvicorn app.main:app --host 0.0.0.0 --port 8000

py -3.10 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Frontend** (in a second terminal):
```bash
cd frontend
npm install
npm run build && npm run start     # or `npm run dev` for local development
```

Then open **http://localhost:3000**. Set `NEXT_PUBLIC_API_URL` in
`frontend/.env.local` if the backend runs somewhere other than
`http://localhost:8000`.

## Running with Docker

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000 (interactive docs at `/docs`)

The compose file mounts `backend/models` and `backend/data` as volumes, so you
can retrain (`python ml_pipeline/train.py`) without rebuilding the image.

## Deployment

The images are stateless aside from the mounted `models/`/`data/` folders, so
they deploy the same way to **Render**, **Railway**, **AWS** (ECS/Fargate or
App Runner), or **Azure** (Container Apps): push the images, set
`NEXT_PUBLIC_API_URL` on the frontend service to the backend's public URL, and
expose ports 3000/8000.

## Retraining on new data

Replace `backend/data/Thyroid_Diff.csv` with a new file using the same 17
columns, then re-run `python ml_pipeline/train.py`. It regenerates every file
in `backend/models/`, including `metadata.json` (which drives the frontend
form) — no frontend or API code changes needed.

## Notes & limitations

- This is a research/educational/portfolio demonstration, **not a certified
  clinical decision tool**. It is not a substitute for clinical judgment.
- No authentication is implemented by design — this is a public,
  no-login AI demonstration.
- SHAP explanations use a permutation-based `shap.Explainer` over a 50-sample
  background set for robustness across whichever model type ends up selected
  as "best" (tree, linear, kernel, or the Keras ANN).
- Docker builds were authored to standard conventions but could not be
  executed inside the sandbox used to build this project (no Docker daemon /
  restricted registry access); please validate `docker compose up --build`
  in your own environment.
