"use client";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
  Select, SelectTrigger, SelectValue, SelectContent, SelectItem,
} from "@/components/ui/select";
import { getFeatures, predictRecurrence, getModels } from "@/lib/api";
import { FeaturesResponse, PredictionResponse } from "@/lib/types";
import { AlertTriangle, CheckCircle2, Loader2, Sparkles, Brain, TrendingUp, BarChart3 } from "lucide-react";

const NICE_LABELS: Record<string, string> = {
  Age: "Age (years)",
  Gender: "Gender",
  Smoking: "Currently smokes",
  "Hx Smoking": "History of smoking",
  "Hx Radiothreapy": "History of radiotherapy",
  "Thyroid Function": "Thyroid function",
  "Physical Examination": "Physical examination",
  Adenopathy: "Adenopathy",
  Pathology: "Pathology",
  Focality: "Focality",
  Risk: "Risk category",
  T: "Tumor stage (T)",
  N: "Node stage (N)",
  M: "Metastasis stage (M)",
  Stage: "Overall stage",
  Response: "Treatment response",
};

const MODEL_TYPE_COLORS = {
  "ML": "bg-blue-100 text-blue-700 border-blue-200",
  "DL": "bg-purple-100 text-purple-700 border-purple-200"
};

const MODEL_COLORS = {
  "Random Forest": "#0f766e",
  "XGBoost": "#2563eb",
  "Neural Network (Keras ANN)": "#d97706",
  "Logistic Regression": "#7c3aed",
  "SVM": "#dc2626",
  "Gradient Boosting": "#059669",
  "LightGBM": "#db2777",
  "KNN": "#0891b2"
};

const DEFAULT_PATIENT: Record<string, string | number> = {
  Age: 45,
  Gender: "F",
  Smoking: "No",
  "Hx Smoking": "No",
  "Hx Radiothreapy": "No",
  "Thyroid Function": "Euthyroid",
  "Physical Examination": "Multinodular goiter",
  Adenopathy: "No",
  Pathology: "Papillary",
  Focality: "Uni-Focal",
  Risk: "Low",
  T: "T1a",
  N: "N0",
  M: "M0",
  Stage: "I",
  Response: "Excellent",
};

// Make path optional to match API response
interface ModelInfo {
  key: string;
  name: string;
  type: string;
  is_keras: boolean;
  is_best: boolean;
  path?: string; // Made optional
  metrics: {
    model?: string;
    type?: string;
    cv_roc_auc_mean: number | null;
    cv_roc_auc_std: number | null;
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc: number;
  };
}

export default function PredictPage() {
  const [features, setFeatures] = useState<FeaturesResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("best");
  const [patient, setPatient] = useState<Record<string, string | number>>(DEFAULT_PATIENT);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Load features and models
    Promise.all([
      getFeatures().catch(() => null),
      getModels().catch(() => null)
    ]).then(([featuresData, modelsData]) => {
      if (featuresData) setFeatures(featuresData);
      if (modelsData) {
        // Map the response to ModelInfo type with proper typing
        const mappedModels: ModelInfo[] = (modelsData.available_models || []).map((m: any) => ({
          key: m.key,
          name: m.name,
          type: m.type,
          is_keras: m.is_keras,
          is_best: m.is_best || false,
          path: m.path || "",
          metrics: {
            accuracy: m.metrics?.accuracy || 0,
            f1: m.metrics?.f1 || 0,
            roc_auc: m.metrics?.roc_auc || 0,
            recall: m.metrics?.recall || 0,
            precision: m.metrics?.precision || 0,
            cv_roc_auc_mean: m.metrics?.cv_roc_auc_mean || null,
            cv_roc_auc_std: m.metrics?.cv_roc_auc_std || null,
          }
        }));
        setModels(mappedModels);
        
        // Set default selected model to best
        const bestModel = mappedModels.find((m: ModelInfo) => m.is_best);
        if (bestModel) setSelectedModel(bestModel.key);
      }
    }).catch(() => setError("Could not reach the ThyroidAI API."));
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictRecurrence(patient, selectedModel);
      setResult(data);
      if (typeof window !== "undefined") {
        window.localStorage.setItem("thyroidai_last_result", JSON.stringify(data));
        window.localStorage.setItem("thyroidai_last_patient", JSON.stringify(patient));
        window.localStorage.setItem("thyroidai_last_model", selectedModel);
      }
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Prediction failed. Please check your inputs.");
    } finally {
      setLoading(false);
    }
  }

  const isHighRisk = result?.prediction === "Yes";
  const riskPct = result ? Math.round(result.probability * 100) : 0;
  
  // Get selected model info
  const selectedModelInfo = models.find(m => m.key === selectedModel);
  const isBestModel = selectedModelInfo?.is_best || false;

  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <div className="mb-8">
        <Badge variant="teal" className="mb-3">Clinical Prediction Dashboard</Badge>
        <h1 className="text-3xl font-bold text-slate-900">Thyroid Recurrence Risk</h1>
        <p className="mt-2 max-w-2xl text-slate-600">
          Enter a patient&apos;s clinicopathologic profile to estimate recurrence risk using
          your preferred model.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-5">
        {/* FORM - Left Side */}
        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Patient profile</span>
              <div className="flex items-center gap-2">
                <Brain className="h-4 w-4 text-teal-600" />
                <span className="text-sm font-normal text-slate-500">16 inputs</span>
              </div>
            </CardTitle>
            <CardDescription>Complete all fields for accurate prediction</CardDescription>
          </CardHeader>
          <CardContent>
            {!features ? (
              <p className="text-sm text-slate-400">Loading form fields from the API…</p>
            ) : (
              <form onSubmit={handleSubmit} className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                {features.feature_cols.map((col) => (
                  <div key={col} className="flex flex-col gap-1.5">
                    <Label>{NICE_LABELS[col] ?? col}</Label>
                    {features.numeric_features.includes(col) ? (
                      <Input
                        type="number"
                        required
                        value={patient[col]}
                        onChange={(e) =>
                          setPatient((p) => ({ ...p, [col]: Number(e.target.value) }))
                        }
                      />
                    ) : (
                      <Select
                        value={String(patient[col])}
                        onValueChange={(v) => setPatient((p) => ({ ...p, [col]: v }))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {features.categorical_options[col]?.map((opt) => (
                            <SelectItem key={opt} value={opt}>
                              {opt}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    )}
                  </div>
                ))}
                <div className="sm:col-span-2 mt-2 flex gap-3">
                  <Button type="submit" variant="primary" size="lg" className="flex-1" disabled={loading}>
                    {loading ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" /> Running prediction…
                      </>
                    ) : (
                      "Predict Recurrence Risk"
                    )}
                  </Button>
                </div>
              </form>
            )}
            {error && (
              <div className="mt-4 flex items-center gap-2 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
                <AlertTriangle className="h-4 w-4" /> {error}
              </div>
            )}
          </CardContent>
        </Card>

        {/* RESULT & MODEL SELECTION - Right Side */}
        <div className="lg:col-span-2 space-y-4">
          {/* Model Selection Card */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <BarChart3 className="h-4 w-4 text-teal-600" />
                Select Model
              </CardTitle>
              <CardDescription>Choose which model to use</CardDescription>
            </CardHeader>
            <CardContent>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {models.map((model) => (
                    <SelectItem key={model.key} value={model.key}>
                      <div className="flex items-center gap-2">
                        <span>{model.name}</span>
                        {model.is_best && (
                          <Badge variant="success" className="text-xs">Best</Badge>
                        )}
                        <Badge 
                          variant="outline" 
                          className={`text-xs ${MODEL_TYPE_COLORS[model.type as keyof typeof MODEL_TYPE_COLORS]}`}
                        >
                          {model.type}
                        </Badge>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {selectedModelInfo && (
                <motion.div
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-3 rounded-lg bg-slate-50 p-3"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{selectedModelInfo.name}</span>
                    {selectedModelInfo.is_best && (
                      <Badge variant="success" className="text-xs">⭐ Best Performing</Badge>
                    )}
                  </div>
                  <div className="mt-2 grid grid-cols-3 gap-1 text-xs">
                    <div className="text-slate-500">Accuracy</div>
                    <div className="text-slate-500">F1</div>
                    <div className="text-slate-500">ROC-AUC</div>
                    <div className="font-mono font-semibold">
                      {(selectedModelInfo.metrics?.accuracy * 100 || 0).toFixed(1)}%
                    </div>
                    <div className="font-mono font-semibold">
                      {(selectedModelInfo.metrics?.f1 || 0).toFixed(3)}
                    </div>
                    <div className="font-mono font-semibold">
                      {(selectedModelInfo.metrics?.roc_auc || 0).toFixed(3)}
                    </div>
                  </div>
                  {selectedModelInfo.is_keras && (
                    <div className="mt-1 text-xs text-purple-600">🧠 Deep Learning Model</div>
                  )}
                </motion.div>
              )}
            </CardContent>
          </Card>

          {/* Prediction Result Card */}
          <Card className="sticky top-20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-teal-700" /> Prediction result
              </CardTitle>
              <CardDescription>
                {result ? (
                  <span className="flex items-center gap-1">
                    Model: {result.model_used}
                    {isBestModel && <Badge variant="success" className="text-xs ml-1">Best</Badge>}
                  </span>
                ) : (
                  "Awaiting patient data"
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <AnimatePresence mode="wait">
                {result ? (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.35 }}
                  >
                    <div className="flex items-center gap-3">
                      {isHighRisk ? (
                        <AlertTriangle className="h-8 w-8 text-red-600" />
                      ) : (
                        <CheckCircle2 className="h-8 w-8 text-emerald-600" />
                      )}
                      <div>
                        <div
                          className={`text-xl font-bold ${
                            isHighRisk ? "text-red-700" : "text-emerald-700"
                          }`}
                        >
                          {isHighRisk ? "HIGH RECURRENCE RISK" : "LOW RECURRENCE RISK"}
                        </div>
                        <div className="text-sm text-slate-500">
                          Prediction: Recurred = {result.prediction}
                        </div>
                      </div>
                    </div>

                    <div className="mt-6">
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-medium text-slate-700">Probability</span>
                        <span className="font-mono font-semibold text-slate-900">{riskPct}%</span>
                      </div>
                      <Progress
                        value={riskPct}
                        className="mt-2"
                        indicatorClassName={isHighRisk ? "bg-red-600" : "bg-emerald-600"}
                      />
                    </div>

                    <div className="mt-4 flex items-center gap-2">
                      <span className="text-sm text-slate-500">Confidence:</span>
                      <Badge
                        variant={
                          result.confidence === "High"
                            ? "success"
                            : result.confidence === "Medium"
                            ? "warning"
                            : "outline"
                        }
                      >
                        {result.confidence}
                      </Badge>
                    </div>

                    <div className="mt-6">
                      <div className="mb-2 text-sm font-semibold text-slate-700">
                        Top contributing factors
                      </div>
                      <ul className="space-y-1.5">
                        {result.explanation.map((line, i) => (
                          <li
                            key={i}
                            className={`flex items-start gap-2 text-sm ${
                              line.startsWith("Increases") ? "text-red-700" : "text-emerald-700"
                            }`}
                          >
                            <span>{line.startsWith("Increases") ? "▲" : "▼"}</span>
                            <span>{line}</span>
                          </li>
                        ))}
                      </ul>
                      <p className="mt-3 text-xs text-slate-400">
                        See the Explainability tab for a full SHAP breakdown of this prediction.
                      </p>
                    </div>
                  </motion.div>
                ) : (
                  <motion.p
                    key="placeholder"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-sm text-slate-400"
                  >
                    Fill in the patient profile and click &ldquo;Predict Recurrence Risk&rdquo; to see
                    the estimated risk here.
                  </motion.p>
                )}
              </AnimatePresence>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}