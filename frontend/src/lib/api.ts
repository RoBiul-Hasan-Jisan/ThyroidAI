import axios from "axios";
import {
  FeaturesResponse,
  ModelInfoResponse,
  PredictionResponse,
  AnalyticsResponse,
  ModelsResponse,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const client = axios.create({ 
  baseURL: API_URL, 
  timeout: 20000,
  headers: {
    'Content-Type': 'application/json',
  }
});

// ============================================
// Core API Functions
// ============================================

export async function getFeatures(): Promise<FeaturesResponse> {
  const { data } = await client.get("/api/features");
  return data;
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  const { data } = await client.get("/api/model-info");
  return data;
}

export async function getAnalytics(): Promise<AnalyticsResponse> {
  const { data } = await client.get("/api/analytics");
  return data;
}

export async function getHealth(): Promise<{ 
  status: string; 
  best_model: string; 
  model_type: string;
  features?: number;
  is_keras?: boolean;
}> {
  const { data } = await client.get("/api/health");
  return data;
}

// ============================================
// Model Management Functions
// ============================================

/**
 * Get all available models with their metrics
 */
export async function getModels(): Promise<ModelsResponse> {
  const { data } = await client.get("/api/models");
  return data;
}

/**
 * Get model comparison metrics
 */
export async function getModelComparison(): Promise<any> {
  const { data } = await client.get("/api/models/comparison");
  return data;
}

// ============================================
// Prediction Functions
// ============================================

/**
 * Make a prediction using a specific model
 * @param payload - Patient data
 * @param modelChoice - Model key ('best', 'random_forest', 'xgboost', 'ann', etc.)
 */
export async function predictRecurrence(
  payload: Record<string, string | number>,
  modelChoice: string = "best"
): Promise<PredictionResponse> {
  const { data } = await client.post("/api/predict", {
    patient: payload,
    model_choice: modelChoice,
  });
  return data;
}

/**
 * Make predictions using all available models for comparison
 */
export async function predictAllModels(
  payload: Record<string, string | number>
): Promise<{
  predictions: Record<string, PredictionResponse>;
  agreement: boolean;
}> {
  const { data } = await client.post("/api/predict-all", {
    patient: payload,
  });
  return data;
}

// ============================================
// Model Information Functions
// ============================================

/**
 * Get detailed model metrics for a specific model
 */
export async function getModelMetrics(modelKey: string): Promise<any> {
  const { data } = await client.get(`/api/models/${modelKey}/metrics`);
  return data;
}

/**
 * Get SHAP explanation for a specific prediction
 */
export async function getExplanation(
  payload: Record<string, string | number>,
  modelChoice: string = "best"
): Promise<{
  shap_factors: any[];
  explanation: string[];
}> {
  const { data } = await client.post("/api/explain", {
    patient: payload,
    model_choice: modelChoice,
  });
  return data;
}

// ============================================
// Feature Information Functions
// ============================================

/**
 * Get feature importance for a specific model
 */
export async function getFeatureImportance(modelKey: string = "best"): Promise<{
  features: { feature: string; importance: number }[];
}> {
  const { data } = await client.get(`/api/features/importance?model_key=${modelKey}`);
  return data;
}

/**
 * Validate patient data before prediction
 */
export async function validatePatientData(
  payload: Record<string, string | number>
): Promise<{ valid: boolean; errors?: string[] }> {
  const { data } = await client.post("/api/validate", payload);
  return data;
}

// ============================================
// Utility Functions
// ============================================

/**
 * Get the best model info
 */
export async function getBestModel(): Promise<{
  name: string;
  key: string;
  type: string;
  metrics: any;
}> {
  const { data } = await client.get("/api/models/best");
  return data;
}

/**
 * Download model artifacts (for admin/debug)
 */
export async function downloadModelArtifacts(modelKey: string): Promise<Blob> {
  const response = await client.get(`/api/models/${modelKey}/download`, {
    responseType: 'blob',
  });
  return response.data;
}