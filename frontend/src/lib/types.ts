export interface ModelsResponse {
  available_models: {
    key: string;
    name: string;
    type: string;
    is_keras: boolean;
    is_best: boolean;
    metrics: {
      accuracy: number;
      f1: number;
      roc_auc: number;
      recall: number;
      precision: number;
      cv_roc_auc_mean: number | null;
      cv_roc_auc_std: number | null;
    };
  }[];
  best_model: string;
  model_comparison: any[];
}

export interface PredictionResponse {
  prediction: string;
  probability: number;
  confidence: string;
  model_used: string;
  model_key: string;
  explanation: string[];
  shap_factors: {
    feature: string;
    value: string;
    impact: number;
    direction: string;
  }[];
  probabilities: {
    No: number;
    Yes: number;
  };
}

export interface FeaturesResponse {
  feature_cols: string[];
  numeric_features: string[];
  categorical_features: string[];
  categorical_options: Record<string, string[]>;
  target_classes: string[];
}

export interface ModelInfoResponse {
  best_model: string;
  best_model_type: string;
  selection_priority: string[];
  all_models: any[];
  best_model_metrics: any;
  roc_curves: Record<string, { fpr: number[]; tpr: number[] }>;
  confusion_matrix: number[][];
  target_classes: string[];
  dataset_info: {
    n_samples: number;
    n_features: number;
    class_balance: Record<string, number>;
  };
}

// ============================================
// RAG (medical context) types
// ============================================

export interface RagEvidenceChunk {
  text: string;
  source: string;
  document: string;
  section: string;
  page: number;
  score: number;
}

export type RagStatus = "completed" | "rag_unavailable" | "no_evidence";

export interface RagExplainResponse {
  status: RagStatus;
  summary?: string;
  clinical_context?: string;
  evidence: RagEvidenceChunk[];
  retrieval_method: string;
  limitations?: string;
  disclaimer: string;
  queries_used: string[];
}

export interface RagStatusResponse {
  rag_enabled: boolean;
  documents_ingested: boolean;
  num_chunks: number;
  ollama_available: boolean;
  ready: boolean;
}

// Add CrosstabRow interface
export interface CrosstabRow {
  category: string;
  No: number;
  Yes: number;
  total?: number;
  percentage?: number;
}

export interface AnalyticsResponse {
  n_samples: number;
  n_features: number;
  target_distribution: { name: string; value: number; percentage: number }[];
  age_histogram: { bin: string; No: number; Yes: number; total: number; recurrence_rate: number }[];
  class_imbalance?: {
    no_count: number;
    yes_count: number;
    total: number;
    no_percentage: number;
    yes_percentage: number;
    imbalance_ratio: number;
    imbalance_severity: string;
    recommendation: string;
  };
  gender_recurrence: CrosstabRow[];
  risk_recurrence: CrosstabRow[];
  stage_recurrence: CrosstabRow[];
  response_recurrence: CrosstabRow[];
  t_stage_recurrence: CrosstabRow[];
  n_stage_recurrence: CrosstabRow[];
  m_stage_recurrence: CrosstabRow[];
  smoking_recurrence: CrosstabRow[];
  adenopathy_recurrence: CrosstabRow[];
  focality_recurrence: CrosstabRow[];
  pathology_recurrence: CrosstabRow[];
  thyroid_function_recurrence?: CrosstabRow[];
  physical_examination_recurrence?: CrosstabRow[];
  correlation_analysis?: {
    top_positive: { feature: string; correlation: number }[];
    top_negative: { feature: string; correlation: number }[];
    all_features: { feature: string; correlation: number; abs_correlation: number }[];
  };
  age_analysis?: {
    overall: { mean: number; median: number; std: number; min: number; max: number };
    by_recurrence: {
      No: { mean: number; median: number; std: number; count: number };
      Yes: { mean: number; median: number; std: number; count: number };
    };
  };
}