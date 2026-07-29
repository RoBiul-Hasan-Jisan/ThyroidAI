"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { getModelInfo } from "@/lib/api";
import { ModelInfoResponse } from "@/lib/types";
import {
  Database, GitBranch, Cpu, ShieldCheck, ArrowRight,
  Microscope, Layers, Sparkles, Loader2,
} from "lucide-react";

const WORKFLOW_STEPS = [
  { icon: Database, title: "Clinical Data", desc: "383 patients, 16 clinicopathologic features collected over 15 years." },
  { icon: Layers, title: "Preprocessing", desc: "Missing-value handling, one-hot encoding, and feature scaling via a sklearn Pipeline." },
  { icon: Cpu, title: "ML + DL Training", desc: "7 classical ML models and a Keras ANN trained and cross-validated." },
  { icon: GitBranch, title: "Model Selection", desc: "Best model auto-selected by ROC-AUC, then F1, then Recall." },
  { icon: Microscope, title: "Explainability", desc: "SHAP attributes every prediction to the clinical factors driving it." },
  { icon: ShieldCheck, title: "Clinical Decision Support", desc: "Served through a REST API to a live prediction dashboard." },
];

export default function HomePage() {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .catch((err) => {
        console.error("Error loading model info:", err);
        setError("Failed to load model performance data.");
      })
      .finally(() => setLoading(false));
  }, []);

  // Safe metric formatting helper
  const formatMetric = (value: any, decimals: number = 3) => {
    if (value === undefined || value === null || isNaN(value)) {
      return "—";
    }
    return value.toFixed(decimals);
  };

  const formatPercent = (value: any) => {
    if (value === undefined || value === null || isNaN(value)) {
      return "—";
    }
    return `${(value * 100).toFixed(1)}%`;
  };

  const best = info?.best_model_metrics;
  const bestModelName = info?.best_model || "—";

  // Check if metrics are available
  const hasMetrics = best && best.accuracy !== undefined && best.accuracy !== null;

  return (
    <div className="mx-auto max-w-7xl px-6 py-14">
      {/* HERO */}
      <div className="grid grid-cols-1 gap-10 lg:grid-cols-2 lg:items-center">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Badge variant="teal" className="mb-4">Explainable AI &middot; Healthcare</Badge>
          <h1 className="text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl">
            Predicting thyroid cancer recurrence — with a model that shows its work.
          </h1>
          <p className="mt-5 max-w-xl text-lg text-slate-600">
            ThyroidAI combines classical machine learning, deep learning, and SHAP-based
            explainability to estimate recurrence risk in differentiated thyroid cancer
            patients from a 15-year clinical cohort.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Button asChild size="lg" variant="primary">
              <Link href="/predict">Run a prediction <ArrowRight className="h-4 w-4" /></Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <Link href="/models">View model bench</Link>
            </Button>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.15 }}
        >
          <Card className="border-teal-100 bg-gradient-to-br from-teal-50 to-white">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-teal-700" /> Model performance summary
              </CardTitle>
              <CardDescription>Best-performing model, held-out test set</CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-8 w-8 animate-spin text-teal-600" />
                  <span className="ml-3 text-slate-500">Loading metrics…</span>
                </div>
              ) : error ? (
                <div className="text-center py-8 text-slate-500">
                  <p className="text-sm">{error}</p>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="mt-3"
                    onClick={() => window.location.reload()}
                  >
                    Retry
                  </Button>
                </div>
              ) : hasMetrics ? (
                <div className="grid grid-cols-2 gap-4">
                  <Stat label="Best model" value={bestModelName} wide />
                  <Stat label="Accuracy" value={formatPercent(best.accuracy)} />
                  <Stat label="F1 Score" value={formatMetric(best.f1)} />
                  <Stat label="ROC-AUC" value={formatMetric(best.roc_auc)} />
                  <Stat label="Recall" value={formatMetric(best.recall)} />
                  <Stat label="Precision" value={formatMetric(best.precision)} />
                </div>
              ) : (
                <div className="text-center py-8 text-slate-500">
                  <p className="text-sm">No model metrics available.</p>
                  <p className="text-xs mt-1">Please ensure models are trained and loaded.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* WORKFLOW */}
      <section className="mt-24">
        <h2 className="text-2xl font-semibold text-slate-900">AI healthcare workflow</h2>
        <p className="mt-2 max-w-2xl text-slate-600">
          From raw clinical records to an explainable, clinician-facing risk score.
        </p>
        <div className="mt-8 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {WORKFLOW_STEPS.map((step, i) => {
            const Icon = step.icon;
            return (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.05 }}
              >
                <Card className="h-full">
                  <CardContent className="pt-6">
                    <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-teal-800 text-white">
                      <Icon className="h-5 w-5" />
                    </div>
                    <div className="font-semibold text-slate-900">{step.title}</div>
                    <p className="mt-1 text-sm text-slate-500">{step.desc}</p>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* DATASET INFO */}
      <section className="mt-24 grid grid-cols-1 gap-10 lg:grid-cols-2">
        <div>
          <h2 className="text-2xl font-semibold text-slate-900">About the dataset</h2>
          <p className="mt-3 text-slate-600">
            The UCI Differentiated Thyroid Cancer Recurrence dataset was collected over
            15 years, with each of the 383 patients followed for at least 10 years. Each
            record captures 16 clinicopathologic features spanning demographics, treatment
            history, pathology, and TNM staging.
          </p>
          <ul className="mt-5 space-y-2 text-sm text-slate-600">
            <li>• 383 patients &middot; 16 features &middot; binary target (Recurred: Yes/No)</li>
            <li>• Real, categorical, and integer feature types</li>
            <li>• Source: Borzooei &amp; Tarokhian (2023), UCI ML Repository, CC BY 4.0</li>
          </ul>
        </div>
        <div>
          <h2 className="text-2xl font-semibold text-slate-900">Research motivation</h2>
          <p className="mt-3 text-slate-600">
            Recurrence risk stratification directly informs follow-up intensity and
            treatment decisions in differentiated thyroid cancer. A model that is both
            accurate <em>and</em> interpretable can support — not replace — clinical judgment,
            by surfacing which specific clinicopathologic factors are driving an individual
            patient&apos;s risk estimate rather than acting as a black box.
          </p>
        </div>
      </section>
    </div>
  );
}

function Stat({ label, value, wide }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={wide ? "col-span-2" : ""}>
      <div className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 text-2xl font-bold text-teal-900">{value}</div>
    </div>
  );
}