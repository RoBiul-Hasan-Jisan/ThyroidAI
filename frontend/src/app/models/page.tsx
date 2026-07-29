"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getModelInfo } from "@/lib/api";
import { ModelInfoResponse } from "@/lib/types";

const COLORS = ["#0f766e", "#dc2626", "#2563eb", "#d97706", "#7c3aed", "#059669", "#db2777", "#0891b2"];

export default function ModelsPage() {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);

  useEffect(() => {
    getModelInfo().then(setInfo).catch(() => {});
  }, []);

  if (!info) {
    return (
      <div className="mx-auto max-w-6xl px-6 py-12">
        <p className="text-slate-400">Loading model performance data…</p>
      </div>
    );
  }

  const rocData = buildRocChartData(info.roc_curves);
  const barData = info.all_models.map((m) => ({
    name: m.model.replace("Neural Network (Keras ANN)", "Keras ANN"),
    Accuracy: m.accuracy,
    F1: m.f1,
    "ROC-AUC": m.roc_auc,
  }));

  const cm = info.confusion_matrix;
  const classes = info.target_classes;

  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <Badge variant="teal" className="mb-3">Model Bench</Badge>
      <h1 className="text-3xl font-bold text-slate-900">Model Performance Dashboard</h1>
      <p className="mt-2 max-w-2xl text-slate-600">
        {info.all_models.length} models trained and evaluated — {info.all_models.filter(m => m.type === "ML").length} classical
        ML models and {info.all_models.filter(m => m.type === "DL").length} deep learning model. Best model selected by{" "}
        {info.selection_priority.join(" → ")}.
      </p>

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
        {/* COMPARISON TABLE */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>Model comparison</CardTitle>
            <CardDescription>Sorted by ROC-AUC (selection priority)</CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-400">
                  <th className="py-2 pr-4">Model</th>
                  <th className="py-2 pr-4">Type</th>
                  <th className="py-2 pr-4 text-right">Accuracy</th>
                  <th className="py-2 pr-4 text-right">Precision</th>
                  <th className="py-2 pr-4 text-right">Recall</th>
                  <th className="py-2 pr-4 text-right">F1</th>
                  <th className="py-2 pr-4 text-right">ROC-AUC</th>
                </tr>
              </thead>
              <tbody>
                {info.all_models.map((m) => (
                  <tr
                    key={m.model}
                    className={`border-b border-slate-100 ${
                      m.model === info.best_model ? "bg-teal-50" : ""
                    }`}
                  >
                    <td className="py-2 pr-4 font-medium text-slate-800">
                      {m.model}
                      {m.model === info.best_model && (
                        <Badge variant="success" className="ml-2">Best</Badge>
                      )}
                    </td>
                    <td className="py-2 pr-4">
                      <Badge variant={m.type === "DL" ? "teal" : "outline"}>{m.type}</Badge>
                    </td>
                    <td className="py-2 pr-4 text-right font-mono">{m.accuracy.toFixed(3)}</td>
                    <td className="py-2 pr-4 text-right font-mono">{m.precision.toFixed(3)}</td>
                    <td className="py-2 pr-4 text-right font-mono">{m.recall.toFixed(3)}</td>
                    <td className="py-2 pr-4 text-right font-mono">{m.f1.toFixed(3)}</td>
                    <td className="py-2 pr-4 text-right font-mono">{m.roc_auc.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>

        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* MODEL COMPARISON BAR CHART */}
          <Card>
            <CardHeader>
              <CardTitle>Metric comparison</CardTitle>
              <CardDescription>Accuracy / F1 / ROC-AUC across all models</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barData} margin={{ left: -10 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-25} textAnchor="end" height={70} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="Accuracy" fill="#0f766e" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="F1" fill="#2563eb" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="ROC-AUC" fill="#d97706" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* ROC CURVE */}
          <Card>
            <CardHeader>
              <CardTitle>ROC curves</CardTitle>
              <CardDescription>True positive rate vs. false positive rate</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart margin={{ left: -10 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      type="number"
                      dataKey="fpr"
                      domain={[0, 1]}
                      tick={{ fontSize: 11 }}
                      label={{ value: "False Positive Rate", position: "bottom", fontSize: 11 }}
                    />
                    <YAxis
                      type="number"
                      domain={[0, 1]}
                      tick={{ fontSize: 11 }}
                      label={{ value: "TPR", angle: -90, position: "insideLeft", fontSize: 11 }}
                    />
                    <Tooltip />
                    <Legend />
                    {Object.keys(info.roc_curves).map((name, i) => (
                      <Line
                        key={name}
                        data={rocData[name]}
                        dataKey="tpr"
                        name={name.replace("Neural Network (Keras ANN)", "Keras ANN")}
                        stroke={COLORS[i % COLORS.length]}
                        dot={false}
                        strokeWidth={name === info.best_model ? 3 : 1.5}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* CONFUSION MATRIX */}
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Confusion matrix — {info.best_model}</CardTitle>
            <CardDescription>Held-out test set</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid max-w-md grid-cols-3 gap-1 text-center text-sm">
              <div />
              <div className="font-semibold text-slate-500">Pred: {classes[0]}</div>
              <div className="font-semibold text-slate-500">Pred: {classes[1]}</div>

              <div className="flex items-center justify-end pr-2 font-semibold text-slate-500">
                Actual: {classes[0]}
              </div>
              <ConfCell value={cm[0][0]} correct />
              <ConfCell value={cm[0][1]} />

              <div className="flex items-center justify-end pr-2 font-semibold text-slate-500">
                Actual: {classes[1]}
              </div>
              <ConfCell value={cm[1][0]} />
              <ConfCell value={cm[1][1]} correct />
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}

function ConfCell({ value, correct }: { value: number; correct?: boolean }) {
  return (
    <div
      className={`flex h-16 items-center justify-center rounded-md font-mono text-lg font-bold ${
        correct ? "bg-teal-100 text-teal-800" : "bg-red-50 text-red-700"
      }`}
    >
      {value}
    </div>
  );
}

function buildRocChartData(rocCurves: Record<string, { fpr: number[]; tpr: number[] }>) {
  const out: Record<string, { fpr: number; tpr: number }[]> = {};
  for (const [name, curve] of Object.entries(rocCurves)) {
    out[name] = curve.fpr.map((fpr, i) => ({ fpr, tpr: curve.tpr[i] }));
  }
  return out;
}
