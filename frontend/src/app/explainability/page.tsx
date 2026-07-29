"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PredictionResponse } from "@/lib/types";
import { ArrowRight, TrendingDown, TrendingUp } from "lucide-react";

export default function ExplainabilityPage() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [patient, setPatient] = useState<Record<string, string | number> | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const raw = window.localStorage.getItem("thyroidai_last_result");
    const rawPatient = window.localStorage.getItem("thyroidai_last_patient");
    if (raw) setResult(JSON.parse(raw));
    if (rawPatient) setPatient(JSON.parse(rawPatient));
  }, []);

  const chartData =
    result?.shap_factors
      .slice()
      .sort((a, b) => a.impact - b.impact)
      .map((f) => ({
        name: `${f.feature} = ${f.value}`,
        impact: f.impact,
      })) ?? [];

  const increasing = result?.shap_factors.filter((f) => f.direction === "increases") ?? [];
  const decreasing = result?.shap_factors.filter((f) => f.direction === "decreases") ?? [];

  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <Badge variant="teal" className="mb-3">Explainable AI</Badge>
      <h1 className="text-3xl font-bold text-slate-900">Explainability Dashboard</h1>
      <p className="mt-2 max-w-2xl text-slate-600">
        Every prediction is explained using SHAP (SHapley Additive exPlanations), attributing
        the model&apos;s output to individual clinicopathologic features for that specific patient.
      </p>

      {!result ? (
        <Card className="mt-8">
          <CardContent className="flex flex-col items-center gap-4 py-16 text-center">
            <p className="text-slate-500">
              No prediction yet. Run a prediction first to see its SHAP explanation here.
            </p>
            <Button asChild variant="primary">
              <Link href="/predict">
                Go to Predict <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
          </CardContent>
        </Card>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-3"
        >
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>SHAP feature contributions</CardTitle>
              <CardDescription>
                Positive bars push the prediction toward recurrence; negative bars push away
                from it. Model: {result.model_used}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-96 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical" margin={{ left: 24, right: 24 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" tick={{ fontSize: 12 }} />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={190}
                      tick={{ fontSize: 11 }}
                    />
                    <Tooltip
                      formatter={(v) => Number(v).toFixed(4)}
                      labelStyle={{ fontSize: 12 }}
                    />
                    <ReferenceLine x={0} stroke="#94a3b8" />
                    <Bar dataKey="impact" radius={[4, 4, 4, 4]}>
                      {chartData.map((entry, idx) => (
                        <Cell key={idx} fill={entry.impact > 0 ? "#dc2626" : "#059669"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <div className="flex flex-col gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-red-700">
                  <TrendingUp className="h-4 w-4" /> Increasing recurrence risk
                </CardTitle>
              </CardHeader>
              <CardContent>
                {increasing.length ? (
                  <ul className="space-y-2 text-sm">
                    {increasing.map((f) => (
                      <li key={f.feature} className="flex justify-between">
                        <span className="text-slate-700">
                          {f.feature} = {f.value}
                        </span>
                        <span className="font-mono text-red-600">+{f.impact.toFixed(3)}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-slate-400">No risk-increasing factors identified.</p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-emerald-700">
                  <TrendingDown className="h-4 w-4" /> Reducing recurrence risk
                </CardTitle>
              </CardHeader>
              <CardContent>
                {decreasing.length ? (
                  <ul className="space-y-2 text-sm">
                    {decreasing.map((f) => (
                      <li key={f.feature} className="flex justify-between">
                        <span className="text-slate-700">
                          {f.feature} = {f.value}
                        </span>
                        <span className="font-mono text-emerald-600">{f.impact.toFixed(3)}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-slate-400">No risk-reducing factors identified.</p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Patient explanation</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-slate-600">
                  This patient was predicted <strong>{result.prediction === "Yes" ? "likely" : "unlikely"}</strong>{" "}
                  to experience recurrence ({(result.probability * 100).toFixed(1)}% probability),
                  with <strong>{result.confidence.toLowerCase()}</strong> model confidence. The
                  strongest driver was{" "}
                  <strong>
                    {result.shap_factors[0]?.feature} = {result.shap_factors[0]?.value}
                  </strong>
                  , which {result.shap_factors[0]?.direction} recurrence risk.
                </p>
              </CardContent>
            </Card>
          </div>
        </motion.div>
      )}
    </div>
  );
}
