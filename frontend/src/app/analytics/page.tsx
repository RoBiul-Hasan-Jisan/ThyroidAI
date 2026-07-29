"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getAnalytics } from "@/lib/api";
import { AnalyticsResponse, CrosstabRow } from "@/lib/types";

const NO_COLOR = "#0f766e";
const YES_COLOR = "#dc2626";
const PIE_COLORS = [NO_COLOR, YES_COLOR];

export default function AnalyticsPage() {
  const [data, setData] = useState<AnalyticsResponse | null>(null);

  useEffect(() => {
    getAnalytics().then(setData).catch(() => {});
  }, []);

  if (!data) {
    return (
      <div className="mx-auto max-w-6xl px-6 py-12">
        <p className="text-slate-400">Loading dataset analytics…</p>
      </div>
    );
  }

  const crosstabs: { title: string; data: CrosstabRow[] }[] = [
    { title: "Recurrence by gender", data: data.gender_recurrence },
    { title: "Recurrence by risk category", data: data.risk_recurrence },
    { title: "Recurrence by overall stage", data: data.stage_recurrence },
    { title: "Recurrence by treatment response", data: data.response_recurrence },
    { title: "Recurrence by T stage", data: data.t_stage_recurrence },
    { title: "Recurrence by N stage", data: data.n_stage_recurrence },
    { title: "Recurrence by M stage", data: data.m_stage_recurrence },
    { title: "Recurrence by smoking status", data: data.smoking_recurrence },
    { title: "Recurrence by adenopathy", data: data.adenopathy_recurrence },
    { title: "Recurrence by focality", data: data.focality_recurrence },
    { title: "Recurrence by pathology", data: data.pathology_recurrence },
  ];

  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <Badge variant="teal" className="mb-3">Dataset Analytics</Badge>
      <h1 className="text-3xl font-bold text-slate-900">Exploratory Data Analysis</h1>
      <p className="mt-2 max-w-2xl text-slate-600">
        {data.n_samples} patients &middot; {data.n_features} clinicopathologic features &middot; interactive charts
      </p>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4 }}
        className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-2"
      >
        {/* TARGET DISTRIBUTION */}
        <Card>
          <CardHeader>
            <CardTitle>Target distribution</CardTitle>
            <CardDescription>Class balance of recurrence outcome</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={data.target_distribution}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={95}
                    label={(entry) => `${entry.name}: ${entry.value}`}
                  >
                    {data.target_distribution.map((_, i) => (
                      <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* AGE HISTOGRAM */}
        <Card>
          <CardHeader>
            <CardTitle>Age distribution</CardTitle>
            <CardDescription>Patient count by 5-year age band, split by recurrence</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.age_histogram} margin={{ left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="No" stackId="a" fill={NO_COLOR} />
                  <Bar dataKey="Yes" stackId="a" fill={YES_COLOR} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {crosstabs.map((ct) => (
          <Card key={ct.title}>
            <CardHeader>
              <CardTitle className="text-base">{ct.title}</CardTitle>
              <CardDescription>Recurrence rate (%) by category</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={ct.data} layout="vertical" margin={{ left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} />
                    <YAxis
                      type="category"
                      dataKey="category"
                      width={110}
                      tick={{ fontSize: 11 }}
                    />
                    <Tooltip formatter={(v) => `${v}%`} />
                    <Legend />
                    <Bar dataKey="No" stackId="a" fill={NO_COLOR} />
                    <Bar dataKey="Yes" stackId="a" fill={YES_COLOR} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        ))}
      </motion.div>
    </div>
  );
}
