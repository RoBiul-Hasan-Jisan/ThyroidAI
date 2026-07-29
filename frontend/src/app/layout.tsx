import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/navbar";

export const metadata: Metadata = {
  title: "ThyroidAI — Explainable Thyroid Cancer Recurrence Prediction",
  description:
    "An explainable ML/DL system for predicting differentiated thyroid cancer recurrence risk.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-slate-50 font-sans">
        <Navbar />
        <main className="flex-1">{children}</main>
        <footer className="border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-400">
          ThyroidAI &middot; Research &amp; educational demonstration only — not a substitute for
          clinical judgment &middot; Dataset: UCI Differentiated Thyroid Cancer Recurrence (CC BY 4.0)
        </footer>
      </body>
    </html>
  );
}
