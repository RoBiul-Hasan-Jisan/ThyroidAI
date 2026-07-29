"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Activity, Brain, LayoutDashboard, LineChart, Sparkles } from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/predict", label: "Predict", icon: Activity },
  { href: "/explainability", label: "Explainability", icon: Sparkles },
  { href: "/models", label: "Model Bench", icon: Brain },
  { href: "/analytics", label: "Analytics", icon: LineChart },
];

export function Navbar() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
        <Link href="/" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-800 text-sm font-bold text-white">
            T
          </div>
          <div className="leading-none">
            <div className="font-semibold text-slate-900">
              Thyroid<span className="text-teal-700">AI</span>
            </div>
            <div className="text-[10px] uppercase tracking-widest text-slate-400">
              Explainable Recurrence Prediction
            </div>
          </div>
        </Link>
        <nav className="flex items-center gap-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  active
                    ? "bg-teal-50 text-teal-800"
                    : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
                )}
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
