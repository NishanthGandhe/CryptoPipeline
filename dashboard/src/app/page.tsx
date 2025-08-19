"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { supabase } from "@/lib/supabase";

const ChartClient = dynamic(() => import("@/components/ChartClient"), { ssr: false });

type SymbolRow = { symbol: string };
type PriceRow = { symbol: string; ds: string; close: number; ma_7: number | null };
type LatestRow = {
  symbol: string;
  forecast_for_day: string;
  forecast_close: number;
  forecast_method: string;
  headline: string;
  details: string;
  created_at: string; // insight timestamp
};
type MultiForecastRow = {
  symbol: string;
  ds: string;               // ISO date of the forecasted day
  forecast_close: number;
  forecast_method: string;
  created_at: string;       // from model_training_runs join
};

function parseDateUTC(iso: string) {
  // iso like "2025-08-19" -> force UTC midnight
  return new Date(iso + "T00:00:00Z");
}

function daysToLabelFromBase(baseISO: string, targetISO: string) {
  const base = parseDateUTC(baseISO);
  const target = parseDateUTC(targetISO);
  const diff = Math.round((+target - +base) / (1000 * 60 * 60 * 24)) + 1; // 1d => 1
  if (diff === 1) return "1d";
  if (diff === 7) return "1w";
  if (diff === 30) return "1m";
  if (diff === 90) return "3m";
  if (diff === 180) return "6m";
  if (diff === 365) return "1y";
  if (diff === 1825) return "5y";
  if (diff === 3650) return "10y";
  return `${diff}d`;
}


function formatUSD(n?: number | null) {
  if (n == null) return "";
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(n);
}

export default function Page() {
  const [mounted, setMounted] = useState(false);
  const [symbols, setSymbols] = useState<string[]>([]);
  const [symbol, setSymbol] = useState<string>("");
  const [series, setSeries] = useState<PriceRow[]>([]);
  const [latest, setLatest] = useState<LatestRow | null>(null);
  const [multi, setMulti] = useState<MultiForecastRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  // Client-only render gate (avoids hydration mismatch with extensions / recharts)
  useEffect(() => { setMounted(true); }, []);

  // Load available symbols once
  useEffect(() => {
    (async () => {
      const { data, error } = await supabase.from("symbols").select("symbol");
      if (error) { setErr(error.message); return; }
      const syms = (data as SymbolRow[]).map(d => d.symbol);
      setSymbols(syms);
      setSymbol(syms[0] || "");
    })();
  }, []);

  // Load series + latest insight + all horizons for selected symbol
  useEffect(() => {
    if (!symbol) return;
    setLoading(true); setErr(null);

    (async () => {
      const [{ data: prices, error: e1 }, { data: latestRows, error: e2 }, { data: mfRows, error: e3 }] = await Promise.all([
        supabase.from("price_last_120d")
          .select("symbol, ds, close, ma_7")
          .eq("symbol", symbol)
          .order("ds", { ascending: true }),
        supabase.from("latest_forecast_insight")
          .select("*")
          .eq("symbol", symbol)
          .limit(1),
        supabase.from("latest_forecasts_all")
          .select("*")
          .eq("symbol", symbol)
          .order("ds", { ascending: true }),
      ]);

      if (e1 || e2 || e3) {
        setErr((e1 || e2 || e3)!.message);
        setLoading(false);
        return;
      }

      setSeries((prices || []) as PriceRow[]);
      setLatest((latestRows && latestRows[0]) as LatestRow);
      setMulti((mfRows || []) as MultiForecastRow[]);
      setLoading(false);
    })();
  }, [symbol]);

  // Map forecasts by friendly horizon label for quick lookup
  const horizonMap = useMemo(() => {
  if (!latest) return {} as Record<string, MultiForecastRow>;
  const wanted = new Set(["1d","1w","1m","3m","6m","1y","5y","10y"]);
  const base = latest.forecast_for_day;     // 👈 anchor on the 1-day forecast date
  const map: Record<string, MultiForecastRow> = {};
  for (const r of multi) {
    const lab = daysToLabelFromBase(latest.forecast_for_day, r.ds);
    if (wanted.has(lab)) map[lab] = r;
  }
  return map;
}, [multi, latest]);

  if (!mounted) {
    return <div className="min-h-screen bg-gray-50 p-6">Loading…</div>;
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <div className="mx-auto max-w-5xl px-6 py-8">
        <header className="mb-6 flex items-center justify-between gap-4">
          <h1 className="text-2xl font-semibold">CryptoPipe — Forecast Dashboard</h1>
          <a
            href="https://github.com/yourname/CryptoPipeline"
            target="_blank"
            rel="noreferrer"
            className="text-sm underline opacity-80 hover:opacity-100"
          >
            GitHub
          </a>
        </header>

        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">Symbol</label>
          <select
            className="rounded-lg border px-3 py-2 bg-white"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
          >
            {symbols.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        {err && (
          <div className="mb-4 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            {err}
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          {/* Chart card */}
          <div className="md:col-span-2 rounded-xl border bg-white p-4 shadow-sm">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-medium">Last 120 days</h2>
              <div className="text-sm opacity-70">close & 7d MA</div>
            </div>
            <ChartClient data={series} />
          </div>

          {/* Latest insight + multi-horizon */}
          <div className="rounded-xl border bg-white p-4 shadow-sm">
            <h2 className="mb-2 text-lg font-medium">Latest insight</h2>
            {loading ? (
              <div className="text-sm opacity-70">Loading…</div>
            ) : latest ? (
              <div className="space-y-3">
                <div className="text-base font-semibold">{latest.headline}</div>
                <div className="text-sm opacity-80">{latest.details}</div>

                {/* Big three at a glance */}
                <div className="grid grid-cols-3 gap-2 text-sm">
                  {["1d","1w","1m"].map((lab) => (
                    <div key={lab} className="rounded-lg border p-3 bg-gray-50">
                      <div className="text-xs opacity-60">{lab}</div>
                      <div className="font-semibold">
                        {horizonMap[lab] ? formatUSD(horizonMap[lab].forecast_close) : "—"}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Full horizon table */}
                <div className="pt-2">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left opacity-60">
                        <th className="py-1">Horizon</th>
                        <th className="py-1">Date</th>
                        <th className="py-1">Forecast</th>
                      </tr>
                    </thead>
                    <tbody>
                      {["1d","1w","1m","3m","6m","1y","5y","10y"].map((lab) => {
                        const r = horizonMap[lab];
                        return (
                          <tr key={lab} className="border-t">
                            <td className="py-1">{lab}</td>
                            <td className="py-1">{r ? r.ds : "—"}</td>
                            <td className="py-1">{r ? formatUSD(r.forecast_close) : "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="pt-2 text-xs opacity-60">
                  Method: <span className="font-mono">{latest.forecast_method}</span>
                </div>
              </div>
            ) : (
              <div className="text-sm opacity-70">No insight yet.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
