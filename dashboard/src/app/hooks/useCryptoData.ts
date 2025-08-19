import { useState, useEffect, useMemo } from "react";
import { supabase } from "@/lib/supabase";
import { HORIZONS, daysToLabelFromBase } from "@/lib/constants";
import { PriceRow, LatestRow, MultiForecastRow, SymbolRow, ProcessedForecast } from "@/lib/types"; // (See types.ts below)

// Centralized data fetching and processing hook
export function useCryptoData(symbol: string) {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<{
    symbol: string;
    series: PriceRow[];
    latestInsight: LatestRow | null;
    forecasts: ProcessedForecast[];
    currentPrice: number | null;
  } | null>(null);

  // Effect to load the list of available symbols once on mount
  useEffect(() => {
    const fetchSymbols = async () => {
      const { data, error } = await supabase.from("symbols").select("symbol");
      if (error) {
        console.error("Failed to fetch symbols:", error.message);
        return;
      }
      const syms = (data as SymbolRow[]).map(d => d.symbol);
      setSymbols(syms);
    };
    fetchSymbols();
  }, []);

  // Effect to fetch and process data when the symbol changes
  useEffect(() => {
    if (!symbol) return;

    const fetchDataForSymbol = async () => {
      setLoading(true);
      setError(null);

      try {
        const [{ data: prices }, { data: latestRows }, { data: mfRows }] = await Promise.all([
          supabase.from("price_last_120d").select("symbol, ds, close, ma_7").eq("symbol", symbol).order("ds", { ascending: true }),
          supabase.from("latest_forecast_insight").select("*").eq("symbol", symbol).limit(1),
          supabase.from("latest_forecasts_all").select("*").eq("symbol", symbol).order("ds", { ascending: true }),
        ]);

        const series = (prices || []) as PriceRow[];
        const latestInsight = (latestRows && latestRows[0]) as LatestRow | null;
        const multiForecasts = (mfRows || []) as MultiForecastRow[];

        // --- Data Processing ---
        const currentPrice = series.length > 0 ? series[series.length - 1].close : null;
        
        const forecastMap: Record<string, MultiForecastRow> = {};
        if (latestInsight) {
          for (const r of multiForecasts) {
            const label = daysToLabelFromBase(latestInsight.forecast_for_day, r.ds);
            forecastMap[label] = r;
          }
        }

        const processedForecasts = HORIZONS.map(label => {
          const forecast = forecastMap[label];
          if (!forecast || currentPrice === null) {
            return { label, date: "—", forecastPrice: null, change: null, percentChange: null };
          }
          const change = forecast.forecast_close - currentPrice;
          const percentChange = (change / currentPrice);
          return {
            label,
            date: forecast.ds,
            forecastPrice: forecast.forecast_close,
            change,
            percentChange,
          };
        });

        setData({
          symbol,
          series,
          latestInsight,
          forecasts: processedForecasts,
          currentPrice,
        });

      } catch (e: any) {
        setError(e.message || "An unexpected error occurred.");
      } finally {
        setLoading(false);
      }
    };

    fetchDataForSymbol();
  }, [symbol]);

  return { data, loading, error, symbols };
}