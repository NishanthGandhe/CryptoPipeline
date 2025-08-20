import { useState, useEffect, useMemo } from "react";
import { supabase } from "@/lib/supabase";
import { QUICK_HORIZONS, ALL_HORIZONS, daysToLabelFromBase } from "@/lib/constants";
import { PriceRow, LatestRow, MultiForecastRow, SymbolRow, ProcessedForecast, ModelInfo } from "@/lib/types"; // (See types.ts below)

// Centralized data fetching and processing hook
export function useCryptoData(symbol: string) {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0); // Add refresh trigger
  const [isRefreshing, setIsRefreshing] = useState(false); // Track pipeline refresh status
  const [lastRefreshTime, setLastRefreshTime] = useState<Date | null>(null); // Track last successful refresh
  const [data, setData] = useState<{
    symbol: string;
    series: PriceRow[];
    latestInsight: LatestRow | null;
    forecasts: ProcessedForecast[];
    currentPrice: number | null;
    modelInfo: ModelInfo | null;
  } | null>(null);

  // Enhanced refresh function to trigger actual data pipeline
  const refresh = async () => {
    setIsRefreshing(true);
    setError(null);
    
    try {
      console.log('Triggering data pipeline refresh...');
      
      // Use external API URL in production
      const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || '';
      const refreshUrl = apiBaseUrl ? `${apiBaseUrl}/refresh-data` : '/api/refresh-data';
      
      const response = await fetch(refreshUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      const result = await response.json();
      
      if (!response.ok) {
        throw new Error(result.message || 'Failed to refresh data');
      }
      
      console.log('Pipeline completed successfully:', result);
      
      // Record successful refresh time
      setLastRefreshTime(new Date());
      
      // After successful pipeline run, trigger data refetch
      setRefreshKey(prev => prev + 1);
      
    } catch (error: any) {
      console.error('Pipeline refresh failed:', error);
      setError(`Data refresh failed: ${error.message}`);
    } finally {
      setIsRefreshing(false);
    }
  };

  // Helper function to generate model information based on forecast method
  const getModelInfo = (method: string, symbol: string): ModelInfo => {
    switch (method) {
      case 'xgboost':
        const isBTC = symbol === 'BTC-USD';
        return {
          method: 'XGBoost Enhanced',
          description: isBTC 
            ? 'Advanced machine learning model using price, volume, and technical indicators for multi-dimensional analysis.'
            : 'Enhanced machine learning model using price patterns, momentum indicators, and volatility analysis.',
          features: isBTC
            ? ['Price Action', 'Trade Volume', 'Total Supply', 'Momentum (3d, 7d)', 'Volatility', 'Moving Averages', 'Volume Trends']
            : ['Price Action', 'Momentum (3d, 7d, 14d)', 'Moving Averages', 'Volatility Ratios', 'Price Positioning', 'Trend Signals'],
          accuracy: 'Optimized for trend-following and dynamic predictions'
        };
      case 'holtwinters':
        return {
          method: 'Holt-Winters',
          description: 'Statistical time series model that captures trend and seasonal patterns in price movements.',
          features: ['Historical Prices', 'Trend Analysis', 'Seasonal Patterns'],
          accuracy: 'Good for stable markets with clear trends'
        };
      case 'naive':
        return {
          method: 'Naive Baseline',
          description: 'Simple baseline model that uses the most recent price as the forecast for future periods.',
          features: ['Last Known Price'],
          accuracy: 'Conservative baseline approach'
        };
      default:
        return {
          method: 'Unknown',
          description: 'Model information not available.',
          features: [],
          accuracy: 'N/A'
        };
    }
  };

  // Effect to load the list of available symbols once on mount
  useEffect(() => {
    const fetchSymbols = async () => {
      try {
        console.log("Fetching symbols...");
        // Try to get symbols from insights table instead
        const { data, error } = await supabase
          .from("insights")
          .select("symbol")
          .order("symbol");
        
        if (error) {
          console.error("Failed to fetch symbols:", error);
          // Fallback to hardcoded list if database fails
          const fallbackSymbols = [
            "BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "LINK-USD",
            "ADA-USD", "XLM-USD", "SOL-USD", "DOT-USD", "DOGE-USD", "AVAX-USD",
            "ALGO-USD", "ICP-USD", "NEAR-USD", "HBAR-USD", "UNI-USD", "AAVE-USD",
            "MKR-USD", "SNX-USD", "INJ-USD", "GRT-USD"
          ];
          console.log("Using fallback symbols:", fallbackSymbols);
          setSymbols(fallbackSymbols);
          return;
        }
        
        // Get unique symbols
        const uniqueSymbols = [...new Set(data?.map(d => d.symbol) || [])];
        console.log("Fetched symbols from DB:", uniqueSymbols);
        setSymbols(uniqueSymbols);
      } catch (e) {
        console.error("Connection error:", e);
        // Fallback to hardcoded list
        const fallbackSymbols = [
          "BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "LINK-USD",
          "ADA-USD", "XLM-USD", "SOL-USD", "DOT-USD", "DOGE-USD", "AVAX-USD",
          "ALGO-USD", "ICP-USD", "NEAR-USD", "HBAR-USD", "UNI-USD", "AAVE-USD",
          "MKR-USD", "SNX-USD", "INJ-USD", "GRT-USD"
        ];
        console.log("Using fallback symbols due to error:", fallbackSymbols);
        setSymbols(fallbackSymbols);
      }
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

        const processedForecasts = ALL_HORIZONS.map(label => {
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

        // Get model info from the first forecast's method
        const forecastMethod = multiForecasts.length > 0 ? multiForecasts[0].forecast_method : 'unknown';
        const modelInfo = getModelInfo(forecastMethod, symbol);

        setData({
          symbol,
          series,
          latestInsight,
          forecasts: processedForecasts,
          currentPrice,
          modelInfo,
        });

      } catch (e: any) {
        setError(e.message || "An unexpected error occurred.");
      } finally {
        setLoading(false);
      }
    };

    fetchDataForSymbol();
  }, [symbol, refreshKey]); // Add refreshKey as dependency

  return { data, loading, error, symbols, refresh, isRefreshing, lastRefreshTime };
}