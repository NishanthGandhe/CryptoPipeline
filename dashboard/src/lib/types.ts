// For raw data fetched from Supabase
export type SymbolRow = { symbol: string };
export type PriceRow = { symbol: string; ds: string; close: number; ma_7: number | null };
export type LatestRow = {
  symbol: string;
  forecast_for_day: string;
  forecast_close: number;
  forecast_method: string;
  headline: string;
  details: string;
  created_at: string;
};
export type MultiForecastRow = {
  symbol: string;
  ds: string;
  forecast_close: number;
  forecast_method: string;
  created_at: string;
};

// For data after it has been processed in the hook
export type ProcessedForecast = {
  label: string;
  date: string;
  forecastPrice: number | null;
  change: number | null;
  percentChange: number | null;
}

export type ModelInfo = {
  method: string;
  description: string;
  features: string[];
  accuracy: string;
}