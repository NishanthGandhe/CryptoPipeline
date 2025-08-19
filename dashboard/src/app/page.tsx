"use client";

import { useState } from "react";
import { useCryptoData } from "./hooks/useCryptoData";
import { SymbolSelector } from "./components/SymbolSelector";
import { ChartCard } from "./components/ChartCard";
import { ForecastCard } from "./components/ForecastCard";
import { SkeletonLoader } from "./components/SkeletonLoader";

export default function Page() {
  // The first symbol from the list is used as the default
  const [symbol, setSymbol] = useState("BTC-USD");
  const { data, loading, error, symbols } = useCryptoData(symbol);

  // Update the default symbol once the full list is loaded
  useState(() => {
    if (symbols.length > 0 && symbol !== symbols[0]) {
      setSymbol(symbols[0]);
    }
  });

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
        <header className="mb-8 flex flex-col items-center justify-between gap-4 sm:flex-row">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-gray-900">
              Crypto Forecast
            </h1>
            <p className="mt-1 text-sm text-gray-500">
              AI-driven price predictions and market insights.
            </p>
          </div>
          <a
            href="https://github.com/yourname/CryptoPipeline"
            target="_blank"
            rel="noreferrer"
            className="text-sm font-medium text-indigo-600 hover:text-indigo-500"
          >
            View on GitHub &rarr;
          </a>
        </header>

        <div className="mb-6 max-w-xs">
          <SymbolSelector
            symbols={symbols}
            selectedSymbol={symbol}
            onSelectSymbol={setSymbol}
            disabled={loading || symbols.length === 0}
          />
        </div>

        {error && (
          <div className="mb-4 rounded-md border border-red-200 bg-red-100 p-4 text-sm text-red-800">
            <strong>Error:</strong> {error}
          </div>
        )}

        {loading ? (
          <SkeletonLoader />
        ) : (
          data && (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3 lg:gap-8">
              <ChartCard
                symbol={data.symbol}
                series={data.series}
                currentPrice={data.currentPrice}
              />
              <ForecastCard
                latestInsight={data.latestInsight}
                forecasts={data.forecasts}
              />
            </div>
          )
        )}
      </div>
    </div>
  );
}