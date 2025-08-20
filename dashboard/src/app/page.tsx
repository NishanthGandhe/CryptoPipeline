"use client";

import { useState, useEffect } from "react";
import { useCryptoData } from "./hooks/useCryptoData";
import { SymbolSelector } from "./components/SymbolSelector";
import { ChartCard } from "./components/ChartCard";
import { ForecastSlider } from "./components/ForecastSlider";
import { SkeletonLoader } from "./components/SkeletonLoader";

export default function Page() {
  const [symbol, setSymbol] = useState("BTC-USD");
  const { data, loading, error, symbols, refresh, isRefreshing, lastRefreshTime } = useCryptoData(symbol);

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'r' && !isRefreshing) {
        e.preventDefault();
        refresh();
      }

      if (symbols.length > 0 && !isRefreshing) {
        const currentIndex = symbols.indexOf(symbol);
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
          e.preventDefault();
          const nextIndex = (currentIndex + 1) % symbols.length;
          setSymbol(symbols[nextIndex]);
        }
        if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
          e.preventDefault();
          const prevIndex = currentIndex === 0 ? symbols.length - 1 : currentIndex - 1;
          setSymbol(symbols[prevIndex]);
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [symbol, symbols, refresh, isRefreshing]);

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
              AI-driven price predictions and market insights
            </p>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={refresh}
              disabled={loading || isRefreshing}
              title="Run data pipeline: Fetch latest prices and generate new forecasts using machine learning models"
              className="inline-flex items-center gap-2 rounded-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              {isRefreshing ? "Updating Data..." : loading ? "Loading..." : "Refresh Data"}
            </button>
            <a
              href="https://github.com/nishanthgandhe/CryptoPipeline"
              target="_blank"
              rel="noreferrer"
              className="text-sm font-medium text-indigo-600 hover:text-indigo-500"
            >
              View on GitHub &rarr;
            </a>
          </div>
        </header>

        <div className="mb-6 flex items-center gap-4">
          <div className="max-w-xs">
            <SymbolSelector
              symbols={symbols}
              selectedSymbol={symbol}
              onSelectSymbol={setSymbol}
              disabled={loading || symbols.length === 0 || isRefreshing}
            />
          </div>
          {data && (
            <div className="flex items-center gap-4 text-sm text-gray-500">
              <span>Last updated: {new Date().toLocaleTimeString()}</span>
              {lastRefreshTime && (
                <>
                  <span className="hidden sm:block">•</span>
                  <span className="hidden sm:block">Pipeline last run: {lastRefreshTime.toLocaleTimeString()}</span>
                </>
              )}
              <span className="hidden sm:block">•</span>
              <span className="hidden sm:block">⌘R to refresh • ←→ to navigate</span>
            </div>
          )}
        </div>

        {error && (
          <div className="mb-4 rounded-md border border-red-200 bg-red-100 p-4 text-sm text-red-800">
            <div className="flex items-center gap-2">
              <svg className="h-4 w-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <strong>Error:</strong> {error}
            </div>
          </div>
        )}

        {/* Data refresh in progress notification */}
        {isRefreshing && (
          <div className="mb-4 rounded-md border border-blue-200 bg-blue-50 p-4 text-sm text-blue-800">
            <div className="flex items-center gap-2">
              <svg className="h-4 w-4 text-blue-500 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <strong>Updating:</strong> Running data pipeline to fetch latest prices and generate new forecasts. This may take a few minutes...
            </div>
          </div>
        )}

        {/* Data freshness warning */}
        {data && data.series.length > 0 && (
          (() => {
            const lastDataDate = new Date(data.series[data.series.length - 1].ds);
            const daysSinceLastData = Math.floor((Date.now() - lastDataDate.getTime()) / (1000 * 60 * 60 * 24));
            
            if (daysSinceLastData > 2) {
              return (
                <div className="mb-4 rounded-md border border-yellow-200 bg-yellow-50 p-4 text-sm text-yellow-800">
                  <div className="flex items-center gap-2">
                    <svg className="h-4 w-4 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.728-.833-2.498 0L4.316 15.5c-.77.833.192 2.5 1.732 2.5z" />
                    </svg>
                    <strong>Notice:</strong> Data may be stale. Last update was {daysSinceLastData} days ago. 
                    <button onClick={refresh} className="underline hover:no-underline ml-1">Refresh</button>
                  </div>
                </div>
              );
            }
            return null;
          })()
        )}

        {loading ? (
          <SkeletonLoader />
        ) : (
          data && (
            <>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-3 lg:gap-8">
                <ChartCard
                  symbol={data.symbol}
                  series={data.series}
                  currentPrice={data.currentPrice}
                />
                <ForecastSlider
                  forecasts={data.forecasts}
                  currentPrice={data.currentPrice}
                />
              </div>
              
              {/* Model Information Section */}
              <div className="mt-8 rounded-xl border bg-white p-6 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-800">Model Information</h3>
                  <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                    {data.symbol}
                  </span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  {data.modelInfo && (
                    <>
                      <div className="space-y-3">
                        <div>
                          <span className="text-sm font-medium text-gray-500">Prediction Model</span>
                          <p className="text-base font-semibold text-gray-900">{data.modelInfo.method}</p>
                        </div>
                        <div>
                          <span className="text-sm font-medium text-gray-500">Description</span>
                          <p className="text-sm text-gray-700">{data.modelInfo.description}</p>
                        </div>
                      </div>
                      
                      {data.modelInfo.features.length > 0 && (
                        <div className="space-y-3">
                          <div>
                            <span className="text-sm font-medium text-gray-500">Features Used ({data.modelInfo.features.length})</span>
                            <div className="mt-1 space-y-1">
                              {data.modelInfo.features.map((feature, index) => (
                                <span key={index} className="inline-block bg-blue-50 text-blue-700 px-2 py-1 rounded text-xs mr-1 mb-1 border border-blue-200">
                                  {feature}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                      
                      <div className="space-y-3">
                        <div>
                          <span className="text-sm font-medium text-gray-500">Model Performance</span>
                          <p className="text-sm font-semibold text-green-700">{data.modelInfo.accuracy}</p>
                        </div>
                        <div>
                          <span className="text-sm font-medium text-gray-500">Training Data</span>
                          <p className="text-sm text-gray-700">2+ years of historical data with technical indicators</p>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        <div>
                          <span className="text-sm font-medium text-gray-500">Prediction Range</span>
                          <p className="text-sm font-semibold text-gray-900">1-30 days ahead</p>
                        </div>
                        <div>
                          <span className="text-sm font-medium text-gray-500">Update Frequency</span>
                          <p className="text-sm text-gray-700">Daily at market close</p>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </>
          )
        )}
      </div>
    </div>
  );
}