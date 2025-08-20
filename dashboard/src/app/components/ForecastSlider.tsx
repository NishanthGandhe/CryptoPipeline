"use client";

import { useState } from "react";
import { formatUSD, formatPercent, cn } from "@/lib/utils";
import { ProcessedForecast } from "@/lib/types";

type ForecastSliderProps = {
  forecasts: ProcessedForecast[];
  currentPrice: number | null;
};

export function ForecastSlider({ forecasts, currentPrice }: ForecastSliderProps) {
  const [selectedDay, setSelectedDay] = useState(1);
  
  // Filter out forecasts with null prices and get the one for selected day
  const validForecasts = forecasts.filter(f => f.forecastPrice !== null);
  const selectedForecast = validForecasts.find(f => f.label === `${selectedDay}d`);
  
  if (validForecasts.length === 0 || !currentPrice) {
    return (
      <div className="rounded-xl border bg-white p-6 shadow-sm">
        <p className="text-gray-500">No forecast data available.</p>
      </div>
    );
  }

  const maxDays = validForecasts.length;
  const isPositive = selectedForecast?.change ? selectedForecast.change > 0 : false;
  const isNeutral = selectedForecast?.change === 0;

  return (
    <div className="rounded-xl border bg-white p-6 shadow-sm">
      <h2 className="text-xl font-semibold text-gray-800 mb-6">Price Forecast</h2>
      
      {/* Main forecast display */}
      <div className="text-center mb-8">
        <div className="text-sm font-medium text-gray-500 mb-2">
          {selectedDay} Day{selectedDay !== 1 ? 's' : ''} from now
        </div>
        <div className="text-4xl font-bold text-gray-900 mb-2">
          {selectedForecast ? formatUSD(selectedForecast.forecastPrice) : '—'}
        </div>
        <div className="text-sm text-gray-500 mb-1">
          Current: {formatUSD(currentPrice)}
        </div>
        {selectedForecast && (
          <div
            className={cn(
              "text-lg font-semibold",
              isNeutral ? "text-gray-500" : isPositive ? "text-green-600" : "text-red-600"
            )}
          >
            {isPositive ? "+" : ""}{formatUSD(selectedForecast.change)} ({formatPercent(selectedForecast.percentChange)})
          </div>
        )}
      </div>

      {/* Slider */}
      <div className="mb-6">
        <div className="flex justify-between text-sm text-gray-500 mb-2">
          <span>1 day</span>
          <span>{maxDays} days</span>
        </div>
        <input
          type="range"
          min="1"
          max={maxDays}
          value={selectedDay}
          onChange={(e) => setSelectedDay(parseInt(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
          style={{
            background: `linear-gradient(to right, #4f46e5 0%, #4f46e5 ${((selectedDay - 1) / (maxDays - 1)) * 100}%, #e5e7eb ${((selectedDay - 1) / (maxDays - 1)) * 100}%, #e5e7eb 100%)`
          }}
        />
        <div className="flex justify-between text-xs text-gray-400 mt-1">
          <span>Short term</span>
          <span>Long term</span>
        </div>
      </div>

      {/* Quick access buttons */}
      <div className="grid grid-cols-4 gap-2">
        {[1, 3, 7, 30].filter(day => day <= maxDays).map((day) => {
          const forecast = validForecasts.find(f => f.label === `${day}d`);
          const isSelected = selectedDay === day;
          const dayIsPositive = forecast?.change ? forecast.change > 0 : false;
          
          return (
            <button
              key={day}
              onClick={() => setSelectedDay(day)}
              className={cn(
                "p-3 rounded-lg border text-center transition-all",
                isSelected 
                  ? "border-indigo-500 bg-indigo-50 text-indigo-700" 
                  : "border-gray-200 bg-white text-gray-700 hover:border-gray-300 hover:bg-gray-50"
              )}
            >
              <div className="text-xs font-medium">{day}d</div>
              {forecast && (
                <>
                  <div className="text-sm font-semibold">
                    {formatUSD(forecast.forecastPrice).replace(/\$|,/g, '').substring(0, 5)}
                  </div>
                  <div className={cn(
                    "text-xs",
                    dayIsPositive ? "text-green-600" : "text-red-600"
                  )}>
                    {formatPercent(forecast.percentChange)}
                  </div>
                </>
              )}
            </button>
          );
        })}
      </div>

      {/* Disclaimer */}
      <div className="mt-6 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
        <p className="text-xs text-yellow-700">
          ⚠️ Forecasts are model predictions and become less reliable over longer time horizons. 
          This is not financial advice.
        </p>
      </div>
    </div>
  );
}
