import { formatUSD, formatPercent, formatUSDShort, cn } from "@/lib/utils";
import { LatestRow, ProcessedForecast } from "@/lib/types";
import { InformationCircleIcon } from '@heroicons/react/24/outline'; // New icon import

// Helper component for the yellow alert box
function Alert({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-md bg-yellow-50 p-4">
      <div className="flex">
        <div className="flex-shrink-0">
          <InformationCircleIcon className="h-5 w-5 text-yellow-400" aria-hidden="true" />
        </div>
        <div className="ml-3">
          <p className="text-sm text-yellow-700">{children}</p>
        </div>
      </div>
    </div>
  );
}

// Updated StatBox with vertical layout and shortened numbers
function StatBox({ forecast }: { forecast: ProcessedForecast }) {
  const isPositive = forecast.change ? forecast.change > 0 : false;
  const isNeutral = forecast.change === 0;

  return (
    <div className="flex flex-col items-center justify-center rounded-lg border bg-gray-50 p-3 text-center">
      <div className="text-xs font-bold uppercase tracking-wider text-gray-500">{forecast.label}</div>
      <div className="mt-1 text-2xl font-semibold tracking-tight text-gray-900">
        {formatUSDShort(forecast.forecastPrice)}
      </div>
      <div
        className={cn(
          "text-sm font-semibold",
          isNeutral ? "text-gray-500" : isPositive ? "text-green-600" : "text-red-600"
        )}
      >
        {formatPercent(forecast.percentChange)}
      </div>
    </div>
  );
}

// A new component to display model metrics cleanly
function ModelDetails({ insight }: { insight: LatestRow }) {
  // Example parsing - adjust if your headline/details format is different
  const maeMatch = insight.details.match(/MAE=(\d+\.?\d*)/);
  const vsNaiveMatch = insight.details.match(/vs naive: ([+-]?\d+%)/);
  const ciMatch = insight.details.match(/95% CI \[\$(.*), \$(.*)\]/);

  return (
    <div className="mt-6">
      <h3 className="text-base font-semibold text-gray-700">Model Details</h3>
      <div className="mt-2 space-y-2 rounded-lg border border-gray-200 bg-white p-4">
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Model Name</span>
          <span className="font-mono text-gray-800">{insight.forecast_method}</span>
        </div>
        {maeMatch && (
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">Mean Absolute Error (1d)</span>
            <span className="font-mono text-gray-800">{formatUSD(parseFloat(maeMatch[1]))}</span>
          </div>
        )}
        {vsNaiveMatch && (
           <div className="flex justify-between text-sm">
            <span className="text-gray-500">Accuracy vs. Naive</span>
            <span className="font-mono text-gray-800">{vsNaiveMatch[1]}</span>
          </div>
        )}
        {ciMatch && (
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">95% Confidence (1d)</span>
            <span className="font-mono text-gray-800">${ciMatch[1]} - ${ciMatch[2]}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export function ForecastCard({ latestInsight, forecasts }: { latestInsight: LatestRow | null; forecasts: ProcessedForecast[] }) {
  if (!latestInsight) {
    return (
      <div className="rounded-xl border bg-white p-6 shadow-sm flex items-center justify-center">
        <p className="text-gray-500">No forecast insight available.</p>
      </div>
    );
  }

  const shortTermForecasts = forecasts.slice(0, 3);
  const longTermForecasts = forecasts.slice(3);

  return (
    <div className="rounded-xl border bg-white p-4 shadow-sm sm:p-6">
      <h2 className="text-xl font-semibold text-gray-800">Forecast Insight</h2>

      <div className="mt-4 grid grid-cols-3 gap-3">
        {shortTermForecasts.map(f => <StatBox key={f.label} forecast={f} />)}
      </div>

      <div className="mt-6">
        <Alert>Long-horizon forecasts have high uncertainty and should be treated as speculative.</Alert>
      </div>

      <div className="mt-6">
        <h3 className="text-base font-semibold text-gray-700">Long-Term Outlook</h3>
        {/*
          START: THIS IS THE RESTORED TABLE CODE
        */}
        <table className="mt-2 w-full text-sm">
          <thead className="text-left">
            <tr className="border-b text-gray-500">
              <th className="py-2 font-medium">Horizon</th>
              <th className="py-2 font-medium">Date</th>
              <th className="py-2 font-medium text-right">Forecast</th>
              <th className="py-2 font-medium text-right">% Change</th>
            </tr>
          </thead>
          <tbody>
            {longTermForecasts.map((f) => {
              const isPositive = f.change ? f.change > 0 : false;
              const isNeutral = f.change === 0;
              return (
                <tr key={f.label} className="border-t border-gray-200">
                  <td className="py-2 font-medium text-gray-700">{f.label}</td>
                  <td className="py-2 text-gray-500">{f.date}</td>
                  <td className="py-2 font-mono text-right text-gray-800">
                    {formatUSD(f.forecastPrice)}
                  </td>
                  <td
                    className={cn(
                      "py-2 font-mono text-right",
                      isNeutral ? "text-gray-500" : isPositive ? "text-green-600" : "text-red-600"
                    )}
                  >
                    {formatPercent(f.percentChange)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
         {/*
          END: RESTORED TABLE CODE
        */}
      </div>

      <ModelDetails insight={latestInsight} />
    </div>
  );
}

// NOTE: You'll need to install @heroicons/react to use the icon
// npm install @heroicons/react