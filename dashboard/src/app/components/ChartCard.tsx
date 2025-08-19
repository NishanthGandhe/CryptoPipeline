import dynamic from "next/dynamic";
import { formatUSD } from "@/lib/utils";
import { PriceRow } from "@/lib/types";

const ChartClient = dynamic(() => import("@/app/components/ChartClient"), { ssr: false });

type ChartCardProps = {
  symbol: string;
  series: PriceRow[];
  currentPrice: number | null;
};

export function ChartCard({ symbol, series, currentPrice }: ChartCardProps) {
  return (
    <div className="lg:col-span-2 rounded-xl border bg-white p-4 shadow-sm sm:p-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-800">{symbol}</h2>
          <p className="text-sm text-gray-500">Last 120 days (close & 7d MA)</p>
        </div>
        <div className="text-right">
          <div className="text-sm text-gray-500">Current Price</div>
          <div className="text-2xl font-bold text-gray-900">
            {formatUSD(currentPrice)}
          </div>
        </div>
      </div>
      <div className="mt-4 h-72 w-full">
        <ChartClient data={series} />
      </div>
    </div>
  );
}