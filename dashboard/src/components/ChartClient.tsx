"use client";

import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend,
} from "recharts";

type PriceRow = { ds: string; close: number; ma_7: number | null };

function formatUSD(n?: number | null) {
  if (n == null) return "";
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(n);
}

export default function ChartClient({ data }: { data: PriceRow[] }) {
  const chartData = data.map((r) => ({ ...r, date: r.ds }));
  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" minTickGap={32} />
          <YAxis tickFormatter={(v) => formatUSD(v)} />
          <Tooltip
            formatter={(val, name) => [formatUSD(Number(val)), name as string]}
            labelFormatter={(l) => `Date: ${l}`}
          />
          <Legend />
          <Line type="monotone" dataKey="close" name="Close" dot={false} />
          <Line type="monotone" dataKey="ma_7" name="MA 7" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
