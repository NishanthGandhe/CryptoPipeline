"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { PriceRow } from '@/lib/types';

type ChartClientProps = {
  data: PriceRow[];
};

// Custom Tooltip for better formatting
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const dataPoint = payload[0].payload;
    return (
      <div className="rounded-md border bg-white p-3 text-sm shadow-lg">
        <p className="font-semibold">{new Date(dataPoint.ds + "T00:00:00Z").toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' })}</p>
        <p style={{ color: '#8884d8' }}>
          Close: {dataPoint.close.toLocaleString('en-US', { style: 'currency', currency: 'USD' })}
        </p>
        {dataPoint.ma_7 && (
          <p style={{ color: '#82ca9d' }}>
            7d MA: {dataPoint.ma_7.toLocaleString('en-US', { style: 'currency', currency: 'USD' })}
          </p>
        )}
      </div>
    );
  }
  return null;
};

export default function ChartClient({ data }: ChartClientProps) {
  // Formatter for the Y-axis (price)
  const formatYAxis = (tickItem: number) => {
    if (tickItem >= 1000) {
      return `$${(tickItem / 1000).toFixed(0)}k`;
    }
    return `$${tickItem}`;
  };

  // Formatter for the X-axis (date)
  const formatXAxis = (tickItem: string) => {
    // e.g., "2025-08-19" -> "Aug 19"
    return new Date(tickItem + "T00:00:00Z").toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
  };
  
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
        data={data}
        margin={{
          top: 5,
          right: 20,
          left: -10,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.5} />
        <XAxis 
          dataKey="ds" 
          tickFormatter={formatXAxis}
          tick={{ fontSize: 12 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis 
          tickFormatter={formatYAxis}
          domain={['dataMin', 'dataMax']}
          tick={{ fontSize: 12 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: '14px' }} />
        <Line 
          type="monotone" 
          dataKey="close" 
          stroke="#8884d8" 
          strokeWidth={2} 
          dot={false}
          name="Close Price"
        />
        <Line 
          type="monotone" 
          dataKey="ma_7" 
          stroke="#82ca9d" 
          strokeWidth={2} 
          dot={false} 
          name="7-day MA"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}