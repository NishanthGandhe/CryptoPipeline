import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

// Utility for combining tailwind classes, especially for conditional styles
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Formatter for USD currency
export function formatUSD(n?: number | null) {
  if (n == null) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(n);
}

// Formatter for percentages
export function formatPercent(n?: number | null) {
    if (n == null) return "—";
    return new Intl.NumberFormat("en-US", {
        style: "percent",
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
        signDisplay: "exceptZero"
    }).format(n)
}

export function formatUSDShort(n?: number | null) {
  if (n == null) return "—";

  if (n >= 1_000_000) {
    return `$${(n / 1_000_000).toFixed(1)}m`;
  }
  if (n >= 1000) {
    return `$${(n / 1000).toFixed(1)}k`;
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0, // No decimals for numbers under 1k
  }).format(n);
}