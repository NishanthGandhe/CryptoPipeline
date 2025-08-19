// Define horizons in one place to avoid repetition
export const HORIZONS = ["1d", "1w", "1m", "3m", "6m", "1y", "5y", "10y"];

function parseDateUTC(iso: string) {
  return new Date(iso + "T00:00:00Z");
}

export function daysToLabelFromBase(baseISO: string, targetISO: string) {
  const base = parseDateUTC(baseISO);
  const target = parseDateUTC(targetISO);
  const diff = Math.round((+target - +base) / (1000 * 60 * 60 * 24)) + 1;
  if (diff === 1) return "1d";
  if (diff === 7) return "1w";
  if (diff === 30) return "1m";
  if (diff === 90) return "3m";
  if (diff === 180) return "6m";
  if (diff === 365) return "1y";
  if (diff === 1825) return "5y";
  if (diff === 3650) return "10y";
  return `${diff}d`;
}