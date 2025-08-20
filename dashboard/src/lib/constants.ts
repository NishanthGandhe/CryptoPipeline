// Define horizons for the forecast display
export const QUICK_HORIZONS = ["1d", "3d", "7d"];  // For quick view boxes
export const ALL_HORIZONS = Array.from({length: 30}, (_, i) => `${i + 1}d`);  // For slider: 1d, 2d, ... 30d

function parseDateUTC(iso: string) {
  return new Date(iso + "T00:00:00Z");
}

export function daysToLabelFromBase(baseISO: string, targetISO: string) {
  const base = parseDateUTC(baseISO);
  const target = parseDateUTC(targetISO);
  const diff = Math.round((+target - +base) / (1000 * 60 * 60 * 24)) + 1;
  return `${diff}d`;
}