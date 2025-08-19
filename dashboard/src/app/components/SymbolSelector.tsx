import { ChevronUpDownIcon } from '@heroicons/react/20/solid'; // New icon import

type SymbolSelectorProps = {
  symbols: string[];
  selectedSymbol: string;
  onSelectSymbol: (symbol: string) => void;
  disabled: boolean;
};

export function SymbolSelector({ symbols, selectedSymbol, onSelectSymbol, disabled }: SymbolSelectorProps) {
  return (
    <div>
      <label htmlFor="symbol" className="block text-sm font-medium text-gray-700 mb-1">
        Select Cryptocurrency
      </label>
      <div className="relative">
        <select
          id="symbol"
          className="w-full cursor-pointer appearance-none rounded-md border border-gray-300 bg-white py-2 pl-3 pr-10 text-base shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:cursor-not-allowed disabled:bg-gray-200 disabled:opacity-75 sm:text-sm"
          value={selectedSymbol}
          onChange={(e) => onSelectSymbol(e.target.value)}
          disabled={disabled}
        >
          {symbols.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
          <ChevronUpDownIcon className="h-5 w-5 text-gray-400" aria-hidden="true" />
        </div>
      </div>
    </div>
  );
}

// NOTE: You'll also need @heroicons/react for this component.
// npm install @heroicons/react