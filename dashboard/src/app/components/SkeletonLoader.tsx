export function SkeletonLoader() {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3 lg:gap-8">
      {/* Chart Skeleton */}
      <div className="lg:col-span-2 rounded-xl border bg-white p-4 shadow-sm animate-pulse">
        <div className="h-8 w-3/4 rounded-md bg-gray-200 mb-4"></div>
        <div className="h-72 w-full rounded-md bg-gray-200"></div>
      </div>
      {/* Insight Skeleton */}
      <div className="rounded-xl border bg-white p-4 shadow-sm animate-pulse space-y-4">
        <div className="h-8 w-1/2 rounded-md bg-gray-200"></div>
        <div className="h-5 w-full rounded-md bg-gray-200"></div>
        <div className="h-5 w-5/6 rounded-md bg-gray-200"></div>
        <div className="grid grid-cols-3 gap-4 pt-2">
          <div className="h-16 rounded-lg bg-gray-200"></div>
          <div className="h-16 rounded-lg bg-gray-200"></div>
          <div className="h-16 rounded-lg bg-gray-200"></div>
        </div>
        <div className="h-40 w-full rounded-md bg-gray-200 pt-2"></div>
      </div>
    </div>
  );
}