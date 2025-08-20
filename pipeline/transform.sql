-- pipeline/transform.sql
-- Idempotent: (re)defines views; no inserts into views.

-- Drop downstream views first to avoid dependency errors
drop view if exists public.price_last_120d cascade;
drop view if exists public.symbols cascade;
drop view if exists public.latest_spot cascade;
drop view if exists public.price_daily cascade;

-- Recreate SILVER (hourly) as a VIEW over bronze_ticks.
-- Handles both payload schemas: {"open","high","low","close","volume"} and {"o","h","l","c","v"}.
create or replace view public.silver_prices as
select
  symbol,
  ts::timestamptz                                           as ts,
  coalesce((payload->>'open')::numeric,  (payload->>'o')::numeric) as open,
  coalesce((payload->>'high')::numeric,  (payload->>'h')::numeric) as high,
  coalesce((payload->>'low')::numeric,   (payload->>'l')::numeric) as low,
  coalesce((payload->>'close')::numeric, (payload->>'c')::numeric) as close,
  coalesce((payload->>'volume')::numeric,(payload->>'v')::numeric) as volume
from public.bronze_ticks
where source = 'crypto';

grant select on public.silver_prices to anon, authenticated;

-- Daily close = last hourly bar per UTC day
create or replace view public.price_daily as
with marked as (
  select
    symbol,
    (ts at time zone 'utc')::date as ds,
    ts,
    close
  from public.silver_prices
),
ranked as (
  select *,
         row_number() over (partition by symbol, ds order by ts desc) as rn
  from marked
)
select symbol, ds, close
from ranked
where rn = 1;

grant select on public.price_daily to anon, authenticated;

-- Gold metrics computed from price_daily
create or replace view public.gold_daily_metrics as
select
  d.symbol,
  d.ds,
  d.close,
  (d.close / lag(d.close) over (partition by d.symbol order by d.ds) - 1.0) as ret_1d,
  avg(d.close) over (
    partition by d.symbol
    order by d.ds
    rows between 6 preceding and current row
  ) as ma_7,
  case
    when stddev_samp(d.close) over (
      partition by d.symbol order by d.ds
      rows between 29 preceding and current row
    ) > 0
    then (d.close - avg(d.close) over (
            partition by d.symbol order by d.ds
            rows between 29 preceding and current row
         ))
       / stddev_samp(d.close) over (
            partition by d.symbol order by d.ds
            rows between 29 preceding and current row
         )
  end as zscore_close
from public.price_daily d;

grant select on public.gold_daily_metrics to anon, authenticated;

-- Symbols to show in the UI (only those with enough data)
create or replace view public.symbols as
select symbol
from public.gold_daily_metrics
where close is not null
group by symbol
having count(*) >= 60;

grant select on public.symbols to anon, authenticated;

-- 120-day chart helper
create or replace view public.price_last_120d as
select
  symbol,
  ds,
  close,
  avg(close) over (
    partition by symbol
    order by ds
    rows between 6 preceding and current row
  ) as ma_7
from public.gold_daily_metrics
where ds >= (current_date - interval '120 days') and close is not null;

grant select on public.price_last_120d to anon, authenticated;

-- Latest hourly close per symbol (for "Current price" tile)
create or replace view public.latest_spot as
select s.symbol, s.ts, s.close
from public.silver_prices s
join (
  select symbol, max(ts) as max_ts
  from public.silver_prices
  group by 1
) m on m.symbol = s.symbol and m.max_ts = s.ts;

grant select on public.latest_spot to anon, authenticated;
