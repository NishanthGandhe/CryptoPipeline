drop view if exists public.price_last_120d cascade;
drop view if exists public.symbols cascade;
drop view if exists public.latest_spot cascade;
drop view if exists public.price_daily cascade;

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

CREATE OR REPLACE VIEW public.gold_daily_metrics AS
SELECT
  pd.symbol,
  pd.ds,
  pd.close,

  (pd.close / LAG(pd.close, 1) OVER (PARTITION BY pd.symbol ORDER BY pd.ds) - 1.0) AS ret_1d,
  AVG(pd.close) OVER (PARTITION BY pd.symbol ORDER BY pd.ds ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma_7d,

  bdm.trade_volume_usd,
  bdm.total_supply

FROM public.price_daily pd
LEFT JOIN public.bronze_daily_metrics bdm
ON pd.symbol = bdm.symbol AND pd.ds = bdm.ds;

grant select on public.gold_daily_metrics to anon, authenticated;

create or replace view public.symbols as
select symbol
from public.gold_daily_metrics
where close is not null
group by symbol
having count(*) >= 60;

grant select on public.symbols to anon, authenticated;

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

create or replace view public.latest_spot as
select s.symbol, s.ts, s.close
from public.silver_prices s
join (
  select symbol, max(ts) as max_ts
  from public.silver_prices
  group by 1
) m on m.symbol = s.symbol and m.max_ts = s.ts;

grant select on public.latest_spot to anon, authenticated;
