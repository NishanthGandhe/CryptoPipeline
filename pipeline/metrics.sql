-- SILVER -> GOLD: compute daily close (UTC) and features

with hourly as (
  select symbol, ts, close
  from silver_prices
),
hourly_utc as (
  select
    symbol,
    ts,
    close,
    (ts at time zone 'UTC')::date as ds   -- day boundaries in UTC
  from hourly
),
daily_last as (
  -- last bar per (symbol, day)
  select
    ds,
    symbol,
    max(ts) as last_ts
  from hourly_utc
  group by 1,2
),
daily_close as (
  select dl.ds, dl.symbol, h.close
  from daily_last dl
  join hourly_utc h
    on h.symbol = dl.symbol
   and h.ts = dl.last_ts
),
features as (
  select
    ds,
    symbol,
    close,
    (close - lag(close) over (partition by symbol order by ds))
      / nullif(lag(close) over (partition by symbol order by ds), 0) as ret_1d,
    avg(close) over (
      partition by symbol order by ds
      rows between 6 preceding and current row
    ) as ma_7,
    (close - avg(close) over (
       partition by symbol order by ds
       rows between 27 preceding and current row
     )) / nullif(stddev_samp(close) over (
       partition by symbol order by ds
       rows between 27 preceding and current row
     ), 0) as zscore_close
  from daily_close
)
insert into gold_daily_metrics (ds, symbol, close, ret_1d, ma_7, zscore_close)
select ds, symbol, close, ret_1d, ma_7, zscore_close
from features
on conflict (ds, symbol) do nothing;
