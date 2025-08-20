create or replace view public.latest_forecasts_all as
with lr as (
  select symbol, max(created_at) as created_at
  from public.model_training_runs
  group by 1
)
select f.symbol, f.ds, f.forecast_close, f.forecast_method, r.created_at
from public.model_training_runs r
join lr on lr.symbol = r.symbol and lr.created_at = r.created_at
join public.model_forecasts f on f.run_id = r.run_id
order by f.symbol, f.ds;

grant select on public.latest_forecasts_all to anon, authenticated;

create or replace view public.latest_forecast_insight as
with li as (
  select symbol, max(created_at) as created_at
  from public.insights
  group by 1
),
joined as (
  select i.symbol, i.run_id, i.headline, i.details, i.created_at
  from public.insights i
  join li on li.symbol = i.symbol and li.created_at = i.created_at
),
first_horizon as (
  select f.run_id, min(f.ds) as first_ds
  from public.model_forecasts f
  group by 1
)
select
  j.symbol,
  f.ds                    as forecast_for_day,
  f.forecast_close,
  f.forecast_method,
  j.headline,
  j.details,
  j.created_at
from joined j
join first_horizon fh on fh.run_id = j.run_id
join public.model_forecasts f on f.run_id = j.run_id and f.ds = fh.first_ds;

grant select on public.latest_forecast_insight to anon, authenticated;
