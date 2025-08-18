insert into silver_prices (symbol, ts, open, high, low, close, volume)
select
  bt.symbol,
  to_timestamp( ((bt.payload->>'openTime')::bigint) / 1000.0 ) as ts,
  (bt.payload->>'o')::numeric as open,
  (bt.payload->>'h')::numeric as high,
  (bt.payload->>'l')::numeric as low,
  (bt.payload->>'c')::numeric as close,
  (bt.payload->>'v')::numeric as volume
from bronze_ticks bt
where bt.source = 'crypto'
  and not exists (
    select 1 from silver_prices sp
    where sp.symbol = bt.symbol
      and sp.ts = to_timestamp( ((bt.payload->>'openTime')::bigint) / 1000.0 )
  );