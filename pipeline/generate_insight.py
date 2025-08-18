import os
from datetime import date, timedelta
import psycopg
from dotenv import load_dotenv

load_dotenv()

# If you prefer using the pooler host and separate fields:
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

SYMBOL = "BTC-USD"

INSIGHT_TEMPLATE = (
    "BTC daily close {close:.2f}; 7-day MA {ma7:.2f}; "
    "{dir} {pct:+.2%} vs yesterday. Anomaly risk: {risk} (z={z:.2f})."
)

def main():
    # Connect
    with psycopg.connect(user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME) as conn:
        with conn.cursor() as cur:
            # Pull the most recent 35 days of gold metrics
            cur.execute("""
                select ds, close, ret_1d, ma_7, zscore_close
                from gold_daily_metrics
                where symbol=%s
                order by ds
                limit 1000
            """, (SYMBOL,))
            rows = cur.fetchall()
            if len(rows) < 2:
                raise RuntimeError("Not enough days in gold_daily_metrics to generate an insight.")

            last_ds, last_close, last_ret, last_ma7, last_z = rows[-1]
            prev_ds, prev_close, *_ = rows[-2]

            # Naive forecast: tomorrow's close = today's close (simple baseline)
            forecast_close = float(last_close)
            ds_next = last_ds + timedelta(days=1)

            # Severity from z-score
            if last_z is None:
                risk = "unknown"
            else:
                zabs = abs(float(last_z))
                risk = "high" if zabs >= 2.0 else ("medium" if zabs >= 1.0 else "low")

            # Direction text
            direction = "Up" if (last_ret or 0) > 0 else "Down" if (last_ret or 0) < 0 else "Flat"

            headline = f"BTC forecast {forecast_close:.2f} for {ds_next.isoformat()} (naïve)"
            details  = INSIGHT_TEMPLATE.format(
                close=float(last_close),
                ma7=float(last_ma7) if last_ma7 is not None else float(last_close),
                dir=direction,
                pct=float(last_ret or 0.0),
                risk=risk,
                z=float(last_z or 0.0)
            )

            # Insert forecast and insight
            cur.execute("""
                insert into model_forecasts (symbol, ds, forecast_close, forecast_method)
                values (%s, %s, %s, %s)
                returning run_id
            """, (SYMBOL, ds_next, forecast_close, 'naive_close'))
            run_id = cur.fetchone()[0]

            cur.execute("""
                insert into insights (run_id, symbol, period, headline, details)
                values (%s, %s, 'daily', %s, %s)
            """, (run_id, SYMBOL, headline, details))

            print("Inserted forecast + insight.")
    print("Done.")

if __name__ == "__main__":
    main()
