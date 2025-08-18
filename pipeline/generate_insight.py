import os
from datetime import timedelta
import psycopg
from dotenv import load_dotenv

load_dotenv()

def get_conn():
    host = os.getenv("DB_HOST")
    db   = os.getenv("DB_NAME", "postgres")
    user = os.getenv("DB_USER")
    pwd  = os.getenv("DB_PASS")
    port = int(os.getenv("DB_PORT", "5432"))
    return psycopg.connect(host=host, dbname=db, user=user, password=pwd, port=port)

def get_symbols():
    raw = os.getenv("SYMBOLS", "BTC-USD")
    return [s.strip() for s in raw.split(",") if s.strip()]

INSIGHT_TEMPLATE = (
    "{sym} close {close:.2f}; MA7 {ma7:.2f}; "
    "{dir} {pct:+.2%} vs yesterday. Anomaly: {risk} (z={z:.2f})."
)

def risk_from_z(z):
    if z is None: return "unknown"
    z = abs(float(z))
    return "high" if z >= 2 else "medium" if z >= 1 else "low"

def latest_row(cur, symbol):
    cur.execute("""
        select ds, close, ret_1d, ma_7, zscore_close
        from gold_daily_metrics
        where symbol=%s
        order by ds
    """, (symbol,))
    rows = cur.fetchall()
    if len(rows) < 2:
        return None, None
    return rows[-2], rows[-1]  # (prev, last)

def main():
    symbols = get_symbols()
    inserted = 0
    with get_conn() as conn, conn.cursor() as cur:
        for sym in symbols:
            prev, last = latest_row(cur, sym)
            if not last:
                print(f"{sym}: not enough data in gold_daily_metrics, skipping.")
                continue
            last_ds, last_close, last_ret, last_ma7, last_z = last
            ds_next = last_ds + timedelta(days=1)
            forecast_close = float(last_close)  # naive baseline

            direction = "Up" if (last_ret or 0) > 0 else "Down" if (last_ret or 0) < 0 else "Flat"
            risk = risk_from_z(last_z)
            headline = f"{sym} forecast {forecast_close:.2f} for {ds_next.isoformat()} (naive)"
            details  = INSIGHT_TEMPLATE.format(
                sym=sym,
                close=float(last_close),
                ma7=float(last_ma7) if last_ma7 is not None else float(last_close),
                dir=direction,
                pct=float(last_ret or 0.0),
                risk=risk,
                z=float(last_z or 0.0),
            )

            # write forecast + insight
            cur.execute("""
                insert into model_forecasts (symbol, ds, forecast_close, forecast_method)
                values (%s, %s, %s, 'naive_close')
                returning run_id
            """, (sym, ds_next, forecast_close))
            run_id = cur.fetchone()[0]
            cur.execute("""
                insert into insights (run_id, symbol, period, headline, details)
                values (%s, %s, 'daily', %s, %s)
            """, (run_id, sym, headline, details))
            inserted += 1

        print(f"Inserted {inserted} forecast/insight pairs.")

if __name__ == "__main__":
    main()
