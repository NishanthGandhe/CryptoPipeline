import os, math, warnings, uuid, json
from datetime import timedelta, date
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
load_dotenv()

HORIZONS = [
    ("1d", 1),
    ("1w", 7),
    ("1m", 30),
    ("3m", 90),
    ("6m", 180),
    ("1y", 365),
    ("5y", 1825),
    ("10y", 3650),
]

# ---------- DB utils ----------
def get_conn():
    host = os.getenv("DB_HOST")
    db   = os.getenv("DB_NAME", "postgres")
    user = os.getenv("DB_USER")
    pwd  = os.getenv("DB_PASS")
    port = int(os.getenv("DB_PORT", "5432"))
    missing = [k for k,v in {"DB_HOST":host,"DB_NAME":db,"DB_USER":user,"DB_PASS":pwd}.items() if not v]
    if missing:
        raise RuntimeError(f"Missing DB envs: {missing}")
    return psycopg.connect(host=host, dbname=db, user=user, password=pwd, port=port)

def get_symbols():
    raw = os.getenv("SYMBOLS", "BTC-USD")
    return [s.strip() for s in raw.split(",") if s.strip()]

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def safe_conf_int_from_resid(forecast, resid, z=1.96):
    sd = float(np.std(resid)) if len(resid) else 0.0
    return (forecast - z*sd, forecast + z*sd)

def load_series(cur, symbol, min_rows=50):
    cur.execute("""
        select ds, close
        from gold_daily_metrics
        where symbol=%s
        order by ds
    """, (symbol,))
    rows = cur.fetchall()
    if len(rows) < min_rows:
        return None
    df = pd.DataFrame(rows, columns=["ds","close"]).dropna()
    df = df.set_index("ds").asfreq("D")
    df["close"] = df["close"].interpolate("time")
    return df

def train_candidates(df):
    y = df["close"].astype(float)
    n = len(y)
    holdout = max(14, n // 10, 7)
    if n <= holdout + 20:
        last = float(y.iloc[-1])
        return [], {
            "method": "naive_close",
            "forecast": last,
            "holdout_mae": None, "vs_naive_improve": 0.0,
            "holdout_len": holdout, "n_train": n - holdout,
            "naive_mae": None,
            "best_model": None, "best_type": "naive"
        }

    y_train = y.iloc[:-holdout]
    y_test  = y.iloc[-holdout:]

    cand = []

    # Naive
    naive_pred = np.repeat(y_train.iloc[-1], holdout)
    naive_mae = mae(y_test, naive_pred)
    cand.append({"method":"naive_close","mae":naive_mae,"type":"naive"})

    # Holt-Winters
    try:
        hw = ExponentialSmoothing(
            y_train, trend="add", seasonal="add", seasonal_periods=7,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=True)
        hw_pred = hw.forecast(holdout)
        hw_mae = mae(y_test, hw_pred)
        cand.append({"method":"holtwinters_add_add_7","mae":hw_mae,"type":"hw","model":hw})
    except Exception as e:
        cand.append({"method":"holtwinters_failed","mae":math.inf,"type":"hw_failed","notes":str(e)})

    # SARIMAX
    try:
        sar = SARIMAX(
            y_train, order=(1,1,1),
            seasonal_order=(0,1,1,7),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
        sar_pred = sar.get_forecast(steps=holdout).predicted_mean
        sar_mae = mae(y_test, sar_pred)
        cand.append({"method":"sarimax_111_0117","mae":sar_mae,"type":"sarimax","model":sar})
    except Exception as e:
        cand.append({"method":"sarimax_failed","mae":math.inf,"type":"sar_failed","notes":str(e)})

    finite = [c for c in cand if math.isfinite(c["mae"])]
    finite.sort(key=lambda c: c["mae"])
    best = finite[0] if finite else cand[0]

    # One-step for the simple headline
    if best["type"] == "hw":
        m = best["model"]
        one_fc = float(m.forecast(1).iloc[0])
    elif best["type"] == "sarimax":
        m = best["model"]
        one_fc = float(m.get_forecast(steps=1).predicted_mean.iloc[0])
    else:
        one_fc = float(y.iloc[-1])

    improve = 0.0
    base = naive_mae if math.isfinite(naive_mae) else None
    if base and math.isfinite(best["mae"]):
        improve = (base - best["mae"]) / base

    return cand, {
        "method": best["method"],
        "forecast": one_fc,
        "holdout_mae": best["mae"] if math.isfinite(best["mae"]) else None,
        "vs_naive_improve": improve,
        "holdout_len": holdout,
        "n_train": len(y_train),
        "naive_mae": base,
        "best_model": best.get("model"),
        "best_type": best["type"],
        "y_train": y_train,
        "y_full": y,
    }

def multi_horizon_forecast(best, max_h):
    """Return (pred[1..max_h], lo95, hi95) lists using the chosen model."""
    btype = best["best_type"]
    y_train = best["y_train"]
    if btype == "sarimax" and best["best_model"] is not None:
        m = best["best_model"]
        gf = m.get_forecast(steps=max_h)
        pred = gf.predicted_mean.to_numpy()
        try:
            ci95 = gf.conf_int(alpha=0.05).to_numpy()
            lo = ci95[:,0]; hi = ci95[:,1]
        except Exception:
            resid = (y_train - m.fittedvalues).to_numpy()
            lo = pred - 1.96*np.std(resid)
            hi = pred + 1.96*np.std(resid)
        return pred, lo, hi
    elif btype == "hw" and best["best_model"] is not None:
        m = best["best_model"]
        pred = m.forecast(max_h).to_numpy()
        resid = (y_train - m.fittedvalues.reindex_like(y_train).fillna(method="bfill")).to_numpy()
        sd = np.std(resid) if len(resid) else 0.0
        lo = pred - 1.96*sd
        hi = pred + 1.96*sd
        return pred, lo, hi
    else:
        # naive: flat line
        last = float(best["y_full"].iloc[-1])
        pred = np.repeat(last, max_h)
        return pred, np.full(max_h, np.nan), np.full(max_h, np.nan)

def main():
    symbols = get_symbols()
    inserted = 0
    with get_conn() as conn, conn.cursor() as cur:
        for sym in symbols:
            df = load_series(cur, sym, min_rows=50)
            if df is None:
                print(f"{sym}: not enough history; skipping.")
                continue

            cand, best = train_candidates(df)
            last_ds = df.index[-1].date()
            ds_next = last_ds + timedelta(days=1)
            max_h = max(h for _, h in HORIZONS)

            # Fresh run_id that we control
            run_id = uuid.uuid4()

            # 1) Log training run FIRST
            cur.execute("""
                insert into model_training_runs
                  (run_id, symbol, ds_next, ds_train_end, holdout_len, n_train,
                   naive_mae, best_method, best_mae, vs_naive_improve)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                run_id, sym, ds_next, last_ds, best["holdout_len"], best["n_train"],
                best["naive_mae"], best["method"], best["holdout_mae"], best["vs_naive_improve"]
            ))

            # Safety check: ensure the FK target exists before forecasts
            cur.execute("select 1 from model_training_runs where run_id = %s", (run_id,))
            assert cur.fetchone(), "training run missing just after insert?!"

            # 2) Log candidates (optional but useful)
            for c in cand:
                params = None
                if c.get("method","").startswith("holtwinters"):
                    params = {"trend":"add","seasonal":"add","seasonal_periods":7}
                elif c.get("method","").startswith("sarimax"):
                    params = {"order":[1,1,1], "seasonal_order":[0,1,1,7]}
                cur.execute("""
                    insert into model_training_candidates (run_id, method, mae, params, notes)
                    values (%s,%s,%s,%s,%s)
                """, (
                    run_id, c["method"],
                    None if not math.isfinite(c.get("mae", math.inf)) else float(c["mae"]),
                    json.dumps(params) if params is not None else None,
                    c.get("notes")
                ))

            # 3) Forecast multiple horizons with the chosen model
            pred, lo95, hi95 = multi_horizon_forecast(best, max_h)

            # UPSERT each horizon so retries are safe
            for label, h in HORIZONS:
                tgt = last_ds + timedelta(days=h)
                fc = float(pred[h-1])  # horizons are 1-indexed
                cur.execute("""
                    insert into model_forecasts (run_id, symbol, ds, forecast_close, forecast_method)
                    values (%s, %s, %s, %s, %s)
                    on conflict (run_id, ds) do update
                      set forecast_close = excluded.forecast_close,
                          forecast_method = excluded.forecast_method
                """, (run_id, sym, tgt, fc, best["method"]))

            # 4) Human-friendly insight (short)
            def fmt(x): return f"${x:,.2f}"
            one_d = float(pred[0]); one_w = float(pred[7-1]); one_m = float(pred[30-1])
            headline = f"{sym} {best['method']} forecasts — 1d: {fmt(one_d)}, 1w: {fmt(one_w)}, 1m: {fmt(one_m)}"

            details_bits = []
            if best["holdout_mae"] is not None:
                details_bits.append(f"MAE={best['holdout_mae']:.2f}")
            if best["vs_naive_improve"] is not None:
                details_bits.append(f"vs naive: {best['vs_naive_improve']:+.0%}")
            if not np.isnan(lo95[0]):
                details_bits.append(f"1d 95% CI [{fmt(lo95[0])}, {fmt(hi95[0])}]")
            details_bits.append("Long-horizon forecasts have high uncertainty.")
            details = "; ".join(details_bits)

            cur.execute("""
                insert into insights (run_id, symbol, period, headline, details)
                values (%s, %s, 'daily', %s, %s)
            """, (run_id, sym, headline, details))

            inserted += 1
            print(f"[{sym}] run_id={run_id} wrote {len(HORIZONS)} horizons + insight")

        print(f"Inserted {inserted} runs")

if __name__ == "__main__":
    main()