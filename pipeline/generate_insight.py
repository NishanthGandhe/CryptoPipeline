import os, math, warnings, uuid, json
from datetime import timedelta
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
load_dotenv()

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

# ---------- helpers ----------
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
    """
    Try multiple models; return:
      candidates: list of dict(method, mae, params, notes, model_obj or preds)
      best: dict with chosen forecast + intervals + metrics
    """
    y = df["close"].astype(float)
    n = len(y)
    holdout = max(14, n // 10, 7)
    if n <= holdout + 20:
        last = float(y.iloc[-1])
        return [], {
            "method": "naive_close",
            "forecast": last,
            "lo80": None, "hi80": None, "lo95": None, "hi95": None,
            "holdout_mae": None, "vs_naive_improve": 0.0,
            "holdout_len": holdout, "n_train": n - holdout, "naive_mae": None
        }

    y_train = y.iloc[:-holdout]
    y_test  = y.iloc[-holdout:]

    cand = []

    # 1) Naive
    naive_pred = np.repeat(y_train.iloc[-1], holdout)
    naive_mae = mae(y_test, naive_pred)
    cand.append({
        "method": "naive_close",
        "mae": naive_mae,
        "params": None,
        "notes": None,
        "pred": naive_pred
    })

    # 2) Holt-Winters (trend+weekly seasonality)
    try:
        hw = ExponentialSmoothing(
            y_train, trend="add", seasonal="add", seasonal_periods=7,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=True)
        hw_pred = hw.forecast(holdout)
        hw_mae = mae(y_test, hw_pred)
        cand.append({
            "method": "holtwinters_add_add_7",
            "mae": hw_mae,
            "params": {"trend":"add","seasonal":"add","seasonal_periods":7},
            "notes": None,
            "model": hw
        })
    except Exception as e:
        cand.append({"method":"holtwinters_failed","mae":math.inf,"params":None,"notes":str(e)})

    # 3) SARIMAX (light weekly seasonality)
    try:
        sar = SARIMAX(
            y_train, order=(1,1,1),
            seasonal_order=(0,1,1,7),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
        sar_pred = sar.get_forecast(steps=holdout).predicted_mean
        sar_mae = mae(y_test, sar_pred)
        cand.append({
            "method":"sarimax_111_0117",
            "mae": sar_mae,
            "params":{"order":[1,1,1],"seasonal_order":[0,1,1,7]},
            "notes": None,
            "model": sar
        })
    except Exception as e:
        cand.append({"method":"sarimax_failed","mae":math.inf,"params":None,"notes":str(e)})

    # choose best finite MAE
    finite = [c for c in cand if math.isfinite(c["mae"])]
    finite.sort(key=lambda c: c["mae"])
    best = finite[0] if finite else cand[0]

    # 1-step forecast + intervals
    if best["method"].startswith("holtwinters") and "model" in best:
        m = best["model"]
        fc = float(m.forecast(1).iloc[0])
        resid = y_train - m.fittedvalues.reindex_like(y_train).fillna(method="bfill")
        lo95, hi95 = safe_conf_int_from_resid(fc, resid, 1.96)
        lo80, hi80 = safe_conf_int_from_resid(fc, resid, 1.28)
    elif best["method"].startswith("sarimax") and "model" in best:
        m = best["model"]
        gf = m.get_forecast(steps=1)
        fc = float(gf.predicted_mean.iloc[0])
        try:
            ci95 = gf.conf_int(alpha=0.05)
            lo95, hi95 = float(ci95.iloc[0,0]), float(ci95.iloc[0,1])
        except Exception:
            resid = y_train - m.fittedvalues
            lo95, hi95 = safe_conf_int_from_resid(fc, resid, 1.96)
        lo80, hi80 = safe_conf_int_from_resid(fc, (y_train - m.fittedvalues), 1.28)
    else:
        fc = float(y.iloc[-1])
        lo80 = hi80 = lo95 = hi95 = None

    improve = 0.0
    base = naive_mae if math.isfinite(naive_mae) else None
    if base and math.isfinite(best["mae"]):
        improve = (base - best["mae"]) / base

    best_out = {
        "method": best["method"],
        "forecast": fc,
        "lo80": lo80, "hi80": hi80, "lo95": lo95, "hi95": hi95,
        "holdout_mae": best["mae"] if math.isfinite(best["mae"]) else None,
        "vs_naive_improve": improve,
        "holdout_len": holdout,
        "n_train": len(y_train),
        "naive_mae": base
    }
    return cand, best_out

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

            # Create a shared run_id so training, forecast, and insight are linked
            run_id = uuid.uuid4()

            # 1) Log training run
            cur.execute("""
                insert into model_training_runs
                  (run_id, symbol, ds_next, ds_train_end, holdout_len, n_train,
                   naive_mae, best_method, best_mae, vs_naive_improve)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                run_id, sym, ds_next, last_ds, best["holdout_len"], best["n_train"],
                best["naive_mae"], best["method"], best["holdout_mae"], best["vs_naive_improve"]
            ))

            # 2) Log all candidates
            for c in cand:
                params = c.get("params")
                # Make sure params is JSON serializable
                if params is not None and not isinstance(params, (dict, list)):
                    params = {"value": str(params)}
                cur.execute("""
                    insert into model_training_candidates (run_id, method, mae, params, notes)
                    values (%s,%s,%s,%s,%s)
                """, (
                    run_id, c["method"],
                    None if not math.isfinite(c.get("mae", math.inf)) else float(c["mae"]),
                    json.dumps(params) if params is not None else None,
                    c.get("notes")
                ))

            # 3) Store forecast + insight (use same run_id for traceability)
            cur.execute("""
                insert into model_forecasts (run_id, symbol, ds, forecast_close, forecast_method)
                values (%s, %s, %s, %s, %s)
            """, (run_id, sym, ds_next, float(best["forecast"]), best["method"]))

            detail_parts = []
            if best["holdout_mae"] is not None: detail_parts.append(f"holdout MAE={best['holdout_mae']:.2f}")
            if best["vs_naive_improve"] is not None: detail_parts.append(f"vs naive: {best['vs_naive_improve']:+.0%}")
            if best["lo95"] is not None: detail_parts.append(f"95% CI [{best['lo95']:.2f}, {best['hi95']:.2f}]")
            details = "; ".join(detail_parts) if detail_parts else "No backtest metrics."
            headline = f"{sym} {best['method']} forecast {best['forecast']:.2f} for {ds_next.isoformat()}"

            cur.execute("""
                insert into insights (run_id, symbol, period, headline, details)
                values (%s, %s, 'daily', %s, %s)
            """, (run_id, sym, headline, details))

            inserted += 1

        print(f"Inserted {inserted} model_forecasts + insights + training logs")

if __name__ == "__main__":
    main()
