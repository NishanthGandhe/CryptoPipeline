import os, math, warnings, uuid, json
from datetime import timedelta
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

# classical models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# optional models
try:
    import pmdarima as pm
    HAVE_PM = True
except Exception:
    HAVE_PM = False

try:
    from prophet import Prophet
    HAVE_PROPHET = True
except Exception:
    HAVE_PROPHET = False

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:
    torch = None
    nn = None
    HAVE_TORCH = False

warnings.filterwarnings("ignore")
load_dotenv()

# ---------- Config ----------
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

TRAIN_WINDOW_DAYS = int(os.getenv("TRAIN_WINDOW_DAYS", "365"))
PREFER_NON_NAIVE_EPS = float(os.getenv("MODEL_PREFER_NON_NAIVE_EPS", "0.02"))  # 2%
PREFER_DYNAMIC_THRESHOLD = float(os.getenv("PREFER_DYNAMIC_THRESHOLD", "0.05"))  # 5%
LOG_SPACE = os.getenv("LOG_SPACE", "1").lower() in ("1", "true", "yes")
USE_PROPHET = os.getenv("USE_PROPHET", "0").lower() in ("1", "true", "yes")
USE_DEEP = os.getenv("USE_DEEP", "0").lower() in ("1", "true", "yes")
DEEP_EPOCHS = int(os.getenv("DEEP_EPOCHS", "40"))
DEEP_LAG = int(os.getenv("DEEP_LAG", "30"))

# ---------- DB utils ----------
def get_conn():
    dsn = os.getenv("SUPABASE_CONNECTION")
    if dsn:
        return psycopg.connect(dsn)
    host = os.getenv("DB_HOST"); db = os.getenv("DB_NAME", "postgres")
    user = os.getenv("DB_USER"); pwd = os.getenv("DB_PASS")
    port = int(os.getenv("DB_PORT", "5432"))
    if not (host and user and pwd):
        raise RuntimeError("Missing DB envs or SUPABASE_CONNECTION")
    return psycopg.connect(host=host, dbname=db, user=user, password=pwd, port=port, sslmode="require")

def get_symbols():
    raw = os.getenv("SYMBOLS", "BTC-USD")
    return [s.strip() for s in raw.split(",") if s.strip()]

# ---------- helpers ----------
def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def mae_pct(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    denom = float(np.mean(np.abs(y_true))) or 1.0
    return mae(y_true, y_pred) / denom

def safe_conf_int_from_resid_level(pred_level, resid, z=1.96):
    pred = np.asarray(pred_level, dtype=float)
    sd = float(np.std(np.asarray(resid, dtype=float))) if len(resid) else 0.0
    lo = pred - z * sd
    hi = pred + z * sd
    return lo, hi

def safe_conf_int_from_resid_log(pred_level, resid_log, z=1.96):
    pred = np.asarray(pred_level, dtype=float)
    mu_log = np.log(np.clip(pred, 1e-12, None))
    sd = float(np.std(np.asarray(resid_log, dtype=float))) if len(resid_log) else 0.0
    lo = np.exp(mu_log - z * sd)
    hi = np.exp(mu_log + z * sd)
    return lo, hi

def load_series(cur, symbol, min_rows=100):
    # Pull daily gold; ignore obvious nulls early
    cur.execute("""
        select ds, close
        from gold_daily_metrics
        where symbol = %s
        order by ds
    """, (symbol,))
    rows = cur.fetchall()
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["ds", "close"])

    # Coerce types & clean
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = (
        df.dropna(subset=["ds", "close"])
          .drop_duplicates(subset=["ds"], keep="last")
          .sort_values("ds")
    )

    # If too few points after cleaning, skip symbol
    if len(df) < min_rows:
        return None

    # Build a proper daily index and fill small gaps
    df = df.set_index("ds")
    start = df.index.min()
    end = df.index.max()
    if pd.isna(start) or pd.isna(end):
        # nothing usable
        return None

    full_idx = pd.date_range(start, end, freq="D")
    df = df.reindex(full_idx)
    df.index.name = "ds"
    df["close"] = df["close"].interpolate(method="time").ffill().bfill()

    # Optional rolling window
    if TRAIN_WINDOW_DAYS > 0:
        cutoff = df.index.max() - pd.Timedelta(days=TRAIN_WINDOW_DAYS - 1)
        df = df[df.index >= cutoff]

    return df

# ---------- optional: tiny LSTM for short-horizon ----------
if HAVE_TORCH:
    class LSTMReg(nn.Module):
        def __init__(self, input_size=1, hidden=32, layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True)
            self.head = nn.Linear(hidden, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    def fit_lstm(series, lag=30, epochs=40):
        series = np.asarray(series, dtype=np.float32)
        x, y = [], []
        for i in range(len(series) - lag):
            x.append(series[i:i+lag])
            y.append(series[i+lag])
        x = np.array(x, dtype=np.float32)[:, :, None]
        y = np.array(y, dtype=np.float32)[:, None]
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        model = LSTMReg()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = torch.nn.L1Loss()
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x_t)
            loss = lossf(pred, y_t)
            loss.backward()
            opt.step()
        model.eval()
        return model

    def lstm_forecast(model, last_window, steps):
        out = []
        cur = np.asarray(last_window, dtype=np.float32).copy()
        for _ in range(steps):
            x = torch.from_numpy(cur[None, :, None])
            with torch.no_grad():
                yhat = model(x).numpy().ravel()[0]
            out.append(float(yhat))
            cur = np.roll(cur, -1); cur[-1] = yhat
        return np.array(out, dtype=float)
else:
    # Safe stubs so accidental calls give a clear error
    def fit_lstm(*args, **kwargs):
        raise RuntimeError("USE_DEEP=1 but PyTorch is not installed on this runner.")
    def lstm_forecast(*args, **kwargs):
        raise RuntimeError("USE_DEEP=1 but PyTorch is not installed on this runner.")

# ---------- training + selection ----------
def train_candidates(df):
    """
    Try several models; select winner by MAE% (scale-normalized).
    Prefer a non-naive if it is within PREFER_NON_NAIVE_EPS of naive.
    Returns (candidates, best_summary).
    """
    y_level = df["close"].astype(float)
    n = len(y_level)

    holdout = max(14, n // 10, 7)
    if n <= holdout + 30:
        # too short for complex models
        last = float(y_level.iloc[-1])
        return [], {
            "method": "naive_close",
            "forecast": last,
            "holdout_mae": None, "holdout_mae_pct": None,
            "vs_naive_improve": 0.0,
            "holdout_len": holdout, "n_train": n - holdout,
            "naive_mae": None, "naive_mae_pct": None,
            "best_model": None, "best_type": "naive",
            "y_train_level": y_level.iloc[:-holdout] if n > holdout else y_level,
            "y_full_level": y_level,
        }

    # train / test split
    y_train_level = y_level.iloc[:-holdout]
    y_test_level  = y_level.iloc[-holdout:]

    # transform (log) if enabled
    if LOG_SPACE:
        y_train = np.log(y_train_level.values)
    else:
        y_train = y_train_level.values

    cand = []

    # 1) Naive
    naive_pred_h = np.repeat(y_train_level.iloc[-1], holdout)
    naive_mae_val = mae(y_test_level, naive_pred_h)
    naive_mae_pct_val = mae_pct(y_test_level, naive_pred_h)
    cand.append({
        "method": "naive_close",
        "mae": naive_mae_val,
        "mae_pct": naive_mae_pct_val,
        "type": "naive"
    })

    # 2) Holt-Winters
    try:
        hw = ExponentialSmoothing(
            y_train if not LOG_SPACE else pd.Series(y_train, index=y_train_level.index),
            trend="add", seasonal="add", seasonal_periods=7,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=True)
        hw_fc_t = hw.forecast(holdout).to_numpy()
        hw_fc = np.exp(hw_fc_t) if LOG_SPACE else hw_fc_t
        hw_mae = mae(y_test_level, hw_fc)
        hw_mae_pct = mae_pct(y_test_level, hw_fc)
        cand.append({"method": "holtwinters_add_add_7", "mae": hw_mae, "mae_pct": hw_mae_pct, "type": "hw", "model": hw})
    except Exception as e:
        cand.append({"method": "holtwinters_failed", "mae": math.inf, "mae_pct": math.inf, "type": "hw_failed", "notes": str(e)})

    # 3) SARIMAX
    try:
        sar = SARIMAX(
            y_train, order=(1,1,1), seasonal_order=(0,1,1,7),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
        sar_fc_t = sar.get_forecast(steps=holdout).predicted_mean.to_numpy()
        sar_fc = np.exp(sar_fc_t) if LOG_SPACE else sar_fc_t
        sar_mae = mae(y_test_level, sar_fc)
        sar_mae_pct = mae_pct(y_test_level, sar_fc)
        cand.append({"method": "sarimax_111_0117", "mae": sar_mae, "mae_pct": sar_mae_pct, "type": "sarimax", "model": sar})
    except Exception as e:
        cand.append({"method": "sarimax_failed", "mae": math.inf, "mae_pct": math.inf, "type": "sar_failed", "notes": str(e)})

    # 4) Auto-ARIMA (pmdarima)
    if HAVE_PM:
        try:
            ar = pm.auto_arima(
                y_train, seasonal=True, m=7, suppress_warnings=True, error_action="ignore",
                max_p=3, max_q=3, max_P=2, max_Q=2, stepwise=True
            )
            ar_fc_t = ar.predict(n_periods=holdout)
            ar_fc = np.exp(ar_fc_t) if LOG_SPACE else ar_fc_t
            ar_mae = mae(y_test_level, ar_fc)
            ar_mae_pct = mae_pct(y_test_level, ar_fc)
            cand.append({"method": "auto_arima_m7", "mae": ar_mae, "mae_pct": ar_mae_pct, "type": "arima", "model": ar})
        except Exception as e:
            cand.append({"method": "auto_arima_failed", "mae": math.inf, "mae_pct": math.inf, "type": "ar_failed", "notes": str(e)})
    else:
        cand.append({"method": "auto_arima_unavailable", "mae": math.inf, "mae_pct": math.inf, "type": "ar_unavail", "notes": "pmdarima not installed"})

    # 5) Prophet (optional)
    if USE_PROPHET and HAVE_PROPHET:
        try:
            dfp = pd.DataFrame({
                "ds": y_train_level.index,
                "y": np.log(y_train_level.values) if LOG_SPACE else y_train_level.values
            })
            m = Prophet(
                weekly_seasonality=True, yearly_seasonality=True,
                daily_seasonality=False, interval_width=0.95
            ).fit(dfp)
            fhold = m.make_future_dataframe(periods=holdout, freq="D")
            fc = m.predict(fhold)
            yhat_t = fc["yhat"].iloc[-holdout:].to_numpy()
            yhat = np.exp(yhat_t) if LOG_SPACE else yhat_t
            p_mae = mae(y_test_level, yhat)
            p_mae_pct = mae_pct(y_test_level, yhat)
            cand.append({"method": "prophet_w_y", "mae": p_mae, "mae_pct": p_mae_pct, "type": "prophet", "model": m})
        except Exception as e:
            cand.append({"method": "prophet_failed", "mae": math.inf, "mae_pct": math.inf, "type": "prophet_failed", "notes": str(e)})
    elif USE_PROPHET and not HAVE_PROPHET:
        cand.append({"method": "prophet_unavailable", "mae": math.inf, "mae_pct": math.inf, "type": "prophet_unavail", "notes": "prophet not installed"})

    # 6) LSTM (optional, short horizon surrogate)
    if USE_DEEP and HAVE_TORCH and len(y_train_level) > (DEEP_LAG + 60):
        try:
            # standardize level series (log-space already stabilizes variance, but we keep it simple here)
            s = y_train_level.values.astype(float)
            mu, sd = float(np.mean(s)), float(np.std(s) or 1.0)
            s_norm = (s - mu) / sd
            model = fit_lstm(s_norm, lag=DEEP_LAG, epochs=DEEP_EPOCHS)
            # recursive forecast on normalized scale then destandardize
            last_win = s_norm[-DEEP_LAG:]
            lstm_fc_norm = lstm_forecast(model, last_win, holdout)
            lstm_fc = lstm_fc_norm * sd + mu
            l_mae = mae(y_test_level, lstm_fc)
            l_mae_pct = mae_pct(y_test_level, lstm_fc)
            cand.append({"method": "lstm_small", "mae": l_mae, "mae_pct": l_mae_pct, "type": "lstm", "model": (model, mu, sd)})
        except Exception as e:
            cand.append({"method": "lstm_failed", "mae": math.inf, "mae_pct": math.inf, "type": "lstm_failed", "notes": str(e)})
    elif USE_DEEP and not HAVE_TORCH:
        cand.append({"method": "lstm_unavailable", "mae": math.inf, "mae_pct": math.inf, "type": "lstm_unavail", "notes": "torch not installed"})

    # --- selection by MAE% ---
    finite = [c for c in cand if math.isfinite(c["mae_pct"])]
    finite.sort(key=lambda c: c["mae"])
    best = finite[0] if finite else cand[0]

    # prefer non-naive if close to naive
    naive_c = next((c for c in cand if c.get("type") == "naive"), None)
    base = naive_c["mae"] if naive_c and math.isfinite(naive_c["mae"]) else None
    if best["method"] == "naive_close" and base and math.isfinite(base):
        alt = next((c for c in finite if c.get("type") in ("hw", "sarimax") and math.isfinite(c["mae"])), None)
        if alt and alt["mae"] <= base * (1.0 + PREFER_DYNAMIC_THRESHOLD):
            best = alt

    # 1-step (next day) for headline
    bt = best["type"]
    if bt == "hw":
        m = best["model"]
        one_t = float(m.forecast(1).iloc[0])
        one_fc = float(math.exp(one_t)) if LOG_SPACE else one_t
    elif bt == "sarimax":
        m = best["model"]
        one_t = float(m.get_forecast(steps=1).predicted_mean.iloc[0])
        one_fc = float(math.exp(one_t)) if LOG_SPACE else one_t
    elif bt == "arima":
        m = best["model"]
        one_t = float(m.predict(n_periods=1)[0])
        one_fc = float(math.exp(one_t)) if LOG_SPACE else one_t
    elif bt == "prophet":
        m = best["model"]
        df_future = m.make_future_dataframe(periods=1, freq="D")
        yhat_t = float(m.predict(df_future)["yhat"].iloc[-1])
        one_fc = float(math.exp(yhat_t)) if LOG_SPACE else yhat_t
    elif bt == "lstm":
        model, mu, sd = best["model"]
        s = y_train_level.values.astype(float)
        s_norm = (s - mu) / sd
        last_win = s_norm[-DEEP_LAG:]
        one_fc = float(lstm_forecast(model, last_win, 1)[0])
    else:
        one_fc = float(y_train_level.iloc[-1])

    # metrics for logging
    naive_mae_val = naive_c["mae"] if naive_c else None
    naive_mae_pct_val = naive_c["mae_pct"] if naive_c else None
    improve = 0.0
    if naive_mae_pct_val and math.isfinite(best["mae_pct"]):
        improve = (naive_mae_pct_val - best["mae_pct"]) / naive_mae_pct_val

    return cand, {
        "method": best["method"],
        "forecast": one_fc,
        "holdout_mae": best["mae"] if math.isfinite(best["mae"]) else None,
        "holdout_mae_pct": best["mae_pct"] if math.isfinite(best["mae_pct"]) else None,
        "vs_naive_improve": improve,
        "holdout_len": holdout,
        "n_train": len(y_train_level),
        "naive_mae": naive_mae_val,
        "naive_mae_pct": naive_mae_pct_val,
        "best_model": best.get("model"),
        "best_type": bt,
        "y_train_level": y_train_level,
        "y_full_level": y_level,
    }

# ---------- multi-horizon forecast ----------
def multi_horizon_forecast(best, max_h):
    bt = best["best_type"]
    y_train_level = best["y_train_level"]

    if bt == "sarimax":
        m = best["best_model"]
        gf = m.get_forecast(steps=max_h)
        pred_t = gf.predicted_mean.to_numpy()
        pred = np.exp(pred_t) if LOG_SPACE else pred_t
        # CI on log if LOG_SPACE, else level
        try:
            ci95 = gf.conf_int(alpha=0.05).to_numpy()
            lo_t, hi_t = ci95[:,0], ci95[:,1]
            if LOG_SPACE:
                lo = np.exp(lo_t); hi = np.exp(hi_t)
            else:
                lo, hi = lo_t, hi_t
        except Exception:
            resid = (y_train_level.values - pred[:len(y_train_level)])  # fallback rough
            lo, hi = safe_conf_int_from_resid_level(pred, resid, 1.96)
        return pred, lo, hi

    if bt == "hw":
        m = best["best_model"]
        pred_t = m.forecast(max_h).to_numpy()
        pred = np.exp(pred_t) if LOG_SPACE else pred_t
        # residuals
        if LOG_SPACE:
            fitted_t = m.fittedvalues.reindex_like(best["y_train_level"]).fillna(method="bfill").to_numpy()
            resid_log = np.log(y_train_level.values) - fitted_t
            lo, hi = safe_conf_int_from_resid_log(pred, resid_log, 1.96)
        else:
            resid = y_train_level.values - m.fittedvalues.reindex_like(y_train_level).fillna(method="bfill").to_numpy()
            lo, hi = safe_conf_int_from_resid_level(pred, resid, 1.96)
        return pred, lo, hi

    if bt == "arima":
        m = best["best_model"]
        if hasattr(m, "predict_in_sample"):  # pmdarima
            pred_t, ci = m.predict(n_periods=max_h, return_conf_int=True, alpha=0.05)
            pred = np.exp(pred_t) if LOG_SPACE else pred_t
            lo_t, hi_t = ci[:,0], ci[:,1]
            lo = np.exp(lo_t) if LOG_SPACE else lo_t
            hi = np.exp(hi_t) if LOG_SPACE else hi_t
            return pred, lo, hi

    if bt == "prophet":
        m = best["best_model"]
        future = m.make_future_dataframe(periods=max_h, freq="D")
        fc = m.predict(future).iloc[-max_h:]
        yhat_t = fc["yhat"].to_numpy()
        pred = np.exp(yhat_t) if LOG_SPACE else yhat_t
        # Prophet gives 80% by default; we asked 95% via interval_width
        lo_t = fc["yhat_lower"].to_numpy(); hi_t = fc["yhat_upper"].to_numpy()
        lo = np.exp(lo_t) if LOG_SPACE else lo_t
        hi = np.exp(hi_t) if LOG_SPACE else hi_t
        return pred, lo, hi

    if bt == "lstm":
        model, mu, sd = best["best_model"]
        s = y_train_level.values.astype(float)
        s_norm = (s - mu) / sd
        last_win = s_norm[-DEEP_LAG:]
        pred = lstm_forecast(model, last_win, max_h)
        # no CI for LSTM
        lo = np.full(max_h, np.nan)
        hi = np.full(max_h, np.nan)
        return pred, lo, hi

    # naive fallback
    last = float(y_train_level.iloc[-1])
    pred = np.repeat(last, max_h)
    lo = np.full(max_h, np.nan)
    hi = np.full(max_h, np.nan)
    return pred, lo, hi

# ---------- main: trains, logs, writes multi-horizon ----------
def main():
    symbols = get_symbols()
    inserted = 0
    with get_conn() as conn, conn.cursor() as cur:
        for sym in symbols:
            try:
                df = load_series(cur, sym, min_rows=100)
                if df is None or df.empty:
                    print(f"[{sym}] not enough or invalid daily history; skipping.")
                    continue

                cand, best = train_candidates(df)
                last_ds = df.index[-1].date()
                ds_next = last_ds + timedelta(days=1)
                max_h = max(h for _, h in HORIZONS)
                run_id = uuid.uuid4()

                # 1) training run
                cur.execute("""
                    insert into model_training_runs
                      (run_id, symbol, ds_next, ds_train_end, holdout_len, n_train,
                       naive_mae, best_method, best_mae, vs_naive_improve)
                    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    run_id, sym, ds_next, last_ds, best["holdout_len"], best["n_train"],
                    best["naive_mae"], best["method"], best["holdout_mae"], best["vs_naive_improve"]
                ))

                # 2) candidates
                for c in cand:
                    params = None
                    if c.get("type") == "hw":
                        params = {"trend":"add","seasonal":"add","seasonal_periods":7, "log_space": LOG_SPACE}
                    elif c.get("type") == "sarimax":
                        params = {"order":[1,1,1], "seasonal_order":[0,1,1,7], "log_space": LOG_SPACE}
                    elif c.get("type") == "arima":
                        params = {"auto": True, "m":7, "log_space": LOG_SPACE}
                    elif c.get("type") == "prophet":
                        params = {"weekly":True, "yearly":True, "interval":"95%", "log_space": LOG_SPACE}
                    elif c.get("type") == "lstm":
                        params = {"lag": DEEP_LAG, "epochs": DEEP_EPOCHS}
                    cur.execute("""
                        insert into model_training_candidates (run_id, method, mae, params, notes)
                        values (%s,%s,%s,%s,%s)
                    """, (
                        run_id, c["method"],
                        None if not math.isfinite(c.get("mae", math.inf)) else float(c["mae"]),
                        json.dumps(params) if params is not None else None,
                        c.get("notes")
                    ))

                # 3) forecasts
                pred, lo95, hi95 = multi_horizon_forecast(best, max_h)
                for label, h in HORIZONS:
                    tgt = last_ds + timedelta(days=h)
                    fc = float(pred[h-1])
                    cur.execute("""
                        insert into model_forecasts (run_id, symbol, ds, forecast_close, forecast_method)
                        values (%s,%s,%s,%s,%s)
                        on conflict (run_id, ds) do update
                          set forecast_close = excluded.forecast_close,
                              forecast_method = excluded.forecast_method
                    """, (run_id, sym, tgt, fc, best["method"]))

                # 4) insight
                def fmt(x): return f"${x:,.2f}"
                one_d = float(pred[0]); one_w = float(pred[7-1]); one_m = float(pred[30-1])
                headline = f"{sym} {best['method']} forecasts — 1d: {fmt(one_d)}, 1w: {fmt(one_w)}, 1m: {fmt(one_m)}"
                details_bits = []
                if best.get("holdout_mae") is not None:
                    details_bits.append(f"MAE={best['holdout_mae']:.2f}")
                if best.get("holdout_mae_pct") is not None:
                    details_bits.append(f"MAE%={best['holdout_mae_pct']:.4f}")
                if best["vs_naive_improve"] is not None:
                    details_bits.append(f"vs naive: {best['vs_naive_improve']:+.0%}")
                if not np.isnan(lo95[0]):
                    details_bits.append(f"1d 95% CI [{fmt(lo95[0])}, {fmt(hi95[0])}]")
                details_bits.append("Long-horizon forecasts are highly uncertain.")
                details = "; ".join(details_bits)

                cur.execute("""
                    insert into insights (run_id, symbol, period, headline, details)
                    values (%s, %s, 'daily', %s, %s)
                """, (run_id, sym, headline, details))

                inserted += 1
                print(f"[{sym}] {best['method']} | run_id={run_id} | wrote {len(HORIZONS)} horizons")

            except Exception as e:
                # keep the pipeline moving
                print(f"[{sym}] generate_insight error: {e!r}; skipping.")
                continue

        print(f"Inserted {inserted} runs")

if __name__ == "__main__":
    main()
