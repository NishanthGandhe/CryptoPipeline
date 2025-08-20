import os
import json
import time
import math
import uuid
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

# New imports for multivariate modeling
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import ExponentialSmoothing

# --- Configuration ---
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

SYMBOLS = os.getenv("SYMBOLS", "BTC-USD").split(",")
TRAIN_WINDOW_DAYS = int(os.getenv("TRAIN_WINDOW_DAYS", 365))

# XGBoost specific configuration
LOOK_BACK_DAYS = 30
HORIZONS = [1, 7, 30, 90, 180, 365, 1825, 3650]

# --- Database Connection ---
def get_conn():
    conn_str = f"user={DB_USER} password={DB_PASS} host={DB_HOST} port={DB_PORT} dbname={DB_NAME}"
    return psycopg.connect(conn_str, row_factory=dict_row, autocommit=True)

# --- Data Loading & Preparation ---
def load_series(conn, symbol):
    sql = "SELECT ds, close, trade_volume_usd, total_supply FROM public.gold_daily_metrics WHERE symbol = %s ORDER BY ds"
    with conn.cursor() as cur:
        cur.execute(sql, (symbol,))
        rows = cur.fetchall()
    
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['ds'] = pd.to_datetime(df['ds'])
    
    numeric_cols = ['close', 'trade_volume_usd', 'total_supply']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.set_index('ds')
    
    if TRAIN_WINDOW_DAYS > 0:
        df = df.last(f'{TRAIN_WINDOW_DAYS}D')
        
    return df

# --- Multivariate Model (XGBoost) ---
def prepare_multivariate_data(df):
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    return scaled_df, scaler

def create_supervised_dataset(df, look_back):
    X, y = [], []
    for i in range(len(df) - look_back - 1):
        X.append(df.iloc[i:(i + look_back)].values.flatten())
        y.append(df.iloc[i + look_back]['close'])
    return np.array(X), np.array(y)

# --- Univariate Model (Holt-Winters) ---
def train_univariate_model(y_train):
    """Trains a simple Holt-Winters model on a price series."""
    model = ExponentialSmoothing(
        y_train, trend="add", seasonal="add", seasonal_periods=7,
        initialization_method="estimated",
    ).fit()
    return model

# --- Main Training & Forecasting Logic ---
def generate_forecasts(model, method, last_data, scaler=None):
    """Generates multi-horizon forecasts based on the winning model."""
    forecasts = {}
    
    if method == "xgboost":
        last_window = last_data.values.flatten()
        last_window = np.array([last_window])
        next_day_pred_scaled = model.predict(last_window)
        
        dummy_array = np.zeros((1, len(last_data.columns)))
        dummy_array[:, 0] = next_day_pred_scaled
        next_day_pred_unscaled = scaler.inverse_transform(dummy_array)[0, 0]
    
    elif method == "holtwinters":
        # Forecast 1 step ahead
        next_day_pred_unscaled = model.forecast(1)[-1]
        
    else: # Naive
        next_day_pred_unscaled = last_data.iloc[-1]

    # Extrapolate for all horizons
    start_date = last_data.index[-1].date()
    for h in HORIZONS:
        forecast_date = start_date + timedelta(days=h)
        forecasts[forecast_date] = float(next_day_pred_unscaled)
        
    return forecasts

# --- DB Operations ---
def write_results_to_db(conn, symbol, best_method, mae, naive_mae, n_train, holdout_len, forecasts, train_end_ds):
    run_id = uuid.uuid4()
    ds_next = train_end_ds + timedelta(days=1)
    vs_naive_improve = (naive_mae - mae) / naive_mae if naive_mae > 0 else 0.0

    print(f"[{symbol}] Writing results with run_id: {run_id}")

    try:
        with conn.cursor() as cur:
            # Insert into model_training_runs
            cur.execute("""
                INSERT INTO public.model_training_runs 
                (run_id, symbol, ds_next, ds_train_end, holdout_len, n_train, naive_mae, best_method, best_mae, vs_naive_improve)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, symbol, ds_next, train_end_ds, holdout_len, n_train, naive_mae, best_method, mae, vs_naive_improve))

            # Insert into model_forecasts
            forecast_data = [(run_id, symbol, dt, val, best_method) for dt, val in forecasts.items()]
            cur.executemany("""
                INSERT INTO public.model_forecasts (run_id, symbol, ds, forecast_close, forecast_method)
                VALUES (%s, %s, %s, %s, %s)
            """, forecast_data)

            # Insert into insights
            headline = f"{symbol} forecast using {best_method}: Next day at ~${forecasts[ds_next]:,.0f}"
            details = f"MAE: {mae:.2f} (vs Naive: {vs_naive_improve:.1%})"
            cur.execute("""
                INSERT INTO public.insights (run_id, symbol, period, headline, details)
                VALUES (%s, %s, %s, %s, %s)
            """, (run_id, symbol, 'daily', headline, details))
            
        print(f"[{symbol}] Successfully wrote results to database.")

    except Exception as e:
        print(f"[{symbol}] Database write failed: {e}")

# --- Main Orchestration ---
def main():
    with get_conn() as conn:
        for symbol in SYMBOLS:
            print(f"\n--- Processing {symbol} ---")
            df = load_series(conn, symbol)
            
            # Check if there's enough data to model at all
            if df.empty or len(df.dropna(subset=['close'])) < 60:
                print(f"Not enough price data for {symbol}. Skipping.")
                continue

            # --- DECISION POINT: Use XGBoost or fall back to simpler model ---
            is_multivariate = df['trade_volume_usd'].notna().any()

            if is_multivariate:
                print(f"[{symbol}] Rich dataset found. Using Multivariate (XGBoost) model.")
                features_df = df[['close', 'trade_volume_usd', 'total_supply']]
                scaled_df, scaler = prepare_multivariate_data(features_df.copy())
                
                if len(scaled_df) < LOOK_BACK_DAYS + 30:
                    print(f"Not enough clean data for {symbol} after prep. Skipping.")
                    continue
                
                X, y = create_supervised_dataset(scaled_df, look_back=LOOK_BACK_DAYS)
                holdout_len = max(14, len(X) // 10)
                X_train, X_test = X[:-holdout_len], X[-holdout_len:]
                y_train, y_test = y[:-holdout_len], y[-holdout_len:]
                y_test_unscaled = df['close'].values[-(holdout_len):]
                
                # Naive model
                naive_pred_unscaled = df['close'].iloc[-holdout_len-1]
                naive_mae = np.mean(np.abs(y_test_unscaled - naive_pred_unscaled))
                
                # XGBoost model
                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                xgb_preds_scaled = model.predict(X_test)
                dummy_array = np.zeros((len(xgb_preds_scaled), scaled_df.shape[1]))
                dummy_array[:, 0] = xgb_preds_scaled
                xgb_preds_unscaled = scaler.inverse_transform(dummy_array)[:, 0]
                xgb_mae = np.mean(np.abs(y_test_unscaled - xgb_preds_unscaled))
                
                # Select winner and generate forecasts
                if xgb_mae < naive_mae:
                    best_method = "xgboost"
                    mae = xgb_mae
                    forecasts = generate_forecasts(model, 'xgboost', scaled_df.iloc[-LOOK_BACK_DAYS:], scaler)
                else:
                    best_method = "naive"
                    mae = naive_mae
                    forecasts = generate_forecasts(None, 'naive', df['close'])

                write_results_to_db(conn, symbol, best_method, mae, naive_mae, len(X_train), holdout_len, forecasts, df.index[-1].date())

            else:
                print(f"[{symbol}] Price data only. Using Univariate (Holt-Winters) model.")
                price_series = df['close'].dropna()
                holdout_len = max(14, len(price_series) // 10)
                y_train, y_test = price_series[:-holdout_len], price_series[-holdout_len:]
                
                # Naive model
                naive_pred = y_train.iloc[-1]
                naive_mae = np.mean(np.abs(y_test - naive_pred))

                # Holt-Winters model
                try:
                    hw_model = train_univariate_model(y_train)
                    hw_preds = hw_model.forecast(holdout_len)
                    hw_mae = np.mean(np.abs(y_test - hw_preds))

                    if hw_mae < naive_mae:
                        best_method = "holtwinters"
                        mae = hw_mae
                        forecasts = generate_forecasts(hw_model, 'holtwinters', y_train)
                    else:
                        best_method = "naive"
                        mae = naive_mae
                        forecasts = generate_forecasts(None, 'naive', y_train)
                    
                    write_results_to_db(conn, symbol, best_method, mae, naive_mae, len(y_train), holdout_len, forecasts, y_train.index[-1].date())

                except Exception as e:
                    print(f"[{symbol}] Holt-Winters model failed: {e}. Skipping.")


if __name__ == "__main__":
    main()