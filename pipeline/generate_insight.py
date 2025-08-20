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

# Use your shorter, preferred horizons
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC-USD").split(",")]
TRAIN_WINDOW_DAYS = int(os.getenv("TRAIN_WINDOW_DAYS", 1095))  # 3 years instead of 1
LOOK_BACK_DAYS = 14  # Reduced to 14 for more responsive, trend-following predictions
HORIZONS = list(range(1, 31))  # Generate forecasts for days 1-30

# --- Database Connection ---
def get_conn():
    conn_str = f"user={DB_USER} password={DB_PASS} host={DB_HOST} port={DB_PORT} dbname={DB_NAME}"
    return psycopg.connect(conn_str, row_factory=dict_row, autocommit=True)

# --- Data Loading & Preparation (FINAL, ROBUST VERSION) ---
def load_and_merge_data(conn, symbol):
    """
    Loads price and metric data using psycopg and merges them in pandas.
    This avoids the pd.read_sql header issue.
    """
    # 1. Load daily prices
    price_sql = "SELECT ds, close FROM public.price_daily WHERE symbol = %s ORDER BY ds"
    with conn.cursor() as cur:
        cur.execute(price_sql, (symbol,))
        price_rows = cur.fetchall()
    
    if not price_rows:
        return pd.DataFrame()
    price_df = pd.DataFrame(price_rows)

    # 2. Load on-chain metrics
    metrics_sql = "SELECT ds, trade_volume_usd, total_supply FROM public.bronze_daily_metrics WHERE symbol = %s ORDER BY ds"
    with conn.cursor() as cur:
        cur.execute(metrics_sql, (symbol,))
        metrics_rows = cur.fetchall()
    
    # Create metrics_df only if data was returned
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
    else:
        metrics_df = pd.DataFrame(columns=['ds', 'trade_volume_usd', 'total_supply'])

    # 3. Convert date columns and merge
    price_df['ds'] = pd.to_datetime(price_df['ds'])
    if not metrics_df.empty:
        metrics_df['ds'] = pd.to_datetime(metrics_df['ds'])
        df = pd.merge(price_df, metrics_df, on='ds', how='left')
    else:
        df = price_df
        df['trade_volume_usd'] = np.nan
        df['total_supply'] = np.nan
    
    # 4. Final data type conversion and processing
    for col in ['close', 'trade_volume_usd', 'total_supply']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.set_index('ds')
    
    if TRAIN_WINDOW_DAYS > 0:
        df = df.last(f'{TRAIN_WINDOW_DAYS}D')
        
    return df

# (The rest of the functions: prepare_multivariate_data, create_supervised_dataset, train_univariate_model, generate_forecasts, and write_results_to_db remain IDENTICAL.)
def prepare_multivariate_data(df):
    """Enhanced but robust feature engineering"""
    original_df = df.copy()
    
    # Only add features if we have enough data
    if len(df) >= 30:
        # Basic technical indicators
        df['price_change'] = df['close'].pct_change().fillna(0)
        df['price_sma_7'] = df['close'].rolling(7, min_periods=1).mean()
        df['volume_sma_7'] = df['trade_volume_usd'].rolling(7, min_periods=1).mean()
        
        # Price momentum over different periods
        df['momentum_3d'] = df['close'].pct_change(3).fillna(0)
        df['momentum_7d'] = df['close'].pct_change(7).fillna(0)
        
        # Volatility
        df['volatility_7d'] = df['close'].rolling(7, min_periods=1).std().fillna(0)
        
        # Price ratios
        df['price_to_sma7'] = (df['close'] / df['price_sma_7']).fillna(1)
        
        # Volume indicators
        df['volume_change'] = df['trade_volume_usd'].pct_change().fillna(0)
        
        feature_columns = [
            'close', 'trade_volume_usd', 'total_supply',
            'price_change', 'momentum_3d', 'momentum_7d', 
            'volatility_7d', 'price_to_sma7', 'volume_change'
        ]
    else:
        # Use basic features only
        feature_columns = ['close', 'trade_volume_usd', 'total_supply']
    
    # Keep only available columns
    available_columns = [col for col in feature_columns if col in df.columns and not df[col].isna().all()]
    df = df[available_columns]
    
    # Clean the data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0)
    
    # Final check
    if len(df) == 0:
        print("Warning: Feature engineering failed, using original data")
        df = original_df
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    return scaled_df, scaler

def prepare_price_only_data(df):
    """Enhanced feature engineering for price-only data"""
    original_df = df.copy()
    
    # Only add features if we have enough data
    if len(df) >= 30:
        # Price momentum and trend features
        df['price_change'] = df['close'].pct_change().fillna(0)
        df['price_change_2d'] = df['close'].pct_change(2).fillna(0)
        df['momentum_3d'] = df['close'].pct_change(3).fillna(0)
        df['momentum_7d'] = df['close'].pct_change(7).fillna(0)
        df['momentum_14d'] = df['close'].pct_change(14).fillna(0)
        
        # Moving averages and trends
        df['price_sma_5'] = df['close'].rolling(5, min_periods=1).mean()
        df['price_sma_10'] = df['close'].rolling(10, min_periods=1).mean()
        df['price_sma_20'] = df['close'].rolling(20, min_periods=1).mean()
        
        # Price ratios (trend indicators)
        df['price_to_sma5'] = (df['close'] / df['price_sma_5']).fillna(1)
        df['price_to_sma10'] = (df['close'] / df['price_sma_10']).fillna(1)
        df['price_to_sma20'] = (df['close'] / df['price_sma_20']).fillna(1)
        df['sma5_to_sma20'] = (df['price_sma_5'] / df['price_sma_20']).fillna(1)
        
        # Volatility indicators
        df['volatility_5d'] = df['close'].rolling(5, min_periods=1).std().fillna(0)
        df['volatility_10d'] = df['close'].rolling(10, min_periods=1).std().fillna(0)
        df['volatility_ratio'] = (df['volatility_5d'] / (df['volatility_10d'] + 0.001)).fillna(1)
        
        # High/low analysis over different periods
        df['high_5d'] = df['close'].rolling(5, min_periods=1).max()
        df['low_5d'] = df['close'].rolling(5, min_periods=1).min()
        df['price_position_5d'] = ((df['close'] - df['low_5d']) / (df['high_5d'] - df['low_5d'] + 0.001)).fillna(0.5)
        
        feature_columns = [
            'close', 'price_change', 'price_change_2d', 'momentum_3d', 'momentum_7d', 'momentum_14d',
            'price_to_sma5', 'price_to_sma10', 'price_to_sma20', 'sma5_to_sma20',
            'volatility_5d', 'volatility_10d', 'volatility_ratio', 'price_position_5d'
        ]
    else:
        # Use basic features only
        feature_columns = ['close']
    
    # Keep only available columns
    available_columns = [col for col in feature_columns if col in df.columns and not df[col].isna().all()]
    df = df[available_columns]
    
    # Clean the data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0)
    
    # Final check
    if len(df) == 0:
        print("Warning: Feature engineering failed, using original data")
        df = original_df[['close']]
    
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

def train_univariate_model(y_train):
    """Enhanced Holt-Winters model with more aggressive trend following"""
    try:
        # Try triple exponential smoothing with optimized parameters
        model = ExponentialSmoothing(
            y_train, 
            trend="add", 
            seasonal="add", 
            seasonal_periods=7,
            initialization_method="estimated",
            damped_trend=False  # Allow stronger trend extrapolation
        ).fit(
            smoothing_level=0.3,    # More responsive to recent values
            smoothing_trend=0.2,    # More responsive to trends
            smoothing_seasonal=0.1  # Moderate seasonal adjustment
        )
        return model
    except:
        # Fallback to additive trend only
        model = ExponentialSmoothing(
            y_train, 
            trend="add", 
            seasonal=None,
            initialization_method="estimated",
            damped_trend=False
        ).fit(
            smoothing_level=0.3,
            smoothing_trend=0.2
        )
        return model

def generate_forecasts(model, method, last_data, scaler=None):
    forecasts = {}
    start_date = last_data.index[-1].date()
    
    if method == "xgboost":
        last_window = last_data.values
        max_horizon = max(HORIZONS)
        predictions_scaled = []
        
        for _ in range(max_horizon):
            next_step_pred_scaled = model.predict(np.array([last_window.flatten()]))[0]
            predictions_scaled.append(next_step_pred_scaled)
            next_step_features = np.zeros(last_window.shape[1])
            next_step_features[0] = next_step_pred_scaled
            if last_window.shape[1] > 1:
                next_step_features[1:] = last_window[-1, 1:]
            last_window = np.vstack([last_window[1:], next_step_features])

        dummy_array = np.zeros((len(predictions_scaled), last_data.shape[1]))
        dummy_array[:, 0] = predictions_scaled
        predictions_unscaled = scaler.inverse_transform(dummy_array)[:, 0]
        
        current_date = start_date
        for pred in predictions_unscaled:
            current_date += timedelta(days=1)
            forecasts[current_date] = float(pred)

    elif method == "holtwinters":
        # Generate actual Holt-Winters forecasts for each horizon
        hw_preds = model.forecast(max(HORIZONS))
        current_date = start_date
        for pred in hw_preds:
            current_date += timedelta(days=1)
            forecasts[current_date] = float(pred)
    
    else:  # Naive method
        last_known_price = last_data.iloc[-1]
        current_date = start_date
        for _ in HORIZONS:
            current_date += timedelta(days=1)
            forecasts[current_date] = float(last_known_price)

    final_forecasts = {}
    for h in HORIZONS:
        forecast_date = start_date + timedelta(days=h)
        if forecast_date in forecasts:
            final_forecasts[forecast_date] = forecasts[forecast_date]
            
    return final_forecasts

def write_results_to_db(conn, symbol, best_method, mae, naive_mae, n_train, holdout_len, forecasts, train_end_ds):
    run_id = uuid.uuid4()
    ds_next = train_end_ds + timedelta(days=1)
    vs_naive_improve = (naive_mae - mae) / naive_mae if naive_mae > 0 else 0.0

    print(f"[{symbol}] Writing results with run_id: {run_id}")

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO public.model_training_runs 
                (run_id, symbol, ds_next, ds_train_end, holdout_len, n_train, naive_mae, best_method, best_mae, vs_naive_improve)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, symbol, ds_next, train_end_ds, holdout_len, n_train, naive_mae, best_method, mae, vs_naive_improve))

            forecast_data = [(run_id, symbol, dt, val, best_method) for dt, val in forecasts.items()]
            cur.executemany("""
                INSERT INTO public.model_forecasts (run_id, symbol, ds, forecast_close, forecast_method)
                VALUES (%s, %s, %s, %s, %s)
            """, forecast_data)

            headline = f"{symbol} forecast using {best_method}: Next day at ~${forecasts[ds_next]:,.0f}"
            details = f"MAE: {mae:.2f} (vs Naive: {vs_naive_improve:.1%})"
            cur.execute("""
                INSERT INTO public.insights (run_id, symbol, period, headline, details)
                VALUES (%s, %s, %s, %s, %s)
            """, (run_id, symbol, 'daily', headline, details))
            
        print(f"[{symbol}] Successfully wrote results to database.")

    except Exception as e:
        print(f"[{symbol}] Database write failed: {e}")

def main():
    # Set your full list of symbols in the .env file
    # SYMBOLS=BTC-USD,ETH-USD,XRP-USD,...
    with get_conn() as conn:
        for symbol in SYMBOLS:
            print(f"\n--- Processing {symbol} ---")
            df = load_and_merge_data(conn, symbol)
            
            if df.empty or len(df.dropna(subset=['close'])) < 60:
                print(f"Not enough price data for {symbol}. Skipping.")
                continue

            is_multivariate = 'trade_volume_usd' in df.columns and df['trade_volume_usd'].notna().any()

            if is_multivariate:
                print(f"[{symbol}] Rich dataset found. Using Enhanced Multivariate (XGBoost) model.")
                # Start with all available columns and let feature engineering handle the rest
                all_cols = ['close', 'trade_volume_usd', 'total_supply']
                available_cols = [col for col in all_cols if col in df.columns]
                features_df = df[available_cols]
                scaled_df, scaler = prepare_multivariate_data(features_df.copy())
                
                if len(scaled_df) < LOOK_BACK_DAYS + 30:
                    print(f"Not enough clean data for {symbol} after prep. Skipping.")
                    continue
                
                X, y = create_supervised_dataset(scaled_df, look_back=LOOK_BACK_DAYS)
                holdout_len = max(14, len(X) // 10)
                X_train, X_test = X[:-holdout_len], X[-holdout_len:]
                y_train, y_test = y[:-holdout_len], y[-holdout_len:]
                y_test_unscaled = df['close'].values[-(holdout_len):]
                
                naive_pred_unscaled = df['close'].iloc[-holdout_len-1]
                naive_mae = np.mean(np.abs(y_test_unscaled - naive_pred_unscaled))
                
                # Trend-responsive XGBoost for bolder, more dynamic predictions
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=80,            # Fewer trees for faster, more responsive learning
                    learning_rate=0.15,         # Higher learning rate for trend responsiveness
                    max_depth=2,                # Shallow trees to avoid overfitting
                    min_child_weight=1,         # Allow aggressive splits
                    subsample=0.9,              # High subsample for stability
                    colsample_bytree=1.0,       # Use all features
                    reg_alpha=0.1,              # Light regularization for responsiveness
                    reg_lambda=0.5,             # Light regularization
                    random_state=42,
                    n_jobs=-1                   # Use all cores
                )
                model.fit(X_train, y_train)
                xgb_preds_scaled = model.predict(X_test)
                dummy_array = np.zeros((len(xgb_preds_scaled), scaled_df.shape[1]))
                dummy_array[:, 0] = xgb_preds_scaled
                xgb_preds_unscaled = scaler.inverse_transform(dummy_array)[:, 0]
                xgb_mae = np.mean(np.abs(y_test_unscaled - xgb_preds_unscaled))
                
                # More aggressive threshold for bolder predictions
                # Use XGBoost if it's within 35% of naive performance (prioritize trend-following over pure accuracy)
                xgb_threshold = naive_mae * 1.35
                
                if xgb_mae < xgb_threshold:
                    best_method = "xgboost"
                    mae = xgb_mae
                    forecasts = generate_forecasts(model, 'xgboost', scaled_df.iloc[-LOOK_BACK_DAYS:], scaler)
                else:
                    best_method = "naive"
                    mae = naive_mae
                    forecasts = generate_forecasts(None, 'naive', df['close'])

                write_results_to_db(conn, symbol, best_method, mae, naive_mae, len(X_train), holdout_len, forecasts, df.index[-1].date())

            else:
                print(f"[{symbol}] Price data only. Using Enhanced Price-Only (XGBoost) model.")
                
                # Check if we have enough data for XGBoost training
                if len(df) < LOOK_BACK_DAYS + 30:
                    print(f"[{symbol}] Not enough data for XGBoost. Need at least {LOOK_BACK_DAYS + 30} records, have {len(df)}. Falling back to Holt-Winters.")
                    
                    # Fallback to Holt-Winters for insufficient data
                    price_series = df['close'].dropna()
                    holdout_len = max(14, len(price_series) // 10)
                    y_train, y_test = price_series[:-holdout_len], price_series[-holdout_len:]
                    
                    naive_pred = y_train.iloc[-1]
                    naive_mae = np.mean(np.abs(y_test - naive_pred))

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
                    
                else:
                    # Use XGBoost with enhanced price-only features
                    try:
                        scaled_df, scaler = prepare_price_only_data(df)
                        
                        X, y = create_supervised_dataset(scaled_df, look_back=LOOK_BACK_DAYS)
                        
                        holdout_len = max(14, len(X) // 10)
                        X_train, X_test = X[:-holdout_len], X[-holdout_len:]
                        y_train, y_test = y[:-holdout_len], y[-holdout_len:]
                        
                        # Calculate naive baseline (unscaled)
                        y_test_unscaled = []
                        for i in range(len(y_test)):
                            dummy_array = np.zeros((1, scaled_df.shape[1]))
                            dummy_array[0, 0] = y_test[i]
                            unscaled = scaler.inverse_transform(dummy_array)[0, 0]
                            y_test_unscaled.append(unscaled)
                        
                        naive_pred_unscaled = []
                        for i in range(len(y_test)):
                            dummy_array = np.zeros((1, scaled_df.shape[1]))
                            dummy_array[0, 0] = y_train[-1]
                            unscaled = scaler.inverse_transform(dummy_array)[0, 0]
                            naive_pred_unscaled.append(unscaled)
                        
                        naive_mae = np.mean(np.abs(np.array(y_test_unscaled) - np.array(naive_pred_unscaled)))
                        
                        # Train XGBoost model (price-only optimized)
                        model = xgb.XGBRegressor(
                            objective='reg:squarederror',
                            n_estimators=100,           # Slightly more trees for price-only data
                            learning_rate=0.12,         # Balanced learning rate
                            max_depth=3,                # Allow some complexity for price patterns
                            min_child_weight=2,         # Moderate constraint
                            subsample=0.9,              # High subsample for stability
                            colsample_bytree=0.95,      # Use most features
                            reg_alpha=0.2,              # Light regularization
                            reg_lambda=1.0,             # Moderate regularization
                            random_state=42,
                            n_jobs=-1
                        )
                        
                        model.fit(X_train, y_train)
                        xgb_preds_scaled = model.predict(X_test)
                        
                        # Unscale XGBoost predictions
                        xgb_preds_unscaled = []
                        for pred in xgb_preds_scaled:
                            dummy_array = np.zeros((1, scaled_df.shape[1]))
                            dummy_array[0, 0] = pred
                            unscaled = scaler.inverse_transform(dummy_array)[0, 0]
                            xgb_preds_unscaled.append(unscaled)
                        
                        xgb_mae = np.mean(np.abs(np.array(y_test_unscaled) - np.array(xgb_preds_unscaled)))
                        
                        # More aggressive threshold for price-only XGBoost (40% tolerance)
                        xgb_threshold = naive_mae * 1.40
                        
                        if xgb_mae < xgb_threshold:
                            best_method = "xgboost"
                            mae = xgb_mae
                            forecasts = generate_forecasts(model, 'xgboost', scaled_df.iloc[-LOOK_BACK_DAYS:], scaler)
                        else:
                            best_method = "naive"
                            mae = naive_mae
                            forecasts = generate_forecasts(None, 'naive', df['close'])

                        write_results_to_db(conn, symbol, best_method, mae, naive_mae, len(X_train), holdout_len, forecasts, df.index[-1].date())
                        
                    except Exception as e:
                        print(f"[{symbol}] XGBoost (price-only) model failed: {e}. Falling back to Holt-Winters.")
                        
                        # Final fallback to Holt-Winters
                        price_series = df['close'].dropna()
                        holdout_len = max(14, len(price_series) // 10)
                        y_train, y_test = price_series[:-holdout_len], price_series[-holdout_len:]
                        
                        naive_pred = y_train.iloc[-1]
                        naive_mae = np.mean(np.abs(y_test - naive_pred))

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
                            print(f"[{symbol}] All models failed: {e}. Skipping.")

if __name__ == "__main__":
    main()