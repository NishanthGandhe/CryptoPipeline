# pipeline/ingest.py

import os
import json
from datetime import datetime, timezone
import requests
import psycopg
from dotenv import load_dotenv

# Load env vars from .env file
load_dotenv()
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")


# ---- Data sources (no API key) ----
HEADERS = {"User-Agent": "cryptopipe/0.1"}
STEP_SEC = 3600  # 1h candles

def fetch_bitstamp(limit: int = 300):
    """Fetches hourly BTCUSD data from Bitstamp."""
    url = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
    params = {"step": STEP_SEC, "limit": limit}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for d in data.get("data", {}).get("ohlc", []):
        t = int(d["timestamp"])
        rows.append({
            "o": d["open"], "h": d["high"], "l": d["low"], "c": d["close"], "v": d["volume"],
            "openTime": t * 1000,
            "closeTime": (t + STEP_SEC) * 1000
        })
    return rows

def fetch_kraken(limit: int = 300):
    """Fetches hourly BTCUSD data from Kraken."""
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": "XBTUSD", "interval": 60}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    arr = data["result"]["XBTUSD"]
    if limit:
        arr = arr[-limit:]
    rows = []
    for k in arr:
        t = int(k[0])
        rows.append({
            "o": str(k[1]), "h": str(k[2]), "l": str(k[3]), "c": str(k[4]), "v": str(k[6]),
            "openTime": t * 1000,
            "closeTime": (t + STEP_SEC) * 1000
        })
    return rows

def fetch_candles(limit: int = 300):
    """Tries Bitstamp, falls back to Kraken."""
    try:
        rows = fetch_bitstamp(limit)
        if rows:
            return rows, "BTC-USD"
    except Exception as e:
        print(f"Bitstamp failed: {repr(e)}. Falling back to Kraken.")
    
    rows = fetch_kraken(limit)
    return rows, "BTC-USD"

# ---- DB helpers ----
def to_rows(normalized_rows, symbol_std):
    """Converts normalized data to DB insert format."""
    rows = []
    for p in normalized_rows:
        ts = datetime.fromtimestamp(p["openTime"] / 1000, tz=timezone.utc)
        rows.append(("crypto", symbol_std, ts, json.dumps(p)))
    return rows

def upsert_bronze(conn, rows):
    """Inserts data into bronze_ticks, ignoring duplicates."""
    if not rows:
        return 0
    sql = """
        INSERT INTO bronze_ticks (source, symbol, ts, payload)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (symbol, ts) DO NOTHING
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)

# ---- Main Execution ----
def main():
    """Main function to run the ingestion pipeline."""
    print("Fetching candle data...")
    candles, symbol_std = fetch_candles(limit=300)
    if not candles:
        raise RuntimeError("Could not fetch candle data from any provider.")
    
    print(f"Fetched {len(candles)} candles.")
    rows = to_rows(candles, symbol_std)

    # **THE FIX IS HERE**: Connect using separate parameters
    conn_info = f"host={HOST} dbname={DBNAME} user={USER} password=... sslmode=require"
    print(f"Connecting to database: {conn_info}")
    
    print(f"DEBUG: Host variable is exactly: {repr(HOST)}")

    with psycopg.connect(user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME) as conn:
        print("Connection successful. Inserting data...")
        inserted_count = upsert_bronze(conn, rows)
    
    print(f"Successfully inserted (or ignored) {inserted_count} rows into bronze_ticks for {symbol_std}.")

if __name__ == "__main__":
    main()