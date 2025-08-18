import os, json
from datetime import datetime, timezone
import requests
import psycopg
from dotenv import load_dotenv

load_dotenv()

# --- DB connection (works with DSN or DB_* envs) ---
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

# --- Symbols to ingest ---
def get_symbols():
    raw = os.getenv("SYMBOLS", "BTC-USD")
    return [s.strip() for s in raw.split(",") if s.strip()]

# --- Providers (no API key) ---
HEADERS = {"User-Agent": "cryptopipe/0.1 (+https://github.com/you/cryptopipe)"}
STEP_SEC = 3600  # 1h candles

def bitstamp_pair(symbol_std: str) -> str:
    # e.g., "BTC-USD" -> "btcusd", "ETH-USD" -> "ethusd"
    return symbol_std.replace("-", "").lower()

def kraken_pair(symbol_std: str) -> str:
    # Kraken pair codes are quirky: BTC-USD -> XBTUSD, ETH-USD -> ETHUSD
    base, quote = symbol_std.split("-")
    base = {"BTC": "XBT"}.get(base.upper(), base.upper())
    return f"{base}{quote.upper()}"

def fetch_bitstamp(symbol_std: str, limit: int = 300):
    url = f"https://www.bitstamp.net/api/v2/ohlc/{bitstamp_pair(symbol_std)}/"
    r = requests.get(url, params={"step": STEP_SEC, "limit": limit}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for d in data.get("data", {}).get("ohlc", []):
        t = int(d["timestamp"])  # seconds
        rows.append({
            "o": d["open"], "h": d["high"], "l": d["low"], "c": d["close"], "v": d["volume"],
            "openTime": t * 1000,
            "closeTime": (t + STEP_SEC) * 1000
        })
    return rows

def fetch_kraken(symbol_std: str, limit: int = 300):
    url = "https://api.kraken.com/0/public/OHLC"
    pair = kraken_pair(symbol_std)
    r = requests.get(url, params={"pair": pair, "interval": 60}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Kraken returns result under the pair key provided (after normalization)
    arr = data["result"][pair]
    if limit:
        arr = arr[-limit:]
    rows = []
    for k in arr:
        t = int(k[0])  # seconds
        rows.append({
            "o": str(k[1]), "h": str(k[2]), "l": str(k[3]), "c": str(k[4]), "v": str(k[6]),
            "openTime": t * 1000,
            "closeTime": (t + STEP_SEC) * 1000
        })
    return rows

def fetch_candles(symbol_std: str, limit: int = 300):
    # try Bitstamp first, fallback to Kraken
    try:
        rows = fetch_bitstamp(symbol_std, limit)
        if rows:
            return rows
    except Exception as e:
        print(f"[{symbol_std}] Bitstamp failed:", repr(e))
    return fetch_kraken(symbol_std, limit)

def normalize_to_bronze_rows(symbol_std: str, payload_rows):
    rows = []
    for p in payload_rows:
        ts = datetime.fromtimestamp(p["openTime"]/1000, tz=timezone.utc)
        rows.append(("crypto", symbol_std, ts, json.dumps(p)))
    return rows

def upsert_bronze(conn, rows):
    if not rows:
        return 0
    sql = """
        insert into bronze_ticks (source, symbol, ts, payload)
        values (%s, %s, %s, %s)
        on conflict (symbol, ts) do nothing
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)

def main():
    symbols = get_symbols()
    print("Symbols to ingest:", symbols)
    total = 0
    with get_conn() as conn:
        for sym in symbols:
            print(f"Fetching {sym} …")
            candles = fetch_candles(sym, limit=300)
            print(f"{sym}: fetched {len(candles)} candles.")
            rows = normalize_to_bronze_rows(sym, candles)
            inserted = upsert_bronze(conn, rows)
            print(f"{sym}: inserted (or deduped) {inserted} rows into bronze_ticks.")
            total += inserted
    print(f"Done. Total rows processed: {total}")

if __name__ == "__main__":
    main()
