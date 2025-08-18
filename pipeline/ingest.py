import os, json,time
from datetime import datetime, timezone
import requests
import psycopg
from dotenv import load_dotenv

load_dotenv()

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

# --- Providers (no API key) ---
HEADERS = {"User-Agent": "cryptopipe/0.1 (+https://github.com/you/cryptopipe)"}
STEP_SEC = 3600
BITSTAMP_MAX = 1000     
KRAKEN_MAX = 720         

def bitstamp_pair(symbol_std: str) -> str:
    return symbol_std.replace("-", "").lower()

def kraken_pair(symbol_std: str) -> str:
    base, quote = symbol_std.split("-")
    base = {"BTC": "XBT"}.get(base.upper(), base.upper())
    return f"{base}{quote.upper()}"

def fetch_bitstamp_page(symbol_std: str, end_ts: int, limit: int = BITSTAMP_MAX):
    """Fetch up to `limit` candles ending at `end_ts` (unix seconds)."""
    url = f"https://www.bitstamp.net/api/v2/ohlc/{bitstamp_pair(symbol_std)}/"
    params = {
        "step": STEP_SEC,
        "limit": min(limit, BITSTAMP_MAX),
        "end": int(end_ts),
        "exclude_current_candle": "true",
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    js = r.json()
    rows = []
    for d in js.get("data", {}).get("ohlc", []):
        t = int(d["timestamp"])  # seconds
        rows.append({
            "o": d["open"], "h": d["high"], "l": d["low"], "c": d["close"], "v": d["volume"],
            "openTime": t * 1000,
            "closeTime": (t + STEP_SEC) * 1000
        })
    # Bitstamp returns ascending by timestamp; ensure sorted just in case
    rows.sort(key=lambda p: p["openTime"])
    return rows

def backfill_bitstamp(symbol_std: str, days: int, conn):
    """Backfill `days` of history using Bitstamp in pages of <=1000 candles."""
    if days <= 0:
        return 0
    now_sec = int(time.time())
    earliest_needed = now_sec - days * 86400
    end = now_sec
    total = 0
    print(f"[{symbol_std}] Backfilling ~{days} days from Bitstamp…")
    while True:
        page = fetch_bitstamp_page(symbol_std, end_ts=end, limit=BITSTAMP_MAX)
        if not page:
            break
        rows = normalize_to_bronze_rows(symbol_std, page)
        inserted = upsert_bronze(conn, rows)
        total += inserted
        earliest_in_page = int(page[0]["openTime"] // 1000)
        # move end just before earliest candle to page older data
        end = earliest_in_page - 1
        if earliest_in_page <= earliest_needed:
            break
    print(f"[{symbol_std}] Backfill inserted (or deduped) {total} rows.")
    return total

# ------------ Kraken fetch (kept as recent fallback only) ------------
def fetch_kraken(symbol_std: str, limit: int = 300):
    url = "https://api.kraken.com/0/public/OHLC"
    pair = kraken_pair(symbol_std)
    r = requests.get(url, params={"pair": pair, "interval": 60}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    arr = data["result"][pair]  # [time, open, high, low, close, vwap, volume, count]
    if limit:
        arr = arr[-min(limit, KRAKEN_MAX):]
    rows = []
    for k in arr:
        t = int(k[0])
        rows.append({
            "o": str(k[1]), "h": str(k[2]), "l": str(k[3]), "c": str(k[4]), "v": str(k[6]),
            "openTime": t * 1000,
            "closeTime": (t + STEP_SEC) * 1000
        })
    rows.sort(key=lambda p: p["openTime"])
    return rows

# ------------ Normalization + DB ------------
def normalize_to_bronze_rows(symbol_std: str, payload_rows):
    out = []
    for p in payload_rows:
        ts = datetime.fromtimestamp(p["openTime"]/1000, tz=timezone.utc)
        out.append(("crypto", symbol_std, ts, json.dumps(p)))
    return out

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

# ------------ Main orchestration ------------
def main():
    symbols = get_symbols()
    backfill_days = int(os.getenv("BACKFILL_DAYS", "0"))
    print("Symbols:", symbols, "| BACKFILL_DAYS:", backfill_days)

    with get_conn() as conn:
        # 1) Optional backfill per symbol (Bitstamp only)
        if backfill_days > 0:
            for sym in symbols:
                backfill_bitstamp(sym, backfill_days, conn)

        # 2) Always do a recent top-up (Bitstamp → Kraken fallback)
        for sym in symbols:
            try:
                page = fetch_bitstamp_page(sym, end_ts=int(time.time()), limit=300)
                source = "Bitstamp"
            except Exception as e:
                print(f"[{sym}] Bitstamp failed, falling back to Kraken:", repr(e))
                page = fetch_kraken(sym, limit=300)
                source = "Kraken"
            print(f"[{sym}] {source} recent fetched {len(page)} candles.")
            rows = normalize_to_bronze_rows(sym, page)
            inserted = upsert_bronze(conn, rows)
            print(f"[{sym}] inserted (or deduped) {inserted} rows.")

    print("Ingest complete.")

if __name__ == "__main__":
    main()