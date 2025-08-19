import os, json, time
from datetime import datetime, timezone
import requests
from requests import HTTPError
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
    # If your Supabase requires SSL, uncomment sslmode="require"
    return psycopg.connect(host=host, dbname=db, user=user, password=pwd, port=port)  # , sslmode="require"

def get_symbols():
    raw = os.getenv("SYMBOLS", "BTC-USD")
    return [s.strip() for s in raw.split(",") if s.strip()]

# --- Providers (no API key) ---
HEADERS = {"User-Agent": "cryptopipe/0.1 (+https://github.com/you/cryptopipe)"}
STEP_SEC = 3600
BITSTAMP_MAX = 1000
KRAKEN_MAX = 720

def bitstamp_pair(symbol_std: str) -> str:
    # "BTC-USD" -> "btcusd"
    return symbol_std.replace("-", "").lower()

def kraken_pair(symbol_std: str) -> str:
    base, quote = symbol_std.split("-")
    base = {"BTC": "XBT"}.get(base.upper(), base.upper())
    return f"{base}{quote.upper()}"

def fetch_bitstamp_page(
    symbol_code: str,  # e.g. "btcusd"
    end_ts: int,
    limit: int = BITSTAMP_MAX,
    session: requests.Session | None = None,
):
    """
    Returns:
      - list[dict] with keys: openTime, closeTime, open, high, low, close, volume
      - [] if no more data
      - None if 404 (pair not listed)
    Raises HTTPError for other non-2xx.
    """
    s = session or requests.Session()
    url = f"https://www.bitstamp.net/api/v2/ohlc/{symbol_code}/"
    params = {
        "step": STEP_SEC,
        "limit": limit,
        "end": end_ts,
        "exclude_current_candle": "true",
    }
    r = s.get(url, params=params, timeout=20)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    j = r.json()
    ohlc = (j or {}).get("data", {}).get("ohlc", []) or []
    out = []
    for c in ohlc:
        ts = int(c["timestamp"])
        out.append({
            "openTime": ts * 1000,
            "closeTime": (ts + STEP_SEC) * 1000,
            "open": c["open"],
            "high": c["high"],
            "low":  c["low"],
            "close": c["close"],
            "volume": c.get("volume", "0"),
        })
    return out

def backfill_bitstamp(symbol: str, backfill_days: int, conn):
    code = bitstamp_pair(symbol)
    now_sec = int(time.time())
    earliest_needed = now_sec - backfill_days * 86400
    end = now_sec

    print(f"[{symbol}] Backfilling ~{backfill_days} days from Bitstamp…")

    s = requests.Session()
    total = 0
    retries = 0

    while True:
        try:
            page = fetch_bitstamp_page(code, end_ts=end, limit=BITSTAMP_MAX, session=s)
        except HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status in (429, 500, 502, 503, 504) and retries < 5:
                wait = 1.5 * (retries + 1)
                print(f"[{symbol}] Transient HTTP {status}; retrying in {wait:.1f}s…")
                time.sleep(wait)
                retries += 1
                continue
            print(f"[{symbol}] HTTP error: {e}. Skipping the rest of this symbol.")
            return
        retries = 0

        if page is None:
            print(f"[{symbol}] Bitstamp pair '{code}' not found (404). Skipping.")
            return
        if not page:
            break

        # normalize to bronze rows and insert
        rows = normalize_to_bronze_rows(symbol, page)
        inserted = upsert_bronze(conn, rows)
        total += inserted

        earliest_in_page = min(int(p["openTime"] // 1000) for p in page)
        end = earliest_in_page - 1
        if earliest_in_page <= earliest_needed:
            break

        time.sleep(0.25)  # be polite

    print(f"[{symbol}] Backfill inserted (or deduped) {total} rows.")

# ------------ Kraken fetch (fallback) ------------
def fetch_kraken(symbol_std: str, limit: int = 300):
    """
    Returns payload-shaped rows like Bitstamp:
    openTime, closeTime, open, high, low, close, volume
    """
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
            "openTime": t * 1000,
            "closeTime": (t + STEP_SEC) * 1000,
            "open": str(k[1]),
            "high": str(k[2]),
            "low":  str(k[3]),
            "close": str(k[4]),
            "volume": str(k[6]),
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
    conn.commit()
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
                try:
                    backfill_bitstamp(sym, backfill_days, conn)
                except Exception as e:
                    print(f"[{sym}] Unhandled backfill error: {e!r}. Skipping.")

        # 2) Recent top-up (Bitstamp → Kraken fallback)
        for sym in symbols:
            try:
                page = fetch_bitstamp_page(bitstamp_pair(sym), end_ts=int(time.time()), limit=300)
                if page is None:
                    print(f"[{sym}] Bitstamp pair not found (404). Skipping this symbol.")
                    continue
                source = "Bitstamp"
            except HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status == 404:
                    print(f"[{sym}] Bitstamp pair not found (404). Skipping this symbol.")
                    continue
                print(f"[{sym}] Bitstamp HTTP {status}; falling back to Kraken…")
                page = fetch_kraken(sym, limit=300)
                source = "Kraken"
            except Exception as e:
                print(f"[{sym}] Bitstamp unexpected error ({e!r}); falling back to Kraken…")
                page = fetch_kraken(sym, limit=300)
                source = "Kraken"

            if not page:
                print(f"[{sym}] No candles from {source}. Skipping.")
                continue

            print(f"[{sym}] {source} recent fetched {len(page)} candles.")
            rows = normalize_to_bronze_rows(sym, page)
            inserted = upsert_bronze(conn, rows)
            print(f"[{sym}] inserted (or deduped) {inserted} rows.")

    print("Ingest complete.")

if __name__ == "__main__":
    main()