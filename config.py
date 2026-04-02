from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CACHE_FILE = DATA_DIR / "market_cache.parquet"
SUMMARY_FILE = DATA_DIR / "summary_cache.parquet"
INDEX_SUMMARY_FILE = DATA_DIR / "index_summary_cache.parquet"
META_FILE = DATA_DIR / "meta.json"

SYMBOLS_FILE = "symbols.csv"

INDEX_MAP = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "SENSEX": "^BSESN",
}

RETURN_WINDOWS = [1, 2, 3, 4, 5, 7, 10, 21, 30, 55, 60, 90, 120, 123, 180, 240, 360]
VOLUME_WINDOWS = [5, 10, 20, 30, 60, 90]

MAX_WORKERS = 8
HISTORY_PERIOD = "2y"
INTERVAL = "1d"
