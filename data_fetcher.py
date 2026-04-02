import pandas as pd
import yfinance as yf
import numpy as np
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import SYMBOLS_FILE, HISTORY_PERIOD, INTERVAL
from indicators import build_latest_summary


BULK_BATCH_SIZE = 80
HISTORY_WORKERS = 6
FUNDAMENTAL_WORKERS = 12

INDEX_ENDPOINTS = {
    "NIFTY50": "NIFTY 50",
    "NIFTY100": "NIFTY 100",
    "NIFTY500": "NIFTY 500",
    "NIFTYMIDCAP": "NIFTY MIDCAP 100",
    "NIFTYSMALLCAP": "NIFTY SMALLCAP 100",
    "NIFTYNEXT50": "NIFTY NEXT 50",
    "NIFTYBANK": "NIFTY BANK",
}


def load_symbols():
    df = pd.read_csv(SYMBOLS_FILE)

    if "Symbol" not in df.columns:
        raise RuntimeError("symbols.csv must contain a Symbol column")

    if "YahooSymbol" not in df.columns:
        df["YahooSymbol"] = df["Symbol"].astype(str).str.strip() + ".NS"
    else:
        df["YahooSymbol"] = df["YahooSymbol"].fillna(
            df["Symbol"].astype(str).str.strip() + ".NS"
        )

    if "Name" not in df.columns:
        df["Name"] = df["Symbol"]
    if "Exchange" not in df.columns:
        df["Exchange"] = "NSE"

    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["YahooSymbol"] = df["YahooSymbol"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Exchange"] = df["Exchange"].astype(str).str.strip()

    df = df.drop_duplicates(subset=["Symbol"], keep="first").reset_index(drop=True)
    return df


def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def normalize_single_ticker_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df.loc[:, ~df.columns.duplicated()]

    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    keep = [c for c in needed if c in df.columns]

    if len(keep) < 6:
        return pd.DataFrame()

    out = df[keep].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])
    out = out.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return out


def bulk_download_chunk(yahoo_symbols):
    try:
        data = yf.download(
            tickers=" ".join(yahoo_symbols),
            period=HISTORY_PERIOD,
            interval=INTERVAL,
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
        return data
    except Exception:
        return pd.DataFrame()


def _fetch_single_chunk(batch):
    return batch, bulk_download_chunk(batch)


def load_index_membership():
    members = {name: set() for name in INDEX_ENDPOINTS.keys()}
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/",
        }
    )

    try:
        session.get("https://www.nseindia.com", timeout=10)
    except Exception:
        return members

    for col_name, index_name in INDEX_ENDPOINTS.items():
        try:
            url = "https://www.nseindia.com/api/equity-stockIndices"
            resp = session.get(url, params={"index": index_name}, timeout=12)
            data = resp.json().get("data", [])
            symbols = {
                str(item.get("symbol", "")).strip()
                for item in data
                if item.get("symbol")
            }
            members[col_name] = symbols
        except Exception:
            members[col_name] = set()

    return members


def _safe_float(value):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def fetch_fundamental_for_symbol(yahoo_symbol):
    out = {"MarketCap": np.nan, "EPS": np.nan, "BookValue": np.nan}
    try:
        t = yf.Ticker(yahoo_symbol)
        fast = getattr(t, "fast_info", {}) or {}
        info = getattr(t, "info", {}) or {}

        market_cap = fast.get("market_cap")
        if market_cap is None:
            market_cap = info.get("marketCap")

        eps = info.get("trailingEps")
        if eps is None:
            eps = info.get("forwardEps")

        book_value = info.get("bookValue")

        out["MarketCap"] = _safe_float(market_cap)
        out["EPS"] = _safe_float(eps)
        out["BookValue"] = _safe_float(book_value)
    except Exception:
        pass
    return yahoo_symbol, out


def fetch_fundamentals_map(yahoo_symbols):
    result = {ys: {"MarketCap": np.nan, "EPS": np.nan, "BookValue": np.nan} for ys in yahoo_symbols}
    with ThreadPoolExecutor(max_workers=FUNDAMENTAL_WORKERS) as executor:
        futures = [executor.submit(fetch_fundamental_for_symbol, ys) for ys in yahoo_symbols]
        for fut in as_completed(futures):
            ys, payload = fut.result()
            result[ys] = payload
    return result


def export_outputs(summary_df):
    export_df = summary_df.copy()
    export_df.to_csv("data/screener_latest.csv", index=False)
    export_df.to_html("data/screener_latest.html", index=False)


def extract_symbol_df(bulk_df, yahoo_symbol):
    if bulk_df is None or bulk_df.empty:
        return pd.DataFrame()

    # Multi-ticker response
    if isinstance(bulk_df.columns, pd.MultiIndex):
        if yahoo_symbol not in bulk_df.columns.get_level_values(0):
            return pd.DataFrame()

        try:
            df = bulk_df[yahoo_symbol].copy()
            return normalize_single_ticker_df(df)
        except Exception:
            return pd.DataFrame()

    # Single-ticker response
    return normalize_single_ticker_df(bulk_df)


def fetch_market_data(force_refresh=False):
    symbols = load_symbols()

    raw_all = []
    summary_all = []

    symbol_map = {
        row.YahooSymbol: {
            "Symbol": row.Symbol,
            "Name": row.Name,
            "Exchange": row.Exchange,
        }
        for row in symbols.itertuples(index=False)
    }

    yahoo_symbols = symbols["YahooSymbol"].tolist()

    index_membership = load_index_membership()
    fundamentals_map = fetch_fundamentals_map(yahoo_symbols)

    batches = list(chunk_list(yahoo_symbols, BULK_BATCH_SIZE))
    with ThreadPoolExecutor(max_workers=HISTORY_WORKERS) as executor:
        futures = [executor.submit(_fetch_single_chunk, batch) for batch in batches]
        batch_results = [f.result() for f in as_completed(futures)]

    for batch, bulk_df in batch_results:
        for ys in batch:
            try:
                df = extract_symbol_df(bulk_df, ys)
                if df.empty:
                    continue

                meta = symbol_map[ys]
                fundamentals = fundamentals_map.get(ys, {})

                df["Symbol"] = meta["Symbol"]
                df["Name"] = meta["Name"]
                df["Exchange"] = meta["Exchange"]

                raw_all.append(df)

                summary = build_latest_summary(
                    meta["Symbol"],
                    meta["Name"],
                    meta["Exchange"],
                    df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy(),
                    fundamentals=fundamentals,
                )

                if summary:
                    for membership_col, symbols_set in index_membership.items():
                        summary[membership_col] = meta["Symbol"] in symbols_set
                    summary_all.append(summary)

            except Exception:
                continue

    success_count = len(summary_all)
    total_symbols = len(symbols)

    if success_count == 0:
        raise RuntimeError(
            "Bulk Yahoo loader ran, but zero symbols fetched successfully."
        )

    raw_df = pd.concat(raw_all, ignore_index=True) if raw_all else pd.DataFrame()
    raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()].copy()

    summary_df = pd.DataFrame(summary_all)
    summary_df = summary_df.loc[:, ~summary_df.columns.duplicated()].copy()
    summary_df = summary_df.sort_values(["Symbol", "Date"], ascending=[True, False])
    summary_df = summary_df.drop_duplicates(subset=["Symbol"], keep="first").reset_index(drop=True)
    summary_df = summary_df.sort_values("Symbol").reset_index(drop=True)

    export_outputs(summary_df)

    index_df = pd.DataFrame()

    meta = {
        "last_refresh": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbols_loaded": total_symbols,
        "symbols_succeeded": success_count,
        "batch_size": BULK_BATCH_SIZE,
    }

    return raw_df, summary_df, index_df, meta


def compute_relative_strength(summary_df, index_df, benchmark_name):
    out = summary_df.copy()
    out["RS_1D_vs_Benchmark"] = None
    out["RS_21D_vs_Benchmark"] = None
    out["RS_55D_vs_Benchmark"] = None
    out["RS_123D_vs_Benchmark"] = None
    out["RS_180D_vs_Benchmark"] = None
    return out
