import pandas as pd
import yfinance as yf
from datetime import datetime

from config import SYMBOLS_FILE, HISTORY_PERIOD, INTERVAL
from indicators import build_latest_summary


BULK_BATCH_SIZE = 80


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
            threads=False,
        )
        return data
    except Exception:
        return pd.DataFrame()


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

    for batch in chunk_list(yahoo_symbols, BULK_BATCH_SIZE):
        bulk_df = bulk_download_chunk(batch)

        for ys in batch:
            try:
                df = extract_symbol_df(bulk_df, ys)
                if df.empty:
                    continue

                meta = symbol_map[ys]

                df["Symbol"] = meta["Symbol"]
                df["Name"] = meta["Name"]
                df["Exchange"] = meta["Exchange"]

                raw_all.append(df)

                summary = build_latest_summary(
                    meta["Symbol"],
                    meta["Name"],
                    meta["Exchange"],
                    df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy(),
                )

                if summary:
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
    summary_df = summary_df.drop_duplicates(subset=["Symbol"], keep="first").reset_index(drop=True)

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
