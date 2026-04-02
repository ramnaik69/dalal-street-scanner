import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from config import SYMBOLS_FILE, HISTORY_PERIOD, INTERVAL, MAX_WORKERS
from indicators import build_latest_summary


def load_symbols():
    df = pd.read_csv(SYMBOLS_FILE)

    if "Symbol" not in df.columns:
        raise RuntimeError("symbols.csv must contain a Symbol column")

    # Auto-create YahooSymbol if missing
    if "YahooSymbol" not in df.columns:
        df["YahooSymbol"] = df["Symbol"].astype(str).str.strip() + ".NS"
    else:
        df["YahooSymbol"] = df["YahooSymbol"].fillna(
            df["Symbol"].astype(str).str.strip() + ".NS"
        )

    # Fill optional columns if missing
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


def fetch_one(yahoo_symbol):
    try:
        df = yf.download(
            yahoo_symbol,
            period=HISTORY_PERIOD,
            interval=INTERVAL,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if df is None or df.empty:
            return pd.DataFrame()

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

    except Exception:
        return pd.DataFrame()


def fetch_index_summaries():
    # keep it simple for now
    return pd.DataFrame()


def fetch_market_data(force_refresh=False):
    symbols = load_symbols()

    raw_all = []
    summary_all = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_one, row.YahooSymbol): row
            for row in symbols.itertuples(index=False)
        }

        for future in as_completed(futures):
            row = futures[future]
            try:
                df = future.result()
                if df.empty:
                    continue

                df["Symbol"] = row.Symbol
                df["Name"] = row.Name
                df["Exchange"] = row.Exchange
                raw_all.append(df)

                summary = build_latest_summary(
                    row.Symbol,
                    row.Name,
                    row.Exchange,
                    df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy(),
                )

                if summary:
                    summary_all.append(summary)

            except Exception:
                continue

    success_count = len(summary_all)
    total_symbols = len(symbols)

    if success_count == 0:
        raise RuntimeError("No symbols fetched successfully. Check symbols.csv / Yahoo symbols.")

    raw_df = pd.concat(raw_all, ignore_index=True) if raw_all else pd.DataFrame()
    summary_df = pd.DataFrame(summary_all)
    index_df = fetch_index_summaries()

    meta = {
        "last_refresh": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbols_loaded": total_symbols,
        "symbols_succeeded": success_count,
    }

    return raw_df, summary_df, index_df, meta


def compute_relative_strength(summary_df, index_df, benchmark_name):
    out = summary_df.copy()

    out["RS_1D_vs_Benchmark"] = None
    out["RS_21D_vs_Benchmark"] = None
    out["RS_55D_vs_Benchmark"] = None
    out["RS_123D_vs_Benchmark"] = None
    out["RS_180D_vs_Benchmark"] = None

    if index_df is None or index_df.empty:
        return out

    if "Index" not in index_df.columns:
        return out

    bench = index_df[index_df["Index"] == benchmark_name]
    if bench.empty:
        return out

    bench_row = bench.iloc[0]

    for col in ["RET_1D", "RET_21D", "RET_55D", "RET_123D", "RET_180D"]:
        if col not in out.columns:
            out[col] = None

    out["RS_1D_vs_Benchmark"] = out["RET_1D"] - bench_row.get("RET_1D", 0)
    out["RS_21D_vs_Benchmark"] = out["RET_21D"] - bench_row.get("RET_21D", 0)
    out["RS_55D_vs_Benchmark"] = out["RET_55D"] - bench_row.get("RET_55D", 0)
    out["RS_123D_vs_Benchmark"] = out["RET_123D"] - bench_row.get("RET_123D", 0)
    out["RS_180D_vs_Benchmark"] = out["RET_180D"] - bench_row.get("RET_180D", 0)

    return out
