import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import yfinance as yf

from config import (
    CACHE_FILE,
    SUMMARY_FILE,
    INDEX_SUMMARY_FILE,
    META_FILE,
    SYMBOLS_FILE,
    INDEX_MAP,
    HISTORY_PERIOD,
    INTERVAL,
    MAX_WORKERS,
)
from indicators import build_latest_summary, add_indicators

def load_symbols():
    df = pd.read_csv(SYMBOLS_FILE)
    return df

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def fetch_one(yahoo_symbol: str) -> pd.DataFrame:
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

    df = flatten_columns(df).reset_index()
    if "Date" not in df.columns:
        return pd.DataFrame()

    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    keep = [c for c in needed if c in df.columns]
    return df[keep].copy()

def fetch_index_summaries():
    rows = []
    for index_name, ticker in INDEX_MAP.items():
        try:
            df = fetch_one(ticker)
            if df.empty:
                continue
            df = add_indicators(df)
            last = df.iloc[-1]
            row = {
                "Index": index_name,
                "Ticker": ticker,
                "Date": last["Date"],
                "Close": round(float(last["Close"]), 2),
                "RSI_D": round(float(last["RSI_D"]), 2) if pd.notna(last["RSI_D"]) else None,
                "ADX_D": round(float(last["ADX_D"]), 2) if pd.notna(last["ADX_D"]) else None,
                "PIVOT_D": round(float(last["PIVOT_D"]), 2) if pd.notna(last["PIVOT_D"]) else None,
                "RET_1D": round(float(last["RET_1D"]), 2) if pd.notna(last["RET_1D"]) else None,
                "RET_21D": round(float(last["RET_21D"]), 2) if pd.notna(last["RET_21D"]) else None,
                "RET_55D": round(float(last["RET_55D"]), 2) if pd.notna(last["RET_55D"]) else None,
                "RET_123D": round(float(last["RET_123D"]), 2) if pd.notna(last["RET_123D"]) else None,
                "RET_180D": round(float(last["RET_180D"]), 2) if pd.notna(last["RET_180D"]) else None,
            }
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)

def fetch_market_data(force_refresh: bool = False):
    symbols = load_symbols()

    if CACHE_FILE.exists() and SUMMARY_FILE.exists() and INDEX_SUMMARY_FILE.exists() and not force_refresh:
        raw = pd.read_parquet(CACHE_FILE)
        summary = pd.read_parquet(SUMMARY_FILE)
        idx_summary = pd.read_parquet(INDEX_SUMMARY_FILE)
        meta = {}
        if META_FILE.exists():
            meta = json.loads(META_FILE.read_text())
        return raw, summary, idx_summary, meta

    raw_frames = []
    summary_rows = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {}
        for row in symbols.itertuples(index=False):
            future = ex.submit(fetch_one, row.YahooSymbol)
            future_map[future] = row

        for future in as_completed(future_map):
            row = future_map[future]
            try:
                df = future.result()
                if df.empty:
                    continue

                df["Symbol"] = row.Symbol
                df["YahooSymbol"] = row.YahooSymbol
                df["Name"] = row.Name
                df["Exchange"] = row.Exchange

                raw_frames.append(df)

                summary_row = build_latest_summary(
                    row.Symbol,
                    row.Name,
                    row.Exchange,
                    df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
                )
                summary_rows.append(summary_row)
            except Exception:
                continue

    if not raw_frames or not summary_rows:
        raise RuntimeError("No market data fetched.")

    raw = pd.concat(raw_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(["Exchange", "Symbol"]).reset_index(drop=True)
    idx_summary = fetch_index_summaries()

    raw.to_parquet(CACHE_FILE, index=False)
    summary.to_parquet(SUMMARY_FILE, index=False)
    idx_summary.to_parquet(INDEX_SUMMARY_FILE, index=False)

    meta = {
        "last_refresh": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "used_fallback": False,
        "rows_raw": int(len(raw)),
        "rows_summary": int(len(summary)),
    }
    META_FILE.write_text(json.dumps(meta, indent=2))
    return raw, summary, idx_summary, meta

def compute_relative_strength(summary_df: pd.DataFrame, index_df: pd.DataFrame, benchmark_name: str):
    bench = index_df[index_df["Index"] == benchmark_name]
    out = summary_df.copy()
    if bench.empty:
        out["RS_55D_vs_Benchmark"] = None
        return out

    bench_row = bench.iloc[0]
    out["RS_1D_vs_Benchmark"] = out["RET_1D"] - bench_row.get("RET_1D", 0)
    out["RS_21D_vs_Benchmark"] = out["RET_21D"] - bench_row.get("RET_21D", 0)
    out["RS_55D_vs_Benchmark"] = out["RET_55D"] - bench_row.get("RET_55D", 0)
    out["RS_123D_vs_Benchmark"] = out["RET_123D"] - bench_row.get("RET_123D", 0)
    out["RS_180D_vs_Benchmark"] = out["RET_180D"] - bench_row.get("RET_180D", 0)
    return out
