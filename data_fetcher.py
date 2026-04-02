import json
import os
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

BATCH_SIZE = 100
MIN_SUCCESS_RATIO = 0.10


def load_symbols():
    df = pd.read_csv(SYMBOLS_FILE)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop_duplicates(subset=["Symbol"], keep="first").reset_index(drop=True)
    return df


def fetch_one(yahoo_symbol: str) -> pd.DataFrame:
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

        if "Date" not in df.columns:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

        needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
        keep = [c for c in needed if c in df.columns]
        if len(keep) < 6:
            return pd.DataFrame()

        out = df[keep].copy()
        out = out.loc[:, ~out.columns.duplicated()]
        return out

    except Exception:
        return pd.DataFrame()


def fetch_batch(batch_df: pd.DataFrame):
    raw_frames = []
    summary_rows = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {
            ex.submit(fetch_one, row.YahooSymbol): row
            for row in batch_df.itertuples(index=False)
        }

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
                df = df.loc[:, ~df.columns.duplicated()]

                raw_frames.append(df)

                summary_row = build_latest_summary(
                    row.Symbol,
                    row.Name,
                    row.Exchange,
                    df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy(),
                )
                if summary_row:
                    summary_rows.append(summary_row)

            except Exception:
                continue

    return raw_frames, summary_rows


def fetch_index_summaries():
    rows = []

    for index_name, ticker in INDEX_MAP.items():
        try:
            df = fetch_one(ticker)
            if df.empty:
                continue

            df = add_indicators(df)
            if df.empty:
                continue

            last = df.iloc[-1]

            row = {
                "Index": index_name,
                "Ticker": ticker,
                "Date": last["Date"],
                "Close": round(float(last["Close"]), 2) if pd.notna(last["Close"]) else None,
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

    idx_df = pd.DataFrame(rows)
    if not idx_df.empty:
        idx_df = idx_df.loc[:, ~idx_df.columns.duplicated()]
        idx_df = idx_df.drop_duplicates(subset=["Index"], keep="first").reset_index(drop=True)
    return idx_df


def fetch_market_data(force_refresh: bool = False):
    symbols = load_symbols()

    raw_frames_all = []
    summary_rows_all = []

    total_symbols = len(symbols)

    for start in range(0, total_symbols, BATCH_SIZE):
        batch_df = symbols.iloc[start:start + BATCH_SIZE].copy()
        raw_frames, summary_rows = fetch_batch(batch_df)
        raw_frames_all.extend(raw_frames)
        summary_rows_all.extend(summary_rows)

    success_count = len(summary_rows_all)
    success_ratio = (success_count / total_symbols) if total_symbols else 0

    if success_count == 0 or success_ratio < MIN_SUCCESS_RATIO:
        raise RuntimeError(
            f"Too few symbols fetched successfully. Success: {success_count}/{total_symbols}"
        )

    raw = pd.concat(raw_frames_all, ignore_index=True)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    raw = raw.reset_index(drop=True)

    summary = pd.DataFrame(summary_rows_all)
    summary = summary.loc[:, ~summary.columns.duplicated()]
    summary = summary.drop_duplicates(subset=["Symbol"], keep="first")
    summary = summary.sort_values(["Exchange", "Symbol"]).reset_index(drop=True)

    idx_summary = fetch_index_summaries()
    if idx_summary.empty:
        idx_summary = pd.DataFrame(
            columns=[
                "Index", "Ticker", "Date", "Close",
                "RSI_D", "ADX_D", "PIVOT_D",
                "RET_1D", "RET_21D", "RET_55D", "RET_123D", "RET_180D"
            ]
        )

    os.makedirs(CACHE_FILE.parent, exist_ok=True)
    raw.to_parquet(CACHE_FILE, index=False)
    summary.to_parquet(SUMMARY_FILE, index=False)
    idx_summary.to_parquet(INDEX_SUMMARY_FILE, index=False)

    meta = {
        "last_refresh": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "used_fallback": False,
        "rows_raw": int(len(raw)),
        "rows_summary": int(len(summary)),
        "symbols_loaded": int(total_symbols),
        "symbols_succeeded": int(success_count),
        "success_ratio": round(success_ratio, 4),
        "batch_size": BATCH_SIZE,
    }
    META_FILE.write_text(json.dumps(meta, indent=2))

    return raw, summary, idx_summary, meta
