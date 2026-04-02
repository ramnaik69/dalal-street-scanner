from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import build_stock_snapshot
from nse_indices import is_fno_symbol, membership_flags

MAX_WORKERS = 16


def _symbols_path() -> Path:
    candidates = [Path("data/symbols.csv"), Path("symbols.csv")]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("symbols.csv not found in data/ or project root")


def load_symbols() -> pd.DataFrame:
    df = pd.read_csv(_symbols_path())
    if "Symbol" not in df.columns:
        raise ValueError("symbols.csv must have Symbol column")

    if "YahooSymbol" not in df.columns:
        df["YahooSymbol"] = df["Symbol"].astype(str).str.strip() + ".NS"

    if "Name" not in df.columns:
        df["Name"] = df["Symbol"]
    if "Exchange" not in df.columns:
        df["Exchange"] = "NSE"

    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df["YahooSymbol"] = df["YahooSymbol"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Exchange"] = df["Exchange"].astype(str).str.strip()
    return df.drop_duplicates(subset=["Symbol"], keep="first").reset_index(drop=True)


def _fundamentals(ticker: yf.Ticker) -> dict:
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    market_cap = info.get("marketCap")
    market_cap_cr = round(market_cap / 1e7, 2) if market_cap else np.nan

    return {
        "Mkt Cap (Cr)": market_cap_cr,
        "Sector": info.get("sector", ""),
        "Industry": info.get("industry", ""),
        "EPS": info.get("trailingEps", np.nan),
        "Book Value": info.get("bookValue", np.nan),
        "Face Val": info.get("faceValue", np.nan),
    }


def _fetch_one(row: dict) -> dict | None:
    symbol = row["Symbol"]
    ys = row["YahooSymbol"]

    ticker = yf.Ticker(ys)
    hist = ticker.history(period="3y", interval="1d", auto_adjust=False)
    if hist.empty:
        return None

    hist = hist.reset_index().rename(columns={"Datetime": "Date"})
    base_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    hist = hist[base_cols].dropna(subset=["Date", "Close"]) if set(base_cols).issubset(hist.columns) else pd.DataFrame()
    if hist.empty:
        return None

    snapshot = build_stock_snapshot(hist)
    if not snapshot:
        return None

    flags, indices_txt = membership_flags(symbol)
    record = {
        "Symbol": symbol,
        "Name": row["Name"],
        "Exchange": row["Exchange"],
        "F&O": is_fno_symbol(symbol),
        "Indices": indices_txt,
    }
    record.update(flags)
    record.update(_fundamentals(ticker))
    record.update(snapshot)
    return record


def _benchmark_returns() -> dict:
    nifty = yf.Ticker("^NSEI").history(period="3y", interval="1d", auto_adjust=False)
    if nifty.empty:
        return {}
    close = nifty["Close"]
    vals = {}
    for w in [21, 55, 90, 123]:
        vals[w] = close.pct_change(w).iloc[-1] * 100 if len(close) > w else np.nan
    return vals


def fetch_market_data(max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    symbols = load_symbols()
    records: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_fetch_one, row._asdict()) for row in symbols.itertuples(index=False)]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                if rec:
                    records.append(rec)
            except Exception:
                continue

    df = pd.DataFrame(records)
    if df.empty:
        return df

    bench = _benchmark_returns()
    for w in [21, 55, 90, 123]:
        if bench.get(w) is not None and not np.isnan(bench.get(w)):
            df[f"RS {w}D"] = (df.get(f"{w}D%") - bench[w]).round(2)
        else:
            df[f"RS {w}D"] = np.nan

    ordered = [
        "Symbol", "Name", "Exchange", "Date", "F&O", "Indices", "Close",
        "Mkt Cap (Cr)", "Sector", "Industry", "EPS", "Book Value", "Face Val",
        "EMA 13", "EMA 21", "EMA 50", "EMA 100", "EMA 200",
        "52W High", "52W Low", "ATH", "ATL", "15% Range", "SMA 200", "Above 200 SMA",
        "RSI D", "RSI W", "RSI M", "ADX D", "ADX W", "ADX M", "MACD D", "MACD W", "MACD M",
        "Jan High", "Jan Low", "Above Jan High", "Below Jan Low",
        "Pivot D", "Pivot W", "Pivot M",
        "1D%", "2D%", "3D%", "5D%", "7D%", "10D%", "21D%", "30D%", "55D%", "60D%", "90D%", "120D%", "180D%", "240D%", "360D%",
        "Vol/5D Avg", "Vol/10D Avg", "Vol/20D Avg", "Vol/30D Avg", "Vol/60D Avg", "Vol/90D Avg",
        "RS 21D", "RS 55D", "RS 90D", "RS 123D",
        "D R3", "D R2", "D R1", "D Pivot", "D S1", "D S2", "D S3",
        "W R3", "W R2", "W R1", "W Pivot", "W S1", "W S2", "W S3",
        "M R3", "M R2", "M R1", "M Pivot", "M S1", "M S2", "M S3",
        "Y R3", "Y R2", "Y R1", "Y Pivot", "Y S1", "Y S2", "Y S3",
    ]
    index_cols = sorted([c for c in df.columns if c.startswith("IDX_")])
    final_cols = ordered + index_cols
    for c in final_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[final_cols].drop_duplicates(subset=["Symbol"], keep="first").sort_values("Symbol").reset_index(drop=True)
    return df
