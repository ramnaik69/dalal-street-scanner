from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from utils import RETURN_WINDOWS, VOLUME_WINDOWS, classic_pivots, pct_return, safe_round, volume_ratio


def _timeframe_indicators(df: pd.DataFrame) -> tuple[float, float, float]:
    if len(df) < 35:
        return np.nan, np.nan, np.nan
    try:
        rsi = ta.momentum.rsi(df["Close"], window=14).iloc[-1]
        adx = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14).iloc[-1]
        macd_hist = ta.trend.MACD(df["Close"], window_fast=12, window_slow=26, window_sign=9).macd_diff().iloc[-1]
        return safe_round(rsi), safe_round(adx), safe_round(macd_hist, 4)
    except Exception:
        return np.nan, np.nan, np.nan


def _prev_period_pivot(df: pd.DataFrame, rule: str) -> dict[str, float]:
    rs = df.set_index("Date").resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    if len(rs) < 2:
        return {k: np.nan for k in ["R3", "R2", "R1", "Pivot", "S1", "S2", "S3"]}
    prev = rs.iloc[-2]
    return classic_pivots(prev["High"], prev["Low"], prev["Close"])


def build_stock_snapshot(df: pd.DataFrame) -> dict:
    df = df.copy()
    if df.empty:
        return {}
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    for span in [13, 21, 50, 100, 200]:
        df[f"EMA {span}"] = close.ewm(span=span, adjust=False).mean()
    df["SMA 200"] = close.rolling(200).mean()

    latest = df.iloc[-1]
    out = {
        "Date": latest["Date"].date().isoformat(),
        "Close": safe_round(latest["Close"]),
        "EMA 13": safe_round(latest["EMA 13"]),
        "EMA 21": safe_round(latest["EMA 21"]),
        "EMA 50": safe_round(latest["EMA 50"]),
        "EMA 100": safe_round(latest["EMA 100"]),
        "EMA 200": safe_round(latest["EMA 200"]),
        "SMA 200": safe_round(latest["SMA 200"]),
        "Above 200 SMA": bool(latest["Close"] > latest["SMA 200"]) if pd.notna(latest["SMA 200"]) else False,
    }

    rsi_d, adx_d, macd_d = _timeframe_indicators(df)
    out.update({"RSI D": rsi_d, "ADX D": adx_d, "MACD D": macd_d})

    weekly = df.set_index("Date").resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna().reset_index()
    monthly = df.set_index("Date").resample("ME").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna().reset_index()

    rsi_w, adx_w, macd_w = _timeframe_indicators(weekly)
    rsi_m, adx_m, macd_m = _timeframe_indicators(monthly)
    out.update({"RSI W": rsi_w, "ADX W": adx_w, "MACD W": macd_w, "RSI M": rsi_m, "ADX M": adx_m, "MACD M": macd_m})

    now = latest["Date"]
    year_df = df[df["Date"].dt.year == now.year]
    jan_df = year_df[year_df["Date"].dt.month == 1]
    jan_high = jan_df["High"].max() if not jan_df.empty else np.nan
    jan_low = jan_df["Low"].min() if not jan_df.empty else np.nan

    high_52w = high.tail(252).max() if len(df) >= 20 else high.max()
    low_52w = low.tail(252).min() if len(df) >= 20 else low.min()

    out.update(
        {
            "52W High": safe_round(high_52w),
            "52W Low": safe_round(low_52w),
            "ATH": safe_round(high.max()),
            "ATL": safe_round(low.min()),
            "15% Range": bool(latest["Close"] >= 0.85 * high_52w) if pd.notna(high_52w) else False,
            "Jan High": safe_round(jan_high),
            "Jan Low": safe_round(jan_low),
            "Above Jan High": bool(latest["Close"] > jan_high) if pd.notna(jan_high) else False,
            "Below Jan Low": bool(latest["Close"] < jan_low) if pd.notna(jan_low) else False,
        }
    )

    daily_prev = df.iloc[-2] if len(df) > 1 else None
    d_piv = classic_pivots(daily_prev["High"], daily_prev["Low"], daily_prev["Close"]) if daily_prev is not None else {k: np.nan for k in ["R3", "R2", "R1", "Pivot", "S1", "S2", "S3"]}
    w_piv = _prev_period_pivot(df, "W-FRI")
    m_piv = _prev_period_pivot(df, "ME")
    y_piv = _prev_period_pivot(df, "YE")

    out.update({f"D {k}": v for k, v in d_piv.items()})
    out.update({f"W {k}": v for k, v in w_piv.items()})
    out.update({f"M {k}": v for k, v in m_piv.items()})
    out.update({f"Y {k}": v for k, v in y_piv.items()})
    out["Pivot D"] = out["D Pivot"]
    out["Pivot W"] = out["W Pivot"]
    out["Pivot M"] = out["M Pivot"]

    for w in RETURN_WINDOWS:
        out[f"{w}D%"] = pct_return(close, w)

    for w in VOLUME_WINDOWS:
        out[f"Vol/{w}D Avg"] = volume_ratio(vol, w)

    return out
