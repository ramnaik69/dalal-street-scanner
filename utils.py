from __future__ import annotations

import numpy as np
import pandas as pd


RETURN_WINDOWS = [1, 2, 3, 5, 7, 10, 21, 30, 55, 60, 90, 120, 180, 240, 360]
VOLUME_WINDOWS = [5, 10, 20, 30, 60, 90]


def safe_round(value, digits=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    try:
        return round(float(value), digits)
    except Exception:
        return np.nan


def pct_return(close: pd.Series, window: int) -> float:
    if len(close) <= window:
        return np.nan
    val = close.pct_change(window).iloc[-1] * 100
    return safe_round(val)


def volume_ratio(volume: pd.Series, window: int) -> float:
    if len(volume) < window:
        return np.nan
    avg = volume.rolling(window).mean().iloc[-1]
    if avg in (0, np.nan) or pd.isna(avg):
        return np.nan
    return safe_round(volume.iloc[-1] / avg, 3)


def classic_pivots(high: float, low: float, close: float) -> dict[str, float]:
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return {
        "R3": safe_round(r3),
        "R2": safe_round(r2),
        "R1": safe_round(r1),
        "Pivot": safe_round(pivot),
        "S1": safe_round(s1),
        "S2": safe_round(s2),
        "S3": safe_round(s3),
    }
