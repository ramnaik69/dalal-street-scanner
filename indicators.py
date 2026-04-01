import numpy as np
import pandas as pd
import ta
from config import RETURN_WINDOWS, VOLUME_WINDOWS

def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df = safe_numeric(df, ["Open", "High", "Low", "Close", "Volume"])

    for w in [13, 21, 50, 100, 200]:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

    df["SMA_200"] = df["Close"].rolling(200, min_periods=1).mean()

    try:
        df["RSI_D"] = ta.momentum.rsi(df["Close"], window=14)
    except Exception:
        df["RSI_D"] = np.nan

    try:
        df["ADX_D"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
    except Exception:
        df["ADX_D"] = np.nan

    try:
        macd = ta.trend.MACD(df["Close"])
        df["MACD_D"] = macd.macd()
    except Exception:
        df["MACD_D"] = np.nan

    for w in RETURN_WINDOWS:
        df[f"RET_{w}D"] = (df["Close"] / df["Close"].shift(w) - 1.0) * 100.0

    for w in VOLUME_WINDOWS:
        avg = df["Volume"].rolling(w, min_periods=1).mean()
        df[f"VOL_AVG_{w}"] = avg
        df[f"VOL_VS_{w}"] = np.where(avg > 0, df["Volume"] / avg, np.nan)

    df["Year"] = df["Date"].dt.year
    jan_df = df[df["Date"].dt.month == 1].groupby("Year").agg(
        JAN_HIGH=("High", "max"),
        JAN_LOW=("Low", "min")
    ).reset_index()

    df = df.merge(jan_df, on="Year", how="left")
    df["Above_Jan_High"] = df["Close"] > df["JAN_HIGH"]
    df["Below_Jan_Low"] = df["Close"] < df["JAN_LOW"]

    prev_high = df["High"].shift(1)
    prev_low = df["Low"].shift(1)
    prev_close = df["Close"].shift(1)

    df["PIVOT_D"] = (prev_high + prev_low + prev_close) / 3
    df["R1_D"] = 2 * df["PIVOT_D"] - prev_low
    df["S1_D"] = 2 * df["PIVOT_D"] - prev_high
    df["R2_D"] = df["PIVOT_D"] + (prev_high - prev_low)
    df["S2_D"] = df["PIVOT_D"] - (prev_high - prev_low)

    df["Above_200_SMA"] = df["Close"] > df["SMA_200"]
    return df

def build_latest_summary(symbol: str, name: str, exchange: str, df: pd.DataFrame) -> dict:
    df = add_indicators(df)
    last = df.iloc[-1]

    row = {
        "Symbol": symbol,
        "Name": name,
        "Exchange": exchange,
        "Date": last["Date"],
        "Close": round(float(last["Close"]), 2),
        "EMA_13": round(float(last["EMA_13"]), 2),
        "EMA_21": round(float(last["EMA_21"]), 2),
        "EMA_50": round(float(last["EMA_50"]), 2),
        "EMA_100": round(float(last["EMA_100"]), 2),
        "EMA_200": round(float(last["EMA_200"]), 2),
        "SMA_200": round(float(last["SMA_200"]), 2),
        "Above_200_SMA": bool(last["Above_200_SMA"]),
        "RSI_D": round(float(last["RSI_D"]), 2) if pd.notna(last["RSI_D"]) else np.nan,
        "ADX_D": round(float(last["ADX_D"]), 2) if pd.notna(last["ADX_D"]) else np.nan,
        "MACD_D": round(float(last["MACD_D"]), 4) if pd.notna(last["MACD_D"]) else np.nan,
        "JAN_HIGH": round(float(last["JAN_HIGH"]), 2) if pd.notna(last["JAN_HIGH"]) else np.nan,
        "JAN_LOW": round(float(last["JAN_LOW"]), 2) if pd.notna(last["JAN_LOW"]) else np.nan,
        "Above_Jan_High": bool(last["Above_Jan_High"]) if pd.notna(last["Above_Jan_High"]) else False,
        "Below_Jan_Low": bool(last["Below_Jan_Low"]) if pd.notna(last["Below_Jan_Low"]) else False,
        "PIVOT_D": round(float(last["PIVOT_D"]), 2) if pd.notna(last["PIVOT_D"]) else np.nan,
    }

    for w in RETURN_WINDOWS:
        val = last.get(f"RET_{w}D", np.nan)
        row[f"RET_{w}D"] = round(float(val), 2) if pd.notna(val) else np.nan

    for w in VOLUME_WINDOWS:
        val = last.get(f"VOL_VS_{w}", np.nan)
        row[f"VOL_VS_{w}"] = round(float(val), 2) if pd.notna(val) else np.nan

    return row
