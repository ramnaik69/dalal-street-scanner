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
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    df = safe_numeric(df, ["Open", "High", "Low", "Close", "Volume"])
    df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

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
    jan_df = (
        df[df["Date"].dt.month == 1]
        .groupby("Year", as_index=False)
        .agg(JAN_HIGH=("High", "max"), JAN_LOW=("Low", "min"))
    )
    df = df.merge(jan_df, on="Year", how="left")

    df["Above_Jan_High"] = df["Close"] > df["JAN_HIGH"]
    df["Below_Jan_Low"] = df["Close"] < df["JAN_LOW"]

    prev_high = df["High"].shift(1)
    prev_low = df["Low"].shift(1)
    prev_close = df["Close"].shift(1)

    df["PIVOT_D"] = (prev_high + prev_low + prev_close) / 3.0
    df["R1_D"] = 2 * df["PIVOT_D"] - prev_low
    df["S1_D"] = 2 * df["PIVOT_D"] - prev_high
    df["R2_D"] = df["PIVOT_D"] + (prev_high - prev_low)
    df["S2_D"] = df["PIVOT_D"] - (prev_high - prev_low)

    df["Above_200_SMA"] = df["Close"] > df["SMA_200"]
    return df


def compute_htf_metrics(df: pd.DataFrame):
    x = df.copy().set_index("Date").sort_index()

    weekly = x.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    monthly = x.resample("M").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    out = {}

    # Weekly
    try:
        out["RSI_W"] = ta.momentum.rsi(weekly["Close"], window=14).iloc[-1] if len(weekly) >= 14 else np.nan
    except Exception:
        out["RSI_W"] = np.nan

    try:
        out["ADX_W"] = ta.trend.adx(weekly["High"], weekly["Low"], weekly["Close"], window=14).iloc[-1] if len(weekly) >= 14 else np.nan
    except Exception:
        out["ADX_W"] = np.nan

    try:
        out["MACD_W"] = ta.trend.MACD(weekly["Close"]).macd().iloc[-1] if len(weekly) >= 26 else np.nan
    except Exception:
        out["MACD_W"] = np.nan

    if len(weekly) >= 2:
        ph, pl, pc = weekly["High"].iloc[-2], weekly["Low"].iloc[-2], weekly["Close"].iloc[-2]
        p = (ph + pl + pc) / 3
        out["PIVOT_W"] = p
        out["R1_W"] = 2 * p - pl
        out["S1_W"] = 2 * p - ph
        out["R2_W"] = p + (ph - pl)
        out["S2_W"] = p - (ph - pl)
    else:
        out["PIVOT_W"] = np.nan
        out["R1_W"] = np.nan
        out["S1_W"] = np.nan
        out["R2_W"] = np.nan
        out["S2_W"] = np.nan

    # Monthly
    try:
        out["RSI_M"] = ta.momentum.rsi(monthly["Close"], window=14).iloc[-1] if len(monthly) >= 14 else np.nan
    except Exception:
        out["RSI_M"] = np.nan

    try:
        out["ADX_M"] = ta.trend.adx(monthly["High"], monthly["Low"], monthly["Close"], window=14).iloc[-1] if len(monthly) >= 14 else np.nan
    except Exception:
        out["ADX_M"] = np.nan

    try:
        out["MACD_M"] = ta.trend.MACD(monthly["Close"]).macd().iloc[-1] if len(monthly) >= 26 else np.nan
    except Exception:
        out["MACD_M"] = np.nan

    if len(monthly) >= 2:
        ph, pl, pc = monthly["High"].iloc[-2], monthly["Low"].iloc[-2], monthly["Close"].iloc[-2]
        p = (ph + pl + pc) / 3
        out["PIVOT_M"] = p
        out["R1_M"] = 2 * p - pl
        out["S1_M"] = 2 * p - ph
        out["R2_M"] = p + (ph - pl)
        out["S2_M"] = p - (ph - pl)
    else:
        out["PIVOT_M"] = np.nan
        out["R1_M"] = np.nan
        out["S1_M"] = np.nan
        out["R2_M"] = np.nan
        out["S2_M"] = np.nan

    return out


def build_latest_summary(symbol: str, name: str, exchange: str, df: pd.DataFrame) -> dict:
    df = add_indicators(df)

    if df.empty:
        return {}

    last = df.iloc[-1]
    htf = compute_htf_metrics(df)

    row = {
        "Symbol": symbol,
        "Name": name,
        "Exchange": exchange,
        "Date": last["Date"],
        "Close": round(float(last["Close"]), 2) if pd.notna(last["Close"]) else np.nan,

        "EMA_13": round(float(last["EMA_13"]), 2) if pd.notna(last["EMA_13"]) else np.nan,
        "EMA_21": round(float(last["EMA_21"]), 2) if pd.notna(last["EMA_21"]) else np.nan,
        "EMA_50": round(float(last["EMA_50"]), 2) if pd.notna(last["EMA_50"]) else np.nan,
        "EMA_100": round(float(last["EMA_100"]), 2) if pd.notna(last["EMA_100"]) else np.nan,
        "EMA_200": round(float(last["EMA_200"]), 2) if pd.notna(last["EMA_200"]) else np.nan,
        "SMA_200": round(float(last["SMA_200"]), 2) if pd.notna(last["SMA_200"]) else np.nan,

        "Above_200_SMA": bool(last["Above_200_SMA"]) if pd.notna(last["Above_200_SMA"]) else False,

        "RSI_D": round(float(last["RSI_D"]), 2) if pd.notna(last["RSI_D"]) else np.nan,
        "ADX_D": round(float(last["ADX_D"]), 2) if pd.notna(last["ADX_D"]) else np.nan,
        "MACD_D": round(float(last["MACD_D"]), 4) if pd.notna(last["MACD_D"]) else np.nan,

        "JAN_HIGH": round(float(last["JAN_HIGH"]), 2) if pd.notna(last["JAN_HIGH"]) else np.nan,
        "JAN_LOW": round(float(last["JAN_LOW"]), 2) if pd.notna(last["JAN_LOW"]) else np.nan,
        "Above_Jan_High": bool(last["Above_Jan_High"]) if pd.notna(last["Above_Jan_High"]) else False,
        "Below_Jan_Low": bool(last["Below_Jan_Low"]) if pd.notna(last["Below_Jan_Low"]) else False,

        "PIVOT_D": round(float(last["PIVOT_D"]), 2) if pd.notna(last["PIVOT_D"]) else np.nan,
        "R1_D": round(float(last["R1_D"]), 2) if pd.notna(last["R1_D"]) else np.nan,
        "S1_D": round(float(last["S1_D"]), 2) if pd.notna(last["S1_D"]) else np.nan,
        "R2_D": round(float(last["R2_D"]), 2) if pd.notna(last["R2_D"]) else np.nan,
        "S2_D": round(float(last["S2_D"]), 2) if pd.notna(last["S2_D"]) else np.nan,
    }

    for key, val in htf.items():
        row[key] = round(float(val), 4) if pd.notna(val) else np.nan

    for w in RETURN_WINDOWS:
        val = last.get(f"RET_{w}D", np.nan)
        row[f"RET_{w}D"] = round(float(val), 2) if pd.notna(val) else np.nan

    for w in VOLUME_WINDOWS:
        val = last.get(f"VOL_VS_{w}", np.nan)
        row[f"VOL_VS_{w}"] = round(float(val), 2) if pd.notna(val) else np.nan

    return row
