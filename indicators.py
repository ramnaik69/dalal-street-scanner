import numpy as np
import pandas as pd
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # EMA / SMA
    for w in [13, 21, 50, 100, 200]:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["Above_200_SMA"] = df["Close"] > df["SMA_200"]

    # DAILY indicators
    try:
        df["RSI_D"] = ta.momentum.rsi(df["Close"], window=14)
        df["ADX_D"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
        df["MACD_D"] = ta.trend.MACD(df["Close"]).macd()
    except:
        df["RSI_D"] = np.nan
        df["ADX_D"] = np.nan
        df["MACD_D"] = np.nan

    # DAILY pivot
    df["PIVOT_D"] = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3

    return df


def compute_htf(df: pd.DataFrame):
    df = df.copy()
    df = df.set_index("Date")

    # WEEKLY
    weekly = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    # MONTHLY
    monthly = df.resample("M").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    result = {}

    # WEEKLY indicators
    try:
        result["RSI_W"] = ta.momentum.rsi(weekly["Close"], 14).iloc[-1]
        result["ADX_W"] = ta.trend.adx(weekly["High"], weekly["Low"], weekly["Close"], 14).iloc[-1]
        result["MACD_W"] = ta.trend.MACD(weekly["Close"]).macd().iloc[-1]
    except:
        result["RSI_W"] = np.nan
        result["ADX_W"] = np.nan
        result["MACD_W"] = np.nan

    if len(weekly) >= 2:
        ph, pl, pc = weekly.iloc[-2][["High", "Low", "Close"]]
        result["PIVOT_W"] = (ph + pl + pc) / 3
    else:
        result["PIVOT_W"] = np.nan

    # MONTHLY indicators
    try:
        result["RSI_M"] = ta.momentum.rsi(monthly["Close"], 14).iloc[-1]
        result["ADX_M"] = ta.trend.adx(monthly["High"], monthly["Low"], monthly["Close"], 14).iloc[-1]
        result["MACD_M"] = ta.trend.MACD(monthly["Close"]).macd().iloc[-1]
    except:
        result["RSI_M"] = np.nan
        result["ADX_M"] = np.nan
        result["MACD_M"] = np.nan

    if len(monthly) >= 2:
        ph, pl, pc = monthly.iloc[-2][["High", "Low", "Close"]]
        result["PIVOT_M"] = (ph + pl + pc) / 3
    else:
        result["PIVOT_M"] = np.nan

    return result


def build_latest_summary(symbol, name, exchange, df):
    df = add_indicators(df)

    if df.empty:
        return {}

    last = df.iloc[-1]

    htf = compute_htf(df)

    return {
        "Symbol": symbol,
        "Name": name,
        "Exchange": exchange,
        "Date": last["Date"],
        "Close": round(last["Close"], 2),

        "EMA_13": last["EMA_13"],
        "EMA_21": last["EMA_21"],
        "EMA_50": last["EMA_50"],
        "EMA_100": last["EMA_100"],
        "EMA_200": last["EMA_200"],
        "SMA_200": last["SMA_200"],
        "Above_200_SMA": last["Above_200_SMA"],

        "RSI_D": last["RSI_D"],
        "ADX_D": last["ADX_D"],
        "MACD_D": last["MACD_D"],
        "PIVOT_D": last["PIVOT_D"],

        "RSI_W": htf["RSI_W"],
        "ADX_W": htf["ADX_W"],
        "MACD_W": htf["MACD_W"],
        "PIVOT_W": htf["PIVOT_W"],

        "RSI_M": htf["RSI_M"],
        "ADX_M": htf["ADX_M"],
        "MACD_M": htf["MACD_M"],
        "PIVOT_M": htf["PIVOT_M"],
    }
