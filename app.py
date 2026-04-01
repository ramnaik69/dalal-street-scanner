import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(layout="wide")
st.title("📊 Dalal Street Scanner")

stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

@st.cache_data
def fetch_data():
    frames = []

    for stock in stocks:
        df = yf.download(stock, period="1y", interval="1d", progress=False, auto_adjust=False)

        if df is None or df.empty:
            continue

        # Flatten columns if yfinance returns MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df["Stock"] = stock

        needed = ["Date", "Open", "High", "Low", "Close", "Volume", "Stock"]
        df = df[[c for c in needed if c in df.columns]]

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

df = fetch_data()

if df.empty:
    st.error("No market data could be fetched right now.")
    st.stop()

# Ensure numeric columns are numeric
for col in ["Open", "High", "Low", "Close", "Volume"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)

# Indicators per stock
def add_indicators(group):
    group = group.copy()
    group["EMA_200"] = group["Close"].ewm(span=200, adjust=False).mean()
    group["RSI"] = ta.momentum.rsi(group["Close"], window=14)
    group["ADX"] = ta.trend.adx(group["High"], group["Low"], group["Close"], window=14)
    group["Above_200"] = group["Close"] > group["EMA_200"]
    return group

df = df.groupby("Stock", group_keys=False).apply(add_indicators)

latest = df.groupby("Stock", group_keys=False).tail(1).copy()

st.subheader("📊 Screener")
st.dataframe(
    latest[["Stock", "Date", "Close", "EMA_200", "RSI", "ADX", "Volume", "Above_200"]],
    use_container_width=True
)

st.subheader("🎯 Focus List")
focus = latest[
    (latest["Above_200"] == True) &
    (latest["RSI"] > 50) &
    (latest["ADX"] > 20)
].copy()

st.dataframe(
    focus[["Stock", "Date", "Close", "EMA_200", "RSI", "ADX", "Volume"]],
    use_container_width=True
)

st.download_button(
    "Download Screener CSV",
    latest.to_csv(index=False).encode("utf-8"),
    "screener.csv",
    "text/csv"
)
