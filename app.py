import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(layout="wide")
st.title("📊 Dalal Street Scanner")

stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

@st.cache_data
def build_summary():
    rows = []

    for stock in stocks:
        try:
            df = yf.download(
                stock,
                period="1y",
                interval="1d",
                progress=False,
                auto_adjust=False
            )

            if df is None or df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()

            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                continue

            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["High", "Low", "Close"]).copy()

            if len(df) < 30:
                continue

            df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
            df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
            df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
            df["Above_200"] = df["Close"] > df["EMA_200"]

            last = df.iloc[-1]

            rows.append({
                "Stock": stock,
                "Date": last["Date"],
                "Close": round(float(last["Close"]), 2) if pd.notna(last["Close"]) else None,
                "EMA_200": round(float(last["EMA_200"]), 2) if pd.notna(last["EMA_200"]) else None,
                "RSI": round(float(last["RSI"]), 2) if pd.notna(last["RSI"]) else None,
                "ADX": round(float(last["ADX"]), 2) if pd.notna(last["ADX"]) else None,
                "Volume": int(last["Volume"]) if pd.notna(last["Volume"]) else None,
                "Above_200": bool(last["Above_200"]) if pd.notna(last["Above_200"]) else False
            })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)

latest = build_summary()

if latest.empty:
    st.error("No market data could be fetched right now.")
    st.stop()

st.subheader("📊 Screener")
st.dataframe(latest, use_container_width=True)

st.subheader("🎯 Focus List")
focus = latest[
    (latest["Above_200"] == True) &
    (latest["RSI"] > 50) &
    (latest["ADX"] > 20)
].copy()

st.dataframe(focus, use_container_width=True)

st.download_button(
    "Download Screener CSV",
    latest.to_csv(index=False).encode("utf-8"),
    "screener.csv",
    "text/csv"
)
