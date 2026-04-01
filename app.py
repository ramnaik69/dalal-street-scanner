import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(layout="wide")
st.title("📊 Dalal Street Scanner")

stocks = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"]

@st.cache_data
def fetch_data():
    all_data = []
    for stock in stocks:
        df = yf.download(stock, period="6mo", interval="1d", progress=False)
        df["Stock"] = stock
        all_data.append(df)
    return pd.concat(all_data)

df = fetch_data()

df["EMA_200"] = df.groupby("Stock")["Close"].transform(lambda x: x.ewm(span=200).mean())
df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)

df["Above_200"] = df["Close"] > df["EMA_200"]

latest = df.groupby("Stock").tail(1)

st.subheader("📊 Screener")
st.dataframe(latest)

st.subheader("🎯 Focus List")
focus = latest[(latest["Above_200"]) & (latest["RSI"]>50) & (latest["ADX"]>20)]
st.dataframe(focus)

st.download_button("Download CSV", latest.to_csv().encode(), "screener.csv")
