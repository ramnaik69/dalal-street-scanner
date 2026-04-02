import streamlit as st
import pandas as pd

from data_fetcher import fetch_market_data

st.set_page_config(page_title="Dalal Street Scanner", layout="wide")
st.title("📊 Dalal Street Scanner (Multi Timeframe)")

st.write("VERSION: MULTI TIMEFRAME ENABLED")


@st.cache_data(ttl=3600)
def load_data(force=False):
    return fetch_market_data(force_refresh=force)


refresh = st.button("🔄 Refresh Data")

raw_df, summary_df, index_df, meta = load_data(force=refresh)

st.caption(f"Last Refresh: {meta.get('last_refresh')} | Stocks: {len(summary_df)}")

# =========================
# SCREENER TABLE
# =========================

cols = [
    "Symbol", "Name", "Exchange", "Close",

    # Trend
    "EMA_13", "EMA_21", "EMA_50", "EMA_100", "EMA_200",
    "SMA_200", "Above_200_SMA",

    # DAILY
    "RSI_D", "ADX_D", "MACD_D", "PIVOT_D",

    # WEEKLY
    "RSI_W", "ADX_W", "MACD_W", "PIVOT_W",

    # MONTHLY
    "RSI_M", "ADX_M", "MACD_M", "PIVOT_M",
]

cols = [c for c in cols if c in summary_df.columns]

view = summary_df[cols].copy()

st.subheader("📋 Screener (D / W / M)")
st.dataframe(view, use_container_width=True, height=600)

st.download_button(
    "Download Screener CSV",
    view.to_csv(index=False),
    "multi_timeframe_screener.csv",
    "text/csv"
)
