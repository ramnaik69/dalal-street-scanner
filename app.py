import streamlit as st
from data_fetcher import fetch_market_data

st.set_page_config(page_title="Dalal Street Scanner", layout="wide")
st.title("📊 Dalal Street Scanner (Multi Timeframe)")
st.warning("APP VERSION: CLEAN-2026-04-02-V1")

@st.cache_data(ttl=60)
def load_data(force=False):
    return fetch_market_data(force_refresh=force)

refresh = st.button("🔄 Refresh Data")

raw_df, summary_df, index_df, meta = load_data(force=refresh)

st.caption(
    f"Last Refresh: {meta.get('last_refresh')} | "
    f"Stocks Loaded: {meta.get('symbols_loaded')} | "
    f"Stocks Succeeded: {meta.get('symbols_succeeded')}"
)

cols = [
    "Symbol", "Name", "Exchange", "Close",
    "EMA_13", "EMA_21", "EMA_50", "EMA_100", "EMA_200",
    "SMA_200", "Above_200_SMA",
    "RSI_D", "ADX_D", "MACD_D", "PIVOT_D",
    "RSI_W", "ADX_W", "MACD_W", "PIVOT_W",
    "RSI_M", "ADX_M", "MACD_M", "PIVOT_M",
]
cols = [c for c in cols if c in summary_df.columns]

view = summary_df[cols].copy()

st.subheader("📋 Screener")
st.dataframe(view, use_container_width=True, height=650)

st.download_button(
    "Download Screener CSV",
    view.to_csv(index=False).encode("utf-8"),
    "multi_timeframe_screener.csv",
    "text/csv"
)
