import streamlit as st
from data_fetcher import fetch_market_data

st.set_page_config(page_title="Dalal Street Scanner", layout="wide")
st.title("📊 Dalal Street Scanner (Multi Timeframe)")
st.warning("APP VERSION: CLEAN-2026-04-02-V1")

@st.cache_data(ttl=60)
def load_data():
    return fetch_market_data(force_refresh=False)

if st.button("🔄 Reload 5PM Snapshot"):
    st.cache_data.clear()

raw_df, summary_df, index_df, meta = load_data()

st.caption(
    f"Last Refresh: {meta.get('last_refresh')} | "
    f"Mode: {meta.get('data_mode', 'snapshot')} | "
    f"Stocks Loaded: {meta.get('symbols_loaded')} | "
    f"Stocks Succeeded: {meta.get('symbols_succeeded')}"
)
if meta.get("error"):
    st.warning(meta["error"])

cols = [
    "Symbol", "Name", "Exchange", "Close",
    "MarketCap", "EPS", "BookValue",
    "NIFTY50", "NIFTY100", "NIFTY500", "NIFTYMIDCAP", "NIFTYSMALLCAP", "NIFTYNEXT50", "NIFTYBANK",
    "EMA_13", "EMA_21", "EMA_50", "EMA_100", "EMA_200",
    "SMA_200", "Above_200_SMA",
    "RSI_D", "ADX_D", "MACD_D", "PIVOT_D",
    "RSI_W", "ADX_W", "MACD_W", "PIVOT_W",
    "RSI_M", "ADX_M", "MACD_M", "PIVOT_M",
]
cols = [c for c in cols if c in summary_df.columns]

view = summary_df[cols].copy()

focus_conditions = [
    "Within_15pct_52W_High",
    "Above_200_SMA",
    "Above_Jan_High",
    "Daily_RSI_GT_50",
    "Daily_ADX_GT_20",
]
focus_conditions = [c for c in focus_conditions if c in summary_df.columns]

focus_df = summary_df.copy()
for c in focus_conditions:
    focus_df = focus_df[focus_df[c] == True]
focus_view = focus_df[cols].copy()

tab_all, tab_focus = st.tabs(["📋 Screener", "🎯 Focus Screener"])
with tab_all:
    st.dataframe(view, use_container_width=True, height=650)

with tab_focus:
    st.dataframe(focus_view, use_container_width=True, height=650)
    st.caption("Filters: within 15% of 52W high, above 200 DMA, above January high, RSI>50, ADX>20")

st.download_button(
    "Download Screener CSV",
    view.to_csv(index=False).encode("utf-8"),
    "multi_timeframe_screener.csv",
    "text/csv"
)

st.download_button(
    "Download Focus Screener CSV",
    focus_view.to_csv(index=False).encode("utf-8"),
    "focus_screener.csv",
    "text/csv"
)
