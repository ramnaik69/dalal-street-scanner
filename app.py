import streamlit as st
import pandas as pd

from data_fetcher import fetch_market_data, compute_relative_strength
from config import INDEX_MAP, RETURN_WINDOWS, VOLUME_WINDOWS

st.set_page_config(page_title="Dalal Street Scanner", layout="wide")
st.title("📊 Dalal Street Scanner")


@st.cache_data(ttl=3600, show_spinner=True)
def load_data(force_refresh=False):
    return fetch_market_data(force_refresh=force_refresh)


col1, col2 = st.columns([1, 2])
with col1:
    refresh_clicked = st.button("🔄 Refresh data")
with col2:
    benchmark = st.selectbox("Benchmark", list(INDEX_MAP.keys()), index=0)

try:
    raw_df, summary_df, index_df, meta = load_data(force_refresh=refresh_clicked)
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

st.caption(
    f"Last refresh: {meta.get('last_refresh', 'NA')} | "
    f"Fallback used: {meta.get('used_fallback', False)} | "
    f"Stocks: {len(summary_df)}"
)

summary_df = summary_df.loc[:, ~summary_df.columns.duplicated()].copy()
index_df = index_df.loc[:, ~index_df.columns.duplicated()].copy()

rs_df = compute_relative_strength(summary_df, index_df, benchmark)

focus_df = rs_df[
    (rs_df["Above_200_SMA"] == True) &
    (rs_df["Above_Jan_High"] == True) &
    (rs_df["ADX_D"] > 20) &
    (rs_df["RSI_D"] > 50) &
    (rs_df["VOL_VS_5"] > 1) &
    (rs_df["RS_55D_vs_Benchmark"] > 0)
].copy()

adv = int((summary_df["RET_1D"] > 0).sum())
dec = int((summary_df["RET_1D"] < 0).sum())
unch = int((summary_df["RET_1D"] == 0).sum())
ratio = round(adv / dec, 2) if dec else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("Advances", adv)
m2.metric("Declines", dec)
m3.metric("Unchanged", unch)
m4.metric("A/D Ratio", ratio if ratio is not None else "NA")

tab1, tab2, tab3, tab4 = st.tabs(["Screener", "Focus List", "Indices", "Relative Strength"])

with tab1:
    screener_cols = [
        "Symbol", "Name", "Exchange", "Date", "Close",
        "EMA_13", "EMA_21", "EMA_50", "EMA_100", "EMA_200", "SMA_200",
        "Above_200_SMA",
        "RSI_D", "ADX_D", "MACD_D",
        "JAN_HIGH", "JAN_LOW", "Above_Jan_High", "Below_Jan_Low",
        "PIVOT_D"
    ]
    screener_cols += [f"RET_{w}D" for w in RETURN_WINDOWS]
    screener_cols += [f"VOL_VS_{w}" for w in VOLUME_WINDOWS]

    screener_cols = [c for c in screener_cols if c in summary_df.columns]
    view = summary_df[screener_cols].copy()

    st.dataframe(view, use_container_width=True, height=520)

    st.download_button(
        "Download Screener CSV",
        view.to_csv(index=False).encode("utf-8"),
        "dalal_street_screener.csv",
        "text/csv"
    )

with tab2:
    st.dataframe(focus_df, use_container_width=True, height=500)
    st.download_button(
        "Download Focus List CSV",
        focus_df.to_csv(index=False).encode("utf-8"),
        "dalal_street_focus_list.csv",
        "text/csv"
    )

with tab3:
    st.dataframe(index_df, use_container_width=True, height=420)
    if not index_df.empty:
        st.download_button(
            "Download Indices CSV",
            index_df.to_csv(index=False).encode("utf-8"),
            "dalal_street_indices.csv",
            "text/csv"
        )

with tab4:
    rs_cols = [
        "Symbol", "Name", "Exchange", "Close",
        "RET_1D", "RET_21D", "RET_55D", "RET_123D", "RET_180D",
        "RS_1D_vs_Benchmark", "RS_21D_vs_Benchmark",
        "RS_55D_vs_Benchmark", "RS_123D_vs_Benchmark", "RS_180D_vs_Benchmark",
        "Above_200_SMA", "Above_Jan_High", "RSI_D", "ADX_D"
    ]
    rs_cols = [c for c in rs_cols if c in rs_df.columns]
    rs_view = rs_df[rs_cols].copy()

    if "RS_55D_vs_Benchmark" in rs_view.columns:
        rs_view = rs_view.sort_values("RS_55D_vs_Benchmark", ascending=False)

    st.dataframe(rs_view, use_container_width=True, height=500)
