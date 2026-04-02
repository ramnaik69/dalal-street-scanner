from __future__ import annotations

import streamlit as st

from data_loader import fetch_market_data
from screener import build_focus_screener

st.set_page_config(page_title="DALAL STREET DON", layout="wide")
st.title("🦁 DALAL STREET DON")
st.caption("Nifty comprehensive screener with 90+ columns and focus strategy.")


@st.cache_data(ttl=900)
def load_data():
    return fetch_market_data()


if st.button("⟳ Refresh Data"):
    st.cache_data.clear()

df = load_data()

if df.empty:
    st.error("No data fetched. Check symbols and internet connectivity.")
    st.stop()

st.success(f"Loaded {len(df)} unique symbols.")

search = st.text_input("Search Symbol or Name")
if search:
    s = search.strip().upper()
    df = df[df["Symbol"].str.contains(s, na=False) | df["Name"].str.upper().str.contains(s, na=False)]

sector_options = ["All"] + sorted([x for x in df["Sector"].dropna().unique() if x])
sector = st.selectbox("Sector", sector_options)
if sector != "All":
    df = df[df["Sector"] == sector]

with st.expander("Quick Filters"):
    near_high = st.checkbox("Within 15% of 52W high")
    above_200 = st.checkbox("Above 200 SMA")
    above_jan = st.checkbox("Above January High")

if near_high:
    df = df[df["15% Range"] == True]
if above_200:
    df = df[df["Above 200 SMA"] == True]
if above_jan:
    df = df[df["Above Jan High"] == True]

focus_df = build_focus_screener(df)

all_tab, focus_tab = st.tabs(["All Stocks", "Focus Screener"])

with all_tab:
    st.dataframe(df, use_container_width=True, height=650)
    st.download_button(
        "⬇ Download Full CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="nse_screener_full.csv",
        mime="text/csv",
    )

with focus_tab:
    st.metric("Matches", len(focus_df))
    st.dataframe(focus_df, use_container_width=True, height=650)
    st.download_button(
        "⬇ Download Focus CSV",
        data=focus_df.to_csv(index=False).encode("utf-8"),
        file_name="nse_focus_screener.csv",
        mime="text/csv",
    )
