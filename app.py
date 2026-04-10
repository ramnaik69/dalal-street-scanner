import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Dalal Street Scanner",
    page_icon="📊",
    layout="wide",
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/final_3125_master_with_metrics_template.csv")
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).fillna("").str.strip()
    return df

df = load_data()

st.title("📊 Dalal Street Scanner")
st.caption("Professional stock screener with RS ranking, sector heatmap, and advanced filters")

# -------- helpers --------
def to_numeric(df_in, cols):
    for c in cols:
        if c in df_in.columns:
            df_in[c] = pd.to_numeric(df_in[c], errors="coerce")
    return df_in

numeric_cols = [
    "Close", "LTP", "Percent_Change", "Turnover_Cr", "Market_Cap_Cr",
    "RS_21D", "RS_55D", "RS_90D", "RS_123D", "MS_RS",
    "RSI_D", "RSI_W", "RSI_M",
    "ADX_D", "ADX_W", "ADX_M",
    "Return_1D", "Return_5D", "Return_21D", "Return_30D", "Return_60D",
    "Return_90D", "Return_120D", "Return_180D", "Return_240D", "Return_365D",
    "Volume_Today", "Avg_Vol_1W", "Avg_Vol_2W", "Avg_Vol_Monthly",
    "Avg_Vol_Quarterly", "Avg_Vol_Yearly"
]
df = to_numeric(df, numeric_cols)

if "Name" not in df.columns and "Company_Name" in df.columns:
    df["Name"] = df["Company_Name"]

if "LTP" not in df.columns and "Close" in df.columns:
    df["LTP"] = df["Close"]

if "Percent_Change" not in df.columns and "Return_1D" in df.columns:
    df["Percent_Change"] = df["Return_1D"] * 100

# RS rank
rank_base = "MS_RS" if "MS_RS" in df.columns else ("RS_55D" if "RS_55D" in df.columns else None)
if rank_base:
    df["RS_Rank"] = df[rank_base].rank(method="dense", ascending=False)
else:
    df["RS_Rank"] = np.nan

# -------- sidebar filters --------
st.sidebar.header("Filters")

search = st.sidebar.text_input("Search Symbol / Name")

sector_options = ["All"]
if "Sector" in df.columns:
    sector_options += sorted([x for x in df["Sector"].dropna().unique() if x])
selected_sector = st.sidebar.selectbox("Sector", sector_options)

industry_options = ["All"]
if "Industry" in df.columns:
    industry_options += sorted([x for x in df["Industry"].dropna().unique() if x])
selected_industry = st.sidebar.selectbox("Industry", industry_options)

min_mcap = st.sidebar.number_input("Min Market Cap (Cr)", min_value=0.0, value=0.0, step=100.0)
min_turnover = st.sidebar.number_input("Min Turnover (Cr)", min_value=0.0, value=0.0, step=10.0)

sort_by = st.sidebar.selectbox(
    "Sort by",
    [
        "RS_Rank", "MS_RS", "RS_21D", "RS_55D", "RS_90D", "RS_123D",
        "Percent_Change", "Turnover_Cr", "Market_Cap_Cr", "LTP"
    ],
)

ascending = st.sidebar.checkbox("Ascending sort", value=(sort_by == "RS_Rank"))

top_n = st.sidebar.slider("Rows to display", 25, 500, 100)

filtered = df.copy()

if search:
    filtered = filtered[
        filtered["Symbol"].str.contains(search, case=False, na=False)
        | filtered["Name"].str.contains(search, case=False, na=False)
    ]

if selected_sector != "All" and "Sector" in filtered.columns:
    filtered = filtered[filtered["Sector"] == selected_sector]

if selected_industry != "All" and "Industry" in filtered.columns:
    filtered = filtered[filtered["Industry"] == selected_industry]

if "Market_Cap_Cr" in filtered.columns:
    filtered = filtered[filtered["Market_Cap_Cr"].fillna(0) >= min_mcap]

if "Turnover_Cr" in filtered.columns:
    filtered = filtered[filtered["Turnover_Cr"].fillna(0) >= min_turnover]

if sort_by in filtered.columns:
    filtered = filtered.sort_values(sort_by, ascending=ascending, na_position="last")

# -------- KPIs --------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Stocks", f"{len(filtered):,}")
with k2:
    if "Sector" in filtered.columns:
        st.metric("Sectors", filtered["Sector"].nunique())
with k3:
    if "Percent_Change" in filtered.columns:
        avg_change = filtered["Percent_Change"].dropna().mean()
        st.metric("Avg % Change", f"{avg_change:.2f}%")
with k4:
    if "MS_RS" in filtered.columns:
        avg_ms = filtered["MS_RS"].dropna().mean()
        st.metric("Avg MS-RS", f"{avg_ms:.2f}")

# -------- top sections --------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("🏆 RS Leaders")
    rs_cols = [c for c in ["Symbol", "Name", "Sector", "Industry", "RS_Rank", "MS_RS", "RS_21D", "RS_55D", "RS_90D", "RS_123D", "LTP", "Percent_Change"] if c in filtered.columns]
    st.dataframe(filtered[rs_cols].head(20), use_container_width=True, height=420)

with right:
    st.subheader("📈 Top Movers")
    mover_cols = [c for c in ["Symbol", "Name", "Sector", "LTP", "Percent_Change", "Turnover_Cr", "Volume_Today"] if c in filtered.columns]
    movers = filtered.sort_values("Percent_Change", ascending=False, na_position="last") if "Percent_Change" in filtered.columns else filtered
    st.dataframe(movers[mover_cols].head(20), use_container_width=True, height=420)

# -------- heatmap / treemap --------
if "Sector" in filtered.columns:
    st.subheader("🟩 Sector Heatmap")

    heat = filtered.copy()
    if "Market_Cap_Cr" not in heat.columns:
        heat["Market_Cap_Cr"] = 1
    if "Percent_Change" not in heat.columns:
        heat["Percent_Change"] = 0

    heat["Sector"] = heat["Sector"].replace("", "Unknown")
    heat["Name"] = heat["Name"].replace("", heat["Symbol"])

    fig = px.treemap(
        heat,
        path=["Sector", "Name"],
        values="Market_Cap_Cr",
        color="Percent_Change",
        hover_data=["Symbol", "LTP", "Percent_Change", "RS_55D", "MS_RS"],
    )
    fig.update_layout(margin=dict(t=30, l=10, r=10, b=10), height=650)
    st.plotly_chart(fig, use_container_width=True)

# -------- sector leaderboard --------
if "Sector" in filtered.columns:
    st.subheader("📊 Sector Leaderboard")
    group_cols = {}
    if "MS_RS" in filtered.columns:
        group_cols["MS_RS"] = "mean"
    if "RS_55D" in filtered.columns:
        group_cols["RS_55D"] = "mean"
    if "Percent_Change" in filtered.columns:
        group_cols["Percent_Change"] = "mean"
    if "Market_Cap_Cr" in filtered.columns:
        group_cols["Market_Cap_Cr"] = "sum"
    group_cols["Symbol"] = "count"

    sector_df = (
        filtered.groupby("Sector", dropna=False)
        .agg(group_cols)
        .rename(columns={"Symbol": "Stock_Count"})
        .reset_index()
    )
    sort_sector_col = "MS_RS" if "MS_RS" in sector_df.columns else ("RS_55D" if "RS_55D" in sector_df.columns else "Stock_Count")
    sector_df = sector_df.sort_values(sort_sector_col, ascending=False)

    st.dataframe(sector_df, use_container_width=True, height=320)

# -------- main table --------
st.subheader("📋 Full Screener")

display_cols = [
    "Symbol", "Name", "Sector", "Industry",
    "RS_Rank", "MS_RS", "RS_21D", "RS_55D", "RS_90D", "RS_123D",
    "LTP", "Percent_Change", "Market_Cap_Cr", "Turnover_Cr",
    "RSI_D", "RSI_W", "RSI_M",
    "ADX_D", "ADX_W", "ADX_M",
    "Above_200_EMA", "Above_Jan_High"
]
display_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(filtered[display_cols].head(top_n), use_container_width=True, height=600)

# -------- download --------
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇ Download filtered CSV",
    data=csv,
    file_name="dalal_street_scanner_filtered.csv",
    mime="text/csv"
)
