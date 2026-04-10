import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dalal Street Scanner", layout="wide")

st.title("Dalal Street Scanner")

df = pd.read_csv("data/final_3125_master_with_metrics_template.csv")

st.write(f"Total stocks: {len(df)}")

search = st.text_input("Search Symbol or Company Name")

if search:
    df = df[
        df["Symbol"].astype(str).str.contains(search, case=False, na=False) |
        df["Company_Name"].astype(str).str.contains(search, case=False, na=False)
    ]

if "Sector" in df.columns:
    sectors = ["All"] + sorted(df["Sector"].dropna().astype(str).unique().tolist())
    selected_sector = st.selectbox("Sector", sectors)
    if selected_sector != "All":
        df = df[df["Sector"].astype(str) == selected_sector]

st.dataframe(df, use_container_width=True)
