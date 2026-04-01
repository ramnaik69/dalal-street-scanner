import pandas as pd
import numpy as np

def add_delivery_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DELIV_PER"] = np.where(
        df["TTL_TRD_QNTY"] > 0,
        (df["DELIV_QTY"] / df["TTL_TRD_QNTY"]) * 100.0,
        np.nan
    )
    return df

def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["Date"] = pd.to_datetime(x["Date"])
    x = x.sort_values(["Symbol", "Date"])

    out = (
        x.groupby(["Symbol", pd.Grouper(key="Date", freq="W-FRI")])
         .agg(
             CLOSE_PRICE=("CLOSE_PRICE", "last"),
             AVG_PRICE=("AVG_PRICE", "mean"),
             TTL_TRD_QNTY=("TTL_TRD_QNTY", "sum"),
             TURNOVER_LACS=("TURNOVER_LACS", "sum"),
             NO_OF_TRADES=("NO_OF_TRADES", "sum"),
             DELIV_QTY=("DELIV_QTY", "sum"),
         )
         .reset_index()
    )
    out = add_delivery_pct(out)
    return out

def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["Date"] = pd.to_datetime(x["Date"])
    x = x.sort_values(["Symbol", "Date"])

    out = (
        x.groupby(["Symbol", pd.Grouper(key="Date", freq="M")])
         .agg(
             CLOSE_PRICE=("CLOSE_PRICE", "last"),
             AVG_PRICE=("AVG_PRICE", "mean"),
             TTL_TRD_QNTY=("TTL_TRD_QNTY", "sum"),
             TURNOVER_LACS=("TURNOVER_LACS", "sum"),
             NO_OF_TRADES=("NO_OF_TRADES", "sum"),
             DELIV_QTY=("DELIV_QTY", "sum"),
         )
         .reset_index()
    )
    out = add_delivery_pct(out)
    return out

def latest_period_snapshot(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    x = df.sort_values(["Symbol", "Date"]).groupby("Symbol", as_index=False).tail(1).copy()

    rename_map = {
        "CLOSE_PRICE": f"CLOSE_PRICE_{suffix}",
        "AVG_PRICE": f"AVG_PRICE_{suffix}",
        "TTL_TRD_QNTY": f"TTL_TRD_QNTY_{suffix}",
        "TURNOVER_LACS": f"TURNOVER_LACS_{suffix}",
        "NO_OF_TRADES": f"NO_OF_TRADES_{suffix}",
        "DELIV_QTY": f"DELIV_QTY_{suffix}",
        "DELIV_PER": f"DELIV_PER_{suffix}",
        "Date": f"DATE_{suffix}",
    }
    return x.rename(columns=rename_map)
