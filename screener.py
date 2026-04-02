from __future__ import annotations

import pandas as pd


FOCUS_CONDITIONS = [
    "15% Range",
    "Above 200 SMA",
    "Above Jan High",
]


def build_focus_screener(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out = out[
        out["15% Range"].fillna(False)
        & out["Above 200 SMA"].fillna(False)
        & out["Above Jan High"].fillna(False)
        & (out["RSI D"].fillna(0) > 50)
        & (out["ADX D"].fillna(0) > 20)
    ]
    return out.sort_values(by=["RSI D", "ADX D"], ascending=False).reset_index(drop=True)
