import pandas as pd
import requests
import zipfile
import io
from datetime import datetime


def get_today_bhavcopy():
    today = datetime.now()

    date_str = today.strftime("%d%m%Y")
    url = f"https://archives.nseindia.com/content/historical/EQUITIES/{today.strftime('%Y')}/{today.strftime('%b').upper()}/cm{date_str}bhav.csv.zip"

    try:
        r = requests.get(url, timeout=10)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]))
        return df
    except:
        return pd.DataFrame()


def process_bhavcopy(df):
    if df.empty:
        return df

    df = df[df["SERIES"] == "EQ"]

    df["TURNOVER_LACS"] = df["TOTTRDVAL"] / 100000

    df.rename(columns={
        "SYMBOL": "Symbol",
        "CLOSE": "CLOSE_PRICE",
        "TOTTRDQTY": "TTL_TRD_QNTY",
        "TOTALTRADES": "NO_OF_TRADES",
        "DELIV_QTY": "DELIV_QTY",
        "DELIV_PER": "DELIV_PER",
        "VWAP": "AVG_PRICE"
    }, inplace=True)

    return df[[
        "Symbol",
        "CLOSE_PRICE",
        "AVG_PRICE",
        "TTL_TRD_QNTY",
        "TURNOVER_LACS",
        "NO_OF_TRADES",
        "DELIV_QTY",
        "DELIV_PER"
    ]]
