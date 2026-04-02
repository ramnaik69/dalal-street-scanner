from __future__ import annotations

INDEX_MEMBERS = {
    "NIFTY_50": {
        "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK", "INFY", "SBIN",
        "ITC", "HINDUNILVR", "LT", "KOTAKBANK", "AXISBANK", "BAJFINANCE", "ASIANPAINT",
        "MARUTI", "SUNPHARMA", "ULTRACEMCO", "TITAN", "NESTLEIND", "NTPC",
    },
    "NIFTY_NEXT_50": {"DMART", "ADANIENT", "HAVELLS", "HAL", "DABUR", "PIDILITIND"},
    "NIFTY_MIDCAP_100": {"BHEL", "IRCTC", "POLYCAB", "MPHASIS", "LUPIN", "COFORGE"},
    "NIFTY_SMALLCAP_100": {"CDSL", "BLS", "CAMS", "KPRMILL", "CLEAN"},
    "NIFTY_BANK": {"HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK"},
}

FNO_SYMBOLS = {
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "LT", "KOTAKBANK",
    "AXISBANK", "BAJFINANCE", "ADANIENT", "SUNPHARMA", "HINDUNILVR", "MARUTI", "ITC",
}


def membership_flags(symbol: str) -> tuple[dict[str, bool], str]:
    symbol = (symbol or "").upper().strip()
    flags = {f"IDX_{name}": symbol in members for name, members in INDEX_MEMBERS.items()}
    labels = [name for name, members in INDEX_MEMBERS.items() if symbol in members]
    return flags, "|".join(labels)


def is_fno_symbol(symbol: str) -> bool:
    return (symbol or "").upper().strip() in FNO_SYMBOLS
