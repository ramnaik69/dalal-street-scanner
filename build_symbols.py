import pandas as pd

RAW_FILE = "nse_symbols_raw.txt"
OUT_FILE = "symbols.csv"

def to_yahoo_symbol(sym: str) -> str:
    return f"{sym}.NS"

def main():
    rows = []
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    # skip header
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        symbol = parts[0].strip()
        series = parts[-1].strip()

        rows.append({
            "Symbol": symbol,
            "YahooSymbol": to_yahoo_symbol(symbol),
            "Name": symbol,
            "Exchange": "NSE",
            "Series": series
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["Symbol", "Series"])

    # recommended default: only EQ in main file
    eq_df = df[df["Series"] == "EQ"].copy()

    eq_df.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(eq_df)} EQ symbols to {OUT_FILE}")

    # optional full file
    df.to_csv("symbols_all_series.csv", index=False)
    print(f"Saved {len(df)} all-series symbols to symbols_all_series.csv")

if __name__ == "__main__":
    main()
