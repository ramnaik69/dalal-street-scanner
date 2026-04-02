from data_fetcher import fetch_market_data

def main():
    raw_df, summary_df, index_df, meta = fetch_market_data(force_refresh=True)
    print("Update completed")
    print(meta)

if __name__ == "__main__":
    main()
