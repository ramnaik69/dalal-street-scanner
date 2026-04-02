# NSE Screener

Streamlit-based NSE stock screener with **90+ columns**, parallel data fetching, indicator stack, pivot matrix, and a Focus Screener strategy.

## Files

- `app.py` - Streamlit UI with All Stocks + Focus Screener tabs.
- `data_loader.py` - Parallel Yahoo Finance loader + fundamentals + RS calculations.
- `indicators.py` - EMA/SMA, RSI/ADX/MACD (D/W/M), pivots, returns, volume ratios.
- `screener.py` - Focus strategy filter.
- `nse_indices.py` - F&O and index flag mapping columns.
- `utils.py` - shared helpers.
- `data/symbols.csv` - symbol universe input.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Focus Screener Logic

A stock is selected when all conditions are true:

1. Within 15% of 52W High
2. Above 200 SMA
3. Above January High
4. RSI D > 50
5. ADX D > 20

## Notes

- Duplicate symbols are removed (`Symbol` uniqueness).
- Data is fetched in parallel for speed (`ThreadPoolExecutor`).
- Relative strength is computed vs Nifty (`^NSEI`) for 21/55/90/123 day windows.
