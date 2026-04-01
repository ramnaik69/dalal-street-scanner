from nse_metrics import aggregate_weekly, aggregate_monthly, latest_period_snapshot

# daily_metrics must contain:
# Symbol, Date, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER

weekly_metrics = aggregate_weekly(daily_metrics)
monthly_metrics = aggregate_monthly(daily_metrics)

daily_snap = latest_period_snapshot(daily_metrics, "D")
weekly_snap = latest_period_snapshot(weekly_metrics, "W")
monthly_snap = latest_period_snapshot(monthly_metrics, "M")

market_metrics_summary = daily_snap.merge(weekly_snap, on="Symbol", how="left").merge(monthly_snap, on="Symbol", how="left")

summary = summary.merge(market_metrics_summary, on="Symbol", how="left")
