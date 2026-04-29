import pandas as pd
from datetime import timedelta

TOP_N = 10
REBALANCE_DAYS = 30


def rolling_coin_selection(trades: list[dict]) -> list[dict]:
    """
    Build rolling selection periods based on per-coin expectancy over past window.
    Returns list of dicts:
        {
            "rebalance_time": timestamp,
            "approved": [symbols...]
        }
    """
    if not trades:
        return []

    df = pd.DataFrame(trades).copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])

    df = df.sort_values("exit_time")

    periods = []
    start = df["exit_time"].min().normalize()
    end = df["exit_time"].max().normalize()

    t = start + timedelta(days=REBALANCE_DAYS)

    while t <= end:
        window_start = t - timedelta(days=REBALANCE_DAYS)
        mask = (df["exit_time"] > window_start) & (df["exit_time"] <= t)
        window_trades = df[mask]

        if window_trades.empty:
            periods.append({"rebalance_time": t, "approved": []})
            t += timedelta(days=REBALANCE_DAYS)
            continue

        grouped = window_trades.groupby("symbol")["pnl_pct"].agg(
            trades="count",
            total_pnl="sum",
        )
        grouped["expectancy"] = grouped["total_pnl"] / grouped["trades"]
        grouped = grouped.sort_values("expectancy", ascending=False)

        approved = grouped.head(TOP_N).index.tolist()

        periods.append({"rebalance_time": t, "approved": approved})

        t += timedelta(days=REBALANCE_DAYS)

    return periods
