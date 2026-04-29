import pandas as pd


def per_coin_report(trades: list[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades).copy()

    grouped = df.groupby("symbol")["pnl_pct"].agg(
        trades="count",
        avg_pnl="mean",
        med_pnl="median",
        win_rate=lambda x: (x > 0).mean(),
        loss_rate=lambda x: (x < 0).mean(),
        best_trade="max",
        worst_trade="min",
        total_pnl="sum",
    )

    grouped["expectancy"] = grouped["total_pnl"] / grouped["trades"]

    return grouped.sort_values("expectancy", ascending=False).reset_index()
