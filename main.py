import pandas as pd

df = pd.read_csv("logs/walkforward/wf_fold_log.csv")
print({
  "folds": len(df),
  "mean_return_pct": df["fold_return_pct"].mean(),
  "median_return_pct": df["fold_return_pct"].median(),
  "pct_negative_folds": (df["fold_return_pct"] < 0).mean(),
  "worst_max_dd": df["max_dd"].min(),
  "avg_trades_per_fold": df["trades"].mean(),
  "median_avg_entry_prob": df["avg_entry_prob"].median()
})

ps = pd.read_csv("logs/walkforward/wf_per_symbol_log.csv")
summary = ps.groupby("symbol").agg(trades=("trades","sum"), avg_prob=("avg_prob","mean")).reset_index()
trouble = summary[(summary["trades"]>=100) & (summary["avg_prob"]<0.515)].sort_values(["trades","avg_prob"], ascending=[False,True])
print(trouble.to_csv(index=False))