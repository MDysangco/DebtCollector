import pandas as pd

def build_backtest_signal_configs(signals_df: pd.DataFrame, price_window: pd.DataFrame):
    """
    Adapt execution signals into per-symbol configs for the portfolio backtest.

    Expected signals_df columns:
      - timestamp
      - symbol
      - final_signal (0/1)
      - prob_long (optional, for logging)
    """
    # ensure sorted
    signals_df = signals_df.sort_values(["timestamp", "symbol"]).copy()

    configs = {}

    for symbol, grp in signals_df.groupby("symbol"):
        # entry bars = 0 -> 1 flips
        grp = grp.sort_values("timestamp")
        sig = grp["final_signal"].fillna(0).astype(int)
        flips = (sig.shift(1, fill_value=0) == 0) & (sig == 1)
        entry_times = grp.loc[flips, "timestamp"].tolist()

        if not entry_times:
            continue

        configs[symbol] = {
            "symbol": symbol,
            "side": "LONG",
            "entries": entry_times,
            # optional: keep probs for diagnostics
            "entry_probs": grp.loc[flips, "prob_long"].tolist() if "prob_long" in grp.columns else None,
        }

    return configs
