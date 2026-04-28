from load_data import load_klines
from features import build_feature_frame, validate_feature_frame
from train import train_multi_horizon_models
from signal_engine import run_full_signal_engine
from portfolio_backtest import backtest_portfolio
from portfolio_report import print_portfolio_report, rolling_coin_selection


def main():

    print("\n==============================")
    print(" STEP 1 — LOAD DATA")
    print("==============================")

    df = load_klines()
    print(f"Loaded klines: {len(df):,} rows, {len(df.columns):,} columns")

    print("\n==============================")
    print(" STEP 2 — BUILD FEATURES")
    print("==============================")

    feature_df, feature_cols, label_cols = build_feature_frame(df)
    print(f"Feature frame built: {len(feature_df):,} rows, {len(feature_df.columns):,} columns")

    print("\n==============================")
    print(" STEP 3 — VALIDATE FEATURES")
    print("==============================")

    validate_feature_frame(feature_df)

    print("\n==============================")
    print(" STEP 4 — TRAIN / TEST SPLIT")
    print("==============================")

    split_date = "2023-01-01"
    train_df = feature_df.loc[:split_date]
    test_df = feature_df.loc[split_date:]

    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows:  {len(test_df):,}")

    print("\n==============================")
    print(" STEP 5 — TRAIN MODELS")
    print("==============================")

    models = train_multi_horizon_models(train_df, feature_cols)
    print(f"Trained {len(models)} models")

    print("\n==============================")
    print(" STEP 6 — GENERATE SIGNALS")
    print("==============================")

    signals = run_full_signal_engine(feature_df, models, feature_cols)

    print("\n==============================")
    print(" STEP 7 — PORTFOLIO BACKTEST (ALL COINS)")
    print("==============================")

    stats, trades = backtest_portfolio(test_df, signals)
    print_portfolio_report(stats, trades)

    # -----------------------------------------
    # STEP 8 — COIN FILTERING
    # -----------------------------------------
    print("\n==============================")
    print(" STEP 8 — ROLLING COIN SELECTION")
    print("==============================")

    if len(trades) == 0:
        print("No trades generated — skipping dynamic selection.")
        print("\n==============================")
        print(" PIPELINE COMPLETE")
        print("==============================\n")
        return

    periods = rolling_coin_selection(trades)

    for p in periods:
        print(f"{p['rebalance_time']} → {p['approved']}")

    import numpy as np

    test_index = test_df.index
    filtered_signal_configs = {}

    for prefix, cfg in signals.items():

        # Slice signal to test period only
        sig_full = cfg["full_final_signal"]
        sig = sig_full[feature_df.index >= split_date].copy()

        # start with everything disallowed
        allowed_mask = np.zeros(len(test_index), dtype=bool)

        # for each period, allow this prefix if it's approved
        for p in periods:
            t = p["rebalance_time"]
            approved = p["approved"]

            if prefix not in approved:
                continue

            # from this rebalance time onward, this coin is allowed
            period_mask = test_index >= t
            allowed_mask |= period_mask

        # final signal: only keep where allowed_mask is True
        sig[~allowed_mask] = 0

        filtered_signal_configs[prefix] = {"full_final_signal": sig}

    print("\n==============================")
    print(" STEP 9 — PORTFOLIO BACKTEST (DYNAMIC)")
    print("==============================")

    stats3, trades3 = backtest_portfolio(test_df, filtered_signal_configs)
    print_portfolio_report(stats3, trades3)

    print("\n==============================")
    print(" PIPELINE COMPLETE")
    print("==============================\n")


if __name__ == "__main__":
    main()
