# pipeline.py (robust drop-in)
import sys
import pandas as pd
from datetime import timedelta

# Project imports (adjust names if your repo differs)
from load_data import load_klines
from features import build_features
from train import train_and_calibrate
from signal_engine import build_raw_signals, apply_execution_logic
from portfolio_backtest import backtest_portfolio

WF_TRAIN_DAYS = 90
WF_TEST_DAYS = 30
WF_STEP_DAYS = 30
INITIAL_CAPITAL = 10000.0


def walkforward_runner(
    feature_df,
    raw_price_df,
    run_train_and_calibrate,
    build_signals_fn,
    backtest_fn,
    wf_train_days: int = WF_TRAIN_DAYS,
    wf_test_days: int = WF_TEST_DAYS,
    wf_step_days: int = WF_STEP_DAYS,
    initial_capital: float = INITIAL_CAPITAL,
    verbose: bool = True,
):
    """
    Robust walk-forward runner:
      - coerces datetimes
      - deduplicates and/or aggregates duplicate timestamps
      - defends against empty/malformed signals
      - tolerates different backtest signatures
    """
    feature_df = feature_df.copy()
    raw_price_df = raw_price_df.copy()

    # Coerce indexes to datetimes
    feature_df.index = pd.to_datetime(feature_df.index, utc=True, errors="coerce")
    raw_price_df.index = pd.to_datetime(raw_price_df.index, utc=True, errors="coerce")

    # Drop rows with invalid timestamps
    feature_df = feature_df[feature_df.index.notna()]
    raw_price_df = raw_price_df[raw_price_df.index.notna()]

    if feature_df.empty:
        raise ValueError("feature_df has no valid datetime index after coercion. Check build_features output.")

    # Ensure unique indexes on both frames (keep last by default)
    if feature_df.index.duplicated().any():
        if verbose:
            print("walkforward_runner: feature_df index has duplicates; deduplicating by keeping last occurrence.")
        feature_df = feature_df[~feature_df.index.duplicated(keep="last")]

    if raw_price_df.index.duplicated().any():
        if verbose:
            print("walkforward_runner: raw_price_df index has duplicates; deduplicating by keeping last occurrence.")
        raw_price_df = raw_price_df[~raw_price_df.index.duplicated(keep="last")]

    # Align price frame to feature_df timestamps (safe)
    raw_price_df = raw_price_df.reindex(feature_df.index)

    start = feature_df.index.min()
    end = feature_df.index.max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("feature_df index min/max are NaT after coercion.")

    fold = 0
    all_trades = []
    fold_stats = []
    skipped_folds = []

    window_start = start
    while True:
        train_start = window_start
        train_end = train_start + pd.Timedelta(days=wf_train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=wf_test_days)

        if test_end > end:
            break

        fold += 1
        if verbose:
            print(f"\n=== Walkforward fold {fold}: train {train_start} → {train_end}, test {test_start} → {test_end}")

        train_df = feature_df.loc[train_start:train_end].copy()
        test_df = feature_df.loc[test_start:test_end].copy()

        # Ensure test_df index unique (defensive)
        if test_df.index.duplicated().any():
            if verbose:
                print(f"Fold {fold}: test_df index has duplicates; deduplicating by keeping first occurrence.")
            test_df = test_df[~test_df.index.duplicated(keep="first")]

        # Skip tiny windows
        if len(train_df) < 200 or len(test_df) < 10:
            if verbose:
                print(f"Fold {fold}: insufficient data (train {len(train_df)} rows, test {len(test_df)} rows). Skipping.")
            skipped_folds.append({"fold": fold, "reason": "insufficient_data", "train_rows": len(train_df), "test_rows": len(test_df)})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        # Train + calibrate
        try:
            models, feature_cols = run_train_and_calibrate(train_df)
        except Exception as e:
            if verbose:
                print(f"Fold {fold}: training failed with error: {e}. Skipping fold.")
            skipped_folds.append({"fold": fold, "reason": "train_error", "error": str(e)})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        # Build raw signals
        try:
            raw_signals = build_signals_fn(test_df, models, feature_cols)
        except Exception as e:
            if verbose:
                print(f"Fold {fold}: build_signals_fn raised error: {e}. Skipping fold.")
            skipped_folds.append({"fold": fold, "reason": "signal_build_error", "error": str(e)})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        if raw_signals is None or raw_signals.empty:
            if verbose:
                print(f"Fold {fold}: no raw signals generated (empty). Skipping fold.")
            skipped_folds.append({"fold": fold, "reason": "no_raw_signals"})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        expected_cols = {"timestamp", "symbol", "prob_long"}
        if not expected_cols.issubset(set(raw_signals.columns)):
            if verbose:
                print(f"Fold {fold}: raw_signals missing expected columns: {expected_cols - set(raw_signals.columns)}. Skipping fold.")
                print("raw_signals columns:", list(raw_signals.columns))
                print(raw_signals.head(5))
            skipped_folds.append({"fold": fold, "reason": "raw_signals_schema", "missing": list(expected_cols - set(raw_signals.columns))})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        # Apply execution logic
        try:
            signals = apply_execution_logic(raw_signals)
        except Exception as e:
            if verbose:
                print(f"Fold {fold}: apply_execution_logic raised error: {e}. Skipping fold.")
            skipped_folds.append({"fold": fold, "reason": "execution_error", "error": str(e)})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        if signals is None or signals.empty or "final_signal" not in signals.columns:
            if verbose:
                print(f"Fold {fold}: no executable signals after execution logic. Skipping fold.")
            skipped_folds.append({"fold": fold, "reason": "no_executable_signals"})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        # Build signal_configs for backtester (compact + time-aligned full arrays)
        signal_configs = {}

        # price_index we will align to (use test_df index; price_window built later)
        price_index = test_df.index

        for prefix, g in signals.groupby("symbol"):
            g = g.sort_values("timestamp")
            # compact arrays for backtester variants that expect event lists
            timestamps = g["timestamp"].values
            compact_final = g["final_signal"].values

            # Build a time-aligned full array (0/1) matching the test window index.
            # This covers backtest implementations that expect a per-timestep signal vector.
            full_series = pd.Series(0, index=price_index)
            # Only set ones for timestamps that exist in the price_index
            # Convert timestamps to the same tz/dtype as price_index for matching
            sig_ts = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
            # Intersect with price_index
            sig_ts = sig_ts[sig_ts.notna()]
            # Keep only timestamps present in price_index
            sig_ts = sig_ts[sig_ts.isin(price_index)]
            if not sig_ts.empty:
                full_series.loc[sig_ts] = 1

            signal_configs[prefix] = {
                "timestamps": timestamps,
                "final_signal": compact_final,
                # Add the time-aligned vector under the name your backtester complained about
                "full_final_signal": full_series.values,
            }

        # Prepare price window (safe reindexing; handle any remaining duplicates by aggregation)
        price_cols = [c for c in raw_price_df.columns if c.endswith("_Close")]
        try:
            price_window = raw_price_df[price_cols].reindex(test_df.index)
        except ValueError as e:
            # likely duplicate labels on raw_price_df.index; aggregate then reindex
            if verbose:
                print(f"Fold {fold}: reindex failed with ValueError: {e}. Aggregating duplicates (last) and retrying.")
            aggregated = raw_price_df[price_cols].groupby(raw_price_df.index).last()
            price_window = aggregated.reindex(test_df.index)

        # Run backtest for this fold (try both signatures)
        try:
            stats, trades = backtest_fn(price_window, signal_configs, initial_capital=initial_capital)
        except TypeError:
            try:
                stats, trades = backtest_fn(price_window, signal_configs)
            except Exception as e:
                if verbose:
                    print(f"Fold {fold}: backtest_fn failed: {e}. Skipping fold.")
                skipped_folds.append({"fold": fold, "reason": "backtest_error", "error": str(e)})
                window_start += pd.Timedelta(days=wf_step_days)
                continue
        except Exception as e:
            if verbose:
                print(f"Fold {fold}: backtest_fn failed: {e}. Skipping fold.")
            skipped_folds.append({"fold": fold, "reason": "backtest_error", "error": str(e)})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        if verbose:
            print(f"Fold {fold} stats:", stats)

        fold_stats.append(stats)
        if trades is None:
            trades = []
        elif isinstance(trades, pd.DataFrame):
            trades = trades.to_dict("records")
        all_trades.extend(trades)

        window_start += pd.Timedelta(days=wf_step_days)

    if verbose:
        print(f"\nWalkforward complete: folds run={len(fold_stats)}, folds skipped={len(skipped_folds)}")

    return fold_stats, all_trades


def single_split_run(feature_df, price_df):
    split_idx = int(len(feature_df) * 0.56)
    train_df = feature_df.iloc[:split_idx].copy()
    test_df = feature_df.iloc[split_idx:].copy()

    models, feature_cols = train_and_calibrate(train_df)
    raw_signals = build_raw_signals(test_df, models, feature_cols)
    signals = apply_execution_logic(raw_signals)

    if signals is None or signals.empty:
        print("single_split_run: no executable signals produced.")
        return {}, []

    signal_configs = {}
    for prefix, g in signals.groupby("symbol"):
        g = g.sort_values("timestamp")
        signal_configs[prefix] = {"timestamps": g["timestamp"].values, "final_signal": g["final_signal"].values}

    price_cols = [c for c in price_df.columns if c.endswith("_Close")]
    # ensure price_df index unique and aggregated
    if price_df.index.duplicated().any():
        price_df = price_df.groupby(price_df.index).last()
    price_window = price_df[price_cols].reindex(test_df.index)

    stats, trades = backtest_portfolio(price_window, signal_configs, initial_capital=INITIAL_CAPITAL)
    print("SINGLE SPLIT STATS:", stats)
    return stats, trades


def main(use_walkforward: bool = True):
    print("\n==============================\n STEP 1 — LOAD DATA\n==============================")
    df = load_klines()
    print(f"Loaded klines: {df.shape[0]} rows, {df.shape[1]} columns")

    # Normalize index to datetimes and diagnostics
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    bad_count = (~df.index.notna()).sum()
    if bad_count:
        print(f"Warning: dropped {bad_count} rows with invalid timestamps during index coercion.")
    df = df[df.index.notna()]
    if df.empty:
        raise ValueError("Loaded price frame has no valid timestamps after coercion.")

    # Deduplicate loaded price frame (keep last)
    if df.index.duplicated().any():
        print("main: loaded price frame has duplicate timestamps; deduplicating by keeping last occurrence.")
        df = df[~df.index.duplicated(keep="last")]

    print("index type:", type(df.index))
    print("index sample:", df.index[:10])
    print("min / max:", df.index.min(), df.index.max())

    print("\n==============================\n STEP 2 — BUILD FEATURES\n==============================")
    feature_df = build_features(df)
    print(f"Feature frame built: {feature_df.shape[0]} rows, {feature_df.shape[1]} columns")

    price_cols = [c for c in df.columns if c.endswith("_Close")]
    price_df = df[price_cols].copy()

    if use_walkforward:
        print("\n==============================\n WALKFORWARD RUN\n==============================")
        stats, trades = walkforward_runner(
            feature_df,
            price_df,
            train_and_calibrate,
            build_raw_signals,
            backtest_portfolio,
            wf_train_days=WF_TRAIN_DAYS,
            wf_test_days=WF_TEST_DAYS,
            wf_step_days=WF_STEP_DAYS,
            initial_capital=INITIAL_CAPITAL,
            verbose=True,
        )
        print("\nWALKFORWARD AGGREGATED STATS:", stats)
    else:
        print("\n==============================\n SINGLE SPLIT RUN\n==============================")
        stats, trades = single_split_run(feature_df, df)

    print("\n==============================\n PIPELINE COMPLETE\n==============================")
    return stats, trades


if __name__ == "__main__":
    use_wf = True
    main(use_walkforward=use_wf)
