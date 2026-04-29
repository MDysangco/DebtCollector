# pipeline.py
from load_data import load_klines
from features import build_features
from train import train_and_calibrate
from signal_engine import build_raw_signals, apply_execution_logic
from portfolio_backtest import backtest_portfolio
import config

import os
import pandas as pd
from datetime import datetime, timezone
import tempfile

WF_TRAIN_DAYS = config.WF_TRAIN_DAYS
WF_TEST_DAYS = config.WF_TEST_DAYS
WF_STEP_DAYS = config.WF_STEP_DAYS
max_folds = config.MAX_FOLDS
INITIAL_CAPITAL = 10000.0

LOG_DIR = "logs/walkforward"
os.makedirs(LOG_DIR, exist_ok=True)


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
    max_folds: int = None,
):
    feature_df = feature_df.copy()
    raw_price_df = raw_price_df.copy()

    feature_df.index = pd.to_datetime(feature_df.index, utc=True, errors="coerce")
    raw_price_df.index = pd.to_datetime(raw_price_df.index, utc=True, errors="coerce")

    feature_df = feature_df[feature_df.index.notna()]
    raw_price_df = raw_price_df[raw_price_df.index.notna()]

    if feature_df.index.duplicated().any():
        if verbose:
            print("walkforward_runner: feature_df index has duplicates; deduplicating by keeping last.")
        feature_df = feature_df[~feature_df.index.duplicated(keep="last")]

    if raw_price_df.index.duplicated().any():
        if verbose:
            print("walkforward_runner: raw_price_df index has duplicates; deduplicating by keeping last.")
        raw_price_df = raw_price_df[~raw_price_df.index.duplicated(keep="last")]

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

        if max_folds is not None and fold > max_folds:
            break

        if verbose:
            print(f"\n=== Walkforward fold {fold}: train {train_start} → {train_end}, test {test_start} → {test_end}")

        train_df = feature_df.loc[train_start:train_end].copy()
        test_df = feature_df.loc[test_start:test_end].copy()

        if test_df.index.duplicated().any():
            if verbose:
                print(f"Fold {fold}: test_df index has duplicates; deduplicating by keeping first.")
            test_df = test_df[~test_df.index.duplicated(keep="first")]

        if len(train_df) < 200 or len(test_df) < 10:
            if verbose:
                print(f"Fold {fold}: insufficient data (train {len(train_df)} rows, test {len(test_df)} rows). Skipping.")
            skipped_folds.append({"fold": fold, "reason": "insufficient_data", "train_rows": len(train_df), "test_rows": len(test_df)})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        try:
            models, feature_cols, thresholds, medians = run_train_and_calibrate(train_df)
        except Exception as e:
            if verbose:
                print(f"Fold {fold}: training failed: {e}. Skipping.")
            skipped_folds.append({"fold": fold, "reason": "train_error", "error": str(e)})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        try:
            raw_signals = build_signals_fn(test_df, models, feature_cols, medians)
        except Exception as e:
            if verbose:
                print(f"Fold {fold}: build_signals_fn error: {e}. Skipping.")
            skipped_folds.append({"fold": fold, "reason": "signal_build_error", "error": str(e)})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        if verbose:
            print(f"Fold {fold}: raw_signals shape:", getattr(raw_signals, "shape", None))
            if isinstance(raw_signals, pd.DataFrame) and "prob_long" in raw_signals.columns:
                print(f"Fold {fold}: raw_signals prob_long stats:\n", raw_signals["prob_long"].describe())

        if raw_signals is None or raw_signals.empty:
            if verbose:
                print(f"Fold {fold}: no raw signals generated (empty). Skipping fold.")
            skipped_folds.append({"fold": fold, "reason": "no_raw_signals"})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        expected_cols = {"timestamp", "symbol", "prob_long"}
        if not expected_cols.issubset(set(raw_signals.columns)):
            if verbose:
                print(f"Fold {fold}: raw_signals missing columns: {expected_cols - set(raw_signals.columns)}. Skipping.")
                print("raw_signals columns:", list(raw_signals.columns))
                print(raw_signals.head(5))
            skipped_folds.append({"fold": fold, "reason": "raw_signals_schema", "missing": list(expected_cols - set(raw_signals.columns))})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        try:
            # use config values for execution logic
            signals = apply_execution_logic(
                raw_signals,
                thresholds=thresholds,
                global_threshold=config.GLOBAL_THRESHOLD,
                cooldown_hours=config.COOLDOWN_HOURS,
            )
        except Exception as e:
            if verbose:
                print(f"Fold {fold}: apply_execution_logic error: {e}. Skipping.")
            skipped_folds.append({"fold": fold, "reason": "execution_error", "error": str(e)})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        if verbose:
            print(f"Fold {fold}: signals shape:", getattr(signals, "shape", None))
            if isinstance(signals, pd.DataFrame) and not signals.empty:
                print(f"Fold {fold}: signals sample:\n", signals.head(5))

        if signals is None or signals.empty or "final_signal" not in signals.columns:
            if verbose:
                print(f"Fold {fold}: no executable signals after execution logic. Skipping fold.")
            skipped_folds.append({"fold": fold, "reason": "no_executable_signals"})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        # Build signal_configs (compact + time-aligned full arrays)
        signal_configs = {}
        price_index = test_df.index
        for prefix, g in signals.groupby("symbol"):
            g = g.sort_values("timestamp")
            timestamps = g["timestamp"].values
            compact_final = g["final_signal"].values

            full_series = pd.Series(0, index=price_index)
            sig_ts = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
            sig_ts = sig_ts[sig_ts.notna()]
            sig_ts = sig_ts[sig_ts.isin(price_index)]
            if not sig_ts.empty:
                full_series.loc[sig_ts] = 1

            signal_configs[prefix] = {
                "timestamps": timestamps,
                "final_signal": compact_final,
                "full_final_signal": full_series.values,
            }

        price_cols = [c for c in raw_price_df.columns if c.endswith("_Close")]
        try:
            price_window = raw_price_df[price_cols].reindex(test_df.index)
        except ValueError as e:
            if verbose:
                print(f"Fold {fold}: price reindex ValueError: {e}. Aggregating duplicates and retrying.")
            aggregated = raw_price_df[price_cols].groupby(raw_price_df.index).last()
            price_window = aggregated.reindex(test_df.index)

        try:
            stats, trades = backtest_fn(price_window, signal_configs, initial_capital=initial_capital)
        except TypeError:
            try:
                stats, trades = backtest_fn(price_window, signal_configs)
            except Exception as e:
                if verbose:
                    print(f"Fold {fold}: backtest_fn failed: {e}. Skipping.")
                skipped_folds.append({"fold": fold, "reason": "backtest_error", "error": str(e)})
                window_start += pd.Timedelta(days=wf_step_days)
                continue
        except Exception as e:
            if verbose:
                print(f"Fold {fold}: backtest_fn failed: {e}. Skipping.")
            skipped_folds.append({"fold": fold, "reason": "backtest_error", "error": str(e)})
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        if verbose:
            print(f"Fold {fold} stats:", stats)

        # --- Persist fold-level summary to CSV (append) ---
        fold_row = {
            "fold": fold,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fold_return_pct": stats.get("return_pct") if isinstance(stats, dict) else None,
            "max_dd": stats.get("max_dd") if isinstance(stats, dict) else None,
            "trades": len(signals),
            "avg_entry_prob": float(signals["prob_long"].mean()) if len(signals) else None,
        }

        fold_csv = os.path.join(LOG_DIR, "wf_fold_log.csv")
        df_fold = pd.DataFrame([fold_row])
        # append safely: write header only if file doesn't exist
        if not os.path.exists(fold_csv):
            df_fold.to_csv(fold_csv, index=False)
        else:
            df_fold.to_csv(fold_csv, mode="a", header=False, index=False)

        # --- Persist per-symbol summary to CSV (append) ---
        if len(signals):
            per_symbol = (
                signals.groupby("symbol")
                .agg(trades=("symbol", "size"), avg_prob=("prob_long", "mean"))
                .reset_index()
            )
            per_symbol["fold"] = fold
            per_symbol["train_start"] = train_start
            per_symbol["train_end"] = train_end
            per_symbol["test_start"] = test_start
            per_symbol["test_end"] = test_end

            per_symbol_csv = os.path.join(LOG_DIR, "wf_per_symbol_log.csv")
            if not os.path.exists(per_symbol_csv):
                per_symbol.to_csv(per_symbol_csv, index=False)
            else:
                per_symbol.to_csv(per_symbol_csv, mode="a", header=False, index=False)

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


def main(use_walkforward: bool = True):
    print("\n==============================\n STEP 1 — LOAD DATA\n==============================")
    df = load_klines()
    print(f"Loaded klines: {df.shape[0]} rows, {df.shape[1]} columns")

    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    bad_count = (~df.index.notna()).sum()
    if bad_count:
        print(f"Warning: dropped {bad_count} rows with invalid timestamps during index coercion.")
    df = df[df.index.notna()]
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

        if config.PRINT_SETTINGS:
            run_settings = {
                "margin": config.MARGIN,
                "global_threshold": config.GLOBAL_THRESHOLD,
                "per_symbol_floor": config.PER_SYMBOL_FLOOR,
                "cooldown_hours": config.COOLDOWN_HOURS,
                "WF_TRAIN_DAYS": config.WF_TRAIN_DAYS,
                "WF_TEST_DAYS": config.WF_TEST_DAYS,
                "WF_STEP_DAYS": config.WF_STEP_DAYS,
                "max_folds": config.MAX_FOLDS,
                "imputation": config.IMPUTATION_METHOD,
            }

            for k, v in run_settings.items():
                print(f"{k:20s}: {v}")

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
            max_folds=config.MAX_FOLDS
        )
        print("\nWALKFORWARD AGGREGATED STATS:", stats)
    else:
        print("\n==============================\n SINGLE SPLIT RUN\n==============================")
        stats, trades = {}, []
        # single_split_run omitted for brevity; use walkforward for now

    print("\n==============================\n PIPELINE COMPLETE\n==============================")
    return stats, trades


if __name__ == "__main__":
    use_wf = True
    main(use_walkforward=use_wf)
