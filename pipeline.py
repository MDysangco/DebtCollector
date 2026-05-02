# pipeline.py
from load_data import load_klines
from features import build_features
from signal_adapter import build_backtest_signal_configs
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
    def debug(msg):
        print(f"[WF DEBUG] {msg}")

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

    feature_df = feature_df.copy()
    raw_price_df = raw_price_df.copy()

    feature_df.index = pd.to_datetime(feature_df.index, utc=True, errors="coerce")
    raw_price_df.index = pd.to_datetime(raw_price_df.index, utc=True, errors="coerce")

    feature_df = feature_df[feature_df.index.notna()]
    raw_price_df = raw_price_df[raw_price_df.index.notna()]

    if feature_df.index.duplicated().any():
        debug("Deduplicating feature_df index")
        feature_df = feature_df[~feature_df.index.duplicated(keep="last")]

    if raw_price_df.index.duplicated().any():
        debug("Deduplicating raw_price_df index")
        raw_price_df = raw_price_df[~raw_price_df.index.duplicated(keep="last")]

    raw_price_df = raw_price_df.reindex(feature_df.index)

    start = feature_df.index.min()
    end = feature_df.index.max()

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
            debug("Stopping: test_end beyond data range")
            break

        fold += 1
        if max_folds is not None and fold > max_folds:
            debug("Stopping: max_folds reached")
            break

        print(f"\n=== Walkforward fold {fold}: train {train_start} → {train_end}, test {test_start} → {test_end}")

        # -------------------------------
        # TRAIN WINDOW
        # -------------------------------
        train_df = feature_df.loc[train_start:train_end]
        debug(f"Train window rows: {len(train_df)}")

        # -------------------------------
        # TEST WINDOW WITH WARM-UP
        # -------------------------------
        warmup_hours = 100
        test_df_full = feature_df.loc[test_start - pd.Timedelta(hours=warmup_hours): test_end]

        if test_df_full.index.duplicated().any():
            debug("Deduplicating test_df_full index")
            test_df_full = test_df_full[~test_df_full.index.duplicated(keep="first")]

        test_df_live = test_df_full.loc[test_start:test_end]
        debug(f"Test window rows: {len(test_df_live)} (full={len(test_df_full)})")

        if len(train_df) < 200 or len(test_df_live) < 10:
            skipped = {"fold": fold, "reason": "insufficient_data"}
            skipped_folds.append(skipped)
            debug(f"SKIPPED: {skipped}")
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        # -------------------------------
        # TRAIN MODEL
        # -------------------------------
        try:
            models, feature_cols, thresholds, medians = run_train_and_calibrate(train_df)
            debug(f"Training OK: models={len(models)}, thresholds={len(thresholds)}")
        except Exception as e:
            skipped = {"fold": fold, "reason": "train_error", "error": str(e)}
            skipped_folds.append(skipped)
            debug(f"SKIPPED: {skipped}")
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        # -------------------------------
        # BUILD RAW SIGNALS
        # -------------------------------
        try:
            raw_signals = build_signals_fn(
                test_df=test_df_full,
                models=models,
                feature_cols=feature_cols,
                medians=medians,
            )
            debug(f"Raw signals: {None if raw_signals is None else raw_signals.shape}")
        except Exception as e:
            skipped = {"fold": fold, "reason": "signal_build_error", "error": str(e)}
            skipped_folds.append(skipped)
            debug(f"SKIPPED: {skipped}")
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        if raw_signals is None or raw_signals.empty:
            skipped = {"fold": fold, "reason": "no_raw_signals"}
            skipped_folds.append(skipped)
            debug(f"SKIPPED: {skipped}")
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        # -------------------------------
        # EXECUTION LOGIC
        # -------------------------------
        try:
            signals = apply_execution_logic(
                raw_signals,
                test_df_full,
                global_threshold=config.GLOBAL_THRESHOLD,
                per_symbol_floor=config.PER_SYMBOL_FLOOR,
                margin=config.MARGIN,
                cooldown_hours=config.COOLDOWN_HOURS,
                thresholds=thresholds,
            )
            debug(f"Signals after execution: {signals.shape}")
            signals = signals[signals["timestamp"] >= test_start]
            debug(f"Signals after live filter: {signals.shape}")
        except Exception as e:
            skipped = {"fold": fold, "reason": "execution_error", "error": str(e)}
            skipped_folds.append(skipped)
            debug(f"SKIPPED: {skipped}")
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        if signals.empty:
            skipped = {"fold": fold, "reason": "no_executable_signals"}
            skipped_folds.append(skipped)
            debug(f"SKIPPED: {skipped}")
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        # -------------------------------
        # BACKTEST
        # -------------------------------
        price_cols = [c for c in raw_price_df.columns if c.endswith("_Close")]
        price_window = raw_price_df[price_cols].reindex(test_df_live.index)
        debug(f"Backtest price window rows: {len(price_window)}")

        try:

            print("[WF DEBUG] signals columns:", signals.columns.tolist())
            print(signals.head())

            signal_configs = build_backtest_signal_configs(signals, price_window)
            stats, trades = backtest_portfolio(price_window, signal_configs)

            debug("Backtest OK — writing logs")

        except Exception as e:
            skipped = {"fold": fold, "reason": "backtest_error", "error": str(e)}
            skipped_folds.append(skipped)
            debug(f"SKIPPED: {skipped}")
            window_start += pd.Timedelta(days=wf_step_days)
            continue

        # -------------------------------
        # CSV LOGGING
        # -------------------------------

        fold_row = {
            "fold": fold,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fold_return_pct": stats.get("return_pct") if isinstance(stats, dict) else None,
            "max_dd": stats.get("max_dd") if isinstance(stats, dict) else None,
            # OLD:
            # "trades": len(signals),
            # NEW:
            "trades": len(trades) if isinstance(trades, list) else None,
            "avg_entry_prob": float(signals["prob_long"].mean()) if len(signals) else None,
        }

        fold_csv = os.path.join(LOG_DIR, f"wf_fold_log_{date_str}.csv")
        df_fold = pd.DataFrame([fold_row])
        df_fold.to_csv(fold_csv, mode="a", header=not os.path.exists(fold_csv), index=False)

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

        per_symbol_csv = os.path.join(LOG_DIR, f"wf_per_symbol_log_{date_str}.csv")
        per_symbol.to_csv(per_symbol_csv, mode="a", header=not os.path.exists(per_symbol_csv), index=False)

        # -------------------------------
        # TRADE LEDGER LOGGING
        # -------------------------------
        if isinstance(trades, list) and len(trades) > 0:
            df_trades = pd.DataFrame(trades)
            df_trades["fold"] = fold
            df_trades["train_start"] = train_start
            df_trades["train_end"] = train_end
            df_trades["test_start"] = test_start
            df_trades["test_end"] = test_end

            trade_ledger_csv = os.path.join(LOG_DIR, f"wf_trade_ledger_{date_str}.csv")
            df_trades.to_csv(
                trade_ledger_csv,
                mode="a",
                header=not os.path.exists(trade_ledger_csv),
                index=False
            )

        # -------------------------------
        # SAVE FOLD RESULTS
        # -------------------------------
        fold_stats.append(stats)
        all_trades.extend(trades if isinstance(trades, list) else [])

        window_start += pd.Timedelta(days=wf_step_days)

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

    # correct price frame
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
        fold_stats, trades = walkforward_runner(
            feature_df,
            price_df,                 # FIXED
            train_and_calibrate,      # correct training function
            build_raw_signals,        # FIXED
            backtest_portfolio,       # FIXED
            wf_train_days=WF_TRAIN_DAYS,
            wf_test_days=WF_TEST_DAYS,
            wf_step_days=WF_STEP_DAYS,
            initial_capital=INITIAL_CAPITAL,
            verbose=True,
            max_folds=config.MAX_FOLDS,
        )

        print("\nWALKFORWARD AGGREGATED STATS:", fold_stats)

    else:
        print("\n==============================\n SINGLE SPLIT RUN\n==============================")
        stats, trades = {}, []
        # single split omitted

    print("\n==============================\n PIPELINE COMPLETE\n==============================")
    return fold_stats, trades

if __name__ == "__main__":
    use_wf = True
    main(use_walkforward=use_wf)
