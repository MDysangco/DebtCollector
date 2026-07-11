"""Walk-forward A/B comparison: Model 1 vs Model 2 on identical folds.

All models are scored on the same yardstick — the realized LABEL_HORIZON-bar
forward return — at three levels:

  gated:  BUY/SELL only when the class probability clears the production
          threshold (what the live pipeline would actually trade), plus the
          existing portfolio backtest on those signals.
  rank_ic: Spearman correlation between the directional score
          (p_buy - p_sell) and the forward return. Threshold-free measure
          of how well a model separates ups from downs.
  decile: mean forward return of the top-10% scores minus the bottom-10%.
          Compares both models on the same number of best-conviction calls.

Training rows within one label horizon of the train/test boundary are purged
for every model so no label peeks into the test window.

Usage:
    python compare_models.py                  # reuses price_cache.parquet if present
    python compare_models.py --refresh        # force re-download of price data
    python compare_models.py --rescore        # re-score stored predictions, no retraining
    python compare_models.py --train-days 180 # longer training window (own predictions file)
"""
import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ZypryxApi import ZypryxApi
from market_data import load_coin_ids, load_price_data_from_api
from features_and_labels import merge_features_and_labels, detect_symbols
from features_v2 import (
    build_features_v2, merge_features_and_labels_v2, merge_features_and_target_v2,
    LEVEL_COLS, EXTERNAL_COLS, CALENDAR_COLS, REVERSAL_COLS,
)
from model_train import train_model, time_split
from model_train_v2 import train_model_v2, train_regressor_v2
from backtest_portfolio import backtest_portfolio

import config
from config import (
    WF_TRAIN_DAYS, WF_TEST_DAYS, WF_STEP_DAYS, MAX_FOLDS,
    LABEL_HORIZON, TRAIN_SPLIT_Q, COIN_IDS, INTERVAL_ID,
)

PRICE_CACHE = Path(__file__).resolve().parent / "price_cache.parquet"


def predictions_file(train_days):
    """One predictions store per training-window length; fold boundaries
    change with the window, so their predictions must not be mixed."""
    suffix = "" if train_days == WF_TRAIN_DAYS else f"_{train_days}d"
    return Path(__file__).resolve().parent / f"predictions{suffix}.parquet"

# Model 2 ablation variants: which feature columns to drop, and how hard to
# rebalance classes. Pre-reversal variants drop REVERSAL_COLS so their stored
# predictions stay reproducible; *_rev variants include the reversal features.
V2_VARIANTS = {
    "model2": {"drop": REVERSAL_COLS, "soften": True, "kind": "clf"},
    "model2_hard": {"drop": REVERSAL_COLS, "soften": False, "kind": "clf"},
    "model2_chg": {"drop": LEVEL_COLS + REVERSAL_COLS, "soften": True, "kind": "clf"},
    "model2_price": {"drop": EXTERNAL_COLS + CALENDAR_COLS + REVERSAL_COLS,
                     "soften": True, "kind": "clf"},
    # Regression on the vol-scaled forward return. Predictions land in the
    # p_buy column (and mirrored negative in p_sell) so the shared scoring
    # applies: score = p_buy - p_sell = 2 * prediction.
    "model2_reg": {"drop": EXTERNAL_COLS + CALENDAR_COLS + REVERSAL_COLS, "kind": "reg"},
    "model2_reg_ext": {"drop": LEVEL_COLS + REVERSAL_COLS, "kind": "reg"},
    # Reversal-feature variants (dist from rolling lows, range position,
    # vol-scaled stretch, RSI x Fear&Greed interaction).
    "model2_rev": {"drop": [], "soften": True, "kind": "clf"},
    "model2_price_rev": {"drop": EXTERNAL_COLS + CALENDAR_COLS,
                         "soften": True, "kind": "clf"},
}


def get_thresholds(name):
    if name == "model1":
        return config.BUY_PROB_THRESHOLD, config.SELL_PROB_THRESHOLD
    if V2_VARIANTS.get(name, {}).get("kind") == "reg":
        # Regression thresholds are in sigmas of predicted move, not probability.
        return config.REG_SIGNAL_SIGMA, config.REG_SIGNAL_SIGMA
    return config.MODEL2_BUY_PROB_THRESHOLD, config.MODEL2_SELL_PROB_THRESHOLD


DECILE = 0.10


# ---------------------------------------------------------
# SHARED YARDSTICK
# ---------------------------------------------------------
def forward_returns(price_df, horizon=LABEL_HORIZON):
    """Realized forward return per (timestamp, symbol), from the full history."""
    frames = []
    for sym in detect_symbols(price_df):
        close = price_df[f"{sym}_Close"]
        fwd = (close.shift(-horizon) / close - 1).rename("fwd_ret").to_frame()
        fwd["symbol"] = sym
        fwd = fwd.reset_index()
        fwd = fwd.rename(columns={fwd.columns[0]: "timestamp"})
        frames.append(fwd.set_index(["timestamp", "symbol"]))
    return pd.concat(frames).sort_index()["fwd_ret"]


# ---------------------------------------------------------
# PREDICTION COLLECTION (the slow part: trains every fold)
# ---------------------------------------------------------
def split_fold(merged, train_end, purge):
    ts = merged.index.get_level_values("timestamp")
    train = merged[ts < (train_end - purge)]
    test = merged[ts >= train_end]
    return train, test


def _probs_frame(fold, name, X_test, probs):
    df = pd.DataFrame(probs, columns=["p_sell", "p_hold", "p_buy"],
                      index=X_test.index).reset_index()
    df["fold"] = fold
    df["model"] = name
    return df


def collect_predictions(price_df, models_to_run, wf_train_days=WF_TRAIN_DAYS):
    bar_delta = pd.Series(price_df.index).diff().median()
    purge = LABEL_HORIZON * bar_delta

    train_days = pd.Timedelta(days=wf_train_days)
    test_days = pd.Timedelta(days=WF_TEST_DAYS)
    step = pd.Timedelta(days=WF_STEP_DAYS)

    start = price_df.index.min()
    end = price_df.index.max()

    v2_variants = {k: v for k, v in V2_VARIANTS.items() if k in models_to_run}

    frames = []
    t = start
    fold = 0

    while t + train_days + test_days <= end:
        if MAX_FOLDS is not None and fold >= MAX_FOLDS:
            break

        train_end = t + train_days
        test_end = train_end + test_days
        df_fold = price_df.loc[t:test_end]

        print(f"[FOLD {fold}] train {t:%Y-%m-%d} -> {train_end:%Y-%m-%d}, "
              f"test -> {test_end:%Y-%m-%d}  models={sorted(models_to_run)}")

        # Features are built over the full fold window so every model enters
        # the test period with warm rolling windows.

        # ---- Model 1, exactly as production trains it -------------------
        if "model1" in models_to_run:
            merged_v1 = merge_features_and_labels(df_fold)
            train_1, test_1 = split_fold(merged_v1, train_end, purge)
            if not train_1.empty and not test_1.empty:
                timestamps = train_1.index.get_level_values("timestamp")
                split_time = pd.Series(timestamps).quantile(TRAIN_SPLIT_Q)
                X_tr, y_tr, X_va, y_va = time_split(train_1, split_time)
                m1 = train_model(X_tr, y_tr, X_va, y_va)

                X_test = test_1.drop(columns=["label"])
                frames.append(_probs_frame(fold, "model1", X_test, m1.predict_proba(X_test)))

        # ---- Model 2 variants --------------------------------------------
        if v2_variants:
            feats = build_features_v2(df_fold)
            merged_clf = None
            merged_reg = None
            if any(s["kind"] == "clf" for s in v2_variants.values()):
                merged_clf = merge_features_and_labels_v2(df_fold, features=feats)
            if any(s["kind"] == "reg" for s in v2_variants.values()):
                merged_reg = merge_features_and_target_v2(df_fold, features=feats)

            for name, spec in v2_variants.items():
                merged = merged_clf if spec["kind"] == "clf" else merged_reg
                y_col = "label" if spec["kind"] == "clf" else "target"

                train_2, test_2 = split_fold(merged, train_end, purge)
                if train_2.empty or test_2.empty:
                    continue

                drop = [c for c in spec["drop"] if c in train_2.columns]
                X_tr2 = train_2.drop(columns=[y_col] + drop)
                y_tr2 = train_2[y_col]
                X_test2 = test_2.drop(columns=[y_col] + drop)

                if spec["kind"] == "clf":
                    m2 = train_model_v2(X_tr2, y_tr2, soften_weights=spec["soften"])
                    probs = m2.predict_proba(X_test2)
                else:
                    m2 = train_regressor_v2(X_tr2, y_tr2)
                    pred = m2.predict(X_test2)
                    probs = np.column_stack([-pred, np.zeros(len(pred)), pred])

                frames.append(_probs_frame(fold, name, X_test2, probs))

        t += step
        fold += 1

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------
# SCORING (cheap post-processing of stored probabilities)
# ---------------------------------------------------------
def score_fold(group, name, fwd_ret, price_df):
    buy_thr, sell_thr = get_thresholds(name)

    idx = pd.MultiIndex.from_frame(group[["timestamp", "symbol"]])
    fwd = fwd_ret.reindex(idx).to_numpy()

    score = group["p_buy"].to_numpy() - group["p_sell"].to_numpy()

    # --- rank IC (threshold-free) ---
    valid = ~np.isnan(fwd)
    ic = spearmanr(score[valid], fwd[valid]).statistic if valid.sum() > 10 else np.nan

    # --- decile spread (count-matched conviction) ---
    n = valid.sum()
    k = max(int(n * DECILE), 1)
    order = np.argsort(score[valid])
    fwd_v = fwd[valid]
    decile_spread = fwd_v[order[-k:]].mean() - fwd_v[order[:k]].mean()

    # --- production-gated signals ---
    buy_mask = group["p_buy"].to_numpy() >= buy_thr
    sell_mask = (group["p_sell"].to_numpy() >= sell_thr) & ~buy_mask

    buy_fwd = fwd[buy_mask & valid]
    sell_fwd = fwd[sell_mask & valid]

    signals = []
    gated = group[buy_mask | sell_mask]
    sides = np.where(gated["p_buy"] >= buy_thr, "BUY", "SELL")
    for (_, row), side in zip(gated.iterrows(), sides):
        price = price_df.loc[row["timestamp"], f"{row['symbol']}_Close"]
        if pd.isna(price):
            continue
        signals.append({
            "timestamp": row["timestamp"],
            "symbol": row["symbol"],
            "side": side,
            "price": float(price),
        })

    if signals:
        bt_stats, _ = backtest_portfolio(price_df, signals)
    else:
        bt_stats = {"return_pct": 0.0, "max_dd": 0.0, "n_trades": 0}

    return pd.Series({
        "rank_ic": ic,
        "decile_spread": decile_spread,
        "n_buy": int(buy_mask.sum()),
        "n_sell": int(sell_mask.sum()),
        "buy_fwd_ret": buy_fwd.mean() if len(buy_fwd) else np.nan,
        "sell_fwd_ret": sell_fwd.mean() if len(sell_fwd) else np.nan,
        "gated_edge": (buy_fwd.mean() - sell_fwd.mean())
                      if len(buy_fwd) and len(sell_fwd) else np.nan,
        "bt_return_pct": float(bt_stats["return_pct"]),
        "bt_max_dd": float(bt_stats["max_dd"]),
        "bt_n_trades": int(bt_stats["n_trades"]),
    })


def score_all(preds, price_df):
    fwd_ret = forward_returns(price_df)
    rows = []
    for (fold, name), group in preds.groupby(["fold", "model"], sort=True):
        stats = score_fold(group, name, fwd_ret, price_df)
        rows.append({"fold": fold, "model": name, **stats.to_dict()})
    return pd.DataFrame(rows)


def print_summary(results):
    if results.empty:
        print("No folds evaluated — not enough history for the walk-forward windows.")
        return

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    print("\n================ PER-FOLD RESULTS ================")
    print(results.set_index(["fold", "model"]))

    print("\n================ AVERAGE ACROSS FOLDS ================")
    agg = results.drop(columns=["fold"]).groupby("model").mean(numeric_only=True)
    print(agg)

    print("\n================ FOLDS WON (higher decile_spread) ================")
    piv = results.pivot(index="fold", columns="model", values="decile_spread")
    for m in piv.columns:
        others = piv.drop(columns=m)
        wins = (piv[m] > others.max(axis=1)).sum()
        print(f"{m}: {wins}/{len(piv)} folds")

    print("\nrank_ic       = Spearman corr of (p_buy - p_sell) vs realized forward return")
    print("decile_spread = mean fwd return of top-10% scores minus bottom-10%")
    print("gated_edge    = buy_fwd_ret - sell_fwd_ret using production thresholds")


# ---------------------------------------------------------
# DATA LOADING (cached)
# ---------------------------------------------------------
def load_price_df(refresh=False):
    if PRICE_CACHE.exists() and not refresh:
        print(f"[INFO] Using cached price data ({PRICE_CACHE.name}); "
              f"pass --refresh to re-download.")
        return pd.read_parquet(PRICE_CACHE)

    async def _fetch():
        async with ZypryxApi(config.API_URL, config.API_TOKEN) as api:
            ids = COIN_IDS or await load_coin_ids(api)
            return await load_price_data_from_api(api, ids, INTERVAL_ID)

    price_df = asyncio.run(_fetch())
    price_df.to_parquet(PRICE_CACHE)
    return price_df


def main():
    refresh = "--refresh" in sys.argv
    rescore_only = "--rescore" in sys.argv

    wf_train_days = WF_TRAIN_DAYS
    if "--train-days" in sys.argv:
        wf_train_days = int(sys.argv[sys.argv.index("--train-days") + 1])
    preds_file = predictions_file(wf_train_days)

    price_df = load_price_df(refresh=refresh)

    all_models = {"model1", *V2_VARIANTS}

    stored = None
    if preds_file.exists() and not refresh:
        stored = pd.read_parquet(preds_file)
        stored = stored[stored["model"].isin(all_models)]

    if rescore_only and stored is not None:
        print(f"[INFO] Re-scoring stored predictions ({preds_file.name}).")
        preds = stored
    else:
        # Fold splits are deterministic given the cached price data, so
        # models that already have stored predictions are not retrained.
        done = set(stored["model"].unique()) if stored is not None else set()
        missing = all_models - done
        if missing:
            new = collect_predictions(price_df, missing, wf_train_days)
            preds = pd.concat([stored, new], ignore_index=True) if stored is not None else new
        else:
            print("[INFO] All models already have stored predictions.")
            preds = stored
        preds.to_parquet(preds_file)

    results = score_all(preds, price_df)
    print_summary(results)

    results.to_csv("compare_results.csv", index=False)
    print(f"\nSaved per-fold results to compare_results.csv "
          f"and raw probabilities to {preds_file.name}")
    print("Re-score without retraining: python compare_models.py --rescore")


if __name__ == "__main__":
    main()
