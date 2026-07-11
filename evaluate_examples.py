"""Benchmark the live Model 2 ensemble against user-flagged moments.

signal_examples.csv holds hand-picked "this should have been a signal"
moments (timestamp_utc, symbol, side). For each one, this script retrains
the ensemble exactly as a live run at that hour would have (data cut at the
timestamp) and reports what the model said, which gate blocked it, and the
realized forward return. It ends with recall at several z-thresholds and a
feature fingerprint of the flagged moments.

Hindsight warning: these examples are hand-picked bottoms. Use them to
diagnose WHAT the model misses, never as a tuning target on their own —
any change they inspire still has to win the walk-forward in
compare_models.py before going live.

Usage:
    python evaluate_examples.py [--parquet <price.parquet>]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from market_data import load_price_data
from features_v2 import build_features_v2
from live_prediction import (
    train_full_model, train_ensemble, ensemble_scores,
    compute_trend_and_vol, select_latest_complete_bar, drop_stale_coins,
)
from config import ENSEMBLE_Z_THRESHOLD, VOL_MIN_THRESHOLD, LABEL_HORIZON

EXAMPLES_FILE = Path(__file__).resolve().parent / "signal_examples.csv"
Z_LEVELS = [0.5, 0.75, 1.0]


def load_examples():
    ex = pd.read_csv(EXAMPLES_FILE)
    ex["timestamp_utc"] = pd.to_datetime(ex["timestamp_utc"], utc=True)
    return ex.sort_values("timestamp_utc")


def realized_fwd(price_full, sym, ts, horizon=LABEL_HORIZON):
    close = price_full[f"{sym}_Close"]
    if ts not in close.index or pd.isna(close.loc[ts]):
        return np.nan
    fwd = close[close.index > ts]
    if fwd.empty:
        return np.nan
    return fwd.iloc[min(horizon - 1, len(fwd) - 1)] / close.loc[ts] - 1


def evaluate_moment(price_full, ts, sym):
    """Reconstruct the live ensemble as of ts and report its view on sym."""
    price_df = price_full.loc[:ts]

    m1_model, m1_features = train_full_model(price_df)
    members = train_ensemble(price_df, m1_model, m1_features)

    latest_ts = select_latest_complete_bar(price_df, config.LIVE_BAR_MIN_COIN_FRAC)
    ens = ensemble_scores(members, latest_ts)
    ens_live = ens[ens.index.get_level_values("timestamp") == latest_ts]

    trend_df, vol_df = compute_trend_and_vol(price_df)

    row = {
        "requested": ts,
        "bar": latest_ts,
        "stale_hours": (ts - latest_ts) / pd.Timedelta(hours=1),
    }

    try:
        row["z"] = float(ens_live.xs(sym, level="symbol").iloc[0])
    except KeyError:
        row["z"] = np.nan
        return row

    # member-level breakdown: each member's cross-sectional rank for sym
    # that hour (1.0 = most bullish in the book)
    for name, model, feats in members:
        probs = model.predict_proba(feats.loc[ens_live.index])
        s_live = probs[:, 2] - probs[:, 0]
        sym_pos = list(ens_live.index.get_level_values("symbol")).index(sym)
        row[f"{name}_rank"] = float(pd.Series(s_live).rank(pct=True).iloc[sym_pos])

    close = price_df.loc[latest_ts, f"{sym}_Close"]
    row["close"] = float(close) if not pd.isna(close) else np.nan
    row["vol_ok"] = bool(vol_df.loc[latest_ts, sym] >= VOL_MIN_THRESHOLD)
    row["above_ema"] = bool(close > trend_df.loc[latest_ts, sym])
    row["rank_in_book"] = int(ens_live.rank(ascending=False).xs(sym, level="symbol").iloc[0])
    row["fwd24"] = realized_fwd(price_full, sym, latest_ts)
    return row


def feature_fingerprint(feats, examples):
    """How do the flagged moments look in feature space vs. all history?
    Reports each feature's average z-score at the flagged bars."""
    rows = []
    for _, ex in examples.iterrows():
        try:
            rows.append(feats.loc[(ex["timestamp_utc"], ex["symbol"])])
        except KeyError:
            pass
    if not rows:
        return pd.Series(dtype=float)
    flagged = pd.DataFrame(rows)

    sym_feats = feats.xs(examples["symbol"].iloc[0], level="symbol")
    z = (flagged.mean() - sym_feats.mean()) / sym_feats.std()
    return z.sort_values(key=abs, ascending=False)


def main():
    if "--parquet" in sys.argv:
        price_full = pd.read_parquet(sys.argv[sys.argv.index("--parquet") + 1])
    else:
        price_full = load_price_data()
    price_full = drop_stale_coins(price_full, config.STALE_COIN_MAX_BARS)

    examples = load_examples()
    print(f"{len(examples)} flagged moments, data through {price_full.index.max()}\n")

    # Incremental: moments already in examples_results.csv are not retrained.
    results_file = Path(__file__).resolve().parent / "examples_results.csv"
    cached = None
    if results_file.exists():
        cached = pd.read_csv(results_file, parse_dates=["requested", "bar"])
        done = set(zip(cached["requested"], cached["side"]))
    else:
        done = set()

    new_rows = []
    for _, ex in examples.iterrows():
        ts, sym = ex["timestamp_utc"], ex["symbol"]
        if (ts, ex["side"]) in done:
            continue
        print(f"[{ts:%Y-%m-%d %H:%M} {ex['side']}] training as-of snapshot...",
              flush=True)
        row = evaluate_moment(price_full, ts, sym)
        row["side"] = ex["side"]
        new_rows.append(row)

    res = pd.DataFrame(new_rows)
    if cached is not None:
        res = pd.concat([cached, res], ignore_index=True)
    res = res.sort_values("requested")

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    print("\n================ PER-MOMENT RESULTS ================")
    cols = ["requested", "side", "z", "rank_in_book", "vol_ok",
            "above_ema", "fwd24"]
    print(res[cols].to_string(index=False))

    valid = res.dropna(subset=["z"])
    for side, sgn in (("BUY", 1), ("SELL", -1)):
        part = valid[valid["side"] == side]
        if part.empty:
            continue
        print(f"\n================ {side} EXAMPLES ({len(part)}) ================")
        for thr in Z_LEVELS:
            hits = (sgn * part["z"] >= thr) & part["vol_ok"]
            print(f"  |z| >= {thr:.2f}: {int(hits.sum())}/{len(part)} fire as {side}")
        print(f"  (live threshold is {ENSEMBLE_Z_THRESHOLD})")
        print(f"  mean z:                 {part['z'].mean():+.2f} "
              f"(want {'positive' if sgn > 0 else 'negative'})")
        print(f"  mean realized next-24h: {part['fwd24'].mean():+.2%}")
        rank_cols = [c for c in part.columns
                     if c.endswith("_rank") and part[c].notna().any()]
        for c in rank_cols:
            print(f"  {c:14s} mean cross-sectional rank {part[c].mean():.2f} "
                  f"(1.0 = most bullish in book)")

    print("\n================ FEATURE FINGERPRINTS ================")
    print("(feature z-score at flagged moments vs. the coin's own history;"
          " |z| > ~0.7 = distinctive)")
    feats = build_features_v2(price_full)
    for side in ("BUY", "SELL"):
        part = examples[examples["side"] == side]
        if part.empty:
            continue
        fp = feature_fingerprint(feats, part)
        print(f"\n--- {side} ---")
        print(fp.head(12).round(2).to_string())

    res.to_csv(results_file, index=False)
    print(f"\nSaved to {results_file.name}")


if __name__ == "__main__":
    main()
