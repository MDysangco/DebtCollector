"""Live prediction loop: one run posts Model 1 readings, then Model 2.

Model 1 (ModelId 1): per-coin technicals -> 3-class XGBoost, fixed +/-2%
labels, probability + trend + volatility filters. Unchanged from the
original pipeline.

Model 2 (ModelId 2): a three-member z-score ensemble. Walk-forward testing
(compare_models.py) at both 90- and 180-day training windows showed no
single feature set robustly beats Model 1, but three nearly uncorrelated
views averaged as z-scores beat it in BOTH settings (rank IC 0.038/0.030
vs Model 1's 0.037/0.012):

  m1        Model 1 verbatim (reused, not retrained)
  m2_price  cross-sectional/market-structure features, vol-scaled labels
  m2_full   the above plus macro & sentiment context (Fear & Greed, VIX,
            S&P 500, USD index, yield curve) — needs long training history
"""
import time
import json
import traceback
import pandas as pd
import numpy as np
import asyncio
import uuid

from ZypryxApi import ZypryxApi
from features_and_labels import merge_features_and_labels, detect_symbols
from live_feature import build_live_features
from market_data import load_coin_ids, load_price_data_from_api
from model_train import train_model

from features_v2 import (
    build_features_v2, merge_features_and_labels_v2,
    EXTERNAL_COLS, CALENDAR_COLS, REVERSAL_COLS,
)
from external_data import load_external_daily
from model_train_v2 import train_model_v2

import config
from config import (
    COIN_IDS, INTERVAL_ID,
    BUY_PROB_THRESHOLD, SELL_PROB_THRESHOLD,
    TREND_EMA_LENGTH,
    VOL_FILTER_WINDOW, VOL_MIN_THRESHOLD,
    ENSEMBLE_Z_THRESHOLD, ENSEMBLE_REF_BARS,
)

# ---------------------------------------------------------
# TREND + VOLATILITY
# ---------------------------------------------------------
def compute_trend_and_vol(price_df: pd.DataFrame):
    symbols = detect_symbols(price_df)
    trend = {}
    vol = {}

    for sym in symbols:
        close = price_df[f"{sym}_Close"]
        ema = close.ewm(span=TREND_EMA_LENGTH, adjust=False).mean()
        ret = close.pct_change()
        vol_series = ret.rolling(VOL_FILTER_WINDOW).std()

        trend[sym] = ema
        vol[sym] = vol_series

    trend_df = pd.DataFrame(trend, index=price_df.index)
    vol_df = pd.DataFrame(vol, index=price_df.index)
    return trend_df, vol_df

# ---------------------------------------------------------
# TRAIN MODEL 1
# ---------------------------------------------------------
def train_full_model(price_df: pd.DataFrame):
    features = build_live_features(price_df)
    merged = merge_features_and_labels(price_df)
    labels = merged["label"]

    X = features.loc[labels.index]
    y = labels

    model = train_model(X, y, X, y)
    return model, features

# ---------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------
def drop_stale_coins(price_df: pd.DataFrame, max_stale_bars: int):
    """Remove coins whose feed has gone dead (no close for > max_stale_bars).
    Their frozen series otherwise produce garbage features, distort the
    cross-sectional features of healthy coins, and hog the top z-scores."""
    keep, dropped = [], []
    for col in price_df.columns:
        if not col.endswith("_Close"):
            continue
        sym = col.split("_")[0]
        lv = price_df[col].last_valid_index()
        stale_bars = len(price_df.loc[lv:]) - 1 if lv is not None else len(price_df)
        (keep if stale_bars <= max_stale_bars else dropped).append(sym)

    if dropped:
        print(f"[WARN] Dropping stale coins (no close for >{max_stale_bars} bars): "
              f"{', '.join(dropped)}")
        cols = [c for c in price_df.columns if c.split("_")[0] in set(keep)]
        price_df = price_df[cols]
    return price_df


def select_latest_complete_bar(price_df: pd.DataFrame, min_coin_frac: float):
    close_cols = [c for c in price_df.columns if c.endswith("_Close")]
    valid_frac = price_df[close_cols].notna().mean(axis=1)
    complete = valid_frac[valid_frac >= min_coin_frac]
    if complete.empty:
        return price_df.index.max()
    return complete.index.max()


async def insert_model_config(api: ZypryxApi, buy_threshold, sell_threshold):
    """Insert a configuration row and return its unique id. The Buy/Sell
    threshold fields carry a probability for Model 1 and the ensemble z-gate
    (in sigmas) for Model 2."""
    config_hash = uuid.uuid4().hex

    config_list = [{
        "UniqueId": config_hash,
        "BuyProbabilityThreshold": buy_threshold,
        "SellProbabilityThreshold": sell_threshold,
        "TrendEMALength": config.TREND_EMA_LENGTH,
        "VolFilterWindow": config.VOL_FILTER_WINDOW,
        "VolMinThreshold": config.VOL_MIN_THRESHOLD,
        "GlobalThreshold": config.GLOBAL_THRESHOLD,
        "PerSymbolFloor": config.PER_SYMBOL_FLOOR,
        "Margin": config.MARGIN,
        "CooldownHours": config.COOLDOWN_HOURS
    }]

    if config.DRY_RUN:
        print("\n[DRY RUN] POST api/configuration payload (not sent):")
        print(json.dumps(config_list, indent=2))
    else:
        await api.insert_configurations(config_list)
    return config_hash


async def post_readings(api: ZypryxApi, readings_batch: list):
    if config.DRY_RUN:
        print(f"\n[DRY RUN] POST api/reading payload (not sent), "
              f"{len(readings_batch)} rows:")
        print(json.dumps(readings_batch, indent=2))
    else:
        await post_readings(api, readings_batch)


# ---------------------------------------------------------
# MODEL 1: GENERATE LIVE SIGNALS
# ---------------------------------------------------------
async def generate_signals_model1(api: ZypryxApi, price_df: pd.DataFrame,
                                  trend_df, vol_df, latest_ts):
    model, features = train_full_model(price_df)

    X_live = features.loc[features.index.get_level_values("timestamp") == latest_ts]
    if X_live.empty:
        return [], model, features

    preds = model.predict(X_live)
    probs = model.predict_proba(X_live)

    config_hash = await insert_model_config(
        api, config.BUY_PROB_THRESHOLD, config.SELL_PROB_THRESHOLD)

    readings_batch = []
    signals = []

    for (ts, sym), pred, prob_vec in zip(X_live.index, preds, probs):

        close = price_df.loc[latest_ts, f"{sym}_Close"]
        if pd.isna(close):
            continue

        p_sell, p_hold, p_buy = prob_vec

        passed_prob = (p_buy >= BUY_PROB_THRESHOLD) or (p_sell >= SELL_PROB_THRESHOLD)

        ema_val = trend_df.loc[latest_ts, sym]
        passed_trend = (
            (p_buy >= BUY_PROB_THRESHOLD and close > ema_val) or
            (p_sell >= SELL_PROB_THRESHOLD and close < ema_val)
        )

        vol_val = vol_df.loc[latest_ts, sym]
        passed_vol = vol_val >= VOL_MIN_THRESHOLD

        if passed_prob and passed_trend and passed_vol:
            final_signal = "BUY" if p_buy > p_sell else "SELL"
        else:
            final_signal = "HOLD"

        readings_batch.append({
            "TimeStampUTC": latest_ts.isoformat(),
            "CoinId": int(sym[1:]),
            "PredictClass": int(pred),
            "ProbSell": float(p_sell),
            "ProbHold": float(p_hold),
            "ProbBuy": float(p_buy),
            "Price": float(close),
            "EMA": float(ema_val),
            "Volatility": float(vol_val),
            "PassedProbFilter": bool(passed_prob),
            "PassedTrendFilter": bool(passed_trend),
            "PassedVolFilter": bool(passed_vol),
            "FinalSignal": final_signal,
            "ModelId": config.MODEL1_ID,
            "ConfigUniqueId": config_hash
        })

        if final_signal != "HOLD":
            signals.append({
                "timestamp": latest_ts,
                "symbol": sym,
                "side": final_signal,
                "price": float(close),
                "p_buy": float(p_buy),
                "p_sell": float(p_sell),
                "ema": float(ema_val),
                "vol": float(vol_val),
            })

    await post_readings(api, readings_batch)

    return signals, model, features


# ---------------------------------------------------------
# MODEL 2: ENSEMBLE
# ---------------------------------------------------------
def train_ensemble(price_df: pd.DataFrame, m1_model, m1_features):
    """Returns [(name, model, scoring_features)] for the three members.
    Model 1 is reused as a member, not retrained."""
    # REVERSAL_COLS are excluded: walk-forward testing showed they don't add
    # ensemble value (avg rank IC 0.027-0.032 vs 0.034 without them), so the
    # live members stay pinned to the validated feature set.
    feats2 = build_features_v2(price_df)
    cols_full = [c for c in feats2.columns if c not in set(REVERSAL_COLS)]
    cols_price = [c for c in cols_full
                  if c not in set(EXTERNAL_COLS) | set(CALENDAR_COLS)]

    merged_v2 = merge_features_and_labels_v2(price_df, features=feats2)

    # Member 2: cross-sectional price features only
    # ("model2_price" in compare_models.py).
    m2 = train_model_v2(merged_v2[cols_price], merged_v2["label"])

    # Member 3: price features plus macro/sentiment context
    # ("model2" in compare_models.py).
    m3 = train_model_v2(merged_v2[cols_full], merged_v2["label"])

    return [
        ("m1", m1_model, m1_features),
        ("m2_price", m2, feats2[cols_price]),
        ("m2_full", m3, feats2[cols_full]),
    ]


def current_fear_greed():
    """Latest Fear & Greed value, or None if the source is unavailable."""
    daily = load_external_daily()
    if "fng" not in daily.columns or daily["fng"].dropna().empty:
        return None
    return float(daily["fng"].dropna().iloc[-1])


def bounce_guard_flags(price_df: pd.DataFrame, latest_ts, fng_now) -> dict:
    """Per-symbol flag: is the coin in dead-cat-bounce conditions right now
    (sharp vol-scaled 4-bar rally + overbought RSI while the market is still
    fearful)? Used to veto BUY signals; see MODEL2_BOUNCE_GUARD in config."""
    if not config.MODEL2_BOUNCE_GUARD:
        return {}
    if fng_now is None or fng_now > config.BOUNCE_GUARD_FNG_MAX:
        return {}  # fail open: no sentiment data, or market not fearful

    flags = {}
    for sym in detect_symbols(price_df):
        close = price_df[f"{sym}_Close"]
        ret4 = close.pct_change(4)
        ret4_z = ret4 / ret4.rolling(168).std()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rsi = 100 - 100 / (1 + gain / loss)

        rz, rs = ret4_z.loc[latest_ts], rsi.loc[latest_ts]
        flags[sym] = (not pd.isna(rz) and not pd.isna(rs)
                      and rz >= config.BOUNCE_GUARD_RET4_Z
                      and rs >= config.BOUNCE_GUARD_RSI)
    return flags


def ensemble_scores(members, latest_ts):
    """Average of member z-scores per (timestamp, symbol) over the reference
    window. Z-scoring each member against its own recent score distribution
    puts differently calibrated classifiers on the same scale."""
    common = members[0][2].index
    for _, _, feats in members[1:]:
        common = common.intersection(feats.index)

    ref_ts = common.get_level_values("timestamp").unique().sort_values()
    ref_ts = ref_ts[ref_ts <= latest_ts][-ENSEMBLE_REF_BARS:]
    ref_index = common[common.get_level_values("timestamp").isin(ref_ts)]

    zs = []
    for name, model, feats in members:
        probs = model.predict_proba(feats.loc[ref_index])
        s = pd.Series(probs[:, 2] - probs[:, 0], index=ref_index)
        zs.append((s - s.mean()) / s.std())

    return pd.concat(zs, axis=1).mean(axis=1)


async def generate_signals_model2(api: ZypryxApi, price_df: pd.DataFrame,
                                  trend_df, vol_df, latest_ts,
                                  m1_model, m1_features):
    members = train_ensemble(price_df, m1_model, m1_features)

    ens = ensemble_scores(members, latest_ts)
    ens_live = ens[ens.index.get_level_values("timestamp") == latest_ts]
    if ens_live.empty:
        return [], ens_live

    # Averaged member probabilities, reported for interpretability.
    live_index = ens_live.index
    avg_probs = np.mean(
        [m.predict_proba(feats.loc[live_index]) for _, m, feats in members],
        axis=0,
    )

    config_hash = await insert_model_config(
        api, ENSEMBLE_Z_THRESHOLD, ENSEMBLE_Z_THRESHOLD)

    fng_now = current_fear_greed()
    guard_flags = bounce_guard_flags(price_df, latest_ts, fng_now)
    greed_veto = (config.MODEL2_GREED_GUARD and fng_now is not None
                  and fng_now >= config.GREED_GUARD_FNG_MIN)

    readings_batch = []
    signals = []

    for (ts, sym), z, prob_vec in zip(ens_live.index, ens_live, avg_probs):

        close = price_df.loc[latest_ts, f"{sym}_Close"]
        if pd.isna(close) or pd.isna(z):
            continue

        p_sell, p_hold, p_buy = prob_vec

        passed_prob = abs(z) >= ENSEMBLE_Z_THRESHOLD

        ema_val = trend_df.loc[latest_ts, sym]
        passed_trend = (
            (z >= ENSEMBLE_Z_THRESHOLD and close > ema_val) or
            (z <= -ENSEMBLE_Z_THRESHOLD and close < ema_val)
        )

        vol_val = vol_df.loc[latest_ts, sym]
        passed_vol = vol_val >= VOL_MIN_THRESHOLD

        trend_ok = passed_trend if config.MODEL2_USE_TREND_FILTER else True
        if passed_prob and trend_ok and passed_vol:
            final_signal = "BUY" if z > 0 else "SELL"
        else:
            final_signal = "HOLD"

        if final_signal == "BUY" and guard_flags.get(sym, False):
            print(f"[GUARD] {sym}: BUY vetoed, dead-cat bounce conditions "
                  f"(z={z:+.2f})")
            final_signal = "HOLD"

        if final_signal == "SELL" and greed_veto:
            print(f"[GUARD] {sym}: SELL vetoed, Fear & Greed at {fng_now:.0f} "
                  f"(greed; z={z:+.2f})")
            final_signal = "HOLD"

        readings_batch.append({
            "TimeStampUTC": latest_ts.isoformat(),
            "CoinId": int(sym[1:]),
            "PredictClass": int(np.argmax(prob_vec)),
            "ProbSell": float(p_sell),
            "ProbHold": float(p_hold),
            "ProbBuy": float(p_buy),
            "Price": float(close),
            "EMA": float(ema_val),
            "Volatility": float(vol_val),
            "PassedProbFilter": bool(passed_prob),
            "PassedTrendFilter": bool(passed_trend),
            "PassedVolFilter": bool(passed_vol),
            "FinalSignal": final_signal,
            "ModelId": config.MODEL2_ID,
            "ConfigUniqueId": config_hash
        })

        if final_signal != "HOLD":
            signals.append({
                "timestamp": latest_ts,
                "symbol": sym,
                "side": final_signal,
                "price": float(close),
                "z": float(z),
                "p_buy": float(p_buy),
                "p_sell": float(p_sell),
                "ema": float(ema_val),
                "vol": float(vol_val),
            })

    await post_readings(api, readings_batch)

    return signals, ens_live


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def _print_signals(signals):
    for s in signals:
        z_part = f"z={s['z']:+.2f} " if "z" in s else ""
        print(
            f"{s['timestamp']} | {s['symbol']} | {s['side']} "
            f"@ {s['price']:.4f} | {z_part}"
            f"p_buy={s['p_buy']:.3f} p_sell={s['p_sell']:.3f} "
            f"| ema={s['ema']:.4f} vol={s['vol']:.4f}"
        )


async def run_once():
    try:
        print("Running LIVE prediction...\n")

        async with ZypryxApi(config.API_URL, config.API_TOKEN) as api:
            coin_ids = await load_coin_ids(api)
            price_df = await load_price_data_from_api(api, coin_ids, config.INTERVAL_ID)
            price_df = drop_stale_coins(price_df, config.STALE_COIN_MAX_BARS)

            trend_df, vol_df = compute_trend_and_vol(price_df)
            latest_ts = select_latest_complete_bar(price_df, config.LIVE_BAR_MIN_COIN_FRAC)

            print("\n--- Model 1 ---")
            signals1, m1_model, m1_features = await generate_signals_model1(
                api, price_df, trend_df, vol_df, latest_ts)
            print(f"Generated {len(signals1)} live signals.")
            _print_signals(signals1)

            print("\n--- Model 2 (ensemble) ---")
            signals2, ens_live = await generate_signals_model2(
                api, price_df, trend_df, vol_df, latest_ts,
                m1_model, m1_features)
            if not ens_live.empty:
                print(f"Ensemble z-scores: min={ens_live.min():.2f} "
                      f"median={ens_live.median():.2f} max={ens_live.max():.2f}")
            print(f"Generated {len(signals2)} live signals.")
            _print_signals(signals2)

        print(f"\nLatest timestamp: {latest_ts}")

    except Exception as e:
        print("Error in live run:", e)
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_once())
