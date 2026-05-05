import time
import traceback

import pandas as pd
import numpy as np
from sqlalchemy import text

import config
from StoredProcedures import insert_raw_reading, insert_configuration
from db_loader import load_price_data_from_db, get_engine
from features_and_labels import merge_features_and_labels, detect_symbols
from live_feature import build_live_features
from model_train import train_model

from config import (
    COIN_IDS, INTERVAL_ID,
    BUY_PROB_THRESHOLD, SELL_PROB_THRESHOLD,
    TREND_EMA_LENGTH,
    VOL_FILTER_WINDOW, VOL_MIN_THRESHOLD,
)


def sql_safe(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return int(v)
    return v


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


def train_full_model(price_df: pd.DataFrame):
    features = build_live_features(price_df)
    merged = merge_features_and_labels(price_df)
    labels = merged["label"]

    X = features.loc[labels.index]
    y = labels

    model = train_model(X, y, X, y)
    return model, features


def generate_live_signals(price_df: pd.DataFrame):
    model, features = train_full_model(price_df)
    latest_ts = price_df.index.max()

    X_live = features.loc[features.index.get_level_values("timestamp") == latest_ts]
    if X_live.empty:
        return []

    trend_df, vol_df = compute_trend_and_vol(price_df)
    preds = model.predict(X_live)
    probs = model.predict_proba(X_live)

    signals = []

    config_id = insert_configuration(
        BuyProbabilityThreshold= config.BUY_PROB_THRESHOLD,
        SellProbabilityThreshold=config.SELL_PROB_THRESHOLD,
        TrendEMALength=config.TREND_EMA_LENGTH,
        VolFilterWindow=config.VOL_FILTER_WINDOW,
        VolMinThreshold=config.VOL_MIN_THRESHOLD,
        GlobalThreshold=config.GLOBAL_THRESHOLD,
        PerSymbolFloor=config.PER_SYMBOL_FLOOR,
        Margin=config.MARGIN,
        CooldownHours=config.COOLDOWN_HOURS
    )

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

        insert_raw_reading(
            TimestampUtc=latest_ts,
            CoinId=int(sym[1:]),
            PredictedClass=int(pred),
            ProbSell=float(p_sell),
            ProbHold=float(p_hold),
            ProbBuy=float(p_buy),
            Price=float(close),
            EMA=float(ema_val),
            Volatility=float(vol_val),
            PassedProbFilter=int(passed_prob),
            PassedTrendFilter=int(passed_trend),
            PassedVolFilter=int(passed_vol),
            FinalSignal=final_signal,
            ModelId=1,
            ConfigRowId=config_id
        )

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

    return signals


def main():
    print("Running LIVE prediction...\n")

    price_df = load_price_data_from_db(coin_ids=COIN_IDS, interval_id=INTERVAL_ID)
    signals = generate_live_signals(price_df)

    print(f"Latest timestamp: {price_df.index.max()}")
    print(f"Generated {len(signals)} live signals.\n")

    for s in signals:
        print(
            f"{s['timestamp']} | {s['symbol']} | {s['side']} "
            f"@ {s['price']:.4f} | p_buy={s['p_buy']:.3f} p_sell={s['p_sell']:.3f} "
            f"| ema={s['ema']:.4f} vol={s['vol']:.4f}"
        )


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print("Error in live loop:", e)
            traceback.print_exc()

        # Always sleep 1 hour no matter what happened
        time.sleep(3600)

