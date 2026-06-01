import time
import traceback
import pandas as pd
import numpy as np
import asyncio
import uuid

from ZypryxApi import ZypryxApi
from features_and_labels import merge_features_and_labels, detect_symbols
from live_feature import build_live_features
from model_train import train_model

import config
from config import (
    COIN_IDS, INTERVAL_ID,
    BUY_PROB_THRESHOLD, SELL_PROB_THRESHOLD,
    TREND_EMA_LENGTH,
    VOL_FILTER_WINDOW, VOL_MIN_THRESHOLD,
)

# ---------------------------------------------------------
# LOAD COINS DYNAMICALLY
# ---------------------------------------------------------
async def load_coin_ids(api):
    coins = await api.get_active_coins()
    if not coins:
        raise ValueError("No active coins returned from API.")
    return [c["Id"] for c in coins]

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
# TRAIN MODEL
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
# LOAD PRICE DATA FROM API
# ---------------------------------------------------------
async def load_price_data_from_api(api: ZypryxApi, coin_ids, interval_id):
    frames = []

    for cid in coin_ids:
        kl = await api.get_klines(cid, interval_id)
        if not kl:
            print(f"[WARN] Coin {cid} returned NO klines.")
            continue

        df = pd.DataFrame(kl)

        df = df.rename(columns={
            "KlineOpenTime": "Timestamp",
            "OpenPrice": "Open",
            "HighPrice": "High",
            "LowPrice": "Low",
            "ClosePrice": "Close",
            "Volume": "Volume",
        })

        df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype("int64"), unit="ms", utc=True)
        df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]

        prefix = f"C{cid}"
        df = df.rename(columns={
            "Open": f"{prefix}_Open",
            "High": f"{prefix}_High",
            "Low": f"{prefix}_Low",
            "Close": f"{prefix}_Close",
            "Volume": f"{prefix}_Volume",
        })

        frames.append(df)

    if not frames:
        raise ValueError("No coins returned klines.")

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="Timestamp", how="outer")

    merged = merged.sort_values("Timestamp").set_index("Timestamp")
    merged = merged[~merged.index.duplicated(keep="first")]

    first_valids = []
    for col in merged.columns:
        fv = merged[col].first_valid_index()
        if fv is not None:
            first_valids.append(fv)

    start_ts = max(first_valids)
    merged = merged.loc[start_ts:]

    print(f"[INFO] Aligned window starts at: {start_ts}")
    print(f"[INFO] Using {len(merged.columns) // 5} valid coins")

    return merged

# ---------------------------------------------------------
# GENERATE LIVE SIGNALS
# ---------------------------------------------------------
async def generate_live_signals(api: ZypryxApi, price_df: pd.DataFrame):
    model, features = train_full_model(price_df)
    latest_ts = price_df.index.max()

    X_live = features.loc[features.index.get_level_values("timestamp") == latest_ts]
    if X_live.empty:
        return []

    trend_df, vol_df = compute_trend_and_vol(price_df)
    preds = model.predict(X_live)
    probs = model.predict_proba(X_live)

    # ---------------------------------------------------------
    # CONFIG AS LIST + PYTHON-GENERATED GUID
    # ---------------------------------------------------------
    config_hash = uuid.uuid4().hex

    config_list = [{
        "UniqueId": config_hash,
        "BuyProbabilityThreshold": config.BUY_PROB_THRESHOLD,
        "SellProbabilityThreshold": config.SELL_PROB_THRESHOLD,
        "TrendEMALength": config.TREND_EMA_LENGTH,
        "VolFilterWindow": config.VOL_FILTER_WINDOW,
        "VolMinThreshold": config.VOL_MIN_THRESHOLD,
        "GlobalThreshold": config.GLOBAL_THRESHOLD,
        "PerSymbolFloor": config.PER_SYMBOL_FLOOR,
        "Margin": config.MARGIN,
        "CooldownHours": config.COOLDOWN_HOURS
    }]

    # API now returns ONLY bool
    await api.insert_configurations(config_list)

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
            "ModelId": 1,
            "ConfigHash": config_hash
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

    print(readings_batch)
    await api.insert_readings_bulk(readings_batch)

    return signals


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
async def run_once():
    try:
        print("Running LIVE prediction...\n")

        async with ZypryxApi(config.API_URL, config.API_TOKEN) as api:
            coin_ids = await load_coin_ids(api)
            price_df = await load_price_data_from_api(api, coin_ids, config.INTERVAL_ID)
            signals = await generate_live_signals(api, price_df)

        print(f"Latest timestamp: {price_df.index.max()}")
        print(f"Generated {len(signals)} live signals.\n")

        for s in signals:
            print(
                f"{s['timestamp']} | {s['symbol']} | {s['side']} "
                f"@ {s['price']:.4f} | p_buy={s['p_buy']:.3f} p_sell={s['p_sell']:.3f} "
                f"| ema={s['ema']:.4f} vol={s['vol']:.4f}"
            )

    except Exception as e:
        print("Error in live run:", e)
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_once())
