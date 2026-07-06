import asyncio

import pandas as pd

from model_train import train_model, time_split
from features_and_labels import merge_features_and_labels
from backtest_portfolio import backtest_portfolio
from market_data import load_coin_ids, load_price_data_from_api

from ZypryxApi import ZypryxApi

import config
from config import (
    WF_TRAIN_DAYS, WF_TEST_DAYS, WF_STEP_DAYS,
    LABEL_HORIZON, TRAIN_SPLIT_Q,
    COIN_IDS, INTERVAL_ID,
)


# ---------------------------------------------------------
# LOAD PRICE DATA FROM API
# ---------------------------------------------------------
async def _fetch_price_df():
    async with ZypryxApi(config.API_URL, config.API_TOKEN) as api:
        coin_ids = COIN_IDS or await load_coin_ids(api)
        return await load_price_data_from_api(api, coin_ids, INTERVAL_ID)


# ---------------------------------------------------------
# SINGLE BACKTEST
# ---------------------------------------------------------
def run_single_backtest(price_df, horizon=LABEL_HORIZON):
    merged = merge_features_and_labels(price_df)

    timestamps = merged.index.get_level_values("timestamp")
    split_time = pd.Series(timestamps).quantile(TRAIN_SPLIT_Q)

    X_train, y_train, X_val, y_val = time_split(merged, split_time)

    model = train_model(X_train, y_train, X_val, y_val)

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    signals = []
    for (ts, sym), p, prob in zip(X_val.index, preds, probs):
        price = price_df.loc[ts, f"{sym}_Close"]

        if p == 2:  # BUY
            signals.append({
                "timestamp": ts,
                "symbol": sym,
                "side": "BUY",
                "price": float(price),
            })
        elif p == 0:  # SELL
            signals.append({
                "timestamp": ts,
                "symbol": sym,
                "side": "SELL",
                "price": float(price),
            })

    stats, trades = backtest_portfolio(price_df, signals)
    return stats, trades, model


# ---------------------------------------------------------
# WALK-FORWARD TUNING
# ---------------------------------------------------------
def run_wf_tuning(price_df):
    all_stats = []
    all_trades = []

    start = price_df.index.min()
    end = price_df.index.max()

    train_days = pd.Timedelta(days=WF_TRAIN_DAYS)
    test_days = pd.Timedelta(days=WF_TEST_DAYS)
    step = pd.Timedelta(days=WF_STEP_DAYS)

    t = start

    while t + train_days + test_days <= end:
        train_start = t
        train_end = t + train_days
        test_end = train_end + test_days

        df_train = price_df.loc[train_start:train_end]
        df_test = price_df.loc[train_end:test_end]

        merged = merge_features_and_labels(df_train)

        timestamps = merged.index.get_level_values("timestamp")
        split_time = pd.Series(timestamps).quantile(TRAIN_SPLIT_Q)

        X_train, y_train, X_val, y_val = time_split(merged, split_time)
        model = train_model(X_train, y_train, X_val, y_val)

        merged_test = merge_features_and_labels(df_test)
        preds = model.predict(merged_test.drop(columns=["label"]))
        probs = model.predict_proba(merged_test.drop(columns=["label"]))

        signals = []
        for (ts, sym), p, prob in zip(merged_test.index, preds, probs):
            price = price_df.loc[ts, f"{sym}_Close"]
            if p == 2:
                signals.append({"timestamp": ts, "symbol": sym, "side": "BUY", "price": float(price)})
            elif p == 0:
                signals.append({"timestamp": ts, "symbol": sym, "side": "SELL", "price": float(price)})

        stats, trades = backtest_portfolio(price_df, signals)
        all_stats.append(stats)
        all_trades.extend(trades)

        t += step

    return all_stats, all_trades


# ---------------------------------------------------------
# PIPELINE ENTRYPOINT
# ---------------------------------------------------------
def run_pipeline(use_tuning=False, horizon=LABEL_HORIZON):
    price_df = asyncio.run(_fetch_price_df())

    if use_tuning:
        stats, trades = run_wf_tuning(price_df)
        return {"wf_stats": stats, "wf_trades": trades}

    stats, trades, model = run_single_backtest(price_df, horizon=horizon)
    return {"stats": stats, "trades": trades, "model": model}
