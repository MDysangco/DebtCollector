import pandas as pd

from model_train import train_model, time_split
from features_and_labels import merge_features_and_labels
from backtest_portfolio import backtest_portfolio

from ZypryxApi import ZypryxApi

from config import (
    WF_TRAIN_DAYS, WF_TEST_DAYS, WF_STEP_DAYS,
    LABEL_HORIZON, TRAIN_SPLIT_Q,
    COIN_IDS, INTERVAL_ID,
)


# ---------------------------------------------------------
# LOAD PRICE DATA FROM API
# ---------------------------------------------------------
def load_price_data_from_api(client: ZypryxApi, coin_ids, interval_id, start=None, end=None):
    frames = []

    for cid in coin_ids:
        kl = client.get_klines(cid, interval_id, start, end)
        if not kl:
            print(f"[WARN] Coin {cid} returned NO klines.")
            continue

        df = pd.DataFrame(kl)

        df["Timestamp"] = pd.to_datetime(df["klineOpenTime"], unit="ms", utc=True)

        df = df.rename(columns={
            "openPrice": "Open",
            "highPrice": "High",
            "lowPrice": "Low",
            "closePrice": "Close",
            "volume": "Volume",
        })

        df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
        df = df.drop_duplicates(subset=["Timestamp"])

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

    # Align start window
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
    client = ZypryxApiClient(config.API_URL, config.API_TOKEN)

    price_df = load_price_data_from_api(client, COIN_IDS, INTERVAL_ID)

    if use_tuning:
        stats, trades = run_wf_tuning(price_df)
        return {"wf_stats": stats, "wf_trades": trades}

    stats, trades, model = run_single_backtest(price_df, horizon=horizon)
    return {"stats": stats, "trades": trades, "model": model}
