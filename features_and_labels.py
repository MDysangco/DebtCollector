import pandas as pd
import numpy as np

from config import (
    RSI_LENGTH, ATR_LENGTH,
    MOM_WINDOWS, VOL_WINDOWS,
    LABEL_HORIZON, LABEL_UP_THRESH, LABEL_DOWN_THRESH
)


def detect_symbols(df):
    return sorted({col.split("_")[0] for col in df.columns if col.endswith("_Close")})


def build_features_for_symbol(df, sym):
    close = df[f"{sym}_Close"]
    high = df[f"{sym}_High"]
    low = df[f"{sym}_Low"]
    vol = df[f"{sym}_Volume"]

    rets = {f"ret_{w}": close.pct_change(w) for w in [1, 2, 4, 8]}
    moms = {f"mom_{w}": close / close.shift(w) - 1 for w in MOM_WINDOWS}
    vols = {f"vol_{w}": close.pct_change().rolling(w).std() for w in VOL_WINDOWS}

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(RSI_LENGTH).mean()
    loss = -delta.clip(upper=0).rolling(RSI_LENGTH).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_LENGTH).mean()

    f = {
        "symbol": sym,
        "close": close,
        "rsi": rsi,
        "atr": atr,
        "volume": vol,
    }
    f.update(rets)
    f.update(moms)
    f.update(vols)

    return pd.DataFrame(f, index=df.index)


def build_labels(close):
    future = close.shift(-LABEL_HORIZON)
    future_ret = (future / close) - 1

    label = pd.Series(index=close.index, dtype="float64")
    label[future_ret >= LABEL_UP_THRESH] = 2
    label[future_ret <= LABEL_DOWN_THRESH] = 0
    label[(future_ret < LABEL_UP_THRESH) & (future_ret > LABEL_DOWN_THRESH)] = 1

    return label


def merge_features_and_labels(df):
    symbols = detect_symbols(df)

    frames = [build_features_for_symbol(df, sym) for sym in symbols]
    features = pd.concat(frames).reset_index()

    if "index" in features.columns:
        features = features.rename(columns={"index": "timestamp"})
    elif "Timestamp" in features.columns:
        features = features.rename(columns={"Timestamp": "timestamp"})
    else:
        features = features.rename(columns={features.columns[0]: "timestamp"})

    features = features.set_index(["timestamp", "symbol"]).sort_index()

    label_frames = []
    for sym in symbols:
        close = df[f"{sym}_Close"]
        lbl = build_labels(close).rename("label").to_frame()
        lbl["symbol"] = sym
        lbl = lbl.reset_index().rename(columns={"Timestamp": "timestamp"})
        lbl = lbl.set_index(["timestamp", "symbol"])
        label_frames.append(lbl)

    labels = pd.concat(label_frames).sort_index()

    merged = features.join(labels, how="inner")
    merged = merged.dropna(subset=["label"])
    merged["label"] = merged["label"].astype(int)

    return merged
