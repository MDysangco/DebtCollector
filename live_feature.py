import pandas as pd
import numpy as np
from features_and_labels import detect_symbols
from config import RSI_LENGTH, ATR_LENGTH, MOM_WINDOWS, VOL_WINDOWS


def build_live_features(price_df):
    symbols = detect_symbols(price_df)
    frames = []

    for sym in symbols:
        close = price_df[f"{sym}_Close"]
        high = price_df[f"{sym}_High"]
        low = price_df[f"{sym}_Low"]
        vol = price_df[f"{sym}_Volume"]

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

        moms = {f"mom_{w}": close / close.shift(w) - 1 for w in MOM_WINDOWS}
        vols = {f"vol_{w}": close.pct_change().rolling(w).std() for w in VOL_WINDOWS}

        f = {
            "symbol": sym,
            "close": close,
            "rsi": rsi,
            "atr": atr,
            "volume": vol,
        }
        f.update(moms)
        f.update(vols)

        df = pd.DataFrame(f, index=price_df.index)
        frames.append(df)

    features = pd.concat(frames)

    # 🔥 FIX: force index → timestamp column
    features = features.reset_index()
    features = features.rename(columns={features.columns[0]: "timestamp"})

    features = features.set_index(["timestamp", "symbol"]).sort_index()
    return features
