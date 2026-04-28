import pandas as pd
import numpy as np


def build_feature_frame(df: pd.DataFrame):
    """
    Builds directional labels and returns:
    - feature_df: full dataframe with features + labels
    - feature_cols: all numeric columns except labels
    - label_cols: all label columns
    """

    # build labels
    label_df = build_multi_horizon_labels(df)

    # merge labels into feature frame
    feature_df = df.copy()
    for col in label_df.columns:
        feature_df[col] = label_df[col]

    # extract label columns
    label_cols = [c for c in feature_df.columns if c.endswith("_Label")]

    # everything else is a feature
    feature_cols = [c for c in feature_df.columns if c not in label_cols]

    print(f"Feature frame built: {len(feature_df):,} rows, {len(feature_df.columns):,} columns")
    return feature_df, feature_cols, label_cols

def validate_feature_frame(df: pd.DataFrame):
    print("\n=== FEATURE FRAME VALIDATION ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    ts_min = df.index.min()
    ts_max = df.index.max()
    print(f"Timestamp range: {ts_min} → {ts_max}")

    # NaN check
    n_nan = df.isna().sum().sum()

    # Inf check (numeric only)
    numeric = df.select_dtypes(include=[np.number])
    n_inf = np.isinf(numeric).sum().sum()

    print(f"NaN values: {n_nan:,}")
    print(f"Inf values: {n_inf:,}")

    prefixes = sorted({c.split("_")[0] for c in df.columns if "_" in c})
    print(f"Detected prefixes: {prefixes}")
    print("=== VALIDATION COMPLETE ===")


def build_multi_horizon_labels(df, horizons=[6, 24, 72], atr_window=14, k=0.5):
    labels = {}
    prefixes = sorted({c.split("_")[0] for c in df.columns if c.endswith("_Close")})

    for prefix in prefixes:
        close = df[f"{prefix}_Close"]
        high = df[f"{prefix}_High"]
        low = df[f"{prefix}_Low"]

        # ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(atr_window).mean()

        for h in horizons:
            future = close.shift(-h)
            future_return = (future - close) / close

            up_thresh = k * atr / close
            down_thresh = -k * atr / close

            label = np.zeros(len(close), dtype=int)
            label[future_return > up_thresh] = 1
            label[future_return < down_thresh] = -1

            label[-h:] = 0

            labels[f"{prefix}_Label_{h}"] = label

    return pd.DataFrame(labels, index=df.index)
