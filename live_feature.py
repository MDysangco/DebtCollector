import pandas as pd

from features_and_labels import detect_symbols, build_features_for_symbol


def build_live_features(price_df):
    symbols = detect_symbols(price_df)
    frames = [build_features_for_symbol(price_df, sym) for sym in symbols]

    features = pd.concat(frames)
    features = features.reset_index()
    features = features.rename(columns={features.columns[0]: "timestamp"})
    features = features.set_index(["timestamp", "symbol"]).sort_index()
    return features
