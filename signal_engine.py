import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build_raw_signals(test_df, models, feature_cols, medians):
    rows = []

    for prefix, model in models.items():
        # feature_cols is a dict[prefix] -> list of cols
        cols = feature_cols.get(prefix, [])
        if not cols:
            continue

        X = test_df[cols].copy()

        # impute using per-prefix medians
        med = medians.get(prefix, {})
        if med:
            X = X.fillna(med)

        try:
            probs = model.predict_proba(X)[:, 1]
        except Exception:
            preds = model.predict(X)
            probs = preds.astype(float)

        for ts, p in zip(test_df.index, probs):
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": prefix,
                    "prob_long": float(p),
                }
            )

    raw = pd.DataFrame(rows)

    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "prob_long"])

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw[raw["timestamp"].notna()]

    return raw


def apply_execution_logic(raw_signals,
                          global_threshold,
                          per_symbol_floor,
                          margin,
                          cooldown_hours):

    print(">>> USING PATCHED EXECUTION LOGIC <<<")

    df = raw_signals.copy()

    # Effective threshold uses only global + floor + margin
    effective_thresh = max(global_threshold, per_symbol_floor) + margin

    df["final_signal"] = 0
    df = df.sort_values(["timestamp", "symbol"])

    last_entry_time = {}

    for idx, row in df.iterrows():
        ts = row["timestamp"]
        sym = row["symbol"]
        prob = row["prob_long"]

        # Cooldown check
        if sym in last_entry_time:
            delta = ts - last_entry_time[sym]
            if delta.total_seconds() < cooldown_hours * 3600:
                continue

        # Entry condition (binary long-only)
        if prob >= effective_thresh:
            df.at[idx, "final_signal"] = 1
            last_entry_time[sym] = ts

    return df
