# signal_engine.py
import pandas as pd
import numpy as np
import config

def build_raw_signals(test_df: pd.DataFrame, models: dict, feature_cols: dict, medians: dict = None) -> pd.DataFrame:
    """
    Build raw signals DataFrame with columns: timestamp, symbol, prob_long.
    medians: optional dict[prefix] -> dict(feature->median) used to impute test features.
    """
    rows = []
    medians = medians or {}

    for prefix, model in models.items():
        cols = feature_cols.get(prefix)
        if not cols:
            continue

        X = test_df[cols].copy()
        # Impute with training medians if available; otherwise forward/backfill as a last resort
        med = medians.get(prefix)
        if med:
            # pandas accepts a dict for fillna(value=...), which will map column names to values
            X = X.fillna(value=med)
        else:
            if config.IMPUTATION_METHOD == "median":
                X = X.fillna(value={col: medians.get(prefix, {}).get(col, X[col].median()) for col in cols})
            else:
                X = X.ffill().bfill()

        if X.empty:
            continue

        try:
            probs = model.predict_proba(X)[:, 1]
        except Exception:
            try:
                probs_raw = model.decision_function(X)
                probs = 1 / (1 + np.exp(-probs_raw))
            except Exception:
                probs = model.predict(X)
                probs = np.asarray(probs, dtype=float)

        df = pd.DataFrame({
            "timestamp": X.index,
            "symbol": prefix,
            "prob_long": probs
        }, index=X.index)
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "prob_long"])
    raw = pd.concat(rows, axis=0)
    raw = raw.reset_index(drop=True)
    return raw


def apply_execution_logic(raw_signals: pd.DataFrame, thresholds: dict = None, global_threshold = config.GLOBAL_THRESHOLD, cooldown_hours = config.COOLDOWN_HOURS) -> pd.DataFrame:
    """
    Convert raw signals into executable signals.
    thresholds: dict[prefix] -> threshold; fallback to global_threshold.
    Returns DataFrame with timestamp, symbol, prob_long, final_signal.
    """
    if raw_signals is None or raw_signals.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "prob_long", "final_signal"])

    thresholds = thresholds or {}
    df = raw_signals.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["timestamp"].notna()]

    # map per-symbol threshold
    floor = config.PER_SYMBOL_FLOOR
    df["threshold"] = df["symbol"].map(lambda s: max(thresholds.get(s, global_threshold), floor))

    margin = config.MARGIN
    df_filtered = df[df["prob_long"] >= (df["threshold"] + margin)].sort_values(["symbol", "timestamp"])

    # If nothing survives for a symbol, do not force fallback here (prefer no trade)
    if df_filtered.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "prob_long", "final_signal"])

    out_rows = []
    cooldown = pd.Timedelta(hours=cooldown_hours)
    for symbol, g in df_filtered.groupby("symbol"):
        last_ts = None
        for _, row in g.iterrows():
            ts = row["timestamp"]
            if last_ts is None or (ts - last_ts) >= cooldown:
                out_rows.append({"timestamp": ts, "symbol": symbol, "prob_long": row["prob_long"], "final_signal": 1})
                last_ts = ts

    out = pd.DataFrame(out_rows)
    return out
