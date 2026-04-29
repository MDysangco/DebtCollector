# signal_engine.py
import pandas as pd
import numpy as np

def build_raw_signals(test_df: pd.DataFrame, models: dict, feature_cols: dict) -> pd.DataFrame:
    """
    Build raw signals DataFrame with columns: timestamp, symbol, prob_long.
    - For each prefix in models, predict probabilities on test_df[feature_cols[prefix]].
    - Returns concatenated DataFrame (may be empty).
    """
    rows = []
    for prefix, model in models.items():
        cols = feature_cols.get(prefix)
        if not cols:
            continue
        X = test_df[cols].copy()
        # drop rows with NaNs for this model's features
        X = X.dropna()
        if X.empty:
            continue
        try:
            probs = model.predict_proba(X)[:, 1]
        except Exception:
            # fallback: if model doesn't support predict_proba, use decision_function or predict
            try:
                probs = model.decision_function(X)
                # scale to 0-1 via logistic
                probs = 1 / (1 + np.exp(-probs))
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


def apply_execution_logic(raw_signals: pd.DataFrame, prob_threshold: float = 0.6, cooldown_hours: int = 24) -> pd.DataFrame:
    """
    Convert raw signals into executable signals.
    - Filters by prob_threshold.
    - Ensures one signal per symbol per cooldown window (simple implementation).
    Returns DataFrame with timestamp, symbol, prob_long, final_signal (1 long, 0 flat).
    """
    if raw_signals is None or raw_signals.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "prob_long", "final_signal"])

    df = raw_signals.copy()
    # ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["timestamp"].notna()]

    # filter by probability threshold
    df = df[df["prob_long"] >= prob_threshold].sort_values(["symbol", "timestamp"])

    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "prob_long", "final_signal"])

    # simple cooldown: keep first signal per symbol per cooldown_hours
    out_rows = []
    cooldown = pd.Timedelta(hours=cooldown_hours)
    for symbol, g in df.groupby("symbol"):
        last_ts = None
        for _, row in g.iterrows():
            ts = row["timestamp"]
            if last_ts is None or (ts - last_ts) >= cooldown:
                out_rows.append({"timestamp": ts, "symbol": symbol, "prob_long": row["prob_long"], "final_signal": 1})
                last_ts = ts

    out = pd.DataFrame(out_rows)
    return out
