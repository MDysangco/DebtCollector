import pandas as pd
from sqlalchemy.util import symbol


def build_raw_signals(test_df, models, feature_cols, medians):
    rows = []

    for prefix, model in models.items():
        cols = feature_cols.get(prefix, [])
        if not cols:
            continue

        X = test_df[cols].copy()

        med = medians.get(prefix, {})
        if med:
            X = X.fillna(med)

        try:
            probs = model.predict_proba(X)[:, 1]
        except Exception:
            preds = model.predict(X)
            probs = preds.astype(float)

        close_col = f"{prefix}_Close"
        ema_fast_col = f"{prefix}_EMA_fast"

        for ts, p in zip(test_df.index, probs):
            row = {
                "timestamp": ts,          # keep UTC-aware
                "symbol": prefix,
                "prob_long": float(p),
            }

            if close_col in test_df.columns:
                row[close_col] = test_df.loc[ts, close_col]

            if ema_fast_col in test_df.columns:
                row[ema_fast_col] = test_df.loc[ts, ema_fast_col]

            rows.append(row)

    raw = pd.DataFrame(rows)

    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "prob_long"])

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw[raw["timestamp"].notna()]

    return raw

def apply_execution_logic(
    raw_signals,
    test_df,
    global_threshold,
    per_symbol_floor,
    margin,
    cooldown_hours,
    thresholds=None,
):
    print(">>> USING PATCHED EXECUTION LOGIC (V1 + Trend Persistence FIXED) <<<")

    df = raw_signals.copy()
    df = df.sort_values(["timestamp", "symbol"])
    df["final_signal"] = 0

    if thresholds is None:
        thresholds = {}

    last_entry_time = {}
    N = 3  # trend persistence window (bars)

    for idx, row in df.iterrows():
        ts = row["timestamp"]
        sym = row["symbol"]
        prob = row["prob_long"]

        # --- Column names in wide test_df ---
        ema_fast_col = f"{sym}_EMA_fast"
        ema_slow_col = f"{sym}_EMA_slow"
        close_col = f"{sym}_Close"

        # --- Trend persistence filter: EMA_fast > EMA_slow for last N bars ---
        if ema_fast_col in test_df.columns and ema_slow_col in test_df.columns:
            # look back on wide frame using timestamp index
            window = test_df.loc[:ts].tail(N)

            # require at least N rows of history
            if len(window) < N:
                continue

            # if any bar has EMA_fast <= EMA_slow → skip
            if (window[ema_fast_col] <= window[ema_slow_col]).any():
                continue

        # --- EMA50 trend filter (price above EMA_fast) ---
        if ema_fast_col in test_df.columns and close_col in test_df.columns:
            window = test_df.loc[:ts].tail(1)
            if len(window) == 0:
                continue

            ema50 = window[ema_fast_col].iloc[0]
            close = window[close_col].iloc[0]

            if pd.notna(ema50) and pd.notna(close):
                if close < ema50:
                    continue

        # --- Dynamic per-symbol threshold ---
        symbol_thresh = thresholds.get(sym, {}).get("long", global_threshold)
        effective_thresh = max(symbol_thresh, per_symbol_floor) + margin

        # --- Cooldown logic ---
        if sym in last_entry_time:
            delta = ts - last_entry_time[sym]
            if delta.total_seconds() < cooldown_hours * 3600:
                continue

        # --- Entry condition ---
        if prob >= effective_thresh:
            df.at[idx, "final_signal"] = 1
            last_entry_time[sym] = ts

    return df

