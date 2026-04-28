import numpy as np
import pandas as pd

BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45

EMA_FAST = 50
EMA_SLOW = 200

MAX_HOLD = 48          # 2 days on 1h bars
STOP_LOSS = -0.05      # -5%
TAKE_PROFIT = 0.08     # +8%

def compute_volatility_position_size(close, window=48, target_vol=0.02):
    """
    Computes volatility-scaled position size.
    close: price series (numpy array)
    window: rolling window for realized volatility
    target_vol: desired volatility per trade
    """
    returns = np.diff(close) / close[:-1]
    vol = pd.Series(returns).rolling(window).std().shift(1).values

    # Align lengths
    vol = np.concatenate([[vol[0]], vol])

    # Avoid division by zero
    vol = np.where(vol == 0, np.nan, vol)

    size = target_vol / vol

    # Clamp to reasonable bounds
    size = np.clip(size, 0.1, 1.0)

    # Replace NaN with minimum size
    size = np.nan_to_num(size, nan=0.1)

    return size

def run_full_signal_engine(df, models, feature_cols):
    signal_configs = {}

    for prefix, horizon_models in models.items():
        print(f"\n=== Building signals for {prefix} ===")

        close_col = f"{prefix}_Close"
        label_6  = f"{prefix}_Label_6"
        label_24 = f"{prefix}_Label_24"
        label_72 = f"{prefix}_Label_72"

        # Basic column checks
        if close_col not in df.columns:
            continue
        if label_6 not in df.columns or label_24 not in df.columns or label_72 not in df.columns:
            continue

        # Require all horizons to have labels
        mask = ~(df[label_6].isna() |
                 df[label_24].isna() |
                 df[label_72].isna())

        if mask.sum() < 200:
            # Not enough signalable rows
            continue

        close = df[close_col]

        # === TREND FILTER ===
        ema_fast = close.ewm(span=EMA_FAST, adjust=False).mean().loc[mask].values
        ema_slow = close.ewm(span=EMA_SLOW, adjust=False).mean().loc[mask].values
        trend = ema_fast - ema_slow

        # === MODEL INPUT ===
        X = df.loc[mask, feature_cols].values

        # === MULTI-HORIZON PREDICTIONS ===
        model6  = horizon_models.get("h6")
        model24 = horizon_models.get("h24")
        model72 = horizon_models.get("h72")

        if model6 is None or model24 is None or model72 is None:
            # Incomplete model set for this prefix
            continue

        p6  = model6.predict_proba(X)[:, 2]
        p24 = model24.predict_proba(X)[:, 2]
        p72 = model72.predict_proba(X)[:, 2]

        # === META-SIGNAL ===
        meta_signal = build_meta_signal(
            pred6=p6,
            pred24=p24,
            pred72=p72,
            span=5,
            persistence=3
        )

        prices = close.loc[mask].values

        # === VOLATILITY-SCALED POSITION SIZE ===
        pos_size_local = compute_volatility_position_size(
            prices,
            window=48,
            target_vol=0.02
        )

        final_signal = []
        final_size = []

        state = 0
        hold_counter = 0
        entry_price = None

        for sig, ts, price, size in zip(meta_signal, trend, prices, pos_size_local):

            # EXIT CONDITIONS
            if state == 1:
                # Stop-loss
                if entry_price and (price - entry_price) / entry_price <= STOP_LOSS:
                    final_signal.append(-1)
                    final_size.append(size)
                    state = 0
                    hold_counter = 0
                    entry_price = None
                    continue

                # Take-profit
                if entry_price and (price - entry_price) / entry_price >= TAKE_PROFIT:
                    final_signal.append(-1)
                    final_size.append(size)
                    state = 0
                    hold_counter = 0
                    entry_price = None
                    continue

                # Trend exit
                if ts < 0:
                    final_signal.append(-1)
                    final_size.append(size)
                    state = 0
                    hold_counter = 0
                    entry_price = None
                    continue

                # Max-hold exit
                if hold_counter >= MAX_HOLD:
                    final_signal.append(-1)
                    final_size.append(size)
                    state = 0
                    hold_counter = 0
                    entry_price = None
                    continue

            # ENTRY CONDITIONS
            if state == 0 and sig == 1 and ts > 0:
                final_signal.append(1)
                final_size.append(size)
                state = 1
                hold_counter = 1
                entry_price = price
                continue

            # HOLD
            if state == 1:
                final_signal.append(1)
                final_size.append(size)
                hold_counter += 1
                continue

            # FLAT
            final_signal.append(0)
            final_size.append(0.0)
            state = 0
            hold_counter = 0
            entry_price = None

        # === EXPAND TO FULL INDEX ===
        full_final_signal = np.zeros(len(df), dtype=int)
        full_position_size = np.zeros(len(df), dtype=float)

        full_final_signal[mask.values] = np.array(final_signal, dtype=int)
        full_position_size[mask.values] = np.array(final_size, dtype=float)

        signal_configs[prefix] = {
            "full_final_signal": full_final_signal,
            "full_position_size": full_position_size,
            "meta_signal": meta_signal,
            "trend": trend,
        }

    return signal_configs

def build_meta_signal(pred6, pred24, pred72, span=5, persistence=3):
    """
    pred6, pred24, pred72 are probability arrays (0..1)
    Returns a smoothed, persistent long/flat signal.
    """

    # 1. Horizon ensemble
    p_meta = 0.2 * pred6 + 0.5 * pred24 + 0.3 * pred72

    # 2. Temporal smoothing (EMA)
    p_smooth = pd.Series(p_meta).ewm(span=span).mean().values

    # 3. Persistence filter
    signal = np.zeros_like(p_smooth, dtype=int)
    above = p_smooth > 0.55

    count = 0
    for i, ok in enumerate(above):
        if ok:
            count += 1
            if count >= persistence:
                signal[i] = 1
        else:
            count = 0

    return signal
