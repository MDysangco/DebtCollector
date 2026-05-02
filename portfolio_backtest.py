import pandas as pd
import numpy as np
import config

def backtest_portfolio(price_window: pd.DataFrame, signal_configs: dict):
    """
    Multi-symbol portfolio backtest with:
      - TP / SL / max-hold exits
      - trend-aware exit (EMA break with buffer + min hold)
      - re-entry cooldown
      - volatility filter (std or ATR band)
      - per-symbol trade cap
      - ATR-based dynamic TP/SL (and optional ATR vol filter)

    price_window: wide DF, columns like 'C1_Close', 'C1_High', 'C1_Low', ...
    signal_configs: dict[symbol] -> {
        'symbol': str,
        'side': 'LONG',
        'entries': [timestamps],
    }
    """

    if price_window.empty or not signal_configs:
        return {"return_pct": 0.0, "max_dd": 0.0}, []

    idx = price_window.index

    # -----------------------------
    # Precompute per-symbol series
    # -----------------------------
    close_map = {}
    high_map = {}
    low_map = {}
    ema_map = {}
    vol_ok_map = {}
    atr_map = {}
    atr_vol_ok_map = {}

    # optional config flags / params with sane defaults
    ATR_WINDOW = getattr(config, "ATR_WINDOW", 14)
    ATR_TP_MULT = getattr(config, "ATR_TP_MULT", 2.0)
    ATR_SL_MULT = getattr(config, "ATR_SL_MULT", 2.0)
    USE_ATR_VOL_FILTER = getattr(config, "USE_ATR_VOL_FILTER", False)
    ATR_VOL_MIN = getattr(config, "ATR_VOL_MIN", 0.0)   # as fraction of price
    ATR_VOL_MAX = getattr(config, "ATR_VOL_MAX", 1.0)   # as fraction of price

    for col in price_window.columns:
        if not col.endswith("_Close"):
            continue

        symbol = col[:-6]  # strip '_Close'

        close = price_window[f"{symbol}_Close"].astype(float)
        close_map[symbol] = close

        # high/low if present (for ATR)
        if f"{symbol}_High" in price_window.columns and f"{symbol}_Low" in price_window.columns:
            high = price_window[f"{symbol}_High"].astype(float)
            low = price_window[f"{symbol}_Low"].astype(float)
        else:
            # fallback: use close as high/low if not provided
            high = close
            low = close

        high_map[symbol] = high
        low_map[symbol] = low

        # trend EMA for exit (slower)
        ema = close.ewm(span=config.TREND_EMA_WINDOW_EXIT, adjust=False).mean()
        ema_map[symbol] = ema

        # classic vol filter (std of returns)
        ret = close.pct_change()
        vol = ret.rolling(config.VOL_LOOKBACK).std()
        vol_ok_map[symbol] = vol > config.VOL_THRESHOLD

        # ATR computation
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=ATR_WINDOW, adjust=False).mean()
        atr_map[symbol] = atr

        # ATR-based vol band (ATR as fraction of price)
        atr_pct = atr / close.replace(0, np.nan)
        atr_vol_ok_map[symbol] = (atr_pct >= ATR_VOL_MIN) & (atr_pct <= ATR_VOL_MAX)

    equity = getattr(config, "INITIAL_CAPITAL", 10000.0)
    peak_equity = equity
    trades = []

    # -----------------------------
    # Per-symbol state
    # -----------------------------
    state = {
        sym: {
            "position": 0,
            "entry_price": None,
            "entry_time": None,
            "bars_held": 0,
            "cooldown_until": None,
            "trade_count": 0,
        }
        for sym in signal_configs.keys()
    }

    # fast lookup for entries
    entry_sets = {
        sym: set(pd.to_datetime(cfg["entries"]))
        for sym, cfg in signal_configs.items()
    }

    bar_delta = idx[1] - idx[0] if len(idx) > 1 else pd.Timedelta(0)

    # -----------------------------
    # Main backtest loop
    # -----------------------------
    for t in idx:
        for sym, cfg in signal_configs.items():
            if sym not in close_map:
                continue

            close = close_map[sym]
            ema = ema_map[sym]
            vol_ok = vol_ok_map[sym]
            atr = atr_map[sym]
            atr_vol_ok = atr_vol_ok_map[sym]
            st = state[sym]

            if pd.isna(close.get(t, np.nan)):
                continue

            price_t = close.loc[t]
            ema_t = ema.loc[t]
            vol_ok_t = bool(vol_ok.loc[t]) if t in vol_ok.index else False
            atr_t = float(atr.loc[t]) if t in atr.index else np.nan
            atr_vol_ok_t = bool(atr_vol_ok.loc[t]) if t in atr_vol_ok.index else True

            # update bars held
            if st["position"] != 0 and st["entry_time"] is not None:
                st["bars_held"] += 1

            # -------------------------
            # Exit logic
            # -------------------------
            if st["position"] != 0:
                entry_price = st["entry_price"]
                pnl_pct = (price_t / entry_price) - 1.0

                exit_reason = None

                # --- ATR-based dynamic TP/SL (if ATR available) ---
                if not np.isnan(atr_t) and ATR_TP_MULT > 0 and ATR_SL_MULT > 0:
                    # ATR as fraction of price at entry
                    atr_frac = atr_t / entry_price if entry_price != 0 else 0.0
                    tp_level = entry_price * (1.0 + ATR_TP_MULT * atr_frac)
                    sl_level = entry_price * (1.0 - ATR_SL_MULT * atr_frac)

                    if price_t >= tp_level:
                        exit_reason = "TP_ATR"
                    elif price_t <= sl_level:
                        exit_reason = "SL_ATR"

                # --- fallback to fixed TP/SL if no ATR exit triggered ---
                if exit_reason is None:
                    if pnl_pct >= config.TAKE_PROFIT:
                        exit_reason = "TP"
                    elif pnl_pct <= config.STOP_LOSS:
                        exit_reason = "SL"

                # max hold
                if exit_reason is None and st["bars_held"] >= config.MAX_HOLD_BARS:
                    exit_reason = "MAX_HOLD"

                # trend break (price < EMA * (1 - buffer)) after min hold
                if (
                    exit_reason is None
                    and st["bars_held"] >= config.TREND_MIN_HOLD_BARS
                    and price_t < ema_t * (1.0 - config.TREND_EXIT_BUFFER)
                ):
                    exit_reason = "TREND_BREAK"

                if exit_reason is not None:
                    trades.append({
                        "symbol": sym,
                        "entry_time": st["entry_time"],
                        "exit_time": t,
                        "entry_price": float(entry_price),
                        "exit_price": float(price_t),
                        "pnl_pct": float(pnl_pct),
                        "side": "LONG",
                        "exit_reason": exit_reason,
                    })

                    st["position"] = 0
                    st["entry_price"] = None
                    st["entry_time"] = None
                    st["bars_held"] = 0
                    st["cooldown_until"] = t + bar_delta * config.REENTRY_COOLDOWN_BARS
                    st["trade_count"] += 1

            # -------------------------
            # Entry logic
            # -------------------------
            if st["position"] == 0:
                # trade cap
                if st["trade_count"] >= config.MAX_TRADES_PER_SYMBOL:
                    continue

                # cooldown
                if st["cooldown_until"] is not None and t < st["cooldown_until"]:
                    continue

                # classic vol filter
                if not vol_ok_t:
                    continue

                # ATR vol band filter (optional)
                if USE_ATR_VOL_FILTER and not atr_vol_ok_t:
                    continue

                # entry signal
                if t in entry_sets[sym]:
                    st["position"] = 1
                    st["entry_price"] = price_t
                    st["entry_time"] = t
                    st["bars_held"] = 0

        # you can later plug a proper equity curve here if you want capital-aware sizing
        # currently we only use trade list for stats

    # -----------------------------
    # Portfolio stats
    # -----------------------------
    if not trades:
        return {"return_pct": 0.0, "max_dd": 0.0}, []

    eq = [1.0]
    for tr in trades:
        eq.append(eq[-1] * (1.0 + tr["pnl_pct"]))
    eq = pd.Series(eq)
    return_pct = eq.iloc[-1] - 1.0
    roll_max = eq.cummax()
    dd = (eq / roll_max) - 1.0
    max_dd = dd.min()

    stats = {
        "return_pct": float(return_pct),
        "max_dd": float(max_dd),
        "n_trades": len(trades),
    }

    return stats, trades
