import pandas as pd
import numpy as np
import config

def backtest_portfolio(price_window: pd.DataFrame, signal_configs: dict):
    """
    Multi-symbol portfolio backtest with:
      - TP / SL / max-hold exits
      - trend-aware exit (EMA break)
      - min holding period
      - re-entry cooldown
      - volatility filter
      - per-symbol trade cap

    price_window: wide DF, columns like 'C1_Close', 'C10_Close', ...
    signal_configs: dict[symbol] -> {
        'symbol': str,
        'side': 'LONG',
        'entries': [timestamps],
    }
    """
    if price_window.empty or not signal_configs:
        return {"return_pct": 0.0, "max_dd": 0.0}, []

    # infer bar index
    idx = price_window.index

    # precompute close series per symbol
    close_map = {}
    ema_map = {}
    vol_ok_map = {}

    for col in price_window.columns:
        if not col.endswith("_Close"):
            continue
        symbol = col[:-6]  # strip '_Close'
        close = price_window[col].astype(float)
        close_map[symbol] = close

        # trend EMA for exit (slower)
        ema = close.ewm(span=config.TREND_EMA_WINDOW_EXIT, adjust=False).mean()
        ema_map[symbol] = ema

        # volatility filter
        ret = close.pct_change()
        vol = ret.rolling(config.VOL_LOOKBACK).std()
        vol_ok_map[symbol] = vol > config.VOL_THRESHOLD

    equity = config.INITIAL_CAPITAL if hasattr(config, "INITIAL_CAPITAL") else 10000.0
    peak_equity = equity
    trades = []

    # per-symbol state
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

    # convert entries to fast lookup per symbol
    entry_sets = {
        sym: set(pd.to_datetime(cfg["entries"]))
        for sym, cfg in signal_configs.items()
    }

    for t in idx:
        for sym, cfg in signal_configs.items():
            if sym not in close_map:
                continue

            close = close_map[sym]
            ema = ema_map[sym]
            vol_ok = vol_ok_map[sym]
            st = state[sym]

            if pd.isna(close.get(t, np.nan)):
                continue

            price_t = close.loc[t]
            ema_t = ema.loc[t]
            vol_ok_t = bool(vol_ok.loc[t]) if t in vol_ok.index else False

            # update bars held if in position
            if st["position"] != 0 and st["entry_time"] is not None:
                st["bars_held"] += 1

            # --- exit logic if in position ---
            if st["position"] != 0:
                entry_price = st["entry_price"]
                pnl_pct = (price_t / entry_price) - 1.0

                exit_reason = None

                # 1) TP / SL
                if pnl_pct >= config.TAKE_PROFIT:
                    exit_reason = "TP"
                elif pnl_pct <= config.STOP_LOSS:
                    exit_reason = "SL"

                # 2) max hold
                if exit_reason is None and st["bars_held"] >= config.MAX_HOLD_BARS:
                    exit_reason = "MAX_HOLD"

                # 3) trend break (price < EMA * (1 - buffer)) after min hold
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

                    # flat position
                    st["position"] = 0
                    st["entry_price"] = None
                    st["entry_time"] = None
                    st["bars_held"] = 0
                    st["cooldown_until"] = t + (idx[1] - idx[0]) * config.REENTRY_COOLDOWN_BARS
                    st["trade_count"] += 1

            # --- entry logic if flat ---
            if st["position"] == 0:
                # per-symbol trade cap
                if st["trade_count"] >= config.MAX_TRADES_PER_SYMBOL:
                    continue

                # cooldown
                if st["cooldown_until"] is not None and t < st["cooldown_until"]:
                    continue

                # volatility filter
                if not vol_ok_t:
                    continue

                # entry signal?
                if t in entry_sets[sym]:
                    st["position"] = 1
                    st["entry_price"] = price_t
                    st["entry_time"] = t
                    st["bars_held"] = 0

        # portfolio equity mark-to-market (simple: equal weight per open position)
        open_positions = [s for s in state.values() if s["position"] != 0]
        if open_positions:
            # assume 1 unit per position for equity curve shape
            # (you can later scale by capital / n_positions)
            # here we just track relative curve via average pnl
            pass  # equity curve not strictly needed for stats below

    # compute portfolio stats from trades
    if not trades:
        return {"return_pct": 0.0, "max_dd": 0.0}, []

    # simple equity curve: start at 1.0, apply trade returns sequentially
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
