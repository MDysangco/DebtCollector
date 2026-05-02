import numpy as np

FEE_RATE = 0.001   # 0.1% per side
STOP_LOSS = -0.05  # -5%
TAKE_PROFIT = 0.08 # +8%
MAX_HOLD = 48      # bars


def backtest_portfolio(df, signal_configs, periods=None):
    equity0 = 10_000.0
    equity = equity0
    equity_curve = []

    prefixes = list(signal_configs.keys())

    positions = {p: 0 for p in prefixes}
    entry_price = {p: None for p in prefixes}
    entry_time = {p: None for p in prefixes}
    entry_idx = {p: None for p in prefixes}
    notional = {p: 0.0 for p in prefixes}

    # track previous signal per prefix for flip-only entries
    prev_sig = {p: 0 for p in prefixes}

    trades = []

    # Universe selection
    if periods:
        periods = sorted(periods, key=lambda x: x["rebalance_time"])

        def get_active(ts):
            active = prefixes
            for p in periods:
                if ts >= p["rebalance_time"]:
                    active = p["approved"]
                else:
                    break
            return set(active)
    else:
        def get_active(ts):
            return set(prefixes)

    n = len(df)

    for i in range(n - 1):
        ts = df.index[i]
        active = get_active(ts)

        for prefix in prefixes:
            close_col = f"{prefix}_Close"
            if close_col not in df.columns:
                continue

            price_now = df.iloc[i][close_col]
            price_next = df.iloc[i + 1][close_col]

            if (
                price_now is None or price_next is None or
                np.isnan(price_now) or np.isnan(price_next) or
                price_now <= 0
            ):
                continue

            sig_series = signal_configs[prefix]["full_final_signal"]
            sig = int(sig_series[i])
            prev = int(prev_sig[prefix])

            # flip-only entry trigger: 0 -> 1
            flip_long_entry = (prev == 0 and sig == 1)

            # Forced exit if coin not in universe
            forced_exit = prefix not in active

            # --- LONG ENTRY LOGIC (flip-only, position-aware) ---
            if positions[prefix] == 0 and not forced_exit:

                if flip_long_entry:
                    positions[prefix] = 1
                    entry_price[prefix] = price_now
                    entry_time[prefix] = ts
                    entry_idx[prefix] = i

                    alloc = equity / len(prefixes)
                    notional[prefix] = alloc

                    equity -= alloc * FEE_RATE  # entry fee

            # --- EXIT LOGIC (LONG ONLY, no exit on sig==0) ---
            elif positions[prefix] != 0:

                bars_held = i - entry_idx[prefix]

                # Long PnL: (price_now - entry_price) / entry_price
                if positions[prefix] == 1:
                    pnl_pct = (price_now - entry_price[prefix]) / entry_price[prefix]
                else:
                    # placeholder for future short logic
                    pnl_pct = (entry_price[prefix] - price_now) / entry_price[prefix]

                exit_now = (
                    forced_exit or
                    pnl_pct <= STOP_LOSS or
                    pnl_pct >= TAKE_PROFIT or
                    bars_held >= MAX_HOLD
                )

                if exit_now:
                    pnl = notional[prefix] * pnl_pct
                    fee = notional[prefix] * FEE_RATE
                    equity += pnl - fee

                    trades.append({
                        "symbol": prefix,
                        "entry_time": entry_time[prefix],
                        "exit_time": ts,
                        "entry_price": float(entry_price[prefix]),
                        "exit_price": float(price_now),
                        "pnl_pct": float(pnl_pct),
                        "side": "LONG" if positions[prefix] == 1 else "SHORT",
                    })

                    positions[prefix] = 0
                    entry_price[prefix] = None
                    entry_time[prefix] = None
                    entry_idx[prefix] = None
                    notional[prefix] = 0.0

            # update prev signal for next bar
            prev_sig[prefix] = sig

        equity_curve.append(equity)

    # Close open positions at final bar
    final_ts = df.index[-1]
    for prefix in prefixes:
        if positions[prefix] != 0:
            price_now = df.iloc[-1][f"{prefix}_Close"]
            if positions[prefix] == 1:
                pnl_pct = (price_now - entry_price[prefix]) / entry_price[prefix]
            else:
                pnl_pct = (entry_price[prefix] - price_now) / entry_price[prefix]

            pnl = notional[prefix] * pnl_pct
            fee = notional[prefix] * FEE_RATE
            equity += pnl - fee

            trades.append({
                "symbol": prefix,
                "entry_time": entry_time[prefix],
                "exit_time": final_ts,
                "entry_price": float(entry_price[prefix]),
                "exit_price": float(price_now),
                "pnl_pct": float(pnl_pct),
                "side": "LONG" if positions[prefix] == 1 else "SHORT",
            })

    # Stats
    if not equity_curve:
        return {
            "initial": equity0,
            "final": equity0,
            "return_pct": 0.0,
            "max_dd": 0.0,
        }, trades

    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak

    stats = {
        "initial": equity0,
        "final": float(eq[-1]),
        "return_pct": float(eq[-1] / equity0 - 1) * 100,
        "max_dd": float(dd.min()) * 100,
    }

    return stats, trades
