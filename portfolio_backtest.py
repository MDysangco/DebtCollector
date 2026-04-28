import numpy as np

FEE_RATE = 0.001  # 0.1% per side


def backtest_portfolio(df, signal_configs):
    equity = 10_000.0
    equity_curve = []

    positions = {prefix: 0 for prefix in signal_configs.keys()}
    entry_price = {prefix: None for prefix in signal_configs.keys()}
    entry_time = {prefix: None for prefix in signal_configs.keys()}

    trades = []

    for i in range(len(df) - 1):
        ts = df.index[i]

        for prefix, cfg in signal_configs.items():
            sig = int(cfg["full_final_signal"][i])
            close_col = f"{prefix}_Close"

            if close_col not in df.columns:
                continue

            price_now = df.iloc[i][close_col]
            price_next = df.iloc[i + 1][close_col]

            if (
                price_now is None
                or price_next is None
                or np.isnan(price_now)
                or np.isnan(price_next)
                or price_now <= 0
            ):
                continue

            # ENTRY
            if positions[prefix] == 0 and sig == 1:
                positions[prefix] = 1
                entry_price[prefix] = price_now
                entry_time[prefix] = ts

                equity *= (1 - FEE_RATE / len(signal_configs))

            # EXIT
            elif positions[prefix] == 1 and sig != 1:
                pnl_pct = (price_now - entry_price[prefix]) / entry_price[prefix]
                equity *= (1 + pnl_pct / len(signal_configs))
                equity *= (1 - FEE_RATE / len(signal_configs))

                trades.append({
                    "symbol": prefix,
                    "entry_time": entry_time[prefix],
                    "exit_time": ts,
                    "entry_price": float(entry_price[prefix]),
                    "exit_price": float(price_now),
                    "pnl_pct": float(pnl_pct),
                })

                positions[prefix] = 0
                entry_price[prefix] = None
                entry_time[prefix] = None

        equity_curve.append(equity)

    # Close open trades at final bar
    final_ts = df.index[-1]
    for prefix in positions:
        if positions[prefix] == 1:
            close_col = f"{prefix}_Close"
            price_now = df.iloc[-1][close_col]
            if price_now and entry_price[prefix]:
                pnl_pct = (price_now - entry_price[prefix]) / entry_price[prefix]
                equity *= (1 + pnl_pct / len(signal_configs))
                equity *= (1 - FEE_RATE / len(signal_configs))

                trades.append({
                    "symbol": prefix,
                    "entry_time": entry_time[prefix],
                    "exit_time": final_ts,
                    "entry_price": float(entry_price[prefix]),
                    "exit_price": float(price_now),
                    "pnl_pct": float(pnl_pct),
                })

    if not equity_curve:
        stats = {
            "initial": 10_000.0,
            "final": 10_000.0,
            "return_pct": 0.0,
            "max_dd": 0.0,
        }
    else:
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak

        stats = {
            "initial": 10_000.0,
            "final": float(eq[-1]),
            "return_pct": float(eq[-1] / 10_000.0 - 1) * 100,
            "max_dd": float(dd.min()) * 100,
        }

    return stats, trades
