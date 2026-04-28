def rolling_coin_selection(trades, window_days=90, rebalance_days=30,
                           min_trades=1, min_expectancy=0.0):
    import pandas as pd

    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df = df.sort_values("entry_time")

    start = df["entry_time"].min()
    end = df["entry_time"].max()

    approved_by_period = []

    current = start

    while current < end:
        window_end = current
        window = df[df["entry_time"] < window_end]

        per_symbol = {}
        for _, t in window.iterrows():
            per_symbol.setdefault(t["symbol"], []).append(t["pnl_pct"])

        approved = []
        for sym, pnl_list in per_symbol.items():
            if len(pnl_list) < min_trades:
                continue
            expectancy = sum(pnl_list) / len(pnl_list)
            if expectancy >= min_expectancy:
                approved.append(sym)

        approved_by_period.append({
            "rebalance_time": current,
            "approved": approved
        })

        current += pd.Timedelta(days=rebalance_days)

    return approved_by_period


def filter_coins_by_expectancy(trades, min_trades=3, min_expectancy=0.0):
    per_symbol = {}

    for t in trades:
        sym = t["symbol"]
        per_symbol.setdefault(sym, []).append(t["pnl_pct"])

    approved = []
    dropped = []

    for sym, pnl_list in per_symbol.items():
        if len(pnl_list) < min_trades:
            dropped.append((sym, "too_few_trades"))
            continue

        expectancy = sum(pnl_list) / len(pnl_list)

        if expectancy >= min_expectancy:
            approved.append(sym)
        else:
            dropped.append((sym, f"expectancy={expectancy:.4f}"))

    return approved, dropped


def print_portfolio_report(stats, trades):
    print("=== PORTFOLIO REPORT ===")
    print(f"Initial equity: {stats['initial']:.2f}")
    print(f"Final equity:   {stats['final']:.2f}")
    print(f"Return:         {stats['return_pct']:.2f}%")
    print(f"Max Drawdown:   {stats['max_dd']:.2f}%")
    print()

    print(f"Total trades:   {len(trades)}")

    if len(trades) == 0:
        print("No trades executed.")
        return

    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]

    print(f"Win rate:       {len(wins) / len(trades) * 100:.2f}%")
    print(f"Avg win:        {sum(t['pnl_pct'] for t in wins) / len(wins) * 100:.2f}%"
          if wins else "Avg win:        N/A")
    print(f"Avg loss:       {sum(t['pnl_pct'] for t in losses) / len(losses) * 100:.2f}%"
          if losses else "Avg loss:       N/A")
    print()

    print("Sample trades (first 5):")
    for t in trades[:5]:
        print(f"{t['symbol']} | {t['entry_time']} → {t['exit_time']} | "
              f"{t['pnl_pct']*100:.2f}%")

    # -----------------------------
    # Per‑symbol performance
    # -----------------------------
    print("\nPer‑symbol performance:")
    per_symbol = {}

    for t in trades:
        sym = t["symbol"]
        per_symbol.setdefault(sym, []).append(t["pnl_pct"])

    for sym, pnl_list in per_symbol.items():
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]

        print(
            f"{sym}: "
            f"Trades={len(pnl_list)}, "
            f"WinRate={len(wins) / len(pnl_list) * 100:.1f}%, "
            f"AvgPnL={sum(pnl_list) / len(pnl_list) * 100:.2f}%"
        )

    # -----------------------------
    # Export trades to CSV
    # -----------------------------
    import pandas as pd
    df_trades = pd.DataFrame(trades)
    df_trades.to_csv("trades.csv", index=False)
    print("\nTrades exported to trades.csv")
