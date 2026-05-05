import pandas as pd
import numpy as np

from config import (
    STARTING_EQUITY,
    MAX_PORTFOLIO_RISK_FRAC,
    MAX_PER_COIN_RISK_FRAC,
    RISK_PER_TRADE_FRAC,
    FEE_BPS,
    SLIPPAGE_BPS,
)


def backtest_portfolio(price_df, signals):
    equity = STARTING_EQUITY
    positions = {}
    trades = []

    if isinstance(signals, list):
        signals = pd.DataFrame(signals)

    signals = signals.sort_values("timestamp")

    for _, row in signals.iterrows():
        ts = row["timestamp"]
        sym = row["symbol"]
        side = row["side"]
        price = float(row["price"])

        if side == "BUY":
            exec_price = price * (1 + SLIPPAGE_BPS / 10_000)
        else:
            exec_price = price * (1 - SLIPPAGE_BPS / 10_000)

        total_exposure = sum(pos["size"] * pos["entry_price"] for pos in positions.values())
        coin_exposure = positions.get(sym, {}).get("size", 0) * positions.get(sym, {}).get("entry_price", 0)

        if side == "BUY":
            if sym in positions:
                continue

            atr = float(row["atr"]) if "atr" in row and not pd.isna(row["atr"]) else None

            if atr and atr > 0:
                risk_amount = equity * RISK_PER_TRADE_FRAC
                size = risk_amount / atr
            else:
                size = (equity * RISK_PER_TRADE_FRAC) / exec_price

            new_coin_exposure = coin_exposure + size * exec_price
            if new_coin_exposure > equity * MAX_PER_COIN_RISK_FRAC:
                allowed = equity * MAX_PER_COIN_RISK_FRAC - coin_exposure
                if allowed <= 0:
                    continue
                size = allowed / exec_price

            new_total_exposure = total_exposure + size * exec_price
            if new_total_exposure > equity * MAX_PORTFOLIO_RISK_FRAC:
                allowed = equity * MAX_PORTFOLIO_RISK_FRAC - total_exposure
                if allowed <= 0:
                    continue
                size = allowed / exec_price

            if size <= 0:
                continue

            fee = size * exec_price * FEE_BPS / 10_000
            equity -= fee

            positions[sym] = {"size": size, "entry_price": exec_price}

            trades.append({
                "timestamp": ts,
                "side": "BUY",
                "price": exec_price,
                "size": size,
                "symbol": sym,
            })

        elif side == "SELL":
            if sym not in positions:
                continue

            pos = positions.pop(sym)
            size = pos["size"]
            entry_price = pos["entry_price"]

            pnl = (exec_price - entry_price) * size
            fee = size * exec_price * FEE_BPS / 10_000
            pnl -= fee

            equity += pnl

            trades.append({
                "timestamp": ts,
                "side": "SELL",
                "price": exec_price,
                "size": size,
                "pnl": pnl,
                "equity_after": equity,
                "symbol": sym,
            })

    if trades:
        eq_series = pd.Series([STARTING_EQUITY] + [t.get("equity_after", np.nan) for t in trades]).ffill()
        ret_pct = equity / STARTING_EQUITY - 1
        max_dd = (eq_series / eq_series.cummax() - 1).min()
    else:
        ret_pct = 0
        max_dd = 0

    stats = {
        "return_pct": np.float64(ret_pct),
        "max_dd": np.float64(max_dd),
        "n_trades": len(trades),
    }

    return stats, trades
