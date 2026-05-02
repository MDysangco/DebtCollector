def build_backtest_signal_configs(signals_df, price_window):
    """
    Convert flat signals DataFrame into the nested dict structure
    expected by backtest_portfolio().

    Output format:
    {
        "C1":  {"full_final_signal": [...]},
        "C10": {"full_final_signal": [...]},
        ...
    }
    """

    idx = price_window.index

    # Extract prefixes from price columns
    prefixes = sorted({c.split("_")[0] for c in price_window.columns})

    # Initialize output dict
    signal_configs = {p: {"full_final_signal": [0] * len(idx)} for p in prefixes}

    # Your execution logic uses "final_signal"
    signal_col = "final_signal"

    if signal_col not in signals_df.columns:
        print("[WF DEBUG] available signal columns:", signals_df.columns.tolist())
        raise ValueError(f"Expected signal column '{signal_col}' not found in signals_df")

    # Build fast lookup: (timestamp, symbol) -> signal
    sig_map = {
        (row.timestamp, row.symbol): int(getattr(row, signal_col))
        for row in signals_df.itertuples()
    }

    # Fill per-prefix arrays aligned to price_window index
    for i, ts in enumerate(idx):
        for prefix in prefixes:
            key = (ts, prefix)
            if key in sig_map:
                signal_configs[prefix]["full_final_signal"][i] = sig_map[key]

    return signal_configs
