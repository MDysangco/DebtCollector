from typing import Optional, Dict, Any

# Walkforward windows
WF_TRAIN_DAYS: int = 180
WF_TEST_DAYS: int = 30
WF_STEP_DAYS: int = 30
MAX_FOLDS: Optional[int] = 3  # set to int for quick smoke runs

# Execution settings
GLOBAL_THRESHOLD = 0.34
PER_SYMBOL_FLOOR = 0.30
MARGIN = 0.00
COOLDOWN_HOURS = 48

# backtest behavior
TAKE_PROFIT = 0.08          # you already have something like this
STOP_LOSS = -0.05
MAX_HOLD_BARS = 48          # e.g. 2 days on 1h bars

# new upgrades
MIN_HOLD_BARS = 6           # e.g. 6 hours minimum
REENTRY_COOLDOWN_BARS = 6   # bars after exit before re-entry allowed
VOL_LOOKBACK = 24           # bars for volatility calc
VOL_THRESHOLD = 0.01        # min std dev of returns to allow entries
MAX_TRADES_PER_SYMBOL = 20  # per fold, per symbol

# trend exit
TREND_EMA_WINDOW_EXIT = 150
TREND_EXIT_BUFFER = 0.05
TREND_MIN_HOLD_BARS = 12

# ATR-based exits
ATR_WINDOW = 14            # ATR smoothing window (classic 14)
ATR_TP_MULT = 2.0          # take-profit = entry + 2 * ATR
ATR_SL_MULT = 2.0          # stop-loss = entry - 2 * ATR

# ATR-based volatility filter (optional)
USE_ATR_VOL_FILTER = False # set True to enable ATR volatility gating
ATR_VOL_MIN = 0.0          # min ATR% of price (e.g., 0.01 = 1%)
ATR_VOL_MAX = 1.0          # max ATR% of price (e.g., 0.10 = 10%)


# Imputation
IMPUTATION_METHOD: str = "median"  # options: "median", "ffill_bfill"

# Logging and debug
DEBUG_THRESHOLD_SAMPLE: int = 20
PRINT_SETTINGS: bool = True

# Per-symbol overrides (optional)
# Example: {"C1": 24, "C10": 48}
PER_SYMBOL_COOLDOWN: Dict[str, int] = {}

# Convenience accessor if you prefer a dict
def as_dict() -> Dict[str, Any]:
    return {
        "WF_TRAIN_DAYS": WF_TRAIN_DAYS,
        "WF_TEST_DAYS": WF_TEST_DAYS,
        "WF_STEP_DAYS": WF_STEP_DAYS,
        "MAX_FOLDS": MAX_FOLDS,
        "GLOBAL_THRESHOLD": GLOBAL_THRESHOLD,
        "PER_SYMBOL_FLOOR": PER_SYMBOL_FLOOR,
        "MARGIN": MARGIN,
        "COOLDOWN_HOURS": COOLDOWN_HOURS,
        "IMPUTATION_METHOD": IMPUTATION_METHOD,
        "DEBUG_THRESHOLD_SAMPLE": DEBUG_THRESHOLD_SAMPLE,
        "PRINT_SETTINGS": PRINT_SETTINGS,
        "PER_SYMBOL_COOLDOWN": PER_SYMBOL_COOLDOWN,
    }
