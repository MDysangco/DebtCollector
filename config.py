# ============================================================
# LIVE SIGNAL / EXECUTION SETTINGS
# ============================================================
from numba.core.types import none

# Class probability thresholds
BUY_PROB_THRESHOLD = 0.55
SELL_PROB_THRESHOLD = 0.55
TREND_EMA_LENGTH = 50

# Volatility filter
VOL_FILTER_WINDOW = 12
VOL_MIN_THRESHOLD = 0.002

# ============================================================
# WALK-FORWARD SETTINGS
# ============================================================

WF_TRAIN_DAYS = 90
WF_TEST_DAYS = 30
WF_STEP_DAYS = 30
MAX_FOLDS = None


# ============================================================
# LABEL SETTINGS
# ============================================================

LABEL_HORIZON = 24
LABEL_UP_THRESH = 0.02
LABEL_DOWN_THRESH = -0.02


# ============================================================
# FEATURE SETTINGS
# ============================================================

RSI_LENGTH = 14
ATR_LENGTH = 14

MOM_WINDOWS = [8, 24]
VOL_WINDOWS = [8, 24]


# ============================================================
# EXECUTION / SIGNAL SETTINGS
# ============================================================

GLOBAL_THRESHOLD = 0.49
PER_SYMBOL_FLOOR = 0.53
MARGIN = 0.01
COOLDOWN_HOURS = 48

PER_SYMBOL_COOLDOWN = {}  # e.g. {"C1": 24}


# ============================================================
# MODEL TRAINING SETTINGS
# ============================================================

TRAIN_SPLIT_Q = 0.8

XGB_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "multi:softprob",
    "num_class": 3,
}


# ============================================================
# BACKTEST RISK MODEL
# ============================================================

STARTING_EQUITY = 10_000.0

MAX_PORTFOLIO_RISK_FRAC = 0.20
MAX_PER_COIN_RISK_FRAC = 0.05
RISK_PER_TRADE_FRAC = 0.005

FEE_BPS = 5
SLIPPAGE_BPS = 5


# ============================================================
# DB LOADER SETTINGS
# ============================================================

COIN_IDS = None
INTERVAL_ID = 6


# ============================================================
# DEBUG SETTINGS
# ============================================================

PRINT_SETTINGS = True
DEBUG_THRESHOLD_SAMPLE = 20


# ============================================================
# CONVENIENCE ACCESSOR
# ============================================================

def as_dict():
    return {k: v for k, v in globals().items() if k.isupper()}


