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
