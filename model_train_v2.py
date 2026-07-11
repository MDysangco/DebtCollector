"""Model 2 training: XGBoost with balanced class weights and early stopping.

Balanced weights stop the dominant HOLD class from drowning out the BUY/SELL
classes; early stopping on a time-ordered tail split picks the tree count
instead of hard-coding it.
"""
import pandas as pd
import xgboost as xgb

from config import (
    XGB2_PARAMS, XGB2_REG_PARAMS, EARLY_STOPPING_ROUNDS, VAL_TAIL_FRAC,
)


def balanced_weights(y: pd.Series, soften: bool = False) -> pd.Series:
    """Inverse-frequency class weights; soften=True takes the square root,
    which corrects imbalance only partially and keeps the model from
    over-firing on the rare directional classes."""
    counts = y.value_counts()
    w = y.map(len(y) / (len(counts) * counts))
    return w.pow(0.5) if soften else w


def tail_time_split(X: pd.DataFrame, y: pd.Series, tail_frac: float = VAL_TAIL_FRAC):
    """Split off the last tail_frac of timestamps as a validation set."""
    ts = pd.Series(X.index.get_level_values("timestamp"))
    split_time = ts.quantile(1 - tail_frac)

    train_mask = X.index.get_level_values("timestamp") < split_time
    return X[train_mask], y[train_mask], X[~train_mask], y[~train_mask]


def train_model_v2(X_train, y_train, X_val=None, y_val=None, soften_weights=True):
    """Train with early stopping. If no explicit validation set is given,
    one is carved off the end of the training window (never train and
    early-stop on the same rows)."""
    if X_val is None or y_val is None or X_train.index.equals(X_val.index):
        X_train, y_train, X_val, y_val = tail_time_split(X_train, y_train)

    model = xgb.XGBClassifier(
        **XGB2_PARAMS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=balanced_weights(y_train, soften=soften_weights),
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def train_regressor_v2(X_train, y_train, X_val=None, y_val=None):
    """Regression on the vol-scaled forward return: keeps the magnitude
    information the 3-class labels throw away. The prediction (in sigmas)
    doubles as a natural signal score."""
    if X_val is None or y_val is None or X_train.index.equals(X_val.index):
        X_train, y_train, X_val, y_val = tail_time_split(X_train, y_train)

    model = xgb.XGBRegressor(
        **XGB2_REG_PARAMS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model
