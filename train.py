import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


def train_and_calibrate(train_df: pd.DataFrame):
    """
    Train per-prefix binary XGBoost models.
    Returns:
      models: dict[prefix] -> calibrated model
      feature_cols: dict[prefix] -> list of feature column names
      thresholds: dict[prefix] -> dict(long=..., short=...)
      medians: dict[prefix] -> dict(feature -> median)
    """

    prefixes = sorted({c.rsplit("_", 1)[0] for c in train_df.columns if c.endswith("_Close")})

    models = {}
    feature_cols = {}
    thresholds = {}
    medians = {}

    for prefix in prefixes:

        close_col = f"{prefix}_Close"
        if close_col not in train_df.columns:
            continue

        # all feature columns for this prefix (excluding Close/Volume)
        cols = [
            c for c in train_df.columns
            if c.startswith(prefix + "_")
            and not c.endswith("_Close")
            and not c.endswith("_Volume")
        ]
        if not cols:
            continue

        # skip prefixes with no usable data
        if train_df[cols].notna().sum().sum() == 0:
            continue

        df = train_df[[close_col] + cols].copy()
        df = df[df[close_col].notna()]
        if df.shape[0] < 200:
            continue

        # --- LABEL: binary long vs not-long ---
        horizon = 6
        future_close = df[close_col].shift(-horizon)
        df["future_ret"] = (future_close - df[close_col]) / df[close_col]

        atr_col = f"{prefix}_ATR"  # use true ATR, not ATR_norm
        if atr_col not in df.columns:
            continue

        atr = df[atr_col].replace(0, np.nan)

        upper = 0.25 * atr
        lower = -0.25 * atr

        df["target"] = 0
        df.loc[df["future_ret"] > upper, "target"] = 1

        df = df.dropna(subset=["future_ret", atr_col])
        if df["target"].sum() < 20:
            continue

        # label diagnostics
        pos_frac = (df["target"] == 1).mean()
        neg_frac = (df["target"] == 0).mean()
        print(f"[LABEL DIAG] {prefix}: pos={pos_frac:.3f}, neg={neg_frac:.3f}")

        X = df[cols].astype(float)
        y = df["target"].astype(int)

        # store medians for test-time imputation
        med = X.median()
        medians[prefix] = med.to_dict()

        # time-ordered split
        split_idx = int(len(X) * 0.8)
        X_train, X_calib = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_calib = y.iloc[:split_idx], y.iloc[split_idx:]

        if X_train.shape[0] < 50 or X_calib.shape[0] < 20:
            X_train, X_calib, y_train, y_calib = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

        # --- XGBoost model ---
        base = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
        base.fit(X_train, y_train)

        # --- Calibration ---
        try:
            calib = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
            calib.fit(X_calib, y_calib)
        except Exception:
            calib = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            calib.fit(X_train, y_train)

        # --- Thresholds ---
        try:
            probs = calib.predict_proba(X_calib)[:, 1]
            long_thresh = float(np.quantile(probs, 0.75))
        except Exception:
            long_thresh = 0.5

        thresholds[prefix] = {
            "long": long_thresh,
            "short": 0.0,  # binary model does not produce short signals
        }

        models[prefix] = calib
        feature_cols[prefix] = cols

        print(
            f"TRAIN: prefix={prefix}, rows={len(df)}, pos={pos_frac:.3f}, "
            f"long_thresh={long_thresh:.3f}"
        )

    return models, feature_cols, thresholds, medians
