# train.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

def train_and_calibrate(train_df: pd.DataFrame):
    """
    Train per-prefix models and return:
      models: dict[prefix] -> calibrated model
      feature_cols: dict[prefix] -> list of feature column names
      thresholds: dict[prefix] -> per-prefix probability threshold (75th percentile on calib set)
      medians: dict[prefix] -> dict(feature -> median) for imputation on test
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

        cols = [c for c in train_df.columns if c.startswith(prefix + "_") and not c.endswith("_Close") and not c.endswith("_Volume")]
        if not cols:
            continue

        df = train_df[[close_col] + cols].copy()
        # drop rows where close is NaN (no price)
        df = df[df[close_col].notna()]
        if df.shape[0] < 200:
            continue

        # target: next period return > 0
        df["target"] = df[close_col].pct_change().shift(-1) > 0
        df = df.dropna()
        if df["target"].nunique() < 2:
            continue

        X = df[cols].astype(float)
        y = df["target"].astype(int)

        # store medians for imputation on test
        med = X.median()
        medians[prefix] = med.to_dict()

        # train/calibration split (time-ordered)
        split_idx = int(len(X) * 0.8)
        X_train, X_calib = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_calib = y.iloc[:split_idx], y.iloc[split_idx:]

        if X_train.shape[0] < 50 or X_calib.shape[0] < 20:
            # fallback to a single fit if calibration split too small
            X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.2, shuffle=False)

        base = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        base.fit(X_train, y_train)

        # Calibrate robustly across sklearn versions and fallbacks
        try:
            calib = CalibratedClassifierCV(estimator=base, method="sigmoid", cv="prefit")
            calib.fit(X_calib, y_calib)
        except TypeError:
            try:
                calib = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv="prefit")
                calib.fit(X_calib, y_calib)
            except Exception:
                calib = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
                calib.fit(X_train, y_train)
        except Exception:
            calib = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
            calib.fit(X_train, y_train)

        # compute a conservative per-symbol threshold from calibration set
        try:
            calib_probs = calib.predict_proba(X_calib)[:, 1]
            per_symbol_threshold = float(np.quantile(calib_probs, 0.75))
        except Exception:
            per_symbol_threshold = 0.5

        models[prefix] = calib
        feature_cols[prefix] = cols
        thresholds[prefix] = per_symbol_threshold

        # lightweight training report
        try:
            pos_frac = float(y.mean())
        except Exception:
            pos_frac = 0.0
        print(f"TRAIN: prefix={prefix}, rows={len(df)}, pos_frac={pos_frac:.3f}, calib_thresh={per_symbol_threshold:.3f}")

    return models, feature_cols, thresholds, medians
