# train.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# Example train_and_calibrate signature used by pipeline
def train_and_calibrate(train_df: pd.DataFrame):
    """
    Train per-prefix models and return (models_dict, feature_cols_dict).
    Expects wide feature_df with columns like <prefix>_<feature>.
    This implementation:
      - Detects prefixes by scanning columns ending with _Close
      - Trains a RandomForest per prefix using a simple target (next-period ret > 0)
      - Calibrates probabilities with CalibratedClassifierCV (robust to sklearn API)
    """
    # detect prefixes
    prefixes = sorted({c.rsplit("_", 1)[0] for c in train_df.columns if c.endswith("_Close")})
    models = {}
    feature_cols = {}

    for prefix in prefixes:
        # define features and target for this prefix
        close_col = f"{prefix}_Close"
        if close_col not in train_df.columns:
            continue

        # choose a small set of features automatically (all columns for this prefix except Close/Volume)
        cols = [c for c in train_df.columns if c.startswith(prefix + "_") and not c.endswith("_Close") and not c.endswith("_Volume")]
        if not cols:
            continue

        df = train_df[[close_col] + cols].dropna()
        if df.shape[0] < 200:
            # not enough data to train a per-prefix model
            continue

        # simple binary target: next period return positive
        df["target"] = df[close_col].pct_change().shift(-1) > 0
        df = df.dropna()
        if df["target"].nunique() < 2:
            continue

        X = df[cols].astype(float)
        y = df["target"].astype(int)

        # train/test split for calibration holdout
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.2, shuffle=False)

        base = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        base.fit(X_train, y_train)

        # Calibrate robustly across sklearn versions and fallbacks
        calib = None
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

        models[prefix] = calib
        feature_cols[prefix] = cols

    return models, feature_cols
