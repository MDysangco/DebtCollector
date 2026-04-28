import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_multi_horizon_models(df, feature_cols):
    models = {}

    horizons = {
        "h6":  "Label_6",
        "h24": "Label_24",
        "h72": "Label_72",
    }

    for prefix in sorted({c.split("_")[0] for c in df.columns if "_Close" in c}):
        print(f"\n=== Training models for {prefix} ===")

        label_cols = {
            h: f"{prefix}_{col}"
            for h, col in horizons.items()
        }

        # Ensure all label columns exist
        if not all(col in df.columns for col in label_cols.values()):
            print(f"Skipping {prefix} — missing horizon labels")
            continue

        # Mask rows where all horizons have labels
        mask = ~(df[label_cols["h6"]].isna() |
                 df[label_cols["h24"]].isna() |
                 df[label_cols["h72"]].isna())

        if mask.sum() < 200:
            print(f"Skipping {prefix} — insufficient training rows")
            continue

        X = df.loc[mask, feature_cols].values
        models[prefix] = {}

        for h, col in label_cols.items():
            y = df.loc[mask, col].values.astype(int)

            # Shift -1,0,1 → 0,1,2
            y = y + 1

            # Skip horizons with only one class
            if len(np.unique(y)) < 2:
                print(f"  Skipping {prefix} {h} — only one class present")
                continue

            clf = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist"
            )

            clf.fit(X, y)
            models[prefix][h] = clf

        # If no horizons trained, remove prefix
        if len(models[prefix]) == 0:
            print(f"Removing {prefix} — no valid horizons")
            del models[prefix]

    return models

