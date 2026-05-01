import pandas as pd

from load_data import load_klines
from features import build_features

df = load_klines()
feat = build_features(df)

train_start = feat.index.min()
train_end   = train_start + pd.Timedelta(days=90)
test_start  = train_end
test_end    = test_start + pd.Timedelta(days=30)

test_df = feat.loc[test_start:test_end]

print("Non-NaN counts per column:")
print(test_df.notna().sum().sort_values())