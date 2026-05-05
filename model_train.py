import pandas as pd
import xgboost as xgb

from config import XGB_PARAMS, TRAIN_SPLIT_Q


def time_split(df, split_time):
    train = df.loc[df.index.get_level_values("timestamp") < split_time]
    val = df.loc[df.index.get_level_values("timestamp") >= split_time]

    X_train = train.drop(columns=["label"])
    y_train = train["label"]

    X_val = val.drop(columns=["label"])
    y_val = val["label"]

    return X_train, y_train, X_val, y_val


def train_model(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model
