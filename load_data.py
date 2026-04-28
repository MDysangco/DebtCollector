import pandas as pd
from sqlalchemy import create_engine
import urllib


def get_engine():
    pipe = r"np:\\.\pipe\LOCALDB#7A4911B8\tsql\query"

    params = urllib.parse.quote_plus(
        "DRIVER=ODBC Driver 17 for SQL Server;"
        f"SERVER={pipe};"
        "DATABASE=Candice;"
        "Trusted_Connection=Yes;"
    )

    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")


def load_klines_for_coin(engine, coin_id, interval_id=6):
    query = f"EXEC [dbo].[GetKlines] {coin_id}, {interval_id}"
    df = pd.read_sql(query, engine)
    df["Timestamp"] = pd.to_datetime(df["KlineOpenTime"], unit="ms", utc=True)

    rename_map = {
        "OpenPrice": "Open",
        "HighPrice": "High",
        "LowPrice": "Low",
        "ClosePrice": "Close",
        "Volume": "Volume",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def load_all_klines(coin_ids=None):
    if coin_ids is None:
        coin_ids = list(range(1, 21))

    engine = get_engine()
    all_data = {}

    for cid in coin_ids:
        df = load_klines_for_coin(engine, cid)
        all_data[cid] = df

    return all_data


def load_klines():
    raw = load_all_klines()
    merged = None

    for cid, df in raw.items():
        prefix = f"C{cid}"

        base_cols = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
        df = df[base_cols].copy()

        df = df.rename(columns={col: f"{prefix}_{col}" for col in df.columns if col != "Timestamp"})

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="Timestamp", how="outer")

    merged = merged.sort_values("Timestamp").reset_index(drop=True)
    merged = merged.set_index("Timestamp")

    # Label creation
    for cid in raw.keys():
        prefix = f"C{cid}"
        close_col = f"{prefix}_Close"

        if close_col not in merged.columns:
            continue

        future_ret = merged[close_col].pct_change().shift(-1)

        # Only label rows with a valid future return
        mask = future_ret.notna()
        valid_ret = future_ret[mask]

        labels = pd.cut(
            valid_ret,
            bins=[-999, -0.002, 0.002, 999],
            labels=[0, 1, 2],  # 0=sell, 1=hold, 2=buy
        )

        merged[f"{prefix}_Label"] = pd.NA
        merged.loc[mask, f"{prefix}_Label"] = labels.astype("Int64")


    return merged
