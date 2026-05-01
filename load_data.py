import pandas as pd
from sqlalchemy import create_engine
import urllib


# =========================================================
# DB ENGINE
# =========================================================
def get_engine():
    pipe = r"np:\\.\pipe\LOCALDB#6970DB69\tsql\query"

    params = urllib.parse.quote_plus(
        "DRIVER=ODBC Driver 17 for SQL Server;"
        f"SERVER={pipe};"
        "DATABASE=Candice;"
        "Trusted_Connection=Yes;"
    )

    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")


# =========================================================
# LOAD KLINES FOR ONE COIN
# =========================================================
def load_klines_for_coin(engine, coin_id, interval_id=6):
    """
    Loads OHLCV for a single coin using your stored procedure.
    Returns a DataFrame with columns:
        Timestamp, Open, High, Low, Close, Volume
    """
    query = f"EXEC [dbo].[GetKlines] {coin_id}, {interval_id}"
    df = pd.read_sql(query, engine)

    # Convert timestamp
    df["Timestamp"] = pd.to_datetime(df["KlineOpenTime"], unit="ms", utc=True)

    # Standardize column names
    rename_map = {
        "OpenPrice": "Open",
        "HighPrice": "High",
        "LowPrice": "Low",
        "ClosePrice": "Close",
        "Volume": "Volume",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]


# =========================================================
# LOAD ALL COINS (1–20)
# =========================================================
def load_all_klines(coin_ids=None):
    """
    Returns dict: { coin_id: df }
    """
    if coin_ids is None:
        coin_ids = list(range(1, 21))

    engine = get_engine()
    all_data = {}

    for cid in coin_ids:
        df = load_klines_for_coin(engine, cid)
        all_data[cid] = df

    return all_data


# =========================================================
# MERGE INTO WIDE DATAFRAME
# =========================================================
def load_klines():
    """
    Returns a wide DataFrame indexed by Timestamp with columns like:
        C1_Open, C1_High, C1_Low, C1_Close, C1_Volume,
        C2_Open, C2_High, ...
    This is the format required by features.py and the ML pipeline.
    """
    raw = load_all_klines()
    merged = None

    for cid, df in raw.items():
        prefix = f"C{cid}"

        # Rename OHLCV columns with prefix
        df = df.rename(columns={
            "Open": f"{prefix}_Open",
            "High": f"{prefix}_High",
            "Low": f"{prefix}_Low",
            "Close": f"{prefix}_Close",
            "Volume": f"{prefix}_Volume",
        })

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="Timestamp", how="outer")

    # Sort and index
    merged = merged.sort_values("Timestamp").reset_index(drop=True)
    merged = merged.set_index("Timestamp")

    # DO NOT create labels here — ML pipeline handles labels internally
    return merged
