import pandas as pd
import urllib
from sqlalchemy import create_engine

from config import COIN_IDS, INTERVAL_ID


# ---------------------------------------------------------
# SHARED ENGINE (TCP, stable, no Named Pipes)
# ---------------------------------------------------------
def get_engine():
    params = urllib.parse.quote_plus(
        "DRIVER=ODBC Driver 17 for SQL Server;"
        "SERVER=(localdb)\\Development;"
        "DATABASE=Candice;"
        "Trusted_Connection=Yes;"
        "Connection Timeout=30;"
    )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")


# ---------------------------------------------------------
# MAIN PRICE LOADER
# ---------------------------------------------------------
def load_price_data_from_db(coin_ids=None, interval_id=None):
    if interval_id is None:
        interval_id = INTERVAL_ID

    engine = get_engine()

    # 1. Pull ALL klines in one call
    query = f"EXEC [dbo].[GetKlines] NULL, {interval_id}"
    df = pd.read_sql(query, engine)

    if df.empty:
        raise ValueError("GetKlines returned NO data.")

    # 2. Convert timestamp
    df["Timestamp"] = pd.to_datetime(df["KlineOpenTime"], unit="ms", utc=True)

    # 3. Standardize column names
    df = df.rename(columns={
        "OpenPrice": "Open",
        "HighPrice": "High",
        "LowPrice": "Low",
        "ClosePrice": "Close",
        "Volume": "Volume",
    })

    # 4. Filter only requested coins (if coin_ids provided)
    if isinstance(coin_ids, (list, tuple, set)) and len(coin_ids) > 0:
        df = df[df["CoinId"].isin(coin_ids)]

    # 5. Pivot into wide format
    wide = df.pivot_table(
        index="Timestamp",
        columns="CoinId",
        values=["Open", "High", "Low", "Close", "Volume"]
    )

    # Flatten MultiIndex columns
    wide.columns = [f"C{cid}_{col}" for col, cid in wide.columns]

    # 6. Sort, drop duplicates
    wide = wide.sort_values("Timestamp")
    wide = wide[~wide.index.duplicated(keep="first")]

    # 7. Align start window
    first_valids = []
    for col in wide.columns:
        fv = wide[col].first_valid_index()
        if fv is not None:
            first_valids.append(fv)

    start_ts = max(first_valids)
    wide = wide.loc[start_ts:]

    print(f"[INFO] Aligned window starts at: {start_ts}")
    print(f"[INFO] Using {len(wide.columns) // 5} valid coins")

    return wide
