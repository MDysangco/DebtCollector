"""Market data loading via the Zypryx API.

Single home for turning API kline responses into the wide price DataFrame
(C{id}_Open/High/Low/Close/Volume columns indexed by Timestamp) that the
feature builders, live prediction, and backtests all consume.
"""
import asyncio

import pandas as pd

import config
from ZypryxApi import ZypryxApi


async def load_coin_ids(api: ZypryxApi):
    coins = await api.get_active_coins()
    if not coins:
        raise ValueError("No active coins returned from API.")
    return [c["Id"] for c in coins]


async def load_price_data_from_api(api: ZypryxApi, coin_ids, interval_id):
    frames = []

    for cid in coin_ids:
        kl = await api.get_klines(cid, interval_id)
        if not kl:
            print(f"[WARN] Coin {cid} returned NO klines.")
            continue

        df = pd.DataFrame(kl)

        df = df.rename(columns={
            "KlineOpenTime": "Timestamp",
            "OpenPrice": "Open",
            "HighPrice": "High",
            "LowPrice": "Low",
            "ClosePrice": "Close",
            "Volume": "Volume",
        })

        df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype("int64"), unit="ms", utc=True)
        df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]

        prefix = f"C{cid}"
        df = df.rename(columns={
            "Open": f"{prefix}_Open",
            "High": f"{prefix}_High",
            "Low": f"{prefix}_Low",
            "Close": f"{prefix}_Close",
            "Volume": f"{prefix}_Volume",
        })

        frames.append(df)

    if not frames:
        raise ValueError("No coins returned klines.")

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="Timestamp", how="outer")

    merged = merged.sort_values("Timestamp").set_index("Timestamp")
    merged = merged[~merged.index.duplicated(keep="first")]

    first_valids = []
    for col in merged.columns:
        fv = merged[col].first_valid_index()
        if fv is not None:
            first_valids.append(fv)

    start_ts = max(first_valids)
    merged = merged.loc[start_ts:]

    print(f"[INFO] Aligned window starts at: {start_ts}")
    print(f"[INFO] Using {len(merged.columns) // 5} valid coins")

    return merged


def load_price_data(coin_ids=None, interval_id=None):
    """Synchronous convenience wrapper for scripts and notebooks."""
    if interval_id is None:
        interval_id = config.INTERVAL_ID

    async def _fetch():
        async with ZypryxApi(config.API_URL, config.API_TOKEN) as api:
            ids = coin_ids or await load_coin_ids(api)
            return await load_price_data_from_api(api, ids, interval_id)

    return asyncio.run(_fetch())
