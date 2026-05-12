import pandas as pd

from ZypryxApi import ZypryxApi
from config import INTERVAL_ID


def load_price_data_from_api(client: ZypryxApi, coin_ids=None, interval_id=None, start=None, end=None):
    if interval_id is None:
        interval_id = INTERVAL_ID

    if coin_ids is None:
        coin_ids

    frames = []

    # Pull klines per coin via API
    for cid in coin_ids:
        kl = client.get_klines(cid, interval_id, start, end)
        if not kl:
            print(f"[WARN] Coin {cid} returned NO rows. Skipping.")
            continue

        df = pd.DataFrame(kl)

        df["Timestamp"] = pd.to_datetime(df["klineOpenTime"], unit="ms", utc=True)

        df = df.rename(columns={
            "openPrice": "Open",
            "highPrice": "High",
            "lowPrice": "Low",
            "closePrice": "Close",
            "volume": "Volume",
        })

        df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
        df = df.drop_duplicates(subset=["Timestamp"])

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
        raise ValueError("No valid coins returned any klines.")

    # Merge all coins
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="Timestamp", how="outer")

    merged = merged.sort_values("Timestamp").set_index("Timestamp")
    merged = merged[~merged.index.duplicated(keep="first")]

    # Align start window
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