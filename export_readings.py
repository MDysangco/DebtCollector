import argparse
import asyncio
import csv
import os
from datetime import datetime, timezone

import config
from ZypryxApi import ZypryxApi

BACKUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


async def run(args):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

    async with ZypryxApi(config.API_URL, config.API_TOKEN) as api:
        coins = await api.get_all_coins()
        if not coins:
            raise ValueError("No coins returned from API.")

        coin_ids = [c["Id"] for c in coins]
        if args.coin_id is not None:
            coin_ids = [cid for cid in coin_ids if cid == args.coin_id]

        rows = []
        for cid in coin_ids:
            readings = await api.get_readings(cid)
            if readings:
                rows.extend(readings)
            print(f"  coin {cid}: {len(readings or [])} readings")

    if not rows:
        print("No readings to export.")
        return

    columns = sorted({k for r in rows for k in r})
    path = os.path.join(BACKUP_DIR, f"backup_readings_{stamp}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nExported {len(rows)} readings -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin-id", type=int, default=None,
                    help="export a single coin id")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
