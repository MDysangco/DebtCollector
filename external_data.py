"""External market-context data for Model 2.

Free, keyless daily sources:
  - Crypto Fear & Greed index (alternative.me)
  - FRED series: VIX, S&P 500, broad USD index, 10y-2y treasury spread

Every series is cached to data_cache/ so live runs survive source outages;
a stale cache is better than no features (XGBoost handles NaN natively).
"""
import time
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
CACHE_MAX_AGE_HOURS = 6
REQUEST_TIMEOUT = 20

# Cap on how far a daily value may be carried forward. If a source dies for
# longer than this, its features go NaN instead of silently feeding the model
# month-old macro data.
MAX_FFILL_DAYS = 7

FRED_SERIES = {
    "vix": "VIXCLS",       # equity fear gauge
    "spx": "SP500",        # broad risk appetite
    "dxy": "DTWEXBGS",     # broad USD index (strong dollar = crypto headwind)
    "t10y2y": "T10Y2Y",    # yield-curve spread (recession/crisis signal)
}


# ---------------------------------------------------------
# CACHING
# ---------------------------------------------------------
def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.csv"


def _cache_is_fresh(name: str) -> bool:
    p = _cache_path(name)
    return p.exists() and (time.time() - p.stat().st_mtime) < CACHE_MAX_AGE_HOURS * 3600


def _read_cache(name: str):
    p = _cache_path(name)
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True)


def _write_cache(name: str, df: pd.DataFrame):
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(_cache_path(name))


def _fetch_or_cache(name: str, fetch_fn):
    if _cache_is_fresh(name):
        return _read_cache(name)
    try:
        df = fetch_fn()
        _write_cache(name, df)
        return df
    except Exception as ex:
        print(f"[WARN] External source '{name}' failed ({ex}); falling back to cache.")
        return _read_cache(name)


# ---------------------------------------------------------
# FETCHERS
# ---------------------------------------------------------
def _fetch_fear_greed() -> pd.DataFrame:
    res = requests.get(
        "https://api.alternative.me/fng/?limit=0&format=json",
        timeout=REQUEST_TIMEOUT,
    )
    res.raise_for_status()
    rows = res.json()["data"]

    idx = pd.to_datetime([int(r["timestamp"]) for r in rows], unit="s")
    vals = pd.to_numeric([r["value"] for r in rows], errors="coerce")
    return pd.DataFrame({"fng": vals}, index=idx).sort_index()


def _fetch_fred(series_id: str, col: str) -> pd.DataFrame:
    res = requests.get(
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
        timeout=REQUEST_TIMEOUT,
    )
    res.raise_for_status()

    from io import StringIO
    raw = pd.read_csv(StringIO(res.text))
    # First column is the observation date, second the value ('' on holidays).
    raw.columns = ["date", col]
    raw["date"] = pd.to_datetime(raw["date"])
    raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw.set_index("date")[[col]].sort_index()


# ---------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------
def load_external_daily() -> pd.DataFrame:
    """Daily DataFrame with columns fng, vix, spx, dxy, t10y2y (NaN-tolerant)."""
    frames = []

    fng = _fetch_or_cache("fng", _fetch_fear_greed)
    if fng is not None:
        frames.append(fng)

    for col, sid in FRED_SERIES.items():
        df = _fetch_or_cache(col, lambda sid=sid, col=col: _fetch_fred(sid, col))
        if df is not None:
            frames.append(df)

    if not frames:
        print("[WARN] No external data available at all; Model 2 runs on price features only.")
        return pd.DataFrame()

    daily = pd.concat(frames, axis=1, sort=True)

    # Fill weekend/holiday gaps but cap staleness.
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx).ffill(limit=MAX_FFILL_DAYS)
    return daily


if __name__ == "__main__":
    df = load_external_daily()
    print(df.tail(10))
    print(f"\n{len(df)} days, columns: {list(df.columns)}")
