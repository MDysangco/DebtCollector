# features.py
import pandas as pd
import numpy as np


def _detect_prefixes(df: pd.DataFrame) -> list:
    return sorted({c.rsplit("_", 1)[0] for c in df.columns if c.endswith("_Close")})


def _ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, length: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1 / length, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / length, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _stoch_rsi(series: pd.Series, length: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    rsi = _rsi(series, length)
    min_rsi = rsi.rolling(window=length, min_periods=1).min()
    max_rsi = rsi.rolling(window=length, min_periods=1).max()
    denom = (max_rsi - min_rsi).replace(0, np.nan)
    stoch = (rsi - min_rsi) / denom
    k = stoch.rolling(window=smooth_k, min_periods=1).mean()
    d = k.rolling(window=smooth_d, min_periods=1).mean()
    return k, d


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=length, min_periods=1).mean()
    return atr


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df[df["timestamp"].notna()].set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df[df.index.notna()]

    # Ensure numeric OHLCV columns are numeric
    ohlcv_suffixes = ("_Open", "_High", "_Low", "_Close", "_Volume")
    for c in df.columns:
        if c.endswith(ohlcv_suffixes):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    prefixes = _detect_prefixes(df)
    features = []

    for prefix in prefixes:
        close_col = f"{prefix}_Close"
        high_col = f"{prefix}_High"
        low_col = f"{prefix}_Low"
        open_col = f"{prefix}_Open"
        vol_col = f"{prefix}_Volume"

        if not {close_col, high_col, low_col, open_col, vol_col}.issubset(df.columns):
            continue

        close = df[close_col].astype(float)
        high = df[high_col].astype(float)
        low = df[low_col].astype(float)
        vol = df[vol_col].astype(float)

        ret1 = close.pct_change().fillna(0.0)
        ret6 = close.pct_change(periods=6).fillna(0.0)

        ema_fast = _ema(close, span=12)
        ema_slow = _ema(close, span=26)
        trend = (ema_fast > ema_slow).astype(int)

        atr = _atr(high, low, close, length=14)
        atr_norm = atr / close.replace(0, np.nan)

        rsi = _rsi(close, length=14)
        stoch_k, stoch_d = _stoch_rsi(close, length=14, smooth_k=3, smooth_d=3)

        macd_line, macd_signal, macd_hist = _macd(close, fast=12, slow=26, signal=9)

        ma20 = close.rolling(window=20, min_periods=1).mean()
        std20 = close.rolling(window=20, min_periods=1).std().replace(0, np.nan)
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        bb_pct = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

        vol_mean = vol.rolling(window=50, min_periods=1).mean()
        vol_std = vol.rolling(window=50, min_periods=1).std().replace(0, np.nan)
        vol_z = (vol - vol_mean) / vol_std

        median_atr = atr_norm.median(skipna=True)
        vol_regime = pd.Series(1, index=atr_norm.index)
        vol_regime = vol_regime.where(atr_norm.notna(), other=-1)
        if pd.notna(median_atr) and median_atr > 0:
            vol_regime = pd.cut(
                atr_norm,
                bins=[-np.inf, median_atr * 0.5, median_atr * 1.5, np.inf],
                labels=[0, 1, 2]
            ).astype("Int64").fillna(-1)

        f = pd.DataFrame(index=df.index)
        f[f"{prefix}_ret1"] = ret1
        f[f"{prefix}_ret6"] = ret6
        f[f"{prefix}_EMA_fast"] = ema_fast
        f[f"{prefix}_EMA_slow"] = ema_slow
        f[f"{prefix}_trend"] = trend
        f[f"{prefix}_ATR"] = atr
        f[f"{prefix}_ATR_norm"] = atr_norm
        f[f"{prefix}_RSI"] = rsi
        f[f"{prefix}_StochK"] = stoch_k
        f[f"{prefix}_StochD"] = stoch_d
        f[f"{prefix}_MACD"] = macd_line
        f[f"{prefix}_MACD_signal"] = macd_signal
        f[f"{prefix}_MACD_hist"] = macd_hist
        f[f"{prefix}_BB_pct"] = bb_pct
        f[f"{prefix}_vol_z"] = vol_z
        f[f"{prefix}_vol_regime"] = vol_regime
        f[f"{prefix}_Close"] = close
        f[f"{prefix}_Volume"] = vol

        features.append(f)

    if not features:
        return pd.DataFrame(index=df.index)

    feat_df = pd.concat(features, axis=1)
    feat_df = feat_df.dropna(how="all")

    close_cols = [c for c in feat_df.columns if c.endswith("_Close")]
    if close_cols:
        mask_any_close = feat_df[close_cols].notna().any(axis=1)
        feat_df = feat_df[mask_any_close]

    for c in feat_df.columns:
        if feat_df[c].dtype == "float64":
            feat_df[c] = feat_df[c].astype("float32")

    return feat_df
