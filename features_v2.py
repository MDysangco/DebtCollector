"""Model 2 features and labels.

Adds to Model 1's per-coin technicals:
  - Market regime: equal-weight market return, breadth, volatility regime.
  - Cross-sectional strength: momentum rank vs other coins, return z-score,
    rolling beta/correlation to the market.
  - External context: Fear & Greed, VIX, S&P 500, USD index, yield curve
    (daily, forward-filled onto bars, lagged so nothing is known early).
  - Calendar: hour-of-day / day-of-week seasonality.

Labels are volatility-scaled instead of fixed +/-2%.
"""
import numpy as np
import pandas as pd

from features_and_labels import detect_symbols, build_features_for_symbol
from external_data import load_external_daily
import config as cfg

# Feature groups, for ablations and for picking each ensemble member's view.
# Levels of slow daily series barely move within a training window, so trees
# can use them to memorize the period ("regime memorization"); the change
# features carry the news without indexing time.
LEVEL_COLS = ["fng", "vix", "t10y2y"]
EXTERNAL_COLS = [
    "fng", "fng_chg_7d", "vix", "vix_chg_5d", "spx_ret_1d", "spx_ret_5d",
    "dxy_ret_5d", "t10y2y", "t10y2y_chg_20d",
    "rsi_x_fng",  # depends on external data, so price-only variants drop it
]
CALENDAR_COLS = ["hour_sin", "hour_cos", "dow"]

# Reversal features (added July 2026): dedicated bottom/top-catching signals.
# Kept in their own group so comparison variants can isolate their effect.
REVERSAL_COLS = [
    "dist_low_72", "dist_low_168", "range_pos_168", "stretch_24", "rsi_x_fng",
]


# ---------------------------------------------------------
# MARKET + CROSS-SECTIONAL FEATURES
# ---------------------------------------------------------
def _close_matrix(price_df, symbols):
    closes = price_df[[f"{s}_Close" for s in symbols]].copy()
    closes.columns = symbols
    return closes


def build_market_features(price_df, symbols):
    """Features shared by every coin: what is the market as a whole doing?"""
    closes = _close_matrix(price_df, symbols)
    ret1 = closes.pct_change()
    mkt_ret1 = ret1.mean(axis=1)

    out = pd.DataFrame(index=price_df.index)
    for w in cfg.MKT_RET_WINDOWS:
        out[f"mkt_ret_{w}"] = closes.pct_change(w).mean(axis=1)

    ema = closes.ewm(span=cfg.TREND_EMA_LENGTH, adjust=False).mean()
    out["breadth_ema"] = (closes > ema).mean(axis=1)
    out["breadth_mom24"] = (closes.pct_change(24) > 0).mean(axis=1)

    out["mkt_vol_24"] = mkt_ret1.rolling(24).std()
    out["vol_regime"] = out["mkt_vol_24"] / out["mkt_vol_24"].rolling(cfg.REGIME_WINDOW).mean()

    return out, ret1, mkt_ret1


def build_cross_features(closes, ret1, mkt_ret1):
    """Per-coin features relative to the rest of the universe.

    Returns {feature_name: DataFrame with one column per symbol}.
    """
    cross = {}

    for w in cfg.CROSS_MOM_WINDOWS:
        mom = closes.pct_change(w)
        cross[f"mom_rank_{w}"] = mom.rank(axis=1, pct=True)

    mom24 = closes.pct_change(24)
    cross["ret_zs_24"] = mom24.sub(mom24.mean(axis=1), axis=0).div(
        mom24.std(axis=1), axis=0
    )

    cross["corr_mkt"] = ret1.rolling(cfg.BETA_WINDOW).corr(mkt_ret1)
    cov = ret1.rolling(cfg.BETA_WINDOW).cov(mkt_ret1)
    cross["beta_mkt"] = cov.div(mkt_ret1.rolling(cfg.BETA_WINDOW).var(), axis=0)

    cross["drawdown_72"] = closes / closes.rolling(72).max() - 1
    rng_hi = closes.rolling(24).max()
    rng_lo = closes.rolling(24).min()
    cross["range_pos_24"] = (closes - rng_lo) / (rng_hi - rng_lo)

    # Reversal features: how stretched is the coin relative to its own
    # recent lows/range/volatility (bottom- and top-catching signals).
    cross["dist_low_72"] = closes / closes.rolling(72).min() - 1
    cross["dist_low_168"] = closes / closes.rolling(168).min() - 1
    rng_hi_w = closes.rolling(168).max()
    rng_lo_w = closes.rolling(168).min()
    cross["range_pos_168"] = (closes - rng_lo_w) / (rng_hi_w - rng_lo_w)
    cross["stretch_24"] = closes.pct_change(24).div(
        ret1.rolling(168).std() * np.sqrt(24))

    return cross


# ---------------------------------------------------------
# EXTERNAL CONTEXT FEATURES
# ---------------------------------------------------------
def build_external_features(bar_index):
    """Daily macro/sentiment series aligned onto the bar index.

    FRED series are values as of market close, so they are lagged one full
    day before alignment; Fear & Greed is stamped at 00:00 UTC of its own
    day and needs no lag. Missing sources simply yield NaN columns.
    """
    daily = load_external_daily()
    if daily.empty:
        return pd.DataFrame(index=bar_index)

    feats = pd.DataFrame(index=daily.index)
    if "fng" in daily:
        feats["fng"] = daily["fng"]
        feats["fng_chg_7d"] = daily["fng"].diff(7)
    if "vix" in daily:
        feats["vix"] = daily["vix"].shift(1)
        feats["vix_chg_5d"] = daily["vix"].diff(5).shift(1)
    if "spx" in daily:
        feats["spx_ret_1d"] = daily["spx"].pct_change().shift(1)
        feats["spx_ret_5d"] = daily["spx"].pct_change(5).shift(1)
    if "dxy" in daily:
        feats["dxy_ret_5d"] = daily["dxy"].pct_change(5).shift(1)
    if "t10y2y" in daily:
        feats["t10y2y"] = daily["t10y2y"].shift(1)
        feats["t10y2y_chg_20d"] = daily["t10y2y"].diff(20).shift(1)

    feats.index = feats.index.tz_localize("UTC")
    union = feats.index.union(bar_index)
    return feats.reindex(union).ffill().reindex(bar_index)


# ---------------------------------------------------------
# FEATURE ASSEMBLY
# ---------------------------------------------------------
def build_features_v2(price_df):
    symbols = detect_symbols(price_df)
    closes = _close_matrix(price_df, symbols)

    market, ret1, mkt_ret1 = build_market_features(price_df, symbols)
    cross = build_cross_features(closes, ret1, mkt_ret1)
    external = build_external_features(price_df.index)

    hours = price_df.index.hour
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    dow = price_df.index.dayofweek

    frames = []
    for sym in symbols:
        f = build_features_for_symbol(price_df, sym)

        for name, mat in cross.items():
            f[name] = mat[sym]
        for col in market.columns:
            f[col] = market[col]
        for col in external.columns:
            f[col] = external[col]

        f["hour_sin"] = hour_sin
        f["hour_cos"] = hour_cos
        f["dow"] = dow

        # Contrarian interaction: positive when the coin is oversold while
        # the market is fearful (or overbought while greedy). Scaled to ~[-1, 1].
        if "fng" in external.columns:
            f["rsi_x_fng"] = (50 - f["rsi"]) * (50 - external["fng"]) / 2500
        else:
            f["rsi_x_fng"] = np.nan

        frames.append(f)

    features = pd.concat(frames).reset_index()
    features = features.rename(columns={features.columns[0]: "timestamp"})
    features = features.set_index(["timestamp", "symbol"]).sort_index()
    return features


# ---------------------------------------------------------
# LABELS / TARGETS (volatility-scaled)
# ---------------------------------------------------------
def vol_scaled_score(close):
    """Forward return over the horizon, in units of recent volatility."""
    fut_ret = close.shift(-cfg.LABEL_HORIZON) / close - 1
    vol_h = close.pct_change().rolling(cfg.VOL_SCALE_WINDOW).std() * np.sqrt(cfg.LABEL_HORIZON)
    return fut_ret / vol_h


def build_labels_v2(close):
    score = vol_scaled_score(close)

    label = pd.Series(1.0, index=close.index)
    label[score >= cfg.LABEL_UP_SIGMA] = 2
    label[score <= -cfg.LABEL_DOWN_SIGMA] = 0
    label[score.isna()] = np.nan
    return label


def build_target_v2(close):
    """Continuous regression target: vol-scaled forward return, winsorized
    so a single 10-sigma candle doesn't dominate the squared-error loss."""
    return vol_scaled_score(close).clip(-cfg.TARGET_CLIP, cfg.TARGET_CLIP)


def _merge_features_and_column(price_df, build_fn, col_name, features=None):
    symbols = detect_symbols(price_df)
    if features is None:
        features = build_features_v2(price_df)

    frames = []
    for sym in symbols:
        close = price_df[f"{sym}_Close"]
        s = build_fn(close).rename(col_name).to_frame()
        s["symbol"] = sym
        s = s.reset_index()
        s = s.rename(columns={s.columns[0]: "timestamp"})
        frames.append(s.set_index(["timestamp", "symbol"]))

    col = pd.concat(frames).sort_index()

    merged = features.join(col, how="inner")
    return merged.dropna(subset=[col_name])


def merge_features_and_labels_v2(price_df, features=None):
    merged = _merge_features_and_column(price_df, build_labels_v2, "label", features)
    merged["label"] = merged["label"].astype(int)
    return merged


def merge_features_and_target_v2(price_df, features=None):
    return _merge_features_and_column(price_df, build_target_v2, "target", features)
