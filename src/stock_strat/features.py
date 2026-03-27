"""Technical indicators (Wilder RSI, SMA, realized vol) — pandas/numpy only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_strat.config import RSI_PERIOD


def rsi_wilder(close: pd.Series, length: int = RSI_PERIOD) -> pd.Series:
    """RSI using Wilder's smoothing (matches common charting defaults)."""
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def compute_features(
    ohlcv: pd.DataFrame,
    *,
    rsi_length: int = RSI_PERIOD,
    sma_trend: int | None = None,
    vol_lookback: int = 20,
) -> pd.DataFrame:
    """
    Append RSI; optional SMA for trend regime; realized vol (annualized) for vol regime.

    ``sma_trend``: if set (e.g. 200), adds ``sma_trend`` and ``above_sma_trend``.
    """
    out = ohlcv.copy()
    close = out["close"].astype(float)
    out["rsi"] = rsi_wilder(close, length=rsi_length)
    lr = np.log(close / close.shift(1))
    rv = lr.rolling(vol_lookback, min_periods=5).std() * np.sqrt(252.0)
    out["realized_vol_ann"] = rv
    if sma_trend is not None and sma_trend > 0:
        out["sma_trend"] = close.rolling(sma_trend, min_periods=min(50, sma_trend)).mean()
        out["above_sma_trend"] = close > out["sma_trend"]
    return out


def merge_index_regime(
    feat: pd.DataFrame,
    index_ohlcv: pd.DataFrame,
    *,
    sma_days: int,
) -> pd.DataFrame:
    """Join index close/SMA regime (e.g. 0050) on ``feat`` index for market gating."""
    iclose = index_ohlcv["close"].astype(float)
    ix_sma = iclose.rolling(sma_days, min_periods=min(50, sma_days)).mean()
    idx_df = pd.DataFrame(
        {
            "index_close": iclose,
            "index_sma": ix_sma,
            "index_above_sma": iclose > ix_sma,
        },
        index=index_ohlcv.index,
    )
    out = feat.join(idx_df, how="left")
    out["index_above_sma"] = out["index_above_sma"].fillna(False)
    return out
