"""Technical indicators (Wilder RSI, pure pandas/numpy — no TA lib required)."""

from __future__ import annotations

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


def compute_features(ohlcv: pd.DataFrame, *, rsi_length: int = RSI_PERIOD) -> pd.DataFrame:
    """Append RSI on adjusted close."""
    out = ohlcv.copy()
    out["rsi"] = rsi_wilder(out["close"], length=rsi_length)
    return out
