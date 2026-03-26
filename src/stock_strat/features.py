"""Technical indicators (pandas-ta)."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from stock_strat.config import RSI_PERIOD


def compute_features(ohlcv: pd.DataFrame, *, rsi_length: int = RSI_PERIOD) -> pd.DataFrame:
    """Append RSI on adjusted close."""
    out = ohlcv.copy()
    close = out["close"].astype(float)
    out["rsi"] = ta.rsi(close, length=rsi_length)
    return out
