"""Optional Yahoo Finance cross-check (e.g. 2317.TW, 2330.TW)."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from stock_strat.config import SYMBOL_YAHOO


def fetch_yahoo_ohlcv(
    start_date: str,
    end_date: str,
    *,
    symbol_yahoo: str = SYMBOL_YAHOO,
) -> pd.DataFrame:
    """Download OHLCV + Adj Close for comparison (Yahoo end date is exclusive in some versions)."""
    t = yf.Ticker(symbol_yahoo)
    df = t.history(start=start_date, end=end_date, auto_adjust=False, actions=True)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
    )
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    keep = [c for c in ["open", "high", "low", "close", "volume", "adj_close"] if c in df.columns]
    return df[keep].sort_index()


def yahoo_adjustment_ratio(df_yahoo: pd.DataFrame) -> pd.Series:
    """Adj Close / Close — multiply raw OHLC by this to align with Yahoo adjustment."""
    if df_yahoo.empty or "adj_close" not in df_yahoo.columns:
        return pd.Series(dtype=float)
    return (df_yahoo["adj_close"] / df_yahoo["close"]).astype(float)
