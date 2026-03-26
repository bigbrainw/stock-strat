"""Clean TWSE daily bars: dividend backward adjustment, calendar QA."""

from __future__ import annotations

import pandas as pd

from stock_strat.data.finmind import fetch_dividends_2317


def _backward_dividend_factors(
    index: pd.DatetimeIndex,
    div_df: pd.DataFrame,
) -> pd.Series:
    """Per-date multipliers so older prices reflect cumulative cash dividends (backward adj)."""
    mult = pd.Series(1.0, index=index, dtype=float)
    if div_df.empty:
        return mult
    # Process most recent ex-date first: scale all strictly earlier dates
    for ex_date, row in div_df.sort_index(ascending=False).iterrows():
        before = float(row["before_price"])
        after = float(row["after_price"])
        if after <= 0 or before <= 0:
            continue
        factor = before / after
        mask = index.normalize() < pd.Timestamp(ex_date).normalize()
        mult.loc[mask] *= factor
    return mult


def clean_ohlcv(
    ohlcv: pd.DataFrame,
    start_date: str,
    end_date: str,
    *,
    dividends: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    - Sort index, drop duplicate dates (keep last).
    - Forward-fill missing OHLC only where safe (optional minimal ffill on close for isolated gaps).
    - Apply backward cash-dividend adjustment to open/high/low/close from FinMind dividend table.

    Volume is left unscaled (cash dividends). Use split-aware logic later if you add split data.
    """
    df = ohlcv.copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df["close"].notna()]

    div = dividends if dividends is not None else fetch_dividends_2317(start_date, end_date)
    mult = _backward_dividend_factors(df.index, div)
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = df[col].astype(float) * mult

    return df


def align_trading_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Drop consecutive duplicate closes (data glitches) — optional noop if none."""
    out = df.copy()
    dup = out["close"].eq(out["close"].shift()) & out["volume"].eq(0)
    out = out.loc[~dup.fillna(False)]
    return out
