"""
RSI mean-reversion with optional regime gates.

Signal evaluated on bar close. Execution is applied on next bar open in `backtest` (shift).

Rules:
- Long entry when RSI crosses into oversold (cross below entry threshold).
- Long exit when RSI crosses above exit threshold.
- Optional: only allow entries when `above_sma_trend` and/or low vol (see params).
"""

from __future__ import annotations

import pandas as pd

from stock_strat.config import RSI_ENTRY, RSI_EXIT


def generate_signals(
    df: pd.DataFrame,
    *,
    rsi_entry: float = RSI_ENTRY,
    rsi_exit: float = RSI_EXIT,
    regime_above_sma: int | None = None,
    regime_require_index_above_sma: bool = False,
    vol_percentile_max: float | None = None,
    vol_lookback: int = 20,
    vol_rank_window: int = 252,
) -> pd.DataFrame:
    """Add signal_* columns. Regime gates apply only to entries, not exits."""
    out = df.copy()
    rsi = out["rsi"].astype(float)
    below = rsi < rsi_entry
    above_exit = rsi > rsi_exit
    prev_below = below.shift(1).fillna(False)
    prev_above_exit = above_exit.shift(1).fillna(False)

    entry_raw = below & ~prev_below
    exit_raw = above_exit & ~prev_above_exit

    allow = pd.Series(True, index=out.index)
    if regime_above_sma is not None and "above_sma_trend" in out.columns:
        allow = allow & out["above_sma_trend"].fillna(False)
    if vol_percentile_max is not None and "realized_vol_ann" in out.columns:
        rv = out["realized_vol_ann"].astype(float)
        thr = rv.rolling(vol_rank_window, min_periods=20).quantile(vol_percentile_max)
        allow = allow & (rv <= thr)
    if regime_require_index_above_sma and "index_above_sma" in out.columns:
        allow = allow & out["index_above_sma"].fillna(False)

    out["signal_entry_cross"] = entry_raw & allow
    out["signal_exit_cross"] = exit_raw

    out["signal_in_oversold"] = below
    out["signal_above_exit_zone"] = above_exit

    return out
