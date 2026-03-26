"""
RSI strategy for 2317: explicit signal columns.

Signal evaluated on bar close. Execution is applied on next bar open in `backtest` (shift).

Rules (educational):
- Long entry when RSI crosses into oversold: RSI < entry and previous RSI >= entry.
- Long exit when RSI crosses into exit zone: RSI > exit and previous RSI <= exit.

Alternatively, first bar where RSI < entry after being above counts as entry — we use cross-into-zone for fewer duplicate orders.
"""

from __future__ import annotations

import pandas as pd

from stock_strat.config import RSI_ENTRY, RSI_EXIT


def generate_signals(
    df: pd.DataFrame,
    *,
    rsi_entry: float = RSI_ENTRY,
    rsi_exit: float = RSI_EXIT,
) -> pd.DataFrame:
    """Add signal_* columns for debugging and backtests."""
    out = df.copy()
    rsi = out["rsi"].astype(float)
    below = rsi < rsi_entry
    above_exit = rsi > rsi_exit
    prev_below = below.shift(1).fillna(False)
    prev_above_exit = above_exit.shift(1).fillna(False)

    # Enter when we cross into oversold band
    out["signal_entry_cross"] = below & ~prev_below

    # Exit when we cross above exit threshold
    out["signal_exit_cross"] = above_exit & ~prev_above_exit

    # Raw zones (for logging)
    out["signal_in_oversold"] = below
    out["signal_above_exit_zone"] = above_exit

    return out
