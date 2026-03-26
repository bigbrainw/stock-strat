"""
Portfolio simulation with vectorbt.

Execution model (documented):
- Indicators and signals use the bar **close** of day t.
- Orders **fill at the next bar open** (t+1): signals are shifted forward by one row.
- **Fees:** `fee_pct_per_trade` applies to each buy and sell notional (default 0.1% per the plan).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

from stock_strat.config import FEE_PCT_PER_TRADE, INITIAL_CAPITAL


@dataclass(frozen=True)
class BacktestResult:
    portfolio: vbt.Portfolio
    entries_exec: pd.Series
    exits_exec: pd.Series


def run_rsi_backtest(
    ohlcv: pd.DataFrame,
    *,
    entries: pd.Series,
    exits: pd.Series,
    initial_cash: float = INITIAL_CAPITAL,
    fee_pct_per_trade: float = FEE_PCT_PER_TRADE,
) -> BacktestResult:
    """
    Long-only RSI strategy. Entries/exits are boolean indexed like ohlcv (close-based signals).
    Shifted by 1 to execute at next open.
    """
    open_ = ohlcv["open"].astype(float)
    close = ohlcv["close"].astype(float)
    # Signal known after close on t -> trade at open on t+1
    entries_exec = entries.shift(1).fillna(False).astype(bool)
    exits_exec = exits.shift(1).fillna(False).astype(bool)

    # vectorbt expects aligned arrays; use open as order price
    pf = vbt.Portfolio.from_signals(
        close=close,
        open=open_,
        entries=entries_exec,
        exits=exits_exec,
        direction="longonly",
        fees=fee_pct_per_trade,
        init_cash=initial_cash,
        freq="1D",
        accumulate=False,
    )
    return BacktestResult(portfolio=pf, entries_exec=entries_exec, exits_exec=exits_exec)


def equity_curve(pf: vbt.Portfolio) -> pd.Series:
    v = pf.value()
    if isinstance(v, pd.DataFrame):
        return v.squeeze()
    return pd.Series(v, index=pf.wrapper.index)
