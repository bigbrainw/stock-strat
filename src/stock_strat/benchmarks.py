"""Simple baselines on the same OHLCV path."""

from __future__ import annotations

import pandas as pd

from stock_strat.backtest import run_rsi_backtest
from stock_strat.metrics import performance_summary


def buy_and_hold_equity(ohlcv: pd.DataFrame, initial_cash: float) -> pd.Series:
    """Invest all cash at first open, hold to last close."""
    o0 = float(ohlcv["open"].astype(float).iloc[0])
    close = ohlcv["close"].astype(float)
    sh = initial_cash / o0 if o0 > 0 else 0.0
    return sh * close


def benchmark_buy_and_hold(ohlcv: pd.DataFrame, *, initial_cash: float = 100_000.0) -> dict[str, float]:
    eq = buy_and_hold_equity(ohlcv, initial_cash)
    return performance_summary(eq, returns=eq.pct_change().fillna(0.0))


def benchmark_ma_trend(
    ohlcv: pd.DataFrame,
    *,
    window: int = 200,
    initial_cash: float = 100_000.0,
    fee_pct_per_trade: float | None = 0.001,
) -> dict[str, float]:
    """Long when close > SMA(window), flat otherwise; rebalance daily at next open (simplified)."""
    close = ohlcv["close"].astype(float)
    sma = close.rolling(window, min_periods=20).mean()
    long = close > sma
    entries = long & ~long.shift(1).fillna(False)
    exits = ~long & long.shift(1).fillna(False)
    res = run_rsi_backtest(
        ohlcv,
        entries=entries,
        exits=exits,
        initial_cash=initial_cash,
        fee_pct_per_trade=fee_pct_per_trade,
    )
    return performance_summary(res.equity, returns=res.returns)


def fantasy_same_close_fills(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    *,
    initial_cash: float = 100_000.0,
) -> dict[str, float]:
    """Ablation: fill at same bar close (unrealistic). Shift entries/exits by 0 for same-bar execution."""
    # Same-bar: use close as fill price by aligning signal t with trade at t
    idx = ohlcv.index
    close = ohlcv["close"].astype(float)
    c = float(initial_cash)
    sh = 0.0
    eq = []
    for i in range(len(idx)):
        if i > 0:
            ent = bool(entries.iloc[i - 1])
            ex = bool(exits.iloc[i - 1])
            px = float(close.iloc[i])
            if ex and sh > 0:
                c = sh * px
                sh = 0.0
            elif ent and sh == 0 and c > 0:
                sh = c / px
                c = 0.0
        eq.append(c + sh * float(close.iloc[i]))
    equity = pd.Series(eq, index=idx)
    return performance_summary(equity, returns=equity.pct_change().fillna(0.0))
