"""Performance metrics (pandas/numpy only)."""

from __future__ import annotations

import math

import pandas as pd


def performance_summary(
    equity: pd.Series,
    *,
    returns: pd.Series | None = None,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: float = 252.0,
) -> dict[str, float]:
    """
    Total return, CAGR, volatility, max drawdown, Sharpe, win rate on **daily** returns.
    Sharpe uses sample std of daily returns; annualized as sqrt(252) * (mean - rf/252) / std.
    """
    eq = equity.astype(float).dropna()
    if eq.empty:
        return {}
    rets = returns if returns is not None else eq.pct_change().dropna()
    if rets.empty:
        rets = eq.pct_change().dropna()

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    n = len(eq)
    years = n / periods_per_year if periods_per_year else 1.0
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else float("nan")

    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    max_dd = float(drawdown.min())

    daily = rets.astype(float)
    vol_ann = float(daily.std() * math.sqrt(periods_per_year))
    excess = daily.mean() - risk_free_rate_annual / periods_per_year
    sharpe = (
        float(excess / daily.std() * math.sqrt(periods_per_year))
        if daily.std() > 0
        else float("nan")
    )

    win_rate = float((daily > 0).mean()) if len(daily) else float("nan")

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility_annual": vol_ann,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "win_rate_daily": win_rate,
        "n_bars": float(n),
    }


def trade_stats(trades: list[dict]) -> dict[str, float]:
    """Round-trip oriented trade count from ledger (buy/sell pairs not explicitly paired)."""
    n = len(trades)
    return {"n_orders": float(n)}
