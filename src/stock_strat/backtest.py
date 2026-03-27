"""
Long-only portfolio simulation (pure pandas).

Execution: signal at close t, fill at next open t+1. Optional slippage, min commission, partial deploy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_strat.config import (
    DEFAULT_COMMISSION_PCT,
    DEFAULT_EQUITY_FRACTION,
    DEFAULT_SELL_TAX_PCT,
    DEFAULT_SLIPPAGE_PCT,
    INITIAL_CAPITAL,
)


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: list[dict]
    cash: pd.Series
    shares: pd.Series


def _apply_min_comm(comm: float, min_commission_twd: float) -> float:
    if min_commission_twd <= 0:
        return comm
    return max(comm, min_commission_twd)


def run_rsi_backtest(
    ohlcv: pd.DataFrame,
    *,
    entries: pd.Series,
    exits: pd.Series,
    initial_cash: float = INITIAL_CAPITAL,
    fee_pct_per_trade: float | None = None,
    commission_pct: float = DEFAULT_COMMISSION_PCT,
    sell_tax_pct: float = DEFAULT_SELL_TAX_PCT,
    min_commission_twd: float = 0.0,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    equity_fraction: float = DEFAULT_EQUITY_FRACTION,
    skip_gap_abs_pct: float | None = None,
    skip_intraday_range_pct: float | None = None,
) -> BacktestResult:
    """
    Long-only. ``equity_fraction`` in (0,1] scales cash deployed on entry (rest stays cash).
    ``slippage_pct``: buy at open * (1+slippage), sell at open * (1-slippage).
    ``skip_gap_abs_pct``: if set, skip buy/sell at open *t* when
    ``abs(open[t]/close[t-1]-1)`` exceeds this (analysis hook for gap risk).
    ``skip_intraday_range_pct``: if set, skip fills when ``(high[t]-low[t])/open[t]`` exceeds this
    (rough proxy for limit / extreme session moves).
    """
    idx = ohlcv.index
    open_ = ohlcv["open"].astype(float).reindex(idx)
    high = ohlcv["high"].astype(float).reindex(idx)
    low = ohlcv["low"].astype(float).reindex(idx)
    close = ohlcv["close"].astype(float).reindex(idx)
    ent = entries.reindex(idx).fillna(False).astype(bool)
    ex = exits.reindex(idx).fillna(False).astype(bool)

    n = len(idx)
    cash_arr = np.zeros(n, dtype=float)
    sh_arr = np.zeros(n, dtype=float)
    trades: list[dict] = []

    c = float(initial_cash)
    sh = 0.0
    eqf = float(np.clip(equity_fraction, 1e-9, 1.0))
    slip = float(slippage_pct)

    cash_arr[0] = c
    sh_arr[0] = sh

    legacy = fee_pct_per_trade is not None
    f = float(fee_pct_per_trade) if legacy else 0.0

    gap_skip = skip_gap_abs_pct
    range_skip = skip_intraday_range_pct

    def _skip_bar(t_bar: int) -> bool:
        if t_bar < 1:
            return False
        prev_c = float(close.iloc[t_bar - 1])
        o_raw = float(open_.iloc[t_bar])
        if gap_skip is not None and prev_c > 0:
            if abs(o_raw / prev_c - 1.0) > gap_skip:
                return True
        if range_skip is not None and o_raw > 0:
            rng = (float(high.iloc[t_bar]) - float(low.iloc[t_bar])) / o_raw
            if rng > range_skip:
                return True
        return False

    for t in range(1, n):
        o_raw = float(open_.iloc[t])
        o_buy = o_raw * (1.0 + slip)
        o_sell = o_raw * (1.0 - slip)
        skip = _skip_bar(t)
        if ex.iloc[t - 1] and sh > 0 and not skip:
            gross = sh * o_sell
            if legacy:
                comm = f * gross
                comm = _apply_min_comm(comm, min_commission_twd)
                fee_total = comm
                c = gross - fee_total
                trades.append(
                    {
                        "bar": idx[t],
                        "side": "sell",
                        "price": o_sell,
                        "shares": sh,
                        "fee": fee_total,
                        "commission": comm,
                        "sell_tax": 0.0,
                    }
                )
            else:
                comm = commission_pct * gross
                comm = _apply_min_comm(comm, min_commission_twd)
                tax = sell_tax_pct * gross
                fee_total = comm + tax
                c = gross - fee_total
                trades.append(
                    {
                        "bar": idx[t],
                        "side": "sell",
                        "price": o_sell,
                        "shares": sh,
                        "fee": fee_total,
                        "commission": comm,
                        "sell_tax": tax,
                    }
                )
            sh = 0.0
        elif ent.iloc[t - 1] and sh == 0 and c > 0 and not skip:
            c_alloc = c * eqf
            c_remain = c - c_alloc
            comm_raw = f * c_alloc if legacy else commission_pct * c_alloc
            comm = _apply_min_comm(comm_raw, min_commission_twd)
            fee_total = comm
            invest = c_alloc - fee_total
            sh = invest / o_buy if o_buy > 0 else 0.0
            trades.append(
                {
                    "bar": idx[t],
                    "side": "buy",
                    "price": o_buy,
                    "shares": sh,
                    "fee": fee_total,
                    "commission": comm,
                    "sell_tax": 0.0,
                }
            )
            c = c_remain
        cash_arr[t] = c
        sh_arr[t] = sh

    cash_s = pd.Series(cash_arr, index=idx)
    sh_s = pd.Series(sh_arr, index=idx)
    equity = cash_s + sh_s * close
    rets = equity.pct_change().fillna(0.0)
    return BacktestResult(
        equity=equity,
        returns=rets,
        trades=trades,
        cash=cash_s,
        shares=sh_s,
    )


def portfolio_daily_table(ohlcv: pd.DataFrame, res: BacktestResult) -> pd.DataFrame:
    idx = ohlcv.index
    close = ohlcv["close"].astype(float).reindex(idx)
    pos = res.shares * close
    return pd.DataFrame(
        {
            "cash": res.cash.reindex(idx),
            "shares": res.shares.reindex(idx),
            "close": close,
            "position_value": pos,
            "equity": res.equity.reindex(idx),
        }
    )
