"""2317-only stress: fee sensitivity and subperiod slices."""

from __future__ import annotations

import pandas as pd

from stock_strat.backtest import run_rsi_backtest
from stock_strat.metrics import performance_summary


def fee_sensitivity(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    fees: list[float],
) -> pd.DataFrame:
    rows = []
    for f in fees:
        res = run_rsi_backtest(ohlcv, entries=entries, exits=exits, fee_pct_per_trade=f)
        m = performance_summary(res.equity, returns=res.returns)
        m["fee_pct"] = f
        rows.append(m)
    return pd.DataFrame(rows)


def subperiod_report(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    *,
    periods: dict[str, tuple[str, str]],
    fee_pct: float,
) -> pd.DataFrame:
    rows = []
    for name, (a, b) in periods.items():
        mask = (ohlcv.index >= a) & (ohlcv.index <= b)
        idx = ohlcv.index[mask]
        if len(idx) == 0:
            continue
        before = ohlcv.index[ohlcv.index < a]
        start = before[-1] if len(before) else idx[0]
        sub = ohlcv.loc[start:b]
        ent = entries.reindex(sub.index).fillna(False)
        ex = exits.reindex(sub.index).fillna(False)
        res = run_rsi_backtest(sub, entries=ent, exits=ex, fee_pct_per_trade=fee_pct)
        eq = res.equity.loc[(res.equity.index >= a) & (res.equity.index <= b)]
        m = performance_summary(eq, returns=eq.pct_change().fillna(0.0))
        m["period"] = name
        rows.append(m)
    return pd.DataFrame(rows)
