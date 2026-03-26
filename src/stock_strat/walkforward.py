"""Walk-forward date splits (in-sample / out-of-sample)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_strat.backtest import BacktestResult, run_rsi_backtest
from stock_strat.metrics import performance_summary


@dataclass
class WalkForwardWindow:
    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


DEFAULT_WINDOWS: list[WalkForwardWindow] = [
    WalkForwardWindow(
        name="wf1",
        train_start="2015-01-01",
        train_end="2019-12-31",
        test_start="2020-01-01",
        test_end="2024-12-31",
    ),
]


def run_walk_forward(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    *,
    windows: list[WalkForwardWindow] | None = None,
    fee_pct: float = 0.001,
) -> pd.DataFrame:
    """
    Run the same signals restricted to train/test periods and report OOS metrics.
    (Signals are precomputed on full series; we slice equity for reporting.)
    """
    windows = windows or DEFAULT_WINDOWS
    rows = []
    for w in windows:
        mask = (ohlcv.index >= w.test_start) & (ohlcv.index <= w.test_end)
        test_idx = ohlcv.index[mask]
        if len(test_idx) == 0:
            continue
        # Include prior trading day so next-bar-open fills have the prior signal
        before = ohlcv.index[ohlcv.index < w.test_start]
        start = before[-1] if len(before) else test_idx[0]
        slice_ohlcv = ohlcv.loc[start : w.test_end]
        ent = entries.reindex(slice_ohlcv.index).fillna(False)
        ex = exits.reindex(slice_ohlcv.index).fillna(False)
        res: BacktestResult = run_rsi_backtest(
            slice_ohlcv,
            entries=ent,
            exits=ex,
            fee_pct_per_trade=fee_pct,
        )
        eq = res.equity.loc[(res.equity.index >= w.test_start) & (res.equity.index <= w.test_end)]
        rets = eq.pct_change().fillna(0.0)
        m = performance_summary(eq, returns=rets)
        m["window"] = w.name
        m["split"] = "test"
        rows.append(m)
    return pd.DataFrame(rows)
