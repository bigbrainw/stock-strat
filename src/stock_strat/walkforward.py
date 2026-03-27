"""Walk-forward date splits and rolling train optimize / test evaluate."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_strat.backtest import BacktestResult, run_rsi_backtest
from stock_strat.config import (
    DEFAULT_EQUITY_FRACTION,
    DEFAULT_SLIPPAGE_PCT,
    SYMBOL_TWSE,
)
from stock_strat.grid_search import RSIGridPoint, default_grid, run_grid_on_clean
from stock_strat.metrics import performance_summary
from stock_strat.pipeline import build_strategy_frame, load_clean_ohlcv


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


def walk_forward_optimize(
    clean: pd.DataFrame,
    *,
    stock_id: str = SYMBOL_TWSE,
    windows: list[WalkForwardWindow] | None = None,
    grid: list[RSIGridPoint] | None = None,
    optimize_metric: str = "sharpe",
    initial_cash: float = 100_000.0,
    fee_pct_per_trade: float | None = 0.001,
    regime_above_sma: int | None = None,
    regime_require_index_above_sma: bool = False,
    index_stock_id: str | None = None,
    index_sma_days: int | None = None,
    vol_percentile_max: float | None = None,
    min_commission_twd: float = 0.0,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    equity_fraction: float = DEFAULT_EQUITY_FRACTION,
    skip_gap_abs_pct: float | None = None,
) -> pd.DataFrame:
    """
    For each window: grid-search on **train** slice (maximize ``optimize_metric``), then backtest
    **test** slice with frozen params. Requires full ``clean`` OHLCV indexed by date.
    """
    windows = windows or DEFAULT_WINDOWS
    grid = grid or default_grid()
    rows: list[dict] = []
    idx_start = str(clean.index.min().date())
    idx_end = str(clean.index.max().date())
    index_full = (
        load_clean_ohlcv(idx_start, idx_end, stock_id=index_stock_id, refresh=False)
        if index_stock_id
        else None
    )
    ix_days = index_sma_days if index_sma_days is not None else (regime_above_sma or 200)
    for w in windows:
        train = clean.loc[(clean.index >= w.train_start) & (clean.index <= w.train_end)]
        if len(train) < 30:
            continue
        index_train = None
        if index_full is not None:
            index_train = index_full.loc[
                (index_full.index >= w.train_start) & (index_full.index <= w.train_end)
            ]
        grid_df = run_grid_on_clean(
            train,
            grid=grid,
            initial_cash=initial_cash,
            fee_pct_per_trade=fee_pct_per_trade,
            regime_above_sma=regime_above_sma,
            regime_require_index_above_sma=regime_require_index_above_sma,
            index_clean=index_train,
            index_sma_days=ix_days,
            vol_percentile_max=vol_percentile_max,
            min_commission_twd=min_commission_twd,
            slippage_pct=slippage_pct,
            equity_fraction=equity_fraction,
            skip_gap_abs_pct=skip_gap_abs_pct,
        )
        if grid_df.empty or optimize_metric not in grid_df.columns:
            continue
        best = grid_df.loc[grid_df[optimize_metric].idxmax()]
        pt = RSIGridPoint(
            int(best["rsi_period"]),
            float(best["rsi_entry"]),
            float(best["rsi_exit"]),
        )
        test_df = build_strategy_frame(
            start_date=w.test_start,
            end_date=w.test_end,
            stock_id=stock_id,
            clean=clean.loc[w.train_start : w.test_end],
            rsi_period=pt.rsi_period,
            rsi_entry=pt.rsi_entry,
            rsi_exit=pt.rsi_exit,
            regime_above_sma=regime_above_sma,
            regime_require_index_above_sma=regime_require_index_above_sma,
            index_stock_id=index_stock_id,
            index_sma_days=index_sma_days,
            vol_percentile_max=vol_percentile_max,
        )
        ohlcv = test_df[["open", "high", "low", "close", "volume"]]
        mask = (ohlcv.index >= w.test_start) & (ohlcv.index <= w.test_end)
        test_idx = ohlcv.index[mask]
        if len(test_idx) == 0:
            continue
        before = ohlcv.index[ohlcv.index < w.test_start]
        start = before[-1] if len(before) else test_idx[0]
        slice_ohlcv = ohlcv.loc[start : w.test_end]
        ent = test_df["signal_entry_cross"].reindex(slice_ohlcv.index).fillna(False)
        ex = test_df["signal_exit_cross"].reindex(slice_ohlcv.index).fillna(False)
        res = run_rsi_backtest(
            slice_ohlcv,
            entries=ent,
            exits=ex,
            initial_cash=initial_cash,
            fee_pct_per_trade=fee_pct_per_trade,
            min_commission_twd=min_commission_twd,
            slippage_pct=slippage_pct,
            equity_fraction=equity_fraction,
            skip_gap_abs_pct=skip_gap_abs_pct,
        )
        eq = res.equity.loc[(res.equity.index >= w.test_start) & (res.equity.index <= w.test_end)]
        oos = performance_summary(eq, returns=eq.pct_change().fillna(0.0))
        oos["window"] = w.name
        oos["rsi_period"] = float(pt.rsi_period)
        oos["rsi_entry"] = pt.rsi_entry
        oos["rsi_exit"] = pt.rsi_exit
        oos[f"train_best_{optimize_metric}"] = float(best[optimize_metric])
        rows.append(oos)
    return pd.DataFrame(rows)
