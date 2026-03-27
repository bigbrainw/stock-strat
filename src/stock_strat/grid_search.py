"""RSI parameter grid search and stability reporting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import pandas as pd

from stock_strat.backtest import run_rsi_backtest
from stock_strat.config import (
    DEFAULT_COMMISSION_PCT,
    DEFAULT_EQUITY_FRACTION,
    DEFAULT_SELL_TAX_PCT,
    DEFAULT_SLIPPAGE_PCT,
    INITIAL_CAPITAL,
)
from stock_strat.features import compute_features, merge_index_regime
from stock_strat.metrics import performance_summary, trade_stats
from stock_strat.strategy import generate_signals


@dataclass(frozen=True)
class RSIGridPoint:
    rsi_period: int
    rsi_entry: float
    rsi_exit: float


def default_grid() -> list[RSIGridPoint]:
    """Default search space (period × entry × exit)."""
    periods = [5, 7, 10, 14, 21]
    entries = [20.0, 25.0, 30.0, 35.0]
    exits = [45.0, 50.0, 55.0, 60.0]
    return [RSIGridPoint(p, e, x) for p, e, x in product(periods, entries, exits) if e < x]


def run_grid_on_clean(
    clean: pd.DataFrame,
    grid: Sequence[RSIGridPoint] | None = None,
    *,
    initial_cash: float = INITIAL_CAPITAL,
    fee_pct_per_trade: float | None = 0.001,
    commission_pct: float = DEFAULT_COMMISSION_PCT,
    sell_tax_pct: float = DEFAULT_SELL_TAX_PCT,
    regime_above_sma: int | None = None,
    regime_require_index_above_sma: bool = False,
    index_clean: pd.DataFrame | None = None,
    index_sma_days: int = 200,
    vol_percentile_max: float | None = None,
    vol_lookback: int = 20,
    min_commission_twd: float = 0.0,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    equity_fraction: float = DEFAULT_EQUITY_FRACTION,
    skip_gap_abs_pct: float | None = None,
) -> pd.DataFrame:
    """
    Backtest each grid point on the same adjusted OHLCV. Uses legacy single fee for speed unless
    caller passes fee_pct_per_trade=None and wires Taiwan fees via a wrapper (see CLI).
    """
    grid = grid or default_grid()
    rows: list[dict] = []
    ohlcv = clean[["open", "high", "low", "close", "volume"]]
    for pt in grid:
        feat = compute_features(
            clean,
            rsi_length=pt.rsi_period,
            sma_trend=regime_above_sma,
            vol_lookback=vol_lookback,
        )
        if index_clean is not None:
            feat = merge_index_regime(feat, index_clean, sma_days=index_sma_days)
        df = generate_signals(
            feat,
            rsi_entry=pt.rsi_entry,
            rsi_exit=pt.rsi_exit,
            regime_above_sma=regime_above_sma,
            regime_require_index_above_sma=regime_require_index_above_sma,
            vol_percentile_max=vol_percentile_max,
            vol_lookback=vol_lookback,
        )
        bt_kw: dict = {
            "initial_cash": initial_cash,
            "min_commission_twd": min_commission_twd,
            "slippage_pct": slippage_pct,
            "equity_fraction": equity_fraction,
            "skip_gap_abs_pct": skip_gap_abs_pct,
        }
        if fee_pct_per_trade is not None:
            res = run_rsi_backtest(
                ohlcv,
                entries=df["signal_entry_cross"],
                exits=df["signal_exit_cross"],
                fee_pct_per_trade=fee_pct_per_trade,
                **bt_kw,
            )
        else:
            res = run_rsi_backtest(
                ohlcv,
                entries=df["signal_entry_cross"],
                exits=df["signal_exit_cross"],
                fee_pct_per_trade=None,
                commission_pct=commission_pct,
                sell_tax_pct=sell_tax_pct,
                **bt_kw,
            )
        m = performance_summary(res.equity, returns=res.returns)
        m.update(trade_stats(res.trades))
        m["rsi_period"] = float(pt.rsi_period)
        m["rsi_entry"] = pt.rsi_entry
        m["rsi_exit"] = pt.rsi_exit
        rows.append(m)
    return pd.DataFrame(rows)


def stability_summary(results: pd.DataFrame, metric: str = "sharpe", top_k: int = 5) -> dict:
    """Top-k by metric and count of combos within 80% of best metric (plateau proxy)."""
    if results.empty or metric not in results.columns:
        return {}
    r = results.dropna(subset=[metric]).sort_values(metric, ascending=False)
    best = float(r[metric].iloc[0]) if len(r) else float("nan")
    thr = best * 0.8 if abs(best) > 1e-9 else best
    plateau = int((r[metric] >= thr).sum()) if len(r) else 0
    return {
        "metric": metric,
        "best": best,
        "plateau_count_80pct": plateau,
        "top_k": r.head(top_k)[["rsi_period", "rsi_entry", "rsi_exit", metric]].to_dict("records"),
    }
