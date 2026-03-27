"""End-to-end load → clean → features → signals."""

from __future__ import annotations

import pandas as pd

from stock_strat.clean import clean_ohlcv
from stock_strat.config import (
    DEFAULT_END,
    DEFAULT_START,
    REGIME_SMA_DAYS,
    RSI_ENTRY,
    RSI_EXIT,
    RSI_PERIOD,
    SYMBOL_TWSE,
)
from stock_strat.data.finmind import load_or_fetch_ohlcv
from stock_strat.features import compute_features, merge_index_regime
from stock_strat.strategy import generate_signals


def load_clean_ohlcv(
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    *,
    stock_id: str = SYMBOL_TWSE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download/cache raw OHLCV, dividend-adjust. Use for grids to avoid repeated FinMind calls."""
    raw = load_or_fetch_ohlcv(start_date, end_date, stock_id=stock_id, refresh=refresh)
    return clean_ohlcv(raw, start_date, end_date, stock_id=stock_id)


def build_strategy_frame(
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    *,
    stock_id: str = SYMBOL_TWSE,
    refresh: bool = False,
    clean: pd.DataFrame | None = None,
    rsi_period: int = RSI_PERIOD,
    rsi_entry: float = RSI_ENTRY,
    rsi_exit: float = RSI_EXIT,
    regime_above_sma: int | None = None,
    regime_require_index_above_sma: bool = False,
    index_stock_id: str | None = None,
    index_sma_days: int | None = None,
    vol_percentile_max: float | None = None,
    vol_lookback: int = 20,
) -> pd.DataFrame:
    """
    If ``clean`` is provided, skip download (caller already has adjusted OHLCV).

    ``regime_above_sma``: if set, only allow entries when close > SMA(n).
    ``vol_percentile_max``: if set, only allow entries when rolling vol percentile <= this (0-1).
    ``index_stock_id``: optional TWSE code (e.g. ``0050``) for index trend regime columns.
    """
    if clean is None:
        clean = load_clean_ohlcv(start_date, end_date, stock_id=stock_id, refresh=refresh)
    feat = compute_features(
        clean,
        rsi_length=rsi_period,
        sma_trend=regime_above_sma,
        vol_lookback=vol_lookback,
    )
    if index_stock_id:
        idx_days = index_sma_days if index_sma_days is not None else (regime_above_sma or REGIME_SMA_DAYS)
        idx_clean = load_clean_ohlcv(start_date, end_date, stock_id=index_stock_id, refresh=refresh)
        feat = merge_index_regime(feat, idx_clean, sma_days=idx_days)
    return generate_signals(
        feat,
        rsi_entry=rsi_entry,
        rsi_exit=rsi_exit,
        regime_above_sma=regime_above_sma,
        regime_require_index_above_sma=regime_require_index_above_sma,
        vol_percentile_max=vol_percentile_max,
        vol_lookback=vol_lookback,
    )
