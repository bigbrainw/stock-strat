"""End-to-end load → clean → features → signals."""

from __future__ import annotations

import pandas as pd

from stock_strat.clean import clean_ohlcv
from stock_strat.config import DEFAULT_END, DEFAULT_START, SYMBOL_TWSE
from stock_strat.data.finmind import load_or_fetch_ohlcv
from stock_strat.features import compute_features
from stock_strat.strategy import generate_signals


def build_strategy_frame(
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    *,
    stock_id: str = SYMBOL_TWSE,
    refresh: bool = False,
) -> pd.DataFrame:
    raw = load_or_fetch_ohlcv(start_date, end_date, stock_id=stock_id, refresh=refresh)
    clean = clean_ohlcv(raw, start_date, end_date, stock_id=stock_id)
    feat = compute_features(clean)
    return generate_signals(feat)
