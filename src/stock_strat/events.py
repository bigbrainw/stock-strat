"""Event subsample analysis (ex-dividend windows; no ML)."""

from __future__ import annotations

import pandas as pd

from stock_strat.backtest import run_rsi_backtest
from stock_strat.data.finmind import fetch_dividends_twse
from stock_strat.metrics import performance_summary
from stock_strat.pipeline import build_strategy_frame, load_clean_ohlcv


def _ex_div_mask(index: pd.DatetimeIndex, div_dates: pd.DatetimeIndex, window_days: int) -> pd.Series:
    """True on bars within ±window_days of any ex-dividend date."""
    mask = pd.Series(False, index=index)
    for d in div_dates:
        lo = pd.Timestamp(d) - pd.Timedelta(days=window_days)
        hi = pd.Timestamp(d) + pd.Timedelta(days=window_days)
        mask |= (index >= lo) & (index <= hi)
    return mask


def event_metrics_split(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    *,
    event_mask: pd.Series,
    fee_pct_per_trade: float | None = 0.001,
    initial_cash: float = 100_000.0,
) -> dict[str, dict]:
    """Performance on event vs non-event days (equity sliced by date mask)."""
    res = run_rsi_backtest(
        ohlcv,
        entries=entries,
        exits=exits,
        initial_cash=initial_cash,
        fee_pct_per_trade=fee_pct_per_trade,
    )
    eq = res.equity
    em = event_mask.reindex(eq.index).fillna(False)
    out: dict[str, dict] = {}
    for name, m in [("event", em), ("non_event", ~em)]:
        sub = eq.loc[m]
        if len(sub) < 2:
            out[name] = {}
            continue
        out[name] = performance_summary(sub, returns=sub.pct_change().fillna(0.0))
    return out


def ex_dividend_experiment(
    stock_id: str,
    start_date: str,
    end_date: str,
    *,
    window_days: int = 3,
    refresh: bool = False,
    fee_pct_per_trade: float | None = 0.001,
) -> dict[str, dict]:
    """Split metrics around ex-dividend dates from FinMind."""
    clean = load_clean_ohlcv(start_date, end_date, stock_id=stock_id, refresh=refresh)
    df = build_strategy_frame(start_date, end_date, stock_id=stock_id, refresh=False, clean=clean)
    ohlcv = df[["open", "high", "low", "close", "volume"]]
    div = fetch_dividends_twse(stock_id, start_date, end_date)
    div_dates = div.index.unique()
    mask = _ex_div_mask(ohlcv.index, div_dates, window_days)
    return event_metrics_split(
        ohlcv,
        df["signal_entry_cross"],
        df["signal_exit_cross"],
        event_mask=mask,
        fee_pct_per_trade=fee_pct_per_trade,
    )
