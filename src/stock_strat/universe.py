"""Cross-sectional runs: multiple TWSE codes with liquidity filters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_strat.backtest import run_rsi_backtest
from stock_strat.metrics import performance_summary, trade_stats
from stock_strat.pipeline import build_strategy_frame, load_clean_ohlcv


def load_ticker_list(path: str | Path) -> list[str]:
    """One TWSE code per line; ``#`` starts a comment."""
    p = Path(path)
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split()[0])
    return out


def liquidity_filter(
    ohlcv: pd.DataFrame,
    *,
    min_avg_turnover_twd: float = 0.0,
    min_price: float = 0.0,
) -> bool:
    """Return True if name passes filters (uses FinMind columns: volume * close ~ turnover proxy)."""
    if ohlcv.empty:
        return False
    close = ohlcv["close"].astype(float)
    vol = ohlcv["volume"].astype(float)
    turnover = (close * vol).mean()
    last_px = float(close.iloc[-1])
    return turnover >= min_avg_turnover_twd and last_px >= min_price


def run_universe(
    tickers: list[str],
    start_date: str,
    end_date: str,
    *,
    refresh: bool = False,
    min_avg_turnover_twd: float = 0.0,
    min_price: float = 5.0,
    fee_pct_per_trade: float | None = 0.001,
    initial_cash: float = 100_000.0,
    min_commission_twd: float = 0.0,
    slippage_pct: float = 0.0,
    equity_fraction: float = 1.0,
    regime_above_sma: int | None = None,
    vol_percentile_max: float | None = None,
    index_stock_id: str | None = None,
    regime_require_index_above_sma: bool = False,
) -> pd.DataFrame:
    """Backtest default RSI strategy per ticker; skip names failing liquidity filter."""
    rows: list[dict] = []
    for t in tickers:
        try:
            clean = load_clean_ohlcv(start_date, end_date, stock_id=t, refresh=refresh)
        except Exception:
            continue
        ohlcv = clean[["open", "high", "low", "close", "volume"]]
        if not liquidity_filter(ohlcv, min_avg_turnover_twd=min_avg_turnover_twd, min_price=min_price):
            rows.append({"stock_id": t, "skipped": 1.0, "reason": "liquidity"})
            continue
        df = build_strategy_frame(
            start_date,
            end_date,
            stock_id=t,
            refresh=False,
            clean=clean,
            regime_above_sma=regime_above_sma,
            vol_percentile_max=vol_percentile_max,
            index_stock_id=index_stock_id,
            regime_require_index_above_sma=regime_require_index_above_sma,
        )
        res = run_rsi_backtest(
            ohlcv,
            entries=df["signal_entry_cross"],
            exits=df["signal_exit_cross"],
            initial_cash=initial_cash,
            fee_pct_per_trade=fee_pct_per_trade,
            min_commission_twd=min_commission_twd,
            slippage_pct=slippage_pct,
            equity_fraction=equity_fraction,
        )
        m = performance_summary(res.equity, returns=res.returns)
        m.update(trade_stats(res.trades))
        m["stock_id"] = t
        m["skipped"] = 0.0
        rows.append(m)
    return pd.DataFrame(rows)
