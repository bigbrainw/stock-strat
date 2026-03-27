import pandas as pd

from stock_strat.backtest import run_rsi_backtest


def test_skip_gap_blocks_fill():
    idx = pd.date_range("2020-01-02", periods=4, freq="B")
    # Bar 2 open gaps up 15% vs prev close -> skip buy at open[2]
    ohlcv = pd.DataFrame(
        {
            "open": [100.0, 100.0, 120.0, 120.0],
            "high": [105.0] * 4,
            "low": [95.0] * 4,
            "close": [100.0, 100.0, 100.0, 120.0],
            "volume": [1e6] * 4,
        },
        index=idx,
    )
    entries = pd.Series([False, True, False, False], index=idx)
    exits = pd.Series([False, False, False, False], index=idx)
    res_skip = run_rsi_backtest(
        ohlcv,
        entries=entries,
        exits=exits,
        initial_cash=10_000.0,
        fee_pct_per_trade=0.0,
        skip_gap_abs_pct=0.10,
    )
    res_no = run_rsi_backtest(
        ohlcv,
        entries=entries,
        exits=exits,
        initial_cash=10_000.0,
        fee_pct_per_trade=0.0,
        skip_gap_abs_pct=None,
    )
    # Skip path never opens a position; baseline opens at open[2] (fee 0 → same mark if flat).
    assert len(res_skip.trades) == 0
    assert len(res_no.trades) >= 1
