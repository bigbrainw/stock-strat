import pandas as pd

from stock_strat.backtest import portfolio_daily_table, run_rsi_backtest


def test_buy_sell_roundtrip():
    idx = pd.date_range("2020-01-02", periods=5, freq="B")
    ohlcv = pd.DataFrame(
        {
            "open": [100.0, 100.0, 110.0, 110.0, 120.0],
            "high": [105.0] * 5,
            "low": [95.0] * 5,
            "close": [100.0, 100.0, 110.0, 110.0, 120.0],
            "volume": [1e6] * 5,
        },
        index=idx,
    )
    entries = pd.Series([False, True, False, False, False], index=idx)
    exits = pd.Series([False, False, False, True, False], index=idx)
    res = run_rsi_backtest(ohlcv, entries=entries, exits=exits, initial_cash=10_000.0, fee_pct_per_trade=0.0)
    # Buy at open idx[2] after signal idx[1]
    assert len(res.trades) >= 2
    assert res.equity.iloc[-1] > 0


def test_portfolio_daily_table_columns():
    idx = pd.date_range("2020-01-02", periods=3, freq="B")
    ohlcv = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0],
            "high": [11.0] * 3,
            "low": [9.0] * 3,
            "close": [10.0, 10.0, 10.0],
            "volume": [1e6] * 3,
        },
        index=idx,
    )
    entries = pd.Series([False, True, False], index=idx)
    exits = pd.Series([False, False, False], index=idx)
    res = run_rsi_backtest(ohlcv, entries=entries, exits=exits, initial_cash=1000.0, fee_pct_per_trade=0.0)
    daily = portfolio_daily_table(ohlcv, res)
    assert list(daily.columns) == ["cash", "shares", "close", "position_value", "equity"]
    assert daily["equity"].iloc[-1] == res.equity.iloc[-1]
