import pandas as pd

from stock_strat.portfolio import equal_weight_combine_equity, simulate_top_n_entry_rotation


def test_equal_weight_combine_equity():
    idx = pd.date_range("2020-01-02", periods=4, freq="B")
    a = pd.Series([100_000.0, 101_000.0, 102_000.0, 103_000.0], index=idx)
    b = pd.Series([100_000.0, 99_000.0, 98_000.0, 97_000.0], index=idx)
    comb = equal_weight_combine_equity({"a": a, "b": b}, initial_cash=100_000.0)
    assert len(comb) == 4
    assert abs(float(comb.iloc[0]) - 100_000.0) < 1e-6


def test_top_n_rotation_smoke():
    idx = pd.date_range("2020-01-02", periods=6, freq="B")
    # Two names, identical prices — focus on plumbing
    def panel(entries: list[bool]) -> pd.DataFrame:
        n = len(idx)
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": [100.0] * n,
                "volume": [1e6] * n,
                "rsi": [25.0] * n,
                "signal_entry_cross": entries,
                "signal_exit_cross": [False] * n,
            },
            index=idx,
        )
        return ohlcv

    ent_a = [False, True, False, False, False, False]
    ent_b = [False, True, False, False, False, False]
    panels = {"1111": panel(ent_a), "2222": panel(ent_b)}
    res = simulate_top_n_entry_rotation(
        panels,
        top_n=2,
        initial_cash=100_000.0,
        fee_pct_per_trade=0.0,
        min_commission_twd=0.0,
        slippage_pct=0.0,
    )
    assert len(res.equity) == len(idx)
    assert res.equity.iloc[-1] > 0
