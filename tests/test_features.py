import pandas as pd

from stock_strat.features import rsi_wilder


def test_rsi_bounds():
    close = pd.Series([100.0, 102.0, 101.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0, 90.0, 89.0, 88.0, 87.0])
    r = rsi_wilder(close, length=14)
    valid = r.dropna()
    assert valid.min() >= 0
    assert valid.max() <= 100
