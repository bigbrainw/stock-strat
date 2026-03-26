import pandas as pd
import pytest

from stock_strat.clean import clean_ohlcv


def test_clean_adjusts_with_dividends():
    idx = pd.date_range("2020-01-02", periods=3, freq="B")
    raw = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1e6, 1e6, 1e6],
        },
        index=idx,
    )
    div = pd.DataFrame(
        {
            "before_price": [110.0],
            "after_price": [100.0],
            "dividend": [10.0],
        },
        index=pd.to_datetime(["2020-01-03"]),
    )
    out = clean_ohlcv(raw, "2020-01-01", "2020-01-31", dividends=div)
    # Before ex-date 2020-01-03, prices scaled by 110/100
    assert out.loc[idx[0], "close"] == pytest.approx(110.0)
    assert out.loc[idx[2], "close"] == pytest.approx(100.0)
