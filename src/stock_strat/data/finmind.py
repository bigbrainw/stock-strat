"""FinMind API: Taiwan OHLCV and dividends (TaiwanStockPrice, TaiwanStockDividendResult)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from stock_strat.config import DATA_CACHE_DIR, FINMIND_BASE, SYMBOL_TSMC, SYMBOL_TWSE


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    out = r.json()
    if out.get("status") != 200 or out.get("msg") != "success":
        raise RuntimeError(f"FinMind error: {out}")
    return out


def _ohlcv_frame_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    out = pd.DataFrame(
        {
            "open": df["open"].astype(float),
            "high": df["max"].astype(float),
            "low": df["min"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["Trading_Volume"].astype(float),
        }
    )
    out.index.name = "date"
    return out


def fetch_ohlcv_twse(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV for a TWSE / Taiwan listing via FinMind v4 ``TaiwanStockPrice``."""
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    raw = _get_json(FINMIND_BASE, params)
    return _ohlcv_frame_from_rows(raw.get("data") or [])


def fetch_ohlcv_2317(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV for 2317 (Hon Hai)."""
    return fetch_ohlcv_twse(SYMBOL_TWSE, start_date, end_date)


def fetch_ohlcv_2330(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV for 2330 (TSMC / 台積電)."""
    return fetch_ohlcv_twse(SYMBOL_TSMC, start_date, end_date)


def fetch_dividends_twse(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Ex-dividend reference prices (``TaiwanStockDividendResult``)."""
    params = {
        "dataset": "TaiwanStockDividendResult",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    raw = _get_json(FINMIND_BASE, params)
    rows = raw.get("data") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(
        columns={
            "before_price": "before_price",
            "after_price": "after_price",
            "stock_and_cache_dividend": "dividend",
        }
    )
    df["before_price"] = df["before_price"].astype(float)
    df["after_price"] = df["after_price"].astype(float)
    df["dividend"] = df["dividend"].astype(float)
    return df.set_index("date").sort_index()


def fetch_dividends_2317(start_date: str, end_date: str) -> pd.DataFrame:
    return fetch_dividends_twse(SYMBOL_TWSE, start_date, end_date)


def fetch_dividends_2330(start_date: str, end_date: str) -> pd.DataFrame:
    return fetch_dividends_twse(SYMBOL_TSMC, start_date, end_date)


def _cache_path(stock_id: str, start_date: str, end_date: str) -> Path:
    return DATA_CACHE_DIR / f"ohlcv_{stock_id}_{start_date}_{end_date}.parquet"


def load_or_fetch_ohlcv(
    start_date: str,
    end_date: str,
    *,
    stock_id: str = SYMBOL_TWSE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load cached Parquet or download from FinMind ``TaiwanStockPrice`` (e.g. ``2317``, ``2330``)."""
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(stock_id, start_date, end_date)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch_ohlcv_twse(stock_id, start_date, end_date)
    if df.empty:
        raise RuntimeError(f"No OHLCV rows returned from FinMind for {stock_id}")
    df.to_parquet(path)
    meta = {
        "source": "FinMind TaiwanStockPrice",
        "symbol": stock_id,
        "start": start_date,
        "end": end_date,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df
