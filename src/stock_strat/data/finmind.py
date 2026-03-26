"""FinMind API: Taiwan OHLCV and dividends for 2317 only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from stock_strat.config import DATA_CACHE_DIR, FINMIND_BASE, SYMBOL_TWSE


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    out = r.json()
    if out.get("status") != 200 or out.get("msg") != "success":
        raise RuntimeError(f"FinMind error: {out}")
    return out


def fetch_ohlcv_2317(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV for 2317 from FinMind v4."""
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": SYMBOL_TWSE,
        "start_date": start_date,
        "end_date": end_date,
    }
    raw = _get_json(FINMIND_BASE, params)
    rows = raw.get("data") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    rename = {
        "open": "open",
        "max": "high",
        "min": "low",
        "close": "close",
        "Trading_Volume": "volume",
    }
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


def fetch_dividends_2317(start_date: str, end_date: str) -> pd.DataFrame:
    """Ex-dividend reference prices (TaiwanStockDividendResult)."""
    params = {
        "dataset": "TaiwanStockDividendResult",
        "data_id": SYMBOL_TWSE,
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


def _cache_path(start_date: str, end_date: str) -> Path:
    return DATA_CACHE_DIR / f"ohlcv_{SYMBOL_TWSE}_{start_date}_{end_date}.parquet"


def load_or_fetch_ohlcv(
    start_date: str,
    end_date: str,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load cached Parquet or download from FinMind."""
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(start_date, end_date)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch_ohlcv_2317(start_date, end_date)
    if df.empty:
        raise RuntimeError("No OHLCV rows returned from FinMind")
    df.to_parquet(path)
    meta = {
        "source": "FinMind TaiwanStockPrice",
        "symbol": SYMBOL_TWSE,
        "start": start_date,
        "end": end_date,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df
