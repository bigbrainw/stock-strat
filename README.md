# stock-strat — Taiwan 2317 (鴻海) daily backtesting

Educational backtest pipeline for **single symbol** TWSE **2317** / Yahoo **2317.TW** ([Yahoo quote](https://tw.stock.yahoo.com/quote/2317.TW)).

## Data assumptions

- **Primary OHLCV:** [FinMind](https://github.com/FinMind/FinMind) `TaiwanStockPrice` (API v4), cached under `data/cache/`.
- **Adjustment:** Cash dividends via FinMind `TaiwanStockDividendResult` (backward adjustment on OHLC). Volume is not scaled for cash dividends.
- **Optional cross-check:** `yfinance` `2317.TW` ratio `Adj Close / Close` aligned by date to validate or blend adjustment.
- **Execution (default):** Signal evaluated at bar **close**; orders fill at **next bar open** (documented in `backtest` module). Costs: configurable % per trade (default **0.1%** round-trip modeled as fee on traded notional unless you change settings).

## Setup

```bash
cd stock-strat
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
stock-strat run --start 2015-01-01 --end 2024-12-31
```

## Tests

```bash
pytest
```

## API limits

FinMind free tier: **600 requests/hour** — cached Parquet minimizes repeat calls.
