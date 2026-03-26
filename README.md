# stock-strat — Taiwan daily backtesting (default 2317 鴻海)

Educational backtest pipeline focused on TWSE listings. Defaults: **2317** / Yahoo [**2317.TW**](https://tw.stock.yahoo.com/quote/2317.TW). **TSMC** is supported as **2330** / [**2330.TW**](https://tw.stock.yahoo.com/quote/2330.TW) via the same FinMind `TaiwanStockPrice` flow.

## Data assumptions

- **Primary OHLCV:** [FinMind](https://github.com/FinMind/FinMind) `TaiwanStockPrice` (API v4), one Parquet cache per `(stock_id, start, end)` under `data/cache/` (e.g. `ohlcv_2330_2015-01-01_2024-12-31.parquet`).
- **Adjustment:** Cash dividends via FinMind `TaiwanStockDividendResult` (backward adjustment on OHLC). Volume is not scaled for cash dividends.
- **Optional cross-check:** `stock_strat.data.yahoo.yahoo_adjustment_ratio` with `fetch_yahoo_ohlcv` — compare Yahoo `Adj Close / Close` to FinMind-adjusted series.
- **Execution:** Signal evaluated at bar **close**; orders fill at **next bar open** (`stock_strat.backtest`).
- **Taiwan costs (defaults on `run`):** Brokerage **commission** on both buy and sell (default ≈ **2 折** on the statutory **0.1425%** cap per side, i.e. ~**0.0285%** of amount, modeled for 永豐大戶投-style electronic discounts — **confirm** on [永豐手續費](https://www.sinotrade.com.tw/newweb/Fee_Rate)). **證交稅** on **sell** only (default **0.3%** of sell gross for normal 現股; 當沖 may differ). Use `--legacy-fee` to use a single old-style rate instead. Minimum **1 TWD** commission per order in reality is **not** modeled.
- **Engine:** Pure **pandas/numpy** simulation (no vectorbt — avoids **numba** on Python 3.14+). RSI is Wilder-smoothed in `features.py`.

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
stock-strat run --capital 500000 --portfolio-csv /tmp/portfolio_2330.csv --stock 2330 --start 2015-01-01 --end 2024-12-31
stock-strat run --stock 2330 --start 2015-01-01 --end 2024-12-31
stock-strat walkforward --start 2015-01-01 --end 2024-12-31
stock-strat stress --start 2015-01-01 --end 2024-12-31
```

`--stock` is the FinMind `data_id` (TWSE code), e.g. `2317` (Hon Hai) or `2330` (TSMC).

**Capital path:** `run` accepts `--capital` (starting cash in TWD) and optional `--portfolio-csv PATH` to write a **daily** table: `cash`, `shares`, `close`, `position_value` (shares × close), `equity`. In code, use `portfolio_daily_table(ohlcv, res)` after `run_rsi_backtest`.

Example notebook: [notebooks/01_2317_rsi.ipynb](notebooks/01_2317_rsi.ipynb).

## Tests

```bash
pytest
```

## API limits

FinMind free tier: **600 requests/hour** — cached Parquet minimizes repeat calls.
