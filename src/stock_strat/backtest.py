"""
Long-only portfolio simulation (pure pandas).

Execution model (documented):
- Signals use bar **close** on day t (`entries`/`exits` aligned to that index).
- Fills at **next bar open** (t+1): on day t+1's open we execute the decision from t.

**Costs (two modes):**

1. **Taiwan-style (default):** brokerage **commission** as a fraction of trade amount on **buy** (cash
   deployed) and **sell** (gross proceeds), plus **證交稅** ``sell_tax_pct`` on **sell gross** only.
2. **Legacy single rate:** ``fee_pct_per_trade`` applied to buy cash and sell gross (no separate sell
   tax). Used when ``fee_pct_per_trade`` is not ``None`` (e.g. stress sweeps).

Slippage can be modeled by raising commission or adding a separate model later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_strat.config import DEFAULT_COMMISSION_PCT, DEFAULT_SELL_TAX_PCT, INITIAL_CAPITAL


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: list[dict]
    cash: pd.Series
    shares: pd.Series


def run_rsi_backtest(
    ohlcv: pd.DataFrame,
    *,
    entries: pd.Series,
    exits: pd.Series,
    initial_cash: float = INITIAL_CAPITAL,
    fee_pct_per_trade: float | None = None,
    commission_pct: float = DEFAULT_COMMISSION_PCT,
    sell_tax_pct: float = DEFAULT_SELL_TAX_PCT,
) -> BacktestResult:
    """
    Long-only: deploy 100% cash on entry, flat on exit.

    If ``fee_pct_per_trade`` is set, it overrides ``commission_pct`` / ``sell_tax_pct`` and applies one
    rate to buy cash and sell gross (legacy). If ``None``, uses commission on both legs and
    ``sell_tax_pct`` on sells only (Taiwan-style defaults).
    """
    idx = ohlcv.index
    open_ = ohlcv["open"].astype(float).reindex(idx)
    close = ohlcv["close"].astype(float).reindex(idx)
    ent = entries.reindex(idx).fillna(False).astype(bool)
    ex = exits.reindex(idx).fillna(False).astype(bool)

    n = len(idx)
    cash_arr = np.zeros(n, dtype=float)
    sh_arr = np.zeros(n, dtype=float)
    trades: list[dict] = []

    c = float(initial_cash)
    sh = 0.0
    cash_arr[0] = c
    sh_arr[0] = sh

    legacy = fee_pct_per_trade is not None
    f = float(fee_pct_per_trade) if legacy else 0.0

    for t in range(1, n):
        o = float(open_.iloc[t])
        if ex.iloc[t - 1] and sh > 0:
            gross = sh * o
            if legacy:
                fee_total = f * gross
                c = gross - fee_total
                trades.append(
                    {
                        "bar": idx[t],
                        "side": "sell",
                        "price": o,
                        "shares": sh,
                        "fee": fee_total,
                        "commission": fee_total,
                        "sell_tax": 0.0,
                    }
                )
            else:
                comm = commission_pct * gross
                tax = sell_tax_pct * gross
                fee_total = comm + tax
                c = gross - fee_total
                trades.append(
                    {
                        "bar": idx[t],
                        "side": "sell",
                        "price": o,
                        "shares": sh,
                        "fee": fee_total,
                        "commission": comm,
                        "sell_tax": tax,
                    }
                )
            sh = 0.0
        elif ent.iloc[t - 1] and sh == 0 and c > 0:
            fee_total = f * c if legacy else commission_pct * c
            invest = c - fee_total
            sh = invest / o if o > 0 else 0.0
            trades.append(
                {
                    "bar": idx[t],
                    "side": "buy",
                    "price": o,
                    "shares": sh,
                    "fee": fee_total,
                    "commission": fee_total,
                    "sell_tax": 0.0,
                }
            )
            c = 0.0
        cash_arr[t] = c
        sh_arr[t] = sh

    cash_s = pd.Series(cash_arr, index=idx)
    sh_s = pd.Series(sh_arr, index=idx)
    equity = cash_s + sh_s * close
    rets = equity.pct_change().fillna(0.0)
    return BacktestResult(
        equity=equity,
        returns=rets,
        trades=trades,
        cash=cash_s,
        shares=sh_s,
    )


def portfolio_daily_table(ohlcv: pd.DataFrame, res: BacktestResult) -> pd.DataFrame:
    """
    Day-by-day account view: cash, shares, mark-to-market at **close**, total equity.

    After a buy, cash goes to ~0 and ``position_value`` tracks shares × close.
    After a sell, cash holds proceeds (net of fee) and position_value is 0.
    """
    idx = ohlcv.index
    close = ohlcv["close"].astype(float).reindex(idx)
    pos = res.shares * close
    return pd.DataFrame(
        {
            "cash": res.cash.reindex(idx),
            "shares": res.shares.reindex(idx),
            "close": close,
            "position_value": pos,
            "equity": res.equity.reindex(idx),
        }
    )
