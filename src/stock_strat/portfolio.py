"""Multi-name equal-weight top-N rotation (next-open execution, legacy fee mode)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_strat.config import (
    DEFAULT_COMMISSION_PCT,
    DEFAULT_EQUITY_FRACTION,
    DEFAULT_SELL_TAX_PCT,
    DEFAULT_SLIPPAGE_PCT,
    INITIAL_CAPITAL,
)


def _apply_min_comm(comm: float, min_commission_twd: float) -> float:
    if min_commission_twd <= 0:
        return comm
    return max(comm, min_commission_twd)


@dataclass
class PortfolioBacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: list[dict]


def equal_weight_combine_equity(
    equity_by_symbol: dict[str, pd.Series],
    *,
    initial_cash: float = INITIAL_CAPITAL,
) -> pd.Series:
    """Intersection of dates; average normalized equity curves, scale by ``initial_cash``."""
    if not equity_by_symbol:
        raise ValueError("equity_by_symbol empty")
    series = [s.dropna().astype(float) for s in equity_by_symbol.values()]
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)
    idx = idx.sort_values()
    if len(idx) < 2:
        raise ValueError("insufficient overlap")
    norm = None
    for s in series:
        sub = s.reindex(idx).ffill()
        n = sub / float(sub.iloc[0])
        norm = n if norm is None else norm + n
    norm = norm / len(series)
    return norm * initial_cash


def simulate_top_n_entry_rotation(
    panels: dict[str, pd.DataFrame],
    *,
    top_n: int = 3,
    initial_cash: float = INITIAL_CAPITAL,
    fee_pct_per_trade: float | None = 0.001,
    commission_pct: float = DEFAULT_COMMISSION_PCT,
    sell_tax_pct: float = DEFAULT_SELL_TAX_PCT,
    min_commission_twd: float = 0.0,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    equity_fraction: float = DEFAULT_EQUITY_FRACTION,
) -> PortfolioBacktestResult:
    """
    Daily: at open *t*, liquidate; among names with ``signal_entry_cross`` on prior close,
    take up to ``top_n`` lowest RSI; deploy ``equity_fraction`` of cash split equally.
    Supports legacy single fee or Taiwan commission + sell tax.
    """
    if not panels:
        raise ValueError("panels empty")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    common: pd.Index | None = None
    for df in panels.values():
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) < 3:
        raise ValueError("no common index")
    common = common.sort_values()
    dates = list(common)

    legacy = fee_pct_per_trade is not None
    f = float(fee_pct_per_trade) if legacy else 0.0
    slip = float(slippage_pct)
    eqf = max(1e-9, min(1.0, float(equity_fraction)))

    cash = float(initial_cash)
    positions: dict[str, float] = {}
    trades: list[dict] = []
    equity_vals: list[float] = []

    def mark_equity(t: pd.Timestamp) -> float:
        tot = cash
        for sym, sh in positions.items():
            if sh <= 0:
                continue
            cl = float(panels[sym].reindex(common).ffill().loc[t, "close"])
            tot += sh * cl
        return tot

    equity_vals.append(mark_equity(dates[0]))

    for ti in range(1, len(dates)):
        t = dates[ti]
        prev = dates[ti - 1]

        for sym, sh in list(positions.items()):
            if sh <= 0:
                continue
            row = panels[sym].reindex(common).ffill().loc[t]
            o_raw = float(row["open"])
            o_sell = o_raw * (1.0 - slip)
            gross = sh * o_sell
            if legacy:
                comm = f * gross
                comm = _apply_min_comm(comm, min_commission_twd)
                cash += gross - comm
                trades.append({"bar": t, "symbol": sym, "side": "sell", "price": o_sell, "shares": sh, "fee": comm})
            else:
                comm = commission_pct * gross
                comm = _apply_min_comm(comm, min_commission_twd)
                tax = sell_tax_pct * gross
                cash += gross - comm - tax
                trades.append(
                    {
                        "bar": t,
                        "symbol": sym,
                        "side": "sell",
                        "price": o_sell,
                        "shares": sh,
                        "fee": comm + tax,
                    }
                )
            del positions[sym]

        scored: list[tuple[str, float]] = []
        for sym, df in panels.items():
            d = df.reindex(common).ffill()
            try:
                if bool(d.loc[prev, "signal_entry_cross"]):
                    scored.append((sym, float(d.loc[prev, "rsi"])))
            except (KeyError, TypeError):
                continue
        scored.sort(key=lambda x: x[1])
        chosen = [s[0] for s in scored[:top_n]]

        deploy = cash * eqf
        leftover = cash - deploy
        if chosen and deploy > 0:
            per = deploy / len(chosen)
            for sym in chosen:
                row = panels[sym].reindex(common).ffill().loc[t]
                o_raw = float(row["open"])
                o_buy = o_raw * (1.0 + slip)
                if legacy:
                    comm_raw = f * per
                    comm = _apply_min_comm(comm_raw, min_commission_twd)
                    invest = per - comm
                else:
                    comm_raw = commission_pct * per
                    comm = _apply_min_comm(comm_raw, min_commission_twd)
                    invest = per - comm
                sh = invest / o_buy if o_buy > 0 else 0.0
                positions[sym] = sh
                trades.append({"bar": t, "symbol": sym, "side": "buy", "price": o_buy, "shares": sh, "fee": comm})
            cash = leftover
        else:
            cash = cash

        equity_vals.append(mark_equity(t))

    eq_s = pd.Series(equity_vals, index=dates[: len(equity_vals)])
    rets = eq_s.pct_change().fillna(0.0)
    return PortfolioBacktestResult(equity=eq_s, returns=rets, trades=trades)
