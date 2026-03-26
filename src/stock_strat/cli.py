"""CLI entry point."""

from __future__ import annotations

import argparse
import json
import sys

from stock_strat.backtest import run_rsi_backtest
from stock_strat.backtest import portfolio_daily_table
from stock_strat.config import (
    DEFAULT_COMMISSION_PCT,
    DEFAULT_END,
    DEFAULT_SELL_TAX_PCT,
    DEFAULT_START,
    FEE_PCT_PER_TRADE,
    INITIAL_CAPITAL,
    SYMBOL_TWSE,
)
from stock_strat.metrics import performance_summary, trade_stats
from stock_strat.pipeline import build_strategy_frame
from stock_strat.stress import fee_sensitivity, subperiod_report
from stock_strat.walkforward import run_walk_forward


def _cmd_run(args: argparse.Namespace) -> int:
    df = build_strategy_frame(
        start_date=args.start,
        end_date=args.end,
        stock_id=args.stock,
        refresh=args.refresh,
    )
    ohlcv = df[["open", "high", "low", "close", "volume"]]
    if args.legacy_fee is not None:
        res = run_rsi_backtest(
            ohlcv,
            entries=df["signal_entry_cross"],
            exits=df["signal_exit_cross"],
            initial_cash=args.capital,
            fee_pct_per_trade=args.legacy_fee,
        )
    else:
        res = run_rsi_backtest(
            ohlcv,
            entries=df["signal_entry_cross"],
            exits=df["signal_exit_cross"],
            initial_cash=args.capital,
            fee_pct_per_trade=None,
            commission_pct=args.commission,
            sell_tax_pct=args.sell_tax,
        )
    perf = performance_summary(res.equity, returns=res.returns)
    perf.update(trade_stats(res.trades))
    perf["stock_id"] = args.stock
    perf["initial_capital"] = args.capital
    perf["final_equity"] = float(res.equity.iloc[-1])
    if args.legacy_fee is not None:
        perf["fee_model"] = "legacy"
        perf["legacy_fee"] = args.legacy_fee
    else:
        perf["fee_model"] = "taiwan"
        perf["commission_pct"] = args.commission
        perf["sell_tax_pct"] = args.sell_tax
    if args.portfolio_csv:
        daily = portfolio_daily_table(ohlcv, res)
        daily.to_csv(args.portfolio_csv)
        perf["portfolio_csv"] = args.portfolio_csv
    print(json.dumps(perf, indent=2))
    return 0


def _cmd_walkforward(args: argparse.Namespace) -> int:
    df = build_strategy_frame(
        start_date=args.start,
        end_date=args.end,
        stock_id=args.stock,
        refresh=args.refresh,
    )
    ohlcv = df[["open", "high", "low", "close", "volume"]]
    wf = run_walk_forward(
        ohlcv,
        df["signal_entry_cross"],
        df["signal_exit_cross"],
        fee_pct=args.fee,
    )
    print(wf.to_json(orient="records", indent=2))
    return 0


def _cmd_stress(args: argparse.Namespace) -> int:
    df = build_strategy_frame(
        start_date=args.start,
        end_date=args.end,
        stock_id=args.stock,
        refresh=args.refresh,
    )
    ohlcv = df[["open", "high", "low", "close", "volume"]]
    fees = [0.0, 0.001, 0.002, 0.005]
    sens = fee_sensitivity(
        ohlcv,
        df["signal_entry_cross"],
        df["signal_exit_cross"],
        fees,
    )
    periods = {
        "covid_slice": ("2020-02-01", "2020-04-30"),
        "full": (args.start, args.end),
    }
    sub = subperiod_report(
        ohlcv,
        df["signal_entry_cross"],
        df["signal_exit_cross"],
        periods=periods,
        fee_pct=args.fee,
    )
    out = {"fee_sensitivity": sens.to_dict(orient="records"), "subperiods": sub.to_dict(orient="records")}
    print(json.dumps(out, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--start", default=DEFAULT_START)
    common.add_argument("--end", default=DEFAULT_END)
    common.add_argument("--refresh", action="store_true", help="Bypass FinMind cache")
    common.add_argument(
        "--stock",
        default=SYMBOL_TWSE,
        metavar="TWSE",
        help="TWSE code for FinMind TaiwanStockPrice (default: %(default)s Hon Hai; e.g. 2330 TSMC)",
    )

    p = argparse.ArgumentParser(prog="stock-strat")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", parents=[common], help="Full-sample backtest metrics")
    run_p.add_argument(
        "--capital",
        type=float,
        default=INITIAL_CAPITAL,
        metavar="TWD",
        help="Starting cash (default: %(default)s)",
    )
    run_p.add_argument(
        "--portfolio-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Write daily cash / shares / position value / equity to CSV",
    )
    run_p.add_argument(
        "--commission",
        type=float,
        default=DEFAULT_COMMISSION_PCT,
        metavar="FRAC",
        help="Brokerage fraction of trade amount (buy and sell); default ≈ 永豐大戶投 2 折 on 0.1425%% cap",
    )
    run_p.add_argument(
        "--sell-tax",
        type=float,
        default=DEFAULT_SELL_TAX_PCT,
        dest="sell_tax",
        metavar="FRAC",
        help="證交稅 fraction of sell gross (現股一般 0.003); 0 to disable",
    )
    run_p.add_argument(
        "--legacy-fee",
        type=float,
        default=None,
        metavar="FRAC",
        dest="legacy_fee",
        help="If set, use single fee on buy cash and sell gross (ignores --commission and --sell-tax)",
    )

    wf_p = sub.add_parser("walkforward", parents=[common], help="Out-of-sample windows")
    wf_p.add_argument(
        "--fee",
        type=float,
        default=FEE_PCT_PER_TRADE,
        help="Legacy single fee rate (same as --legacy-fee on run)",
    )

    st_p = sub.add_parser("stress", parents=[common], help="Fee grid + subperiod report")
    st_p.add_argument(
        "--fee",
        type=float,
        default=FEE_PCT_PER_TRADE,
        help="Legacy single fee rate for subperiod runs",
    )

    args = p.parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "walkforward":
        return _cmd_walkforward(args)
    if args.cmd == "stress":
        return _cmd_stress(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
