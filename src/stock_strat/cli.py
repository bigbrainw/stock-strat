"""CLI entry point."""

from __future__ import annotations

import argparse
import json
import sys

from stock_strat.backtest import run_rsi_backtest
from stock_strat.backtest import portfolio_daily_table
from stock_strat.benchmarks import (
    benchmark_buy_and_hold,
    benchmark_ma_trend,
    fantasy_same_close_fills,
)
from stock_strat.config import (
    DEFAULT_COMMISSION_PCT,
    DEFAULT_END,
    DEFAULT_SLIPPAGE_PCT,
    DEFAULT_SELL_TAX_PCT,
    DEFAULT_START,
    FEE_PCT_PER_TRADE,
    INITIAL_CAPITAL,
    REGIME_SMA_DAYS,
    RSI_ENTRY,
    RSI_EXIT,
    RSI_PERIOD,
    SYMBOL_TWSE,
)
from stock_strat.events import ex_dividend_experiment
from stock_strat.grid_search import default_grid, run_grid_on_clean, stability_summary
from stock_strat.metrics import performance_summary, trade_stats
from stock_strat.pipeline import build_strategy_frame, load_clean_ohlcv
from stock_strat.portfolio import equal_weight_combine_equity, simulate_top_n_entry_rotation
from stock_strat.stats_validation import bootstrap_sharpe, bonferroni_alpha, permutation_sharpe_pvalue
from stock_strat.stress import fee_sensitivity, subperiod_report
from stock_strat.universe import load_ticker_list, run_universe
from stock_strat.walkforward import run_walk_forward, walk_forward_optimize


def _cmd_run(args: argparse.Namespace) -> int:
    df = build_strategy_frame(
        start_date=args.start,
        end_date=args.end,
        stock_id=args.stock,
        refresh=args.refresh,
        rsi_period=args.rsi_period,
        rsi_entry=args.rsi_entry,
        rsi_exit=args.rsi_exit,
        regime_above_sma=args.regime_sma,
        regime_require_index_above_sma=args.regime_require_index,
        index_stock_id=args.index_stock,
        index_sma_days=args.index_sma_days,
        vol_percentile_max=args.vol_pct_max,
    )
    ohlcv = df[["open", "high", "low", "close", "volume"]]
    bt_kw = dict(
        initial_cash=args.capital,
        min_commission_twd=args.min_commission,
        slippage_pct=args.slippage,
        equity_fraction=args.equity_fraction,
        skip_gap_abs_pct=args.skip_gap_pct,
        skip_intraday_range_pct=args.skip_range_pct,
    )
    if args.legacy_fee is not None:
        res = run_rsi_backtest(
            ohlcv,
            entries=df["signal_entry_cross"],
            exits=df["signal_exit_cross"],
            fee_pct_per_trade=args.legacy_fee,
            **bt_kw,
        )
    else:
        res = run_rsi_backtest(
            ohlcv,
            entries=df["signal_entry_cross"],
            exits=df["signal_exit_cross"],
            fee_pct_per_trade=None,
            commission_pct=args.commission,
            sell_tax_pct=args.sell_tax,
            **bt_kw,
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

    bh = benchmark_buy_and_hold(ohlcv, initial_cash=args.capital)
    perf["buy_and_hold"] = bh
    perf["excess_total_return_vs_buy_hold"] = float(perf["total_return"]) - float(bh.get("total_return", 0.0))
    perf["excess_cagr_vs_buy_hold"] = float(perf["cagr"]) - float(bh.get("cagr", 0.0))

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
        regime_above_sma=args.regime_sma,
        regime_require_index_above_sma=args.regime_require_index,
        index_stock_id=args.index_stock,
        vol_percentile_max=args.vol_pct_max,
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


def _cmd_grid(args: argparse.Namespace) -> int:
    clean = load_clean_ohlcv(args.start, args.end, stock_id=args.stock, refresh=args.refresh)
    index_clean = None
    if args.index_stock:
        index_clean = load_clean_ohlcv(args.start, args.end, stock_id=args.index_stock, refresh=args.refresh)
    grid = default_grid()
    df = run_grid_on_clean(
        clean,
        grid=grid,
        initial_cash=args.capital,
        fee_pct_per_trade=args.legacy_fee,
        commission_pct=args.commission,
        sell_tax_pct=args.sell_tax,
        regime_above_sma=args.regime_sma,
        regime_require_index_above_sma=args.regime_require_index,
        index_clean=index_clean,
        index_sma_days=args.index_sma_days or REGIME_SMA_DAYS,
        vol_percentile_max=args.vol_pct_max,
        min_commission_twd=args.min_commission,
        slippage_pct=args.slippage,
        equity_fraction=args.equity_fraction,
        skip_gap_abs_pct=args.skip_gap_pct,
    )
    out: dict = {"rows": df.to_dict(orient="records"), "stability": stability_summary(df, metric=args.metric)}
    if args.legacy_fee is None:
        out["fee_model"] = "taiwan"
    else:
        out["fee_model"] = "legacy"
    print(json.dumps(out, indent=2, default=str))
    return 0


def _cmd_wf_optimize(args: argparse.Namespace) -> int:
    clean = load_clean_ohlcv(args.start, args.end, stock_id=args.stock, refresh=args.refresh)
    wf = walk_forward_optimize(
        clean,
        stock_id=args.stock,
        optimize_metric=args.metric,
        initial_cash=args.capital,
        fee_pct_per_trade=args.legacy_fee,
        regime_above_sma=args.regime_sma,
        regime_require_index_above_sma=args.regime_require_index,
        index_stock_id=args.index_stock,
        index_sma_days=args.index_sma_days,
        vol_percentile_max=args.vol_pct_max,
        min_commission_twd=args.min_commission,
        slippage_pct=args.slippage,
        equity_fraction=args.equity_fraction,
        skip_gap_abs_pct=args.skip_gap_pct,
    )
    print(wf.to_json(orient="records", indent=2, default_handler=str))
    return 0


def _cmd_run_universe(args: argparse.Namespace) -> int:
    tickers = load_ticker_list(args.ticker_file)
    df = run_universe(
        tickers,
        args.start,
        args.end,
        refresh=args.refresh,
        min_avg_turnover_twd=args.min_turnover,
        min_price=args.min_price,
        fee_pct_per_trade=args.legacy_fee,
        initial_cash=args.capital,
        min_commission_twd=args.min_commission,
        slippage_pct=args.slippage,
        equity_fraction=args.equity_fraction,
        regime_above_sma=args.regime_sma,
        vol_percentile_max=args.vol_pct_max,
        index_stock_id=args.index_stock,
        regime_require_index_above_sma=args.regime_require_index,
    )
    print(df.to_json(orient="records", indent=2, default_handler=str))
    return 0


def _cmd_benchmarks(args: argparse.Namespace) -> int:
    df = build_strategy_frame(
        start_date=args.start,
        end_date=args.end,
        stock_id=args.stock,
        refresh=args.refresh,
    )
    ohlcv = df[["open", "high", "low", "close", "volume"]]
    rsi_res = run_rsi_backtest(
        ohlcv,
        entries=df["signal_entry_cross"],
        exits=df["signal_exit_cross"],
        initial_cash=args.capital,
        fee_pct_per_trade=args.legacy_fee,
    )
    out = {
        "buy_and_hold": benchmark_buy_and_hold(ohlcv, initial_cash=args.capital),
        "ma_trend": benchmark_ma_trend(ohlcv, initial_cash=args.capital, fee_pct_per_trade=args.legacy_fee),
        "rsi_strategy": performance_summary(rsi_res.equity, returns=rsi_res.returns),
        "fantasy_same_close": fantasy_same_close_fills(
            ohlcv,
            df["signal_entry_cross"],
            df["signal_exit_cross"],
            initial_cash=args.capital,
        ),
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    df = build_strategy_frame(
        start_date=args.start,
        end_date=args.end,
        stock_id=args.stock,
        refresh=args.refresh,
    )
    ohlcv = df[["open", "high", "low", "close", "volume"]]
    res = run_rsi_backtest(
        ohlcv,
        entries=df["signal_entry_cross"],
        exits=df["signal_exit_cross"],
        initial_cash=args.capital,
        fee_pct_per_trade=args.legacy_fee,
    )
    r = res.returns
    perf = performance_summary(res.equity, returns=r)
    obs = float(perf.get("sharpe", float("nan")))
    out = {
        "performance": perf,
        "bootstrap_sharpe": bootstrap_sharpe(r, n_bootstrap=args.bootstrap_n, seed=args.seed),
        "permutation_pvalue_one_sided": permutation_sharpe_pvalue(
            r, observed_sharpe=obs, n_perm=args.perm_n, seed=args.seed
        ),
        "bonferroni_alpha": bonferroni_alpha(args.alpha, args.n_grid_tests),
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    out = ex_dividend_experiment(
        args.stock,
        args.start,
        args.end,
        window_days=args.ex_div_window,
        refresh=args.refresh,
        fee_pct_per_trade=args.legacy_fee,
    )
    print(json.dumps(out, indent=2))
    return 0


def _cmd_portfolio_top_n(args: argparse.Namespace) -> int:
    tickers = load_ticker_list(args.ticker_file)[: args.max_names]
    panels: dict[str, object] = {}
    for t in tickers:
        try:
            clean = load_clean_ohlcv(args.start, args.end, stock_id=t, refresh=args.refresh)
        except Exception:
            continue
        df = build_strategy_frame(
            args.start,
            args.end,
            stock_id=t,
            refresh=False,
            clean=clean,
            regime_above_sma=args.regime_sma,
            vol_percentile_max=args.vol_pct_max,
        )
        panels[t] = df
    if len(panels) < 2:
        print(json.dumps({"error": "need at least 2 valid tickers"}))
        return 1
    pres = simulate_top_n_entry_rotation(
        panels,
        top_n=args.top_n,
        initial_cash=args.capital,
        fee_pct_per_trade=args.legacy_fee,
        min_commission_twd=args.min_commission,
        slippage_pct=args.slippage,
        equity_fraction=args.equity_fraction,
    )
    perf = performance_summary(pres.equity, returns=pres.returns)
    perf["n_trades"] = len(pres.trades)
    print(json.dumps(perf, indent=2))
    return 0


def _cmd_portfolio_ew_combine(args: argparse.Namespace) -> int:
    tickers = load_ticker_list(args.ticker_file)[: args.max_names]
    curves: dict[str, object] = {}
    for t in tickers:
        try:
            df = build_strategy_frame(args.start, args.end, stock_id=t, refresh=args.refresh)
        except Exception:
            continue
        ohlcv = df[["open", "high", "low", "close", "volume"]]
        res = run_rsi_backtest(
            ohlcv,
            entries=df["signal_entry_cross"],
            exits=df["signal_exit_cross"],
            initial_cash=args.capital,
            fee_pct_per_trade=args.legacy_fee,
        )
        curves[t] = res.equity
    if len(curves) < 2:
        print(json.dumps({"error": "need at least 2 valid tickers"}))
        return 1
    combined = equal_weight_combine_equity(curves, initial_cash=args.capital)
    perf = performance_summary(combined, returns=combined.pct_change().fillna(0.0))
    print(json.dumps(perf, indent=2))
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

    regime = argparse.ArgumentParser(add_help=False)
    regime.add_argument("--regime-sma", type=int, default=None, help="Stock SMA days for trend gate (entries)")
    regime.add_argument("--vol-pct-max", type=float, default=None, dest="vol_pct_max", help="Vol percentile cap 0-1")
    regime.add_argument(
        "--index-stock",
        type=str,
        default=None,
        dest="index_stock",
        help="TWSE code for index regime merge (e.g. 0050)",
    )
    regime.add_argument(
        "--regime-require-index",
        action="store_true",
        dest="regime_require_index",
        help="Require index close > index SMA for entries",
    )
    regime.add_argument(
        "--index-sma-days",
        type=int,
        default=None,
        dest="index_sma_days",
        help="SMA window for index series (default: regime-sma or 200)",
    )

    exec_flags = argparse.ArgumentParser(add_help=False)
    exec_flags.add_argument(
        "--min-commission",
        type=float,
        default=0.0,
        dest="min_commission",
        metavar="TWD",
        help="Minimum commission per brokerage leg (0 = off; e.g. 1 for typical TW floor)",
    )
    exec_flags.add_argument(
        "--slippage",
        type=float,
        default=DEFAULT_SLIPPAGE_PCT,
        help="Fraction added to buy open / subtracted from sell open",
    )
    exec_flags.add_argument(
        "--equity-fraction",
        type=float,
        default=1.0,
        dest="equity_fraction",
        help="Fraction of cash to deploy on each entry",
    )
    exec_flags.add_argument(
        "--skip-gap-pct",
        type=float,
        default=None,
        dest="skip_gap_pct",
        metavar="FRAC",
        help="Skip fills when abs(open/close_prev-1) exceeds this",
    )
    exec_flags.add_argument(
        "--skip-range-pct",
        type=float,
        default=None,
        dest="skip_range_pct",
        metavar="FRAC",
        help="Skip fills when (high-low)/open exceeds this",
    )

    p = argparse.ArgumentParser(prog="stock-strat")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", parents=[common, regime, exec_flags], help="Full-sample backtest metrics")
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
    run_p.add_argument("--rsi-period", type=int, default=RSI_PERIOD, dest="rsi_period")
    run_p.add_argument("--rsi-entry", type=float, default=RSI_ENTRY, dest="rsi_entry")
    run_p.add_argument("--rsi-exit", type=float, default=RSI_EXIT, dest="rsi_exit")
    run_p.add_argument(
        "--commission",
        type=float,
        default=DEFAULT_COMMISSION_PCT,
        metavar="FRAC",
        help="Brokerage fraction of trade amount (buy and sell)",
    )
    run_p.add_argument(
        "--sell-tax",
        type=float,
        default=DEFAULT_SELL_TAX_PCT,
        dest="sell_tax",
        metavar="FRAC",
        help="證交稅 fraction of sell gross",
    )
    run_p.add_argument(
        "--legacy-fee",
        type=float,
        default=None,
        metavar="FRAC",
        dest="legacy_fee",
        help="If set, legacy single fee on buy cash and sell gross",
    )

    wf_p = sub.add_parser("walkforward", parents=[common, regime], help="Out-of-sample windows")
    wf_p.add_argument(
        "--fee",
        type=float,
        default=FEE_PCT_PER_TRADE,
        help="Legacy single fee rate",
    )

    st_p = sub.add_parser("stress", parents=[common], help="Fee grid + subperiod report")
    st_p.add_argument(
        "--fee",
        type=float,
        default=FEE_PCT_PER_TRADE,
        help="Legacy single fee rate for subperiod runs",
    )

    grid_p = sub.add_parser("grid", parents=[common, regime, exec_flags], help="RSI grid search + stability")
    grid_p.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    grid_p.add_argument(
        "--legacy-fee",
        type=float,
        default=FEE_PCT_PER_TRADE,
        dest="legacy_fee",
        help="Legacy single fee (default 0.001); omit by using --use-taiwan-fees",
    )
    grid_p.add_argument("--use-taiwan-fees", action="store_true", help="Use commission + sell tax instead of legacy")
    grid_p.add_argument("--commission", type=float, default=DEFAULT_COMMISSION_PCT)
    grid_p.add_argument("--sell-tax", type=float, default=DEFAULT_SELL_TAX_PCT, dest="sell_tax")
    grid_p.add_argument("--metric", type=str, default="sharpe", help="Metric for stability_summary top-k")

    wfo_p = sub.add_parser("wf-optimize", parents=[common, regime, exec_flags], help="Walk-forward optimize grid")
    wfo_p.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    wfo_p.add_argument("--legacy-fee", type=float, default=FEE_PCT_PER_TRADE, dest="legacy_fee")
    wfo_p.add_argument("--metric", type=str, default="sharpe", dest="metric")

    uni_p = sub.add_parser("run-universe", parents=[common, regime, exec_flags], help="Batch tickers + liquidity")
    uni_p.add_argument(
        "ticker_file",
        type=str,
        help="Path to text file of TWSE codes",
    )
    uni_p.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    uni_p.add_argument("--legacy-fee", type=float, default=FEE_PCT_PER_TRADE, dest="legacy_fee")
    uni_p.add_argument("--min-turnover", type=float, default=0.0, dest="min_turnover")
    uni_p.add_argument("--min-price", type=float, default=5.0, dest="min_price")

    bench_p = sub.add_parser("benchmarks", parents=[common], help="Buy-hold, MA trend, RSI, fantasy")
    bench_p.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    bench_p.add_argument("--legacy-fee", type=float, default=FEE_PCT_PER_TRADE, dest="legacy_fee")

    stats_p = sub.add_parser("stats", parents=[common], help="Bootstrap / permutation / Bonferroni")
    stats_p.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    stats_p.add_argument("--legacy-fee", type=float, default=FEE_PCT_PER_TRADE, dest="legacy_fee")
    stats_p.add_argument("--bootstrap-n", type=int, default=1000, dest="bootstrap_n")
    stats_p.add_argument("--perm-n", type=int, default=1000, dest="perm_n")
    stats_p.add_argument("--seed", type=int, default=0)
    stats_p.add_argument("--alpha", type=float, default=0.05)
    stats_p.add_argument("--n-grid-tests", type=int, default=1, dest="n_grid_tests")

    ev_p = sub.add_parser("events", parents=[common], help="Ex-dividend window split metrics")
    ev_p.add_argument("--legacy-fee", type=float, default=FEE_PCT_PER_TRADE, dest="legacy_fee")
    ev_p.add_argument("--ex-div-window", type=int, default=3, dest="ex_div_window")

    ptop_p = sub.add_parser("portfolio-top-n", parents=[common, exec_flags], help="Top-N daily rotation")
    ptop_p.add_argument("ticker_file", type=str)
    ptop_p.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    ptop_p.add_argument("--legacy-fee", type=float, default=FEE_PCT_PER_TRADE, dest="legacy_fee")
    ptop_p.add_argument("--top-n", type=int, default=3, dest="top_n")
    ptop_p.add_argument("--max-names", type=int, default=20, dest="max_names")
    ptop_p.add_argument("--regime-sma", type=int, default=None, dest="regime_sma")
    ptop_p.add_argument("--vol-pct-max", type=float, default=None, dest="vol_pct_max")

    pew_p = sub.add_parser("portfolio-ew-combine", parents=[common], help="Equal-weight combine single-name curves")
    pew_p.add_argument("ticker_file", type=str)
    pew_p.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    pew_p.add_argument("--legacy-fee", type=float, default=FEE_PCT_PER_TRADE, dest="legacy_fee")
    pew_p.add_argument("--max-names", type=int, default=20, dest="max_names")

    args = p.parse_args(argv)

    if args.cmd == "grid" and getattr(args, "use_taiwan_fees", False):
        args.legacy_fee = None

    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "walkforward":
        return _cmd_walkforward(args)
    if args.cmd == "stress":
        return _cmd_stress(args)
    if args.cmd == "grid":
        return _cmd_grid(args)
    if args.cmd == "wf-optimize":
        return _cmd_wf_optimize(args)
    if args.cmd == "run-universe":
        return _cmd_run_universe(args)
    if args.cmd == "benchmarks":
        return _cmd_benchmarks(args)
    if args.cmd == "stats":
        return _cmd_stats(args)
    if args.cmd == "events":
        return _cmd_events(args)
    if args.cmd == "portfolio-top-n":
        return _cmd_portfolio_top_n(args)
    if args.cmd == "portfolio-ew-combine":
        return _cmd_portfolio_ew_combine(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
