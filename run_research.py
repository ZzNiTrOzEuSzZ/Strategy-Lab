"""
run_research.py
----------------
Universal research runner for StratLab. Works for any strategy, any asset class.
This is the script Layer 4 will call programmatically.

CLI usage examples:
  python run_research.py --strategy bb_breakout
  python run_research.py --strategy bb_breakout --assets fx
  python run_research.py --strategy bb_breakout --assets crypto
  python run_research.py --strategy bb_breakout --ticker EURUSD
  python run_research.py --strategy bb_breakout --mode simple
  python run_research.py --strategy bb_breakout --trials 50 --start 2021-01-01

Arguments:
  --strategy   REQUIRED. Module name inside strategies/ e.g. bb_breakout
  --assets     Asset class: fx | crypto | equities | all
               Default: uses strategy's ASSET_CLASSES attribute
  --ticker     Single ticker. Overrides --assets.
  --mode       simple | full. Default: full
               simple = one backtest with default params, no optimization
               full   = walk_forward + sensitivity + perturbation + cost_stress
  --train      Training bars per fold. Default: from config/assets.yaml
  --test       Test bars per fold. Default: from config/assets.yaml
  --trials     Optuna trials per fold. Default: from config/assets.yaml
  --start      Start date YYYY-MM-DD. Default: from config/assets.yaml
  --end        End date YYYY-MM-DD. Default: all available data
  --no-report  Flag. If set, skip saving JSON reports to results/
"""

from __future__ import annotations

import importlib
import inspect
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from infrastructure.backtester import (
    Research, build_report, save_report, list_available,
)
from strategies.base import BaseStrategy


# ---------------------------------------------------------------------------
# Strategy loader
# ---------------------------------------------------------------------------

def load_strategy_class(module_name: str):
    """
    Dynamically import a strategy module and find the BaseStrategy subclass.
    This is how Layer 4 will inject generated strategy classes.
    """
    module = importlib.import_module(f"strategies.{module_name}")
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
            return obj
    raise ValueError(
        f"No BaseStrategy subclass found in strategies/{module_name}.py"
    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config() -> dict:
    config_path = Path(__file__).parent / "config" / "assets.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Ticker resolution
# ---------------------------------------------------------------------------

def get_tickers(asset_class: str, config: dict) -> list:
    """
    Read tickers from config/assets.yaml for the given asset class.
    Returns list of ticker_display strings.
    """
    assets = config.get(asset_class, [])
    tickers = []
    for a in assets:
        display = (
            a.get("ticker_display")
            or a.get("ticker_binance")
            or a.get("ticker_yfinance")
        )
        if display:
            tickers.append(display)
    return tickers


def _infer_asset_class(ticker: str, config: dict) -> str:
    """Find which asset class a ticker belongs to."""
    for ac in ("fx", "crypto", "equities"):
        for a in config.get(ac, []):
            display = (
                a.get("ticker_display")
                or a.get("ticker_binance")
                or a.get("ticker_yfinance")
            )
            if display == ticker:
                return ac
    return "fx"  # fallback


# ---------------------------------------------------------------------------
# Per-asset runner
# ---------------------------------------------------------------------------

def _save_charts(report_data: dict, report: dict, ticker: str, charts_dir: Path):
    """Save all HTML charts for one asset to charts_dir."""
    charts_dir.mkdir(parents=True, exist_ok=True)

    wf_results = report_data.get("wf_results", {})
    full_metrics = report_data.get("full_metrics")
    sensitivity  = report_data.get("sensitivity")

    try:
        from infrastructure.backtester.visualizer import (
            plot_results, plot_direction_split,
        )
        from infrastructure.backtester.wf_visualizer import (
            plot_fold_performance, plot_parameter_evolution,
            plot_oos_equity, plot_walk_forward_results,
        )
    except ImportError as e:
        print(f"  WARNING: Could not import visualizers: {e}")
        return

    results_df   = wf_results.get("results_df")
    oos_metrics  = wf_results.get("oos_metrics")
    oos_combined = wf_results.get("oos_combined_df")
    fold_records = wf_results.get("fold_records", [])
    stability_df = wf_results.get("stability_df")

    saved = []

    # 1. Full-period equity + drawdown
    if full_metrics is not None:
        try:
            p = str(charts_dir / "equity.html")
            plot_results(full_metrics, show=False, save_html=p)
            saved.append("equity.html")
        except Exception as e:
            print(f"  WARNING equity chart: {e}")

    # 2. Long/short direction split
    if full_metrics is not None and full_metrics.get("direction_split"):
        try:
            p = str(charts_dir / "direction_split.html")
            plot_direction_split(full_metrics["direction_split"], show=False, save_html=p)
            saved.append("direction_split.html")
        except Exception as e:
            print(f"  WARNING direction split chart: {e}")

    # 3. Walk-forward fold IS vs OOS performance
    if results_df is not None and len(results_df) > 0:
        try:
            p = str(charts_dir / "wf_folds.html")
            plot_fold_performance(results_df, show=False, save_html=p)
            saved.append("wf_folds.html")
        except Exception as e:
            print(f"  WARNING wf_folds chart: {e}")

    # 4. Parameter evolution across folds
    if results_df is not None and len(results_df) > 0:
        try:
            from strategies.base import BaseStrategy
            p = str(charts_dir / "params.html")
            plot_parameter_evolution(results_df, {}, show=False, save_html=p)
            saved.append("params.html")
        except Exception as e:
            print(f"  WARNING params chart: {e}")

    # 5. OOS equity curve
    if oos_metrics is not None and oos_combined is not None:
        try:
            boundaries = [r["test_start"] for r in fold_records[1:]] if fold_records else None
            p = str(charts_dir / "oos_equity.html")
            plot_oos_equity(oos_metrics, oos_combined,
                           fold_boundaries=boundaries,
                           show=False, save_html=p)
            saved.append("oos_equity.html")
        except Exception as e:
            print(f"  WARNING oos_equity chart: {e}")

    # 6. Full WF dashboard
    if results_df is not None and len(results_df) > 0:
        try:
            p = str(charts_dir / "wf_dashboard.html")
            plot_walk_forward_results(
                results_df       = results_df,
                oos_metrics      = oos_metrics,
                oos_combined_df  = oos_combined,
                stability_df     = stability_df,
                show             = False,
                save_html        = p,
            )
            saved.append("wf_dashboard.html")
        except Exception as e:
            print(f"  WARNING wf_dashboard chart: {e}")

    if saved:
        print(f"  Charts saved -> {charts_dir}/")
        for s in saved:
            print(f"    {s}")


def _run_one_asset(
    strategy_class,
    ticker:          str,
    asset_class:     str,
    cost:            float,
    mode:            str,
    train_bars:      int,
    test_bars:       int,
    n_trials:        int,
    start_date:      str | None,
    end_date:        str | None,
    save_reports:    bool,
    skip_robustness: bool = False,
) -> dict | None:
    """
    Run the research pipeline for a single ticker.
    Returns a summary dict for the cross-asset table, or None on failure.
    """
    print(f"\n{'='*70}")
    print(f"  ASSET: {ticker}  |  Class: {asset_class.upper()}")
    print(f"{'='*70}")

    strategy_instance = strategy_class()

    try:
        r = Research(
            strategy_fn    = strategy_instance.generate_signals,
            param_space    = strategy_class.PARAM_SPACE,
            primary_ticker = ticker,
            primary_tf     = strategy_class.TIMEFRAMES["primary"],
            aux_timeframes = strategy_class.TIMEFRAMES["aux"],
            cost           = cost,
            start_date     = start_date,
            end_date       = end_date,
        )
    except FileNotFoundError as e:
        print(f"  DATA NOT FOUND: {e}")
        return None
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return None

    # ── Simple mode: single backtest with default params ──────────────────
    if mode == "simple":
        print(f"\n[simple] Running with default parameters...")
        try:
            metrics = r.simple_backtest(params=strategy_class.get_default_params())
        except Exception as e:
            print(f"  ERROR: {e}")
            return None

        print(f"  Trades: {metrics['num_trades']}")
        return {
            "ticker":        ticker,
            "asset_class":   asset_class,
            "num_trades":    metrics["num_trades"],
            "sharpe":        metrics.get("sharpe_ratio", 0) or 0,
            "oos_sharpe":    None,
            "overfit_ratio": None,
            "total_return":  metrics.get("total_return", 0) or 0,
            "max_drawdown":  metrics.get("max_drawdown", 0) or 0,
            "win_rate":      metrics.get("win_rate", 0) or 0,
            "long_sharpe":   None,
            "short_sharpe":  None,
            "tradeable":     None,
            "conclusion":    "simple mode — no walk-forward",
        }

    # ── Full mode: walk-forward + report ──────────────────────────────────
    print(f"\n[1/3] Sanity check — default parameters...")
    try:
        sanity = r.simple_backtest(params=strategy_class.get_default_params())
        print(f"       Trades fired: {sanity['num_trades']}")
        if sanity["num_trades"] == 0:
            print(f"  WARNING: Zero trades with default params.")
            print(f"  Continuing — optimizer may find params that work.")
    except Exception as e:
        print(f"  ERROR in sanity check: {e}")
        return None

    print(f"\n[2/3] Walk-forward optimization ({n_trials} trials per fold)...")
    try:
        report_data = r.full_report(
            train_bars      = train_bars,
            test_bars       = test_bars,
            n_trials        = n_trials,
            skip_robustness = skip_robustness,
        )
    except Exception as e:
        print(f"  ERROR in full_report: {e}")
        return None

    print(f"\n[3/3] Building report and charts...")
    try:
        report = build_report(
            research_results     = report_data,
            strategy_name        = strategy_class.NAME,
            strategy_description = strategy_class.DESCRIPTION,
            ticker               = ticker,
            timeframe            = strategy_class.TIMEFRAMES["primary"],
            cost                 = cost,
        )
        if save_reports:
            save_report(report)
            charts_dir = Path("results") / "charts" / ticker
            _save_charts(report_data, report, ticker, charts_dir)
    except Exception as e:
        print(f"  ERROR building report: {e}")
        return None

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'-'*70}")
    print(f"  RESULTS: {ticker}")
    print(f"{'-'*70}")

    v        = report.get("verdict", {})
    fp       = report.get("full_period", {})
    wf       = report.get("walk_forward", {})
    combined = fp.get("combined") or {}
    lo       = fp.get("long_only") or {}
    so       = fp.get("short_only") or {}

    print(f"\n  FULL PERIOD (consensus params):")
    print(f"    Total Return:    {(combined.get('total_return') or 0)*100:.1f}%")
    print(f"    Sharpe Ratio:    {combined.get('sharpe_ratio') or 0:.2f}")
    print(f"    Max Drawdown:    {(combined.get('max_drawdown') or 0)*100:.1f}%")
    print(f"    Calmar Ratio:    {combined.get('calmar_ratio') or 0:.2f}")
    print(f"    Win Rate:        {(combined.get('win_rate') or 0)*100:.1f}%")
    print(f"    Num Trades:      {combined.get('num_trades') or 0}")
    print(f"    Profit Factor:   {combined.get('profit_factor') or 0:.2f}")

    print(f"\n  LONG ONLY:")
    if lo:
        print(f"    Sharpe: {lo.get('sharpe_ratio') or 0:.2f}  "
              f"Win Rate: {(lo.get('win_rate') or 0)*100:.1f}%  "
              f"Trades: {lo.get('num_trades') or 0}")
    else:
        print(f"    No long trades")

    print(f"\n  SHORT ONLY:")
    if so:
        print(f"    Sharpe: {so.get('sharpe_ratio') or 0:.2f}  "
              f"Win Rate: {(so.get('win_rate') or 0)*100:.1f}%  "
              f"Trades: {so.get('num_trades') or 0}")
    else:
        print(f"    No short trades")

    print(f"\n  WALK-FORWARD OUT-OF-SAMPLE:")
    print(f"    OOS Sharpe:         {wf.get('avg_oos_sharpe') or 0:.2f}")
    print(f"    OOS Return:         {(wf.get('avg_oos_return') or 0)*100:.1f}%")
    print(f"    OOS Drawdown:       {(wf.get('avg_oos_drawdown') or 0)*100:.1f}%")
    print(f"    Folds profitable:   {(wf.get('pct_folds_profitable') or 0)*100:.0f}%")
    print(f"    Overfit ratio:      {wf.get('overfit_ratio') or 0:.2f}")

    tradeable_str = "TRADEABLE" if v.get("tradeable") else "NOT TRADEABLE"
    print(f"\n  VERDICT: {tradeable_str}")
    print(f"    {v.get('conclusion', '')}")
    for w in v.get("warnings", []):
        print(f"    WARNING: {w}")

    return {
        "ticker":        ticker,
        "asset_class":   asset_class,
        "num_trades":    combined.get("num_trades") or 0,
        "sharpe":        combined.get("sharpe_ratio") or 0,
        "oos_sharpe":    wf.get("avg_oos_sharpe") or 0,
        "overfit_ratio": wf.get("overfit_ratio") or 0,
        "total_return":  combined.get("total_return") or 0,
        "max_drawdown":  combined.get("max_drawdown") or 0,
        "win_rate":      combined.get("win_rate") or 0,
        "long_sharpe":   lo.get("sharpe_ratio") if lo else None,
        "short_sharpe":  so.get("sharpe_ratio") if so else None,
        "tradeable":     v.get("tradeable", False),
        "conclusion":    v.get("conclusion", ""),
    }


# ---------------------------------------------------------------------------
# Cross-asset summary table
# ---------------------------------------------------------------------------

def _print_summary_table(rows: list, mode: str):
    if not rows:
        return

    df = pd.DataFrame(rows)

    print(f"\n\n{'#'*70}")
    print(f"  CROSS-ASSET SUMMARY")
    print(f"{'#'*70}\n")

    if mode == "full":
        df = df.sort_values("oos_sharpe", ascending=False)

        header = (
            f"{'Ticker':<12} {'Class':<8} {'Trades':>7} "
            f"{'Sharpe':>8} {'OOS Sharpe':>11} {'Overfit':>8} "
            f"{'Return':>8} {'DD':>8} {'WinRate':>8} "
            f"{'L Sharpe':>9} {'S Sharpe':>9} {'Verdict':>12}"
        )
        print(header)
        print("-" * len(header))

        for _, row in df.iterrows():
            verdict_str = "YES" if row.get("tradeable") else "NO"
            l_sh = f"{row['long_sharpe']:.2f}"  if row.get("long_sharpe")  is not None else "N/A"
            s_sh = f"{row['short_sharpe']:.2f}" if row.get("short_sharpe") is not None else "N/A"
            oos  = row.get("oos_sharpe") or 0
            ovf  = row.get("overfit_ratio") or 0
            print(
                f"{row['ticker']:<12} {row['asset_class']:<8} "
                f"{row['num_trades']:>7} "
                f"{row['sharpe']:>8.2f} {oos:>11.2f} "
                f"{ovf:>8.2f} "
                f"{row['total_return']*100:>7.1f}% "
                f"{row['max_drawdown']*100:>7.1f}% "
                f"{row['win_rate']*100:>7.1f}% "
                f"{l_sh:>9} {s_sh:>9} "
                f"{verdict_str:>12}"
            )

        print(f"\nSUMMARY STATISTICS:")
        tradeable = df[df["tradeable"] == True]
        print(f"  Assets tested:      {len(df)}")
        print(f"  Assets tradeable:   {len(tradeable)}")
        oos_vals = df["oos_sharpe"].dropna()
        ovf_vals = df["overfit_ratio"].dropna()
        print(f"  Avg OOS Sharpe:     {oos_vals.mean():.2f}" if len(oos_vals) else "  Avg OOS Sharpe:     N/A")
        print(f"  Avg Overfit Ratio:  {ovf_vals.mean():.2f}" if len(ovf_vals) else "  Avg Overfit Ratio:  N/A")
        print(f"  Avg Win Rate:       {df['win_rate'].mean()*100:.1f}%")
        if len(df) > 0:
            best = df.iloc[0]
            print(f"  Best asset:         {best['ticker']} (OOS Sharpe {best.get('oos_sharpe') or 0:.2f})")

    else:
        # Simple mode — just show trades + sharpe
        df = df.sort_values("sharpe", ascending=False)
        header = f"{'Ticker':<12} {'Class':<8} {'Trades':>7} {'Sharpe':>8} {'Return':>8} {'WinRate':>8}"
        print(header)
        print("-" * len(header))
        for _, row in df.iterrows():
            print(
                f"{row['ticker']:<12} {row['asset_class']:<8} "
                f"{row['num_trades']:>7} "
                f"{row['sharpe']:>8.2f} "
                f"{row['total_return']*100:>7.1f}% "
                f"{row['win_rate']*100:>7.1f}%"
            )

    print(f"\n{'#'*70}")
    print(f"  RESEARCH COMPLETE  |  Reports saved to: results/")
    print(f"{'#'*70}\n")


# ---------------------------------------------------------------------------
# Main run_research function (callable by Layer 4)
# ---------------------------------------------------------------------------

def run_research(
    strategy_class,
    assets:          str | None  = None,
    ticker:          str | None  = None,
    mode:            str         = "full",
    train_bars:      int | None  = None,
    test_bars:       int | None  = None,
    n_trials:        int | None  = None,
    start_date:      str | None  = None,
    end_date:        str | None  = None,
    save_reports:    bool        = True,
    skip_robustness: bool        = False,
) -> dict:
    """
    Run the complete research pipeline for a strategy.

    This is the single callable that Layer 4 will use.
    Returns a dict keyed by ticker containing each asset's full report.

    Parameters
    ----------
    strategy_class : BaseStrategy subclass
        The strategy to test. Not an instance — the class itself.
    assets : str, optional
        Asset class to test. If None, uses strategy_class.ASSET_CLASSES.
        If "all", runs all three asset classes.
    ticker : str, optional
        Run on a single ticker only. Overrides assets.
    mode : str
        "simple" for a quick single backtest.
        "full" for the complete research pipeline.
    train_bars, test_bars, n_trials : int, optional
        Walk-forward parameters. If None, reads from config/assets.yaml.
    start_date, end_date : str, optional
        Date range. If None, reads from config/assets.yaml.
    save_reports : bool
        Whether to save JSON reports to results/. Default True.

    Returns
    -------
    dict
        {ticker: row_dict} for every asset that was run.
        Also contains a "summary" key with cross-asset stats.
    """
    config   = load_config()
    defaults = config.get("research_defaults", {})
    costs    = config.get("costs", {})

    # Fill defaults from config
    if train_bars  is None: train_bars  = defaults.get("train_bars",  8760)
    if test_bars   is None: test_bars   = defaults.get("test_bars",   4380)
    if n_trials    is None: n_trials    = defaults.get("n_trials",    100)
    if start_date  is None: start_date  = defaults.get("start_date",  "2020-01-01")

    print(f"\n{'#'*70}")
    print(f"  STRATLAB RESEARCH RUNNER")
    print(f"  Strategy: {strategy_class.NAME}")
    print(f"  Mode: {mode.upper()}")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'#'*70}")

    # Confirm data availability
    available = list_available()
    print(f"\nData available:")
    for tf, tickers in sorted(available.items()):
        print(f"  {tf}: {len(tickers)} assets")

    # Build run list
    if ticker:
        asset_class = _infer_asset_class(ticker, config)
        run_list    = [(ticker, asset_class)]
    elif assets == "all":
        run_list = []
        for ac in ("fx", "crypto", "equities"):
            for t in get_tickers(ac, config):
                run_list.append((t, ac))
    elif assets:
        run_list = [(t, assets) for t in get_tickers(assets, config)]
    else:
        run_list = []
        for ac in strategy_class.ASSET_CLASSES:
            for t in get_tickers(ac, config):
                run_list.append((t, ac))

    # Filter to tickers with available 1H data
    primary_tf = strategy_class.TIMEFRAMES["primary"]
    available_tickers = set(available.get(primary_tf, []))
    run_list_filtered = []
    for t, ac in run_list:
        if t not in available_tickers:
            print(f"\nSkipping {t} — no {primary_tf} data")
        else:
            run_list_filtered.append((t, ac))

    print(f"\nRunning {len(run_list_filtered)} asset(s): "
          f"{[t for t, _ in run_list_filtered]}")
    if mode == "full":
        print(f"  train={train_bars}  test={test_bars}  trials={n_trials}  start={start_date}")

    results  = {}
    rows     = []

    for t, ac in run_list_filtered:
        cost = costs.get(ac, 0.001)
        row  = _run_one_asset(
            strategy_class   = strategy_class,
            ticker           = t,
            asset_class      = ac,
            cost             = cost,
            mode             = mode,
            train_bars       = train_bars,
            test_bars        = test_bars,
            n_trials         = n_trials,
            start_date       = start_date,
            end_date         = end_date,
            save_reports     = save_reports,
            skip_robustness  = skip_robustness,
        )
        if row:
            results[t] = row
            rows.append(row)

    _print_summary_table(rows, mode)

    # Build summary stats
    if rows:
        df = pd.DataFrame(rows)
        tradeable_count = int((df["tradeable"] == True).sum()) if "tradeable" in df else 0
        results["summary"] = {
            "assets_tested":    len(rows),
            "assets_tradeable": tradeable_count,
            "avg_oos_sharpe":   float(df["oos_sharpe"].dropna().mean()) if "oos_sharpe" in df else None,
            "avg_win_rate":     float(df["win_rate"].mean()),
        }

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="StratLab Research Runner")
    parser.add_argument("--strategy", required=True,
                        help="Module name inside strategies/ e.g. bb_breakout")
    parser.add_argument("--assets",   default=None,
                        help="fx | crypto | equities | all")
    parser.add_argument("--ticker",   default=None,
                        help="Single ticker, e.g. EURUSD")
    parser.add_argument("--mode",     default="full",
                        choices=["simple", "full"])
    parser.add_argument("--train",    type=int, default=None,
                        help="Training bars per fold")
    parser.add_argument("--test",     type=int, default=None,
                        help="Test bars per fold")
    parser.add_argument("--trials",   type=int, default=None,
                        help="Optuna trials per fold")
    parser.add_argument("--start",    default=None,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      default=None,
                        help="End date YYYY-MM-DD")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip saving JSON reports to results/")
    parser.add_argument("--skip-robustness", action="store_true",
                        help="Skip sensitivity/perturbation/cost_stress (faster runs)")
    args = parser.parse_args()

    strategy_class = load_strategy_class(args.strategy)

    run_research(
        strategy_class   = strategy_class,
        assets           = args.assets,
        ticker           = args.ticker,
        mode             = args.mode,
        train_bars       = args.train,
        test_bars        = args.test,
        n_trials         = args.trials,
        start_date       = args.start,
        end_date         = args.end,
        save_reports     = not args.no_report,
        skip_robustness  = args.skip_robustness,
    )
