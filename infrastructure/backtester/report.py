"""
infrastructure/backtester/report.py
--------------------------------------
Structured research report builder.

Takes the output of Research.full_report() and produces a JSON-serialisable
report dict plus a plain-English verdict. This is the contract between Layer 2
and any future Layer 4 (narrative generation / dashboard).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _to_python(obj):
    """
    Recursively convert numpy/pandas types to native Python types so that
    the result is JSON-serialisable.
    """
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.reset_index().to_dict(orient="records")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, float) and np.isinf(obj):
        return None
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------

def _param_stability_label(stability_df) -> str:
    if stability_df is None or len(stability_df) == 0:
        return "unknown"
    avg_cv = stability_df["cv"].mean()
    if avg_cv < 0.20:
        return "stable"
    elif avg_cv <= 0.40:
        return "mixed"
    else:
        return "unstable"


def _long_short_balance(direction_split) -> str:
    if direction_split is None:
        return "unknown"
    lo = direction_split.get("long_only")
    so = direction_split.get("short_only")

    lo_sharpe = lo["sharpe_ratio"] if lo else None
    so_sharpe = so["sharpe_ratio"] if so else None

    if lo is None or (lo.get("num_trades", 0) == 0):
        return "short-only"
    if so is None or (so.get("num_trades", 0) == 0):
        return "long-only"

    lo_s = lo_sharpe or 0.0
    so_s = so_sharpe or 0.0

    if lo_s > 0.3 and so_s > 0.3:
        return "balanced"
    elif lo_s > 0.5 and so_s < 0.3:
        return "long-dominated"
    elif so_s > 0.5 and lo_s < 0.3:
        return "short-dominated"
    return "balanced"


def _build_verdict(
    oos_sharpe,
    overfit_ratio,
    avg_oos_return,
    pct_folds_profitable,
    max_drawdown,
    num_trades,
    stability_df,
    direction_split,
) -> dict:
    """Compute the verdict block following the documented rules exactly."""

    warnings_list = []

    # Tradeable flags
    tradeable = (
        oos_sharpe is not None and oos_sharpe >= 0.5
        and overfit_ratio is not None and overfit_ratio >= 0.5
        and avg_oos_return is not None and avg_oos_return > 0
        and pct_folds_profitable is not None and pct_folds_profitable >= 0.5
    )

    if oos_sharpe is None or oos_sharpe < 0.5:
        warnings_list.append("Out-of-sample Sharpe ratio below minimum threshold")

    if overfit_ratio is None or overfit_ratio < 0.5:
        warnings_list.append(
            "Significant performance degradation from in-sample to out-of-sample "
            "-- possible overfitting"
        )

    param_stab = _param_stability_label(stability_df)
    if param_stab == "unstable":
        warnings_list.append(
            "Parameters unstable across walk-forward folds -- strategy may be curve-fitted"
        )

    if pct_folds_profitable is not None and pct_folds_profitable < 0.5:
        warnings_list.append("Less than half of walk-forward folds were profitable")

    if max_drawdown is not None and max_drawdown < -0.30:
        warnings_list.append("Maximum drawdown exceeds 30%")

    if num_trades is not None and num_trades < 30:
        warnings_list.append("Insufficient trade count for statistical confidence")

    ls_balance = _long_short_balance(direction_split)

    # Plain-English conclusion
    if tradeable:
        conclusion = (
            f"Strategy passes all quality gates: "
            f"OOS Sharpe {oos_sharpe:.2f}, "
            f"overfit ratio {overfit_ratio:.2f}, "
            f"parameters {param_stab}."
        )
    else:
        conclusion = (
            "Strategy does not meet minimum quality thresholds. "
            f"Issues: {'; '.join(warnings_list) if warnings_list else 'see metrics'}."
        )

    return {
        "tradeable":          tradeable,
        "oos_sharpe":         oos_sharpe,
        "overfit_ratio":      overfit_ratio,
        "param_stability":    param_stab,
        "long_short_balance": ls_balance,
        "conclusion":         conclusion,
        "warnings":           warnings_list,
    }


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

def build_report(
    research_results: dict,
    strategy_name:    str,
    strategy_description: str,
    ticker:           str,
    timeframe:        str,
    cost:             float,
) -> dict:
    """
    Build a structured, JSON-serialisable research report.

    Parameters
    ----------
    research_results : dict
        Output of Research.full_report(). Expected keys:
        wf_results, full_metrics, sensitivity, perturbation, cost_stress.
    strategy_name : str
        Human-readable strategy name, e.g. "MA Crossover Trend".
    strategy_description : str
        One-sentence description of what the strategy does.
    ticker : str
        Primary ticker, e.g. "EURUSD".
    timeframe : str
        Primary timeframe, e.g. "1H".
    cost : float
        Round-trip cost fraction used in the research run.

    Returns
    -------
    dict
        Fully structured report dict. Every key is always present (None if
        the data is unavailable). Safe to pass to save_report().

    Example
    -------
    >>> report = build_report(r.full_report(), "MA Cross", "...", "SPY", "1D", 0.001)
    >>> save_report(report)
    """
    wf           = research_results.get("wf_results", {})
    full_metrics = research_results.get("full_metrics")
    sensitivity  = research_results.get("sensitivity")
    perturbation = research_results.get("perturbation")
    cost_stress  = research_results.get("cost_stress")

    oos_metrics  = wf.get("oos_metrics")
    results_df   = wf.get("results_df")
    stability_df = wf.get("stability_df")
    fold_records = wf.get("fold_records", [])
    cp           = wf.get("consensus_params", {})

    # -- metadata --------------------------------------------------------------
    primary_df   = None
    data_start   = None
    data_end     = None
    total_bars   = None

    if full_metrics is not None and full_metrics.get("equity_curve") is not None:
        eq = full_metrics["equity_curve"]
        data_start = str(eq.index[0])
        data_end   = str(eq.index[-1])
        total_bars = len(eq)

    metadata = {
        "strategy_name":        strategy_name,
        "strategy_description": strategy_description,
        "ticker":               ticker,
        "timeframe":            timeframe,
        "cost":                 cost,
        "run_date":             datetime.now(timezone.utc).isoformat(),
        "data_start":           data_start,
        "data_end":             data_end,
        "total_bars":           total_bars,
        "num_folds":            len(fold_records),
    }

    # -- full period -----------------------------------------------------------
    ds = full_metrics.get("direction_split") if full_metrics else None

    def _slim_metrics(m):
        if m is None:
            return None
        return {
            "total_return":       m.get("total_return"),
            "sharpe_ratio":       m.get("sharpe_ratio"),
            "max_drawdown":       m.get("max_drawdown"),
            "win_rate":           m.get("win_rate"),
            "num_trades":         m.get("num_trades"),
            "avg_win_loss_ratio": m.get("avg_win_loss_ratio"),
            "profit_factor":      m.get("profit_factor"),
            "calmar_ratio":       m.get("calmar_ratio"),
        }

    full_period = {
        "combined":   _slim_metrics(ds.get("combined") if ds else full_metrics),
        "long_only":  _slim_metrics(ds.get("long_only") if ds else None),
        "short_only": _slim_metrics(ds.get("short_only") if ds else None),
    }

    # -- walk-forward ----------------------------------------------------------
    avg_oos_sharpe    = None
    avg_oos_return    = None
    avg_oos_drawdown  = None
    pct_folds_prof    = None
    overfit_ratio     = None
    oos_sharpe_val    = None

    if results_df is not None and len(results_df) > 0:
        valid = results_df[
            results_df["test_return"].notna() & results_df["train_return"].notna()
        ]
        if len(valid) > 0:
            avg_oos_sharpe   = float(valid["test_sharpe"].mean())
            avg_oos_return   = float(valid["test_return"].mean())
            avg_oos_drawdown = float(valid["test_drawdown"].mean())
            pct_folds_prof   = float((valid["test_return"] > 0).sum() / len(valid))
            if valid["train_sharpe"].mean() != 0:
                overfit_ratio = float(
                    valid["test_sharpe"].mean() / valid["train_sharpe"].mean()
                )

    if oos_metrics is not None:
        oos_sharpe_val = oos_metrics.get("sharpe_ratio")

    oos_sharpe_for_verdict = oos_sharpe_val if oos_sharpe_val is not None else avg_oos_sharpe

    walk_forward_block = {
        "fold_records":         fold_records,
        "oos_metrics":          _slim_metrics(oos_metrics),
        "avg_oos_sharpe":       avg_oos_sharpe,
        "avg_oos_return":       avg_oos_return,
        "avg_oos_drawdown":     avg_oos_drawdown,
        "pct_folds_profitable": pct_folds_prof,
        "overfit_ratio":        overfit_ratio,
        "consensus_params":     cp,
        "stability_df": (
            stability_df.to_dict(orient="records") if stability_df is not None else []
        ),
    }

    # -- robustness ------------------------------------------------------------
    def _sens_to_json(sens):
        if not sens:
            return None
        return {k: v.to_dict(orient="records") for k, v in sens.items()}

    robustness = {
        "sensitivity":  _sens_to_json(sensitivity),
        "perturbation": (
            perturbation.to_dict(orient="records")
            if isinstance(perturbation, pd.DataFrame) and len(perturbation) > 0
            else None
        ),
        "cost_stress": (
            cost_stress.to_dict(orient="records")
            if isinstance(cost_stress, pd.DataFrame) and len(cost_stress) > 0
            else None
        ),
    }

    # -- verdict ---------------------------------------------------------------
    max_dd_for_verdict = (
        full_period["combined"]["max_drawdown"]
        if full_period["combined"] else None
    )
    num_trades_for_verdict = (
        full_period["combined"]["num_trades"]
        if full_period["combined"] else None
    )

    verdict = _build_verdict(
        oos_sharpe          = oos_sharpe_for_verdict,
        overfit_ratio       = overfit_ratio,
        avg_oos_return      = avg_oos_return,
        pct_folds_profitable= pct_folds_prof,
        max_drawdown        = max_dd_for_verdict,
        num_trades          = num_trades_for_verdict,
        stability_df        = stability_df,
        direction_split     = ds,
    )

    report = {
        "metadata":     metadata,
        "full_period":  full_period,
        "walk_forward": walk_forward_block,
        "robustness":   robustness,
        "verdict":      verdict,
    }

    return _to_python(report)


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

def save_report(report: dict, results_dir: str = "results") -> str:
    """
    Serialise the report dict to a timestamped JSON file.

    Parameters
    ----------
    report : dict
        Output of build_report(). Must be JSON-serialisable (all non-native
        types already converted by build_report).
    results_dir : str
        Directory where the file is saved. Created if it does not exist.
        Default "results".

    Returns
    -------
    str
        Absolute path of the saved JSON file.

    Example
    -------
    >>> path = save_report(report, results_dir="results")
    >>> print(path)
    results/MA_Cross_EURUSD_1H_20250408T143012.json
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta      = report.get("metadata", {})
    name      = str(meta.get("strategy_name", "strategy")).replace(" ", "_")
    ticker    = str(meta.get("ticker", "unknown"))
    timeframe = str(meta.get("timeframe", ""))
    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    filename  = f"{name}_{ticker}_{timeframe}_{ts}.json"
    filepath  = out_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"OK Report saved -> {filepath}")
    return str(filepath)
