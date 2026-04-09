"""
infrastructure/backtester/research.py
---------------------------------------
Research class: wraps the walk-forward engine, sensitivity analysis,
perturbation testing, and cost stress testing into a single callable object.

Ported from Epsilon Fund / infrastructure / walkforward / wf_engine.py.
All logic (fold building, Optuna setup, scoring, OOS stitching, plateau
analysis, perturbation test, cost stress test) is preserved exactly.
The only structural change is that everything is a method on Research instead
of a standalone function, and the engine call uses BacktestContext.
"""

from __future__ import annotations

import os
import warnings
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import optuna

from .data_loader import BacktestContext, get_context
from .engine      import backtest

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers (ported verbatim from wf_engine.py)
# ---------------------------------------------------------------------------

def _calmar(metrics):
    if metrics is None or metrics["max_drawdown"] == 0:
        return 0.0
    return metrics["total_return"] / abs(metrics["max_drawdown"])


def _fmt(val, pct=False, dp=2):
    if val is None:
        return "N/A"
    if pct:
        return f"{val*100:.{dp}f}%"
    return f"{val:.{dp}f}"


def _metrics_to_row(m, prefix):
    if m is None:
        keys = ["return", "sharpe", "drawdown", "calmar", "trades", "winrate", "profit_factor"]
        return {f"{prefix}_{k}": None for k in keys}
    return {
        f"{prefix}_return":        m["total_return"],
        f"{prefix}_sharpe":        m["sharpe_ratio"],
        f"{prefix}_drawdown":      m["max_drawdown"],
        f"{prefix}_calmar":        _calmar(m),
        f"{prefix}_trades":        m["num_trades"],
        f"{prefix}_winrate":       m["win_rate"],
        f"{prefix}_profit_factor": m["profit_factor"],
    }


def _build_stability_df(all_best_params, param_defs, fixed_params):
    rows = []
    for name in param_defs:
        vals   = [p[name] for p in all_best_params]
        median = float(np.median(vals))
        std    = float(np.std(vals))
        cv     = std / abs(median) if median != 0 else 999.0
        rows.append({
            "param":  name,
            "median": median,
            "std":    std,
            "cv":     cv,
            "fixed":  name in fixed_params,
            "stable": cv < 0.15,
        })
    return pd.DataFrame(rows)


def _consensus_params(all_best_params, param_defs):
    int_params = {k for k, (t, _, _) in param_defs.items() if t == "int"}
    result = {}
    for name in param_defs:
        vals = [p[name] for p in all_best_params]
        med  = np.median(vals)
        result[name] = int(round(med)) if name in int_params else round(float(med), 4)
    return result


# ---------------------------------------------------------------------------
# Research class
# ---------------------------------------------------------------------------

class Research:
    """
    Single entry point for systematic strategy research.

    Wraps walk-forward optimisation, sensitivity analysis, perturbation
    testing, and cost stress testing into one object.

    Parameters
    ----------
    strategy_fn : callable
        The generate_signals function from a strategy. Signature:
        strategy_fn(context: BacktestContext, params: dict) -> pd.DataFrame
    param_space : dict
        Parameter search space: {name: ('int'|'float', lo, hi)}.
    primary_ticker : str
        Ticker symbol, e.g. "EURUSD".
    primary_tf : str
        Primary timeframe, e.g. "1H".
    aux_timeframes : list of str, optional
        Additional timeframes, e.g. ["4H"].
    cost : float
        Round-trip trading cost fraction. Default 0.001.
    start_date : str, optional
        ISO date string to filter data start.
    end_date : str, optional
        ISO date string to filter data end.

    Example
    -------
    >>> r = Research(my_strategy.generate_signals, my_strategy.PARAM_SPACE,
    ...              "EURUSD", "1H", aux_timeframes=["4H"])
    >>> results = r.walk_forward(train_bars=504, test_bars=252, n_trials=200)
    """

    def __init__(
        self,
        strategy_fn:    callable,
        param_space:    dict,
        primary_ticker: str,
        primary_tf:     str,
        aux_timeframes: list[str] | None = None,
        cost:           float            = 0.001,
        start_date:     str | None       = None,
        end_date:       str | None       = None,
    ):
        self.strategy_fn    = strategy_fn
        self.param_space    = param_space
        self.primary_ticker = primary_ticker
        self.primary_tf     = primary_tf
        self.aux_timeframes = aux_timeframes or []
        self.cost           = cost
        self.start_date     = start_date
        self.end_date       = end_date

        # Load full context once -- sliced per fold during walk-forward
        self._full_context = get_context(
            primary_ticker  = primary_ticker,
            primary_tf      = primary_tf,
            aux_timeframes  = self.aux_timeframes,
            start_date      = start_date,
            end_date        = end_date,
        )

    # -----------------------------------------------------------------------
    # Default score / reject (preserved from Epsilon Fund)
    # -----------------------------------------------------------------------

    def _default_score(self, metrics) -> float:
        SHARPE_MAX = 2.5
        CALMAR_MAX = 60.0
        RETURN_MAX = 15.0
        calmar = _calmar(metrics)
        s = np.clip(metrics["sharpe_ratio"] / SHARPE_MAX, 0, 1)
        c = np.clip(calmar               / CALMAR_MAX, 0, 1)
        r = np.clip(metrics["total_return"] / RETURN_MAX, 0, 1)
        return 0.50 * s + 0.30 * c + 0.20 * r

    def _default_reject(self, metrics) -> bool:
        if metrics is None:                    return True
        if metrics["num_trades"]   < 7:        return True
        if metrics["win_rate"]     < 0.35:     return True
        if metrics["max_drawdown"] < -0.80:    return True
        if metrics["profit_factor"] < 0.8:     return True
        return False

    # -----------------------------------------------------------------------
    # Internal: run one backtest safely
    # -----------------------------------------------------------------------

    def _run(self, context: BacktestContext, params: dict):
        try:
            return backtest(context, self.strategy_fn, params,
                            cost=self.cost, show_plot=False)
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # Internal: build Optuna objective for one fold
    # -----------------------------------------------------------------------

    def _make_objective(self, fold_context, fixed_params, score_fn, reject_fn):
        def objective(trial):
            params = {}
            for name, (dtype, lo, hi) in self.param_space.items():
                if name in fixed_params:
                    params[name] = fixed_params[name]
                elif dtype == "int":
                    params[name] = trial.suggest_int(name, lo, hi)
                else:
                    params[name] = trial.suggest_float(name, lo, hi)

            m = self._run(fold_context, params)

            if reject_fn(m):
                return -999.0

            trial.set_user_attr("sharpe",        m["sharpe_ratio"])
            trial.set_user_attr("calmar",        _calmar(m))
            trial.set_user_attr("total_return",  m["total_return"])
            trial.set_user_attr("max_drawdown",  m["max_drawdown"])
            trial.set_user_attr("num_trades",    m["num_trades"])
            trial.set_user_attr("win_rate",      m["win_rate"])
            trial.set_user_attr("profit_factor", m["profit_factor"])

            return score_fn(m)

        return objective

    # -----------------------------------------------------------------------
    # simple_backtest
    # -----------------------------------------------------------------------

    def simple_backtest(self, params: dict | None = None) -> dict:
        """
        Run a single backtest on the full available history.

        Parameters
        ----------
        params : dict, optional
            If None, uses midpoint values from param_space for every parameter.

        Returns
        -------
        dict
            Full results dict from engine.backtest().

        Example
        -------
        >>> r = Research(strategy_fn, PARAM_SPACE, "SPY", "1D")
        >>> results = r.simple_backtest()
        >>> print(results['sharpe_ratio'])
        """
        if params is None:
            int_params = {k for k, (t, _, _) in self.param_space.items() if t == "int"}
            params = {}
            for name, (dtype, lo, hi) in self.param_space.items():
                mid = (lo + hi) / 2
                params[name] = int(round(mid)) if dtype == "int" else round(float(mid), 4)

        metrics = backtest(
            self._full_context, self.strategy_fn, params,
            cost=self.cost, show_plot=False,
        )

        print(f"\n{'-'*50}")
        print(f"Simple Backtest: {self.primary_ticker} {self.primary_tf}")
        print(f"{'-'*50}")
        print(f"  Total Return:   {_fmt(metrics['total_return'], pct=True)}")
        print(f"  Sharpe Ratio:   {_fmt(metrics['sharpe_ratio'])}")
        print(f"  Max Drawdown:   {_fmt(metrics['max_drawdown'], pct=True)}")
        print(f"  Calmar Ratio:   {_fmt(metrics['calmar_ratio'])}")
        print(f"  Win Rate:       {_fmt(metrics['win_rate'], pct=True)}")
        print(f"  Num Trades:     {metrics['num_trades']}")
        print(f"  Profit Factor:  {_fmt(metrics['profit_factor'])}")
        print(f"{'-'*50}")

        return metrics

    # -----------------------------------------------------------------------
    # walk_forward
    # -----------------------------------------------------------------------

    def walk_forward(
        self,
        train_bars:   int            = 504,
        test_bars:    int            = 252,
        burnin_bars:  int            = 60,
        n_trials:     int            = 200,
        seed_base:    int            = 42,
        fixed_params: dict | None    = None,
        score_fn:     callable | None = None,
        reject_fn:    callable | None = None,
        save_csv:     str | None      = None,
    ) -> dict:
        """
        Rolling walk-forward optimisation with Optuna TPE sampler.

        Each fold:
        1. Optimises parameters on a training window.
        2. Evaluates best params on an out-of-sample test window.
        3. The context for each fold is a slice of the full loaded data
           (no re-reading from disk).

        Parameters
        ----------
        train_bars : int
            Length of training window in bars. Default 504 (~2 years daily).
        test_bars : int
            Length of test window in bars. Default 252 (~1 year daily).
        burnin_bars : int
            Bars prepended to the test slice for indicator warmup (trimmed
            before evaluation). Default 60.
        n_trials : int
            Optuna trials per fold. Default 200.
        seed_base : int
            Fold i uses seed seed_base + i for reproducibility.
        fixed_params : dict, optional
            Parameters anchored to a fixed value across all folds.
        score_fn : callable, optional
            Custom scoring function: fn(metrics) -> float.
            Defaults to Sharpe 50% + Calmar 30% + Return 20%.
        reject_fn : callable, optional
            Custom rejection filter: fn(metrics) -> bool.
            Defaults to min 7 trades, 35% win rate, -80% drawdown, 0.8 PF.
        save_csv : str, optional
            If provided, saves fold results to this CSV path.

        Returns
        -------
        dict
            fold_records, results_df, all_best_params, consensus_params,
            stability_df, oos_combined_df, oos_metrics.
        """
        if fixed_params is None:
            fixed_params = {}
        if score_fn is None:
            score_fn = self._default_score
        if reject_fn is None:
            reject_fn = self._default_reject

        primary = self._full_context.primary()

        # Build folds
        folds = []
        start = 0
        while start + train_bars + test_bars <= len(primary):
            folds.append({
                "train_start": primary.index[start],
                "train_end":   primary.index[start + train_bars - 1],
                "test_start":  primary.index[start + train_bars],
                "test_end":    primary.index[start + train_bars + test_bars - 1],
                "burnin_start_idx": start + train_bars - burnin_bars,
                "i_train_start": start,
                "i_train_end":   start + train_bars,
                "i_test_end":    start + train_bars + test_bars,
                "burnin_bars":   burnin_bars,
            })
            start += test_bars

        print(f"Walk-forward: {len(folds)} fold(s)  "
              f"train={train_bars}  test={test_bars}  burnin={burnin_bars}  "
              f"trials={n_trials}")
        for i, f in enumerate(folds):
            print(f"  Fold {i+1}: train {f['train_start'].date()} -> {f['train_end'].date()} "
                  f"| test {f['test_start'].date()} -> {f['test_end'].date()}")

        free = [k for k in self.param_space if k not in fixed_params]
        if fixed_params:
            print(f"\nFixed ({len(fixed_params)}): {list(fixed_params.keys())}")
            print(f"Free  ({len(free)}): {free}")

        fold_records    = []
        all_best_params = []
        oos_slices      = []

        for i, fold in enumerate(folds):
            print(f"\n{'-'*60}")
            print(f"Fold {i+1}/{len(folds)}  "
                  f"train: {fold['train_start'].date()} -> {fold['train_end'].date()}  "
                  f"test: {fold['test_start'].date()} -> {fold['test_end'].date()}")

            # Build train context slice
            train_ctx = self._full_context.slice(
                start=fold["train_start"],
                end=fold["train_end"],
            )

            # Optimise on training window
            study = optuna.create_study(
                direction  = "maximize",
                study_name = f"wf_fold_{i+1}",
                sampler    = optuna.samplers.TPESampler(seed=seed_base + i),
            )
            study.optimize(
                self._make_objective(train_ctx, fixed_params, score_fn, reject_fn),
                n_trials          = n_trials,
                show_progress_bar = True,
            )

            best_params = {**fixed_params, **study.best_params}
            all_best_params.append(best_params)

            # IS performance
            train_m = self._run(train_ctx, best_params)

            # OOS performance (with burnin prepended, then trim)
            burnin_start = primary.index[fold["burnin_start_idx"]]
            test_ctx_full = self._full_context.slice(
                start=burnin_start,
                end=fold["test_end"],
            )

            test_m  = None
            oos_df  = None
            try:
                test_m = self._run(test_ctx_full, best_params)
                if test_m is not None:
                    # Trim burnin from OOS equity (use test_start onwards)
                    oos_primary = test_ctx_full.primary()
                    oos_df = oos_primary[oos_primary.index >= fold["test_start"]].copy()
                    oos_slices.append(oos_df)
            except Exception as e:
                logger.warning(f"Fold {i+1} OOS failed: {e}")

            record = {
                "fold":         i + 1,
                "train_start":  str(fold["train_start"].date()),
                "train_end":    str(fold["train_end"].date()),
                "test_start":   str(fold["test_start"].date()),
                "test_end":     str(fold["test_end"].date()),
                "optuna_score": study.best_value,
                **_metrics_to_row(train_m, "train"),
                **_metrics_to_row(test_m,  "test"),
                **{f"param_{k}": v for k, v in best_params.items()},
            }
            fold_records.append(record)

            print(f"\n  IS  -> Sharpe: {_fmt(record['train_sharpe'])}  "
                  f"Return: {_fmt(record['train_return'], pct=True)}  "
                  f"DD: {_fmt(record['train_drawdown'], pct=True)}  "
                  f"Trades: {record['train_trades']}")
            print(f"  OOS -> Sharpe: {_fmt(record['test_sharpe'])}  "
                  f"Return: {_fmt(record['test_return'], pct=True)}  "
                  f"DD: {_fmt(record['test_drawdown'], pct=True)}  "
                  f"Trades: {record['test_trades']}")
            print(f"\n  Best params: {best_params}")

        results_df   = pd.DataFrame(fold_records)
        cp           = _consensus_params(all_best_params, self.param_space)
        stability_df = _build_stability_df(all_best_params, self.param_space, fixed_params)

        # Summary
        print(f"\n{'='*60}")
        print("WALK-FORWARD SUMMARY")
        print(f"{'='*60}")
        valid = results_df[
            results_df["test_return"].notna() & results_df["train_return"].notna()
        ]
        if len(valid) == 0:
            print("WARNING: no valid folds -- loosen reject_fn filters or check strategy output")
        else:
            print(f"\nOut-of-sample across {len(valid)} fold(s):")
            print(f"  Avg Sharpe:       {valid['test_sharpe'].mean():.2f}")
            print(f"  Avg Return:       {valid['test_return'].mean()*100:.1f}%")
            print(f"  Avg Max Drawdown: {valid['test_drawdown'].mean()*100:.1f}%")
            print(f"  Folds profitable: {(valid['test_return'] > 0).sum()}/{len(valid)}")
            if valid["train_sharpe"].mean() != 0:
                deg   = valid["test_sharpe"].mean() / valid["train_sharpe"].mean()
                label = "good" if deg > 0.70 else ("acceptable" if deg > 0.50 else "poor")
                print(f"  Sharpe OOS/IS:    {deg:.2f}  ({label})")

        if save_csv:
            os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
            results_df.to_csv(save_csv, index=False)
            print(f"\nOK Fold results saved -> {save_csv}")

        # Combined OOS backtest
        oos_metrics  = None
        oos_combined = None

        if oos_slices:
            # Re-run the combined OOS window as a single backtest for proper metrics
            oos_start = pd.to_datetime(fold_records[0]["test_start"])
            oos_end   = pd.to_datetime(fold_records[-1]["test_end"])
            oos_ctx   = self._full_context.slice(start=oos_start, end=oos_end)
            oos_metrics = self._run(oos_ctx, cp)

            oos_combined = oos_ctx.primary().copy()

            if oos_metrics:
                print(f"\n{'-'*60}")
                print(f"COMBINED OOS: {oos_start.date()} -> {oos_end.date()}")
                print(f"  Return:        {oos_metrics['total_return']*100:.2f}%")
                print(f"  Sharpe:        {oos_metrics['sharpe_ratio']:.2f}")
                print(f"  Max Drawdown:  {oos_metrics['max_drawdown']*100:.2f}%")
                print(f"  Profit Factor: {oos_metrics['profit_factor']:.2f}")
                print(f"  Win Rate:      {oos_metrics['win_rate']*100:.2f}%")
                print(f"  Num Trades:    {oos_metrics['num_trades']}")

        return {
            "fold_records":     fold_records,
            "results_df":       results_df,
            "all_best_params":  all_best_params,
            "consensus_params": cp,
            "stability_df":     stability_df,
            "oos_combined_df":  oos_combined,
            "oos_metrics":      oos_metrics,
        }

    # -----------------------------------------------------------------------
    # sensitivity (plateau analysis)
    # -----------------------------------------------------------------------

    def sensitivity(
        self,
        base_params:  dict,
        n_steps:      int           = 20,
        fixed_params: dict | None   = None,
        score_fn:     callable | None = None,
        reject_fn:    callable | None = None,
    ) -> dict:
        """
        1-D sensitivity sweeps: vary each free parameter while holding others fixed.

        Ported from plateau_analysis() in Epsilon Fund wf_engine.py.

        Parameters
        ----------
        base_params : dict
            Parameter values to hold fixed while sweeping each one.
            Typically consensus_params from walk_forward().
        n_steps : int
            Number of points per parameter sweep. Default 20.
        fixed_params : dict, optional
            Parameters to exclude from sweeping.
        score_fn : callable, optional
        reject_fn : callable, optional

        Returns
        -------
        dict
            {param_name: pd.DataFrame with columns [value, score, sharpe,
            calmar, return, drawdown, trades]}
        """
        if fixed_params is None:
            fixed_params = {}
        if score_fn is None:
            score_fn = self._default_score
        if reject_fn is None:
            reject_fn = self._default_reject

        free_params   = [k for k in self.param_space if k not in fixed_params]
        sweep_results = {}

        for name in free_params:
            dtype, lo, hi = self.param_space[name]
            if dtype == "int":
                values = np.unique(np.linspace(lo, hi, n_steps).astype(int))
            else:
                values = np.linspace(lo, hi, n_steps)

            rows = []
            for val in values:
                trial_params = {
                    **base_params,
                    name: int(val) if dtype == "int" else float(val),
                }
                m = self._run(self._full_context, trial_params)
                if m is None or reject_fn(m):
                    rows.append({"value": val, "score": None})
                    continue
                rows.append({
                    "value":    val,
                    "score":    score_fn(m),
                    "sharpe":   m["sharpe_ratio"],
                    "calmar":   _calmar(m),
                    "return":   m["total_return"],
                    "drawdown": m["max_drawdown"],
                    "trades":   m["num_trades"],
                })
            sweep_results[name] = pd.DataFrame(rows)

        return sweep_results

    # -----------------------------------------------------------------------
    # perturbation
    # -----------------------------------------------------------------------

    def perturbation(
        self,
        base_params:  dict,
        pct_offsets:  tuple      = (0.05, 0.10, 0.20),
        n_samples:    int        = 50,
        seed:         int        = 42,
        fixed_params: dict | None = None,
        score_fn:     callable | None = None,
        reject_fn:    callable | None = None,
    ) -> pd.DataFrame:
        """
        Randomly perturb ALL free parameters simultaneously and measure score degradation.

        Ported from perturbation_test() in Epsilon Fund wf_engine.py.

        Parameters
        ----------
        base_params : dict
            Centre-point for perturbation (typically consensus_params).
        pct_offsets : tuple of float
            Fractional perturbation levels, e.g. (0.05, 0.10, 0.20).
        n_samples : int
            Number of random samples per offset level. Default 50.
        seed : int
            Random seed for reproducibility.
        fixed_params : dict, optional
        score_fn : callable, optional
        reject_fn : callable, optional

        Returns
        -------
        pd.DataFrame
            Columns: offset_pct, n_valid, mean_score, median_score, std_score,
            min_score, degradation.
        """
        if fixed_params is None:
            fixed_params = {}
        if score_fn is None:
            score_fn = self._default_score
        if reject_fn is None:
            reject_fn = self._default_reject

        rng  = np.random.default_rng(seed)
        free = [k for k in self.param_space if k not in fixed_params]

        # Baseline score
        base_m = self._run(self._full_context, base_params)
        if base_m is None:
            print("ERROR: base_params failed to produce a valid backtest")
            return pd.DataFrame()
        base_score = score_fn(base_m)

        records = []
        for pct in pct_offsets:
            scores = []
            for _ in range(n_samples):
                trial_params = dict(base_params)
                for name in free:
                    dtype, lo, hi = self.param_space[name]
                    rng_width = (hi - lo) * pct
                    noise     = rng.uniform(-rng_width, rng_width)
                    new_val   = np.clip(base_params[name] + noise, lo, hi)
                    trial_params[name] = int(round(new_val)) if dtype == "int" else float(new_val)

                m = self._run(self._full_context, trial_params)
                if m is None or reject_fn(m):
                    continue
                scores.append(score_fn(m))

            if scores:
                mean_s = float(np.mean(scores))
                records.append({
                    "offset_pct":    pct,
                    "n_valid":       len(scores),
                    "mean_score":    mean_s,
                    "median_score":  float(np.median(scores)),
                    "std_score":     float(np.std(scores)),
                    "min_score":     float(np.min(scores)),
                    "degradation":   (base_score - mean_s) / base_score if base_score else 0.0,
                })

        result_df = pd.DataFrame(records)

        print(f"\n{'='*75}")
        print("PERTURBATION TEST -- NEIGHBOURHOOD ROBUSTNESS")
        print(f"{'='*75}")
        print(f"Base score: {base_score:.4f}")
        print(f"{'Offset':>8} {'N valid':>8} {'Mean':>8} {'Median':>8} "
              f"{'Std':>8} {'Min':>8} {'Degradation':>12}")
        print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
        for _, row in result_df.iterrows():
            print(f"{row['offset_pct']*100:>7.0f}% {row['n_valid']:>8.0f} "
                  f"{row['mean_score']:>8.4f} {row['median_score']:>8.4f} "
                  f"{row['std_score']:>8.4f} {row['min_score']:>8.4f} "
                  f"{row['degradation']*100:>10.1f}%")

        return result_df

    # -----------------------------------------------------------------------
    # cost_stress
    # -----------------------------------------------------------------------

    def cost_stress(
        self,
        oos_combined_df:   pd.DataFrame,
        cost_multipliers:  tuple = (1.0, 1.5, 2.0, 3.0),
    ) -> pd.DataFrame:
        """
        Re-run the OOS period at escalating transaction costs.

        Ported from cost_stress_test() in Epsilon Fund wf_engine.py.

        Parameters
        ----------
        oos_combined_df : pd.DataFrame
            The combined OOS primary DataFrame (from walk_forward result).
        cost_multipliers : tuple of float
            Cost scaling factors, e.g. (1.0, 1.5, 2.0, 3.0).

        Returns
        -------
        pd.DataFrame
            One row per cost level with performance metrics.
        """
        # We need to run as a BacktestContext
        from .data_loader import BacktestContext as BTC

        oos_start = oos_combined_df.index[0]
        oos_end   = oos_combined_df.index[-1]
        oos_ctx   = self._full_context.slice(start=oos_start, end=oos_end)

        # Use consensus params for this run
        records = []
        for mult in cost_multipliers:
            c = self.cost * mult
            try:
                m = backtest(oos_ctx, self.strategy_fn,
                             params={}, cost=c, show_plot=False)
            except Exception:
                continue
            if m:
                records.append({
                    "cost":          c,
                    "cost_mult":     mult,
                    "sharpe":        m["sharpe_ratio"],
                    "total_return":  m["total_return"],
                    "max_drawdown":  m["max_drawdown"],
                    "calmar":        _calmar(m),
                    "profit_factor": m["profit_factor"],
                    "num_trades":    m["num_trades"],
                })

        result_df = pd.DataFrame(records)

        print(f"\n{'='*75}")
        print("TRANSACTION COST STRESS TEST")
        print(f"{'='*75}")
        for _, r in result_df.iterrows():
            print(f"  {r['cost_mult']:.1f}x  Sharpe: {r['sharpe']:.2f}  "
                  f"Return: {r['total_return']*100:.2f}%  "
                  f"DD: {r['max_drawdown']*100:.2f}%  "
                  f"PF: {r['profit_factor']:.2f}")

        return result_df

    # -----------------------------------------------------------------------
    # full_report
    # -----------------------------------------------------------------------

    def full_report(
        self,
        train_bars:       int  = 504,
        test_bars:        int  = 252,
        n_trials:         int  = 200,
        skip_robustness:  bool = False,
    ) -> dict:
        """
        Run the complete research pipeline and return all results in one dict.

        Executes in order:
        1. walk_forward
        2. sensitivity on consensus params
        3. perturbation on consensus params
        4. cost_stress on combined OOS

        Parameters
        ----------
        train_bars : int
        test_bars  : int
        n_trials   : int

        Returns
        -------
        dict
            Keys: wf_results, sensitivity, perturbation, cost_stress.
            This is the dict consumed by report.build_report().

        Example
        -------
        >>> r = Research(strategy_fn, PARAM_SPACE, "EURUSD", "1H")
        >>> report_data = r.full_report()
        >>> from infrastructure.backtester.report import build_report, save_report
        >>> report = build_report(report_data, "MyStrategy", "...", "EURUSD", "1H", 0.001)
        >>> save_report(report)
        """
        print(f"\n{'='*60}")
        print(f"FULL RESEARCH REPORT")
        print(f"  {self.primary_ticker} @ {self.primary_tf}")
        print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*60}\n")

        # 1. Walk-forward
        wf = self.walk_forward(
            train_bars=train_bars,
            test_bars=test_bars,
            n_trials=n_trials,
        )

        cp = wf["consensus_params"]

        sens   = None
        pert   = None
        cost_st = pd.DataFrame()

        if not skip_robustness:
            # 2. Sensitivity
            print("\n--- Sensitivity Analysis ---")
            sens = self.sensitivity(cp)

            # 3. Perturbation
            print("\n--- Perturbation Test ---")
            pert = self.perturbation(cp)

            # 4. Cost stress
            if wf["oos_combined_df"] is not None:
                print("\n--- Cost Stress Test ---")
                cost_st = self.cost_stress(wf["oos_combined_df"])
        else:
            print("\n[Robustness analysis skipped]")

        # Full-period simple backtest with consensus params
        print("\n--- Full-Period Backtest (consensus params) ---")
        full_metrics = self.simple_backtest(params=cp)

        print(f"\n{'='*60}")
        print("FULL REPORT COMPLETE")
        print(f"{'='*60}\n")

        return {
            "wf_results":   wf,
            "full_metrics": full_metrics,
            "sensitivity":  sens,
            "perturbation": pert,
            "cost_stress":  cost_st,
        }
