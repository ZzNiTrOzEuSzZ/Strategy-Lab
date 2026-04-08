"""
infrastructure/backtester/engine.py
--------------------------------------
Core backtesting engine for StratLab Layer 2.

Ported from Epsilon Fund / infrastructure / backtester / engine.py.
The returns calculation, cost application, and metrics call are preserved
exactly. Changes vs the original:
  - Accepts BacktestContext + strategy_fn instead of a pre-built DataFrame
  - Adds a bar-by-bar stop/target monitoring loop
  - Adds direction_split to the results dict
  - Removed build_pair_df (moved to strategies/utils)
"""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np

from .data_loader import BacktestContext
from .metrics     import calculate_all_metrics, calculate_direction_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main backtest function
# ---------------------------------------------------------------------------

def backtest(
    context:        BacktestContext,
    strategy_fn:    callable,
    params:         dict,
    cost:           float               = 0.001,
    show_plot:      bool                = False,
    save_html:      str | None          = None,
    benchmark_data: pd.DataFrame | None = None,
) -> dict:
    """
    Run a full backtest of ``strategy_fn`` on the data in ``context``.

    Execution order
    ---------------
    1. Call strategy_fn(context, params) to get the signal DataFrame.
    2. Validate the returned DataFrame has a 'position' column in [-1, 1].
    3. If 'stop_loss' column is present and non-zero, run the bar-by-bar
       stop/target monitoring loop (the only loop in the engine).
    4. Apply position sizing: uses 'position_size' col if present, else
       abs(position) clipped to [0, 1].
    5. Shift effective position by 1 bar (fills at prior close — no lookahead).
    6. Compute returns, subtract costs, run metrics.
    7. Add direction_split breakdown to the results dict.

    Parameters
    ----------
    context : BacktestContext
        Multi-timeframe data context providing primary and auxiliary data.
    strategy_fn : callable
        Signature: strategy_fn(context, params) -> pd.DataFrame.
        Must return a DataFrame with at minimum a 'position' column.
        Optional columns: 'position_size' (float in [0,1]),
                          'stop_loss' (float price, 0 = no stop).
    params : dict
        Parameter values passed directly to strategy_fn.
    cost : float
        Round-trip trading cost as a fraction of trade size. Default 0.001.
    show_plot : bool
        Display an interactive Plotly chart after the backtest. Default False.
    save_html : str, optional
        Save the results chart to this path.
    benchmark_data : pd.DataFrame, optional
        DataFrame with a 'close' column for buy-and-hold comparison.

    Returns
    -------
    dict
        All keys from calculate_all_metrics() plus 'direction_split'.
        Standard keys: total_return, sharpe_ratio, max_drawdown, win_rate,
        num_trades, avg_win_loss_ratio, profit_factor, calmar_ratio,
        yearly_returns, yearly_sharpe, yearly_max_drawdown, cost_percent,
        equity_curve, trades, direction_split.

    Raises
    ------
    ValueError
        If strategy_fn does not return a DataFrame with a 'position' column.
    """
    # --- 1. Generate signals ---
    signals = strategy_fn(context, params)

    if not isinstance(signals, pd.DataFrame) or "position" not in signals.columns:
        raise ValueError(
            "strategy_fn must return a pd.DataFrame with a 'position' column "
            "(values in [-1, 0, 1])."
        )

    # Merge price data with signals (align on index)
    primary = context.primary()
    df      = primary.copy()

    for col in signals.columns:
        df[col] = signals[col]

    # Rename 'close' -> 'Close' for compatibility with metrics/identify_trades
    df["Close"] = df["close"]

    # Ensure position is float, filled, clipped
    df["position"] = df["position"].fillna(0.0).astype(float).clip(-1.0, 1.0)

    # --- 2. Stop/target monitoring loop ---
    if "stop_loss" in df.columns and df["stop_loss"].abs().sum() > 0:
        df = _apply_stops_and_targets(df)

    # --- 3. Position sizing ---
    if "position_size" in df.columns:
        df["position_size"] = df["position_size"].fillna(1.0).clip(0.0, 1.0)
        df["effective_position"] = (df["position"] * df["position_size"]).shift(1)
    else:
        df["effective_position"] = df["position"].shift(1)

    # --- 4. Returns (preserved exactly from Epsilon Fund engine.py) ---
    df["returns"]          = df["Close"].pct_change()
    df["strategy_returns"] = df["effective_position"] * df["returns"]

    # --- 5. Costs ---
    df["position_change"] = df["position"].diff().abs()
    df["trade_cost"]      = df["position_change"] * cost
    df["net_returns"]     = df["strategy_returns"] - df["trade_cost"]

    # --- 6. Metrics ---
    metrics = calculate_all_metrics(
        data        = df,
        net_returns = df["net_returns"],
        cost        = cost,
    )

    # --- 7. Direction split ---
    metrics["direction_split"] = calculate_direction_split(
        trades_df    = metrics["trades"],
        net_returns  = df["net_returns"],
        equity_curve = metrics["equity_curve"],
        data         = df,
        cost         = cost,
    )

    # --- 8. Optional plot ---
    if show_plot or save_html:
        try:
            from .visualizer import plot_results
            bm = None
            if benchmark_data is not None:
                bm = benchmark_data.rename(columns={"close": "Close"})
            plot_results(metrics, benchmark_data=bm, show=show_plot, save_html=save_html)
        except Exception as e:
            logger.warning(f"Could not render plot: {e}")

    return metrics


# ---------------------------------------------------------------------------
# Stop / target monitoring loop
# ---------------------------------------------------------------------------

def _apply_stops_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bar-by-bar loop that forces position to 0 when a stop or 1R target is hit.

    This is the ONLY bar-by-bar loop in the engine. Everything else is
    vectorised. It runs before the vectorised returns calculation so that
    forced exits are reflected correctly in the P&L.

    Stop logic
    ----------
    - Long  (position > 0): if low[i] <= stop_loss[i-1], force position[i] = 0
    - Short (position < 0): if high[i] >= stop_loss[i-1], force position[i] = 0

    1R target logic
    ---------------
    Entry price is recorded when position changes from 0 to non-zero.
    - target_1r for longs:  entry_price + (entry_price - stop_loss)
    - target_1r for shorts: entry_price - (stop_loss - entry_price)
    - Long:  if high[i] >= target_1r, force position[i] = 0
    - Short: if low[i]  <= target_1r, force position[i] = 0

    Parameters
    ----------
    df : pd.DataFrame
        Strategy DataFrame containing at minimum: 'position', 'stop_loss',
        'high', 'low', 'Close'.

    Returns
    -------
    pd.DataFrame
        Copy of df with position values updated where stops/targets fired.
    """
    pos   = df["position"].values.copy().astype(float)
    stops = df["stop_loss"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows  = df["low"].values.astype(float)
    closes = df["Close"].values.astype(float)

    entry_price = 0.0
    in_position = False
    target_1r   = np.nan

    for i in range(1, len(df)):
        curr_pos  = pos[i]
        prev_stop = stops[i - 1]

        # Detect new entry (position goes from 0 to non-zero)
        if not in_position and curr_pos != 0:
            in_position = True
            entry_price = closes[i - 1]   # filled at prior close (1-bar lag)
            if prev_stop != 0 and not np.isnan(prev_stop):
                risk = abs(entry_price - prev_stop)
                target_1r = entry_price + risk if curr_pos > 0 else entry_price - risk
            else:
                target_1r = np.nan

        # While in a position, check whether stop or target is hit this bar
        if in_position and curr_pos != 0:
            stopped = False

            # Stop check
            if prev_stop != 0 and not np.isnan(prev_stop):
                if curr_pos > 0 and lows[i] <= prev_stop:
                    pos[i]  = 0.0
                    stopped = True
                elif curr_pos < 0 and highs[i] >= prev_stop:
                    pos[i]  = 0.0
                    stopped = True

            # 1R target check (only if stop not already triggered)
            if not stopped and not np.isnan(target_1r):
                if curr_pos > 0 and highs[i] >= target_1r:
                    pos[i]  = 0.0
                    stopped = True
                elif curr_pos < 0 and lows[i] <= target_1r:
                    pos[i]  = 0.0
                    stopped = True

            if stopped:
                in_position = False
                target_1r   = np.nan

        # Natural exit: strategy set position to 0
        if in_position and pos[i] == 0:
            in_position = False
            target_1r   = np.nan

    result        = df.copy()
    result["position"] = pos
    return result
