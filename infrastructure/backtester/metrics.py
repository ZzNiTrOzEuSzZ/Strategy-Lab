"""
infrastructure/backtester/metrics.py
-------------------------------------
Ported from Epsilon Fund / infrastructure / backtester / performance_metrics.py.
Logic is preserved exactly. Added: calculate_direction_split() at the bottom.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Frequency inference
# ---------------------------------------------------------------------------

def infer_frequency(index):
    """
    Infer the number of trading periods per year from a DatetimeIndex.

    Parameters
    ----------
    index : pd.DatetimeIndex

    Returns
    -------
    int
        8760 (hourly), 2190 (4h), 365 (daily), 52 (weekly), 12 (monthly).
    """
    if len(index) < 2:
        return 365
    time_diffs  = index.to_series().diff().dropna()
    median_diff = time_diffs.median()
    hours       = median_diff.total_seconds() / 3600
    if hours <= 1:
        return 8760
    elif hours <= 4:
        return 2190
    elif hours <= 24:
        return 365
    elif hours <= 168:
        return 52
    else:
        return 12


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def calculate_total_return(equity_curve):
    """Return final equity minus 1."""
    return equity_curve.iloc[-1] - 1


def calculate_sharpe_ratio(returns, periods_per_year):
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
    periods_per_year : int

    Returns
    -------
    float
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0
    mean_return = returns.mean()
    std_return  = returns.std()
    if std_return == 0:
        return 0.0
    return (mean_return / std_return) * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve):
    """
    Maximum peak-to-trough drawdown.

    Returns
    -------
    float
        Negative value, e.g. -0.25 means -25%.
    """
    running_max = equity_curve.cummax()
    drawdown    = (equity_curve - running_max) / running_max
    return drawdown.min()


def identify_trades(data):
    """
    Parse position changes into individual completed trade records.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'position', 'position_change', and 'Close' columns.

    Returns
    -------
    pd.DataFrame
        Columns: entry_time, exit_time, entry_price, exit_price,
        direction ('Long'/'Short'), pnl.
    """
    trades = []

    if len(data[data['position_change'] > 0]) == 0:
        return pd.DataFrame(columns=['entry_time', 'exit_time', 'entry_price', 'exit_price', 'direction', 'pnl'])

    in_position     = False
    entry_price     = None
    entry_direction = None
    entry_time      = None

    for idx, row in data.iterrows():
        current_position = row['position']

        if not in_position and current_position != 0:
            in_position     = True
            entry_price     = row['Close']
            entry_direction = current_position
            entry_time      = idx

        elif in_position and (
            current_position == 0
            or (current_position != 0 and current_position != entry_direction)
        ):
            exit_price = row['Close']
            exit_time  = idx

            pnl = (
                (exit_price - entry_price) / entry_price
                if entry_direction == 1
                else (entry_price - exit_price) / entry_price
            )

            trades.append({
                'entry_time':  entry_time,
                'exit_time':   exit_time,
                'entry_price': entry_price,
                'exit_price':  exit_price,
                'direction':   'Long' if entry_direction == 1 else 'Short',
                'pnl':         pnl,
            })

            if current_position != 0:
                entry_price     = row['Close']
                entry_direction = current_position
                entry_time      = idx
            else:
                in_position = False

    return pd.DataFrame(trades)


def calculate_win_rate(trades_df):
    """Fraction of trades with positive P&L. Returns 0.0 if no trades."""
    if len(trades_df) == 0:
        return 0.0
    return len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)


def calculate_num_trades(trades_df):
    """Return the number of completed trades."""
    return len(trades_df)


def calculate_avg_win_loss_ratio(trades_df):
    """
    Average win / average absolute loss.

    Returns
    -------
    float
        0.0 if no trades or one side is missing.
    """
    if len(trades_df) == 0:
        return 0.0
    winning = trades_df[trades_df['pnl'] > 0]['pnl']
    losing  = trades_df[trades_df['pnl'] < 0]['pnl']
    if len(winning) == 0 or len(losing) == 0:
        return 0.0
    return winning.mean() / abs(losing.mean())


def calculate_profit_factor(trades_df):
    """
    Gross profit / gross loss.

    Returns
    -------
    float
        np.inf if no losing trades, 0.0 if no trades.
    """
    if len(trades_df) == 0:
        return 0.0
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss   = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calculate_calmar_ratio(total_return, max_drawdown):
    """total_return / |max_drawdown|. Returns 0.0 if max_drawdown is zero."""
    if max_drawdown == 0:
        return 0.0
    return total_return / abs(max_drawdown)


def build_equity_curve(returns, return_type="arithmetic"):
    """
    Build a normalised equity curve from a returns series.

    Parameters
    ----------
    returns : pd.Series
    return_type : str
        'arithmetic' -> (1+r).cumprod()   |   'log' -> exp(cumsum(r))

    Returns
    -------
    pd.Series
    """
    returns = returns.fillna(0.0)
    if return_type == "log":
        equity_curve = np.exp(returns.cumsum())
    else:
        equity_curve = (1 + returns).cumprod()
    equity_curve = pd.Series(equity_curve, index=returns.index)
    equity_curve.fillna(1.0, inplace=True)
    return equity_curve


def to_arithmetic_returns(returns, return_type="arithmetic"):
    """Convert log returns to arithmetic if needed; no-op otherwise."""
    returns = returns.copy()
    if return_type == "log":
        return np.exp(returns) - 1.0
    return returns


def calculate_yearly_metrics(returns, equity_curve, periods_per_year):
    """
    Break performance down by calendar year.

    Parameters
    ----------
    returns : pd.Series
        Arithmetic per-bar returns.
    equity_curve : pd.Series
    periods_per_year : int

    Returns
    -------
    dict
        'yearly_returns', 'yearly_sharpe', 'yearly_max_drawdown' —
        each maps year (int) -> float.
    """
    returns_by_year = returns.groupby(returns.index.year)
    equity_by_year  = equity_curve.groupby(equity_curve.index.year)

    yearly_returns = {}
    yearly_sharpe  = {}
    yearly_max_dd  = {}

    for year in returns_by_year.groups.keys():
        year_returns = returns_by_year.get_group(year)
        year_equity  = equity_by_year.get_group(year)

        start_value = year_equity.iloc[0]
        end_value   = year_equity.iloc[-1]
        yearly_returns[year] = (
            (end_value - start_value) / start_value if start_value != 0 else 0.0
        )

        std_ret = year_returns.std()
        yearly_sharpe[year] = (
            (year_returns.mean() / std_ret) * np.sqrt(periods_per_year)
            if std_ret > 0 else 0.0
        )

        running_max = year_equity.cummax()
        yearly_max_dd[year] = ((year_equity - running_max) / running_max).min()

    return {
        'yearly_returns':      yearly_returns,
        'yearly_sharpe':       yearly_sharpe,
        'yearly_max_drawdown': yearly_max_dd,
    }


def calculate_all_metrics(data, net_returns, cost, return_type="arithmetic"):
    """
    Compute and compile the full set of backtest performance metrics.

    Parameters
    ----------
    data : pd.DataFrame
        Strategy DataFrame with 'position', 'position_change', 'Close'.
    net_returns : pd.Series
        Per-bar net returns after costs.
    cost : float
        Round-trip cost fraction (stored for reporting).
    return_type : str
        'arithmetic' or 'log'.

    Returns
    -------
    dict
        Keys: total_return, sharpe_ratio, max_drawdown, win_rate, num_trades,
        avg_win_loss_ratio, profit_factor, calmar_ratio, yearly_returns,
        yearly_sharpe, yearly_max_drawdown, cost_percent, equity_curve, trades.
    """
    arith_returns    = to_arithmetic_returns(net_returns, return_type=return_type)
    equity_curve     = build_equity_curve(net_returns, return_type=return_type)
    periods_per_year = infer_frequency(data.index)
    trades_df        = identify_trades(data)

    total_return  = calculate_total_return(equity_curve)
    sharpe_ratio  = calculate_sharpe_ratio(arith_returns, periods_per_year)
    max_drawdown  = calculate_max_drawdown(equity_curve)
    win_rate      = calculate_win_rate(trades_df)
    num_trades    = calculate_num_trades(trades_df)
    avg_win_loss  = calculate_avg_win_loss_ratio(trades_df)
    profit_factor = calculate_profit_factor(trades_df)
    calmar_ratio  = calculate_calmar_ratio(total_return, max_drawdown)
    yearly        = calculate_yearly_metrics(arith_returns, equity_curve, periods_per_year)

    return {
        'total_return':        total_return,
        'sharpe_ratio':        sharpe_ratio,
        'max_drawdown':        max_drawdown,
        'win_rate':            win_rate,
        'num_trades':          num_trades,
        'avg_win_loss_ratio':  avg_win_loss,
        'profit_factor':       profit_factor,
        'calmar_ratio':        calmar_ratio,
        'yearly_returns':      yearly['yearly_returns'],
        'yearly_sharpe':       yearly['yearly_sharpe'],
        'yearly_max_drawdown': yearly['yearly_max_drawdown'],
        'cost_percent':        cost,
        'equity_curve':        equity_curve,
        'trades':              trades_df,
    }


# ---------------------------------------------------------------------------
# Direction split (new)
# ---------------------------------------------------------------------------

def calculate_direction_split(trades_df, net_returns, equity_curve, data, cost):
    """
    Split backtest performance into long-only and short-only breakdowns.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Output of identify_trades() — must have a 'direction' column.
    net_returns : pd.Series
        Full per-bar net returns.
    equity_curve : pd.Series
        Full equity curve.
    data : pd.DataFrame
        Full strategy DataFrame (used only to infer periods_per_year).
    cost : float
        Round-trip cost fraction.

    Returns
    -------
    dict
        {
            'combined':   dict of all standard metrics,
            'long_only':  dict or None  (None if zero long trades),
            'short_only': dict or None  (None if zero short trades),
        }

    Example
    -------
    >>> split = calculate_direction_split(trades_df, net_returns, equity_curve, df, 0.001)
    >>> split['long_only']['sharpe_ratio']
    1.24
    """
    periods_per_year = infer_frequency(data.index)

    combined = {
        'total_return':       calculate_total_return(equity_curve),
        'sharpe_ratio':       calculate_sharpe_ratio(net_returns.fillna(0), periods_per_year),
        'max_drawdown':       calculate_max_drawdown(equity_curve),
        'win_rate':           calculate_win_rate(trades_df),
        'num_trades':         calculate_num_trades(trades_df),
        'avg_win_loss_ratio': calculate_avg_win_loss_ratio(trades_df),
        'profit_factor':      calculate_profit_factor(trades_df),
        'calmar_ratio':       calculate_calmar_ratio(
            calculate_total_return(equity_curve),
            calculate_max_drawdown(equity_curve),
        ),
        'cost_percent': cost,
    }

    def _metrics_for(direction_label):
        subset = trades_df[trades_df['direction'] == direction_label]
        if len(subset) == 0:
            return None

        win_rate      = calculate_win_rate(subset)
        num_trades    = calculate_num_trades(subset)
        avg_wl        = calculate_avg_win_loss_ratio(subset)
        pf            = calculate_profit_factor(subset)
        total_pnl     = float(subset['pnl'].sum())

        # Approximate Sharpe from trade-level P&L series
        pnl_std = subset['pnl'].std()
        if len(subset) > 1 and pnl_std > 0:
            sharpe = (subset['pnl'].mean() / pnl_std) * np.sqrt(
                min(periods_per_year, num_trades)
            )
        else:
            sharpe = 0.0

        # Approximate max drawdown from cumulative trade P&L
        cum_pnl = subset['pnl'].cumsum()
        max_dd  = float((cum_pnl - cum_pnl.cummax()).min()) if len(subset) > 0 else 0.0

        return {
            'total_return':       total_pnl,
            'sharpe_ratio':       float(sharpe),
            'max_drawdown':       max_dd,
            'win_rate':           win_rate,
            'num_trades':         num_trades,
            'avg_win_loss_ratio': avg_wl,
            'profit_factor':      pf,
            'calmar_ratio':       calculate_calmar_ratio(total_pnl, max_dd),
            'cost_percent':       cost,
        }

    return {
        'combined':   combined,
        'long_only':  _metrics_for('Long'),
        'short_only': _metrics_for('Short'),
    }
