"""
infrastructure/backtester/sizing.py
-------------------------------------
Position sizing models.

All functions return a float or pd.Series with values in [0, 1] representing
the fraction of maximum capital to deploy. The engine multiplies the position
signal by this fraction to get the effective position size.
"""

import pandas as pd
import numpy as np

from .indicators import atr as calc_atr


# ---------------------------------------------------------------------------
# Fixed size
# ---------------------------------------------------------------------------

def fixed_size(size: float = 1.0) -> float:
    """
    Always return a constant position size.

    The simplest sizing model. Use this for strategies where you want full
    position sizing at all times, or when you want to isolate signal quality
    from sizing effects.

    Parameters
    ----------
    size : float
        Fraction of capital, in [0, 1]. Default 1.0 (full position).

    Returns
    -------
    float
        The ``size`` value unchanged.

    Example
    -------
    >>> fixed_size()
    1.0
    >>> fixed_size(0.5)
    0.5
    """
    return float(size)


# ---------------------------------------------------------------------------
# ATR-based sizing
# ---------------------------------------------------------------------------

def atr_size(
    df:               pd.DataFrame,
    entry_price:      pd.Series,
    stop_price:       pd.Series,
    account_risk_pct: float = 0.01,
    atr_period:       int   = 14,
) -> pd.Series:
    """
    Size position so that a stop-out costs exactly ``account_risk_pct`` of capital.

    Formula (per unit of position):
        risk_per_unit = |entry_price − stop_price|
        size_fraction = account_risk_pct / risk_per_unit

    The result is clipped to [0, 1]. When stop_price is 0 or NaN (no stop
    defined), falls back to fixed_size(1.0).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame — used only for ATR calculation as a fallback when
        stop_price is NaN (in that case the ATR is used as a proxy for risk).
    entry_price : pd.Series
        Entry price at each bar. Typically df['close'].
    stop_price : pd.Series
        Stop loss price at each bar. 0 or NaN means no stop.
    account_risk_pct : float
        Fraction of account to risk per trade. Default 0.01 (1%).
    atr_period : int
        ATR period used only as a fallback when stop_price is unavailable.

    Returns
    -------
    pd.Series
        Position size fraction, clipped to [0, 1].

    Example
    -------
    >>> sizes = atr_size(df, df['close'], stop_series, account_risk_pct=0.01)
    >>> sizes.describe()
    """
    stop = stop_price.copy().replace(0, np.nan)

    risk_per_unit = (entry_price - stop).abs()

    # Where stop is NaN, fall back to using ATR as a proxy for risk
    atr_series = calc_atr(df, atr_period)
    risk_per_unit = risk_per_unit.fillna(atr_series)

    # Avoid division by zero
    risk_per_unit = risk_per_unit.replace(0, np.nan)

    size = account_risk_pct / risk_per_unit

    # Where we still have NaN (no ATR data either), use full size
    size = size.fillna(1.0)

    return size.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Volatility-targeted sizing
# ---------------------------------------------------------------------------

def volatility_target_size(
    df:         pd.DataFrame,
    target_vol: float = 0.15,
    period:     int   = 20,
) -> pd.Series:
    """
    Scale position size so that annualised realised volatility equals ``target_vol``.

    Formula:
        realised_vol = rolling_std(returns, period) × sqrt(periods_per_year)
        size         = target_vol / realised_vol

    The result is clipped to [0, 1].

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a 'close' column.
    target_vol : float
        Annualised volatility target, e.g. 0.15 = 15%. Default 0.15.
    period : int
        Rolling window for realised volatility estimation. Default 20.

    Returns
    -------
    pd.Series
        Position size fraction in [0, 1].

    Notes
    -----
    periods_per_year is inferred from the median bar spacing in df.index.
    For hourly data this is 8760; for daily data it is 365.

    Example
    -------
    >>> sizes = volatility_target_size(df, target_vol=0.10, period=20)
    """
    from .metrics import infer_frequency

    returns          = df["close"].pct_change()
    periods_per_year = infer_frequency(df.index)

    realised_vol = returns.rolling(period).std() * np.sqrt(periods_per_year)

    # Avoid division by zero
    realised_vol = realised_vol.replace(0, np.nan)

    size = target_vol / realised_vol
    size = size.fillna(1.0)

    return size.clip(0.0, 1.0)
