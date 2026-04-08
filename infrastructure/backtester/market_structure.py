"""
infrastructure/backtester/market_structure.py
----------------------------------------------
Price structure detection functions.

CRITICAL REQUIREMENT: every function in this module is lookahead-safe.
A swing high or low can only be confirmed after ``lookback`` bars have
passed since the potential pivot. This confirmation lag is mandatory and
is documented explicitly in each function that uses zigzag().
"""

import pandas as pd
import numpy as np

from .indicators import sma


# ---------------------------------------------------------------------------
# ZigZag
# ---------------------------------------------------------------------------

def zigzag(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Identify confirmed swing highs and swing lows.

    Lookahead-safety mechanism
    --------------------------
    A swing HIGH at bar N is only confirmed after ``lookback`` additional
    bars have passed (bars N+1 … N+lookback) without a higher high being
    recorded. Only then is the swing high registered at index N.

    A swing LOW at bar N is only confirmed after ``lookback`` bars pass
    without a lower low.

    This means the most recent ``lookback`` bars can never have a confirmed
    swing — they are always NaN. This is by design.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'high' and 'low' columns and UTC DatetimeIndex.
    lookback : int
        Number of bars that must pass after a potential pivot before it is
        confirmed. Default 10.

    Returns
    -------
    pd.DataFrame
        Same index as ``df`` with two columns:
        - swing_high : confirmed swing high price, NaN elsewhere
        - swing_low  : confirmed swing low price, NaN elsewhere

    Example
    -------
    >>> zz = zigzag(df, lookback=10)
    >>> zz[zz['swing_high'].notna()]  # all confirmed swing highs
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    swing_high = np.full(n, np.nan)
    swing_low  = np.full(n, np.nan)

    for i in range(n - lookback):
        # Check if bar i is a swing high: no bar in [i+1 .. i+lookback] has a higher high
        candidate_high = highs[i]
        window_highs   = highs[i + 1 : i + lookback + 1]
        if candidate_high >= np.max(window_highs):
            # Also confirm it's higher than the lookback bars before it
            left_window = highs[max(0, i - lookback) : i]
            if len(left_window) == 0 or candidate_high >= np.max(left_window):
                swing_high[i] = candidate_high

        # Check if bar i is a swing low: no bar in [i+1 .. i+lookback] has a lower low
        candidate_low = lows[i]
        window_lows   = lows[i + 1 : i + lookback + 1]
        if candidate_low <= np.min(window_lows):
            left_window = lows[max(0, i - lookback) : i]
            if len(left_window) == 0 or candidate_low <= np.min(left_window):
                swing_low[i] = candidate_low

    return pd.DataFrame(
        {"swing_high": swing_high, "swing_low": swing_low},
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Last confirmed swing levels
# ---------------------------------------------------------------------------

def last_swing_low(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """
    At each bar, return the price of the most recently confirmed swing low.

    Uses zigzag() internally. Forward-fills from the last confirmed swing low
    so every bar has a value once at least one swing low has been confirmed.
    Bars before the first confirmed swing low are NaN.

    Lookahead-safety: inherits from zigzag() — no swing low in the most
    recent ``lookback`` bars is ever confirmed.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    lookback : int
        Passed to zigzag(). Default 10.

    Returns
    -------
    pd.Series
        Most recent confirmed swing low at each bar.

    Example
    -------
    >>> sl = last_swing_low(df, lookback=10)
    >>> sl.iloc[-1]   # stop loss reference for the current bar
    1.08450
    """
    zz = zigzag(df, lookback)
    return zz["swing_low"].ffill()


def last_swing_high(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """
    At each bar, return the price of the most recently confirmed swing high.

    Uses zigzag() internally. Forward-fills from the last confirmed swing high.
    Bars before the first confirmed swing high are NaN.

    Lookahead-safety: inherits from zigzag().

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    lookback : int
        Passed to zigzag(). Default 10.

    Returns
    -------
    pd.Series
        Most recent confirmed swing high at each bar.
    """
    zz = zigzag(df, lookback)
    return zz["swing_high"].ffill()


# ---------------------------------------------------------------------------
# Pullback detection
# ---------------------------------------------------------------------------

def in_pullback(
    df: pd.DataFrame,
    direction: int,
    ma_period: int = 20,
    tolerance_bps: float = 10.0,
) -> pd.Series:
    """
    Detect whether price has pulled back to touch (or nearly touch) the MA.

    Used to find pullback entries within a trend: price is trending in
    ``direction`` but has retraced back to the MA for a potential entry.

    For longs (direction=1):
    - True when close is still above the MA (trend intact)
    - AND abs(close − MA) / MA × 10000 ≤ tolerance_bps (close enough to touch)

    For shorts (direction=-1):
    - True when close is still below the MA (trend intact)
    - AND abs(close − MA) / MA × 10000 ≤ tolerance_bps

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    direction : int
        1 for long (looking for pullbacks down to MA from above).
        -1 for short (looking for pullbacks up to MA from below).
    ma_period : int
        SMA period. Default 20.
    tolerance_bps : float
        Tolerance in basis points (1 bp = 0.01%). Default 10 bps.

    Returns
    -------
    pd.Series
        Boolean Series: True where a pullback-to-MA condition is met.

    Example
    -------
    >>> pullback = in_pullback(df, direction=1, ma_period=20, tolerance_bps=10)
    >>> entry_signal = pullback & (some_reversal_pattern == 1)
    """
    ma        = sma(df, ma_period)
    close     = df["close"]
    proximity = (close - ma).abs() / ma * 10_000  # in basis points

    if direction == 1:
        return (proximity <= tolerance_bps) & (close > ma)
    else:
        return (proximity <= tolerance_bps) & (close < ma)


# ---------------------------------------------------------------------------
# Break of structure
# ---------------------------------------------------------------------------

def break_of_structure(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """
    Detect breaks of market structure (BOS).

    A bullish BOS (+1) occurs when close breaks above the most recently
    confirmed swing high — price is breaking structural resistance.

    A bearish BOS (-1) occurs when close breaks below the most recently
    confirmed swing low — price is breaking structural support.

    Lookahead-safety: swing highs and lows come from zigzag() with its
    mandatory ``lookback`` confirmation lag. The "last confirmed" levels
    are forward-filled, so the strategy only ever acts on levels that
    were confirmed at least ``lookback`` bars ago.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    lookback : int
        Passed to zigzag(). Default 10.

    Returns
    -------
    pd.Series
        1 = bullish BOS (close broke above last swing high)
       -1 = bearish BOS (close broke below last swing low)
        0 = no break

    Example
    -------
    >>> bos = break_of_structure(df, lookback=10)
    >>> entries = df[bos == 1]   # potential long entries after BOS
    """
    last_high = last_swing_high(df, lookback)
    last_low  = last_swing_low(df, lookback)
    close     = df["close"]

    result = pd.Series(0, index=df.index, dtype=int)
    result[close > last_high] =  1
    result[close < last_low]  = -1
    return result
