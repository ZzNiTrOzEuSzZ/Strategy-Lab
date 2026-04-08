"""
infrastructure/backtester/indicators.py
-----------------------------------------
Pure technical indicator functions. No classes, no state, no side effects.

Every function:
- Takes a DataFrame with standard OHLCV columns (open, high, low, close, volume)
- Returns a pd.Series (or tuple of Series) with the same index as the input
- Uses only current and past data — lookahead is structurally impossible
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Trend indicators
# ---------------------------------------------------------------------------

def sma(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Simple moving average of close price.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a 'close' column.
    period : int
        Number of bars for the rolling window.

    Returns
    -------
    pd.Series
        SMA values, NaN for the first (period - 1) bars.
    """
    return df["close"].rolling(period).mean()


def ema(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Exponential moving average of close price.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a 'close' column.
    period : int
        EMA span (equivalent to the lookback period for the smoothing factor).

    Returns
    -------
    pd.Series
        EMA values with the same index as df.
    """
    return df["close"].ewm(span=period, adjust=False).mean()


def ma_slope(df: pd.DataFrame, period: int, lookback: int = 1) -> pd.Series:
    """
    Difference between the current SMA value and ``lookback`` bars ago.

    A positive value means the MA is rising (price trend is up); a negative
    value means it is falling. This is commonly used to confirm that the MA
    is "moving towards the move" before entering a trend trade.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    period : int
        SMA period.
    lookback : int
        Number of bars to look back for the comparison. Default 1.

    Returns
    -------
    pd.Series
        Slope values. NaN for the first (period + lookback - 1) bars.
    """
    moving_avg = sma(df, period)
    return moving_avg - moving_avg.shift(lookback)


# ---------------------------------------------------------------------------
# Volatility indicators
# ---------------------------------------------------------------------------

def bollinger_bands(
    df: pd.DataFrame, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands: upper, middle, and lower bands.

    Middle = SMA(close, period)
    Upper  = middle + num_std × rolling_std(close, period)
    Lower  = middle − num_std × rolling_std(close, period)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a 'close' column.
    period : int
        Rolling window for SMA and standard deviation. Default 20.
    num_std : float
        Number of standard deviations for the band width. Default 2.0.

    Returns
    -------
    tuple of (pd.Series, pd.Series, pd.Series)
        (upper, middle, lower) — all with the same index as df.
    """
    middle = df["close"].rolling(period).mean()
    std    = df["close"].rolling(period).std()
    upper  = middle + num_std * std
    lower  = middle - num_std * std
    return upper, middle, lower


def bb_width(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Bollinger Band width: (upper − lower) / middle.

    Measures the width of the bands relative to price. Increasing bandwidth
    indicates expanding volatility; decreasing indicates a squeeze.

    Parameters
    ----------
    df : pd.DataFrame
    period : int
    num_std : float

    Returns
    -------
    pd.Series
        Band width ratio. NaN where middle is zero or insufficient history.
    """
    upper, middle, lower = bollinger_bands(df, period, num_std)
    return (upper - lower) / middle


def bb_percent_b(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Bollinger %B: position of close within the band.

    Formula: (close − lower) / (upper − lower)

    Values:
    - 1.0 = close is at the upper band
    - 0.5 = close is at the middle band
    - 0.0 = close is at the lower band
    - > 1 or < 0 = outside the bands

    Parameters
    ----------
    df : pd.DataFrame
    period : int
    num_std : float

    Returns
    -------
    pd.Series
    """
    upper, middle, lower = bollinger_bands(df, period, num_std)
    band_width = upper - lower
    return (df["close"] - lower) / band_width.replace(0, float("nan"))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.

    True range at each bar = max(high − low,
                                 |high − prev_close|,
                                 |low  − prev_close|)
    ATR = EMA(true_range, period)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'high', 'low', 'close' columns.
    period : int
        Smoothing period. Default 14.

    Returns
    -------
    pd.Series
        ATR values in price units.
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def candle_range(df: pd.DataFrame) -> pd.Series:
    """
    High minus low for each bar.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'high' and 'low' columns.

    Returns
    -------
    pd.Series
        Bar range in price units.
    """
    return df["high"] - df["low"]


def avg_candle_range(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Rolling mean of bar range (high − low) over ``period`` bars.

    Use this to contextualise whether the current bar is "bigger than
    recent moves" — e.g. a candle whose range is ≥ 2× avg_candle_range
    is unusually large.

    Parameters
    ----------
    df : pd.DataFrame
    period : int
        Rolling window. Default 10.

    Returns
    -------
    pd.Series
    """
    return candle_range(df).rolling(period).mean()


# ---------------------------------------------------------------------------
# Candlestick pattern detection
# ---------------------------------------------------------------------------
# All pattern functions return pd.Series of integers:
#   1  = bullish signal
#  -1  = bearish signal
#   0  = no pattern
#
# Every calculation uses only current and past bars.

def engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Detect bullish and bearish engulfing candle patterns.

    Bullish engulfing:
    - Current bar is green (close > open)
    - Previous bar is red (close < open)
    - Current bar's body fully contains the previous bar's body
      (current open ≤ prev close  AND  current close ≥ prev open)

    Bearish engulfing:
    - Current bar is red (close < open)
    - Previous bar is green (close > open)
    - Current bar's body fully contains the previous bar's body

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'open' and 'close' columns.

    Returns
    -------
    pd.Series
        1 = bullish engulfing, -1 = bearish engulfing, 0 = none.
    """
    o = df["open"]
    c = df["close"]
    po = o.shift(1)
    pc = c.shift(1)

    prev_green = pc > po
    prev_red   = pc < po

    # Bullish: current green, prev red, current body engulfs prev body
    bullish = (
        (c > o)          # current green
        & prev_red       # previous red
        & (o <= pc)      # current open below or at prev close
        & (c >= po)      # current close above or at prev open
    )

    # Bearish: current red, prev green, current body engulfs prev body
    bearish = (
        (c < o)          # current red
        & prev_green     # previous green
        & (o >= pc)      # current open above or at prev close
        & (c <= po)      # current close below or at prev open
    )

    result = pd.Series(0, index=df.index, dtype=int)
    result[bullish] =  1
    result[bearish] = -1
    return result


def three_bar_reversal(df: pd.DataFrame) -> pd.Series:
    """
    Detect three-bar reversal patterns.

    Bullish three-bar reversal:
    - Bar[n-2] is a down bar (close < open)
    - Bar[n-1] is a down bar (close < open)
    - Bar[n]   is a strong up bar that closes above bar[n-2]'s open

    Bearish three-bar reversal:
    - Bar[n-2] is an up bar
    - Bar[n-1] is an up bar
    - Bar[n]   is a strong down bar that closes below bar[n-2]'s open

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'open' and 'close' columns.

    Returns
    -------
    pd.Series
        1 = bullish reversal, -1 = bearish reversal, 0 = none.
    """
    o  = df["open"]
    c  = df["close"]
    o2 = o.shift(2)   # two bars ago
    c1 = c.shift(1)   # one bar ago
    o1 = o.shift(1)
    c2 = c.shift(2)

    bullish = (
        (c2 < o2)      # bar[n-2] down
        & (c1 < o1)    # bar[n-1] down
        & (c > o)      # current up
        & (c > o2)     # closes decisively above bar[n-2]'s open
    )

    bearish = (
        (c2 > o2)      # bar[n-2] up
        & (c1 > o1)    # bar[n-1] up
        & (c < o)      # current down
        & (c < o2)     # closes decisively below bar[n-2]'s open
    )

    result = pd.Series(0, index=df.index, dtype=int)
    result[bullish] =  1
    result[bearish] = -1
    return result


def pin_bar(df: pd.DataFrame, wick_ratio: float = 2.0) -> pd.Series:
    """
    Detect pin bar (hammer / shooting star) patterns.

    Bullish pin bar:
    - Lower wick ≥ wick_ratio × body size
    - Upper wick is small (< body size)

    Bearish pin bar:
    - Upper wick ≥ wick_ratio × body size
    - Lower wick is small (< body size)

    Definitions:
    - body       = |close − open|
    - lower_wick = min(open, close) − low
    - upper_wick = high − max(open, close)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'open', 'high', 'low', 'close'.
    wick_ratio : float
        Minimum ratio of wick-to-body for a valid pin. Default 2.0.

    Returns
    -------
    pd.Series
        1 = bullish pin, -1 = bearish pin, 0 = none.
    """
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    body        = (c - o).abs()
    lower_wick  = pd.concat([o, c], axis=1).min(axis=1) - l
    upper_wick  = h - pd.concat([o, c], axis=1).max(axis=1)

    # Avoid division by zero for doji-like candles
    body_safe = body.replace(0, float("nan"))

    bullish = (
        (lower_wick >= wick_ratio * body_safe)
        & (upper_wick < body_safe)
    )

    bearish = (
        (upper_wick >= wick_ratio * body_safe)
        & (lower_wick < body_safe)
    )

    result = pd.Series(0, index=df.index, dtype=int)
    result[bullish] =  1
    result[bearish] = -1
    return result


def any_reversal_pattern(df: pd.DataFrame, wick_ratio: float = 2.0) -> pd.Series:
    """
    Return the first reversal pattern that fires at each bar.

    Priority order: engulfing → three_bar_reversal → pin_bar.

    If multiple patterns fire on the same bar, the highest-priority one wins.
    Mixed signals (one bullish, one bearish) at the same bar return the
    highest-priority signal's direction.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    wick_ratio : float
        Passed to pin_bar(). Default 2.0.

    Returns
    -------
    pd.Series
        1 = bullish signal, -1 = bearish signal, 0 = no pattern.
    """
    eng  = engulfing(df)
    tbr  = three_bar_reversal(df)
    pin  = pin_bar(df, wick_ratio)

    # Start with zeros; fill in priority order (last overwrite wins in reverse,
    # so apply lowest priority first)
    result = pd.Series(0, index=df.index, dtype=int)
    result[pin  != 0] = pin[pin   != 0]
    result[tbr  != 0] = tbr[tbr   != 0]
    result[eng  != 0] = eng[eng   != 0]   # highest priority — overwrites others
    return result
