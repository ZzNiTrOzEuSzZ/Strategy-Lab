"""
infrastructure/backtester/data_loader.py
-----------------------------------------
Bridge between Layer 1 (Gold Parquet files) and Layer 2 (backtesting engine).

The engine never reads Parquet files directly — it always goes through this
module. If storage moves to a different backend in the future, only this file
needs to change.

The central concept is BacktestContext: the object passed to every strategy's
generate_signals() method. It controls what data is visible at each point in
time, making lookahead structurally impossible by design.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Project root is three levels up from this file:
# infrastructure/backtester/data_loader.py -> infrastructure -> project root
_ROOT = Path(__file__).resolve().parent.parent.parent
_GOLD_DIR    = _ROOT / "data" / "gold"
_ASSETS_YAML = _ROOT / "config" / "assets.yaml"


# ---------------------------------------------------------------------------
# BacktestContext
# ---------------------------------------------------------------------------

class BacktestContext:
    """
    Passed to strategy.generate_signals(). Provides lookahead-safe access to
    one primary timeframe and zero or more auxiliary timeframes.

    The primary timeframe is the one the strategy generates signals on.
    Auxiliary timeframes only expose fully closed bars — the current forming
    bar is never visible — achieved by aligning to the primary index with
    forward-fill.

    Parameters
    ----------
    primary_df : pd.DataFrame
        Full primary timeframe OHLCV DataFrame with UTC DatetimeIndex.
    aux_dfs : dict, optional
        Mapping of timeframe label -> DataFrame for each auxiliary timeframe.

    Example
    -------
    >>> ctx = get_context("EURUSD", "1H", aux_timeframes=["4H"])
    >>> h1 = ctx.primary()          # full 1H DataFrame
    >>> h4 = ctx.aux("4H")          # 4H aligned to 1H index via ffill
    """

    def __init__(
        self,
        primary_df: pd.DataFrame,
        aux_dfs: dict[str, pd.DataFrame] | None = None,
    ):
        self._primary = primary_df
        self._aux     = aux_dfs or {}

    def primary(self) -> pd.DataFrame:
        """
        Return the full primary timeframe DataFrame.

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame with UTC DatetimeIndex named 'time'.
        """
        return self._primary

    def aux(self, timeframe: str) -> pd.DataFrame:
        """
        Return an auxiliary timeframe DataFrame aligned to the primary index.

        Uses forward-fill so that at each primary bar the strategy sees only
        the last *fully closed* auxiliary bar — never the bar currently forming.

        Parameters
        ----------
        timeframe : str
            Timeframe label, e.g. "4H", "1D". Must have been requested in
            get_context(aux_timeframes=[...]).

        Returns
        -------
        pd.DataFrame
            Auxiliary OHLCV DataFrame reindexed to the primary index.

        Raises
        ------
        KeyError
            If the requested timeframe was not loaded into this context.

        Example
        -------
        >>> h4 = ctx.aux("4H")
        >>> h4.loc["2024-01-15 13:00+00:00", "close"]  # sees 12:00 closed bar
        """
        if timeframe not in self._aux:
            raise KeyError(
                f"Auxiliary timeframe '{timeframe}' not loaded in this context. "
                f"Available: {list(self._aux.keys())}"
            )
        return self._aux[timeframe]

    def slice(self, start=None, end=None) -> "BacktestContext":
        """
        Return a new BacktestContext restricted to [start, end].

        Both primary and all auxiliary DataFrames are sliced to the same
        date range. Used internally by the walk-forward engine to build
        fold-level contexts without re-reading Parquet files.

        Parameters
        ----------
        start : str or pd.Timestamp, optional
        end   : str or pd.Timestamp, optional

        Returns
        -------
        BacktestContext
        """
        primary_sliced = self._primary
        if start is not None:
            primary_sliced = primary_sliced[primary_sliced.index >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            primary_sliced = primary_sliced[primary_sliced.index <= pd.Timestamp(end, tz="UTC")]

        aux_sliced = {}
        for tf, df in self._aux.items():
            sliced = df
            if start is not None:
                sliced = sliced[sliced.index >= pd.Timestamp(start, tz="UTC")]
            if end is not None:
                sliced = sliced[sliced.index <= pd.Timestamp(end, tz="UTC")]
            aux_sliced[tf] = sliced

        return BacktestContext(primary_sliced, aux_sliced)


# ---------------------------------------------------------------------------
# load_asset
# ---------------------------------------------------------------------------

def load_asset(
    ticker: str,
    timeframe: str,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> pd.DataFrame:
    """
    Load a single Gold-layer Parquet file for one ticker and timeframe.

    Parameters
    ----------
    ticker : str
        Ticker symbol as stored in data/gold/, e.g. "EURUSD", "BTCUSDT", "SPY".
    timeframe : str
        One of "1H", "4H", "1D", "1W".
    start_date : str, optional
        ISO date string "YYYY-MM-DD". Rows before this date are dropped.
    end_date : str, optional
        ISO date string "YYYY-MM-DD". Rows after this date are dropped.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with UTC-aware DatetimeIndex named 'time' and
        float64 columns: open, high, low, close, volume.

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist. The error message lists what IS
        available in data/gold/{timeframe}/.

    Example
    -------
    >>> df = load_asset("EURUSD", "1H", start_date="2020-01-01")
    >>> print(df.shape)
    (35040, 5)
    """
    path = _GOLD_DIR / timeframe / f"{ticker}.parquet"

    if not path.exists():
        # List available tickers for this timeframe to help the caller
        tf_dir = _GOLD_DIR / timeframe
        available = []
        if tf_dir.exists():
            available = sorted(p.stem for p in tf_dir.glob("*.parquet"))
        raise FileNotFoundError(
            f"No data found for {ticker} at {timeframe}.\n"
            f"Expected path: {path}\n"
            f"Available tickers in {timeframe}: {available}"
        )

    df = pd.read_parquet(path, engine="pyarrow")

    # Ensure UTC-aware DatetimeIndex named 'time'
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "time"

    # Apply date filters
    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]
    if end_date is not None:
        df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]

    return df


# ---------------------------------------------------------------------------
# load_multiple
# ---------------------------------------------------------------------------

def load_multiple(
    tickers:    list[str],
    timeframe:  str,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load Gold-layer data for a list of tickers at one timeframe.

    One missing ticker does not crash the rest — failures are logged and
    the ticker is omitted from the returned dict.

    Parameters
    ----------
    tickers : list of str
    timeframe : str
        One of "1H", "4H", "1D", "1W".
    start_date : str, optional
    end_date : str, optional

    Returns
    -------
    dict
        Mapping ticker -> DataFrame. Missing tickers are absent.

    Example
    -------
    >>> frames = load_multiple(["EURUSD", "GBPUSD"], "1D")
    >>> frames["EURUSD"].shape
    (5587, 5)
    """
    result = {}
    for ticker in tickers:
        try:
            result[ticker] = load_asset(ticker, timeframe, start_date, end_date)
        except Exception as e:
            logger.warning(f"load_multiple: skipping {ticker} @ {timeframe} — {e}")
    return result


# ---------------------------------------------------------------------------
# get_context
# ---------------------------------------------------------------------------

def get_context(
    primary_ticker: str,
    primary_tf:     str,
    aux_timeframes: list[str] | None = None,
    start_date:     str | None = None,
    end_date:       str | None = None,
) -> BacktestContext:
    """
    Build a BacktestContext for a strategy run.

    Loads the primary timeframe and any auxiliary timeframes. Auxiliary
    DataFrames are reindexed to the primary index using forward-fill so
    strategies can never accidentally see a bar that has not yet closed.

    Parameters
    ----------
    primary_ticker : str
        Ticker symbol, e.g. "EURUSD".
    primary_tf : str
        Primary timeframe, e.g. "1H".
    aux_timeframes : list of str, optional
        Additional timeframes to load, e.g. ["4H", "1D"].
    start_date : str, optional
        ISO date string applied to all loaded data.
    end_date : str, optional
        ISO date string applied to all loaded data.

    Returns
    -------
    BacktestContext

    Example
    -------
    >>> ctx = get_context("EURUSD", "1H", aux_timeframes=["4H"])
    >>> h4  = ctx.aux("4H")
    >>> assert len(h4) == len(ctx.primary())  # aligned
    """
    primary_df = load_asset(primary_ticker, primary_tf, start_date, end_date)

    aux_dfs = {}
    if aux_timeframes:
        for tf in aux_timeframes:
            try:
                raw = load_asset(primary_ticker, tf, start_date, end_date)
                # Align to primary index: reindex with ffill so each primary
                # bar only sees the last *closed* auxiliary bar.
                aligned = raw.reindex(primary_df.index, method="ffill")
                aux_dfs[tf] = aligned
            except Exception as e:
                logger.warning(
                    f"get_context: could not load aux timeframe {tf} for "
                    f"{primary_ticker} — {e}"
                )

    return BacktestContext(primary_df, aux_dfs)


# ---------------------------------------------------------------------------
# list_available
# ---------------------------------------------------------------------------

def list_available(asset_class: str | None = None) -> dict[str, list[str]]:
    """
    Scan data/gold/ and return a dict of available tickers per timeframe.

    Parameters
    ----------
    asset_class : str, optional
        If provided ('fx', 'crypto', 'equities'), filters the ticker list
        by cross-referencing config/assets.yaml. Otherwise returns all tickers.

    Returns
    -------
    dict
        {"1H": ["EURUSD", "BTCUSDT", ...], "1D": [...], ...}

    Example
    -------
    >>> list_available()
    {'1H': ['AUDUSD', 'BTCUSDT', ...], '1D': [...], ...}
    >>> list_available(asset_class='fx')
    {'1H': ['AUDUSD', 'EURUSD', ...], ...}
    """
    if not _GOLD_DIR.exists():
        return {}

    # Optionally build a set of allowed tickers from assets.yaml
    allowed = None
    if asset_class is not None and _ASSETS_YAML.exists():
        with open(_ASSETS_YAML, "r") as f:
            cfg = yaml.safe_load(f)
        assets = cfg.get(asset_class, [])
        allowed = set()
        for a in assets:
            display = a.get("ticker_display") or a.get("ticker")
            if display:
                allowed.add(display)

    result = {}
    for tf_dir in sorted(_GOLD_DIR.iterdir()):
        if not tf_dir.is_dir():
            continue
        tickers = sorted(p.stem for p in tf_dir.glob("*.parquet"))
        if allowed is not None:
            tickers = [t for t in tickers if t in allowed]
        result[tf_dir.name] = tickers

    return result
