"""
strategies/base.py
-------------------
Abstract base class that every StratLab strategy must inherit from.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class for all StratLab strategies.

    Every strategy must:
    1. Define NAME (str) — human readable name.
    2. Define DESCRIPTION (str) — what the strategy does in plain English.
    3. Define PARAM_SPACE (dict) — parameter search space for optimisation.
       Format: {"param_name": ("int"|"float", min_value, max_value)}
    4. Implement generate_signals(context, params) -> pd.DataFrame.

    The generate_signals function receives a BacktestContext and a params dict.
    It must return a DataFrame with at minimum a "position" column.

    Optional columns in the returned DataFrame:
    - position_size : float in [0, 1], defaults to abs(position)
    - stop_loss     : float price level, 0 = no stop

    Example
    -------
    >>> class MACross(BaseStrategy):
    ...     NAME = "MA Crossover"
    ...     DESCRIPTION = "Long when fast MA crosses above slow MA"
    ...     PARAM_SPACE = {
    ...         "fast": ("int", 5,  50),
    ...         "slow": ("int", 20, 200),
    ...     }
    ...     def generate_signals(self, context, params):
    ...         df = context.primary().copy()
    ...         from infrastructure.backtester.indicators import sma
    ...         fast = sma(df, params["fast"])
    ...         slow = sma(df, params["slow"])
    ...         df["position"] = 0.0
    ...         df.loc[fast > slow, "position"] = 1.0
    ...         df.loc[fast < slow, "position"] = -1.0
    ...         return df[["position"]]
    """

    NAME:        str  = ""
    DESCRIPTION: str  = ""
    PARAM_SPACE: dict = {}

    TIMEFRAMES: dict = {
        "primary": "1H",
        "aux":     ["4H"],
    }
    # Example for a daily strategy with weekly filter:
    # TIMEFRAMES = {"primary": "1D", "aux": ["1W"]}
    # Example for a simple single-timeframe strategy:
    # TIMEFRAMES = {"primary": "1D", "aux": []}

    ASSET_CLASSES: list = ["fx", "crypto", "equities"]
    # Declares which asset classes this strategy is designed for.
    # The runner uses this as the default when --assets is not specified.
    # A strategy that only makes sense on FX would declare: ["fx"]

    @abstractmethod
    def generate_signals(self, context, params: dict) -> pd.DataFrame:
        """
        Generate trading signals for the given context and parameters.

        Parameters
        ----------
        context : BacktestContext
            Provides access to primary and auxiliary timeframe data in a
            lookahead-safe way via context.primary() and context.aux(tf).
        params : dict
            Parameter values drawn from PARAM_SPACE, e.g.
            {"fast": 10, "slow": 50}.

        Returns
        -------
        pd.DataFrame
            Must contain at minimum a "position" column with float values
            in [-1, 1]:
            - 1.0 = full long
            - 0.0 = flat
            - -1.0 = full short

            Optional columns:
            - "position_size" : float in [0, 1]
            - "stop_loss"     : float price level (0 = no stop)

        Requirements
        ------------
        - Use only context.primary() and context.aux() for data access.
        - Never access data beyond what the context exposes.
        - Handle NaN values gracefully (fill or skip).
        - Position values must be in [-1, 1].
        """
        raise NotImplementedError

    @classmethod
    def get_param_space(cls) -> dict:
        """Return the PARAM_SPACE dict for this strategy."""
        return cls.PARAM_SPACE

    @classmethod
    def get_default_params(cls) -> dict:
        """
        Return midpoint values for all parameters.

        Useful for quick testing without optimisation.

        Returns
        -------
        dict
            {param_name: midpoint_value} where midpoint is (lo + hi) / 2,
            rounded to int for "int" params.

        Example
        -------
        >>> params = MACross.get_default_params()
        >>> {"fast": 27, "slow": 110}
        """
        defaults = {}
        for name, (dtype, lo, hi) in cls.PARAM_SPACE.items():
            mid = (lo + hi) / 2
            defaults[name] = int(round(mid)) if dtype == "int" else round(float(mid), 4)
        return defaults
