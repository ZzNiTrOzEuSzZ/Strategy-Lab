"""
strategies/bb_breakout.py
--------------------------
Bollinger Band Breakout Strategy — Denislav Dantev

COMPLETE LOGIC:

4H SETUP (checked at every 4H bar):
  Condition 1 — Directional volatility breakout:
    - Last 2 closed 4H candles are BOTH green (close > open) for a long setup
    - Last 2 closed 4H candles are BOTH red (close < open) for a short setup
    - BOTH candles have a range (high - low) greater than
      breakout_atr_mult × ATR(4H, atr_period)
    - This captures "market was quiet, then sudden directional explosion"

  Condition 2 — Bollinger Bands expanding:
    - bb_width(4H, bb_period, bb_std) > bb_width(4H, bb_period, bb_std).shift(1)

  Condition 3 — SMA sloping in direction of move:
    - ma_slope(4H, ma_period) > 0 for long setup
    - ma_slope(4H, ma_period) < 0 for short setup

  All three conditions must be true simultaneously on the same 4H bar.

1H ENTRY (state machine — tracked bar by bar):
  Once a 4H setup fires, the setup becomes "active".
  At each subsequent 1H bar, check in this exact order:

  EXPIRY CHECK (invalidates setup — check these first):
    E1 — Too many bars: bars_since_setup > max_1h_bars → invalidate
    E2 — Pullback candle too large:
         candle_range(1H current bar) > pullback_atr_mult × ATR(1H, atr_period)
         → invalidate (price spiking not drifting)
    E3 — Price overshot SMA:
         For long setup: close < sma - (sma × pullback_bps / 10000)
         → price has gone more than pullback_bps below the SMA → invalidate
         For short setup: close > sma + (sma × pullback_bps / 10000)
         → price has gone more than pullback_bps above the SMA → invalidate

  ENTRY CHECK (only if setup still active after expiry checks):
    P1 — Price in pullback zone:
         abs(close - sma) / sma × 10000 <= pullback_bps
         AND for long: close > sma (still above, just touching from above)
         AND for short: close < sma (still below, just touching from below)
    P2 — Reversal pattern fired: any_reversal_pattern() returns 1 (long)
         or -1 (short) matching the setup direction

  If P1 AND P2 are both true: ENTRY
    - position = 1.0 (long) or -1.0 (short)
    - Entry executes at NEXT bar open (engine shifts position by 1 bar)
    - stop_loss = last_swing_low(1H, swing_lookback) for longs
    - stop_loss = last_swing_high(1H, swing_lookback) for shorts
    - After entry, deactivate the setup (one trade per setup)

PARAM_SPACE:
  bb_period:          int,   10, 40     Bollinger Band period
  bb_std:             float, 1.5, 3.0   BB standard deviation multiplier
  atr_period:         int,   5,  20     ATR period (used on both 4H and 1H)
  breakout_atr_mult:  float, 1.2, 3.0   4H candle must be > this × ATR to count as big
  ma_period:          int,   10, 30     SMA period (same on both 4H and 1H)
  pullback_bps:       int,   5,  30     Basis points tolerance for SMA touch AND overshoot
  max_1h_bars:        int,   12, 48     Max 1H bars to wait for entry after 4H setup
  pullback_atr_mult:  float, 0.5, 2.0   Max 1H candle size during pullback phase
  swing_lookback:     int,   5,  20     ZigZag lookback for stop placement
  pattern_wick:       float, 1.5, 3.0   Pin bar wick-to-body ratio minimum
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from infrastructure.backtester.indicators import (
    bollinger_bands, bb_width, ma_slope, sma,
    atr, candle_range, any_reversal_pattern,
)
from infrastructure.backtester.market_structure import (
    last_swing_low, last_swing_high,
)


class BBBreakout(BaseStrategy):

    NAME = "Bollinger Band Breakout"
    DESCRIPTION = (
        "4H volatility breakout with expanding BBs and directional SMA. "
        "1H entry on pullback to SMA confirmed by reversal pattern. "
        "Stop at previous 1H swing high/low. Target 1:1."
    )

    TIMEFRAMES = {
        "primary": "1H",
        "aux":     ["4H"],
    }

    ASSET_CLASSES = ["fx", "crypto"]

    PARAM_SPACE = {
        "bb_period":         ("int",   10, 40),
        "bb_std":            ("float", 1.5, 3.0),
        "atr_period":        ("int",   5,  20),
        "breakout_atr_mult": ("float", 1.2, 3.0),
        "ma_period":         ("int",   10, 30),
        "pullback_bps":      ("int",   5,  30),
        "max_1h_bars":       ("int",   12, 48),
        "pullback_atr_mult": ("float", 0.5, 2.0),
        "swing_lookback":    ("int",   5,  20),
        "pattern_wick":      ("float", 1.5, 3.0),
    }

    def generate_signals(self, context, params: dict) -> pd.DataFrame:
        """
        Generate signals using the two-timeframe BB breakout logic.

        The 4H conditions are computed vectorially.
        The 1H state machine (setup tracking, expiry, entry) runs bar by bar.
        This is the only loop in the strategy — it is necessary because the
        state (is setup active? how many bars have passed?) cannot be expressed
        as a vectorized pandas operation.

        Returns a DataFrame with columns:
            position   : float in [-1, 1]
            stop_loss  : float price level (0 = no stop)
        """
        h4 = context.aux("4H")
        h1 = context.primary()

        # ── Pre-compute all 4H indicators ──────────────────────────────────
        h4_atr      = atr(h4, params["atr_period"])
        h4_range    = candle_range(h4)
        h4_bw       = bb_width(h4, params["bb_period"], params["bb_std"])
        h4_slope    = ma_slope(h4, params["ma_period"])
        h4_sma      = sma(h4, params["ma_period"])

        # Candle direction
        h4_green = (h4["close"] > h4["open"])
        h4_red   = (h4["close"] < h4["open"])

        # Both last 2 candles big AND same direction
        big = h4_range > params["breakout_atr_mult"] * h4_atr
        two_big_green = big & big.shift(1) & h4_green & h4_green.shift(1)
        two_big_red   = big & big.shift(1) & h4_red   & h4_red.shift(1)

        # BB expanding
        bb_exp = h4_bw > h4_bw.shift(1)

        # Full 4H setup conditions
        h4_long  = two_big_green & bb_exp & (h4_slope > 0)
        h4_short = two_big_red   & bb_exp & (h4_slope < 0)

        # ── Pre-compute all 1H indicators ───────────────────────────────────
        h1_atr     = atr(h1, params["atr_period"])
        h1_range   = candle_range(h1)
        h1_sma     = sma(h1, params["ma_period"])
        h1_pattern = any_reversal_pattern(h1, wick_ratio=params["pattern_wick"])
        h1_swl     = last_swing_low(h1,  params["swing_lookback"])
        h1_swh     = last_swing_high(h1, params["swing_lookback"])

        # Align 4H signals to 1H index (forward-fill — last closed 4H bar)
        # A new setup can only be detected on a newly closed 4H bar.
        # We use diff() to find the exact 1H bars where a 4H setup FIRST fires.
        h4_long_1h  = h4_long.reindex(h1.index,  method="ffill").fillna(False)
        h4_short_1h = h4_short.reindex(h1.index, method="ffill").fillna(False)

        # Detect the 1H bar where a 4H setup fires for the first time
        # (rising edge — False→True transition)
        long_setup_fires  = h4_long_1h  & ~h4_long_1h.shift(1).fillna(False)
        short_setup_fires = h4_short_1h & ~h4_short_1h.shift(1).fillna(False)

        # ── State machine — bar by bar ──────────────────────────────────────
        n = len(h1)
        position   = np.zeros(n)
        stop_loss  = np.zeros(n)

        # State variables
        setup_active    = False   # is a setup currently live?
        setup_direction = 0       # 1 = long setup, -1 = short setup
        bars_since      = 0       # 1H bars elapsed since setup fired

        for i in range(1, n):

            # ── Check if a new 4H setup fires at this bar ──────────────────
            # Only accept a new setup if no setup is currently active
            if not setup_active:
                if long_setup_fires.iloc[i]:
                    setup_active    = True
                    setup_direction = 1
                    bars_since      = 0
                elif short_setup_fires.iloc[i]:
                    setup_active    = True
                    setup_direction = -1
                    bars_since      = 0

            if not setup_active:
                continue

            bars_since += 1
            close  = h1["close"].iloc[i]
            s_ma   = h1_sma.iloc[i]
            h1_rng = h1_range.iloc[i]
            h1_at  = h1_atr.iloc[i]

            # Skip bar if indicators not yet available
            if pd.isna(s_ma) or pd.isna(h1_at) or s_ma == 0:
                continue

            bps_from_sma = abs(close - s_ma) / s_ma * 10000

            # ── Expiry checks ──────────────────────────────────────────────

            # E1: too many bars
            if bars_since > params["max_1h_bars"]:
                setup_active = False
                continue

            # E2: pullback candle too large (spiking not drifting)
            if h1_rng > params["pullback_atr_mult"] * h1_at:
                setup_active = False
                continue

            # E3: price overshot SMA (gone too far past it)
            if setup_direction == 1:
                # Long setup: price should be above SMA touching from above
                # Invalidate if price drops more than pullback_bps below SMA
                if close < s_ma - (s_ma * params["pullback_bps"] / 10000):
                    setup_active = False
                    continue
            else:
                # Short setup: price should be below SMA touching from below
                # Invalidate if price rises more than pullback_bps above SMA
                if close > s_ma + (s_ma * params["pullback_bps"] / 10000):
                    setup_active = False
                    continue

            # ── Entry checks ───────────────────────────────────────────────

            # P1: price in pullback zone (within pullback_bps of SMA,
            #     on the correct side)
            in_zone = bps_from_sma <= params["pullback_bps"]
            correct_side = (
                (setup_direction == 1  and close > s_ma) or
                (setup_direction == -1 and close < s_ma)
            )

            # P2: reversal pattern matching setup direction
            pat = h1_pattern.iloc[i]
            pattern_ok = (
                (setup_direction == 1  and pat == 1) or
                (setup_direction == -1 and pat == -1)
            )

            if in_zone and correct_side and pattern_ok:
                # Entry confirmed
                position[i]  = float(setup_direction)

                # Stop: previous swing low (longs) or swing high (shorts)
                if setup_direction == 1:
                    sl = h1_swl.iloc[i]
                else:
                    sl = h1_swh.iloc[i]

                stop_loss[i] = float(sl) if not pd.isna(sl) else 0.0

                # Deactivate setup — one trade per setup
                setup_active = False

        # ── Build output DataFrame ──────────────────────────────────────────
        out = h1[[]].copy()   # empty DataFrame with h1 index
        out["position"]  = position
        out["stop_loss"] = stop_loss
        return out
