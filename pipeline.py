"""
StratLab Data Pipeline -- Layer 1

Single entry point: ``python pipeline.py``

Reads config/assets.yaml, checks the catalog for stale or missing data,
fetches only what is needed via the correct source per asset/timeframe,
and stores analysis-ready Parquet files across three data layers
(Bronze -> Silver -> Gold).
"""

import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import yaml
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from infrastructure.data import (
    yfinance_fetcher,
    dukascopy_fetcher,
    normalizer,
    resampler,
    catalog,
)

ASSETS_CONFIG = ROOT / "config" / "assets.yaml"
CATALOG_PATH = ROOT / "catalog.json"
DATA_RAW = ROOT / "data" / "raw"
DATA_SILVER = ROOT / "data" / "silver"
DATA_GOLD = ROOT / "data" / "gold"

# Start dates per source
DUKASCOPY_START = "2016-01-01"   # ~10 years of FX intraday
YFINANCE_DAILY_START = "2010-01-01"
YFINANCE_HOURLY_MAX_DAYS = 700

# Fetch groups: each base fetch produces multiple gold timeframes.
# key = (source, base_interval) -> gold timeframes it produces
FETCH_GROUPS = {
    ("dukascopy", "1h"):  ["1H", "4H"],
    ("yfinance",  "1h"):  ["1H", "4H"],
    ("yfinance",  "1d"):  ["1D", "1W"],
    ("binance",   "1h"):  ["1H", "4H"],
    ("binance",   "1d"):  ["1D", "1W"],
}

# For each gold timeframe + source, which base interval to fetch
TF_TO_BASE = {
    ("dukascopy", "1H"): "1h",
    ("dukascopy", "4H"): "1h",
    ("yfinance",  "1H"): "1h",
    ("yfinance",  "4H"): "1h",
    ("yfinance",  "1D"): "1d",
    ("yfinance",  "1W"): "1d",
    ("binance",   "1H"): "1h",
    ("binance",   "4H"): "1h",
    ("binance",   "1D"): "1d",
    ("binance",   "1W"): "1d",
}

GOLD_TIMEFRAMES = ["1H", "4H", "1D", "1W"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Binance helper
# ---------------------------------------------------------------------------
_binance_client_instance = None


def _get_binance_client():
    global _binance_client_instance
    if _binance_client_instance is None:
        from infrastructure.data.binance_client import get_binance_client
        _binance_client_instance = get_binance_client()
    return _binance_client_instance


def _fetch_binance(ticker, interval):
    from infrastructure.data.binance_client import get_data
    client = _get_binance_client()
    interval_map = {
        "1h": ("1h", 1000),
        "1d": ("1d", 5000),
    }
    binance_interval, lookback = interval_map[interval]
    return get_data(client, ticker, binance_interval, lookback)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_ticker(ticker):
    """Clean ticker for use in filenames."""
    return ticker.replace("=", "_").replace("/", "_").replace("^", "_")


def _save_bronze(df, asset_class, ticker, interval):
    clean = _clean_ticker(ticker)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    bronze_dir = DATA_RAW / asset_class
    bronze_dir.mkdir(parents=True, exist_ok=True)
    filepath = bronze_dir / f"{clean}_{interval}_{ts}.parquet"
    df.to_parquet(filepath, engine="pyarrow")
    return filepath


def _save_silver(df, asset_class, ticker, interval):
    clean = _clean_ticker(ticker)
    silver_dir = DATA_SILVER / asset_class
    silver_dir.mkdir(parents=True, exist_ok=True)
    filepath = silver_dir / f"{clean}_{interval}.parquet"

    if filepath.exists():
        existing = pd.read_parquet(filepath, engine="pyarrow")
        if existing.index.tz is None and df.index.tz is not None:
            existing.index = existing.index.tz_localize("UTC")
        merged = pd.concat([existing, df])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.sort_index()
        df = merged

    df.to_parquet(filepath, engine="pyarrow")
    return filepath, len(df)


def _get_ticker_for_source(asset_info, source):
    """Extract the correct ticker field for a given source."""
    if source == "dukascopy":
        return asset_info["ticker_dukascopy"]
    elif source == "binance":
        return asset_info["ticker_binance"]
    elif source == "yfinance":
        return asset_info["ticker_yfinance"]
    raise ValueError(f"Unknown source: {source}")


# ---------------------------------------------------------------------------
# Core: process one asset
# ---------------------------------------------------------------------------

def _process_asset(asset_class, asset_info, cat):
    """Process a single asset through the pipeline.

    Groups fetches by base interval so that e.g. one Dukascopy 1h fetch
    produces both 1H and 4H gold timeframes, and one yfinance 1d fetch
    produces both 1D and 1W.

    Returns (display_ticker, was_updated, None).
    """
    display = asset_info["ticker_display"]
    sources = asset_info["sources"]  # e.g. {"1H": "dukascopy", "4H": "dukascopy", "1D": "yfinance", "1W": "yfinance"}

    # Check if all gold timeframes are fresh
    all_fresh = all(
        not catalog.is_stale(cat, display, tf)
        for tf in GOLD_TIMEFRAMES
    )
    if all_fresh:
        return (display, False, None)

    print(f"\n  > Processing {display} ({asset_info['name']})")

    # Group needed timeframes by (source, base_interval) to avoid double fetching
    # e.g. {("dukascopy","1h"): ["1H","4H"], ("yfinance","1d"): ["1D","1W"]}
    fetch_plan = {}
    for tf in GOLD_TIMEFRAMES:
        if not catalog.is_stale(cat, display, tf):
            continue
        src = sources[tf]
        base = TF_TO_BASE[(src, tf)]
        key = (src, base)
        if key not in fetch_plan:
            fetch_plan[key] = []
        fetch_plan[key].append(tf)

    if not fetch_plan:
        return (display, False, None)

    # Execute each fetch group
    for (src, base_interval), target_tfs in fetch_plan.items():
        ticker = _get_ticker_for_source(asset_info, src)
        tf_str = "+".join(target_tfs)
        print(f"    [{src}] Fetching {base_interval} -> {tf_str}...", end=" ", flush=True)

        # --- Fetch ---
        if src == "dukascopy":
            raw_df = dukascopy_fetcher.get_data(ticker, DUKASCOPY_START)
        elif src == "yfinance":
            if base_interval == "1h":
                start = YFINANCE_DAILY_START
            else:
                start = YFINANCE_DAILY_START
            yf_interval = {"1h": "1h", "1d": "1d"}[base_interval]
            raw_df = yfinance_fetcher.get_data(ticker, yf_interval, start)
        elif src == "binance":
            raw_df = _fetch_binance(ticker, base_interval)
        else:
            raise ValueError(f"Unknown source: {src}")

        # --- Bronze ---
        _save_bronze(raw_df, asset_class, ticker, base_interval)

        # --- Normalize ---
        norm_df = normalizer.normalize(raw_df, src, display)

        # --- Silver ---
        _, silver_rows = _save_silver(norm_df, asset_class, display, base_interval)
        print(f"{silver_rows} rows in silver")

        # --- Gold: resample to target timeframes ---
        print(f"    Resampling {base_interval} -> {tf_str}...")
        resampled = resampler.resample_to_timeframes(
            norm_df, display, asset_class, base_resolution=base_interval
        )
        resampler.save_gold(resampled, display, str(DATA_GOLD))

        # --- Depth warning ---
        depth_warn = None
        if src == "yfinance" and base_interval == "1h":
            depth_warn = "yfinance 1h limited to ~2 years"

        # --- Update catalog for produced timeframes ---
        for tf_label, tf_df in resampled.items():
            if tf_label in target_tfs:
                start_dt = tf_df.index.min().isoformat()
                end_dt = tf_df.index.max().isoformat()
                catalog.update_catalog(
                    cat, display, asset_class, src, tf_label,
                    start_dt, end_dt, len(tf_df),
                    depth_warning=depth_warn if base_interval == "1h" and src == "yfinance" else None,
                )

    return (display, True, None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*50}")
    print(f"  === StratLab Data Pipeline ===")
    print(f"  {now_utc}")
    print(f"{'='*50}\n")

    with open(ASSETS_CONFIG, "r") as f:
        assets_config = yaml.safe_load(f)

    cat = catalog.get_catalog(str(CATALOG_PATH))

    updated = 0
    skipped = 0
    errors = []

    for asset_class, assets in assets_config.items():
        print(f"\n-- {asset_class.upper()} --")
        for asset_info in assets:
            display = asset_info["ticker_display"]
            try:
                _, was_updated, _ = _process_asset(asset_class, asset_info, cat)
                if was_updated:
                    updated += 1
                else:
                    skipped += 1
                    print(f"  -- Skipping {display} (all timeframes fresh)")
            except Exception as e:
                errors.append((display, str(e)))
                logger.error(f"Failed {display}: {e}", exc_info=True)
                print(f"  x {display} -- {e}")

    catalog.save_catalog(cat, str(CATALOG_PATH))

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n{'='*50}")
    print(f"  === StratLab Pipeline Complete ===")
    print(f"  Updated:  {updated} assets")
    print(f"  Skipped:  {skipped} assets")
    print(f"  Errors:   {len(errors)} asset(s)")
    for ticker, msg in errors:
        print(f"    x {ticker} -- {msg}")
    print(f"  Total time: {mins}m {secs}s")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
