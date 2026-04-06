"""
StratLab Data Pipeline — Layer 1

Single entry point: ``python pipeline.py``

Reads config/assets.yaml, checks the catalog for stale or missing data,
fetches only what is needed, and stores analysis-ready Parquet files
across three data layers (Bronze → Silver → Gold).
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
# Paths — all relative to this file's location (project root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from infrastructure.data import yfinance_fetcher, normalizer, resampler, catalog

ASSETS_CONFIG = ROOT / "config" / "assets.yaml"
CATALOG_PATH = ROOT / "catalog.json"
DATA_RAW = ROOT / "data" / "raw"
DATA_SILVER = ROOT / "data" / "silver"
DATA_GOLD = ROOT / "data" / "gold"

# Base start date for historical data
DEFAULT_START = "2010-01-01"

# Timeframes we track in the catalog
GOLD_TIMEFRAMES = ["1W", "1D", "4H", "1H"]

# Map source → base intervals we need to fetch
# yfinance: fetch 1h and 1d (4H is resampled from 1h, 1W from 1d)
# binance:  fetch 1h and 1d via the existing client
YFINANCE_INTERVALS = ["1h", "1d"]
BINANCE_INTERVALS = ["1h", "1d"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Binance helper — wraps the existing binance_client.py interface
# ---------------------------------------------------------------------------
_binance_client_instance = None


def _get_binance_client():
    """Lazy-initialise the Binance client once per run."""
    global _binance_client_instance
    if _binance_client_instance is None:
        from infrastructure.data.binance_client import get_binance_client
        _binance_client_instance = get_binance_client()
    return _binance_client_instance


def _fetch_binance(ticker, interval):
    """Fetch a single ticker/interval from Binance using the existing client.

    Returns a pandas DataFrame or raises on failure.
    """
    from infrastructure.data.binance_client import get_data

    client = _get_binance_client()

    # Map interval string to Binance constant and lookback days
    interval_map = {
        "1h": ("1h", 1000),
        "1d": ("1d", 5000),
    }
    binance_interval, lookback = interval_map[interval]
    return get_data(client, ticker, binance_interval, lookback)


# ---------------------------------------------------------------------------
# Core pipeline logic
# ---------------------------------------------------------------------------

def _clean_ticker_name(ticker):
    """Remove special characters for filenames (e.g. EURUSD=X → EURUSD_X)."""
    return ticker.replace("=", "_").replace("/", "_").replace("^", "_")


def _save_bronze(df, asset_class, ticker, interval):
    """Save raw fetched data to Bronze layer."""
    clean = _clean_ticker_name(ticker)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    bronze_dir = DATA_RAW / asset_class
    bronze_dir.mkdir(parents=True, exist_ok=True)
    filepath = bronze_dir / f"{clean}_{interval}_{ts}.parquet"
    df.to_parquet(filepath, engine="pyarrow")
    return filepath


def _save_silver(df, asset_class, ticker, interval):
    """Save normalised data to Silver layer, merging with existing."""
    clean = _clean_ticker_name(ticker)
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


def _process_asset(asset_class, asset_info, cat):
    """Process a single asset: fetch → normalise → resample → save.

    Returns a tuple (ticker, was_updated, error_msg).
    """
    ticker = asset_info["ticker"]
    source = asset_info["source"]
    clean = _clean_ticker_name(ticker)

    # Check if all gold timeframes are fresh
    all_fresh = all(
        not catalog.is_stale(cat, clean, tf)
        for tf in GOLD_TIMEFRAMES
    )
    if all_fresh:
        return (ticker, False, None)

    print(f"\n  ▸ Processing {ticker} ({source})")

    # Determine which base intervals to fetch
    intervals = YFINANCE_INTERVALS if source == "yfinance" else BINANCE_INTERVALS

    # Collect normalised dataframes per interval for resampling
    silver_frames = {}

    for interval in intervals:
        print(f"    Fetching {interval}...", end=" ", flush=True)

        # --- Fetch ---
        if source == "yfinance":
            raw_df = yfinance_fetcher.get_data(ticker, interval, DEFAULT_START)
        elif source == "binance":
            raw_df = _fetch_binance(ticker, interval)
        else:
            raise ValueError(f"Unknown source: {source}")

        # --- Bronze ---
        _save_bronze(raw_df, asset_class, ticker, interval)

        # --- Normalise ---
        norm_df = normalizer.normalize(raw_df, source, ticker)

        # --- Silver ---
        silver_path, silver_rows = _save_silver(norm_df, asset_class, ticker, interval)
        print(f"✓ ({silver_rows} rows)")

        silver_frames[interval] = norm_df

    # --- Gold: resample from the highest-resolution data ---
    # Use 1h data if available (preferred), fall back to 1d
    if "1h" in silver_frames:
        base_df = silver_frames["1h"]
    elif "1d" in silver_frames:
        base_df = silver_frames["1d"]
    else:
        raise ValueError(f"No usable interval data for {ticker}")

    print(f"    Resampling to gold timeframes...")
    resampled = resampler.resample_to_timeframes(base_df, clean, asset_class)
    resampler.save_gold(resampled, clean, str(DATA_GOLD))

    # --- Update catalog for every timeframe we produced ---
    for tf_label, tf_df in resampled.items():
        start_dt = tf_df.index.min().isoformat()
        end_dt = tf_df.index.max().isoformat()
        catalog.update_catalog(
            cat, clean, asset_class, source, tf_label,
            start_dt, end_dt, len(tf_df),
        )

    return (ticker, True, None)


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

    # Load config
    with open(ASSETS_CONFIG, "r") as f:
        assets_config = yaml.safe_load(f)

    # Load catalog
    cat = catalog.get_catalog(str(CATALOG_PATH))

    updated = 0
    skipped = 0
    errors = []

    for asset_class, assets in assets_config.items():
        print(f"\n── {asset_class.upper()} ──")
        for asset_info in assets:
            ticker = asset_info["ticker"]
            try:
                _, was_updated, err = _process_asset(asset_class, asset_info, cat)
                if was_updated:
                    updated += 1
                else:
                    skipped += 1
                    print(f"  — Skipping {ticker} (all timeframes fresh)")
            except Exception as e:
                errors.append((ticker, str(e)))
                logger.error(f"Failed {ticker}: {e}")
                print(f"  ✗ {ticker} — {e}")

    # Save catalog
    catalog.save_catalog(cat, str(CATALOG_PATH))

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"  === Pipeline complete ===")
    print(f"  Assets updated:  {updated}")
    print(f"  Assets skipped:  {skipped}")
    print(f"  Errors:          {len(errors)}")
    for ticker, msg in errors:
        print(f"    ✗ {ticker} — {msg}")
    print(f"  Total time: {elapsed:.0f}s")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
