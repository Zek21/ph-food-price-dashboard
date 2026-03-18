"""
Daily auto-updater for the Philippines Food Price Dashboard.
Downloads the latest WFP data, retrains the model, and rebuilds the dashboard.

Usage:
  python daily_update.py          # Run manually
  python daily_update.py --force  # Force update even if data hasn't changed
  python daily_update.py --dry-run  # Download and validate only, skip retraining

Scheduling (Windows Task Scheduler):
  Run: schtasks /create /tn "FoodPriceDashboard" /tr "python D:\\ML\\Website\\daily_update.py" /sc daily /st 06:00
  Delete: schtasks /delete /tn "FoodPriceDashboard" /f
"""

import hashlib
import logging
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd  # signed: beta

# ─── Configuration ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ML_DIR = BASE_DIR.parent  # D:/ML
WFP_CSV = ML_DIR / "WFP" / "wfp_food_prices_phl_latest.csv"
HASH_FILE = BASE_DIR / ".last_data_hash"
LOG_FILE = BASE_DIR / "update_log.txt"

DATA_URL = (
    "https://data.humdata.org/dataset/ea251823-8694-47b4-82d0-7d27f00e8aba"
    "/resource/9a842d72-0d7d-4922-ad0e-eb8106c1ab0e/download/wfp_food_prices_phl.csv"
)

# ─── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("daily_update")


def file_hash(path: Path) -> str:
    """SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def download_latest(max_retries: int = 3) -> bool:
    """Download the latest WFP CSV with exponential backoff retry.

    Returns True if data changed.
    """
    # signed: beta
    log.info("Downloading latest WFP data from Humanitarian Data Exchange...")
    WFP_CSV.parent.mkdir(parents=True, exist_ok=True)

    old_hash = ""
    if WFP_CSV.exists():
        old_hash = file_hash(WFP_CSV)

    tmp = WFP_CSV.with_suffix(".tmp")

    # Retry with exponential backoff: 2s, 4s, 8s  # signed: beta
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            log.info("  Download attempt %d/%d...", attempt, max_retries)
            urllib.request.urlretrieve(DATA_URL, tmp)
            last_error = None
            break
        except Exception as e:
            last_error = e
            if tmp.exists():
                tmp.unlink()
            if attempt < max_retries:
                delay = 2 ** attempt  # 2s, 4s, 8s
                log.warning("  Download failed (attempt %d): %s. Retrying in %ds...", attempt, e, delay)
                time.sleep(delay)
            else:
                log.error("Download failed after %d attempts: %s", max_retries, e)

    if last_error is not None:
        return False

    new_hash = file_hash(tmp)
    if new_hash == old_hash:
        log.info("Data unchanged (hash match). Skipping retrain.")
        tmp.unlink()
        return False

    # Atomic replace
    if WFP_CSV.exists():
        WFP_CSV.unlink()
    tmp.rename(WFP_CSV)

    log.info("New data downloaded (%s bytes, hash: %s...)", WFP_CSV.stat().st_size, new_hash[:12])

    # Save hash
    HASH_FILE.write_text(new_hash)
    return True


def check_data_quality() -> bool:
    """Validate downloaded CSV data quality before retraining.

    Checks: file exists, minimum row count, required columns, null ratios,
    date range sanity, price value sanity.
    Returns True if data passes all checks.
    """
    # signed: beta
    if not WFP_CSV.exists():
        log.error("Data quality check FAILED: CSV file not found at %s", WFP_CSV)
        return False

    try:
        df = pd.read_csv(WFP_CSV)
    except Exception as e:
        log.error("Data quality check FAILED: Cannot parse CSV: %s", e)
        return False

    # Required columns
    required = ["date", "price", "commodity", "admin1", "pricetype"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error("Data quality check FAILED: Missing columns: %s", missing)
        return False

    # Minimum row count
    if len(df) < 1000:
        log.error("Data quality check FAILED: Only %d rows (need >= 1000)", len(df))
        return False

    # Null ratio check (allow up to 10% nulls in critical columns)
    for col in ["date", "price", "commodity"]:
        null_pct = df[col].isnull().sum() / len(df) * 100
        if null_pct > 10:
            log.error("Data quality check FAILED: Column '%s' has %.1f%% nulls (max 10%%)", col, null_pct)
            return False

    # Date range sanity — should span at least 2 years
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    valid_dates = df["_date"].dropna()
    if len(valid_dates) > 0:
        date_range_years = (valid_dates.max() - valid_dates.min()).days / 365.25
        if date_range_years < 2:
            log.error("Data quality check FAILED: Date range only %.1f years (need >= 2)", date_range_years)
            return False
        log.info("  Data quality: %d rows, %d commodities, date range: %s to %s (%.1f years)",
                 len(df), df["commodity"].nunique(),
                 valid_dates.min().strftime("%Y-%m-%d"),
                 valid_dates.max().strftime("%Y-%m-%d"),
                 date_range_years)

    # Price sanity — at least 90% should be positive numbers
    prices = pd.to_numeric(df["price"], errors="coerce")
    valid_prices = prices.dropna()
    positive_pct = (valid_prices > 0).sum() / len(valid_prices) * 100 if len(valid_prices) > 0 else 0
    if positive_pct < 90:
        log.error("Data quality check FAILED: Only %.1f%% positive prices (need >= 90%%)", positive_pct)
        return False

    log.info("  Data quality checks PASSED.")
    return True


def retrain_model():
    """Run retrain_model.py to rebuild the model and comparison JSON."""
    log.info("Retraining model...")
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "retrain_model.py")],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        log.error("retrain_model.py failed:\n%s\n%s", result.stdout[-2000:], result.stderr[-2000:])
        raise RuntimeError("Model retraining failed")
    log.info("Model retrained successfully.")
    # Log last few lines of output
    for line in result.stdout.strip().split("\n")[-5:]:
        log.info("  %s", line)


def rebuild_dashboard():
    """Run build_dashboard.py to regenerate dashboard_data.json."""
    log.info("Rebuilding dashboard data...")
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "build_dashboard.py")],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        log.error("build_dashboard.py failed:\n%s\n%s", result.stdout[-2000:], result.stderr[-2000:])
        raise RuntimeError("Dashboard rebuild failed")
    log.info("Dashboard rebuilt successfully.")


def validate_outputs():
    """Run consistency checks so generated counters stay accurate."""
    log.info("Validating generated dashboard outputs...")
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "validate_dashboard_outputs.py")],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        log.error(
            "validate_dashboard_outputs.py failed:\n%s\n%s",
            result.stdout[-2000:],
            result.stderr[-2000:],
        )
        raise RuntimeError("Generated dashboard validation failed")
    output = result.stdout.strip() or "Validation passed."
    log.info(output)


def main():
    force = "--force" in sys.argv
    dry_run = "--dry-run" in sys.argv  # signed: beta
    log.info("=" * 50)
    log.info("Daily update started%s%s", " (forced)" if force else "", " (DRY RUN)" if dry_run else "")

    changed = download_latest()

    if changed or force:
        # Data quality gate — validate before retraining  # signed: beta
        if not check_data_quality():
            log.error("DATA QUALITY CHECK FAILED — aborting retraining to protect model integrity.")
            log.info("Fix the data issues and re-run with --force.")
            sys.exit(1)

        if dry_run:
            log.info("DRY RUN: Data downloaded and validated. Skipping retraining and dashboard rebuild.")
            log.info("Done (dry run).\n")
            return

        retrain_model()
        rebuild_dashboard()
        validate_outputs()
        log.info("Dashboard data updated at %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    else:
        log.info("No update needed -- data unchanged since last check.")

    log.info("Done.\n")


if __name__ == "__main__":
    main()
