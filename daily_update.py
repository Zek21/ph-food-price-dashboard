"""
Daily auto-updater for the Philippines Food Price Dashboard.
Downloads the latest WFP data, retrains the model, and rebuilds the dashboard.

Usage:
  python daily_update.py          # Run manually
  python daily_update.py --force  # Force update even if data hasn't changed

Scheduling (Windows Task Scheduler):
  Run: schtasks /create /tn "FoodPriceDashboard" /tr "python daily_update.py" /sc daily /st 06:00
  Delete: schtasks /delete /tn "FoodPriceDashboard" /f

Scheduling (Linux/macOS cron):
  Add via crontab -e:  0 6 * * * cd /path/to/repo && python daily_update.py

Environment variables (override defaults):
  WFP_DATA_PATH  — path to the WFP CSV file (default: ../WFP/wfp_food_prices_phl_latest.csv)
"""

import hashlib
import logging
import os
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent
_default_wfp_csv = str(BASE_DIR.parent / "WFP" / "wfp_food_prices_phl_latest.csv")
WFP_CSV = Path(os.environ.get("WFP_DATA_PATH", _default_wfp_csv))
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


def download_latest() -> bool:
    """Download the latest WFP CSV. Returns True if data changed."""
    log.info("Downloading latest WFP data from Humanitarian Data Exchange...")
    WFP_CSV.parent.mkdir(parents=True, exist_ok=True)

    old_hash = ""
    if WFP_CSV.exists():
        old_hash = file_hash(WFP_CSV)

    tmp = WFP_CSV.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(DATA_URL, tmp)
    except Exception as e:
        log.error("Download failed: %s", e)
        if tmp.exists():
            tmp.unlink()
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


def retrain_model():
    """Run retrain_model.py to rebuild the model and comparison JSON."""
    log.info("Retraining model...")
    env = os.environ.copy()
    env["WFP_DATA_PATH"] = str(WFP_CSV)
    env["OUTPUT_PATH"] = str(BASE_DIR / "model_comparison.json")
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "retrain_model.py")],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    if result.returncode != 0:
        log.error("retrain_model.py failed:\n%s\n%s", result.stdout[-2000:], result.stderr[-2000:])
        raise RuntimeError("Model retraining failed")
    log.info("Model retrained successfully.")
    # Log last few lines of output
    for line in result.stdout.strip().split("\n")[-5:]:
        log.info("  %s", line)


def main():
    force = "--force" in sys.argv
    log.info("=" * 50)
    log.info("Daily update started%s", " (forced)" if force else "")

    changed = download_latest()

    if changed or force:
        retrain_model()
        log.info("Dashboard data updated at %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    else:
        log.info("No update needed — data unchanged since last check.")

    log.info("Done.\n")


if __name__ == "__main__":
    main()
