"""Read private forecasts in memory; emit only counts, hashes and scan outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
PRIVATE = Path(r"D:\ML\AR20260720_private_20260928_rc6\predictions_current.json")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def scan_bytes(data, private_hash, series):
    assert sha(data) != private_hash, "Private artifact bytes found"
    text = data.decode("utf-8-sig", errors="replace")
    # Detect JSON date/value sets even if renamed or stripped of commodity keys.
    pairs = {(month, float(value)) for month, value in re.findall(
        r'"(\d{4}-\d{2})"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', text)}
    for row in series:
        assert len(pairs.intersection(row.items())) < 3, "Private date/value series found"
    # Also detect a full numeric forward vector in a table, list or prose export.
    numbers = [float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", text)]
    for row in series:
        values = list(row.values())
        for start, number in enumerate(numbers):
            if number == values[0]:
                assert numbers[start:start + len(values)] != values, "Private numeric vector found"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=ROOT / "gpu_driver_evidence/rc6-audit-20260928.zip")
    parser.add_argument("--output", type=Path, default=EVIDENCE / "campaign_payload_scan.json")
    args = parser.parse_args()
    private_bytes = PRIVATE.read_bytes()
    private_hash = sha(private_bytes)
    private = json.loads(private_bytes)
    audit = json.loads((EVIDENCE / "private_prediction_audit.json").read_bytes())
    assert private_hash == audit["private_sha256"]
    series = list(private["forecasts"].values())
    assert len(series) == 59 and sum(map(len, series)) == 1062
    assert private["publication_gate"]["status"] == "withheld_failed_validation"
    # A synthetic canary confirms detection without exposing a private value.
    try:
        scan_bytes(b'{"2030-01": 11.11, "2030-02": 22.22, "2030-03": 33.33}',
                   "not-a-hash", [{"2030-01": 11.11, "2030-02": 22.22, "2030-03": 33.33}])
    except AssertionError:
        pass
    else:
        raise AssertionError("Scanner canary failed")
    paths = git("ls-files", "-z").decode().rstrip("\0").split("\0")
    staged = git("diff", "--cached", "--name-only", "-z").decode().rstrip("\0").split("\0")
    count = 0
    for name in paths:
        assert Path(name).name != "predictions_current.json", "Private artifact name in index"
        for data in (git("show", ":" + name), (ROOT / name).read_bytes()):
            try:
                scan_bytes(data, private_hash, series)
            except AssertionError as exc:
                raise AssertionError(f"{name}: {exc}") from None
        count += 1
    # Scan public evidence files including untracked receipts; skip ignored local
    # report/readback files, which are scanned separately before final delivery.
    public_evidence = list(EVIDENCE.glob("*.json")) + list(EVIDENCE.glob("*.txt"))
    for path in public_evidence:
        scan_bytes(path.read_bytes(), private_hash, series)
    payload = json.loads((EVIDENCE / "release_payload.json").read_bytes())
    assert len(payload["files"]) == 45
    assert "blog_amd_gpu_section.html" not in payload["files"]
    for name in payload["files"]:
        scan_bytes((ROOT / name).read_bytes(), private_hash, series)
    with zipfile.ZipFile(args.bundle) as bundle:
        assert len(bundle.namelist()) == len(set(bundle.namelist())) == 45
        assert set(bundle.namelist()) == set(payload["files"])
        for name in bundle.namelist():
            data = bundle.read(name)
            assert data == (ROOT / name).read_bytes(), name
            scan_bytes(data, private_hash, series)
        checks = bundle.read("SHA256SUMS-rc6.txt").decode().splitlines()
        for line in checks:
            expected, name = line.split("  ", 1)
            assert sha(bundle.read(name)) == expected, name
    result = {
        "ticket": "AR-20260720-ML-NATIVE-GPU-DRIVER-RELEASE-PREDICTIONS",
        "private_sha256": private_hash, "private_models": 59, "private_points": 1062,
        "tracked_worktree_and_index_files_scanned": count,
        "staged_files_scanned": len([x for x in staged if x]),
        "public_evidence_files_scanned": len(public_evidence),
        "payload_files_scanned": 45, "zip_members_scanned": 45,
        "checksums_verified": len(checks), "bundle_sha256": sha(args.bundle.read_bytes()),
        "private_artifact_found": False, "current_forecast_value_set_found": False,
        "scanner_canary_verified": True,
        "method": "Exact private artifact hash; date/value series (3 matching pairs); complete numeric vectors; index blobs, working files, evidence, allowlist and every ZIP member.",
        "scope_limit": "Current artifact/value sets only. Isolated rounded numbers can coincide with historical observations; arbitrary re-encoding is not a provable absence. Historical tags/archives and unrelated classical forecasts are preserved.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
