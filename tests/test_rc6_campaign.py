"""Release regeneration must preserve editorial copy and the established scope."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("gpu_driver_evidence/rerun_20260928_rc6")


@pytest.fixture
def release_copy(tmp_path):
    payload = json.loads((ROOT / EVIDENCE / "release_payload.json").read_text())
    for name in payload["files"]:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    return tmp_path, payload


def run_refresh(root):
    return subprocess.run(
        [sys.executable, str(root / EVIDENCE / "build_release_receipts.py"), "--campaign-only"],
        capture_output=True, text=True, encoding="utf-8", cwd=root,
    )


def test_refresh_preserves_campaign_and_ignores_unlisted_evidence(release_copy):
    root, payload = release_copy
    campaign = root / "INVESTOR_CAMPAIGN_RC6.md"
    campaign.write_text("Editorial revision must survive regeneration.\n", encoding="utf-8")
    before = campaign.read_bytes()
    (root / EVIDENCE / "unexpected_private.json").write_text('{"forecasts":{"synthetic":[123]}}')
    result = run_refresh(root)
    assert result.returncode == 0, result.stderr
    assert campaign.read_bytes() == before
    assert json.loads((root / EVIDENCE / "release_payload.json").read_text()) == payload
    checksums = (root / "SHA256SUMS-rc6.txt").read_text()
    assert "unexpected_private.json" not in checksums
    assert "blog_amd_gpu_section.html" not in checksums
    for line in checksums.splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected


def test_refresh_rejects_changed_authoritative_receipt(release_copy):
    root, _ = release_copy
    (root / EVIDENCE / "validation_current.json").write_text("{}")
    before = (root / "SHA256SUMS-rc6.txt").read_bytes()
    result = run_refresh(root)
    assert result.returncode != 0
    assert "Receipt changed" in result.stderr
    assert (root / "SHA256SUMS-rc6.txt").read_bytes() == before


def test_refresh_rejects_payload_scope_drift(release_copy):
    root, payload = release_copy
    payload["files"].append("blog_amd_gpu_section.html")
    (root / EVIDENCE / "release_payload.json").write_text(json.dumps(payload))
    result = run_refresh(root)
    assert result.returncode != 0
    assert "Release scope changed" in result.stderr
