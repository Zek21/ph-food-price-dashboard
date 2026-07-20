"""
Tests for daily_update.py — Daily auto-updater for the Philippines Food Price Dashboard.

Tests cover:
  - file_hash() SHA-256 computation
  - download_latest() hash-based change detection
  - retrain_model() subprocess invocation
  - rebuild_dashboard() subprocess invocation
  - validate_outputs() subprocess invocation
  - main() orchestration with --force flag
  - DATA_URL construction
  - Error handling (download failures, subprocess failures)
  - Hash file I/O
"""
# signed: delta

import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Import daily_update module
# ---------------------------------------------------------------------------
DAILY_UPDATE_PATH = Path(__file__).resolve().parent.parent / "daily_update.py"

# We can't import daily_update directly as a module because it's in D:\ML\Website
# and uses relative paths.  Instead we add its parent to sys.path temporarily.
_website_dir = str(DAILY_UPDATE_PATH.parent)


@pytest.fixture(autouse=True)
def _add_website_to_path():
    """Temporarily add the Website directory to sys.path for imports."""
    if _website_dir not in sys.path:
        sys.path.insert(0, _website_dir)
    yield
    if _website_dir in sys.path:
        sys.path.remove(_website_dir)


def _import_daily_update():
    """Import daily_update module fresh."""
    if _website_dir not in sys.path:
        sys.path.insert(0, _website_dir)
    import importlib
    if "daily_update" in sys.modules:
        return importlib.reload(sys.modules["daily_update"])
    return importlib.import_module("daily_update")


# ===================================================================
# TEST CLASS: file_hash Function
# ===================================================================

class TestFileHash:
    """Test the SHA-256 file hashing function."""

    def test_hash_known_content(self, tmp_path):
        """Hash of known content should match expected SHA-256."""
        du = _import_daily_update()
        f = tmp_path / "test.txt"
        content = b"Hello, World!"
        f.write_bytes(content)
        result = du.file_hash(f)
        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_hash_empty_file(self, tmp_path):
        """Hash of empty file should be SHA-256 of empty bytes."""
        du = _import_daily_update()
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        result = du.file_hash(f)
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_hash_deterministic(self, tmp_path):
        """Same file content should always produce the same hash."""
        du = _import_daily_update()
        f = tmp_path / "det.txt"
        f.write_bytes(b"deterministic content test")
        h1 = du.file_hash(f)
        h2 = du.file_hash(f)
        assert h1 == h2

    def test_hash_different_content(self, tmp_path):
        """Different content should produce different hashes."""
        du = _import_daily_update()
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert du.file_hash(f1) != du.file_hash(f2)

    def test_hash_large_file(self, tmp_path):
        """Should handle files larger than the 64KB chunk size."""
        du = _import_daily_update()
        f = tmp_path / "large.bin"
        data = b"x" * (1 << 17)  # 128 KB
        f.write_bytes(data)
        result = du.file_hash(f)
        expected = hashlib.sha256(data).hexdigest()
        assert result == expected

    def test_hash_returns_hex_string(self, tmp_path):
        """Hash should be a 64-character hex string."""
        du = _import_daily_update()
        f = tmp_path / "hex.txt"
        f.write_bytes(b"test")
        result = du.file_hash(f)
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


# ===================================================================
# TEST CLASS: download_latest Function
# ===================================================================

class TestDownloadLatest:
    """Test the download and change detection logic."""

    def test_new_download_returns_true(self, tmp_path):
        """First download (no existing file) should return True."""
        du = _import_daily_update()
        csv_path = tmp_path / "WFP" / "data.csv"
        with patch.object(du, "WFP_CSV", csv_path), \
             patch.object(du, "HASH_FILE", tmp_path / ".hash"), \
             patch("urllib.request.urlretrieve") as mock_dl:
            # urlretrieve creates the temp file
            def fake_download(url, dest):
                Path(dest).write_text("date,price\n2024-01-01,50\n")
            mock_dl.side_effect = fake_download
            result = du.download_latest()
            assert result is True
            assert csv_path.exists()

    def test_unchanged_data_returns_false(self, tmp_path):
        """If downloaded data matches existing, should return False."""
        du = _import_daily_update()
        csv_path = tmp_path / "WFP" / "data.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        content = "date,price\n2024-01-01,50\n"
        csv_path.write_text(content)
        with patch.object(du, "WFP_CSV", csv_path), \
             patch.object(du, "HASH_FILE", tmp_path / ".hash"), \
             patch("urllib.request.urlretrieve") as mock_dl:
            def fake_download(url, dest):
                Path(dest).write_text(content)  # same content
            mock_dl.side_effect = fake_download
            result = du.download_latest()
            assert result is False

    def test_changed_data_returns_true(self, tmp_path):
        """If downloaded data differs from existing, should return True."""
        du = _import_daily_update()
        csv_path = tmp_path / "WFP" / "data.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("old content")
        with patch.object(du, "WFP_CSV", csv_path), \
             patch.object(du, "HASH_FILE", tmp_path / ".hash"), \
             patch("urllib.request.urlretrieve") as mock_dl:
            def fake_download(url, dest):
                Path(dest).write_text("new content")
            mock_dl.side_effect = fake_download
            result = du.download_latest()
            assert result is True

    def test_download_failure_returns_false(self, tmp_path):
        """Network error during download should return False."""
        du = _import_daily_update()
        csv_path = tmp_path / "WFP" / "data.csv"
        with patch.object(du, "WFP_CSV", csv_path), \
             patch.object(du, "HASH_FILE", tmp_path / ".hash"), \
             patch("urllib.request.urlretrieve", side_effect=ConnectionError("Network down")), \
             patch("time.sleep"):  # skip retry delays
            result = du.download_latest()
            assert result is False

    def test_hash_file_written_on_change(self, tmp_path):
        """HASH_FILE should be updated when data changes."""
        du = _import_daily_update()
        csv_path = tmp_path / "WFP" / "data.csv"
        hash_file = tmp_path / ".hash"
        with patch.object(du, "WFP_CSV", csv_path), \
             patch.object(du, "HASH_FILE", hash_file), \
             patch("urllib.request.urlretrieve") as mock_dl:
            def fake_download(url, dest):
                Path(dest).write_text("new data content")
            mock_dl.side_effect = fake_download
            du.download_latest()
            assert hash_file.exists()
            saved_hash = hash_file.read_text().strip()
            assert len(saved_hash) == 64  # SHA-256 hex length


# ===================================================================
# TEST CLASS: Subprocess Wrappers
# ===================================================================

class TestSubprocessWrappers:
    """Test retrain_model(), rebuild_dashboard(), validate_outputs() wrappers."""

    def test_retrain_model_calls_subprocess(self):
        """retrain_model() should invoke retrain_model.py via subprocess."""
        du = _import_daily_update()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
            du.retrain_model()
            mock_run.assert_called_once()
            args = mock_run.call_args
            cmd = args[0][0] if args[0] else args[1].get("args", [])
            # Should invoke python retrain_model.py
            assert "retrain_model.py" in str(cmd[-1])

    def test_retrain_model_passes_canonical_wfp_path(self):
        """The child trainer must receive the updater's canonical WFP CSV."""
        du = _import_daily_update()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
            du.retrain_model()
            child_env = mock_run.call_args.kwargs["env"]
            assert child_env["WFP_DATA_PATH"] == str(du.WFP_CSV.resolve())
            assert mock_run.call_args.kwargs["timeout"] == du.RETRAIN_TIMEOUT_SECONDS

    def test_retrain_model_raises_on_failure(self):
        """retrain_model() should raise RuntimeError on non-zero exit."""
        du = _import_daily_update()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="error", stderr="crash")
            with pytest.raises(RuntimeError, match="retraining failed"):
                du.retrain_model()

    def test_rebuild_dashboard_calls_subprocess(self):
        """rebuild_dashboard() should invoke build_dashboard.py."""
        du = _import_daily_update()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
            du.rebuild_dashboard()
            mock_run.assert_called_once()
            args = mock_run.call_args
            cmd = args[0][0] if args[0] else args[1].get("args", [])
            assert "build_dashboard.py" in str(cmd[-1])

    def test_rebuild_dashboard_raises_on_failure(self):
        """rebuild_dashboard() should raise RuntimeError on non-zero exit."""
        du = _import_daily_update()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="error", stderr="crash")
            with pytest.raises(RuntimeError, match="rebuild failed"):
                du.rebuild_dashboard()

    def test_validate_outputs_calls_subprocess(self):
        """validate_outputs() should invoke validate_dashboard_outputs.py."""
        du = _import_daily_update()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
            du.validate_outputs()
            mock_run.assert_called_once()
            args = mock_run.call_args
            cmd = args[0][0] if args[0] else args[1].get("args", [])
            assert "validate_dashboard_outputs.py" in str(cmd[-1])

    def test_validate_outputs_raises_on_failure(self):
        """validate_outputs() should raise RuntimeError on non-zero exit."""
        du = _import_daily_update()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="error", stderr="crash")
            with pytest.raises(RuntimeError, match="validation failed"):
                du.validate_outputs()

    def test_retrain_model_has_timeout(self):
        """retrain_model() subprocess should have a 30-minute timeout."""
        du = _import_daily_update()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
            du.retrain_model()
            call_kwargs = mock_run.call_args[1] if mock_run.call_args[1] else {}
            timeout = call_kwargs.get("timeout", 0)
            assert timeout >= 300, f"Timeout {timeout}s is too short for model training"


# ===================================================================
# TEST CLASS: main() Orchestration
# ===================================================================

class TestMainOrchestration:
    """Test the main() function flow."""

    def test_main_force_flag_triggers_pipeline(self):
        """--force should trigger retrain even if data unchanged."""
        du = _import_daily_update()
        with patch.object(du, "download_latest", return_value=False), \
             patch.object(du, "check_data_quality", return_value=True), \
             patch.object(du, "retrain_model") as mock_retrain, \
             patch.object(du, "rebuild_dashboard") as mock_rebuild, \
             patch.object(du, "validate_outputs") as mock_validate, \
             patch("sys.argv", ["daily_update.py", "--force"]):
            du.main()
            mock_retrain.assert_called_once()
            mock_rebuild.assert_called_once()
            mock_validate.assert_called_once()

    def test_main_no_change_skips_pipeline(self):
        """If data unchanged and no --force, pipeline should be skipped."""
        du = _import_daily_update()
        with patch.object(du, "download_latest", return_value=False), \
             patch.object(du, "retrain_model") as mock_retrain, \
             patch.object(du, "rebuild_dashboard") as mock_rebuild, \
             patch.object(du, "validate_outputs") as mock_validate, \
             patch("sys.argv", ["daily_update.py"]):
            du.main()
            mock_retrain.assert_not_called()
            mock_rebuild.assert_not_called()
            mock_validate.assert_not_called()

    def test_main_data_changed_triggers_pipeline(self):
        """If download_latest() returns True, pipeline should run."""
        du = _import_daily_update()
        with patch.object(du, "download_latest", return_value=True), \
             patch.object(du, "check_data_quality", return_value=True), \
             patch.object(du, "retrain_model") as mock_retrain, \
             patch.object(du, "rebuild_dashboard") as mock_rebuild, \
             patch.object(du, "validate_outputs") as mock_validate, \
             patch("sys.argv", ["daily_update.py"]):
            du.main()
            mock_retrain.assert_called_once()
            mock_rebuild.assert_called_once()
            mock_validate.assert_called_once()

    def test_main_pipeline_order(self):
        """Pipeline steps should execute in correct order: retrain -> rebuild -> validate."""
        du = _import_daily_update()
        call_order = []
        with patch.object(du, "download_latest", return_value=True), \
             patch.object(du, "check_data_quality", return_value=True), \
             patch.object(du, "retrain_model", side_effect=lambda: call_order.append("retrain")), \
             patch.object(du, "rebuild_dashboard", side_effect=lambda: call_order.append("rebuild")), \
             patch.object(du, "validate_outputs", side_effect=lambda: call_order.append("validate")), \
             patch("sys.argv", ["daily_update.py"]):
            du.main()
        assert call_order == ["retrain", "rebuild", "validate"]

    def test_main_data_quality_failure_aborts(self):
        """If check_data_quality returns False, pipeline should abort."""
        du = _import_daily_update()
        with patch.object(du, "download_latest", return_value=True), \
             patch.object(du, "check_data_quality", return_value=False), \
             patch.object(du, "retrain_model") as mock_retrain, \
             patch("sys.argv", ["daily_update.py"]):
            with pytest.raises(SystemExit):
                du.main()
            mock_retrain.assert_not_called()

    def test_main_dry_run_skips_retrain(self):
        """--dry-run should download+validate but skip retrain and rebuild."""
        du = _import_daily_update()
        with patch.object(du, "download_latest", return_value=True), \
             patch.object(du, "check_data_quality", return_value=True), \
             patch.object(du, "retrain_model") as mock_retrain, \
             patch.object(du, "rebuild_dashboard") as mock_rebuild, \
             patch("sys.argv", ["daily_update.py", "--dry-run"]):
            du.main()
            mock_retrain.assert_not_called()
            mock_rebuild.assert_not_called()


# ===================================================================
# TEST CLASS: Configuration Constants
# ===================================================================

class TestConfiguration:
    """Test the configuration constants in daily_update.py."""

    def test_data_url_is_humdata(self):
        """DATA_URL should point to Humanitarian Data Exchange."""
        du = _import_daily_update()
        assert "data.humdata.org" in du.DATA_URL

    def test_data_url_targets_philippines(self):
        """DATA_URL should download Philippines food price data."""
        du = _import_daily_update()
        assert "phl" in du.DATA_URL.lower()

    def test_wfp_csv_path_under_ml(self):
        """WFP_CSV should be under the ML/WFP directory."""
        du = _import_daily_update()
        assert "WFP" in str(du.WFP_CSV) or "wfp" in str(du.WFP_CSV)

    def test_hash_file_in_base_dir(self):
        """HASH_FILE should be in the same directory as the script."""
        du = _import_daily_update()
        assert du.HASH_FILE.parent == du.BASE_DIR

    def test_log_file_in_base_dir(self):
        """LOG_FILE should be in the same directory as the script."""
        du = _import_daily_update()
        assert du.LOG_FILE.parent == du.BASE_DIR

    def test_base_dir_is_website(self):
        """BASE_DIR should point to the Website directory."""
        du = _import_daily_update()
        assert du.BASE_DIR.name == "Website"


# ===================================================================
# TEST CLASS: Edge Cases
# ===================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_download_cleans_up_tmp_on_failure(self, tmp_path):
        """Temp file should be removed if download fails."""
        du = _import_daily_update()
        csv_path = tmp_path / "WFP" / "data.csv"
        tmp_file = csv_path.with_suffix(".tmp")
        with patch.object(du, "WFP_CSV", csv_path), \
             patch.object(du, "HASH_FILE", tmp_path / ".hash"), \
             patch("time.sleep"):  # skip retry delays
            # Mock urlretrieve to create tmp file then raise (simulates partial download)
            def partial_download(url, dest):
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                Path(dest).write_text("partial")
                raise Exception("Connection lost mid-download")
            with patch("urllib.request.urlretrieve", side_effect=partial_download):
                du.download_latest()
            assert not tmp_file.exists(), "Temp file should be cleaned up on failure"

    def test_download_creates_parent_dirs(self, tmp_path):
        """download_latest should create parent dirs for WFP_CSV."""
        du = _import_daily_update()
        csv_path = tmp_path / "new" / "deep" / "dir" / "data.csv"
        with patch.object(du, "WFP_CSV", csv_path), \
             patch.object(du, "HASH_FILE", tmp_path / ".hash"), \
             patch("urllib.request.urlretrieve") as mock_dl:
            def fake_download(url, dest):
                Path(dest).write_text("data")
            mock_dl.side_effect = fake_download
            du.download_latest()
            assert csv_path.parent.exists()

    def test_data_url_is_valid_url(self):
        """DATA_URL should be a valid HTTP/HTTPS URL."""
        du = _import_daily_update()
        assert du.DATA_URL.startswith("https://")
        assert " " not in du.DATA_URL
