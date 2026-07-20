import json
from pathlib import Path

import gpu_forecast_driver as driver


def test_regression_metrics_are_exact_for_simple_population():
    metrics = driver._regression_metrics([10.0, 20.0], [9.0, 22.0])
    assert metrics == {
        "mape": 10.0,
        "mae": 1.5,
        "rmse": 1.5811,
        "r2": 0.9,
        "n": 2,
    }


def test_publication_gate_requires_model_to_beat_naive_on_mape_and_mae():
    passed = driver._publication_gate(
        {"mape": 8.0, "mae": 3.0, "n": 40},
        {"mape": 10.0, "mae": 4.0, "n": 40},
        model_count=12,
    )
    assert passed["passed"] is True
    assert passed["status"] == "passed_out_of_time_naive_baseline"

    failed = driver._publication_gate(
        {"mape": 8.0, "mae": 5.0, "n": 40},
        {"mape": 10.0, "mae": 4.0, "n": 40},
        model_count=12,
    )
    assert failed["passed"] is False
    assert any("did not beat naive MAE" in reason for reason in failed["reasons"])


def test_training_cutoff_is_inferred_from_earliest_forecast(tmp_path: Path):
    artifact = tmp_path / "predictions.json"
    artifact.write_text(
        json.dumps({"forecasts": {"Rice": {"2026-02": 1.0, "2026-03": 2.0}}}),
        encoding="utf-8",
    )
    proof = driver._infer_training_cutoff(artifact)
    assert proof["forecast_start"] == "2026-02"
    assert proof["training_cutoff"] == "2026-01"
    assert len(proof["prediction_artifact_sha256"]) == 64


def test_predictions_without_validation_are_explicitly_withheld(monkeypatch, tmp_path: Path):
    class FakeSession:
        def get_modelmeta(self):
            return type("Meta", (), {"custom_metadata_map": {
                "commodity": "Rice",
                "scaler_mean": "[10.0]",
                "scaler_scale": "[2.0]",
                "source_checkpoint_sha256": "abc",
                "source_checkpoint_mtime_utc": "2026-01-01T00:00:00+00:00",
            }})()

        def get_providers(self):
            return [driver.CPU_PROVIDER]

        def get_inputs(self):
            return [type("Input", (), {"name": "features"})()]

        def run(self, *_args, **_kwargs):
            return [[10.0]]

    monthly = __import__("pandas").DataFrame({
        "commodity": ["Rice"] * 12,
        "date": __import__("pandas").date_range("2025-07-01", periods=12, freq="MS"),
        "price": [10.0] * 12,
        "region_enc": [0] * 12,
        "pt_enc": [0] * 12,
        "year": [2025] * 6 + [2026] * 6,
    })
    model = tmp_path / "lstm_Rice.onnx"
    model.write_bytes(b"model")
    monkeypatch.setattr(driver, "probe", lambda: {
        "directml_available": True,
        "available_providers": [driver.DML_PROVIDER, driver.CPU_PROVIDER],
    })
    monkeypatch.setattr(driver, "_load_monthly", lambda _path: (monthly, {"max_date": "2026-06-01"}))
    monkeypatch.setattr(driver, "_session", lambda *_args, **_kwargs: FakeSession())
    output = tmp_path / "predictions.json"
    result = driver.generate_predictions(
        tmp_path / "data.csv", tmp_path, output, horizon=1, prefer_gpu=True
    )
    assert result["publication_gate"]["status"] == "withheld_validation_missing"
    assert result["publication_gate"]["passed"] is False


def test_directml_error_names_project_pinned_interpreter(monkeypatch, tmp_path: Path):
    expected = tmp_path / ".venv-directml" / "Scripts" / "python.exe"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"")
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    message = driver._directml_unavailable_message()
    assert "DmlExecutionProvider is unavailable" in message
    assert str(expected.resolve()) in message
