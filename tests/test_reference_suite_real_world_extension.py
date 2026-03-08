import importlib.util
import json
import pathlib
import sys


def _load_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "benchmarks" / "reference_suite.py"
    spec = importlib.util.spec_from_file_location("reference_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["reference_suite"] = module
    spec.loader.exec_module(module)
    return module


def test_real_world_extension_reports_available_when_files_exist(tmp_path):
    mod = _load_module()

    p = tmp_path / "sample.jsonl"
    row = {
        "session_id": "s1",
        "timestamp": 1.0,
        "condition": "adaptive",
        "scenario": {
            "lighting": "normal",
            "occlusion": "none",
            "head_pose": "frontal",
            "background_noise": "low",
        },
        "ground_truth_emotion": "happy",
        "predicted_emotion": "happy",
        "prediction_confidence": 0.9,
        "guardrail_active": False,
    }
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")

    ext = mod._run_real_world_uncertainty_extension([str(p)], low_conf_threshold=0.45)

    assert ext["available"] is True
    assert ext["n_valid_records"] == 1
    assert "uncertainty" in ext


def test_real_world_extension_reports_unavailable_without_files():
    mod = _load_module()

    ext = mod._run_real_world_uncertainty_extension(["does-not-exist/*.jsonl"], low_conf_threshold=0.45)

    assert ext["available"] is False
    assert ext["input_files"] == []
