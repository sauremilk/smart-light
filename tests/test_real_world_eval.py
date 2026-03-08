import importlib.util
import json
import pathlib


def _load_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "benchmarks" / "real_world_eval.py"
    spec = importlib.util.spec_from_file_location("real_world_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_eval_reports_core_metrics_and_uncertainty(tmp_path):
    mod = _load_module()
    src = tmp_path / "sample.jsonl"
    rows = [
        {
            "session_id": "s1",
            "timestamp": 1.0,
            "condition": "adaptive",
            "scenario": {
                "lighting": "low_light",
                "occlusion": "partial",
                "head_pose": "moderate",
                "background_noise": "medium",
            },
            "ground_truth_emotion": "happy",
            "predicted_emotion": "happy",
            "prediction_confidence": 0.80,
            "guardrail_active": False,
        },
        {
            "session_id": "s1",
            "timestamp": 2.0,
            "condition": "adaptive",
            "scenario": {
                "lighting": "low_light",
                "occlusion": "partial",
                "head_pose": "moderate",
                "background_noise": "medium",
            },
            "ground_truth_emotion": "sad",
            "predicted_emotion": "happy",
            "prediction_confidence": 0.20,
            "guardrail_active": True,
        },
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    report = mod.run_eval([str(src)], low_conf_threshold=0.45)

    assert report["n_records"] == 2
    assert report["n_valid_records"] == 2
    assert report["metrics"]["accuracy"] == 0.5
    assert "macro_f1" in report["metrics"]
    assert report["uncertainty"]["low_conf_rate"] == 0.5
    assert report["uncertainty"]["guardrail_activation_rate"] == 0.5
    assert len(report["scenarios"]) == 1
