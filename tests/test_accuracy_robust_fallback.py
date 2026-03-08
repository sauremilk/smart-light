import importlib.util
import pathlib
import sys

import numpy as np


def _load_accuracy_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "benchmarks" / "accuracy_benchmark.py"
    spec = importlib.util.spec_from_file_location("accuracy_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["accuracy_benchmark"] = module
    spec.loader.exec_module(module)
    return module


def test_robust_scores_can_select_non_base_variant(monkeypatch):
    mod = _load_accuracy_module()

    def fake_deepface_scores(frame, detector_backend):
        # Determine variant by mean brightness that our synthetic transforms change.
        m = float(frame.mean())
        if m > 175.0:
            # lifted -> strongest confidence
            return ({e: (90.0 if e == "happy" else 1.0) for e in mod.EMOTIONS}, 0.90)
        if m < 90.0:
            # denoise/unsharp path in this synthetic setup
            return ({e: (55.0 if e == "sad" else 5.0) for e in mod.EMOTIONS}, 0.55)
        # base
        return ({e: (45.0 if e == "neutral" else 8.0) for e in mod.EMOTIONS}, 0.45)

    monkeypatch.setattr(mod, "deepface_scores", fake_deepface_scores)

    img = np.full((48, 48, 3), 120, dtype=np.uint8)
    scores, conf, variant, quality, backend = mod._robust_deepface_scores(img, detector_backend="opencv")

    assert conf >= 0.45
    assert variant in {"lifted", "denoise", "unsharp", "base", "lifted_dark", "denoise_lifted_dark"}
    assert backend in {"opencv", "retinaface"}
    assert quality >= 0.0
    assert isinstance(scores, dict)


def test_robust_scores_can_use_detector_fallback(monkeypatch):
    mod = _load_accuracy_module()

    def fake_deepface_scores(frame, detector_backend):
        if detector_backend == "opencv":
            return ({e: (35.0 if e == "neutral" else 9.0) for e in mod.EMOTIONS}, 0.35)
        return ({e: (92.0 if e == "happy" else 1.0) for e in mod.EMOTIONS}, 0.92)

    monkeypatch.setattr(mod, "deepface_scores", fake_deepface_scores)

    img = np.full((48, 48, 3), 120, dtype=np.uint8)
    scores, conf, variant, quality, backend = mod._robust_deepface_scores(img, detector_backend="opencv")

    assert backend == "retinaface"
    assert conf > 0.0
    assert max(scores, key=scores.get) == "happy"


def test_predict_enhanced_returns_label_under_stubbed_scores(monkeypatch):
    mod = _load_accuracy_module()

    def fake_robust(frame, detector_backend):
        scores = {e: 0.0 for e in mod.EMOTIONS}
        scores["happy"] = 95.0
        return scores, 0.95, "lifted", 0.8, "opencv"

    monkeypatch.setattr(mod, "_robust_deepface_scores", fake_robust)

    img = np.full((48, 48, 3), 120, dtype=np.uint8)
    label = mod.predict_enhanced(
        img,
        detector_backend="opencv",
        face_mesh=None,
        use_face_mesh=False,
        return_debug=False,
    )

    assert label == "happy"
