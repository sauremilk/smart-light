import importlib
import sys
import types

from core.overlay import _build_status_text


def import_main_with_stubs():
    deepface_module = types.ModuleType("deepface")

    class _DeepFace:
        @staticmethod
        def analyze(*args, **kwargs):
            return []

    setattr(deepface_module, "DeepFace", _DeepFace)

    phue_module = types.ModuleType("phue")

    class _Bridge:
        def __init__(self, ip):
            self.ip = ip

        def connect(self):
            return None

        def set_light(self, *args, **kwargs):
            return None

    setattr(phue_module, "Bridge", _Bridge)

    sys.modules["deepface"] = deepface_module
    sys.modules["phue"] = phue_module

    if "main" in sys.modules:
        del sys.modules["main"]

    return importlib.import_module("main")


def test_prediction_quality_prefers_peaked_distribution():
    main = import_main_with_stubs()
    analyzer = main.EmotionAnalyzer(calibration={})

    peaked = {
        "happy": 95.0,
        "sad": 1.0,
        "angry": 1.0,
        "fear": 1.0,
        "surprise": 1.0,
        "disgust": 0.5,
        "neutral": 0.5,
    }
    flat = {
        "happy": 14.0,
        "sad": 14.0,
        "angry": 14.0,
        "fear": 14.0,
        "surprise": 14.0,
        "disgust": 15.0,
        "neutral": 15.0,
    }

    q_peaked = analyzer._compute_prediction_quality(peaked)
    q_flat = analyzer._compute_prediction_quality(flat)

    assert 0.0 <= q_peaked <= 1.0
    assert 0.0 <= q_flat <= 1.0
    assert q_peaked > q_flat


def test_status_text_marks_low_quality_guardrail():
    status = _build_status_text(
        fps_display=20.0,
        analysis_every_n=8,
        has_audio=True,
        has_pose=False,
        has_face_mesh=False,
        has_calibration=False,
        burst_active=False,
        low_fps_guard=False,
        has_hrv=False,
        has_breathing=False,
        low_quality_guardrail=True,
    )

    assert "LOW-Q" in status
