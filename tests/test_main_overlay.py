import importlib
import sys
import types

import numpy as np

from core.overlay import _build_status_text, _build_top3_text


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


def test_build_top3_text_with_fused_ema_returns_ranked_values():
    text = _build_top3_text(
        fused_ema={"happy": 0.5, "sad": 0.3, "neutral": 0.2, "angry": 0.1},
        emotion="neutral",
        confidence=0.0,
    )

    assert text == "happy 50% | sad 30% | neutral 20%"


def test_build_top3_text_falls_back_to_emotion_confidence_without_fusion():
    text = _build_top3_text(fused_ema={}, emotion="neutral", confidence=0.42)

    assert text == "neutral (42%)"


def test_build_status_text_includes_enabled_modules_and_burst_suffix():
    status = _build_status_text(
        fps_display=28.7,
        analysis_every_n=5,
        has_audio=True,
        has_pose=False,
        has_face_mesh=False,
        has_calibration=True,
        burst_active=True,
    )

    assert status == "FPS:29  1/5  [Video+Audio+Cal]  BURST"


def test_build_status_text_includes_face_mesh_module():
    status = _build_status_text(
        fps_display=24.0,
        analysis_every_n=7,
        has_audio=False,
        has_pose=True,
        has_face_mesh=True,
        has_calibration=False,
        burst_active=False,
    )

    assert status == "FPS:24  1/7  [Video+Pose+FaceMesh]"


def test_draw_overlay_shows_no_face_message_for_hrv_when_face_missing(monkeypatch):
    main = import_main_with_stubs()

    drawn_texts = []

    def _fake_put_text(img, text, *args, **kwargs):
        drawn_texts.append(text)
        return img

    monkeypatch.setattr(main.cv2, "putText", _fake_put_text)

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    main._draw_overlay(
        frame=frame,
        fused_ema={},
        emotion="neutral",
        confidence=0.0,
        valence=0.0,
        arousal=0.0,
        trend_v=0.0,
        params={"hue": 10000, "bri": 128, "sat": 180},
        transition=4,
        fps_display=24.0,
        analysis_every_n=5,
        has_audio=False,
        has_pose=False,
        has_face_mesh=False,
        has_calibration=False,
        burst_active=False,
        low_fps_guard=False,
        hrv_result={
            "hr_bpm": 72.0,
            "hrv_rmssd": 35.0,
            "confidence": 0.9,
            "face_detected": False,
        },
        breathing_result=None,
    )

    assert "HR: kein Gesicht" in drawn_texts
