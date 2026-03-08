"""Tests fuer face_mesh_analyzer.py und Color Constancy."""

import importlib
import sys
import types
import numpy as np


def import_main_with_stubs():
    """Import main.py while stubbing heavy external modules."""
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


# ────────────── Color Constancy Tests ──────────────────────────


def test_gray_world_correction_normalizes_color_channels():
    main = import_main_with_stubs()

    # Bild mit starkem Blaustich (simuliert blaue Hue-Lampe)
    frame = np.full((100, 100, 3), [50, 50, 200], dtype=np.uint8)
    corrected = main.gray_world_correction(frame)

    # Alle Kanaele sollten nach Korrektur naeher an 128 liegen
    for c in range(3):
        avg = corrected[:, :, c].mean()
        assert 100 < avg < 160, f"Kanal {c}: avg={avg:.1f} nicht nahe 128"


def test_gray_world_correction_preserves_shape():
    main = import_main_with_stubs()

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    corrected = main.gray_world_correction(frame)

    assert corrected.shape == frame.shape
    assert corrected.dtype == np.uint8


def test_gray_world_correction_handles_dark_image():
    main = import_main_with_stubs()

    # Fast schwarzes Bild → sollte nicht durch 0 dividieren
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    corrected = main.gray_world_correction(frame)

    assert corrected.shape == frame.shape


# ────────────── Face Mesh AU → Emotion Tests ───────────────────


def test_aus_to_emotions_happy_with_duchenne_smile():
    """AU6 (Cheek Raiser) + AU12 (Lip Corner Puller) → happy dominant."""
    from face_mesh_analyzer import FaceMeshAnalyzer

    aus = {k: 0.0 for k in ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU15", "AU20", "AU25", "AU26"]}
    aus["AU6"] = 1.2
    aus["AU12"] = 1.5

    scores = FaceMeshAnalyzer._aus_to_emotions(aus)

    assert scores["happy"] > scores["neutral"]
    assert scores["happy"] > scores["sad"]
    total = sum(scores.values())
    assert abs(total - 100.0) < 0.1


def test_aus_to_emotions_neutral_with_low_activity():
    """Keine signifikanten AUs → neutral dominant."""
    from face_mesh_analyzer import FaceMeshAnalyzer

    aus = {k: 0.0 for k in ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU15", "AU20", "AU25", "AU26"]}

    scores = FaceMeshAnalyzer._aus_to_emotions(aus)

    assert scores["neutral"] > scores["happy"]
    assert scores["neutral"] > scores["angry"]


def test_aus_to_emotions_surprise_with_raised_brows_and_jaw():
    """AU1 + AU2 (Brauen hoch) + AU25/AU26 (Mund offen) → surprise."""
    from face_mesh_analyzer import FaceMeshAnalyzer

    aus = {k: 0.0 for k in ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU15", "AU20", "AU25", "AU26"]}
    aus["AU1"] = 1.5
    aus["AU2"] = 1.3
    aus["AU25"] = 1.8
    aus["AU26"] = 1.5

    scores = FaceMeshAnalyzer._aus_to_emotions(aus)

    assert scores["surprise"] > scores["neutral"]


def test_aus_to_emotions_scores_sum_to_100():
    """Alle Score-Vektoren summieren sich auf 100."""
    from face_mesh_analyzer import FaceMeshAnalyzer

    aus = {k: 0.5 for k in ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU15", "AU20", "AU25", "AU26"]}

    scores = FaceMeshAnalyzer._aus_to_emotions(aus)

    total = sum(scores.values())
    assert abs(total - 100.0) < 0.1


# ────────────── Head Pose Confidence Tests ─────────────────────


def test_head_pose_confidence_full_at_frontal():
    """Frontale Kopfhaltung → Confidence-Faktor = 1.0."""
    from face_mesh_analyzer import FaceMeshAnalyzer

    factor = FaceMeshAnalyzer._head_pose_to_confidence(
        {"yaw": 5.0, "pitch": 3.0, "roll": 0.0}
    )
    assert factor == 1.0


def test_head_pose_confidence_reduced_at_extreme_yaw():
    """Starke Seitendrehung → Confidence-Faktor < 1.0."""
    from face_mesh_analyzer import FaceMeshAnalyzer

    factor = FaceMeshAnalyzer._head_pose_to_confidence(
        {"yaw": 50.0, "pitch": 0.0, "roll": 0.0}
    )
    assert factor < 1.0
    assert factor >= 0.5  # Nicht unter MAX_ATTENUATION


def test_head_pose_confidence_reduced_at_extreme_pitch():
    """Starke Kopfneigung → Confidence-Faktor < 1.0."""
    from face_mesh_analyzer import FaceMeshAnalyzer

    factor = FaceMeshAnalyzer._head_pose_to_confidence(
        {"yaw": 0.0, "pitch": 40.0, "roll": 0.0}
    )
    assert factor < 1.0
    assert factor >= 0.5


def test_head_pose_confidence_never_below_minimum():
    """Auch bei extremer Pose nie unter HEAD_POSE_MAX_ATTENUATION."""
    from face_mesh_analyzer import FaceMeshAnalyzer

    factor = FaceMeshAnalyzer._head_pose_to_confidence(
        {"yaw": 90.0, "pitch": 80.0, "roll": 45.0}
    )
    assert factor >= 0.5
