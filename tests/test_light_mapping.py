import importlib
import sys
import types


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


def test_valence_arousal_to_light_clamps_output_ranges():
    main = import_main_with_stubs()

    params = main.valence_arousal_to_light(valence=5.0, arousal=-5.0)

    assert 0 <= params["hue"] <= 65535
    assert 1 <= params["bri"] <= 254
    assert 0 <= params["sat"] <= 254


def test_blend_emotion_colors_returns_fallback_for_empty_signal():
    main = import_main_with_stubs()

    ema = {emotion: 0.0 for emotion in main.EMOTION_MAP.keys()}
    params = main.blend_emotion_colors(ema)

    assert params == main.FALLBACK_LIGHT


def test_blend_emotion_colors_returns_valid_light_params():
    main = import_main_with_stubs()

    emotions = list(main.EMOTION_MAP.keys())
    ema = {e: 0.0 for e in emotions}
    ema[emotions[0]] = 0.7
    ema[emotions[1]] = 0.3

    params = main.blend_emotion_colors(ema)

    assert 0 <= params["hue"] <= 65535
    assert 1 <= params["bri"] <= 254
    assert 0 <= params["sat"] <= 254


def test_fuse_modalities_without_audio_normalizes_distribution():
    main = import_main_with_stubs()

    video_ema = {e: 0.0 for e in main.EMOTION_MAP.keys()}
    first, second = list(video_ema.keys())[:2]
    video_ema[first] = 5.0
    video_ema[second] = 5.0

    fused = main.fuse_modalities(video_ema, audio_ema=None, pose_arousal_offset=0.0, audio_weight=0.0)

    assert abs(sum(fused.values()) - 1.0) < 1e-9
    assert fused[first] == 0.5
    assert fused[second] == 0.5


def test_fuse_modalities_with_audio_uses_weighted_mix_and_normalizes():
    main = import_main_with_stubs()

    emotions = list(main.EMOTION_MAP.keys())
    video_ema = {e: 0.0 for e in emotions}
    audio_ema = {e: 0.0 for e in emotions}

    video_ema[emotions[0]] = 1.0
    audio_ema[emotions[1]] = 1.0

    fused = main.fuse_modalities(video_ema, audio_ema=audio_ema, pose_arousal_offset=0.0, audio_weight=0.35)

    assert abs(sum(fused.values()) - 1.0) < 1e-9
    assert fused[emotions[0]] == 0.65
    assert fused[emotions[1]] == 0.35


def test_fuse_modalities_with_face_mesh_blends_scores():
    main = import_main_with_stubs()

    emotions = list(main.EMOTION_MAP.keys())
    video_ema = {e: 0.0 for e in emotions}
    face_mesh_scores = {e: 0.0 for e in emotions}

    video_ema[emotions[0]] = 1.0
    face_mesh_scores[emotions[1]] = 100.0  # AU-Scores sind 0-100

    fused = main.fuse_modalities(
        video_ema, audio_ema=None, pose_arousal_offset=0.0, audio_weight=0.0,
        face_mesh_scores=face_mesh_scores, face_mesh_weight=0.25,
    )

    assert abs(sum(fused.values()) - 1.0) < 1e-9
    assert fused[emotions[0]] > 0.5  # Video dominiert (75%)
    assert fused[emotions[1]] > 0.1  # Face Mesh traegt bei (25%)
