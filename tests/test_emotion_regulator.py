from emotion_regulator import EmotionRegulator
from light_mapping import compute_va_from_ema


def test_regulator_moves_state_towards_target():
    regulator = EmotionRegulator(
        target_v=0.7,
        target_a=0.0,
        blend_strength=0.45,
        blend_max=0.8,
        progress_timeout=30.0,
        escalation=0.1,
        at_target_threshold=0.2,
    )

    info = regulator.update(current_v=-0.9, current_a=1.0)

    assert info["reg_v"] > -0.9
    assert info["reg_a"] < 1.0
    assert info["blend"] == 0.45
    assert info["at_target"] is False


def test_regulator_reports_stable_when_near_target():
    regulator = EmotionRegulator(
        target_v=0.7,
        target_a=0.0,
        blend_strength=0.45,
        blend_max=0.8,
        progress_timeout=30.0,
        escalation=0.1,
        at_target_threshold=0.2,
    )

    info = regulator.update(current_v=0.65, current_a=0.05)

    assert info["at_target"] is True
    assert info["blend"] == 0.0
    assert "Stabil" in info["label"]


def test_regulator_escalates_blend_after_no_progress_timeout():
    regulator = EmotionRegulator(
        target_v=0.7,
        target_a=0.0,
        blend_strength=0.45,
        blend_max=0.8,
        progress_timeout=30.0,
        escalation=0.1,
        at_target_threshold=0.2,
    )

    # Initial update sets baseline distance.
    regulator.update(current_v=-0.8, current_a=0.9)

    # Force a virtual timeout without sleeping.
    regulator._last_update_time -= 31.0  # noqa: SLF001
    info = regulator.update(current_v=-0.8, current_a=0.9)

    assert info["blend"] > 0.45
    assert info["blend"] <= 0.8


def test_compute_va_from_ema_returns_weighted_result():
    ema = {
        "happy": 0.8,
        "sad": 0.1,
        "angry": 0.02,
        "fear": 0.02,
        "surprise": 0.02,
        "disgust": 0.02,
        "neutral": 0.02,
    }

    valence, arousal = compute_va_from_ema(ema)

    assert valence > 0.5
    assert arousal > 0.0


def test_compute_va_from_ema_handles_empty_signal():
    valence, arousal = compute_va_from_ema({
        "happy": 0.0,
        "sad": 0.0,
        "angry": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "neutral": 0.0,
    })

    assert valence == 0.0
    assert arousal == 0.0
