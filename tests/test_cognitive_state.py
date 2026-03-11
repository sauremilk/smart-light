"""Tests fuer core/cognitive_state.py – Kognitiver Zustandsklassifikator."""

import time

from core.cognitive_state import CognitiveClassifier, CognitiveState, _clamp01


def test_clamp01():
    assert _clamp01(-0.5) == 0.0
    assert _clamp01(1.5) == 1.0
    assert _clamp01(0.5) == 0.5


def test_initial_state_is_neutral():
    c = CognitiveClassifier()
    result = c.update()
    assert result.state == "NEUTRAL"
    assert isinstance(result, CognitiveState)
    assert 0.0 <= result.confidence <= 1.0


def test_focus_detection():
    c = CognitiveClassifier()
    # Simulate focus conditions repeatedly
    for _ in range(20):
        result = c.update(
            blink_rate=6.0,  # Low blink = focused
            cognitive_load=0.8,  # High activity
            torso_lean=0.5,  # Leaning forward
            arousal=0.3,  # Moderate arousal
        )
    assert result.state == "FOCUS"
    assert result.confidence > 0.3


def test_fatigue_detection():
    c = CognitiveClassifier()
    for _ in range(25):
        result = c.update(
            blink_rate=35.0,  # High blink = tired
            shoulder_drop=0.5,  # Slouching
            cognitive_load=0.0,  # No activity
            arousal=-0.5,  # Low arousal
        )
    assert result.state == "FATIGUE"


def test_stress_detection():
    c = CognitiveClassifier()
    for _ in range(25):
        result = c.update(
            hr_bpm=110.0,  # High heart rate
            br_bpm=24.0,  # Fast breathing
            valence=-0.7,  # Negative valence
            arousal=0.8,  # High arousal
            hrv_rmssd=15.0,  # Low HRV
        )
    assert result.state == "STRESS"


def test_state_duration_tracking():
    c = CognitiveClassifier()
    # First call
    result1 = c.update()
    assert result1.duration_s >= 0.0

    time.sleep(0.05)

    # Second call should show increased duration if state didn't change
    result2 = c.update()
    assert result2.duration_s >= result1.duration_s


def test_scores_dict_has_all_states():
    c = CognitiveClassifier()
    result = c.update()
    for state in ["FOCUS", "FLOW", "FATIGUE", "STRESS", "NEUTRAL"]:
        assert state in result.scores


def test_properties():
    c = CognitiveClassifier()
    c.update()
    assert c.current_state in ("FOCUS", "FLOW", "FATIGUE", "STRESS", "NEUTRAL")
    assert c.state_duration >= 0.0
