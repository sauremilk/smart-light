"""Tests fuer core/feedback.py – Feedback-System."""

import time

from core.feedback import FeedbackCollector, FeedbackEntry


def test_initial_state():
    fc = FeedbackCollector()
    assert fc.total_count == 0
    assert fc.positive_count == 0
    assert fc.negative_count == 0
    assert fc.recent_satisfaction == 0.5  # Neutral bei leer


def test_record_positive():
    fc = FeedbackCollector(cooldown_s=0.0)
    ok = fc.record("positive", cognitive_state="FOCUS", valence=0.5, arousal=0.3)
    assert ok
    assert fc.total_count == 1
    assert fc.positive_count == 1


def test_record_negative():
    fc = FeedbackCollector(cooldown_s=0.0)
    fc.record("negative")
    assert fc.negative_count == 1


def test_cooldown_prevents_rapid():
    fc = FeedbackCollector(cooldown_s=1.0)
    ok1 = fc.record("positive")
    ok2 = fc.record("positive")  # Too fast
    assert ok1
    assert not ok2
    assert fc.total_count == 1


def test_flash_active():
    fc = FeedbackCollector(cooldown_s=0.0)
    fc.record("positive")
    active, kind = fc.flash_active
    assert active
    assert kind == "positive"


def test_flash_expires():
    fc = FeedbackCollector(cooldown_s=0.0)
    fc.record("positive")
    # Manually expire flash
    fc._flash_until = time.monotonic() - 1.0
    active, kind = fc.flash_active
    assert not active


def test_satisfaction_score():
    fc = FeedbackCollector(cooldown_s=0.0)
    fc.record("positive")
    fc.record("positive")
    fc.record("negative")
    # 2 positive / 3 total = 0.666...
    assert abs(fc.recent_satisfaction - 2.0 / 3.0) < 0.01


def test_feedback_for_log_returns_recent():
    fc = FeedbackCollector(cooldown_s=0.0)
    fc.record("negative", cognitive_state="STRESS", active_mode="RELAX")
    data = fc.get_feedback_for_log()
    assert data is not None
    assert data["feedback_kind"] == "negative"
    assert data["feedback_state"] == "STRESS"


def test_feedback_for_log_returns_none_after_timeout():
    fc = FeedbackCollector(cooldown_s=0.0)
    fc.record("positive")
    # Fake old timestamp
    fc._entries[-1] = FeedbackEntry(
        timestamp=time.monotonic() - 10.0,
        kind="positive",
        cognitive_state="NEUTRAL",
        active_mode="AUTO",
        valence=0.0,
        arousal=0.0,
    )
    data = fc.get_feedback_for_log()
    assert data is None
