"""Tests fuer core/break_manager.py – Pausen-Manager."""

import time

from core.break_manager import BreakEvent, BreakManager


def test_initial_state_no_break():
    bm = BreakManager()
    event = bm.update("NEUTRAL")
    assert isinstance(event, BreakEvent)
    assert not event.break_recommended
    assert not event.break_active
    assert event.work_duration_s >= 0.0


def test_fatigue_triggers_break():
    bm = BreakManager(fatigue_trigger_s=0.1)
    # Simulate fatigue for > trigger time
    time.sleep(0.15)
    event = bm.update("FATIGUE")
    assert event.break_recommended
    assert event.reason == "fatigue"


def test_max_work_triggers_break():
    bm = BreakManager(max_work_minutes=0.001)  # ~60ms
    time.sleep(0.1)
    event = bm.update("NEUTRAL")
    assert event.break_recommended
    assert event.reason == "max_work"


def test_pomodoro_triggers_break():
    bm = BreakManager(pomodoro_enabled=True, pomodoro_work_minutes=0.001)
    time.sleep(0.1)
    event = bm.update("NEUTRAL")
    assert event.break_recommended
    assert event.reason == "pomodoro"


def test_start_and_end_break():
    bm = BreakManager(min_break_minutes=0.001)
    bm.start_break()
    assert bm.is_break_active
    event = bm.update("NEUTRAL")
    assert event.break_active

    # Wait for min break duration
    time.sleep(0.1)
    event = bm.update("FOCUS")
    # Break should auto-end after min duration
    assert not bm.is_break_active


def test_skip_break():
    bm = BreakManager()
    bm.start_break()
    assert bm.is_break_active
    bm.skip_break()
    assert not bm.is_break_active


def test_dismiss_break():
    bm = BreakManager(fatigue_trigger_s=0.05)
    time.sleep(0.1)
    event = bm.update("FATIGUE")
    assert event.break_recommended

    bm.dismiss_break()
    event = bm.update("FATIGUE")
    assert not event.break_recommended  # Dismissed


def test_recovery_quality_tracking():
    bm = BreakManager(min_break_minutes=0.001)
    bm.start_break()

    # Simulate recovery
    bm.update("FOCUS")
    bm.update("NEUTRAL")
    bm.update("FATIGUE")
    time.sleep(0.1)
    bm.update("FOCUS")

    # After break ends, recovery quality should reflect positive samples
    assert bm._recovery_quality > 0.0


def test_pomodoro_cycle_counting():
    bm = BreakManager(
        pomodoro_enabled=True,
        pomodoro_work_minutes=0.001,
        pomodoro_break_minutes=0.001,
    )

    # Trigger pomodoro break
    time.sleep(0.1)
    bm.update("NEUTRAL")
    bm.start_break()
    time.sleep(0.1)
    bm.update("NEUTRAL")  # Should auto-end break

    event = bm.update("NEUTRAL")
    assert event.pomodoro_cycle == 1
