"""Tests fuer core/mode_manager.py – Modus-System."""

import time

from core.mode_manager import (
    MODE_FOCUS,
    MODES,
    ModeManager,
    ModeProfile,
)


def test_modes_dict_complete():
    assert set(MODES.keys()) == {"FOCUS", "ENERGY", "RELAX", "RECOVERY"}


def test_mode_profiles_are_frozen():
    for profile in MODES.values():
        assert isinstance(profile, ModeProfile)
        # Frozen dataclass should raise on assignment
        try:
            profile.name = "test"
            assert False, "Should have raised"
        except AttributeError:
            pass


def test_manual_mode_setting():
    mm = ModeManager(initial_mode="FOCUS")
    assert not mm.is_auto
    assert mm.active_mode == "FOCUS"
    assert mm.active_profile == MODE_FOCUS


def test_auto_mode_initial():
    mm = ModeManager(initial_mode="AUTO")
    assert mm.is_auto
    assert mm.active_mode == "FOCUS"  # Default auto


def test_set_mode_to_auto():
    mm = ModeManager(initial_mode="RELAX")
    assert mm.active_mode == "RELAX"
    mm.set_mode("AUTO")
    assert mm.is_auto


def test_cycle_mode():
    mm = ModeManager(initial_mode="AUTO")
    # AUTO → FOCUS → ENERGY → RELAX → RECOVERY → AUTO
    mm.cycle_mode()
    assert mm.active_mode == "FOCUS"
    mm.cycle_mode()
    assert mm.active_mode == "ENERGY"
    mm.cycle_mode()
    assert mm.active_mode == "RELAX"
    mm.cycle_mode()
    assert mm.active_mode == "RECOVERY"
    mm.cycle_mode()
    assert mm.is_auto


def test_auto_mode_hysteresis():
    mm = ModeManager(initial_mode="AUTO")
    mm._hysteresis_s = 0.05  # Very short for testing
    now = time.monotonic()

    # Initial state should be FOCUS
    mm.update_auto("FATIGUE", now)
    # Still FOCUS (hysteresis not expired)
    assert mm.active_mode == "FOCUS"

    # After hysteresis period
    mm.update_auto("FATIGUE", now + 0.06)
    assert mm.active_mode == "RECOVERY"


def test_manual_mode_ignores_auto_update():
    mm = ModeManager(initial_mode="ENERGY")
    now = time.monotonic()
    mm.update_auto("FATIGUE", now)
    mm.update_auto("FATIGUE", now + 100)
    assert mm.active_mode == "ENERGY"  # Manual stays


def test_mode_profiles_have_valid_values():
    for profile in MODES.values():
        assert -1.0 <= profile.target_v <= 1.0
        assert -1.0 <= profile.target_a <= 1.0
        assert 0 <= profile.hue_negative <= 65535
        assert 0 <= profile.hue_positive <= 65535
        assert 1 <= profile.bri_max <= 254
        assert 0.0 <= profile.blend_strength <= 1.0
