"""Tests für core.alexa_controller – Mood-Mapping, Hysterese, Volume, Controller-Lifecycle."""

import asyncio
import time
from unittest.mock import MagicMock, patch

from core.alexa_controller import (
    AlexaController,
    _hide_email,
    _va_to_mood,
    _va_to_volume,
)

# ---------------------------------------------------------------------------
# _va_to_mood – Basiskategorien
# ---------------------------------------------------------------------------


class TestVaToMood:
    def test_positive_high_arousal(self):
        assert _va_to_mood(0.5, 0.3) == "energetic_positive"

    def test_positive_low_arousal(self):
        assert _va_to_mood(0.5, 0.0) == "calm_positive"

    def test_negative_high_arousal(self):
        assert _va_to_mood(-0.5, 0.3) == "energetic_negative"

    def test_negative_low_arousal(self):
        assert _va_to_mood(-0.5, 0.0) == "calm_negative"

    def test_neutral(self):
        assert _va_to_mood(0.0, 0.0) == "neutral"

    def test_boundary_positive(self):
        assert _va_to_mood(0.25, 0.0) == "calm_positive"

    def test_boundary_negative(self):
        assert _va_to_mood(-0.20, 0.0) == "neutral"  # >= -0.20 bleibt neutral
        assert _va_to_mood(-0.21, 0.0) == "calm_negative"


# ---------------------------------------------------------------------------
# _va_to_mood – Hysterese
# ---------------------------------------------------------------------------


class TestMoodHysteresis:
    """Einmal in einem Mood, braucht es einen deutlicheren Wechsel zum Verlassen."""

    def test_stay_positive_below_enter_threshold(self):
        # Valence 0.20 < 0.25 (enter), aber >= 0.18 (leave) → bleibt positiv
        assert _va_to_mood(0.20, 0.0, current_mood="calm_positive") == "calm_positive"

    def test_leave_positive_below_leave_threshold(self):
        # Valence 0.15 < 0.18 (leave) → wechselt zu neutral
        assert _va_to_mood(0.15, 0.0, current_mood="calm_positive") == "neutral"

    def test_stay_negative_above_enter_threshold(self):
        # Valence -0.15 > -0.20 (enter), aber < -0.13 (leave) → bleibt negativ
        assert _va_to_mood(-0.15, 0.0, current_mood="calm_negative") == "calm_negative"

    def test_leave_negative_above_leave_threshold(self):
        # Valence -0.10 > -0.13 (leave) → wechselt zu neutral
        assert _va_to_mood(-0.10, 0.0, current_mood="calm_negative") == "neutral"

    def test_no_hysteresis_from_neutral(self):
        # Von neutral braucht es den vollen Enter-Schwellwert
        assert _va_to_mood(0.20, 0.0, current_mood="neutral") == "neutral"
        assert _va_to_mood(0.25, 0.0, current_mood="neutral") == "calm_positive"

    def test_no_hysteresis_without_current_mood(self):
        # Ohne aktuelle Stimmung gelten Enter-Schwellen
        assert _va_to_mood(0.20, 0.0, current_mood=None) == "neutral"


# ---------------------------------------------------------------------------
# _va_to_volume
# ---------------------------------------------------------------------------


class TestVaToVolume:
    def test_low_arousal(self):
        vol = _va_to_volume(0.0, -1.0)
        assert 10 <= vol <= 70

    def test_high_arousal(self):
        vol = _va_to_volume(0.0, 1.0)
        assert 10 <= vol <= 70

    def test_clamp_min(self):
        vol = _va_to_volume(0.0, -10.0)
        assert vol == 10

    def test_clamp_max(self):
        vol = _va_to_volume(0.0, 10.0)
        assert vol == 70


# ---------------------------------------------------------------------------
# _hide_email
# ---------------------------------------------------------------------------


class TestHideEmail:
    def test_masks_email(self):
        assert _hide_email("user@example.com") == "us***@example.com"

    def test_no_at_sign(self):
        assert _hide_email("noemail") == "****"

    def test_short_local(self):
        assert _hide_email("a@b.de") == "a***@b.de"


# ---------------------------------------------------------------------------
# AlexaController – Lifecycle
# ---------------------------------------------------------------------------


class TestAlexaControllerInit:
    def test_defaults(self):
        ctrl = AlexaController(email="a@b.de", password="pw", device_name="Echo")
        assert ctrl._amazon_url == "amazon.de"
        assert ctrl._cooldown == 30.0
        assert ctrl.ready is False

    def test_custom_playlists(self):
        playlists = {"neutral": "jazz"}
        ctrl = AlexaController(
            email="a@b.de",
            password="pw",
            device_name="Echo",
            mood_playlists=playlists,
        )
        assert ctrl._playlists == playlists


class TestAlexaControllerUpdate:
    def test_no_crash_before_start(self):
        """update() soll nicht crashen wenn kein Loop gestartet ist."""
        ctrl = AlexaController(email="a@b.de", password="pw", device_name="Echo")
        ctrl.update(0.5, 0.5, "happy")  # Kein Fehler

    def test_shutdown_without_start(self):
        """shutdown() soll nicht crashen wenn nie gestartet."""
        ctrl = AlexaController(email="a@b.de", password="pw", device_name="Echo")
        ctrl.shutdown()  # Kein Fehler


class TestAlexaControllerAbsence:
    def test_no_warning_when_not_ready(self):
        ctrl = AlexaController(email="a@b.de", password="pw", device_name="Echo")
        ctrl.check_absence(timeout_seconds=0.0)  # Kein Fehler, kein Warning

    def test_warning_after_timeout(self):
        ctrl = AlexaController(email="a@b.de", password="pw", device_name="Echo")
        ctrl._ready = True
        ctrl._last_update_time = time.time() - 200
        with patch("core.alexa_controller.log") as mock_log:
            ctrl.check_absence(timeout_seconds=180.0)
            mock_log.warning.assert_called_once()

    def test_no_duplicate_warning(self):
        ctrl = AlexaController(email="a@b.de", password="pw", device_name="Echo")
        ctrl._ready = True
        ctrl._last_update_time = time.time() - 200
        with patch("core.alexa_controller.log") as mock_log:
            ctrl.check_absence(timeout_seconds=180.0)
            ctrl.check_absence(timeout_seconds=180.0)
            assert mock_log.warning.call_count == 1

    def test_update_resets_absence(self):
        ctrl = AlexaController(email="a@b.de", password="pw", device_name="Echo")
        ctrl._ready = True
        ctrl._last_update_time = time.time() - 200
        ctrl._absence_warned = True
        ctrl.update(0.5, 0.5, "happy")
        assert ctrl._absence_warned is False


class TestAlexaControllerReadyProperty:
    def test_not_ready_initially(self):
        ctrl = AlexaController(email="a@b.de", password="pw", device_name="Echo")
        assert ctrl.ready is False

    def test_ready_after_login(self):
        ctrl = AlexaController(email="a@b.de", password="pw", device_name="Echo")
        ctrl._ready = True
        assert ctrl.ready is True


class TestAlexaControllerPasswordCleanup:
    def test_password_cleared_after_login(self):
        """_main_async soll Passwort nach Login löschen."""
        ctrl = AlexaController(email="a@b.de", password="secret", device_name="Echo")

        async def fake_login(self_inner):
            self_inner._ready = True

        with (
            patch.dict("sys.modules", {"alexapy": MagicMock()}),
            patch.object(
                AlexaController,
                "_initialize_login",
                new=fake_login,
            ),
        ):
            loop = asyncio.new_event_loop()

            # _stop_event wird in _main_async erstellt; wir setzen es nach kurzem Delay
            async def run():
                task = asyncio.ensure_future(ctrl._main_async())
                await asyncio.sleep(0.05)
                ctrl._stop_event.set()
                await task

            loop.run_until_complete(run())
            loop.close()

        assert ctrl._password == ""
