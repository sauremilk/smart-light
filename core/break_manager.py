"""Pausen-Manager mit Muedigkeitserkennung und Pomodoro-Unterstuetzung.

Funktionen:
  1. Muedigkeitsbasierte Pausen-Empfehlung: erkennt aus kognitivem Zustand
     (FATIGUE) und kontinuierlicher Arbeitszeit den Moment fuer eine Pause.
  2. Pomodoro-Timer: optionale Arbeit/Pausen-Zyklen (25/5 oder benutzerdefiniert).
  3. Erholungs-Tracking: misst ob eine Pause tatsaechlich erholsam war
     (Zustandswechsel von FATIGUE → FOCUS/NEUTRAL innerhalb Pausenzeit).
  4. Licht-Signale: Pausen werden ueber sanftes Pulsieren (amber) signalisiert.

Ausgabe: BreakEvent-Objekte die main.py fuer Overlay-Anzeige und Modus-Wechsel nutzt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class BreakEvent:
    """Snapshot des aktuellen Pausen-Zustands."""

    break_recommended: bool  # Pause jetzt empfohlen
    break_active: bool  # Nutzer ist gerade in Pause
    reason: str  # Grund: "fatigue" | "pomodoro" | "manual" | ""
    work_duration_s: float  # Sekunden seit letzter Pause
    break_duration_s: float  # Sekunden in aktueller Pause (0 wenn nicht)
    pomodoro_cycle: int  # Aktueller Pomodoro-Zyklus (0 wenn aus)
    recovery_quality: float  # 0..1 wie gut die letzte Pause war


class BreakManager:
    """Verwaltet Arbeits-/Pausenzyklen und Muedigkeitserkennung.

    Parameters
    ----------
    max_work_minutes : float
        Maximale Arbeitszeit ohne Pause (Default: 50 min).
    fatigue_trigger_s : float
        Sekunden im FATIGUE-Zustand bis Pause empfohlen wird (Default: 120s).
    min_break_minutes : float
        Mindestdauer einer Pause (Default: 5 min).
    pomodoro_enabled : bool
        Pomodoro-Modus aktivieren.
    pomodoro_work_minutes : float
        Arbeitszeit pro Pomodoro-Zyklus (Default: 25 min).
    pomodoro_break_minutes : float
        Pausenzeit pro Zyklus (Default: 5 min).
    pomodoro_long_break_minutes : float
        Lange Pause nach N Zyklen (Default: 15 min).
    pomodoro_long_break_after : int
        Anzahl Zyklen bis zur langen Pause (Default: 4).
    """

    def __init__(
        self,
        max_work_minutes: float = 50.0,
        fatigue_trigger_s: float = 120.0,
        min_break_minutes: float = 5.0,
        pomodoro_enabled: bool = False,
        pomodoro_work_minutes: float = 25.0,
        pomodoro_break_minutes: float = 5.0,
        pomodoro_long_break_minutes: float = 15.0,
        pomodoro_long_break_after: int = 4,
    ):
        self._max_work_s = max_work_minutes * 60.0
        self._fatigue_trigger = fatigue_trigger_s
        self._min_break_s = min_break_minutes * 60.0

        # Pomodoro
        self._pomo_enabled = pomodoro_enabled
        self._pomo_work_s = pomodoro_work_minutes * 60.0
        self._pomo_break_s = pomodoro_break_minutes * 60.0
        self._pomo_long_break_s = pomodoro_long_break_minutes * 60.0
        self._pomo_long_after = pomodoro_long_break_after

        # State
        self._work_start = time.monotonic()
        self._break_start: float | None = None
        self._fatigue_counter = 0.0
        self._last_update = time.monotonic()
        self._break_recommended = False
        self._break_reason = ""
        self._pomo_cycle = 0
        self._recovery_quality = 0.0
        self._recovery_samples = 0
        self._recovery_positive = 0
        self._dismissed = False  # Nutzer hat Empfehlung abgelehnt

    @property
    def is_break_active(self) -> bool:
        return self._break_start is not None

    def update(self, cognitive_state: str) -> BreakEvent:
        """Aktualisiert den Pausen-Status basierend auf kognitivem Zustand.

        Parameters
        ----------
        cognitive_state : str
            Aktueller kognitiver Zustand (FOCUS/FLOW/FATIGUE/STRESS/NEUTRAL).

        Returns BreakEvent mit aktuellem Status.
        """
        now = time.monotonic()
        dt = now - self._last_update
        self._last_update = now

        # ── In Pause ──
        if self._break_start is not None:
            break_dur = now - self._break_start

            # Recovery-Qualitaet tracken
            self._recovery_samples += 1
            if cognitive_state in ("FOCUS", "FLOW", "NEUTRAL"):
                self._recovery_positive += 1

            # Pause beenden wenn Mindestdauer erreicht und Zustand erholt
            break_target = self._current_break_target()
            if break_dur >= break_target:
                self._end_break(now)

            return self._build_event(now)

        # ── Arbeitsphase ──
        work_dur = now - self._work_start

        # Muedigkeits-Counter
        if cognitive_state == "FATIGUE":
            self._fatigue_counter += dt
        else:
            # Langsam abklingen bei Nicht-Muedigkeit
            self._fatigue_counter = max(0.0, self._fatigue_counter - dt * 0.3)

        # Pausen-Empfehlung pruefen
        if not self._dismissed:
            reason = ""

            # Pomodoro-Timer
            if self._pomo_enabled and work_dur >= self._pomo_work_s:
                reason = "pomodoro"

            # Muedigkeit
            elif self._fatigue_counter >= self._fatigue_trigger:
                reason = "fatigue"

            # Maximale Arbeitszeit
            elif work_dur >= self._max_work_s:
                reason = "max_work"

            if reason:
                self._break_recommended = True
                self._break_reason = reason

        return self._build_event(now)

    def start_break(self) -> None:
        """Startet eine Pause manuell oder als Reaktion auf Empfehlung."""
        if self._break_start is None:
            self._break_start = time.monotonic()
            self._break_recommended = False
            self._recovery_samples = 0
            self._recovery_positive = 0
            self._dismissed = False

    def dismiss_break(self) -> None:
        """Lehnt die aktuelle Pausen-Empfehlung ab (fuer 10 Minuten)."""
        self._break_recommended = False
        self._dismissed = True
        self._fatigue_counter = 0.0

    def skip_break(self) -> None:
        """Beendet die aktuelle Pause vorzeitig."""
        if self._break_start is not None:
            self._end_break(time.monotonic())

    def _end_break(self, now: float) -> None:
        """Interne Methode: Pause beenden und Statistiken aktualisieren."""
        if self._recovery_samples > 0:
            self._recovery_quality = self._recovery_positive / self._recovery_samples
        self._break_start = None
        self._work_start = now
        self._fatigue_counter = 0.0
        self._break_recommended = False
        self._dismissed = False
        if self._pomo_enabled:
            self._pomo_cycle += 1

    def _current_break_target(self) -> float:
        """Gibt die Soll-Pausendauer fuer den aktuellen Zyklus zurueck."""
        if self._pomo_enabled:
            if (
                self._pomo_long_after > 0
                and self._pomo_cycle > 0
                and self._pomo_cycle % self._pomo_long_after == 0
            ):
                return self._pomo_long_break_s
            return self._pomo_break_s
        return self._min_break_s

    def _build_event(self, now: float) -> BreakEvent:
        work_dur = now - self._work_start if self._break_start is None else 0.0
        break_dur = (now - self._break_start) if self._break_start is not None else 0.0

        return BreakEvent(
            break_recommended=self._break_recommended,
            break_active=self._break_start is not None,
            reason=self._break_reason if self._break_recommended or self._break_start is not None else "",
            work_duration_s=work_dur,
            break_duration_s=break_dur,
            pomodoro_cycle=self._pomo_cycle if self._pomo_enabled else 0,
            recovery_quality=self._recovery_quality,
        )
