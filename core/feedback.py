"""Feedback-System fuer Ground-Truth Sammlung.

Erlaubt dem Benutzer per Tastendruck schnelles Feedback zu geben:
  - Thumbs Up:   aktueller Zustand/Licht ist angenehm
  - Thumbs Down: aktueller Zustand/Licht ist unangenehm
  - Zustand-Override: manuell einen kognitiven Zustand melden

Das Feedback wird in der Session-Log gespeichert und kann spaeter
fuer Kalibrierung und Modell-Verbesserung genutzt werden.

Tastenbelegung (in main.py integriert):
  'f' → positives Feedback (thumbs up)
  'd' → negatives Feedback (thumbs down)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackEntry:
    """Einzelner Feedback-Eintrag."""

    timestamp: float
    kind: str  # "positive" | "negative"
    cognitive_state: str  # Aktueller klassifizierter Zustand
    active_mode: str  # Aktueller Modus
    valence: float
    arousal: float


class FeedbackCollector:
    """Sammelt und verwaltet Benutzer-Feedback.

    Haelt die letzten N Feedback-Eintraege in einem Ring-Buffer
    und berechnet damit eine lokale Zufriedenheitsmetrik.
    """

    def __init__(self, max_entries: int = 200, cooldown_s: float = 3.0):
        self._entries: deque[FeedbackEntry] = deque(maxlen=max_entries)
        self._cooldown_s = cooldown_s
        self._last_feedback_time = 0.0
        self._flash_until = 0.0  # Fuer visuelles Overlay-Feedback
        self._flash_kind = ""

    def record(
        self,
        kind: str,
        *,
        cognitive_state: str = "NEUTRAL",
        active_mode: str = "AUTO",
        valence: float = 0.0,
        arousal: float = 0.0,
    ) -> bool:
        """Zeichnet Feedback auf. Returns True wenn akzeptiert (Cooldown beachten)."""
        now = time.monotonic()
        if now - self._last_feedback_time < self._cooldown_s:
            return False

        entry = FeedbackEntry(
            timestamp=now,
            kind=kind,
            cognitive_state=cognitive_state,
            active_mode=active_mode,
            valence=valence,
            arousal=arousal,
        )
        self._entries.append(entry)
        self._last_feedback_time = now

        # Visuelles Feedback (3 Sekunden Flash)
        self._flash_until = now + 3.0
        self._flash_kind = kind

        return True

    @property
    def flash_active(self) -> tuple[bool, str]:
        """Gibt (aktiv, kind) zurueck fuer Overlay-Anzeige."""
        now = time.monotonic()
        if now < self._flash_until:
            return True, self._flash_kind
        return False, ""

    @property
    def recent_satisfaction(self) -> float:
        """Lokale Zufriedenheit aus den letzten 20 Feedbacks (0..1)."""
        recent = list(self._entries)[-20:]
        if not recent:
            return 0.5  # Neutral bei keinem Feedback
        positive = sum(1 for e in recent if e.kind == "positive")
        return positive / len(recent)

    @property
    def total_count(self) -> int:
        return len(self._entries)

    @property
    def positive_count(self) -> int:
        return sum(1 for e in self._entries if e.kind == "positive")

    @property
    def negative_count(self) -> int:
        return sum(1 for e in self._entries if e.kind == "negative")

    def get_feedback_for_log(self) -> dict | None:
        """Gibt den letzten Feedback-Eintrag als dict fuer Session-Log zurueck.

        Returns None wenn kein neuer Eintrag seit letztem Aufruf.
        """
        if not self._entries:
            return None
        last = self._entries[-1]
        # Nur zurueckgeben wenn innerhalb der letzten 2 Sekunden
        if time.monotonic() - last.timestamp > 2.0:
            return None
        return {
            "feedback_kind": last.kind,
            "feedback_state": last.cognitive_state,
            "feedback_mode": last.active_mode,
            "feedback_valence": last.valence,
            "feedback_arousal": last.arousal,
        }
