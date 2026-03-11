"""Adaptive Emotionsregulation via Licht.

Der EmotionRegulator berechnet aus dem aktuell erkannten Valence/Arousal-Wert
einen *regulierten* Ausgangswert, der den Nutzer sanft Richtung Zielzustand
(standardmäßig: positiv + ruhig) führt.

Mechanismus
-----------
  regulated_v = current_v + blend * (target_v - current_v)
  regulated_a = current_a + blend * (target_a - current_a)

Der `blend`-Parameter startet bei `blend_strength` und wird eskaliert, wenn
nach `progress_timeout` Sekunden keine messbare Annäherung an das Ziel erkennbar
ist. Bei echter Verbesserung fällt blend langsam auf den Ausgangswert zurück.

Quadrantenspezifische Strategien passen Blend-Parameter automatisch an den
emotionalen Zustand an: Stress wird aggressiver reguliert als Freude.
"""

import math
import time


class EmotionRegulator:
    """Reguliert den erkannten Emotionszustand Richtung eines Zielzustands."""

    _LABEL_STABLE = "Stabil \u2713"
    _BLEND_SOFT_RESET_ALPHA = 0.03  # Wie schnell blend auf Ausgangswert zurückfällt

    # Quadrantenspezifische Regulationsparameter:
    # Q1 (+v, +a) Aufgeregt/Freudig:  sanft, nur Arousal senken
    # Q2 (+v, -a) Entspannt/Ruhig:    minimal, fast stabil
    # Q3 (-v, +a) Stress/Angst/Wut:   aggressiv beruhigen
    # Q4 (-v, -a) Traurig/Erschoepft: sanft Valenz heben
    _QUADRANT_PARAMS = {
        "Q1": {"blend_start": 0.20, "blend_max": 0.50, "escalation": 0.08, "timeout": 45.0},
        "Q2": {"blend_start": 0.10, "blend_max": 0.30, "escalation": 0.05, "timeout": 60.0},
        "Q3": {"blend_start": 0.55, "blend_max": 0.90, "escalation": 0.12, "timeout": 20.0},
        "Q4": {"blend_start": 0.40, "blend_max": 0.75, "escalation": 0.10, "timeout": 35.0},
    }

    def __init__(
        self,
        target_v: float,
        target_a: float,
        blend_strength: float,
        blend_max: float,
        progress_timeout: float,
        escalation: float,
        at_target_threshold: float,
    ):
        self._target_v = target_v
        self._target_a = target_a
        self._blend_base = blend_strength
        self._blend_max = blend_max
        self._progress_timeout = progress_timeout
        self._escalation = escalation
        self._at_target_thr = at_target_threshold

        self._blend = blend_strength
        self._prev_dist: float | None = None
        self._no_progress_timer: float = 0.0
        self._last_update_time: float = time.monotonic()

        # Boost-Mechanismus fuer vorausschauende Intervention
        self._boost_factor: float = 1.0
        self._boost_end_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_target(self, target_v: float, target_a: float) -> None:
        """Aktualisiert den Zielzustand (z.B. durch Zirkadian-Profil)."""
        self._target_v = target_v
        self._target_a = target_a

    def boost_blend(self, factor: float = 1.5, duration_s: float = 15.0) -> None:
        """Temporaer den Blend-Faktor verstaerken (vorausschauende Intervention).

        Der Boost-Faktor wird auf die aktuelle Blend-Staerke multipliziert
        und verfaellt nach ``duration_s`` Sekunden mit sanftem Decay.
        """
        self._boost_factor = max(1.0, factor)
        self._boost_end_time = time.monotonic() + duration_s

    def update(self, current_v: float, current_a: float) -> dict:
        """Berechnet den regulierten VA-Punkt und aktualisiert internen Zustand.

        Returns
        -------
        dict mit Schlüsseln:
          reg_v, reg_a     – regulierter Ausgangswert (geht in valence_arousal_to_light)
          current_v/a      – unveränderter Ist-Wert (für Overlay)
          target_v/a       – Zielwert (für Overlay)
          blend            – aktuelle Blend-Stärke
          label            – menschlesbare Beschreibung der Regulationsrichtung
          at_target        – True wenn Ist-Zustand bereits nah am Ziel
        """
        now = time.monotonic()
        dt = now - self._last_update_time
        self._last_update_time = now

        dist = math.sqrt(
            (self._target_v - current_v) ** 2 + (self._target_a - current_a) ** 2
        )

        at_target = dist < self._at_target_thr

        if at_target:
            self._no_progress_timer = 0.0
            self._prev_dist = dist
            self._boost_factor = 1.0
            return {
                "reg_v": current_v,
                "reg_a": current_a,
                "current_v": current_v,
                "current_a": current_a,
                "target_v": self._target_v,
                "target_a": self._target_a,
                "blend": 0.0,
                "label": self._LABEL_STABLE,
                "at_target": True,
            }

        # Quadrantenspezifische Parameter anwenden
        qp = self._get_quadrant_params(current_v, current_a)
        effective_blend_max = qp["blend_max"]
        effective_escalation = qp["escalation"]
        effective_timeout = qp["timeout"]

        # Fortschritts-Tracking
        if self._prev_dist is not None:
            improved = dist < (self._prev_dist - 0.005)  # mind. 0.005 Verbesserung
            if improved:
                # Blend sanft Richtung quadrantenspezifischem Ausgangswert zurückführen
                target_blend = qp["blend_start"]
                self._blend = (
                    (1.0 - self._BLEND_SOFT_RESET_ALPHA) * self._blend
                    + self._BLEND_SOFT_RESET_ALPHA * target_blend
                )
                self._no_progress_timer = max(0.0, self._no_progress_timer - dt * 2)
            else:
                self._no_progress_timer += dt

            if self._no_progress_timer >= effective_timeout:
                self._blend = min(effective_blend_max, self._blend + effective_escalation)
                self._no_progress_timer = 0.0

        self._prev_dist = dist

        # Boost-Faktor anwenden (vorausschauende Intervention)
        if now > self._boost_end_time:
            # Sanfter Decay: Boost-Faktor exponentiell zurueck auf 1.0
            self._boost_factor = 1.0 + (self._boost_factor - 1.0) * 0.95
            if self._boost_factor < 1.01:
                self._boost_factor = 1.0

        # Regulierten Ausgangswert berechnen
        b = min(effective_blend_max, self._blend * self._boost_factor)
        reg_v = current_v + b * (self._target_v - current_v)
        reg_a = current_a + b * (self._target_a - current_a)

        label = self._compute_label(current_v, current_a)

        return {
            "reg_v": reg_v,
            "reg_a": reg_a,
            "current_v": current_v,
            "current_a": current_a,
            "target_v": self._target_v,
            "target_a": self._target_a,
            "blend": b,
            "label": label,
            "at_target": False,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_quadrant_params(self, current_v: float, current_a: float) -> dict:
        """Gibt quadrantenspezifische Blend-Parameter basierend auf Emotionszustand zurueck."""
        if current_v >= 0 and current_a >= 0:
            return self._QUADRANT_PARAMS["Q1"]  # Aufgeregt/Freudig
        if current_v >= 0 and current_a < 0:
            return self._QUADRANT_PARAMS["Q2"]  # Entspannt/Ruhig
        if current_v < 0 and current_a >= 0:
            return self._QUADRANT_PARAMS["Q3"]  # Stress/Angst/Wut
        return self._QUADRANT_PARAMS["Q4"]      # Traurig/Erschoepft

    def _compute_label(self, current_v: float, current_a: float) -> str:
        dv = self._target_v - current_v  # >0 → aufheitern nötig
        da = self._target_a - current_a  # <0 → beruhigen nötig, >0 → aktivieren nötig

        need_brighten = dv > 0.25
        need_calm = da < -0.25
        need_activate = da > 0.25

        if need_brighten and need_calm:
            return "Aufheitern & Beruhigen"
        if need_brighten and need_activate:
            return "Aufheitern & Aktivieren"
        if need_brighten:
            return "Aufheitern"
        if need_calm:
            return "Beruhigen"
        if need_activate:
            return "Aktivieren"
        return "Feinabstimmung"
