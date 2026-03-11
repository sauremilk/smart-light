"""Kognitiver Zustandsklassifikator.

Leitet aus Sensorwerten (HRV, Blink-Rate, kognitive Last, Pose, V/A)
einen diskreten kognitiven Zustand ab:

  FOCUS   – konzentriertes Arbeiten (niedrige Blink-Rate, hohe Aktivitaet,
            Vorneigung, moderate HR)
  FLOW    – optimaler Leistungszustand (stabil bei Ziel-V/A, niedrige Blink,
            gleichmaessige Aktivitaet)
  FATIGUE – Muedigkeit/Erschoepfung (hohe Blink-Rate, Schulterabsenkung,
            niedrige HR-Variabilitaet)
  STRESS  – Anspannung (hohe HR, schnelle Atmung, negative Valence, hohe
            Arousal)
  NEUTRAL – kein klarer Zustand erkennbar

Jeder Zustand wird mit einem Konfidenzwert (0-1) versehen.
Die Klassifikation nutzt ein gewichtetes Scoring-Modell, kein ML.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class CognitiveState:
    """Immutable snapshot eines kognitiven Zustands."""

    state: str  # FOCUS | FLOW | FATIGUE | STRESS | NEUTRAL
    confidence: float  # 0.0 - 1.0
    scores: dict  # Alle Zustandsscores
    duration_s: float  # Sekunden im aktuellen Zustand


# Schwellwerte (koennen spaeter konfigurierbar gemacht werden)
_FOCUS_THRESHOLD = 0.45
_FLOW_THRESHOLD = 0.55
_FATIGUE_THRESHOLD = 0.45
_STRESS_THRESHOLD = 0.45


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


class CognitiveClassifier:
    """Regelbasierter Klassifikator fuer kognitive Zustaende.

    Eingabe-Signale (alle optional, fehlende werden ignoriert):
      hr_bpm          – Herzfrequenz (Beats per Minute)
      hrv_rmssd       – HRV in Millisekunden
      blink_rate      – Blinzelrate pro Minute
      cognitive_load  – 0..1 aus Aktivitaets-Analyzer
      torso_lean      – -1..+1 (positiv = vorgelehnt)
      shoulder_drop   – 0..+1 (hoch = starke Absenkung)
      valence         – -1..+1
      arousal         – -1..+1
      br_bpm          – Atemfrequenz pro Minute
      at_target       – True wenn Regulator am Ziel
    """

    # EMA-Alpha fuer Score-Glaettung
    _SCORE_EMA = 0.25

    def __init__(self, stability_window_s: float = 30.0):
        self._current_state = "NEUTRAL"
        self._state_since = time.monotonic()
        self._stability_window = stability_window_s

        # Geglättete Scores
        self._ema_scores: dict[str, float] = {
            "FOCUS": 0.0,
            "FLOW": 0.0,
            "FATIGUE": 0.0,
            "STRESS": 0.0,
            "NEUTRAL": 0.2,
        }

        # Zustandshistorie fuer Flow-Erkennung (letzte N Sekunden)
        self._state_history: deque[tuple[float, str]] = deque(maxlen=300)

    def update(
        self,
        *,
        hr_bpm: float = 0.0,
        hrv_rmssd: float = 0.0,
        blink_rate: float = 0.0,
        cognitive_load: float = 0.0,
        torso_lean: float = 0.0,
        shoulder_drop: float = 0.0,
        valence: float = 0.0,
        arousal: float = 0.0,
        br_bpm: float = 0.0,
        at_target: bool = False,
    ) -> CognitiveState:
        """Aktualisiert den Zustand basierend auf aktuellen Sensorwerten.

        Returns CognitiveState mit aktuellem Zustand und Konfidenz.
        """
        raw = self._compute_raw_scores(
            hr_bpm=hr_bpm,
            hrv_rmssd=hrv_rmssd,
            blink_rate=blink_rate,
            cognitive_load=cognitive_load,
            torso_lean=torso_lean,
            shoulder_drop=shoulder_drop,
            valence=valence,
            arousal=arousal,
            br_bpm=br_bpm,
            at_target=at_target,
        )

        # EMA-Glaettung der Scores
        a = self._SCORE_EMA
        for k in self._ema_scores:
            self._ema_scores[k] = (1.0 - a) * self._ema_scores[k] + a * raw.get(k, 0.0)

        # Bester Zustand waehlen
        best_state = max(self._ema_scores, key=self._ema_scores.get)  # type: ignore[arg-type]
        best_score = self._ema_scores[best_state]

        # Mindest-Schwelle pruefen
        thresholds = {
            "FOCUS": _FOCUS_THRESHOLD,
            "FLOW": _FLOW_THRESHOLD,
            "FATIGUE": _FATIGUE_THRESHOLD,
            "STRESS": _STRESS_THRESHOLD,
            "NEUTRAL": 0.0,
        }
        if best_score < thresholds.get(best_state, 0.0):
            best_state = "NEUTRAL"
            best_score = self._ema_scores["NEUTRAL"]

        # Zustandswechsel tracken
        now = time.monotonic()
        if best_state != self._current_state:
            self._current_state = best_state
            self._state_since = now

        self._state_history.append((now, self._current_state))

        duration = now - self._state_since

        return CognitiveState(
            state=self._current_state,
            confidence=_clamp01(best_score),
            scores=dict(self._ema_scores),
            duration_s=duration,
        )

    def _compute_raw_scores(
        self,
        *,
        hr_bpm: float,
        hrv_rmssd: float,
        blink_rate: float,
        cognitive_load: float,
        torso_lean: float,
        shoulder_drop: float,
        valence: float,
        arousal: float,
        br_bpm: float,
        at_target: bool,
    ) -> dict[str, float]:
        """Berechnet Roh-Scores fuer jeden kognitiven Zustand."""

        scores: dict[str, float] = {}

        # ── FOCUS ──
        # Niedrige Blink-Rate, hohe Aktivitaet, Vorneigung, moderate Arousal
        focus = 0.0
        n_focus = 0

        if blink_rate > 0:
            # < 10/min = stark fokussiert, 10-15 = normal, > 20 = unfokussiert
            focus += _clamp01(1.0 - (blink_rate - 8.0) / 15.0)
            n_focus += 1

        if cognitive_load > 0:
            # Hohe kognitive Last = vermutlich fokussiert
            focus += _clamp01(cognitive_load)
            n_focus += 1

        if torso_lean > 0:
            # Vorneigung = Engagement
            focus += _clamp01(torso_lean * 2.0)
            n_focus += 1

        if arousal != 0:
            # Moderate Arousal (0.1 - 0.5) = optimal fuer Fokus
            focus += _clamp01(1.0 - abs(arousal - 0.3) * 2.0)
            n_focus += 1

        scores["FOCUS"] = (focus / max(1, n_focus)) if n_focus > 0 else 0.0

        # ── FLOW ──
        # Stabiler Fokus-Zustand + am Regulationsziel + positive Valence
        flow = 0.0
        n_flow = 0

        # Baseline: Fokus-Score muss hoch sein
        flow += scores["FOCUS"] * 0.6
        n_flow += 1

        if at_target:
            # Am Regulationsziel = innerlich im Gleichgewicht
            flow += 0.8
            n_flow += 1

        if valence > 0:
            # Positive Stimmung
            flow += _clamp01(valence)
            n_flow += 1

        # Stabilitaet: wie lange schon im Fokus?
        focus_duration = self._get_recent_state_duration("FOCUS", window_s=120.0)
        if focus_duration > 0:
            # > 60s im Fokus = guter Flow-Indikator
            flow += _clamp01(focus_duration / 60.0)
            n_flow += 1

        if hrv_rmssd > 0:
            # Moderate HRV (30-60ms) = kohaerenter Zustand
            flow += _clamp01(1.0 - abs(hrv_rmssd - 45.0) / 30.0)
            n_flow += 1

        scores["FLOW"] = (flow / max(1, n_flow)) if n_flow > 0 else 0.0

        # ── FATIGUE ──
        # Hohe Blink-Rate, Schulterabsenkung, niedrige Aktivitaet, niedrige Arousal
        fatigue = 0.0
        n_fat = 0

        if blink_rate > 0:
            # > 25/min = muede, > 35 = sehr muede
            fatigue += _clamp01((blink_rate - 20.0) / 15.0)
            n_fat += 1

        if shoulder_drop > 0:
            # Starke Schulterabsenkung = Erschoepfung
            fatigue += _clamp01(shoulder_drop * 3.0)
            n_fat += 1

        if cognitive_load >= 0:
            # Niedrige Aktivitaet = moeglicherweise muede
            fatigue += _clamp01(1.0 - cognitive_load * 2.0)
            n_fat += 1

        if arousal != 0:
            # Niedrige Arousal = Energiemangel
            fatigue += _clamp01(-arousal * 0.8)
            n_fat += 1

        if valence < 0:
            fatigue += _clamp01(-valence * 0.3)
            n_fat += 1

        scores["FATIGUE"] = (fatigue / max(1, n_fat)) if n_fat > 0 else 0.0

        # ── STRESS ──
        # Hohe HR, schnelle Atmung, negative Valence, hohe Arousal
        stress = 0.0
        n_str = 0

        if hr_bpm > 0:
            # > 85 bpm in Ruhe = erhoehter Stress
            stress += _clamp01((hr_bpm - 75.0) / 30.0)
            n_str += 1

        if br_bpm > 0:
            # > 18/min = schnelle Atmung
            stress += _clamp01((br_bpm - 15.0) / 10.0)
            n_str += 1

        if valence < 0:
            stress += _clamp01(-valence)
            n_str += 1

        if arousal > 0:
            # Hohe Arousal bei negativer Valence = Stress
            stress += _clamp01(arousal)
            n_str += 1

        if hrv_rmssd > 0:
            # Niedrige HRV = Stress-Indikator
            stress += _clamp01(1.0 - hrv_rmssd / 50.0)
            n_str += 1

        scores["STRESS"] = (stress / max(1, n_str)) if n_str > 0 else 0.0

        # ── NEUTRAL ──
        # Invertierter Maximalwert der anderen Zustaende
        max_other = max(scores["FOCUS"], scores["FLOW"], scores["FATIGUE"], scores["STRESS"])
        scores["NEUTRAL"] = _clamp01(1.0 - max_other)

        return scores

    def _get_recent_state_duration(self, state: str, window_s: float = 120.0) -> float:
        """Summiert Sekunden im angegebenen Zustand innerhalb des Zeitfensters."""
        now = time.monotonic()
        cutoff = now - window_s
        total = 0.0
        prev_time = cutoff
        for ts, s in self._state_history:
            if ts < cutoff:
                continue
            if s == state:
                total += ts - prev_time
            prev_time = ts
        # Letztes Segment bis jetzt
        if self._state_history and self._state_history[-1][1] == state:
            total += now - prev_time
        return total

    @property
    def current_state(self) -> str:
        return self._current_state

    @property
    def state_duration(self) -> float:
        return time.monotonic() - self._state_since
