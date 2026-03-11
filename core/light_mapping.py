"""Pure light-mapping and multimodal fusion helpers."""

import math
import time

from config import (
    EMA_MIN_WEIGHT,
    EMOTION_MAP,
    FALLBACK_LIGHT,
    USE_VALENCE_AROUSAL,
    VA_BRI_HIGH,
    VA_BRI_LOW,
    VA_CT_AROUSAL_SHIFT,
    VA_CT_NEGATIVE,
    VA_CT_NEUTRAL,
    VA_CT_POSITIVE,
    VA_HUE_NEGATIVE,
    VA_HUE_NEUTRAL,
    VA_HUE_POSITIVE,
    VA_SAT_HIGH,
    VA_SAT_LOW,
    VALENCE_AROUSAL_MAP,
)

_TWO_PI = 2.0 * math.pi
_HUE_MAX = 65535


def _lerp(a: float, b: float, t: float) -> float:
    """Lineare Interpolation a->b mit t in [0, 1]."""
    return a + (b - a) * t


def valence_arousal_to_light(valence: float, arousal: float) -> dict:
    """Wandelt Valence (-1..+1) und Arousal (-1..+1) in Hue/Bri/Sat/ct um.

    Gibt zusaetzlich ``ct`` (Mirek-Farbtemperatur, 153–500) zurueck:
    negative Valence → warmeres Amber, positive Valence → kuehles Weiss (Fokus).
    Hohes Arousal verschiebt ct leicht Richtung kuehler (aktivierend).
    ``ct`` funktioniert auf White-Ambiance- UND Farb-Lampen.
    """
    if valence >= 0:
        hue = _lerp(VA_HUE_NEUTRAL, VA_HUE_POSITIVE, valence)
        ct = _lerp(VA_CT_NEUTRAL, VA_CT_POSITIVE, valence)
    else:
        hue = _lerp(VA_HUE_NEUTRAL, VA_HUE_NEGATIVE, -valence)
        ct = _lerp(VA_CT_NEUTRAL, VA_CT_NEGATIVE, -valence)

    # Hohes Arousal kühlt die Farbtemperatur leicht (aktivierender Effekt)
    ct -= arousal * VA_CT_AROUSAL_SHIFT

    a_norm = (arousal + 1.0) / 2.0
    bri = _lerp(VA_BRI_LOW, VA_BRI_HIGH, a_norm)
    sat = _lerp(VA_SAT_LOW, VA_SAT_HIGH, a_norm)

    return {
        "hue": max(0, min(_HUE_MAX, int(round(hue)))),
        "bri": max(1, min(254, int(round(bri)))),
        "sat": max(0, min(254, int(round(sat)))),
        "ct": max(153, min(500, int(round(ct)))),
    }


def blend_emotion_colors(ema_vector: dict) -> dict:
    """Berechnet Lichtparameter aus dem EMA-Emotionsvektor."""
    filtered = {e: w for e, w in ema_vector.items() if w >= EMA_MIN_WEIGHT and e in EMOTION_MAP}
    if not filtered:
        return FALLBACK_LIGHT.copy()

    total = sum(filtered.values())
    weights = {e: w / total for e, w in filtered.items()}

    if USE_VALENCE_AROUSAL:
        valence = sum(w * VALENCE_AROUSAL_MAP[e]["valence"] for e, w in weights.items())
        arousal = sum(w * VALENCE_AROUSAL_MAP[e]["arousal"] for e, w in weights.items())
        return valence_arousal_to_light(valence, arousal)

    x = sum(w * math.cos(EMOTION_MAP[e]["hue"] * _TWO_PI / _HUE_MAX) for e, w in weights.items())
    y = sum(w * math.sin(EMOTION_MAP[e]["hue"] * _TWO_PI / _HUE_MAX) for e, w in weights.items())
    blended_hue = math.atan2(y, x) * _HUE_MAX / _TWO_PI
    if blended_hue < 0:
        blended_hue += _HUE_MAX

    blended_bri = sum(w * EMOTION_MAP[e]["bri"] for e, w in weights.items())
    blended_sat = sum(w * EMOTION_MAP[e]["sat"] for e, w in weights.items())

    return {
        "hue": max(0, min(_HUE_MAX, int(round(blended_hue)))),
        "bri": max(1, min(254, int(round(blended_bri)))),
        "sat": max(0, min(254, int(round(blended_sat)))),
    }


def compute_va_from_ema(ema_vector: dict) -> tuple:
    """Berechnet (valence, arousal) aus einem normierten EMA-Vektor.

    Gibt (0.0, 0.0) zurück wenn der Vektor leer oder alle Gewichte unter EMA_MIN_WEIGHT sind.
    """
    filtered = {
        e: w for e, w in ema_vector.items() if w >= EMA_MIN_WEIGHT and e in VALENCE_AROUSAL_MAP
    }
    if not filtered:
        return 0.0, 0.0
    total = sum(filtered.values())
    weights = {e: w / total for e, w in filtered.items()}
    valence = sum(w * VALENCE_AROUSAL_MAP[e]["valence"] for e, w in weights.items())
    arousal = sum(w * VALENCE_AROUSAL_MAP[e]["arousal"] for e, w in weights.items())
    return valence, arousal


def fuse_modalities(
    video_ema: dict,
    audio_ema: dict | None,
    pose_arousal_offset: float,
    audio_weight: float,
    face_mesh_scores: dict | None = None,
    face_mesh_weight: float = 0.0,
) -> dict:
    """Fusioniert Video-EMA mit optionalem Audio-EMA und Face-Mesh-Scores.

    pose_arousal_offset bleibt Teil der Signatur fuer API-Kompatibilitaet.
    """
    _ = pose_arousal_offset
    emotions = list(EMOTION_MAP.keys())

    # Basis: Video-EMA
    fused = video_ema.copy()

    # Audio-Fusion: gewichtete Mischung
    if audio_ema and audio_weight > 0:
        vw = 1.0 - audio_weight
        aw = audio_weight
        fused = {e: vw * fused.get(e, 0) + aw * audio_ema.get(e, 0) for e in emotions}

    # Face-Mesh-Fusion: gewichtete Mischung
    if face_mesh_scores and face_mesh_weight > 0:
        fm_total = sum(face_mesh_scores.values())
        if fm_total > 0:
            fm_norm = {e: face_mesh_scores.get(e, 0) / fm_total for e in emotions}
            remaining = 1.0 - face_mesh_weight
            fused = {
                e: remaining * fused.get(e, 0) + face_mesh_weight * fm_norm.get(e, 0)
                for e in emotions
            }

    total = sum(fused.values())
    if total > 0:
        fused = {e: v / total for e, v in fused.items()}

    return fused


# ──────────── Atemführungs-Entrainment (Breathing Pacer) ──────────────


class BreathingPacer:
    """Pulsiert Helligkeit bei 0.1 Hz (6/min) zur parasympathischen Atemfuehrung.

    Aktiviert sich wenn die erkannte Atemfrequenz ueber einem Schwellwert liegt
    und blendet sanft ein, um den Nutzer nicht abrupt zu stoeren.
    """

    def __init__(
        self,
        guide_bpm: float = 6.0,
        amplitude: float = 0.08,
        fade_in_seconds: float = 30.0,
    ):
        self._guide_hz = guide_bpm / 60.0
        self._amplitude = amplitude
        self._fade_in_seconds = fade_in_seconds
        self._active_since: float | None = None

    def set_active(self, active: bool) -> None:
        """Setzt den Pacer aktiv/inaktiv mit Fade-In-Tracking."""
        if active and self._active_since is None:
            self._active_since = time.monotonic()
        elif not active:
            self._active_since = None

    @property
    def is_active(self) -> bool:
        return self._active_since is not None

    def get_fade_pct(self) -> float:
        """Gibt den aktuellen Fade-In-Fortschritt zurueck (0.0 - 1.0)."""
        if self._active_since is None:
            return 0.0
        elapsed = time.monotonic() - self._active_since
        return min(1.0, elapsed / max(0.1, self._fade_in_seconds))

    def get_pulsation_factor(self) -> float:
        """Gibt den Helligkeits-Multiplikator zurueck (z.B. 0.92 - 1.08).

        Sinuskurve bei Ziel-Atemfrequenz, mit Fade-In-Modulation.
        Inaktiv: gibt 1.0 zurueck (kein Effekt).
        """
        if self._active_since is None:
            return 1.0
        fade = self.get_fade_pct()
        t = time.monotonic()
        # Sinuswelle: positiver Halbzyklus = Einatmen (heller), negativer = Ausatmen (dunkler)
        pulse = math.sin(2.0 * math.pi * self._guide_hz * t)
        return 1.0 + pulse * self._amplitude * fade


# ──────────── Multi-Licht-Szenen-Komposition ──────────────────────────


def compose_multi_light_scene(primary: dict, role: str) -> dict:
    """Berechnet lichtrollenspezifische Parameter aus dem Primaerlicht.

    Rollen:
      primary – unveraendert
      accent  – leicht waermer (+1500 Hue / +15 Mirek), 90% Saettigung
      ambient – deutlich waermer (+2000 Hue / +30 Mirek), 60% Saettigung, 70% Helligkeit
    """
    if role == "primary":
        return primary.copy()

    result = primary.copy()
    if role == "accent":
        result["hue"] = max(0, min(_HUE_MAX, result["hue"] + 1500))
        result["sat"] = max(0, min(254, int(result["sat"] * 0.9)))
        if "ct" in result:
            result["ct"] = max(153, min(500, result["ct"] + 15))  # etwas waermer
    elif role == "ambient":
        result["hue"] = max(0, min(_HUE_MAX, result["hue"] + 2000))
        result["sat"] = max(0, min(254, int(result["sat"] * 0.6)))
        result["bri"] = max(1, min(254, int(result["bri"] * 0.7)))
        if "ct" in result:
            result["ct"] = max(153, min(500, result["ct"] + 30))  # deutlich waermer

    return result
