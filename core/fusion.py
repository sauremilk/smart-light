"""Multimodale Sensor-Fusion: Offset-Anwendung und Transition-Berechnung.

Dieses Modul buendelt die feingranularen Lichtparameter-Anpassungen,
die aus den verschiedenen Sensormodalitaeten (HRV, Atmung, Pupillen,
Blink-Rate, Pose, Aktivitaet, Prosodie) berechnet werden.
"""

from __future__ import annotations

from config import (
    ACTIVITY_AROUSAL_INFLUENCE,
    ACTIVITY_TRANSITION_INFLUENCE,
    BLINK_RATE_FATIGUE_THRESHOLD,
    BLINK_RATE_FOCUS_THRESHOLD,
    BLINK_VALENCE_INFLUENCE,
    BREATHING_TRANSITION_INFLUENCE,
    POSE_WEIGHT,
    PROSODIC_AROUSAL_INFLUENCE,
    PROSODIC_PITCH_CALM_HZ,
    PROSODIC_PITCH_STRESS_HZ,
    PROSODIC_SPEECH_RATE_HIGH,
    PROSODIC_SPEECH_RATE_LOW,
    PUPIL_AROUSAL_INFLUENCE,
    PUPIL_DILATION_BASELINE,
    PUPIL_OFFSET_CLAMP,
    SHOULDER_DROP_VALENCE_INFLUENCE,
    TORSO_LEAN_AROUSAL_INFLUENCE,
    TRANSITION_TIME,
    TREND_INFLUENCE,
    VA_BRI_HIGH,
    VA_BRI_LOW,
    VA_CT_AROUSAL_SHIFT,
    VA_CT_NEGATIVE,
    VA_CT_NEUTRAL,
    VA_CT_POSITIVE,
    VA_HUE_NEUTRAL,
    VA_SAT_HIGH,
    VA_SAT_LOW,
)


def circadian_va_to_light(
    valence: float,
    arousal: float,
    hue_neg: int,
    hue_pos: int,
    bri_max: int,
    ct_neg: int = VA_CT_NEGATIVE,
    ct_pos: int = VA_CT_POSITIVE,
) -> dict:
    """Wie valence_arousal_to_light, aber mit zirkadianen Hue-/Bri-/CT-Overrides."""
    if valence >= 0:
        hue = VA_HUE_NEUTRAL + (hue_pos - VA_HUE_NEUTRAL) * valence
        ct = VA_CT_NEUTRAL + (ct_pos - VA_CT_NEUTRAL) * valence
    else:
        hue = VA_HUE_NEUTRAL + (hue_neg - VA_HUE_NEUTRAL) * (-valence)
        ct = VA_CT_NEUTRAL + (ct_neg - VA_CT_NEUTRAL) * (-valence)
    ct -= arousal * VA_CT_AROUSAL_SHIFT
    a_norm = (arousal + 1.0) / 2.0
    bri = VA_BRI_LOW + (min(bri_max, VA_BRI_HIGH) - VA_BRI_LOW) * a_norm
    sat = VA_SAT_LOW + (VA_SAT_HIGH - VA_SAT_LOW) * a_norm
    return {
        "hue": max(0, min(65535, int(round(hue)))),
        "bri": max(1, min(254, int(round(bri)))),
        "sat": max(0, min(254, int(round(sat)))),
        "ct": max(153, min(500, int(round(ct)))),
    }


# ---------------------------------------------------------------------------
# Modality-Offset-Anwendung
# ---------------------------------------------------------------------------


def apply_modality_offsets(
    params: dict,
    *,
    pose_arousal_offset: float = 0.0,
    hrv_arousal_offset: float = 0.0,
    breathing_arousal_offset: float = 0.0,
    pupil_dilation: float = 0.0,
    blink_rate: float = 0.0,
    torso_lean: float = 0.0,
    shoulder_drop: float = 0.0,
    cognitive_load: float = 0.0,
    pitch_mean_hz: float = 0.0,
    speech_rate: float = 0.0,
    pacer_factor: float = 1.0,
    use_pupil_blink: bool = False,
    use_extended_pose: bool = False,
    use_prosodic: bool = False,
    has_activity: bool = False,
) -> dict:
    """Wendet alle Sensor-Offsets auf die Basis-Lichtparameter an.

    Veraendert *params* in-place und gibt es zurueck.
    """
    p = params  # Kurzreferenz

    # Pose-Arousal → Helligkeit
    if pose_arousal_offset != 0.0 and POSE_WEIGHT > 0:
        bri_adj = int(p["bri"] * (1.0 + pose_arousal_offset * POSE_WEIGHT))
        p["bri"] = max(1, min(254, bri_adj))

    # HRV-Arousal → Helligkeit
    if hrv_arousal_offset != 0.0:
        bri_adj = int(p["bri"] * (1.0 + hrv_arousal_offset))
        p["bri"] = max(1, min(254, bri_adj))

    # Atemfrequenz → Helligkeit + Saettigung
    if breathing_arousal_offset != 0.0:
        bri_adj = int(p["bri"] * (1.0 + breathing_arousal_offset))
        p["bri"] = max(1, min(254, bri_adj))
        sat_adj = int(p["sat"] * (1.0 + breathing_arousal_offset * 0.5))
        p["sat"] = max(0, min(254, sat_adj))

    # Pupille → Helligkeit
    if use_pupil_blink and pupil_dilation > 0:
        pupil_offset = (
            (pupil_dilation - PUPIL_DILATION_BASELINE) / max(0.1, PUPIL_DILATION_BASELINE)
        ) * PUPIL_AROUSAL_INFLUENCE
        pupil_offset = max(-PUPIL_OFFSET_CLAMP, min(PUPIL_OFFSET_CLAMP, pupil_offset))
        if pupil_offset != 0.0:
            bri_adj = int(p["bri"] * (1.0 + pupil_offset))
            p["bri"] = max(1, min(254, bri_adj))

    # Blink-Rate → Helligkeit (Muedigkeit / Fokus)
    if use_pupil_blink and blink_rate > 0:
        if blink_rate > BLINK_RATE_FATIGUE_THRESHOLD:
            fatigue_factor = min(1.0, (blink_rate - BLINK_RATE_FATIGUE_THRESHOLD) / 15.0)
            p["bri"] = max(
                1,
                int(p["bri"] * (1.0 - fatigue_factor * BLINK_VALENCE_INFLUENCE)),
            )
        elif blink_rate < BLINK_RATE_FOCUS_THRESHOLD:
            focus_factor = min(1.0, (BLINK_RATE_FOCUS_THRESHOLD - blink_rate) / 8.0)
            p["bri"] = min(
                254,
                int(p["bri"] * (1.0 + focus_factor * BLINK_VALENCE_INFLUENCE)),
            )

    # Erweiterte Pose-Signale
    if use_extended_pose:
        if torso_lean > 0 and TORSO_LEAN_AROUSAL_INFLUENCE > 0:
            lean_boost = torso_lean * TORSO_LEAN_AROUSAL_INFLUENCE
            p["bri"] = min(254, int(p["bri"] * (1.0 + lean_boost)))
        if shoulder_drop > 0 and SHOULDER_DROP_VALENCE_INFLUENCE > 0:
            drop_dim = shoulder_drop * SHOULDER_DROP_VALENCE_INFLUENCE
            p["sat"] = max(0, int(p["sat"] * (1.0 - drop_dim)))

    # Kognitive Last → Helligkeit
    if has_activity and cognitive_load > 0 and ACTIVITY_AROUSAL_INFLUENCE > 0:
        activity_boost = cognitive_load * ACTIVITY_AROUSAL_INFLUENCE
        p["bri"] = min(254, int(p["bri"] * (1.0 + activity_boost)))

    # Prosodie → Helligkeit + Saettigung
    if use_prosodic and PROSODIC_AROUSAL_INFLUENCE > 0:
        if pitch_mean_hz > 0:
            mid_hz = (PROSODIC_PITCH_STRESS_HZ + PROSODIC_PITCH_CALM_HZ) / 2.0
            range_hz = (PROSODIC_PITCH_STRESS_HZ - PROSODIC_PITCH_CALM_HZ) / 2.0
            if range_hz > 0:
                pitch_offset = ((pitch_mean_hz - mid_hz) / range_hz) * PROSODIC_AROUSAL_INFLUENCE
                pitch_offset = max(-0.15, min(0.15, pitch_offset))
                p["bri"] = max(1, min(254, int(p["bri"] * (1.0 + pitch_offset))))
        if speech_rate > 0:
            mid_sr = (PROSODIC_SPEECH_RATE_HIGH + PROSODIC_SPEECH_RATE_LOW) / 2.0
            range_sr = (PROSODIC_SPEECH_RATE_HIGH - PROSODIC_SPEECH_RATE_LOW) / 2.0
            if range_sr > 0:
                sr_offset = ((speech_rate - mid_sr) / range_sr) * PROSODIC_AROUSAL_INFLUENCE * 0.5
                sr_offset = max(-0.1, min(0.1, sr_offset))
                p["sat"] = max(0, min(254, int(p["sat"] * (1.0 + sr_offset))))

    # Atem-Pacer Pulsation
    if pacer_factor != 1.0:
        p["bri"] = max(1, min(254, int(p["bri"] * pacer_factor)))

    return p


# ---------------------------------------------------------------------------
# Transition-Zeit-Berechnung
# ---------------------------------------------------------------------------


def compute_transition(
    *,
    trend_v: float = 0.0,
    breathing_arousal_offset: float = 0.0,
    cognitive_load: float = 0.0,
    has_activity: bool = False,
) -> int:
    """Berechnet die Hue-Transition-Zeit unter Beruecksichtigung aller Modulationen."""
    transition = TRANSITION_TIME

    # Trend → laengere Transition bei negativem Verlauf
    if TREND_INFLUENCE > 0 and trend_v < -0.01:
        transition = int(TRANSITION_TIME * (1.0 + TREND_INFLUENCE * abs(trend_v) * 10))
        transition = min(transition, 50)

    # Atmung → schneller/langsamer je nach Atemfrequenz
    if BREATHING_TRANSITION_INFLUENCE > 0 and breathing_arousal_offset != 0.0:
        factor = 1.0 - (breathing_arousal_offset * BREATHING_TRANSITION_INFLUENCE)
        factor = max(0.6, min(1.8, factor))
        transition = int(round(transition * factor))
        transition = max(1, min(50, transition))

    # Aktivitaet → reaktiveres Licht bei hoher Eingaberate
    if has_activity and cognitive_load > 0 and ACTIVITY_TRANSITION_INFLUENCE > 0:
        act_factor = 1.0 - (cognitive_load * ACTIVITY_TRANSITION_INFLUENCE)
        act_factor = max(0.6, min(1.0, act_factor))
        transition = max(1, min(50, int(round(transition * act_factor))))

    return transition
