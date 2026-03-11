"""Session-Logging: Aufzeichnung der Sitzungsdaten im JSONL-Format."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from config import ADAPTIVE_TARGET_AROUSAL, ADAPTIVE_TARGET_VALENCE

if TYPE_CHECKING:
    from core.break_manager import BreakEvent

log = logging.getLogger("emotion-light")


def append_session_log(path: str, payload: dict) -> None:
    """Haengt einen JSON-Datensatz atomar an die JSONL-Session-Datei an."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def pseudonymize_identity(value: str | None, salt: str) -> str | None:
    """Ersetzt Identitaeten durch deterministische, gesalzene Hash-IDs."""
    if value is None:
        return None
    raw = f"{salt}:{value}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def build_session_payload(
    *,
    session_start_ts: float,
    participant: str | None,
    session_id: str | None,
    condition: str,
    pseudonymize: bool = False,
    salt: str = "",
    # Emotion-Zustand
    emotion: str,
    confidence: float,
    audio_confidence: float,
    audio_quality: float,
    dynamic_audio_weight: float,
    quality: float,
    low_quality_guardrail: bool,
    fused_v: float,
    fused_a: float,
    # Regulation
    reg_info: dict | None,
    adaptive_enabled: bool,
    # Biometrie
    pupil_dilation: float,
    blink_rate: float,
    cognitive_load: float,
    torso_lean: float,
    shoulder_drop: float,
    head_tilt: float,
    cognitive_state: str,
    cognitive_confidence: float,
    active_mode: str,
    # Pausen
    break_event: BreakEvent | None = None,
    # Feedback
    feedback_data: dict | None = None,
) -> dict:
    """Baut den vollstaendigen Session-Log-Payload zusammen."""
    target_v = reg_info["target_v"] if reg_info is not None else ADAPTIVE_TARGET_VALENCE
    target_a = reg_info["target_a"] if reg_info is not None else ADAPTIVE_TARGET_AROUSAL
    output_v = reg_info["reg_v"] if reg_info is not None else fused_v
    output_a = reg_info["reg_a"] if reg_info is not None else fused_a
    current_dist = float(np.sqrt((target_v - fused_v) ** 2 + (target_a - fused_a) ** 2))
    output_dist = float(np.sqrt((target_v - output_v) ** 2 + (target_a - output_a) ** 2))

    payload: dict = {
        "timestamp": time.time(),
        "runtime_sec": time.time() - session_start_ts,
        "participant": (pseudonymize_identity(participant, salt) if pseudonymize else participant),
        "session_id": (pseudonymize_identity(session_id, salt) if pseudonymize else session_id),
        "condition": condition,
        "adaptive_enabled": adaptive_enabled,
        "emotion": emotion,
        "confidence": float(confidence),
        "audio_confidence": float(audio_confidence),
        "audio_quality": float(audio_quality),
        "audio_weight_dynamic": float(dynamic_audio_weight),
        "model_quality": float(quality),
        "low_quality_guardrail": bool(low_quality_guardrail),
        "current_valence": float(fused_v),
        "current_arousal": float(fused_a),
        "output_valence": float(output_v),
        "output_arousal": float(output_a),
        "target_valence": float(target_v),
        "target_arousal": float(target_a),
        "distance_current": current_dist,
        "distance_output": output_dist,
        "blend": float(reg_info["blend"]) if reg_info is not None else 0.0,
        "at_target": bool(reg_info["at_target"]) if reg_info is not None else False,
        "label": reg_info["label"] if reg_info is not None else "n/a",
        "pupil_dilation": float(pupil_dilation),
        "blink_rate": float(blink_rate),
        "cognitive_load": float(cognitive_load),
        "torso_lean": float(torso_lean),
        "shoulder_drop": float(shoulder_drop),
        "head_tilt": float(head_tilt),
        "cognitive_state": cognitive_state,
        "cognitive_confidence": float(cognitive_confidence),
        "active_mode": active_mode,
        "break_active": break_event.break_active if break_event else False,
        "break_recommended": break_event.break_recommended if break_event else False,
        "break_reason": break_event.reason if break_event else "",
        "work_duration_s": break_event.work_duration_s if break_event else 0.0,
        "recovery_quality": break_event.recovery_quality if break_event else 0.0,
    }

    if feedback_data:
        payload.update(feedback_data)

    return payload
