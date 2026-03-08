"""Audio quality helpers for noise robustness and dynamic fusion weighting."""

from __future__ import annotations

import math
import numpy as np


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_snr_proxy_db(audio: np.ndarray, noise_floor_rms: float) -> float:
    """Returns a robust SNR proxy in dB based on chunk RMS and adaptive noise floor."""
    if audio.size == 0:
        return -40.0

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    noise = max(1e-6, float(noise_floor_rms))
    return float(20.0 * math.log10(max(rms, 1e-8) / noise))


def quality_from_snr_db(snr_db: float, snr_floor_db: float, snr_ceil_db: float) -> float:
    """Maps SNR proxy to [0,1] quality."""
    lo = float(min(snr_floor_db, snr_ceil_db))
    hi = float(max(snr_floor_db, snr_ceil_db))
    if hi - lo <= 1e-6:
        return 0.0
    return clamp01((float(snr_db) - lo) / (hi - lo))


def effective_audio_weight(
    base_weight: float,
    audio_quality: float,
    audio_confidence: float,
    min_factor: float,
    quality_exponent: float = 1.0,
) -> float:
    """Scales base fusion weight by audio quality and confidence.

    At very low quality/confidence, weight stays near `base_weight * min_factor`.
    At high quality/confidence, weight approaches `base_weight`.
    """
    base = clamp01(base_weight)
    if base <= 0.0:
        return 0.0

    q = clamp01(audio_quality) ** max(0.1, float(quality_exponent))
    c = clamp01(audio_confidence)
    mf = clamp01(min_factor)

    quality_term = mf + (1.0 - mf) * q
    confidence_term = 0.5 + 0.5 * c
    scaled = base * quality_term * confidence_term
    return max(0.0, min(base, float(scaled)))
