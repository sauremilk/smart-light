from analyzers.audio_quality import (
    compute_snr_proxy_db,
    quality_from_snr_db,
    effective_audio_weight,
)

import numpy as np


def test_quality_from_snr_db_monotonic():
    low = quality_from_snr_db(-5.0, snr_floor_db=0.0, snr_ceil_db=20.0)
    mid = quality_from_snr_db(10.0, snr_floor_db=0.0, snr_ceil_db=20.0)
    high = quality_from_snr_db(30.0, snr_floor_db=0.0, snr_ceil_db=20.0)

    assert 0.0 <= low <= mid <= high <= 1.0


def test_compute_snr_proxy_db_reflects_noise_ratio():
    clean = np.ones(1600, dtype=np.float32) * 0.1
    noisy = np.ones(1600, dtype=np.float32) * 0.02

    snr_clean = compute_snr_proxy_db(clean, noise_floor_rms=0.01)
    snr_noisy = compute_snr_proxy_db(noisy, noise_floor_rms=0.01)

    assert snr_clean > snr_noisy


def test_effective_audio_weight_tracks_quality_and_confidence():
    base = 0.35
    w_low = effective_audio_weight(
        base_weight=base,
        audio_quality=0.1,
        audio_confidence=0.1,
        min_factor=0.2,
        quality_exponent=1.2,
    )
    w_high = effective_audio_weight(
        base_weight=base,
        audio_quality=0.9,
        audio_confidence=0.9,
        min_factor=0.2,
        quality_exponent=1.2,
    )

    assert 0.0 <= w_low <= base
    assert 0.0 <= w_high <= base
    assert w_high > w_low
