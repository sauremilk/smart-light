import numpy as np

from hrv_analyzer import HRVAnalyzer


def test_decay_result_reduces_confidence_and_clears_stale_metrics():
    analyzer = HRVAnalyzer(window_seconds=30.0, target_fps=24.0)

    analyzer._result = {
        "hr_bpm": 72.0,
        "hrv_rmssd": 38.0,
        "hrv_sdnn": 52.0,
        "confidence": 1.0,
        "face_detected": True,
    }

    for _ in range(20):
        analyzer._decay_result(face_detected=False, factor=0.78)

    result = analyzer.get()
    assert result["face_detected"] is False
    assert result["confidence"] < 0.05
    assert result["hr_bpm"] == 0.0
    assert result["hrv_rmssd"] == 0.0
    assert result["hrv_sdnn"] == 0.0


def test_process_signal_confidence_not_permanently_saturated_to_one():
    analyzer = HRVAnalyzer(window_seconds=30.0, target_fps=24.0)

    fs = 24.0
    duration_s = 30.0
    t = np.arange(0.0, duration_s, 1.0 / fs)

    # Synthetic pulse-like RGB stream around 72 BPM with moderate variability.
    base = 120.0 + 8.0 * np.sin(2.0 * np.pi * 1.2 * t)

    analyzer._buffer.clear()
    t0 = 1_700_000_000.0
    for i, ts in enumerate(t):
        analyzer._buffer.append((t0 + float(ts), float(base[i]), 120.0, 120.0))

    ok = analyzer._process_signal()
    assert ok is True

    result = analyzer.get()
    assert result["hr_bpm"] > 0.0
    assert 0.0 < result["confidence"] < 1.0
