"""Tests fuer tools/session_analyzer.py – Session-Analyse."""

import json
import tempfile

from tools.session_analyzer import analyze_session, compare_sessions


def _write_jsonl(lines: list[dict]) -> str:
    """Erstellt eine temporaere JSONL-Datei und gibt den Pfad zurueck."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for line in lines:
        f.write(json.dumps(line) + "\n")
    f.close()
    return f.name


def test_empty_file():
    path = _write_jsonl([])
    report = analyze_session(path)
    assert report.sample_count == 0
    assert report.quality_score == 0.0


def test_basic_analysis():
    records = [
        {"timestamp": 1000.0, "valence": 0.5, "arousal": 0.3, "cognitive_state": "FOCUS"},
        {"timestamp": 1001.0, "valence": 0.6, "arousal": 0.2, "cognitive_state": "FOCUS"},
        {"timestamp": 1002.0, "valence": 0.4, "arousal": 0.1, "cognitive_state": "NEUTRAL"},
        {"timestamp": 1003.0, "valence": -0.2, "arousal": 0.5, "cognitive_state": "STRESS"},
        {"timestamp": 1004.0, "valence": 0.3, "arousal": -0.1, "cognitive_state": "FOCUS"},
    ]
    path = _write_jsonl(records)
    report = analyze_session(path)

    assert report.sample_count == 5
    assert report.total_seconds == 4.0
    assert report.focus_ratio == 3.0 / 5.0
    assert report.stress_ratio == 1.0 / 5.0
    assert report.neutral_ratio == 1.0 / 5.0
    assert report.mean_valence > 0.0
    assert report.quality_score > 0.0


def test_mode_distribution():
    records = [
        {"timestamp": 100.0, "active_mode": "FOCUS"},
        {"timestamp": 101.0, "active_mode": "FOCUS"},
        {"timestamp": 102.0, "active_mode": "RELAX"},
    ]
    path = _write_jsonl(records)
    report = analyze_session(path)

    assert "FOCUS" in report.mode_distribution
    assert abs(report.mode_distribution["FOCUS"] - 2.0 / 3.0) < 0.01


def test_break_counting():
    records = [
        {"timestamp": 100.0, "break_active": False},
        {"timestamp": 101.0, "break_active": True},
        {"timestamp": 102.0, "break_active": True},
        {"timestamp": 103.0, "break_active": False, "recovery_quality": 0.8},
        {"timestamp": 104.0, "break_active": True},
        {"timestamp": 105.0, "break_active": False, "recovery_quality": 0.6},
    ]
    path = _write_jsonl(records)
    report = analyze_session(path)

    assert report.break_count == 2
    assert report.avg_recovery_quality > 0.0


def test_summary_string():
    records = [
        {"timestamp": 100.0, "valence": 0.5, "arousal": 0.3, "cognitive_state": "FOCUS"},
        {"timestamp": 200.0, "valence": 0.4, "arousal": 0.2, "cognitive_state": "FLOW"},
    ]
    path = _write_jsonl(records)
    report = analyze_session(path)
    summary = report.summary()

    assert "Session-Analyse" in summary
    assert "Fokus" in summary
    assert "Qualitaet" in summary


def test_compare_sessions():
    recs1 = [
        {"timestamp": 100.0, "valence": 0.5, "arousal": 0.3, "cognitive_state": "FOCUS"},
        {"timestamp": 200.0, "valence": 0.4, "arousal": 0.2, "cognitive_state": "FOCUS"},
    ]
    recs2 = [
        {"timestamp": 100.0, "valence": 0.2, "arousal": 0.1, "cognitive_state": "STRESS"},
        {"timestamp": 200.0, "valence": 0.1, "arousal": 0.0, "cognitive_state": "STRESS"},
    ]
    p1 = _write_jsonl(recs1)
    p2 = _write_jsonl(recs2)
    result = compare_sessions([p1, p2])
    assert "Vergleich" in result
    assert "Trend" in result
