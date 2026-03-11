"""Session-Analyzer: wertet JSONL-Session-Logs nachtraeglich aus.

Nutzung als CLI-Tool:
    python -m tools.session_analyzer path/to/session_log.jsonl

Oder als Modul:
    from tools.session_analyzer import SessionReport, analyze_session
    report = analyze_session("path/to/session.jsonl")
    print(report.summary())

Metriken:
  - Gesamtdauer
  - Fokus-Anteil (%)
  - Stress-Anteil (%)
  - Flow-Anteil (%)
  - Durchschnittliche Valence/Arousal
  - Pausen-Statistik
  - Vergleich mit vorherigen Sessions (wenn Verzeichnis angegeben)
"""

from __future__ import annotations

import json
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SessionReport:
    """Zusammenfassung einer analysierten Session."""

    file_path: str = ""
    total_seconds: float = 0.0
    sample_count: int = 0

    # Kognitive Zustaende (Anteil 0..1)
    focus_ratio: float = 0.0
    flow_ratio: float = 0.0
    fatigue_ratio: float = 0.0
    stress_ratio: float = 0.0
    neutral_ratio: float = 0.0

    # Valence/Arousal-Statistiken
    mean_valence: float = 0.0
    mean_arousal: float = 0.0
    std_valence: float = 0.0
    std_arousal: float = 0.0

    # Pausen
    break_count: int = 0
    total_break_seconds: float = 0.0
    avg_recovery_quality: float = 0.0

    # Modus-Verteilung
    mode_distribution: dict[str, float] = field(default_factory=dict)

    # Qualitaets-Score (0..100)
    quality_score: float = 0.0

    def summary(self) -> str:
        """Lesbare Text-Zusammenfassung."""
        dur_min = self.total_seconds / 60.0
        lines = [
            f"=== Session-Analyse: {self.file_path} ===",
            f"Dauer:        {dur_min:.1f} min  ({self.sample_count} Samples)",
            f"Fokus:        {self.focus_ratio * 100:.1f}%",
            f"Flow:         {self.flow_ratio * 100:.1f}%",
            f"Muedigkeit:   {self.fatigue_ratio * 100:.1f}%",
            f"Stress:       {self.stress_ratio * 100:.1f}%",
            f"Neutral:      {self.neutral_ratio * 100:.1f}%",
            "",
            f"Valence:      {self.mean_valence:+.3f} (std {self.std_valence:.3f})",
            f"Arousal:      {self.mean_arousal:+.3f} (std {self.std_arousal:.3f})",
            "",
            f"Pausen:       {self.break_count}x,  {self.total_break_seconds / 60:.1f} min gesamt",
            f"Erholungs-Q:  {self.avg_recovery_quality:.0%}",
            "",
            f"Qualitaet:    {self.quality_score:.0f}/100",
        ]
        if self.mode_distribution:
            lines.append("")
            lines.append("Modi-Verteilung:")
            for mode, ratio in sorted(self.mode_distribution.items(), key=lambda x: -x[1]):
                lines.append(f"  {mode:12s} {ratio * 100:.1f}%")
        return "\n".join(lines)


def analyze_session(path: str | Path) -> SessionReport:
    """Analysiert eine JSONL-Session-Datei.

    Jede Zeile sollte ein JSON-Objekt mit diesen Feldern enthalten:
      - timestamp (float): Zeitstempel
      - valence/arousal (float): Emotionswerte
      - cognitive_state (str): FOCUS/FLOW/FATIGUE/STRESS/NEUTRAL (optional)
      - active_mode (str): Aktueller Modus (optional)
      - break_active (bool): Pause aktiv (optional)
      - recovery_quality (float): Erholungsqualitaet (optional)
    """
    path = Path(path)
    records: list[dict] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return SessionReport(file_path=str(path))

    report = SessionReport(file_path=str(path), sample_count=len(records))

    # ── Dauer berechnen ──
    timestamps = [r.get("timestamp", 0.0) for r in records]
    if len(timestamps) >= 2:
        report.total_seconds = max(timestamps) - min(timestamps)

    # ── Kognitive Zustaende ──
    states = [r.get("cognitive_state", "NEUTRAL") for r in records]
    n = len(states)
    if n > 0:
        report.focus_ratio = states.count("FOCUS") / n
        report.flow_ratio = states.count("FLOW") / n
        report.fatigue_ratio = states.count("FATIGUE") / n
        report.stress_ratio = states.count("STRESS") / n
        report.neutral_ratio = states.count("NEUTRAL") / n

    # ── Valence/Arousal ──
    val_list = [r.get("valence", 0.0) for r in records if "valence" in r]
    aro_list = [r.get("arousal", 0.0) for r in records if "arousal" in r]
    if val_list:
        report.mean_valence = statistics.mean(val_list)
        report.std_valence = statistics.stdev(val_list) if len(val_list) > 1 else 0.0
    if aro_list:
        report.mean_arousal = statistics.mean(aro_list)
        report.std_arousal = statistics.stdev(aro_list) if len(aro_list) > 1 else 0.0

    # ── Pausen ──
    in_break = False
    break_starts: list[int] = []
    break_count = 0
    recovery_quals: list[float] = []

    for i, r in enumerate(records):
        ba = r.get("break_active", False)
        if ba and not in_break:
            break_count += 1
            break_starts.append(i)
        if not ba and in_break:
            rq = r.get("recovery_quality", 0.0)
            if rq > 0:
                recovery_quals.append(rq)
        in_break = ba

    report.break_count = break_count
    break_samples = sum(1 for r in records if r.get("break_active", False))
    if report.total_seconds > 0 and n > 0:
        report.total_break_seconds = report.total_seconds * (break_samples / n)
    if recovery_quals:
        report.avg_recovery_quality = statistics.mean(recovery_quals)

    # ── Modus-Verteilung ──
    modes = [r.get("active_mode", "") for r in records if r.get("active_mode")]
    if modes:
        unique_modes = set(modes)
        total_m = len(modes)
        report.mode_distribution = {m: modes.count(m) / total_m for m in unique_modes}

    # ── Qualitaets-Score ──
    report.quality_score = _compute_quality_score(report)

    return report


def _compute_quality_score(r: SessionReport) -> float:
    """Berechnet einen Qualitaets-Score (0-100) fuer die Session.

    Gewichtung:
      - Fokus+Flow-Anteil: 40%
      - Niedriger Stress: 20%
      - Niedrige Muedigkeit: 15%
      - V/A Stabilitaet: 15%
      - Pausen-Qualitaet: 10%
    """
    if r.sample_count == 0:
        return 0.0

    # Produktivitaet (Fokus + Flow)
    productivity = min(1.0, r.focus_ratio + r.flow_ratio)

    # Stress-Freiheit
    low_stress = 1.0 - min(1.0, r.stress_ratio * 2.0)

    # Muedigkeits-Freiheit
    low_fatigue = 1.0 - min(1.0, r.fatigue_ratio * 2.0)

    # V/A Stabilitaet (niedrige Varianz = stabiler)
    va_stability = max(0.0, 1.0 - (r.std_valence + r.std_arousal))

    # Pausen-Qualitaet
    break_quality = r.avg_recovery_quality if r.break_count > 0 else 0.5

    score = (
        productivity * 40.0
        + low_stress * 20.0
        + low_fatigue * 15.0
        + va_stability * 15.0
        + break_quality * 10.0
    )

    return max(0.0, min(100.0, score))


def compare_sessions(paths: Sequence[str | Path]) -> str:
    """Vergleicht mehrere Sessions und gibt einen Trend-Bericht zurueck."""
    reports = [analyze_session(p) for p in paths]
    reports = [r for r in reports if r.sample_count > 0]

    if not reports:
        return "Keine verwertbaren Sessions gefunden."

    if len(reports) == 1:
        return reports[0].summary()

    lines = [f"=== Vergleich von {len(reports)} Sessions ===", ""]
    lines.append(
        f"{'Datei':30s} {'Dauer':>8s} {'Fokus':>7s} {'Flow':>7s} {'Stress':>7s} {'Score':>6s}"
    )
    lines.append("-" * 72)

    for r in reports:
        dur = f"{r.total_seconds / 60:.0f}m"
        lines.append(
            f"{Path(r.file_path).name:30s} {dur:>8s}"
            f" {r.focus_ratio * 100:>6.1f}%"
            f" {r.flow_ratio * 100:>6.1f}%"
            f" {r.stress_ratio * 100:>6.1f}%"
            f" {r.quality_score:>5.0f}"
        )

    # Trend
    scores = [r.quality_score for r in reports]
    if len(scores) >= 2:
        delta = scores[-1] - scores[0]
        trend = "aufwaerts" if delta > 2 else ("abwaerts" if delta < -2 else "stabil")
        lines.append("")
        lines.append(f"Trend: {trend} ({delta:+.1f} Punkte von erster zu letzter Session)")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Nutzung: python -m tools.session_analyzer <session.jsonl> [session2.jsonl ...]")
        sys.exit(1)

    if len(sys.argv) == 2:
        rep = analyze_session(sys.argv[1])
        print(rep.summary())
    else:
        print(compare_sessions(sys.argv[1:]))
