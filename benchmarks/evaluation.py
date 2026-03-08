#!/usr/bin/env python3
"""Evaluate emotional effect from session JSONL logs.

The script summarizes how strongly sessions move toward the target VA state.
It supports A/B analysis with conditions "adaptive" and "control".

Log format expected (produced by main.py --session-log):
- runtime_sec
- condition
- participant (optional)
- distance_current
- at_target

Usage examples:
  c:/Users/mickg/smart-light/.venv/Scripts/python.exe benchmarks/evaluation.py \
      --glob "benchmarks/results/sessions/*.jsonl"

  c:/Users/mickg/smart-light/.venv/Scripts/python.exe benchmarks/evaluation.py \
      --input logs/session_adaptive_p01.jsonl logs/session_control_p01.jsonl
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from dataclasses import dataclass


@dataclass
class SessionMetrics:
    file: str
    participant: str | None
    session_id: str | None
    condition: str
    n_samples: int
    duration_sec: float
    start_dist: float
    end_dist: float
    delta_dist: float
    slope_dist_per_sec: float
    at_target_ratio: float


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    var_x = sum((x - x_mean) ** 2 for x in xs)
    if var_x <= 1e-12:
        return 0.0
    cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return float(cov_xy / var_x)


def _load_records(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "runtime_sec" not in rec or "distance_current" not in rec:
                continue
            records.append(rec)
    records.sort(key=lambda r: float(r.get("runtime_sec", 0.0)))
    return records


def _window(values: list[tuple[float, float]], begin: float, end: float) -> list[float]:
    return [v for t, v in values if begin <= t <= end]


def _derive_session_metrics(path: str, records: list[dict], edge_window_sec: float) -> SessionMetrics | None:
    if len(records) < 5:
        return None

    duration = float(records[-1].get("runtime_sec", 0.0))
    if duration <= 5.0:
        return None

    distances = [(float(r.get("runtime_sec", 0.0)), float(r.get("distance_current", 0.0))) for r in records]
    at_target_vals = [1.0 if bool(r.get("at_target", False)) else 0.0 for r in records]

    start_window_end = min(edge_window_sec, duration * 0.25)
    end_window_start = max(0.0, duration - edge_window_sec)

    start_vals = _window(distances, 0.0, start_window_end)
    end_vals = _window(distances, end_window_start, duration)

    if not start_vals:
        start_vals = [distances[0][1]]
    if not end_vals:
        end_vals = [distances[-1][1]]

    start_dist = _median(start_vals)
    end_dist = _median(end_vals)
    delta = end_dist - start_dist

    xs = [t for t, _ in distances]
    ys = [d for _, d in distances]

    first = records[0]
    return SessionMetrics(
        file=path,
        participant=first.get("participant"),
        session_id=first.get("session_id"),
        condition=str(first.get("condition", "unknown")),
        n_samples=len(records),
        duration_sec=duration,
        start_dist=start_dist,
        end_dist=end_dist,
        delta_dist=delta,
        slope_dist_per_sec=_linear_slope(xs, ys),
        at_target_ratio=_mean(at_target_vals),
    )


def _group_by_condition(sessions: list[SessionMetrics]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cond in sorted(set(s.condition for s in sessions)):
        rows = [s for s in sessions if s.condition == cond]
        out[cond] = {
            "n_sessions": len(rows),
            "delta_dist_mean": _mean([s.delta_dist for s in rows]),
            "delta_dist_median": _median([s.delta_dist for s in rows]),
            "slope_mean": _mean([s.slope_dist_per_sec for s in rows]),
            "at_target_ratio_mean": _mean([s.at_target_ratio for s in rows]),
            "duration_mean_sec": _mean([s.duration_sec for s in rows]),
        }
    return out


def _paired_effect(sessions: list[SessionMetrics]) -> dict:
    by_participant: dict[str, dict[str, list[SessionMetrics]]] = {}
    for s in sessions:
        if not s.participant:
            continue
        by_participant.setdefault(s.participant, {}).setdefault(s.condition, []).append(s)

    deltas: list[float] = []
    # Positive value means adaptive is better (smaller/more negative delta_dist).
    for participant, cond_map in by_participant.items():
        if "adaptive" not in cond_map or "control" not in cond_map:
            continue
        adaptive_mean = _mean([x.delta_dist for x in cond_map["adaptive"]])
        control_mean = _mean([x.delta_dist for x in cond_map["control"]])
        deltas.append(control_mean - adaptive_mean)

    return {
        "n_participants_paired": len(deltas),
        "adaptive_advantage_mean": _mean(deltas),
        "adaptive_advantage_median": _median(deltas),
        "details": "Positive value => adaptive improved distance-to-target more than control.",
    }


def _resolve_inputs(args: argparse.Namespace) -> list[str]:
    files: list[str] = []
    if args.input:
        files.extend(args.input)
    if args.glob:
        for pattern in args.glob:
            files.extend(glob.glob(pattern))
    unique = sorted(set(os.path.normpath(p) for p in files if os.path.isfile(p)))
    return unique


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def _load_self_report(path: str) -> dict:
    """Load self-report CSV and return lookup maps for session/participant matching.

    Expected columns (recommended):
      participant,session_id,condition,mood_pre,mood_post,calm_pre,calm_post,stress_pre,stress_post
    """
    by_session_id: dict[str, dict] = {}
    by_participant_condition: dict[tuple[str, str], list[dict]] = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            participant = (row.get("participant") or "").strip()
            session_id = (row.get("session_id") or "").strip()
            condition = (row.get("condition") or "").strip()

            mood_pre = _safe_float(row.get("mood_pre"))
            mood_post = _safe_float(row.get("mood_post"))
            calm_pre = _safe_float(row.get("calm_pre"))
            calm_post = _safe_float(row.get("calm_post"))
            stress_pre = _safe_float(row.get("stress_pre"))
            stress_post = _safe_float(row.get("stress_post"))

            mood_delta = None if mood_pre is None or mood_post is None else (mood_post - mood_pre)
            calm_delta = None if calm_pre is None or calm_post is None else (calm_post - calm_pre)
            stress_delta = None if stress_pre is None or stress_post is None else (stress_post - stress_pre)

            wellbeing_parts = []
            if mood_delta is not None:
                wellbeing_parts.append(mood_delta)
            if calm_delta is not None:
                wellbeing_parts.append(calm_delta)
            if stress_delta is not None:
                wellbeing_parts.append(-stress_delta)
            wellbeing_delta = _mean(wellbeing_parts) if wellbeing_parts else None

            parsed = {
                "participant": participant or None,
                "session_id": session_id or None,
                "condition": condition or None,
                "mood_delta": mood_delta,
                "calm_delta": calm_delta,
                "stress_delta": stress_delta,
                "wellbeing_delta": wellbeing_delta,
                "raw": row,
            }

            if session_id:
                by_session_id[session_id] = parsed
            if participant and condition:
                by_participant_condition.setdefault((participant, condition), []).append(parsed)

    return {
        "by_session_id": by_session_id,
        "by_participant_condition": by_participant_condition,
    }


def _join_self_report(sessions: list[SessionMetrics], self_report_maps: dict) -> list[dict]:
    """Join sessions with self-reports by session_id first, then participant+condition."""
    linked: list[dict] = []
    by_id = self_report_maps.get("by_session_id", {})
    by_pc = self_report_maps.get("by_participant_condition", {})

    for s in sessions:
        match = None
        if s.session_id and s.session_id in by_id:
            match = by_id[s.session_id]
        elif s.participant:
            candidates = by_pc.get((s.participant, s.condition), [])
            if len(candidates) == 1:
                match = candidates[0]

        if match is None:
            continue

        linked.append(
            {
                "participant": s.participant,
                "session_id": s.session_id,
                "condition": s.condition,
                "delta_dist": s.delta_dist,
                "slope_dist_per_sec": s.slope_dist_per_sec,
                "mood_delta": match.get("mood_delta"),
                "calm_delta": match.get("calm_delta"),
                "stress_delta": match.get("stress_delta"),
                "wellbeing_delta": match.get("wellbeing_delta"),
            }
        )

    return linked


def _mean_non_null(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return _mean(vals)


def _self_report_summary(linked_rows: list[dict]) -> dict:
    if not linked_rows:
        return {
            "n_linked_sessions": 0,
            "by_condition": {},
            "paired_adaptive_advantage": {
                "n_participants_paired": 0,
                "wellbeing_advantage_mean": None,
                "details": "Positive value => adaptive better subjective improvement.",
            },
        }

    by_condition: dict[str, dict] = {}
    conditions = sorted(set(r["condition"] for r in linked_rows))
    for cond in conditions:
        rows = [r for r in linked_rows if r["condition"] == cond]
        by_condition[cond] = {
            "n_sessions": len(rows),
            "mood_delta_mean": _mean_non_null(rows, "mood_delta"),
            "calm_delta_mean": _mean_non_null(rows, "calm_delta"),
            "stress_delta_mean": _mean_non_null(rows, "stress_delta"),
            "wellbeing_delta_mean": _mean_non_null(rows, "wellbeing_delta"),
        }

    # Paired advantage on wellbeing_delta.
    by_participant: dict[str, dict[str, list[dict]]] = {}
    for row in linked_rows:
        participant = row.get("participant")
        if not participant:
            continue
        by_participant.setdefault(participant, {}).setdefault(row["condition"], []).append(row)

    paired_adv: list[float] = []
    for _participant, cond_map in by_participant.items():
        if "adaptive" not in cond_map or "control" not in cond_map:
            continue
        adaptive_vals = [r["wellbeing_delta"] for r in cond_map["adaptive"] if r.get("wellbeing_delta") is not None]
        control_vals = [r["wellbeing_delta"] for r in cond_map["control"] if r.get("wellbeing_delta") is not None]
        if not adaptive_vals or not control_vals:
            continue
        paired_adv.append(_mean(adaptive_vals) - _mean(control_vals))

    return {
        "n_linked_sessions": len(linked_rows),
        "by_condition": by_condition,
        "paired_adaptive_advantage": {
            "n_participants_paired": len(paired_adv),
            "wellbeing_advantage_mean": _mean(paired_adv) if paired_adv else None,
            "details": "Positive value => adaptive better subjective improvement.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate emotional effect from session logs")
    parser.add_argument("--input", nargs="*", default=None, help="Explicit JSONL files")
    parser.add_argument("--glob", nargs="*", default=None, help="Glob patterns for JSONL files")
    parser.add_argument("--edge-window-sec", type=float, default=15.0,
                        help="Seconds at start/end used for robust median distance")
    parser.add_argument("--self-report-csv", default=None,
                        help="Optional CSV with pre/post self-report scores")
    parser.add_argument("--output", default="benchmarks/results/effect_report.json",
                        help="Path to JSON report")
    args = parser.parse_args()

    files = _resolve_inputs(args)
    if not files:
        raise SystemExit("No input files found. Use --input or --glob.")

    sessions: list[SessionMetrics] = []
    for path in files:
        records = _load_records(path)
        metrics = _derive_session_metrics(path, records, edge_window_sec=float(args.edge_window_sec))
        if metrics is not None:
            sessions.append(metrics)

    if not sessions:
        raise SystemExit("No valid sessions in input files.")

    by_condition = _group_by_condition(sessions)
    paired = _paired_effect(sessions)

    linked_self_report_rows: list[dict] = []
    self_report_summary = None
    if args.self_report_csv:
        maps = _load_self_report(args.self_report_csv)
        linked_self_report_rows = _join_self_report(sessions, maps)
        self_report_summary = _self_report_summary(linked_self_report_rows)

    report = {
        "inputs": files,
        "n_sessions": len(sessions),
        "by_condition": by_condition,
        "paired_effect": paired,
        "self_report_csv": args.self_report_csv,
        "self_report_summary": self_report_summary,
        "self_report_linked_sessions": linked_self_report_rows,
        "sessions": [
            {
                "file": s.file,
                "participant": s.participant,
                "session_id": s.session_id,
                "condition": s.condition,
                "n_samples": s.n_samples,
                "duration_sec": s.duration_sec,
                "start_dist": s.start_dist,
                "end_dist": s.end_dist,
                "delta_dist": s.delta_dist,
                "slope_dist_per_sec": s.slope_dist_per_sec,
                "at_target_ratio": s.at_target_ratio,
            }
            for s in sessions
        ],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Emotional Effect Evaluation ===")
    print(f"Sessions: {len(sessions)}")
    for cond, values in by_condition.items():
        print(
            f"{cond:>8}: n={values['n_sessions']}, "
            f"mean_delta={values['delta_dist_mean']:+.4f}, "
            f"median_delta={values['delta_dist_median']:+.4f}, "
            f"mean_slope={values['slope_mean']:+.5f}/s"
        )
    print(
        "Paired adaptive advantage: "
        f"n={paired['n_participants_paired']}, "
        f"mean={paired['adaptive_advantage_mean']:+.4f}"
    )
    if self_report_summary is not None:
        print("Self-report linked sessions:", self_report_summary["n_linked_sessions"])
        for cond, values in self_report_summary["by_condition"].items():
            w = values["wellbeing_delta_mean"]
            w_text = "n/a" if w is None else f"{w:+.3f}"
            print(f"  {cond:>8}: subjective wellbeing_delta_mean={w_text}")
        paired_w = self_report_summary["paired_adaptive_advantage"]["wellbeing_advantage_mean"]
        if paired_w is not None:
            print(f"  paired subjective adaptive advantage mean={paired_w:+.3f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
