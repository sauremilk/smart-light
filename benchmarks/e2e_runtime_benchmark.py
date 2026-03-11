#!/usr/bin/env python3
"""End-to-end runtime benchmark for mock-mode system load and stability.

This benchmark executes `main.py` in mock mode for fixed scenarios and captures:
- control-loop timing from session log cadence
- process CPU and memory behavior over time
- runtime stability (unexpected early exits)

It is intentionally hardware-aware but deterministic in scenario definition.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

try:
    import psutil
except Exception as exc:  # pragma: no cover - import error path
    raise RuntimeError(
        "psutil is required for e2e runtime benchmark. Install with: pip install psutil"
    ) from exc


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class Scenario:
    name: str
    args: list[str]


PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "duration_seconds": 12,
        "warmup_seconds": 4,
        "sample_interval_seconds": 0.5,
        "scenarios": [
            Scenario(
                name="video_minimal",
                args=["--no-audio", "--no-pose", "--no-face-mesh", "--no-hrv", "--no-breathing"],
            ),
            Scenario(
                name="video_face_mesh",
                args=["--no-audio", "--no-pose", "--no-hrv", "--no-breathing"],
            ),
        ],
    },
    "standard": {
        "duration_seconds": 16,
        "warmup_seconds": 5,
        "sample_interval_seconds": 0.5,
        "scenarios": [
            Scenario(
                name="video_minimal",
                args=["--no-audio", "--no-pose", "--no-face-mesh", "--no-hrv", "--no-breathing"],
            ),
            Scenario(
                name="video_face_mesh",
                args=["--no-audio", "--no-pose", "--no-hrv", "--no-breathing"],
            ),
            Scenario(
                name="full_multimodal",
                args=[],
            ),
        ],
    },
    "strict": {
        "duration_seconds": 24,
        "warmup_seconds": 6,
        "sample_interval_seconds": 0.5,
        "scenarios": [
            Scenario(
                name="video_minimal",
                args=["--no-audio", "--no-pose", "--no-face-mesh", "--no-hrv", "--no-breathing"],
            ),
            Scenario(
                name="video_face_mesh",
                args=["--no-audio", "--no-pose", "--no-hrv", "--no-breathing"],
            ),
            Scenario(
                name="full_multimodal",
                args=[],
            ),
        ],
    },
}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    pos = (len(arr) - 1) * (q / 100.0)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return arr[low]
    frac = pos - low
    return arr[low] * (1.0 - frac) + arr[high] * frac


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _score_to_index(score01: float) -> int:
    return int(round(_clamp01(score01) * 1000.0))


def _safe_mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _load_session_entries(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _compute_timing_metrics(
    entries: list[dict[str, Any]], duration_seconds: float
) -> dict[str, float]:
    ts = [float(e.get("timestamp", 0.0)) for e in entries if e.get("timestamp") is not None]
    ts = [t for t in ts if t > 0.0]
    ts.sort()

    intervals_ms: list[float] = []
    for i in range(1, len(ts)):
        intervals_ms.append(max(0.0, (ts[i] - ts[i - 1]) * 1000.0))

    if intervals_ms:
        mean_interval_ms = _safe_mean(intervals_ms)
        loop_hz = 1000.0 / max(1e-9, mean_interval_ms)
    else:
        mean_interval_ms = 0.0
        loop_hz = 0.0

    expected_ticks = max(1.0, float(duration_seconds))
    observed_ticks = float(len(ts))
    drop_rate = _clamp01(1.0 - (observed_ticks / expected_ticks))

    return {
        "observed_ticks": observed_ticks,
        "expected_ticks": expected_ticks,
        "drop_rate": drop_rate,
        "loop_hz": loop_hz,
        "latency_p50_ms": _percentile(intervals_ms, 50.0),
        "latency_p95_ms": _percentile(intervals_ms, 95.0),
        "latency_p99_ms": _percentile(intervals_ms, 99.0),
        "interval_mean_ms": mean_interval_ms,
    }


def _compute_process_metrics(
    cpu_samples: list[float], rss_samples_mb: list[float], duration_seconds: float
) -> dict[str, float]:
    mem_drift_mb_per_min = 0.0
    if len(rss_samples_mb) >= 2 and duration_seconds > 0:
        mem_drift_mb = float(rss_samples_mb[-1] - rss_samples_mb[0])
        mem_drift_mb_per_min = mem_drift_mb * (60.0 / float(duration_seconds))

    return {
        "cpu_mean_percent": _safe_mean(cpu_samples),
        "cpu_p95_percent": _percentile(cpu_samples, 95.0),
        "rss_mean_mb": _safe_mean(rss_samples_mb),
        "rss_peak_mb": max(rss_samples_mb) if rss_samples_mb else 0.0,
        "rss_drift_mb_per_min": mem_drift_mb_per_min,
    }


def _score_scenario(
    timing: dict[str, float], proc_metrics: dict[str, float], stable_exit: bool
) -> dict[str, Any]:
    # Timing scores: 1.0 means no control-loop drops and <=1.0s cadence.
    s_loop = _clamp01(timing["loop_hz"] / 1.0)
    s_drop = 1.0 - _clamp01(timing["drop_rate"])

    # Latency score: full points at <=1000ms p95, linearly down to 0 at 3000ms.
    p95 = float(timing["latency_p95_ms"])
    s_latency = _clamp01((3000.0 - p95) / 2000.0)

    # Resource scores: preserve headroom and avoid memory growth.
    cpu_mean = float(proc_metrics["cpu_mean_percent"])
    s_cpu = _clamp01((100.0 - max(0.0, cpu_mean - 70.0)) / 100.0)

    mem_drift = abs(float(proc_metrics["rss_drift_mb_per_min"]))
    s_mem = _clamp01((250.0 - mem_drift) / 250.0)

    s_stability = 1.0 if stable_exit else 0.0

    score01 = _clamp01(
        0.35 * s_loop
        + 0.20 * s_drop
        + 0.20 * s_latency
        + 0.10 * s_cpu
        + 0.10 * s_mem
        + 0.05 * s_stability
    )

    return {
        "score01": score01,
        "score_index": _score_to_index(score01),
        "subscores": {
            "loop_hz": s_loop,
            "drop_rate": s_drop,
            "latency_p95": s_latency,
            "cpu_headroom": s_cpu,
            "memory_drift": s_mem,
            "stability": s_stability,
        },
    }


def _kill_process_tree(pid: int) -> None:
    """Forcefully kill a process and all its children (cross-platform via psutil)."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
    except psutil.NoSuchProcess:
        pass


def _run_scenario(
    scenario: Scenario,
    duration_seconds: float,
    warmup_seconds: float,
    sample_interval_seconds: float,
    python_executable: str,
) -> dict[str, Any]:
    temp_log = tempfile.NamedTemporaryFile(
        prefix=f"e2e_{scenario.name}_", suffix=".jsonl", delete=False
    )
    temp_log.close()

    cmd = [
        python_executable,
        os.path.join(ROOT_DIR, "main.py"),
        "--mock",
        "--session-log",
        temp_log.name,
        "--condition",
        "control",
        *scenario.args,
    ]

    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    p = psutil.Process(proc.pid)
    cpu_samples: list[float] = []
    rss_samples_mb: list[float] = []

    try:
        p.cpu_percent(interval=None)
        while True:
            now = time.time()
            elapsed = now - started
            if elapsed >= duration_seconds:
                break
            if proc.poll() is not None:
                break

            try:
                cpu = float(p.cpu_percent(interval=None))
                rss = float(p.memory_info().rss) / (1024.0 * 1024.0)
                if elapsed >= warmup_seconds:
                    cpu_samples.append(cpu)
                    rss_samples_mb.append(rss)
            except Exception:
                pass

            time.sleep(sample_interval_seconds)
    finally:
        still_running = proc.poll() is None
        if still_running:
            # Graceful shutdown first, then force-kill entire process tree.
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            if proc.poll() is None:
                try:
                    _kill_process_tree(proc.pid)
                except Exception:
                    proc.kill()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass  # Best-effort; log and continue

    ended = time.time()
    runtime_seconds = float(ended - started)
    return_code = proc.returncode if proc.returncode is not None else -999

    entries = _load_session_entries(temp_log.name)
    try:
        os.unlink(temp_log.name)
    except Exception:
        pass

    stable_exit = (runtime_seconds + 0.5) >= duration_seconds
    timing = _compute_timing_metrics(entries=entries, duration_seconds=duration_seconds)
    proc_metrics = _compute_process_metrics(
        cpu_samples=cpu_samples,
        rss_samples_mb=rss_samples_mb,
        duration_seconds=max(1e-9, runtime_seconds - warmup_seconds),
    )
    score = _score_scenario(timing=timing, proc_metrics=proc_metrics, stable_exit=stable_exit)

    return {
        "scenario": scenario.name,
        "command": " ".join(cmd),
        "duration_seconds": runtime_seconds,
        "requested_duration_seconds": float(duration_seconds),
        "return_code": int(return_code),
        "stable_exit": bool(stable_exit),
        "timing": timing,
        "process": proc_metrics,
        "samples": {
            "cpu_n": len(cpu_samples),
            "rss_n": len(rss_samples_mb),
            "session_log_rows": len(entries),
        },
        "score": score,
    }


def run_e2e_runtime_benchmark(profile: str, python_executable: str | None = None) -> dict[str, Any]:
    if profile not in PROFILE_PRESETS:
        raise ValueError(f"Unknown profile: {profile}")

    preset = PROFILE_PRESETS[profile]
    py = python_executable or sys.executable

    scenario_reports = []
    for scenario in preset["scenarios"]:
        scenario_reports.append(
            _run_scenario(
                scenario=scenario,
                duration_seconds=float(preset["duration_seconds"]),
                warmup_seconds=float(preset["warmup_seconds"]),
                sample_interval_seconds=float(preset["sample_interval_seconds"]),
                python_executable=py,
            )
        )

    scenario_scores = [float(s["score"]["score01"]) for s in scenario_reports]
    aggregate_score = _safe_mean(scenario_scores)

    return {
        "benchmark": "e2e_runtime_v1",
        "profile": profile,
        "settings": {
            "duration_seconds": float(preset["duration_seconds"]),
            "warmup_seconds": float(preset["warmup_seconds"]),
            "sample_interval_seconds": float(preset["sample_interval_seconds"]),
            "scenarios": [s.name for s in preset["scenarios"]],
            "python_executable": py,
        },
        "aggregate": {
            "score01": aggregate_score,
            "index": _score_to_index(aggregate_score),
        },
        "scenarios": scenario_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end runtime benchmark")
    parser.add_argument("--profile", choices=["quick", "standard", "strict"], default="quick")
    parser.add_argument(
        "--output",
        default=os.path.join("benchmarks", "results", "e2e_runtime.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    report = run_e2e_runtime_benchmark(profile=args.profile)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== E2E Runtime Benchmark ===")
    print(f"Profile: {report['profile']}")
    print(f"Aggregate index: {report['aggregate']['index']}")
    for s in report["scenarios"]:
        print(
            f"- {s['scenario']}: idx={s['score']['score_index']} "
            f"p95={s['timing']['latency_p95_ms']:.1f}ms "
            f"cpu={s['process']['cpu_mean_percent']:.1f}% "
            f"rss_peak={s['process']['rss_peak_mb']:.1f}MB"
        )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
