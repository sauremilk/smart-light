#!/usr/bin/env python3
"""Reference benchmark suite for disciplined, evidence-based optimization.

This suite combines multiple benchmark pillars so improvements cannot game a
single metric. It is designed for repeatable runs by humans and coding agents.

Pillars:
1. Extreme visual robustness benchmark (very hard perturbations)
2. Multi-seed stability benchmark (variance-aware)
3. Test quality benchmark (local project test suite)
4. Module sanity benchmark (core subsystem contract checks)

Output:
- benchmarks/results/reference_suite_latest.json
- optional gate check against a baseline json file
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

PROFILE_PRESETS = {
    "quick": {
        "extreme_limit": 21,
        "extreme_variants": 1,
        "accuracy_limit": 21,
        "seeds": [1],
        "e2e_profile": "quick",
    },
    "standard": {
        "extreme_limit": 56,
        "extreme_variants": 2,
        "accuracy_limit": 42,
        "seeds": [1, 2, 3],
        "e2e_profile": "standard",
    },
    "strict": {
        "extreme_limit": 84,
        "extreme_variants": 3,
        "accuracy_limit": 56,
        "seeds": [1, 2, 3, 4, 5],
        "e2e_profile": "strict",
    },
}


@dataclass
class ComponentResult:
    name: str
    score01: float
    details: dict


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _weighted_metric(acc: float, macro_f1: float) -> float:
    return _clamp01(0.4 * float(acc) + 0.6 * float(macro_f1))


def _index(score01: float) -> int:
    return int(round(_clamp01(score01) * 1000.0))


def _mean_std_ci95(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "ci95_half_width": 0.0,
        }

    mean = float(statistics.fmean(values))
    std = float(statistics.stdev(values)) if n > 1 else 0.0
    sem = (std / (n ** 0.5)) if n > 1 else 0.0
    # Normal approximation is sufficient for quick trend diagnostics.
    half = 1.96 * sem
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "ci95_half_width": half,
    }


def _collect_environment_metadata() -> dict:
    metadata = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "libraries": {},
        "hardware": {},
    }

    for lib_name, module_name in (
        ("numpy", "numpy"),
        ("opencv", "cv2"),
        ("tensorflow", "tensorflow"),
        ("torch", "torch"),
        ("mediapipe", "mediapipe"),
    ):
        try:
            mod = __import__(module_name)
            metadata["libraries"][lib_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            metadata["libraries"][lib_name] = None

    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        gpu_mem = None
        if cuda_available:
            gpu_mem = int(torch.cuda.get_device_properties(0).total_memory)
        metadata["hardware"]["torch_cuda"] = {
            "available": cuda_available,
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "device_name": gpu_name,
            "total_memory_bytes": gpu_mem,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        }
    except Exception:
        metadata["hardware"]["torch_cuda"] = {
            "available": False,
            "device_count": 0,
            "device_name": None,
            "total_memory_bytes": None,
            "cuda_version": None,
            "cudnn_version": None,
        }

    return metadata


def _run_extreme_component(preset: dict, detector: str, seed: int) -> ComponentResult:
    from accuracy_benchmark import FACE_MESH_WEIGHT, HEAD_POSE_STRENGTH
    from extreme_reference_benchmark import run_extreme_benchmark

    runs = {}
    run_no_fm = run_extreme_benchmark(
        limit=int(preset["extreme_limit"]),
        seed=int(seed),
        detector_backend=detector,
        variants_per_sample=int(preset["extreme_variants"]),
        use_face_mesh=False,
        face_mesh_weight=float(FACE_MESH_WEIGHT),
        head_pose_strength=float(HEAD_POSE_STRENGTH),
    )
    runs["no_face_mesh"] = run_no_fm

    run_with_fm = run_extreme_benchmark(
        limit=int(preset["extreme_limit"]),
        seed=int(seed),
        detector_backend=detector,
        variants_per_sample=int(preset["extreme_variants"]),
        use_face_mesh=True,
        face_mesh_weight=float(FACE_MESH_WEIGHT),
        head_pose_strength=float(HEAD_POSE_STRENGTH),
    )
    runs["with_face_mesh"] = run_with_fm

    s_no = float(run_no_fm["enhanced"]["weighted_score"])
    s_fm = float(run_with_fm["enhanced"]["weighted_score"])
    score = _clamp01((s_no + s_fm) / 2.0)

    details = {
        "score_index": _index(score),
        "enhanced_score_no_face_mesh": s_no,
        "enhanced_score_with_face_mesh": s_fm,
        "delta_score_with_minus_without_face_mesh": s_fm - s_no,
        "runs": runs,
    }
    return ComponentResult(name="extreme_visual_robustness", score01=score, details=details)


def _run_stability_component(preset: dict, detector: str) -> ComponentResult:
    from accuracy_benchmark import FACE_MESH_WEIGHT, HEAD_POSE_STRENGTH, run as run_accuracy

    seed_rows = []
    weighted_scores = []
    delta_scores = []

    for seed in preset["seeds"]:
        report = run_accuracy(
            limit=int(preset["accuracy_limit"]),
            seed=int(seed),
            detector_backend=detector,
            use_face_mesh=True,
            face_mesh_weight=float(FACE_MESH_WEIGHT),
            head_pose_strength=float(HEAD_POSE_STRENGTH),
            collect_diagnostics=False,
        )

        score = _weighted_metric(report["enhanced"]["accuracy"], report["enhanced"]["macro_f1"])
        delta_score = _weighted_metric(report["delta"]["accuracy"], report["delta"]["macro_f1"])
        weighted_scores.append(score)
        delta_scores.append(delta_score)

        seed_rows.append(
            {
                "seed": seed,
                "enhanced_accuracy": report["enhanced"]["accuracy"],
                "enhanced_macro_f1": report["enhanced"]["macro_f1"],
                "enhanced_weighted_score": score,
                "delta_accuracy": report["delta"]["accuracy"],
                "delta_macro_f1": report["delta"]["macro_f1"],
                "delta_weighted_score": delta_score,
            }
        )

    score_stats = _mean_std_ci95(weighted_scores)
    delta_stats = _mean_std_ci95(delta_scores)
    mean_score = float(score_stats["mean"])
    std_score = float(score_stats["std"])

    # Penalize instability directly so a noisy setup cannot score high.
    score = _clamp01(mean_score - 0.5 * std_score)

    details = {
        "score_index": _index(score),
        "mean_enhanced_weighted_score": mean_score,
        "std_enhanced_weighted_score": std_score,
        "mean_delta_weighted_score": float(delta_stats["mean"]),
        "enhanced_weighted_score_stats": score_stats,
        "delta_weighted_score_stats": delta_stats,
        "per_seed": seed_rows,
    }
    return ComponentResult(name="multi_seed_stability", score01=score, details=details)


def _parse_pytest_summary(text: str) -> tuple[int, int, bool]:
    # Example endings: "7 passed in 1.20s" or "1 failed, 6 passed in ..."
    passed = 0
    failed = 0

    m_passed = re.findall(r"(\d+)\s+passed", text)
    m_failed = re.findall(r"(\d+)\s+failed", text)

    if m_passed:
        passed = int(m_passed[-1])
    if m_failed:
        failed = int(m_failed[-1])

    total = passed + failed
    ok = failed == 0 and passed > 0
    return passed, total, ok


def _run_test_component() -> ComponentResult:
    local_tests = [
        "tests/test_ema_utils.py",
        "tests/test_config_local_override.py",
        "tests/test_emotion_regulator.py",
        "tests/test_face_mesh_and_color.py",
        "tests/test_hrv_analyzer.py",
        "tests/test_light_mapping.py",
        "tests/test_main_overlay.py",
        "tests/test_main_loop_safeguards.py",
        "tests/test_reference_suite_gate.py",
        "tests/test_runtime_error_telemetry.py",
        "tests/test_audio_quality.py",
        "tests/test_reference_suite_real_world_extension.py",
        "tests/test_session_log_privacy.py",
        "tests/test_smart_features.py",
    ]

    cmd = [sys.executable, "-m", "pytest", "-q", *local_tests]
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=1200,
            check=False,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        passed, total, ok = _parse_pytest_summary(output)
        score = (float(passed) / float(total)) if total > 0 else 0.0

        details = {
            "score_index": _index(score),
            "command": " ".join(cmd),
            "exit_code": int(proc.returncode),
            "passed": passed,
            "total": total,
            "all_passed": ok,
            "output_tail": "\n".join(output.strip().splitlines()[-25:]),
        }
        return ComponentResult(name="test_quality", score01=_clamp01(score), details=details)
    except Exception as exc:
        details = {
            "score_index": 0,
            "command": " ".join(cmd),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "all_passed": False,
        }
        return ComponentResult(name="test_quality", score01=0.0, details=details)


def _run_module_sanity_component() -> ComponentResult:
    checks = []

    try:
        from core.light_mapping import fuse_modalities

        out = fuse_modalities(
            video_ema={"happy": 1.0, "sad": 0.0, "neutral": 0.0, "angry": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0},
            audio_ema={"happy": 0.0, "sad": 1.0, "neutral": 0.0, "angry": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0},
            pose_arousal_offset=0.0,
            audio_weight=0.5,
        )
        checks.append(("fusion_normalized", abs(sum(out.values()) - 1.0) < 1e-6))
    except Exception:
        checks.append(("fusion_normalized", False))

    try:
        from core.emotion_regulator import EmotionRegulator

        reg = EmotionRegulator(0.65, 0.35, 0.45, 0.8, 30.0, 0.1, 0.18)
        info = reg.update(-0.8, 0.8)
        checks.append(("regulator_moves_toward_target", info["reg_v"] > -0.8 and info["reg_a"] < 0.8))
    except Exception:
        checks.append(("regulator_moves_toward_target", False))

    try:
        from analyzers.hrv_analyzer import _compute_hr_hrv

        ibis = np.array([0.80, 0.82, 0.79, 0.81], dtype=np.float64)
        hr, rmssd, sdnn = _compute_hr_hrv(ibis)
        checks.append(("hrv_signal_math", 65.0 <= hr <= 80.0 and rmssd >= 0.0 and sdnn >= 0.0))
    except Exception:
        checks.append(("hrv_signal_math", False))

    try:
        from analyzers.breathing_analyzer import _find_peaks_simple

        signal = np.array([0.0, 1.0, 0.0, 1.1, 0.0, 1.2, 0.0], dtype=np.float64)
        peaks = _find_peaks_simple(signal, min_dist=1)
        checks.append(("breathing_peak_detection", len(peaks) >= 3))
    except Exception:
        checks.append(("breathing_peak_detection", False))

    try:
        from analyzers.face_mesh_analyzer import FaceMeshAnalyzer

        aus = {
            "AU1": 0.2,
            "AU2": 0.2,
            "AU4": 0.1,
            "AU6": 0.8,
            "AU9": 0.1,
            "AU12": 0.9,
            "AU15": 0.1,
            "AU20": 0.1,
            "AU25": 0.2,
            "AU26": 0.2,
        }
        scores = FaceMeshAnalyzer._aus_to_emotions(aus)
        checks.append(("face_mesh_emotion_mapping", abs(sum(scores.values()) - 100.0) < 1e-6))
    except Exception:
        checks.append(("face_mesh_emotion_mapping", False))

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    score = float(passed) / float(total) if total > 0 else 0.0

    details = {
        "score_index": _index(score),
        "passed": passed,
        "total": total,
        "checks": [{"name": name, "ok": ok} for name, ok in checks],
    }
    return ComponentResult(name="module_sanity", score01=_clamp01(score), details=details)


def _run_e2e_runtime_component(preset: dict) -> ComponentResult:
    from e2e_runtime_benchmark import run_e2e_runtime_benchmark

    e2e_profile = str(preset.get("e2e_profile", "quick"))
    report = run_e2e_runtime_benchmark(profile=e2e_profile, python_executable=sys.executable)
    score = _clamp01(float(report.get("aggregate", {}).get("score01", 0.0)))

    details = {
        "score_index": _index(score),
        "e2e_profile": e2e_profile,
        "aggregate": report.get("aggregate", {}),
        "settings": report.get("settings", {}),
        "scenarios": report.get("scenarios", []),
    }
    return ComponentResult(name="e2e_runtime", score01=score, details=details)


def _load_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _gate(
    current: dict,
    baseline: dict | None,
    max_composite_drop: int,
    max_component_drop: float,
    fail_on_benchmark_mismatch: bool = True,
    skipped_components: list[str] | None = None,
) -> dict:
    skipped = set(skipped_components or [])

    if baseline is None:
        return {
            "baseline_present": False,
            "pass": True,
            "reasons": ["No baseline file found. Gate skipped."],
            "failures": [],
            "warnings": ["No baseline file found. Gate skipped."],
        }

    failures: list[str] = []
    warnings: list[str] = []
    baseline_benchmark = str(baseline.get("benchmark", ""))
    current_benchmark = str(current.get("benchmark", ""))
    benchmark_compatible = baseline_benchmark == current_benchmark
    current_idx = int(current["composite"]["index"])
    base_idx = int(baseline.get("composite", {}).get("index", 0))

    if skipped:
        warnings.append(
            "Gate partial-check mode: skipped components="
            + ", ".join(sorted(skipped))
            + ". Composite drop check disabled."
        )

    if benchmark_compatible:
        if (not skipped) and current_idx < (base_idx - int(max_composite_drop)):
            failures.append(
                f"Composite index dropped too much: current={current_idx}, baseline={base_idx}, allowed_drop={max_composite_drop}"
            )
    else:
        mismatch_msg = (
            "Benchmark version/schema mismatch between current and baseline "
            f"({current_benchmark} vs {baseline_benchmark})."
        )
        if fail_on_benchmark_mismatch:
            failures.append(
                mismatch_msg
                + " Gate failed by policy. Refresh baseline only after explicit approval and evidence."
            )
        else:
            warnings.append(mismatch_msg + " Composite gate check skipped due to override.")

    cur_components = current.get("components", {})
    base_components = baseline.get("components", {})
    for name, cur in cur_components.items():
        if name in skipped:
            continue
        if name not in base_components:
            continue
        base = base_components[name]
        cur_score = float(cur.get("score01", 0.0))
        base_score = float(base.get("score01", 0.0))
        if cur_score < (base_score - float(max_component_drop)):
            failures.append(
                f"Component regression {name}: current={cur_score:.4f}, baseline={base_score:.4f}, allowed_drop={max_component_drop:.4f}"
            )

    reasons = failures + warnings

    return {
        "baseline_present": True,
        "pass": len(failures) == 0,
        "reasons": reasons if reasons else ["All gate checks passed."],
        "failures": failures,
        "warnings": warnings,
        "baseline_index": base_idx,
        "current_index": current_idx,
        "benchmark_compatible": benchmark_compatible,
        "baseline_benchmark": baseline_benchmark,
        "current_benchmark": current_benchmark,
    }


def _relative_to_baseline(current: dict, baseline: dict | None) -> dict:
    if baseline is None:
        return {
            "baseline_present": False,
            "composite_index_percent_of_baseline": None,
            "components_index_percent_of_baseline": {},
            "components_score01_percent_of_baseline": {},
            "note": "No baseline file found.",
        }

    base_comp = baseline.get("composite", {})
    cur_comp = current.get("composite", {})
    base_idx = float(base_comp.get("index", 0))
    cur_idx = float(cur_comp.get("index", 0))

    composite_pct = None
    if base_idx > 0.0:
        composite_pct = (cur_idx / base_idx) * 100.0

    per_component_idx_pct = {}
    per_component_score_pct = {}
    current_components = current.get("components", {})
    baseline_components = baseline.get("components", {})

    for name, cur_data in current_components.items():
        base_data = baseline_components.get(name, {})
        cur_score = float(cur_data.get("score01", 0.0))
        base_score = float(base_data.get("score01", 0.0))
        cur_index = float(_index(cur_score))
        base_index = float(_index(base_score))

        per_component_score_pct[name] = ((cur_score / base_score) * 100.0) if base_score > 0.0 else None
        per_component_idx_pct[name] = ((cur_index / base_index) * 100.0) if base_index > 0.0 else None

    return {
        "baseline_present": True,
        "composite_index_percent_of_baseline": composite_pct,
        "components_index_percent_of_baseline": per_component_idx_pct,
        "components_score01_percent_of_baseline": per_component_score_pct,
    }


def _derive_improvement_targets(components: dict) -> list[str]:
    rows = [(name, float(data.get("score01", 0.0))) for name, data in components.items()]
    rows.sort(key=lambda x: x[1])
    suggestions = []
    for name, score in rows[:2]:
        if name == "extreme_visual_robustness":
            suggestions.append(f"Prioritaet: visuelle Robustheit steigern (aktuell {score:.3f}).")
        elif name == "multi_seed_stability":
            suggestions.append(f"Prioritaet: seed-uebergreifende Stabilitaet verbessern (aktuell {score:.3f}).")
        elif name == "test_quality":
            suggestions.append(f"Prioritaet: fehlschlagende Tests beheben (aktuell {score:.3f}).")
        elif name == "module_sanity":
            suggestions.append(f"Prioritaet: Modul-Sanity-Checks reparieren (aktuell {score:.3f}).")
    return suggestions


def _run_real_world_uncertainty_extension(real_world_globs: list[str], low_conf_threshold: float) -> dict:
    """Runs optional real-world and uncertainty evaluation from JSONL session data."""
    files: list[str] = []
    for pattern in real_world_globs:
        files.extend(glob.glob(pattern))
    files = sorted(set(os.path.normpath(p) for p in files if os.path.isfile(p)))

    if not files:
        return {
            "available": False,
            "reason": "No real-world JSONL files found.",
            "input_files": [],
        }

    from real_world_eval import run_eval

    report = run_eval(files, low_conf_threshold=float(low_conf_threshold))
    weighted_score = float(report.get("metrics", {}).get("weighted_score", 0.0))
    uncertainty = report.get("uncertainty", {})

    return {
        "available": True,
        "input_files": files,
        "score01": _clamp01(weighted_score),
        "score_index": _index(weighted_score),
        "metrics": report.get("metrics", {}),
        "uncertainty": uncertainty,
        "scenarios": report.get("scenarios", {}),
        "n_valid_records": int(report.get("n_valid_records", 0)),
        "n_records": int(report.get("n_records", 0)),
    }


def run_suite(args: argparse.Namespace) -> dict:
    from accuracy_benchmark import FACE_MESH_WEIGHT, HEAD_POSE_STRENGTH

    preset = PROFILE_PRESETS[args.profile]

    run_started = time.perf_counter()
    env_metadata = _collect_environment_metadata()
    skipped_components: list[str] = []

    t0 = time.perf_counter()
    comp_extreme = _run_extreme_component(preset=preset, detector=args.detector, seed=args.seed)
    t_extreme = time.perf_counter() - t0

    t0 = time.perf_counter()
    comp_stability = _run_stability_component(preset=preset, detector=args.detector)
    t_stability = time.perf_counter() - t0

    if args.skip_tests:
        skipped_components.append("test_quality")
        comp_tests = ComponentResult(
            name="test_quality",
            score01=0.0,
            details={
                "score_index": 0,
                "skipped": True,
                "reason": "Skipped by --skip-tests",
            },
        )
        t_tests = 0.0
    else:
        t0 = time.perf_counter()
        comp_tests = _run_test_component()
        t_tests = time.perf_counter() - t0

    t0 = time.perf_counter()
    comp_sanity = _run_module_sanity_component()
    t_sanity = time.perf_counter() - t0

    if args.skip_e2e:
        skipped_components.append("e2e_runtime")
        comp_e2e = ComponentResult(
            name="e2e_runtime",
            score01=0.0,
            details={
                "score_index": 0,
                "skipped": True,
                "reason": "Skipped by --skip-e2e",
            },
        )
        t_e2e = 0.0
    else:
        t0 = time.perf_counter()
        comp_e2e = _run_e2e_runtime_component(preset=preset)
        t_e2e = time.perf_counter() - t0

    t0 = time.perf_counter()
    real_world_uncertainty = _run_real_world_uncertainty_extension(
        real_world_globs=list(args.real_world_glob),
        low_conf_threshold=float(args.real_world_low_conf_threshold),
    )
    t_real_world_uncertainty = time.perf_counter() - t0

    components = {
        comp_extreme.name: {"score01": comp_extreme.score01, "details": comp_extreme.details},
        comp_stability.name: {"score01": comp_stability.score01, "details": comp_stability.details},
        comp_tests.name: {"score01": comp_tests.score01, "details": comp_tests.details},
        comp_sanity.name: {"score01": comp_sanity.score01, "details": comp_sanity.details},
        comp_e2e.name: {"score01": comp_e2e.score01, "details": comp_e2e.details},
    }

    # Fixed weights to prevent hiding weak areas by re-weighting.
    weights = {
        "extreme_visual_robustness": 0.40,
        "multi_seed_stability": 0.23,
        "test_quality": 0.17,
        "module_sanity": 0.10,
        "e2e_runtime": 0.10,
    }

    active_components = [name for name in weights if name not in skipped_components]
    active_weight_sum = sum(float(weights[name]) for name in active_components)
    composite_score = 0.0
    if active_weight_sum > 0.0:
        for name in active_components:
            normalized_weight = float(weights[name]) / active_weight_sum
            composite_score += float(components[name]["score01"]) * normalized_weight
        composite_score = _clamp01(composite_score)

    report = {
        "benchmark": "reference_suite_v2",
        "generated_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "profile": args.profile,
        "settings": {
            "seed": int(args.seed),
            "detector": args.detector,
            "preset": preset,
            "weights": weights,
            "active_components": active_components,
            "skipped_components": sorted(skipped_components),
            "face_mesh_weight": float(FACE_MESH_WEIGHT),
            "head_pose_strength": float(HEAD_POSE_STRENGTH),
        },
        "environment": env_metadata,
        "components": components,
        "composite": {
            "score01": composite_score,
            "index": _index(composite_score),
        },
        "runtime": {
            "total_seconds": float(time.perf_counter() - run_started),
            "component_seconds": {
                "extreme_visual_robustness": float(t_extreme),
                "multi_seed_stability": float(t_stability),
                "test_quality": float(t_tests),
                "module_sanity": float(t_sanity),
                "e2e_runtime": float(t_e2e),
                "real_world_uncertainty": float(t_real_world_uncertainty),
            },
        },
        "extensions": {
            "real_world_uncertainty": real_world_uncertainty,
        },
    }

    baseline = _load_json(args.baseline)
    report["gate"] = _gate(
        current=report,
        baseline=baseline,
        max_composite_drop=args.max_composite_drop,
        max_component_drop=args.max_component_drop,
        fail_on_benchmark_mismatch=not args.allow_incompatible_baseline,
        skipped_components=skipped_components,
    )
    report["relative_to_baseline"] = _relative_to_baseline(current=report, baseline=baseline)
    report["improvement_targets"] = _derive_improvement_targets(components)

    return report


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_last_history_entry(history_path: str, profile: str, detector: str) -> dict | None:
    """Load the latest comparable history entry.

    Preference order:
    1) same profile + detector
    2) same profile
    3) latest any entry
    """
    if not os.path.isfile(history_path):
        return None

    entries: list[dict] = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue

    if not entries:
        return None

    same_profile_detector = [
        e for e in entries
        if e.get("profile") == profile and e.get("detector") == detector
    ]
    if same_profile_detector:
        return same_profile_detector[-1]

    same_profile = [e for e in entries if e.get("profile") == profile]
    if same_profile:
        return same_profile[-1]

    return entries[-1]


def _make_history_entry(report: dict) -> dict:
    component_indices = {
        name: _index(float(data.get("score01", 0.0)))
        for name, data in report.get("components", {}).items()
    }
    return {
        "generated_at_utc": report.get("generated_at_utc"),
        "profile": report.get("profile"),
        "detector": report.get("settings", {}).get("detector"),
        "composite_index": int(report.get("composite", {}).get("index", 0)),
        "component_indices": component_indices,
        "gate_pass": bool(report.get("gate", {}).get("pass", False)),
    }


def _append_history(history_path: str, entry: dict) -> None:
    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _build_trend_block(report: dict, previous_entry: dict | None) -> dict:
    current_entry = _make_history_entry(report)
    if previous_entry is None:
        return {
            "has_previous": False,
            "basis": "none",
            "current": current_entry,
            "delta": None,
            "summary": "No previous run in history.",
        }

    prev_comp = previous_entry.get("component_indices", {})
    cur_comp = current_entry.get("component_indices", {})

    per_component_delta = {}
    for name, cur_idx in cur_comp.items():
        per_component_delta[name] = int(cur_idx) - int(prev_comp.get(name, 0))

    comp_delta = int(current_entry["composite_index"]) - int(previous_entry.get("composite_index", 0))
    direction = "improved" if comp_delta > 0 else ("regressed" if comp_delta < 0 else "flat")
    basis = "same-profile+detector" if (
        previous_entry.get("profile") == current_entry.get("profile")
        and previous_entry.get("detector") == current_entry.get("detector")
    ) else (
        "same-profile" if previous_entry.get("profile") == current_entry.get("profile") else "latest-any"
    )

    return {
        "has_previous": True,
        "basis": basis,
        "previous": previous_entry,
        "current": current_entry,
        "delta": {
            "composite_index": comp_delta,
            "component_indices": per_component_delta,
        },
        "summary": f"Composite {direction} by {comp_delta:+d} points vs previous comparable run.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full reference benchmark suite")
    parser.add_argument("--profile", "--mode", dest="profile", choices=["quick", "standard", "strict"], default="standard")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--detector", type=str, default="opencv")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("benchmarks", "results", "reference_suite_latest.json"),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=os.path.join("benchmarks", "results", "reference_suite_baseline.json"),
    )
    parser.add_argument("--max-composite-drop", type=int, default=15)
    parser.add_argument("--max-component-drop", type=float, default=0.04)
    parser.add_argument(
        "--allow-incompatible-baseline",
        action="store_true",
        help="Allow gate to continue when current and baseline benchmark schemas differ.",
    )
    parser.add_argument(
        "--enforce-gate",
        action="store_true",
        help="Exit non-zero when gate checks fail.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Also write current run to baseline path.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip test_quality component for faster local iteration (not valid with --enforce-gate).",
    )
    parser.add_argument(
        "--skip-e2e",
        action="store_true",
        help="Skip e2e_runtime component for faster local iteration (not valid with --enforce-gate).",
    )
    parser.add_argument(
        "--history",
        type=str,
        default=os.path.join("benchmarks", "results", "reference_suite_history.jsonl"),
        help="Path to append run history as JSONL.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable reading/appending history entries.",
    )
    parser.add_argument(
        "--real-world-glob",
        nargs="+",
        default=[os.path.join("benchmarks", "results", "real_world", "*.jsonl")],
        help="Glob pattern(s) for real-world JSONL evaluation inputs.",
    )
    parser.add_argument(
        "--real-world-low-conf-threshold",
        type=float,
        default=0.45,
        help="Low-confidence threshold used by real-world uncertainty evaluation.",
    )
    args = parser.parse_args()

    if args.enforce_gate and (args.skip_tests or args.skip_e2e):
        parser.error("--enforce-gate cannot be used together with --skip-tests or --skip-e2e")

    previous_entry = None
    if not args.no_history:
        previous_entry = _load_last_history_entry(
            history_path=args.history,
            profile=args.profile,
            detector=args.detector,
        )

    report = run_suite(args)
    report["trend"] = _build_trend_block(report, previous_entry)
    _write_json(args.output, report)

    if not args.no_history:
        _append_history(args.history, _make_history_entry(report))

    if args.write_baseline:
        _write_json(args.baseline, report)

    print("\n=== Reference Suite ===")
    print(f"Profile: {report['profile']}")
    print(f"Composite index: {report['composite']['index']}")
    for cname, cdata in report["components"].items():
        print(f"- {cname}: idx={_index(cdata['score01'])}")
    print(f"Trend: {report['trend']['summary']}")
    print(f"Gate: {'PASS' if report['gate']['pass'] else 'FAIL'}")
    for line in report["gate"]["reasons"]:
        print(f"  * {line}")
    print(f"Report: {args.output}")

    if args.enforce_gate and not report["gate"]["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
