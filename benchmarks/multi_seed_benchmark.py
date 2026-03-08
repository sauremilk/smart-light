#!/usr/bin/env python3
"""Robust multi-seed benchmark runner.

Runs the project benchmark for multiple seeds and reports aggregate metrics:
- baseline vs enhanced (without face mesh)
- baseline vs enhanced (with face mesh)
- direct enhanced-with-vs-without-face-mesh delta

Usage:
  c:/Users/mickg/smart-light/.venv/Scripts/python.exe benchmarks/multi_seed_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import statistics
import argparse
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from accuracy_benchmark import run, FACE_MESH_WEIGHT


@dataclass
class RunResult:
    seed: int
    no_fm: dict
    with_fm: dict


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed benchmark runner")
    parser.add_argument(
        "--face-mesh-weight",
        type=float,
        default=None,
        help="Override face mesh fusion weight used in accuracy benchmark",
    )
    parser.add_argument(
        "--head-pose-strength",
        type=float,
        default=1.0,
        help="Strength of head-pose confidence attenuation (0.0-1.0)",
    )
    parser.add_argument(
        "--no-head-pose-penalty",
        action="store_true",
        help="Disable head-pose confidence attenuation",
    )
    args = parser.parse_args()

    seeds = [1, 2, 3, 4, 5]
    limit = 48
    detector = "opencv"
    effective_weight = (
        float(args.face_mesh_weight) if args.face_mesh_weight is not None else float(FACE_MESH_WEIGHT)
    )
    effective_head_pose_strength = 0.0 if args.no_head_pose_penalty else float(args.head_pose_strength)

    per_seed: list[RunResult] = []

    for seed in seeds:
        print(f"Running seed={seed} no-face-mesh...")
        no_fm = run(
            limit=limit,
            seed=seed,
            detector_backend=detector,
            use_face_mesh=False,
            face_mesh_weight=effective_weight,
            head_pose_strength=effective_head_pose_strength,
        )

        print(f"Running seed={seed} with-face-mesh...")
        with_fm = run(
            limit=limit,
            seed=seed,
            detector_backend=detector,
            use_face_mesh=True,
            face_mesh_weight=effective_weight,
            head_pose_strength=effective_head_pose_strength,
        )

        per_seed.append(RunResult(seed=seed, no_fm=no_fm, with_fm=with_fm))

    no_fm_acc_delta = [r.no_fm["delta"]["accuracy"] for r in per_seed]
    no_fm_f1_delta = [r.no_fm["delta"]["macro_f1"] for r in per_seed]

    with_fm_acc_delta = [r.with_fm["delta"]["accuracy"] for r in per_seed]
    with_fm_f1_delta = [r.with_fm["delta"]["macro_f1"] for r in per_seed]

    face_mesh_extra_acc = [
        r.with_fm["enhanced"]["accuracy"] - r.no_fm["enhanced"]["accuracy"]
        for r in per_seed
    ]
    face_mesh_extra_f1 = [
        r.with_fm["enhanced"]["macro_f1"] - r.no_fm["enhanced"]["macro_f1"]
        for r in per_seed
    ]

    summary = {
        "settings": {
            "seeds": seeds,
            "limit": limit,
            "detector": detector,
            "face_mesh_weight": effective_weight,
            "head_pose_strength": effective_head_pose_strength,
        },
        "aggregate": {
            "enhanced_minus_baseline_no_face_mesh": {
                "accuracy_mean": _mean(no_fm_acc_delta),
                "accuracy_stdev": _stdev(no_fm_acc_delta),
                "macro_f1_mean": _mean(no_fm_f1_delta),
                "macro_f1_stdev": _stdev(no_fm_f1_delta),
            },
            "enhanced_minus_baseline_with_face_mesh": {
                "accuracy_mean": _mean(with_fm_acc_delta),
                "accuracy_stdev": _stdev(with_fm_acc_delta),
                "macro_f1_mean": _mean(with_fm_f1_delta),
                "macro_f1_stdev": _stdev(with_fm_f1_delta),
            },
            "enhanced_with_face_mesh_minus_enhanced_no_face_mesh": {
                "accuracy_mean": _mean(face_mesh_extra_acc),
                "accuracy_stdev": _stdev(face_mesh_extra_acc),
                "macro_f1_mean": _mean(face_mesh_extra_f1),
                "macro_f1_stdev": _stdev(face_mesh_extra_f1),
            },
        },
        "per_seed": [
            {
                "seed": r.seed,
                "no_face_mesh": {
                    "baseline_accuracy": r.no_fm["baseline"]["accuracy"],
                    "baseline_macro_f1": r.no_fm["baseline"]["macro_f1"],
                    "enhanced_accuracy": r.no_fm["enhanced"]["accuracy"],
                    "enhanced_macro_f1": r.no_fm["enhanced"]["macro_f1"],
                    "delta_accuracy": r.no_fm["delta"]["accuracy"],
                    "delta_macro_f1": r.no_fm["delta"]["macro_f1"],
                },
                "with_face_mesh": {
                    "baseline_accuracy": r.with_fm["baseline"]["accuracy"],
                    "baseline_macro_f1": r.with_fm["baseline"]["macro_f1"],
                    "enhanced_accuracy": r.with_fm["enhanced"]["accuracy"],
                    "enhanced_macro_f1": r.with_fm["enhanced"]["macro_f1"],
                    "delta_accuracy": r.with_fm["delta"]["accuracy"],
                    "delta_macro_f1": r.with_fm["delta"]["macro_f1"],
                },
                "face_mesh_extra": {
                    "enhanced_accuracy_delta": (
                        r.with_fm["enhanced"]["accuracy"] - r.no_fm["enhanced"]["accuracy"]
                    ),
                    "enhanced_macro_f1_delta": (
                        r.with_fm["enhanced"]["macro_f1"] - r.no_fm["enhanced"]["macro_f1"]
                    ),
                },
            }
            for r in per_seed
        ],
    }

    os.makedirs("benchmarks/results", exist_ok=True)
    out_path = "benchmarks/results/multi_seed_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Multi-Seed Summary ===")
    agg = summary["aggregate"]
    no_fm = agg["enhanced_minus_baseline_no_face_mesh"]
    fm = agg["enhanced_minus_baseline_with_face_mesh"]
    fm_extra = agg["enhanced_with_face_mesh_minus_enhanced_no_face_mesh"]
    print(
        "No-FM delta mean: "
        f"acc={no_fm['accuracy_mean']:+.4f} +- {no_fm['accuracy_stdev']:.4f}, "
        f"f1={no_fm['macro_f1_mean']:+.4f} +- {no_fm['macro_f1_stdev']:.4f}"
    )
    print(
        "With-FM delta mean: "
        f"acc={fm['accuracy_mean']:+.4f} +- {fm['accuracy_stdev']:.4f}, "
        f"f1={fm['macro_f1_mean']:+.4f} +- {fm['macro_f1_stdev']:.4f}"
    )
    print(
        "Face-Mesh extra mean: "
        f"acc={fm_extra['accuracy_mean']:+.4f} +- {fm_extra['accuracy_stdev']:.4f}, "
        f"f1={fm_extra['macro_f1_mean']:+.4f} +- {fm_extra['macro_f1_stdev']:.4f}"
    )
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
