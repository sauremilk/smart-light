#!/usr/bin/env python3
"""Quick grid search for benchmark tuning.

Purpose:
- Keep tuning reproducible and comparable across runs.
- Evaluate key levers that impacted quality in this project:
  1) face_mesh_weight
  2) head_pose_strength

Usage:
  c:/Users/mickg/smart-light/.venv/Scripts/python.exe benchmarks/tune_accuracy.py
  c:/Users/mickg/smart-light/.venv/Scripts/python.exe benchmarks/tune_accuracy.py --limit 48 --seeds 1,2,3,4,5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from accuracy_benchmark import run, FACE_MESH_WEIGHT, HEAD_POSE_STRENGTH


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _parse_csv_floats(raw: str) -> list[float]:
    out: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out


def _parse_csv_ints(raw: str) -> list[int]:
    out: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def evaluate_setting(
    seeds: list[int],
    limit: int,
    detector: str,
    face_mesh_weight: float,
    head_pose_strength: float,
) -> dict:
    no_fm_enh_acc: list[float] = []
    with_fm_enh_acc: list[float] = []
    no_fm_delta_acc: list[float] = []
    with_fm_delta_acc: list[float] = []
    no_fm_delta_f1: list[float] = []
    with_fm_delta_f1: list[float] = []

    for seed in seeds:
        no_fm = run(
            limit=limit,
            seed=seed,
            detector_backend=detector,
            use_face_mesh=False,
            face_mesh_weight=face_mesh_weight,
            head_pose_strength=head_pose_strength,
        )
        with_fm = run(
            limit=limit,
            seed=seed,
            detector_backend=detector,
            use_face_mesh=True,
            face_mesh_weight=face_mesh_weight,
            head_pose_strength=head_pose_strength,
        )

        no_fm_enh_acc.append(float(no_fm["enhanced"]["accuracy"]))
        with_fm_enh_acc.append(float(with_fm["enhanced"]["accuracy"]))
        no_fm_delta_acc.append(float(no_fm["delta"]["accuracy"]))
        with_fm_delta_acc.append(float(with_fm["delta"]["accuracy"]))
        no_fm_delta_f1.append(float(no_fm["delta"]["macro_f1"]))
        with_fm_delta_f1.append(float(with_fm["delta"]["macro_f1"]))

    fm_extra_acc = [w - n for w, n in zip(with_fm_enh_acc, no_fm_enh_acc)]

    return {
        "face_mesh_weight": face_mesh_weight,
        "head_pose_strength": head_pose_strength,
        "metrics": {
            "enhanced_no_fm_acc_mean": _mean(no_fm_enh_acc),
            "enhanced_with_fm_acc_mean": _mean(with_fm_enh_acc),
            "no_fm_delta_acc_mean": _mean(no_fm_delta_acc),
            "with_fm_delta_acc_mean": _mean(with_fm_delta_acc),
            "no_fm_delta_f1_mean": _mean(no_fm_delta_f1),
            "with_fm_delta_f1_mean": _mean(with_fm_delta_f1),
            "fm_extra_acc_mean": _mean(fm_extra_acc),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick accuracy tuning grid")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--detector", type=str, default="opencv")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument(
        "--face-mesh-weights",
        type=str,
        default=f"{FACE_MESH_WEIGHT:.2f},0.10,0.20,0.30",
        help="Comma-separated list, e.g. 0.1,0.2,0.3",
    )
    parser.add_argument(
        "--head-pose-strengths",
        type=str,
        default=f"{HEAD_POSE_STRENGTH:.2f},0.00,0.20,0.50",
        help="Comma-separated list, e.g. 0.0,0.2,0.5",
    )
    args = parser.parse_args()

    seeds = _parse_csv_ints(args.seeds)
    fm_weights = _parse_csv_floats(args.face_mesh_weights)
    hp_strengths = _parse_csv_floats(args.head_pose_strengths)

    # Keep insertion order but drop duplicates from defaults.
    fm_weights = list(dict.fromkeys(round(v, 4) for v in fm_weights))
    hp_strengths = list(dict.fromkeys(round(v, 4) for v in hp_strengths))

    rows: list[dict] = []
    for fmw in fm_weights:
        for hps in hp_strengths:
            row = evaluate_setting(
                seeds=seeds,
                limit=args.limit,
                detector=args.detector,
                face_mesh_weight=fmw,
                head_pose_strength=hps,
            )
            rows.append(row)
            m = row["metrics"]
            print(
                f"fm_w={fmw:.2f} hp_s={hps:.2f} "
                f"with_fm_acc={m['enhanced_with_fm_acc_mean']:.4f} "
                f"with_fm_delta={m['with_fm_delta_acc_mean']:+.4f} "
                f"fm_extra={m['fm_extra_acc_mean']:+.4f}"
            )

    # Prioritize setups that maximize enhanced_with_fm accuracy.
    ranked = sorted(rows, key=lambda r: r["metrics"]["enhanced_with_fm_acc_mean"], reverse=True)

    summary = {
        "settings": {
            "limit": args.limit,
            "detector": args.detector,
            "seeds": seeds,
            "face_mesh_weights": fm_weights,
            "head_pose_strengths": hp_strengths,
        },
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
    }

    os.makedirs("benchmarks/results", exist_ok=True)
    out_path = "benchmarks/results/tune_accuracy_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if ranked:
        best = ranked[0]
        print("\n=== Best Setting ===")
        print(
            "face_mesh_weight={:.2f}, head_pose_strength={:.2f}, enhanced_with_fm_acc_mean={:.4f}".format(
                best["face_mesh_weight"],
                best["head_pose_strength"],
                best["metrics"]["enhanced_with_fm_acc_mean"],
            )
        )
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
