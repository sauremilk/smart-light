#!/usr/bin/env python3
"""Real-world evaluation for emotion inference and uncertainty behavior.

Expected input: JSONL records following benchmarks/real_world_eval_schema.json.
This script intentionally avoids heavy dependencies so it can run in CI/dev setups.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import time
from collections import defaultdict


EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _load_jsonl(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rows.append(rec)
    return rows


def _accuracy(records: list[dict]) -> float:
    if not records:
        return 0.0
    correct = 0
    for r in records:
        if r.get("ground_truth_emotion") == r.get("predicted_emotion"):
            correct += 1
    return correct / float(len(records))


def _macro_f1(records: list[dict]) -> float:
    if not records:
        return 0.0

    f1_values: list[float] = []
    for label in EMOTIONS:
        tp = fp = fn = 0
        for r in records:
            y_true = r.get("ground_truth_emotion")
            y_pred = r.get("predicted_emotion")
            if y_pred == label and y_true == label:
                tp += 1
            elif y_pred == label and y_true != label:
                fp += 1
            elif y_pred != label and y_true == label:
                fn += 1

        if tp == 0 and fp == 0 and fn == 0:
            # Class not present in this slice; skip instead of forcing 0.
            continue

        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall <= 1e-12:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1_values.append(f1)

    return _safe_mean(f1_values)


def _ece(records: list[dict], bins: int = 10) -> float:
    if not records:
        return 0.0

    bucket_total = [0 for _ in range(bins)]
    bucket_acc_sum = [0.0 for _ in range(bins)]
    bucket_conf_sum = [0.0 for _ in range(bins)]

    for r in records:
        conf = _clamp01(float(r.get("prediction_confidence", 0.0)))
        correct = 1.0 if r.get("ground_truth_emotion") == r.get("predicted_emotion") else 0.0
        idx = min(bins - 1, int(conf * bins))
        bucket_total[idx] += 1
        bucket_acc_sum[idx] += correct
        bucket_conf_sum[idx] += conf

    n = float(len(records))
    ece = 0.0
    for i in range(bins):
        if bucket_total[i] == 0:
            continue
        w = bucket_total[i] / n
        avg_acc = bucket_acc_sum[i] / float(bucket_total[i])
        avg_conf = bucket_conf_sum[i] / float(bucket_total[i])
        ece += w * abs(avg_acc - avg_conf)
    return ece


def _scenario_key(rec: dict) -> str:
    s = rec.get("scenario") or {}
    lighting = str(s.get("lighting", "unknown"))
    occlusion = str(s.get("occlusion", "unknown"))
    head_pose = str(s.get("head_pose", "unknown"))
    noise = str(s.get("background_noise", "unknown"))
    return f"lighting={lighting}|occlusion={occlusion}|head_pose={head_pose}|noise={noise}"


def _uncertainty_metrics(records: list[dict], low_conf_threshold: float) -> dict:
    if not records:
        return {
            "low_conf_threshold": low_conf_threshold,
            "low_conf_rate": 0.0,
            "error_rate_low_conf": 0.0,
            "error_rate_high_conf": 0.0,
            "ece": 0.0,
            "guardrail_activation_rate": 0.0,
        }

    low = []
    high = []
    guardrail_count = 0
    for r in records:
        conf = _clamp01(float(r.get("prediction_confidence", 0.0)))
        err = 0.0 if r.get("ground_truth_emotion") == r.get("predicted_emotion") else 1.0
        if bool(r.get("guardrail_active", False)):
            guardrail_count += 1
        if conf < low_conf_threshold:
            low.append(err)
        else:
            high.append(err)

    return {
        "low_conf_threshold": low_conf_threshold,
        "low_conf_rate": len(low) / float(len(records)),
        "error_rate_low_conf": _safe_mean(low),
        "error_rate_high_conf": _safe_mean(high),
        "ece": _ece(records),
        "guardrail_activation_rate": guardrail_count / float(len(records)),
    }


def _by_scenario(records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[_scenario_key(r)].append(r)

    out: dict[str, dict] = {}
    for key, rows in sorted(groups.items()):
        out[key] = {
            "n": len(rows),
            "accuracy": _accuracy(rows),
            "macro_f1": _macro_f1(rows),
        }
    return out


def _resolve_inputs(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    if args.input:
        paths.extend(args.input)
    if args.glob:
        for pattern in args.glob:
            paths.extend(glob.glob(pattern))
    uniq = sorted(set(os.path.normpath(p) for p in paths if os.path.isfile(p)))
    return uniq


def run_eval(paths: list[str], low_conf_threshold: float) -> dict:
    records = _load_jsonl(paths)

    valid = [
        r for r in records
        if r.get("ground_truth_emotion") in EMOTIONS
        and r.get("predicted_emotion") in EMOTIONS
    ]

    accuracy = _accuracy(valid)
    macro_f1 = _macro_f1(valid)
    weighted_score = _clamp01(0.4 * accuracy + 0.6 * macro_f1)

    return {
        "generated_at": time.time(),
        "input_files": paths,
        "n_records": len(records),
        "n_valid_records": len(valid),
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_score": weighted_score,
            "index": int(round(weighted_score * 1000.0)),
        },
        "uncertainty": _uncertainty_metrics(valid, low_conf_threshold=low_conf_threshold),
        "scenarios": _by_scenario(valid),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-world emotion evaluation")
    parser.add_argument("--input", nargs="*", default=None, help="Input JSONL file(s)")
    parser.add_argument("--glob", nargs="*", default=None, help="Input glob(s), e.g. benchmarks/results/real_world/*.jsonl")
    parser.add_argument("--low-conf-threshold", type=float, default=0.45, help="Confidence threshold for low/high split")
    parser.add_argument(
        "--output",
        default=os.path.join("benchmarks", "results", "real_world_eval_latest.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    paths = _resolve_inputs(args)
    if not paths:
        print("No input files found. Use --input or --glob.")
        return 2

    report = run_eval(paths, low_conf_threshold=float(args.low_conf_threshold))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Real-World Eval ===")
    print(f"Records: {report['n_valid_records']}/{report['n_records']}")
    print(f"Accuracy: {report['metrics']['accuracy']:.4f}")
    print(f"Macro-F1: {report['metrics']['macro_f1']:.4f}")
    print(f"Index: {report['metrics']['index']}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
