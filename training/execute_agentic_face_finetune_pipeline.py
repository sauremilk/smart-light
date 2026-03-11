#!/usr/bin/env python3
"""Master orchestrator for fully automated face fine-tuning.

Pipeline stages:
1) Dataset generation gate (>= 1000 samples)
2) Fine-tune gate (val_accuracy >= target)
3) Reference suite gate (standard or strict)
4) Optional retry with half learning rate if gate fails
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path


log = logging.getLogger("agentic-face-pipeline")
ROOT = Path(__file__).resolve().parent


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    log.info("Running command: %s", " ".join(cmd))
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.stdout:
        log.info("stdout:\n%s", proc.stdout)
    if proc.stderr:
        log.info("stderr:\n%s", proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_reference_suite(py: str, profile: str, enforce_gate: bool) -> dict:
    cmd = [py, "benchmarks/reference_suite.py", "--profile", profile]
    if enforce_gate:
        cmd.append("--enforce-gate")
    _run_cmd(cmd, cwd=ROOT)
    latest = ROOT / "benchmarks" / "results" / "reference_suite_latest.json"
    if not latest.exists():
        raise RuntimeError("reference_suite_latest.json not found after benchmark run")
    return _load_json(latest)


def _run_finetune(py: str, lr: float, target_acc: float, backend: str | None) -> dict:
    cmd = [
        py,
        "finetune_face_agentic.py",
        "--dataset-dir",
        "dataset/face_finetune",
        "--output-dir",
        "artifacts/face_finetune",
        "--target-val-acc",
        str(target_acc),
        "--lr",
        str(lr),
    ]
    if backend:
        cmd.extend(["--backend", backend])
    _run_cmd(cmd, cwd=ROOT)
    summary = ROOT / "artifacts" / "face_finetune" / "train_summary.json"
    if not summary.exists():
        raise RuntimeError("Train summary missing: artifacts/face_finetune/train_summary.json")
    return _load_json(summary)


def _run_dataset(py: str, min_samples: int, webcam_index: int, skip_dataset: bool) -> dict:
    if skip_dataset:
        manifest = ROOT / "dataset" / "face_finetune" / "dataset_manifest.json"
        if not manifest.exists():
            raise RuntimeError("--skip-dataset set but dataset manifest is missing")
        return _load_json(manifest)

    cmd = [
        py,
        "agentic_dataset_gen.py",
        "--output-dir",
        "dataset/face_finetune",
        "--min-total-samples",
        str(min_samples),
        "--webcam-index",
        str(webcam_index),
    ]
    _run_cmd(cmd, cwd=ROOT)
    manifest = ROOT / "dataset" / "face_finetune" / "dataset_manifest.json"
    if not manifest.exists():
        raise RuntimeError("Dataset manifest missing: dataset/face_finetune/dataset_manifest.json")
    return _load_json(manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute complete agentic face fine-tuning workflow.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for sub-commands.")
    parser.add_argument("--min-samples", type=int, default=1000, help="Dataset minimum sample gate.")
    parser.add_argument("--target-val-acc", type=float, default=0.82, help="Validation target gate.")
    parser.add_argument("--benchmark-profile", choices=["quick", "standard", "strict"], default="strict")
    parser.add_argument("--webcam-index", type=int, default=0, help="Webcam index for dataset generation.")
    parser.add_argument("--backend", choices=["autotrain", "torch"], default=None, help="Force train backend.")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset creation and reuse existing manifest.")
    parser.add_argument("--retry-on-fail", action="store_true", help="Retry training with lr/2 after benchmark gate fail.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    t0 = time.time()

    dataset_manifest = _run_dataset(
        py=args.python,
        min_samples=int(args.min_samples),
        webcam_index=int(args.webcam_index),
        skip_dataset=bool(args.skip_dataset),
    )

    if int(dataset_manifest.get("total_samples", 0)) < int(args.min_samples):
        raise RuntimeError("Dataset gate failed in master pipeline.")

    train_summary = _run_finetune(
        py=args.python,
        lr=2e-4,
        target_acc=float(args.target_val_acc),
        backend=args.backend,
    )

    bench_report = _run_reference_suite(
        py=args.python,
        profile=args.benchmark_profile,
        enforce_gate=True,
    )
    gate = bench_report.get("gate", {})
    gate_pass = bool(gate.get("pass", False))

    retry_summary = None
    retry_bench = None
    if (not gate_pass) and bool(args.retry_on_fail):
        retry_summary = _run_finetune(
            py=args.python,
            lr=1e-4,
            target_acc=float(args.target_val_acc),
            backend=args.backend,
        )
        retry_bench = _run_reference_suite(
            py=args.python,
            profile=args.benchmark_profile,
            enforce_gate=True,
        )
        gate_pass = bool(retry_bench.get("gate", {}).get("pass", False))

    output = {
        "started_at_unix": int(t0),
        "runtime_seconds": float(time.time() - t0),
        "dataset": dataset_manifest,
        "training": train_summary,
        "benchmark": {
            "profile": args.benchmark_profile,
            "composite_index": bench_report.get("composite", {}).get("index"),
            "gate": gate,
        },
        "retry_training": retry_summary,
        "retry_benchmark": retry_bench,
        "final_gate_pass": gate_pass,
    }

    out_path = ROOT / "artifacts" / "face_finetune" / "pipeline_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
