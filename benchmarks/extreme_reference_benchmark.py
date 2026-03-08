#!/usr/bin/env python3
"""Very challenging, reproducible benchmark for long-term reference tracking.

The goal is not to maximize absolute scores, but to create a stable "stress test"
that can be used as a regression and improvement reference over time.

What this benchmark does:
1. Loads FER-style labeled face samples (same source strategy as accuracy benchmark)
2. Creates deterministic hard variants per sample (low light, blur, noise, occlusion,
   compression artifacts, geometric distortions, color cast, mixed stress)
3. Evaluates baseline and enhanced prediction pipelines on the same hard set
4. Writes a JSON report with a compact reference index for easy run-to-run comparison

Usage:
  c:/Users/mickg/smart-light/.venv/Scripts/python.exe benchmarks/extreme_reference_benchmark.py

Quick smoke run:
  c:/Users/mickg/smart-light/.venv/Scripts/python.exe benchmarks/extreme_reference_benchmark.py --limit 14 --variants-per-sample 1 --no-face-mesh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from accuracy_benchmark import (  # noqa: E402
    FACE_MESH_WEIGHT,
    HEAD_POSE_STRENGTH,
    _HAS_MEDIAPIPE,
    accuracy,
    confusion_matrix,
    create_face_mesh,
    ensure_fer_csv,
    load_fer_subset,
    load_samples_from_urls,
    macro_f1,
    predict_baseline,
    predict_enhanced,
)


@dataclass
class HardSample:
    label: str
    profile: str
    image: np.ndarray


PROFILES = [
    "low_light_noise",
    "motion_blur",
    "jpeg_artifacts",
    "occlusion",
    "rotation_scale",
    "color_cast_shadow",
    "mixed_extreme",
]

PROFILE_SPECS = {
    "low_light_noise": {
        "description": "Gamma darkening, brightness drop, gaussian sensor noise",
        "parameters": {
            "brightness_range": [0.32, 0.58],
            "gamma_range": [1.25, 1.8],
            "noise_std_range": [8.0, 18.0],
        },
    },
    "motion_blur": {
        "description": "Directional blur kernel plus mild gaussian blur",
        "parameters": {
            "kernel_sizes": [5, 7, 9, 11],
            "gaussian_sigma_range": [0.8, 1.6],
        },
    },
    "jpeg_artifacts": {
        "description": "Low quality JPEG compression plus down-up resampling",
        "parameters": {
            "jpeg_quality_range": [8, 28],
            "downsample_factor": 2,
        },
    },
    "occlusion": {
        "description": "Solid block occlusion or patch replacement",
        "parameters": {
            "occlusion_width_fraction": [0.2, 0.45],
            "occlusion_height_fraction": [0.18, 0.4],
            "modes": ["solid_rect", "patch_shuffle"],
        },
    },
    "rotation_scale": {
        "description": "Affine transform with rotation, scale and translation",
        "parameters": {
            "angle_range_deg": [-22.0, 22.0],
            "scale_range": [0.78, 1.18],
            "translation_fraction": [-0.08, 0.08],
        },
    },
    "color_cast_shadow": {
        "description": "Channel gains and directional shadow gradient",
        "parameters": {
            "channel_gain_range": [0.6, 1.35],
            "shadow_strength_range": [0.25, 0.55],
            "shadow_axis": ["x", "y"],
        },
    },
    "mixed_extreme": {
        "description": "Chained transform: rotation -> low light/noise -> occlusion -> JPEG artifacts",
        "parameters": {
            "composition_order": ["rotation_scale", "low_light_noise", "occlusion", "jpeg_artifacts"],
        },
    },
}


def _clamp_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _apply_low_light_noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.astype(np.float32)
    brightness = rng.uniform(0.32, 0.58)
    gamma = rng.uniform(1.25, 1.8)
    out = 255.0 * ((out / 255.0) ** gamma)
    out *= brightness
    noise_std = rng.uniform(8.0, 18.0)
    out += np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0.0, noise_std, out.shape)
    return _clamp_u8(out)


def _apply_motion_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.copy()
    k = int(rng.choice([5, 7, 9, 11]))
    kernel = np.zeros((k, k), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= kernel.sum()
    out = cv2.filter2D(out, -1, kernel)
    out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.8, 1.6))
    return out


def _apply_jpeg_artifacts(img: np.ndarray, rng: random.Random) -> np.ndarray:
    quality = int(rng.randint(8, 28))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if out is None:
        return img
    # Down-up sample to amplify block artifacts.
    h, w = out.shape[:2]
    small = cv2.resize(out, (max(16, w // 2), max(16, h // 2)), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def _apply_occlusion(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    occ_w = int(w * rng.uniform(0.2, 0.45))
    occ_h = int(h * rng.uniform(0.18, 0.4))
    x = int(rng.uniform(0, max(1, w - occ_w)))
    y = int(rng.uniform(0, max(1, h - occ_h)))

    if rng.random() < 0.5:
        color = int(rng.randint(0, 80))
        cv2.rectangle(out, (x, y), (x + occ_w, y + occ_h), (color, color, color), thickness=-1)
    else:
        # Patch shuffle occlusion: replace area with a random crop from another area.
        sx = int(rng.uniform(0, max(1, w - occ_w)))
        sy = int(rng.uniform(0, max(1, h - occ_h)))
        out[y : y + occ_h, x : x + occ_w] = out[sy : sy + occ_h, sx : sx + occ_w]

    return out


def _apply_rotation_scale(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    angle = rng.uniform(-22.0, 22.0)
    scale = rng.uniform(0.78, 1.18)
    tx = rng.uniform(-0.08 * w, 0.08 * w)
    ty = rng.uniform(-0.08 * h, 0.08 * h)

    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return out


def _apply_color_cast_shadow(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.astype(np.float32)
    gains = np.array(
        [
            rng.uniform(0.6, 1.35),
            rng.uniform(0.6, 1.35),
            rng.uniform(0.6, 1.35),
        ],
        dtype=np.float32,
    )
    out *= gains.reshape(1, 1, 3)

    h, w = out.shape[:2]
    shadow_strength = rng.uniform(0.25, 0.55)
    axis = rng.choice([0, 1])
    if axis == 0:
        grad = np.linspace(1.0 - shadow_strength, 1.0, h, dtype=np.float32).reshape(h, 1, 1)
    else:
        grad = np.linspace(1.0 - shadow_strength, 1.0, w, dtype=np.float32).reshape(1, w, 1)
    out *= grad

    return _clamp_u8(out)


def apply_profile(img: np.ndarray, profile: str, rng: random.Random) -> np.ndarray:
    if profile == "low_light_noise":
        return _apply_low_light_noise(img, rng)
    if profile == "motion_blur":
        return _apply_motion_blur(img, rng)
    if profile == "jpeg_artifacts":
        return _apply_jpeg_artifacts(img, rng)
    if profile == "occlusion":
        return _apply_occlusion(img, rng)
    if profile == "rotation_scale":
        return _apply_rotation_scale(img, rng)
    if profile == "color_cast_shadow":
        return _apply_color_cast_shadow(img, rng)
    if profile == "mixed_extreme":
        out = _apply_rotation_scale(img, rng)
        out = _apply_low_light_noise(out, rng)
        out = _apply_occlusion(out, rng)
        out = _apply_jpeg_artifacts(out, rng)
        return out
    return img


def build_hard_samples(limit: int, seed: int, variants_per_sample: int) -> list[HardSample]:
    samples = load_samples_from_urls(limit=limit, seed=seed)
    if not samples:
        csv_path = ensure_fer_csv(os.path.join("data", "fer2013", "fer2013.csv"))
        samples = load_fer_subset(csv_path, limit=limit, seed=seed)
    if not samples:
        raise RuntimeError("No samples loaded for extreme benchmark")

    rng = random.Random(seed)
    out: list[HardSample] = []

    # Ensure profile coverage while keeping deterministic profile assignment.
    base_profiles = list(PROFILES)

    for i, s in enumerate(samples):
        profiles = []
        profiles.append(base_profiles[i % len(base_profiles)])
        while len(profiles) < max(1, variants_per_sample):
            p = rng.choice(PROFILES)
            if p not in profiles:
                profiles.append(p)

        for j, profile in enumerate(profiles):
            prng = random.Random((seed * 1000003) + i * 911 + j * 101)
            hard = apply_profile(s.image, profile, prng)
            out.append(HardSample(label=s.label, profile=profile, image=hard))

    return out


def _empty_metrics() -> dict:
    elapsed = float(time.perf_counter() - started)

    return {
        "n": 0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
    }


def _weighted_score(acc_value: float, f1_value: float) -> float:
    # Heavier weight on macro-F1 to reward class balance on difficult data.
    return 0.4 * float(acc_value) + 0.6 * float(f1_value)


def _score_to_index(score: float) -> int:
    # Compact integer index [0..1000] for dashboards and quick trend reading.
    return int(round(max(0.0, min(1.0, score)) * 1000.0))


def _percent_of_baseline(current: float, baseline: float) -> float | None:
    if float(baseline) <= 0.0:
        return None
    return (float(current) / float(baseline)) * 100.0


def run_extreme_benchmark(
    limit: int,
    seed: int,
    detector_backend: str,
    variants_per_sample: int,
    use_face_mesh: bool,
    face_mesh_weight: float,
    head_pose_strength: float,
) -> dict:
    started = time.perf_counter()
    hard_samples = build_hard_samples(limit=limit, seed=seed, variants_per_sample=variants_per_sample)

    face_mesh = None
    if _HAS_MEDIAPIPE and use_face_mesh:
        face_mesh = create_face_mesh()

    y_true: list[str] = []
    y_base: list[str] = []
    y_enh: list[str] = []

    per_profile_true: dict[str, list[str]] = {k: [] for k in PROFILES}
    per_profile_base: dict[str, list[str]] = {k: [] for k in PROFILES}
    per_profile_enh: dict[str, list[str]] = {k: [] for k in PROFILES}

    failed = 0
    for i, hs in enumerate(hard_samples, start=1):
        y_true.append(hs.label)
        per_profile_true[hs.profile].append(hs.label)

        try:
            pb = predict_baseline(hs.image, detector_backend=detector_backend)
        except Exception:
            pb = "neutral"
            failed += 1

        try:
            pe = predict_enhanced(
                hs.image,
                detector_backend=detector_backend,
                face_mesh=face_mesh,
                use_face_mesh=use_face_mesh,
                face_mesh_weight=face_mesh_weight,
                head_pose_strength=head_pose_strength,
                return_debug=False,
                hard_profile=hs.profile,
            )
            if isinstance(pe, tuple):
                pe = pe[0]
        except Exception:
            pe = "neutral"
            failed += 1

        y_base.append(pb)
        y_enh.append(pe)
        per_profile_base[hs.profile].append(pb)
        per_profile_enh[hs.profile].append(pe)

        if i % 40 == 0:
            print(f"Processed {i}/{len(hard_samples)} hard samples")

    if face_mesh is not None:
        try:
            face_mesh.close()
        except Exception:
            pass

    baseline_acc = accuracy(y_true, y_base)
    baseline_f1 = macro_f1(y_true, y_base)
    enhanced_acc = accuracy(y_true, y_enh)
    enhanced_f1 = macro_f1(y_true, y_enh)

    baseline_score = _weighted_score(baseline_acc, baseline_f1)
    enhanced_score = _weighted_score(enhanced_acc, enhanced_f1)

    per_profile = {}
    hardest_profile = None
    hardest_profile_score = 10.0
    for p in PROFILES:
        yt = per_profile_true[p]
        if not yt:
            per_profile[p] = _empty_metrics()
            continue
        yb = per_profile_base[p]
        ye = per_profile_enh[p]
        p_acc = accuracy(yt, ye)
        p_f1 = macro_f1(yt, ye)
        p_score = _weighted_score(p_acc, p_f1)
        per_profile[p] = {
            "n": len(yt),
            "baseline_accuracy": accuracy(yt, yb),
            "baseline_macro_f1": macro_f1(yt, yb),
            "enhanced_accuracy": p_acc,
            "enhanced_macro_f1": p_f1,
            "enhanced_weighted_score": p_score,
        }
        if p_score < hardest_profile_score:
            hardest_profile_score = p_score
            hardest_profile = p

    fingerprint_payload = {
        "limit": limit,
        "seed": seed,
        "variants_per_sample": variants_per_sample,
        "profiles": PROFILES,
        "detector_backend": detector_backend,
        "face_mesh": bool(use_face_mesh and _HAS_MEDIAPIPE),
        "face_mesh_weight": float(face_mesh_weight),
        "head_pose_strength": float(head_pose_strength),
    }
    fingerprint = hashlib.sha1(json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    elapsed = float(time.perf_counter() - started)

    return {
        "benchmark": "extreme_reference_v1",
        "benchmark_fingerprint": fingerprint,
        "settings": fingerprint_payload,
        "profile_specs": PROFILE_SPECS,
        "dataset": {
            "base_samples": limit,
            "hard_samples": len(hard_samples),
            "profiles": PROFILES,
            "failed_predictions": failed,
        },
        "baseline": {
            "accuracy": baseline_acc,
            "macro_f1": baseline_f1,
            "weighted_score": baseline_score,
            "index": _score_to_index(baseline_score),
            "confusion_matrix": confusion_matrix(y_true, y_base),
        },
        "enhanced": {
            "accuracy": enhanced_acc,
            "macro_f1": enhanced_f1,
            "weighted_score": enhanced_score,
            "index": _score_to_index(enhanced_score),
            "confusion_matrix": confusion_matrix(y_true, y_enh),
        },
        "delta": {
            "accuracy": enhanced_acc - baseline_acc,
            "macro_f1": enhanced_f1 - baseline_f1,
            "weighted_score": enhanced_score - baseline_score,
            "index": _score_to_index(enhanced_score) - _score_to_index(baseline_score),
        },
        "relative": {
            "enhanced_percent_of_baseline": {
                "accuracy": _percent_of_baseline(enhanced_acc, baseline_acc),
                "macro_f1": _percent_of_baseline(enhanced_f1, baseline_f1),
                "weighted_score": _percent_of_baseline(enhanced_score, baseline_score),
                "index": _percent_of_baseline(
                    float(_score_to_index(enhanced_score)),
                    float(_score_to_index(baseline_score)),
                ),
            }
        },
        "hardness": {
            "hardest_profile": hardest_profile,
            "hardest_profile_enhanced_weighted_score": hardest_profile_score if hardest_profile else 0.0,
            "per_profile": per_profile,
        },
        "runtime": {
            "total_seconds": elapsed,
            "hard_samples_per_second": float(len(hard_samples) / max(1e-9, elapsed)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Very challenging benchmark for reference tracking")
    parser.add_argument("--limit", type=int, default=84, help="Number of base samples before hard augmentation")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for deterministic benchmark generation")
    parser.add_argument("--variants-per-sample", type=int, default=3, help="How many hard variants to create per sample")
    parser.add_argument("--detector", type=str, default="opencv", help="DeepFace detector backend")
    parser.add_argument("--no-face-mesh", action="store_true", help="Disable face mesh fusion")
    parser.add_argument(
        "--face-mesh-weight",
        type=float,
        default=FACE_MESH_WEIGHT,
        help="Fusion weight for face mesh probabilities (0.0-1.0)",
    )
    parser.add_argument(
        "--head-pose-strength",
        type=float,
        default=HEAD_POSE_STRENGTH,
        help="Strength of head-pose confidence attenuation (0.0-1.0)",
    )
    parser.add_argument(
        "--no-head-pose-penalty",
        action="store_true",
        help="Disable head-pose confidence attenuation (same as --head-pose-strength 0)",
    )
    args = parser.parse_args()

    report = run_extreme_benchmark(
        limit=max(7, int(args.limit)),
        seed=int(args.seed),
        detector_backend=args.detector,
        variants_per_sample=max(1, int(args.variants_per_sample)),
        use_face_mesh=not args.no_face_mesh,
        face_mesh_weight=float(args.face_mesh_weight),
        head_pose_strength=(0.0 if args.no_head_pose_penalty else float(args.head_pose_strength)),
    )

    os.makedirs(os.path.join("benchmarks", "results"), exist_ok=True)
    out_path = os.path.join("benchmarks", "results", "extreme_reference.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Extreme Reference Benchmark ===")
    print(f"Fingerprint: {report['benchmark_fingerprint']}")
    print(
        "Baseline  acc={:.4f}  f1={:.4f}  idx={}".format(
            report["baseline"]["accuracy"],
            report["baseline"]["macro_f1"],
            report["baseline"]["index"],
        )
    )
    print(
        "Enhanced  acc={:.4f}  f1={:.4f}  idx={}".format(
            report["enhanced"]["accuracy"],
            report["enhanced"]["macro_f1"],
            report["enhanced"]["index"],
        )
    )
    print(
        "Delta     acc={:+.4f}  f1={:+.4f}  idx={:+d}".format(
            report["delta"]["accuracy"],
            report["delta"]["macro_f1"],
            int(report["delta"]["index"]),
        )
    )
    print(f"Hardest profile: {report['hardness']['hardest_profile']}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
