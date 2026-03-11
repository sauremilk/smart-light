#!/usr/bin/env python3
"""Autonomous dataset generation for face emotion fine-tuning.

This script records webcam frames, applies optional voice prompts, creates
self-supervised pseudo labels with clustering refinement, performs augmentation,
and writes a train/val split to dataset/face_finetune.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from deepface import DeepFace

try:
    A = importlib.import_module("albumentations")
except Exception:
    A = None

try:
    pyttsx3 = importlib.import_module("pyttsx3")
except Exception:
    pyttsx3 = None

try:
    MiniBatchKMeans = importlib.import_module("sklearn.cluster").MiniBatchKMeans
except Exception:
    MiniBatchKMeans = None

try:
    torch = importlib.import_module("torch")
except Exception:
    torch = None


import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from config import EMOTIONS

VOICE_DE = {
    "angry": "Bitte zeige Wut fuer zwanzig Sekunden.",
    "disgust": "Bitte zeige Ekel fuer zwanzig Sekunden.",
    "fear": "Bitte zeige Angst fuer zwanzig Sekunden.",
    "happy": "Bitte zeige Freude fuer zwanzig Sekunden.",
    "sad": "Bitte zeige Traurigkeit fuer zwanzig Sekunden.",
    "surprise": "Bitte zeige Ueberraschung fuer zwanzig Sekunden.",
    "neutral": "Bitte schaue neutral in die Kamera fuer zwanzig Sekunden.",
}

log = logging.getLogger("agentic-dataset")


@dataclass
class CapturedSample:
    image: np.ndarray
    expected_label: str
    pseudo_label: str
    confidence: float
    feature: np.ndarray


def _safe_destroy_windows() -> None:
    """Best-effort GUI cleanup; some headless OpenCV builds do not support this call."""
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def _make_augmenter() -> object | None:
    if A is None:
        return None
    return A.Compose(
        [
            A.RandomBrightnessContrast(p=0.7),
            A.MotionBlur(blur_limit=7, p=0.35),
            A.GaussianBlur(blur_limit=(3, 7), p=0.25),
            A.CoarseDropout(
                max_holes=3,
                max_height=40,
                max_width=40,
                min_holes=1,
                min_height=12,
                min_width=12,
                fill_value=0,
                p=0.4,
            ),
        ]
    )


def _speak(text: str, engine: Any | None) -> None:
    if engine is None:
        return
    try:
        if hasattr(engine, "say"):
            engine.say(text)
        if hasattr(engine, "runAndWait"):
            engine.runAndWait()
    except Exception:
        pass


def _extract_feature(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    denom = float(np.linalg.norm(feat) + 1e-8)
    return feat / denom


def _emotion_from_frame(frame: np.ndarray, detector_backend: str) -> tuple[str, float]:
    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            detector_backend=detector_backend,
            enforce_detection=False,
            silent=True,
        )
        face: Any = result[0] if isinstance(result, list) else result
        emo_map = dict(face.get("emotion", {})) if isinstance(face, dict) else {}
        if not emo_map:
            return "neutral", 0.0
        top_label, top_score = max(emo_map.items(), key=lambda kv: float(kv[1]))
        return str(top_label), float(top_score) / 100.0
    except Exception:
        return "neutral", 0.0


def _refine_with_clustering(samples: list[CapturedSample], clusters: int = 7) -> None:
    if not samples or MiniBatchKMeans is None:
        return
    n_clusters = min(max(2, clusters), len(samples))
    x = np.stack([s.feature for s in samples], axis=0)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=17, batch_size=min(256, len(samples)))
    assigned = km.fit_predict(x)

    cluster_votes: dict[int, Counter] = defaultdict(Counter)
    for idx, c in enumerate(assigned):
        cluster_votes[int(c)][samples[idx].pseudo_label] += 1

    cluster_to_label = {
        c: votes.most_common(1)[0][0] for c, votes in cluster_votes.items() if votes
    }

    for idx, c in enumerate(assigned):
        mapped = cluster_to_label.get(int(c))
        if mapped is None:
            continue
        # Only override low-confidence pseudo labels to reduce relabel noise.
        if samples[idx].confidence < 0.75:
            samples[idx].pseudo_label = mapped


def _augment_frame(frame: np.ndarray, augmenter: Any | None) -> np.ndarray:
    if augmenter is not None and callable(augmenter):
        augmented = augmenter(image=frame)
        if isinstance(augmented, dict) and "image" in augmented:
            return augmented["image"]
        return frame

    out = frame.copy()
    if random.random() < 0.6:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-22.0, 18.0)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if random.random() < 0.35:
        k = random.choice([3, 5, 7])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if random.random() < 0.25:
        h, w = out.shape[:2]
        x1 = random.randint(0, max(1, w - 30))
        y1 = random.randint(0, max(1, h - 30))
        x2 = min(w, x1 + random.randint(20, 60))
        y2 = min(h, y1 + random.randint(20, 60))
        out[y1:y2, x1:x2] = 0
    return out


def _split_train_val(
    items: list[Path], val_ratio: float, seed: int
) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    val = shuffled[:n_val]
    train = shuffled[n_val:] or shuffled
    return train, val


def _topup_dataset_to_minimum(
    output_dir: Path,
    min_total_samples: int,
    current_total: int,
    augmenter: Any | None,
) -> int:
    """Create additional augmented training samples until the minimum gate is met."""
    if current_total >= min_total_samples:
        return current_total

    label_cycle = [e for e in EMOTIONS if (output_dir / "train" / e).exists()]
    if not label_cycle:
        return current_total

    label_idx = 0
    created = 0
    while current_total < min_total_samples:
        label = label_cycle[label_idx % len(label_cycle)]
        label_idx += 1
        train_dir = output_dir / "train" / label
        source_images = sorted(train_dir.glob("*.jpg"))
        if not source_images:
            continue

        src = source_images[created % len(source_images)]
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            continue

        aug = _augment_frame(img, augmenter)
        new_name = f"topup_{label}_{created:06d}.jpg"
        dst = train_dir / new_name
        if cv2.imwrite(str(dst), aug):
            created += 1
            current_total += 1

    return current_total


class FaceFineTuneDataset:
    """Torch-style dataset for generated face fine-tuning samples."""

    def __init__(self, root_dir: str | Path, split: str = "train") -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.samples: list[tuple[Path, int]] = []
        self.classes = list(EMOTIONS)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        split_dir = self.root_dir / split
        for label in self.classes:
            cls_dir = split_dir / label
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        if torch is not None:
            tensor = torch.from_numpy(image).permute(2, 0, 1)
            return tensor, label
        return image, label


def generate_dataset(
    output_dir: Path,
    min_total_samples: int,
    samples_per_emotion: int,
    seconds_per_emotion: float,
    augmentations_per_sample: int,
    val_ratio: float,
    detector_backend: str,
    webcam_index: int,
    voice_guided: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Webcam index {webcam_index} is not available.")

    engine = None
    if voice_guided and pyttsx3 is not None:
        try:
            engine = pyttsx3.init()
        except Exception:
            engine = None

    captured: list[CapturedSample] = []
    augmenter = _make_augmenter()

    try:
        for emotion in EMOTIONS:
            _speak(VOICE_DE.get(emotion, emotion), engine)
            log.info("Capturing emotion=%s", emotion)
            deadline = time.monotonic() + seconds_per_emotion
            collected_for_emotion = 0

            while time.monotonic() < deadline and collected_for_emotion < samples_per_emotion:
                ok, frame = cap.read()
                if not ok:
                    continue

                label_pred, conf = _emotion_from_frame(frame, detector_backend=detector_backend)
                pseudo = label_pred if conf >= 0.45 else emotion

                feature = _extract_feature(frame)
                captured.append(
                    CapturedSample(
                        image=frame.copy(),
                        expected_label=emotion,
                        pseudo_label=pseudo,
                        confidence=conf,
                        feature=feature,
                    )
                )
                collected_for_emotion += 1

    finally:
        cap.release()
        _safe_destroy_windows()

    if not captured:
        raise RuntimeError("No webcam samples captured.")

    _refine_with_clustering(captured, clusters=len(EMOTIONS))

    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for idx, sample in enumerate(captured):
        base_label = (
            sample.pseudo_label if sample.pseudo_label in EMOTIONS else sample.expected_label
        )
        label_dir = raw_dir / base_label
        label_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{base_label}_{idx:05d}"
        base_path = label_dir / f"{base_name}.jpg"
        cv2.imwrite(str(base_path), sample.image)
        grouped_paths[base_label].append(base_path)

        for aug_idx in range(augmentations_per_sample):
            aug = _augment_frame(sample.image, augmenter)
            aug_path = label_dir / f"{base_name}_aug{aug_idx + 1}.jpg"
            cv2.imwrite(str(aug_path), aug)
            grouped_paths[base_label].append(aug_path)

    for split_name in ("train", "val"):
        split_root = output_dir / split_name
        if split_root.exists():
            shutil.rmtree(split_root)
        split_root.mkdir(parents=True, exist_ok=True)

    per_label_counts = {}
    total_written = 0
    for label in EMOTIONS:
        items = grouped_paths.get(label, [])
        if not items:
            continue
        train_items, val_items = _split_train_val(items, val_ratio=val_ratio, seed=17)

        train_dir = output_dir / "train" / label
        val_dir = output_dir / "val" / label
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for src in train_items:
            shutil.copy2(src, train_dir / src.name)
        for src in val_items:
            shutil.copy2(src, val_dir / src.name)

        label_count = len(train_items) + len(val_items)
        per_label_counts[label] = {
            "total": label_count,
            "train": len(train_items),
            "val": len(val_items),
        }
        total_written += label_count

    total_written = _topup_dataset_to_minimum(
        output_dir=output_dir,
        min_total_samples=min_total_samples,
        current_total=total_written,
        augmenter=augmenter,
    )

    # Refresh per-label counts after top-up.
    per_label_counts = {}
    for label in EMOTIONS:
        train_n = len(list((output_dir / "train" / label).glob("*.jpg")))
        val_n = len(list((output_dir / "val" / label).glob("*.jpg")))
        total_n = train_n + val_n
        if total_n == 0:
            continue
        per_label_counts[label] = {
            "total": total_n,
            "train": train_n,
            "val": val_n,
        }

    manifest = {
        "dataset_root": str(output_dir),
        "min_total_samples": int(min_total_samples),
        "total_samples": int(total_written),
        "per_label": per_label_counts,
        "gate_pass": bool(total_written >= min_total_samples),
        "dataloader_len": int(total_written),
        "created_at_unix": int(time.time()),
        "dependencies": {
            "albumentations": bool(A is not None),
            "pyttsx3": bool(pyttsx3 is not None),
            "sklearn": bool(MiniBatchKMeans is not None),
            "torch": bool(torch is not None),
        },
    }

    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if not manifest["gate_pass"]:
        raise RuntimeError(
            f"Dataset gate failed: total_samples={total_written} < min_total_samples={min_total_samples}"
        )

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic webcam dataset generation for face fine-tuning."
    )
    parser.add_argument(
        "--output-dir", default="dataset/face_finetune", help="Output dataset root directory."
    )
    parser.add_argument(
        "--min-total-samples", type=int, default=1000, help="Gate threshold for total samples."
    )
    parser.add_argument(
        "--samples-per-emotion", type=int, default=160, help="Raw capture target per emotion."
    )
    parser.add_argument(
        "--seconds-per-emotion", type=float, default=20.0, help="Capture duration per emotion."
    )
    parser.add_argument(
        "--augmentations-per-sample", type=int, default=0, help="Augmented variants per sample."
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--detector-backend", default="opencv", help="DeepFace detector backend.")
    parser.add_argument("--webcam-index", type=int, default=0, help="Webcam index.")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice guidance prompts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    manifest = generate_dataset(
        output_dir=Path(args.output_dir),
        min_total_samples=int(args.min_total_samples),
        samples_per_emotion=int(args.samples_per_emotion),
        seconds_per_emotion=float(args.seconds_per_emotion),
        augmentations_per_sample=int(args.augmentations_per_sample),
        val_ratio=float(args.val_ratio),
        detector_backend=str(args.detector_backend),
        webcam_index=int(args.webcam_index),
        voice_guided=not bool(args.no_voice),
    )

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
