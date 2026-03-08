#!/usr/bin/env python3
"""Automatic A/B accuracy benchmark for emotion classification.

What it does:
1. Downloads FER2013 CSV automatically (if missing)
2. Builds a reproducible subset from PublicTest split
3. Runs two pipelines on the same images:
   - baseline: old settings (no gray-world, 192px, threshold 0.45)
   - enhanced: current settings (gray-world, 320px, threshold 0.55, optional face-mesh fusion)
4. Prints accuracy, macro-F1 and confusion matrix summaries
5. Writes JSON result to benchmarks/results/latest_accuracy.json

Run:
    c:/Users/mickg/smart-light/.venv/Scripts/python.exe benchmarks/accuracy_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np
from deepface import DeepFace
from asset_integrity import ensure_mediapipe_model

try:
    from config import FACE_MESH_WEIGHT as _CFG_FACE_MESH_WEIGHT
except Exception:
    _CFG_FACE_MESH_WEIGHT = 0.25

try:
    from config import HEAD_POSE_CONFIDENCE_STRENGTH as _CFG_HEAD_POSE_STRENGTH
except Exception:
    _CFG_HEAD_POSE_STRENGTH = 1.0

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except Exception:
    mp = None
    _HAS_MEDIAPIPE = False

FACE_MESH_WEIGHT = min(1.0, max(0.0, float(_CFG_FACE_MESH_WEIGHT)))
HEAD_POSE_STRENGTH = min(1.0, max(0.0, float(_CFG_HEAD_POSE_STRENGTH)))


def create_face_mesh():
    """Create a Face Mesh instance across mediapipe API variants."""
    if not _HAS_MEDIAPIPE:
        return None
    try:
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    except Exception:
        try:
            mt = mp.tasks
            BaseOptions = mt.BaseOptions
            FaceLandmarker = mt.vision.FaceLandmarker
            FaceLandmarkerOptions = mt.vision.FaceLandmarkerOptions
            VisionRunningMode = mt.vision.RunningMode

            model_dir = os.path.join("pretrained_models", "mediapipe")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "face_landmarker.task")
            model_path = ensure_mediapipe_model("face_landmarker.task", model_dir=model_dir)

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return FaceLandmarker.create_from_options(options)
        except Exception:
            return None

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
FER_TO_EMOTION = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

FER_URLS = [
    "https://raw.githubusercontent.com/harshit158/ser-based-conditional-gan/dcb579be0c076a2a8a9543ccf890edbb208c80dd/conditional_gan/data/fer2013.csv",
    "https://raw.githubusercontent.com/thoughtworksarts/EmoPy/037b26afc92c5f9d36262c41902c6012796c157b/EmoPy/examples/image_data/sample.csv",
    "https://raw.githubusercontent.com/SaturdaysAI/Projects/40e5642fe66508b2d799636d36c45ac37ad450c0/Guadalajara/March2021/EmotionsDetector-main/fer2013/fer2021.csv",
    "https://raw.githubusercontent.com/SaturdaysAI/Projects/40e5642fe66508b2d799636d36c45ac37ad450c0/Guadalajara/March2021/EmotionsDetector-main/fer2013/sample_fer2013.csv",
    "https://raw.githubusercontent.com/rohitnarwani/Cummins_2023Hackathon/c892ef52b7737696af6f1164c848795869a98759/WellBeingAssesment_StarTechies/pythonlib/week3.csv",
    "https://raw.githubusercontent.com/rohitnarwani/Cummins_2023Hackathon/c892ef52b7737696af6f1164c848795869a98759/WellBeingAssesment_StarTechies/pythonlib/week4.csv",
    "https://raw.githubusercontent.com/thoo/resnet-facial-expression/4fbccaeab264fe7a06ac3f49b62a32c7618fb22e/sample.csv",
]

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)
POSE_IDS = [1, 152, 33, 263, 61, 291]


@dataclass
class Sample:
    image: np.ndarray
    label: str


def ensure_fer_csv(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                first = (f.readline() or "").lower()
            if "emotion" in first and "pixels" in first:
                return path
            os.remove(path)
        except Exception:
            try:
                os.remove(path)
            except Exception:
                pass

    last_exc = None
    for url in FER_URLS:
        try:
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(f"Could not download FER2013 CSV: {last_exc}")


def load_fer_subset(path: str, limit: int, seed: int) -> list[Sample]:
    by_class_public: dict[str, list[np.ndarray]] = {e: [] for e in EMOTIONS}
    by_class_all: dict[str, list[np.ndarray]] = {e: [] for e in EMOTIONS}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            usage = (row.get("Usage") or row.get("usage") or "").strip()
            emotion_raw = (row.get("emotion") or row.get("Emotion") or "").strip()
            if emotion_raw == "":
                continue
            y = int(emotion_raw)
            if y not in FER_TO_EMOTION:
                continue
            label = FER_TO_EMOTION[y]
            pixels_raw = (row.get("pixels") or row.get("Pixels") or "").strip()
            if not pixels_raw:
                continue
            pix = np.fromstring(pixels_raw, dtype=np.uint8, sep=" ")
            if pix.size != 48 * 48:
                continue
            img = pix.reshape(48, 48)
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            by_class_all[label].append(bgr)
            if usage == "PublicTest":
                by_class_public[label].append(bgr)

    by_class = by_class_public
    if sum(len(v) for v in by_class.values()) == 0:
        by_class = by_class_all

    rng = random.Random(seed)
    per_class = max(1, limit // len(EMOTIONS))
    picked: list[Sample] = []
    for label in EMOTIONS:
        items = by_class[label]
        rng.shuffle(items)
        for img in items[:per_class]:
            picked.append(Sample(image=img, label=label))

    rng.shuffle(picked)
    return picked[:limit]


def _samples_from_csv_text(content: str) -> list[Sample]:
    out: list[Sample] = []
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        emotion_raw = (row.get("emotion") or row.get("Emotion") or "").strip()
        pixels_raw = (row.get("pixels") or row.get("Pixels") or "").strip()
        if not emotion_raw or not pixels_raw:
            continue
        try:
            y = int(emotion_raw)
        except Exception:
            continue
        label = FER_TO_EMOTION.get(y)
        if label is None:
            continue
        pix = np.fromstring(pixels_raw, dtype=np.uint8, sep=" ")
        if pix.size != 48 * 48:
            continue
        img = pix.reshape(48, 48)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out.append(Sample(image=bgr, label=label))
    return out


def load_samples_from_urls(limit: int, seed: int) -> list[Sample]:
    all_samples: list[Sample] = []
    for url in FER_URLS:
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                text = r.read().decode("utf-8", errors="ignore")
            all_samples.extend(_samples_from_csv_text(text))
        except Exception:
            continue

    if not all_samples:
        return []

    by_class: dict[str, list[np.ndarray]] = {e: [] for e in EMOTIONS}
    for s in all_samples:
        by_class[s.label].append(s.image)

    rng = random.Random(seed)
    per_class = max(1, limit // len(EMOTIONS))
    picked: list[Sample] = []
    for label in EMOTIONS:
        items = by_class[label]
        rng.shuffle(items)
        for img in items[:per_class]:
            picked.append(Sample(image=img, label=label))

    rng.shuffle(picked)
    return picked[:limit]


def gray_world(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    for c in range(3):
        avg = float(f[:, :, c].mean())
        if avg > 1.0:
            f[:, :, c] *= 128.0 / avg
    return np.clip(f, 0, 255).astype(np.uint8)


def clahe_l(frame: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def resize_for_width(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / float(w)
    return cv2.resize(frame, (target_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def norm_scores(scores: dict[str, float]) -> dict[str, float]:
    vals = {e: max(0.0, float(scores.get(e, 0.0))) for e in EMOTIONS}
    s = sum(vals.values())
    if s <= 0:
        return {e: (1.0 / len(EMOTIONS)) for e in EMOTIONS}
    return {e: vals[e] / s for e in EMOTIONS}


def _face_mesh_landmarks(face_mesh, rgb):
    """Return first landmarks for solutions or tasks face mesh APIs."""
    if face_mesh is None:
        return None

    if hasattr(face_mesh, "process"):
        rs = face_mesh.process(rgb)
        if rs is not None and getattr(rs, "multi_face_landmarks", None):
            return rs.multi_face_landmarks[0].landmark
        return None

    if hasattr(face_mesh, "detect"):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        rs = face_mesh.detect(mp_image)
        if rs is not None and getattr(rs, "face_landmarks", None):
            return rs.face_landmarks[0]
        return None

    return None


def _dist(lm, i, j, w, h):
    dx = (lm[i].x - lm[j].x) * w
    dy = (lm[i].y - lm[j].y) * h
    return math.sqrt(dx * dx + dy * dy)


def aus_from_landmarks(lm, w, h) -> dict[str, float]:
    iod = _dist(lm, 33, 263, w, h)
    if iod < 1.0:
        iod = 1.0

    au1 = (_dist(lm, 107, 159, w, h) / iod + _dist(lm, 336, 386, w, h) / iod) / 2.0
    au2 = (_dist(lm, 70, 130, w, h) / iod + _dist(lm, 300, 359, w, h) / iod) / 2.0
    au4_raw = _dist(lm, 9, 168, w, h) / iod
    au4 = max(0.0, 0.35 - au4_raw) * 5.0

    eye_open = (_dist(lm, 159, 145, w, h) / iod + _dist(lm, 386, 374, w, h) / iod) / 2.0
    au6 = max(0.0, 0.18 - eye_open) * 10.0

    mouth_width = _dist(lm, 61, 291, w, h) / iod
    au12 = max(0.0, mouth_width - 0.55) * 4.0

    mouth_center_y = (lm[13].y + lm[14].y) / 2.0
    corner_avg_y = (lm[61].y + lm[291].y) / 2.0
    au15 = max(0.0, (corner_avg_y - mouth_center_y) * h / iod) * 3.0

    au20 = max(0.0, mouth_width - 0.6) * 3.0
    au25 = (_dist(lm, 13, 14, w, h) / iod) * 5.0
    au26 = max(0.0, (_dist(lm, 152, 1, w, h) / iod) - 0.65) * 4.0
    au9 = max(0.0, (_dist(lm, 97, 326, w, h) / iod) - 0.28) * 5.0

    return {
        "AU1": min(au1, 2.0),
        "AU2": min(au2, 2.0),
        "AU4": min(au4, 2.0),
        "AU6": min(au6, 2.0),
        "AU9": min(au9, 2.0),
        "AU12": min(au12, 2.0),
        "AU15": min(au15, 2.0),
        "AU20": min(au20, 2.0),
        "AU25": min(au25, 2.0),
        "AU26": min(au26, 2.0),
    }


def aus_to_emotions(aus: dict[str, float]) -> dict[str, float]:
    s = {}
    s["happy"] = min(100.0, (aus["AU6"] + aus["AU12"]) * 40.0)
    s["sad"] = min(100.0, (aus["AU1"] * 0.3 + aus["AU4"] * 0.3 + aus["AU15"] * 0.4) * 60.0)
    s["angry"] = min(100.0, (aus["AU4"] * 0.5 + aus["AU9"] * 0.3) * 50.0 * max(0.2, 1.0 - aus["AU1"]))
    s["fear"] = min(100.0, (aus["AU1"] * 0.3 + aus["AU2"] * 0.2 + aus["AU4"] * 0.2 + aus["AU20"] * 0.3) * 55.0)
    s["surprise"] = min(100.0, (aus["AU1"] * 0.2 + aus["AU2"] * 0.2 + aus["AU25"] * 0.3 + aus["AU26"] * 0.3) * 50.0)
    s["disgust"] = min(100.0, (aus["AU9"] * 0.5 + aus["AU15"] * 0.5) * 55.0)
    total_au = sum(aus.values())
    s["neutral"] = max(0.0, 100.0 - total_au * 15.0)
    return s


def estimate_head_pose(lm, w, h) -> dict[str, float]:
    pts2d = np.array([(lm[i].x * w, lm[i].y * h) for i in POSE_IDS], dtype=np.float64)
    focal = w
    center = (w / 2.0, h / 2.0)
    camera = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, pts2d, camera, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
        roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
        roll = 0.0
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def head_pose_factor(head_pose: dict[str, float]) -> float:
    yaw_abs = abs(head_pose["yaw"])
    pitch_abs = abs(head_pose["pitch"])
    factor = 1.0

    if yaw_abs > 30.0:
        factor *= max(0.5, 1.0 - ((yaw_abs - 30.0) / 30.0))
    if pitch_abs > 25.0:
        factor *= max(0.5, 1.0 - ((pitch_abs - 25.0) / 25.0))

    return max(0.5, factor)


def deepface_scores(frame: np.ndarray, detector_backend: str) -> tuple[dict[str, float], float]:
    result = DeepFace.analyze(
        img_path=frame,
        actions=["emotion"],
        detector_backend=detector_backend,
        enforce_detection=False,
        silent=True,
    )
    face = result[0] if isinstance(result, list) else result
    scores = face.get("emotion", {})
    dominant = face.get("dominant_emotion", "neutral")
    conf = float(scores.get(dominant, 0.0)) / 100.0
    return ({e: float(scores.get(e, 0.0)) for e in EMOTIONS}, conf)


def _gamma_correct(frame: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        return frame
    f = frame.astype(np.float32) / 255.0
    out = np.power(np.clip(f, 0.0, 1.0), gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _unsharp(frame: np.ndarray, sigma: float = 1.0, amount: float = 1.1) -> np.ndarray:
    blur = cv2.GaussianBlur(frame, (0, 0), sigma)
    out = cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _denoise_bilateral(frame: np.ndarray) -> np.ndarray:
    # Bilateral filtering keeps edges while reducing sensor/compression noise.
    return cv2.bilateralFilter(frame, d=5, sigmaColor=45, sigmaSpace=45)


def _top_margin(probs: dict[str, float]) -> float:
    vals = sorted((float(probs.get(e, 0.0)) for e in EMOTIONS), reverse=True)
    if len(vals) < 2:
        return 0.0
    return max(0.0, vals[0] - vals[1])


def _robust_deepface_scores(
    frame: np.ndarray,
    detector_backend: str,
    profile_hint: str | None = None,
) -> tuple[dict[str, float], float, str, float, str]:
    """Runs DeepFace over robust preprocess variants and selects the best candidate.

    Returns: (scores, confidence, selected_variant, selected_quality, selected_backend)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_luma = float(gray.mean())

    candidates = [
        ("base", frame),
        ("lifted", _gamma_correct(frame, gamma=0.72)),
        ("denoise", _denoise_bilateral(frame)),
        ("unsharp", _unsharp(frame, sigma=1.0, amount=1.1)),
    ]

    # Extra variants only for genuinely dark frames to reduce overfitting/regressions.
    if mean_luma < 80.0:
        candidates.extend(
            [
                ("lifted_dark", _gamma_correct(frame, gamma=0.58)),
                ("denoise_lifted_dark", _denoise_bilateral(_gamma_correct(frame, gamma=0.62))),
            ]
        )

    # Optional hint from extreme benchmark to probe profile-specific robustness.
    if profile_hint == "color_cast_shadow":
        cc = gray_world(frame)
        candidates.extend(
            [
                ("cc_grayworld", cc),
                ("cc_grayworld_clahe", clahe_l(cc, 2.0)),
            ]
        )
    elif profile_hint == "mixed_extreme":
        candidates.extend(
            [
                ("mixed_denoise_unsharp", _unsharp(_denoise_bilateral(frame), sigma=1.0, amount=0.9)),
            ]
        )

    best_scores: dict[str, float] | None = None
    best_image: np.ndarray | None = None
    best_conf = -1.0
    best_quality = -1.0
    best_name = "base"
    best_backend = detector_backend

    backends = [detector_backend]
    # Optional robust fallback for hard perturbations.
    if detector_backend == "opencv":
        backends.append("retinaface")

    for backend in backends:
        # Skip expensive fallback when primary already looks reliable.
        if backend != detector_backend and best_quality >= 0.52:
            continue

        for idx, (name, img) in enumerate(candidates):
            try:
                scores, conf = deepface_scores(img, backend)
            except Exception:
                continue
            probs = norm_scores(scores)
            margin = _top_margin(probs)
            quality = 0.65 * float(conf) + 0.35 * float(margin)

            if quality > best_quality:
                best_scores = scores
                best_image = img
                best_conf = float(conf)
                best_quality = float(quality)
                best_name = name
                best_backend = backend

            # Early stop for strong/clear prediction to limit runtime overhead.
            if idx == 0 and conf >= 0.70 and margin >= 0.25:
                break

    if best_scores is None:
        return ({e: 0.0 for e in EMOTIONS}, 0.0, "base", 0.0, detector_backend)

    # Low-quality TTA: blend with horizontally flipped inference for extra robustness.
    if best_image is not None and best_quality < 0.55:
        try:
            flipped = cv2.flip(best_image, 1)
            flip_scores, flip_conf = deepface_scores(flipped, best_backend)
            probs_a = norm_scores(best_scores)
            probs_b = norm_scores(flip_scores)
            merged_probs = {
                e: 0.7 * float(probs_a.get(e, 0.0)) + 0.3 * float(probs_b.get(e, 0.0))
                for e in EMOTIONS
            }
            s = sum(merged_probs.values())
            if s > 0:
                merged_probs = {e: merged_probs[e] / s for e in EMOTIONS}
                best_scores = {e: 100.0 * merged_probs[e] for e in EMOTIONS}
                best_conf = max(float(best_conf), float(flip_conf), float(max(merged_probs.values())))
        except Exception:
            pass

    return best_scores, max(0.0, best_conf), best_name, max(0.0, best_quality), best_backend


def predict_baseline(img: np.ndarray, detector_backend: str = "opencv") -> str:
    proc = resize_for_width(img, 192)
    proc = clahe_l(proc, 2.0)
    scores, conf = deepface_scores(proc, detector_backend)
    if conf < 0.45:
        return "neutral"
    probs = norm_scores(scores)
    return max(EMOTIONS, key=lambda e: probs[e])


def predict_enhanced(
    img: np.ndarray,
    detector_backend: str = "opencv",
    face_mesh=None,
    use_face_mesh: bool = True,
    face_mesh_weight: float = FACE_MESH_WEIGHT,
    head_pose_strength: float = HEAD_POSE_STRENGTH,
    return_debug: bool = False,
    hard_profile: str | None = None,
) -> str | tuple[str, dict[str, object]]:
    proc = resize_for_width(img, 320)
    proc = gray_world(proc)
    proc = clahe_l(proc, 2.0)

    (
        df_scores,
        conf,
        preprocess_selected,
        preprocess_quality,
        detector_selected,
    ) = _robust_deepface_scores(proc, detector_backend, profile_hint=hard_profile)

    fm_probs = None
    pose_factor = 1.0
    conf_before_pose = conf
    if use_face_mesh and face_mesh is not None:
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        lm = _face_mesh_landmarks(face_mesh, rgb)
        if lm is not None:
            aus = aus_from_landmarks(lm, proc.shape[1], proc.shape[0])
            fm_probs = norm_scores(aus_to_emotions(aus))

            pose = estimate_head_pose(lm, proc.shape[1], proc.shape[0])
            raw_pose_factor = head_pose_factor(pose)
            strength = min(1.0, max(0.0, float(head_pose_strength)))
            pose_factor = (1.0 - strength) + (strength * raw_pose_factor)
            conf *= pose_factor

    probs = norm_scores(df_scores)
    conf_before_fm_boost = conf
    w = min(1.0, max(0.0, float(face_mesh_weight)))
    if fm_probs is not None:
        probs = {e: (1.0 - w) * probs[e] + w * fm_probs[e] for e in EMOTIONS}
        s = sum(probs.values())
        probs = {e: probs[e] / s for e in EMOTIONS}
        # Allow face-mesh evidence to lift low-DeepFace-confidence cases.
        conf = max(conf, max(fm_probs.values()))

    mean_luma = float(cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY).mean())
    darkness = max(0.0, min(1.0, (95.0 - mean_luma) / 95.0))
    decision_score = 0.65 * float(conf) + 0.35 * float(_top_margin(probs))
    # In low-light scenarios, avoid overly aggressive neutral-gating.
    gate_threshold = 0.42 - (0.14 * darkness)
    top_label = max(EMOTIONS, key=lambda e: probs[e])
    top_prob = float(probs.get(top_label, 0.0))
    gated_neutral = decision_score < gate_threshold

    # Rescue rule for very dark frames: if there is a stable non-neutral top class,
    # avoid collapsing to neutral too aggressively.
    dark_rescue = (
        darkness >= 0.55
        and top_label != "neutral"
        and top_prob >= 0.30
        and float(conf) >= 0.15
    )

    label = "neutral" if (gated_neutral and not dark_rescue) else top_label

    if return_debug:
        df_top = max(df_scores, key=df_scores.get) if df_scores else "neutral"
        fm_top = max(fm_probs, key=fm_probs.get) if fm_probs else None
        debug = {
            "deepface_top": df_top,
            "deepface_top_prob": probs.get(df_top, 0.0),
            "deepface_conf_raw": float(conf_before_pose),
            "deepface_preprocess_selected": preprocess_selected,
            "deepface_preprocess_quality": float(preprocess_quality),
            "deepface_detector_selected": detector_selected,
            "pose_factor": float(pose_factor),
            "conf_after_pose": float(conf_before_fm_boost),
            "conf_after_fm": float(conf),
            "decision_score": float(decision_score),
            "gate_threshold": float(gate_threshold),
            "mean_luma": float(mean_luma),
            "darkness": float(darkness),
            "dark_rescue": bool(dark_rescue),
            "top_label": top_label,
            "top_prob": top_prob,
            "gated_neutral": bool(gated_neutral),
            "face_mesh_used": bool(fm_probs is not None),
            "face_mesh_weight": float(w),
            "face_mesh_top": fm_top,
            "face_mesh_top_prob": (float(max(fm_probs.values())) if fm_probs else 0.0),
            "final_top": max(probs, key=probs.get),
            "final_top_prob": float(max(probs.values())),
            "final_label": label,
        }
        return label, debug

    return label


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> list[list[int]]:
    idx = {e: i for i, e in enumerate(EMOTIONS)}
    m = [[0 for _ in EMOTIONS] for _ in EMOTIONS]
    for t, p in zip(y_true, y_pred):
        m[idx[t]][idx[p]] += 1
    return m


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    ok = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return ok / len(y_true)


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    m = confusion_matrix(y_true, y_pred)
    scores = []
    for i in range(len(EMOTIONS)):
        tp = m[i][i]
        fp = sum(m[r][i] for r in range(len(EMOTIONS)) if r != i)
        fn = sum(m[i][c] for c in range(len(EMOTIONS)) if c != i)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def run(
    limit: int,
    seed: int,
    detector_backend: str,
    use_face_mesh: bool,
    face_mesh_weight: float = FACE_MESH_WEIGHT,
    head_pose_strength: float = HEAD_POSE_STRENGTH,
    collect_diagnostics: bool = False,
) -> dict:
    # Prefer multi-source remote loading for fully automatic benchmarking.
    samples = load_samples_from_urls(limit=limit, seed=seed)

    # Fallback to local cached CSV if remote loading is unavailable.
    if not samples:
        csv_path = ensure_fer_csv(os.path.join("data", "fer2013", "fer2013.csv"))
        samples = load_fer_subset(csv_path, limit=limit, seed=seed)

    if not samples:
        raise RuntimeError("No samples loaded from FER2013")

    face_mesh = None
    if _HAS_MEDIAPIPE and use_face_mesh:
        face_mesh = create_face_mesh()

    fm_frames_total = 0
    fm_frames_detected = 0

    y_true = [s.label for s in samples]
    y_pred_base = []
    y_pred_new = []
    diagnostics = []

    skipped = 0
    for i, s in enumerate(samples, start=1):
        try:
            fm_frames_total += 1 if (use_face_mesh and face_mesh is not None) else 0
            y_pred_base.append(predict_baseline(s.image, detector_backend=detector_backend))
            # Count if Face Mesh produced landmarks for this sample.
            if use_face_mesh and face_mesh is not None:
                proc = resize_for_width(s.image, 320)
                proc = gray_world(proc)
                proc = clahe_l(proc, 2.0)
                rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
                lm = _face_mesh_landmarks(face_mesh, rgb)
                if lm is not None:
                    fm_frames_detected += 1

            pred_new = predict_enhanced(
                s.image,
                detector_backend=detector_backend,
                face_mesh=face_mesh,
                use_face_mesh=use_face_mesh,
                face_mesh_weight=face_mesh_weight,
                head_pose_strength=head_pose_strength,
                return_debug=collect_diagnostics,
            )
            if collect_diagnostics:
                label_new, dbg = pred_new
                y_pred_new.append(label_new)
                diagnostics.append(
                    {
                        "index": i,
                        "true_label": s.label,
                        "baseline_label": y_pred_base[-1],
                        "enhanced_label": label_new,
                        **dbg,
                    }
                )
            else:
                y_pred_new.append(pred_new)
        except Exception:
            skipped += 1
            y_pred_base.append("neutral")
            y_pred_new.append("neutral")

        if i % 20 == 0:
            print(f"Processed {i}/{len(samples)}")

    if face_mesh is not None:
        try:
            face_mesh.close()
        except Exception:
            pass

    base_acc = accuracy(y_true, y_pred_base)
    new_acc = accuracy(y_true, y_pred_new)
    base_f1 = macro_f1(y_true, y_pred_base)
    new_f1 = macro_f1(y_true, y_pred_new)

    return {
        "dataset": "Multi-source FER-style labeled CSV benchmark",
        "samples": len(samples),
        "skipped_or_failed": skipped,
        "detector_backend": detector_backend,
        "face_mesh_enabled": bool(_HAS_MEDIAPIPE and use_face_mesh),
        "face_mesh_weight": float(face_mesh_weight),
        "head_pose_strength": float(head_pose_strength),
        "face_mesh_runtime": {
            "requested": bool(use_face_mesh),
            "initialized": bool(face_mesh is not None),
            "frames_checked": fm_frames_total,
            "frames_with_landmarks": fm_frames_detected,
            "landmark_ratio": (fm_frames_detected / fm_frames_total) if fm_frames_total > 0 else 0.0,
        },
        "baseline": {
            "accuracy": base_acc,
            "macro_f1": base_f1,
            "confusion_matrix": confusion_matrix(y_true, y_pred_base),
        },
        "enhanced": {
            "accuracy": new_acc,
            "macro_f1": new_f1,
            "confusion_matrix": confusion_matrix(y_true, y_pred_new),
        },
        "delta": {
            "accuracy": new_acc - base_acc,
            "macro_f1": new_f1 - base_f1,
        },
        "diagnostics": diagnostics if collect_diagnostics else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Automatic A/B accuracy benchmark")
    parser.add_argument("--limit", type=int, default=140, help="Number of evaluation samples")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--detector", default="opencv", help="DeepFace detector backend")
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
    parser.add_argument(
        "--collect-diagnostics",
        action="store_true",
        help="Include per-sample decision diagnostics in latest_accuracy.json",
    )
    args = parser.parse_args()

    report = run(
        limit=args.limit,
        seed=args.seed,
        detector_backend=args.detector,
        use_face_mesh=not args.no_face_mesh,
        face_mesh_weight=args.face_mesh_weight,
        head_pose_strength=(0.0 if args.no_head_pose_penalty else args.head_pose_strength),
        collect_diagnostics=args.collect_diagnostics,
    )

    os.makedirs(os.path.join("benchmarks", "results"), exist_ok=True)
    out_path = os.path.join("benchmarks", "results", "latest_accuracy.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Accuracy Benchmark ===")
    print(f"Samples: {report['samples']} (failed={report['skipped_or_failed']})")
    print(f"Baseline  acc={report['baseline']['accuracy']:.4f}  f1={report['baseline']['macro_f1']:.4f}")
    print(f"Enhanced  acc={report['enhanced']['accuracy']:.4f}  f1={report['enhanced']['macro_f1']:.4f}")
    print(f"Delta     acc={report['delta']['accuracy']:+.4f}  f1={report['delta']['macro_f1']:+.4f}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
