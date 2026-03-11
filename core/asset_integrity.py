"""Helpers for verified model asset handling.

This module enforces SHA256 verification for downloaded model files so runtime
initialization cannot silently consume tampered or drifting artifacts.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request


MEDIAPIPE_MODEL_ASSETS = {
    "face_landmarker.task": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        ),
        "sha256": "64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF",
    },
    "pose_landmarker_lite.task": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/latest/"
            "pose_landmarker_lite.task"
        ),
        "sha256": "59929E1D1EE95287735DDD833B19CF4AC46D29BC7AFDDBBF6753C459690D574A",
    },
}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _download_to_file(url: str, path: str, timeout_seconds: float = 60.0) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path) or ".") as tmp:
        tmp_path = tmp.name

    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response, open(tmp_path, "wb") as out:
            while True:
                buf = response.read(1024 * 1024)
                if not buf:
                    break
                out.write(buf)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def ensure_verified_asset(path: str, url: str, expected_sha256: str) -> str:
    """Ensures a local asset exists and matches the expected SHA256 digest."""
    expected = expected_sha256.strip().upper()

    if os.path.isfile(path):
        local = _sha256_file(path)
        if local == expected:
            return path

    _download_to_file(url=url, path=path)
    downloaded = _sha256_file(path)
    if downloaded != expected:
        raise RuntimeError(
            "Asset integrity check failed for "
            f"{path}. Expected SHA256={expected}, got SHA256={downloaded}."
        )
    return path


def ensure_mediapipe_model(model_file_name: str, model_dir: str = os.path.join("pretrained_models", "mediapipe")) -> str:
    """Ensures a verified MediaPipe model exists locally and returns its path."""
    if model_file_name not in MEDIAPIPE_MODEL_ASSETS:
        raise ValueError(f"Unknown MediaPipe model asset: {model_file_name}")

    spec = MEDIAPIPE_MODEL_ASSETS[model_file_name]
    os.makedirs(model_dir, exist_ok=True)
    target_path = os.path.join(model_dir, model_file_name)
    return ensure_verified_asset(
        path=target_path,
        url=str(spec["url"]),
        expected_sha256=str(spec["sha256"]),
    )
