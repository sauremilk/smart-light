"""ONNX-backed emotion classifier with DeepFace-compatible output shape."""

import logging
import os

import cv2
import numpy as np

from config import EMOTIONS, FACE_FINETUNE_ONNX_PATH, USE_FACE_FINETUNE_ONNX

log = logging.getLogger("emotion-light")

try:
    import onnxruntime as ort
except Exception:
    ort = None


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x)
    exp = np.exp(x)
    denom = float(np.sum(exp))
    if denom <= 1e-12:
        return np.full_like(exp, fill_value=1.0 / max(1, exp.size), dtype=np.float64)
    return exp / denom


class OnnxEmotionModel:
    """ONNX-backed emotion classifier with DeepFace-compatible output shape."""

    _labels = EMOTIONS

    def __init__(self, model_path: str):
        if ort is None:
            raise RuntimeError("onnxruntime is not available")
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def analyze(self, frame: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        logits = self._session.run(None, {self._input_name: x})[0]
        probs = _softmax(logits[0])

        scores = {label: float(probs[i] * 100.0) for i, label in enumerate(self._labels)}
        dominant = max(scores, key=scores.get)
        return {"emotion": scores, "dominant_emotion": dominant}


def init_optional_onnx_model() -> OnnxEmotionModel | None:
    """Tries to load the fine-tuned ONNX model; returns None on failure."""
    if not USE_FACE_FINETUNE_ONNX:
        return None
    if not os.path.exists(FACE_FINETUNE_ONNX_PATH):
        log.warning(
            "USE_FACE_FINETUNE_ONNX=True but model file is missing: %s. Falling back to DeepFace.",
            FACE_FINETUNE_ONNX_PATH,
        )
        return None
    try:
        model = OnnxEmotionModel(FACE_FINETUNE_ONNX_PATH)
        log.info("ONNX emotion backend active: %s", FACE_FINETUNE_ONNX_PATH)
        return model
    except Exception as exc:
        log.warning("Failed to initialize ONNX backend (%s). Falling back to DeepFace.", exc)
        return None
