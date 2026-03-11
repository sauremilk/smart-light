"""Async emotion analysis thread with EMA smoothing and burst detection."""

import logging
import queue
import threading
import time

import cv2
import numpy as np
from deepface import DeepFace

from config import (
    ANALYSIS_FRAME_SIZE,
    BURST_CONFIDENCE_DELTA,
    BURST_FRAMES,
    DETECTOR_BACKEND,
    EMA_ALPHA,
    EMA_MIN_WEIGHT,
    EMOTION_MAP,
    FALLBACK_AFTER_SECONDS,
    FALLBACK_DECAY,
    LOW_CONFIDENCE_ALPHA_SCALE,
    MIN_CONFIDENCE,
    SOFT_MIN_CONFIDENCE,
    UNCERTAINTY_ENTROPY_WEIGHT,
    UNCERTAINTY_MARGIN_WEIGHT,
    VALENCE_AROUSAL_MAP,
)
from core.ema_utils import normalize_vector_inplace, update_ema_vector_inplace
from core.error_taxonomy import DEEPFACE_ANALYZE_FAILED
from core.onnx_model import init_optional_onnx_model
from core.preprocessing import normalize_lighting
from core.telemetry import RuntimeErrorTelemetry

log = logging.getLogger("emotion-light")

_ONNX_MODEL = init_optional_onnx_model()


def analyze_emotion_frame(frame: np.ndarray) -> dict | None:
    """Returns DeepFace-like result dict with keys: emotion, dominant_emotion."""
    if _ONNX_MODEL is not None:
        return _ONNX_MODEL.analyze(frame)

    results = DeepFace.analyze(
        img_path=frame,
        actions=["emotion"],
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
        silent=True,
    )
    if isinstance(results, list):
        if not results:
            return None
        face = results[0]
        return face if isinstance(face, dict) else None
    return results if isinstance(results, dict) else None


class EmotionAnalyzer:
    """Asynchrone Emotionserkennung mit EMA-Smoothing, Confidence-Gewichtung,
    CLAHE-Normalisierung, Trend-Analyse und Mikro-Expressions-Burst."""

    _EMOTIONS = list(EMOTION_MAP.keys())  # 7 Emotionen

    def __init__(
        self,
        calibration: dict | None = None,
        *,
        telemetry: RuntimeErrorTelemetry | None = None,
    ):
        self._telemetry = telemetry
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._running = False
        self._last_face_time = time.time()
        self._calibration = calibration or {}
        # EMA-Vektor: gleichverteilt starten
        n = len(self._EMOTIONS)
        self._ema = {e: 1.0 / n for e in self._EMOTIONS}
        self._ema_prev = self._ema.copy()  # Vorheriger EMA fuer Trend
        self._avg_confidence = 0.5  # Laufender Durchschnitt der Confidence (fuer Burst)
        self._burst_remaining = 0  # Verbleibende Burst-Frames
        self._result = {
            "emotion": "neutral",
            "confidence": 0.0,
            "quality": 0.0,
            "ema_vector": self._ema.copy(),
            "valence": 0.0,
            "arousal": 0.0,
            "trend_valence": 0.0,
        }

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, frame):
        """Uebergibt einen Frame (non-blocking, verwirft alten Frame)."""
        try:
            while not self._q.empty():
                self._q.get_nowait()
            self._q.put_nowait(frame)
        except queue.Full:
            pass

    @property
    def burst_active(self) -> bool:
        return self._burst_remaining > 0

    @property
    def last_face_time(self) -> float:
        """Zeitstempel des letzten erkannten Gesichts."""
        return self._last_face_time

    def _apply_calibration(self, scores: dict) -> dict:
        """Wendet Kalibrierungs-Offsets auf rohe Scores an."""
        if not self._calibration:
            return scores
        calibrated = {}
        for e in self._EMOTIONS:
            raw = scores.get(e, 0.0)
            offset = self._calibration.get(e, 0.0)
            calibrated[e] = max(0.0, min(100.0, raw + offset))
        return calibrated

    def _update_ema(self, scores: dict, confidence: float):
        """EMA-Update mit Confidence-Gewichtung: niedrige Konfidenz = weniger Einfluss."""
        self._ema_prev = self._ema.copy()
        alpha = EMA_ALPHA * confidence  # Confidence-gewichtetes Alpha
        update_ema_vector_inplace(self._ema, scores, alpha, self._EMOTIONS)

    def _compute_valence_arousal(self) -> tuple:
        """Berechnet gewichteten Valence/Arousal aus EMA-Vektor."""
        filtered = {e: w for e, w in self._ema.items() if w >= EMA_MIN_WEIGHT}
        if not filtered:
            return 0.0, 0.0
        total = sum(filtered.values())
        valence = sum(
            (w / total) * VALENCE_AROUSAL_MAP[e]["valence"] for e, w in filtered.items()
        )
        arousal = sum(
            (w / total) * VALENCE_AROUSAL_MAP[e]["arousal"] for e, w in filtered.items()
        )
        return valence, arousal

    def _compute_trend(self) -> float:
        """Berechnet Valence-Trend (Differenz zum vorherigen EMA)."""
        v_now = sum(
            self._ema.get(e, 0) * VALENCE_AROUSAL_MAP[e]["valence"]
            for e in self._EMOTIONS
        )
        v_prev = sum(
            self._ema_prev.get(e, 0) * VALENCE_AROUSAL_MAP[e]["valence"]
            for e in self._EMOTIONS
        )
        return v_now - v_prev

    def _dominant_emotion(self) -> tuple:
        """Gibt (Emotionsname, EMA-Gewicht) der staerksten Emotion zurueck."""
        best = max(self._EMOTIONS, key=lambda e: self._ema[e])
        return best, self._ema[best]

    def _check_burst(self, confidence: float):
        """Prueft ob ein Mikro-Expressions-Burst ausgeloest werden soll."""
        delta = abs(confidence - self._avg_confidence)
        # Laufenden Durchschnitt aktualisieren
        self._avg_confidence = 0.95 * self._avg_confidence + 0.05 * confidence
        if delta >= BURST_CONFIDENCE_DELTA and self._burst_remaining <= 0:
            self._burst_remaining = BURST_FRAMES
            log.debug("Burst ausgeloest: delta=%.2f", delta)

    def _compute_prediction_quality(self, scores: dict) -> float:
        """Berechnet ein robustes Qualitaetsmass aus Margin und Entropie."""
        raw = np.array(
            [max(0.0, float(scores.get(e, 0.0))) for e in self._EMOTIONS],
            dtype=np.float64,
        )
        total = float(raw.sum())
        if total <= 1e-12:
            return 0.0

        probs = raw / total
        sorted_probs = np.sort(probs)
        top1 = float(sorted_probs[-1])
        top2 = float(sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0
        margin = max(0.0, top1 - top2)

        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        max_entropy = float(np.log(max(2, len(self._EMOTIONS))))
        entropy_norm = entropy / max_entropy if max_entropy > 1e-12 else 1.0
        entropy_conf = 1.0 - max(0.0, min(1.0, entropy_norm))

        q = (
            float(UNCERTAINTY_MARGIN_WEIGHT) * margin
            + float(UNCERTAINTY_ENTROPY_WEIGHT) * entropy_conf
        )
        return max(0.0, min(1.0, float(q)))

    def _loop(self):
        while self._running:
            timeout = 0.05 if self._burst_remaining > 0 else 1.0
            try:
                frame = self._q.get(timeout=timeout)
            except queue.Empty:
                self._maybe_fallback()
                continue

            if self._burst_remaining > 0:
                self._burst_remaining -= 1

            # Frame verkleinern fuer schnellere Analyse
            h, w = frame.shape[:2]
            if w > ANALYSIS_FRAME_SIZE:
                scale = ANALYSIS_FRAME_SIZE / w
                small = cv2.resize(frame, (ANALYSIS_FRAME_SIZE, int(h * scale)))
            else:
                small = frame

            # Beleuchtungsnormalisierung
            small = normalize_lighting(small)

            try:
                face = analyze_emotion_frame(small)

                if isinstance(face, dict):
                    scores = face.get("emotion", {})
                    dominant = face.get("dominant_emotion", "neutral")
                    confidence = scores.get(dominant, 0.0) / 100.0

                    # Kalibrierung auch fuer unsichere Frames anwenden,
                    # damit der Soft-Update-Pfad konsistent bleibt.
                    scores = self._apply_calibration(scores)
                    quality = self._compute_prediction_quality(scores)

                    if confidence >= MIN_CONFIDENCE:
                        self._last_face_time = time.time()

                        # EMA mit Confidence-Gewichtung aktualisieren
                        self._update_ema(scores, confidence)

                        # Burst pruefen
                        self._check_burst(confidence)

                        best, best_w = self._dominant_emotion()
                        valence, arousal = self._compute_valence_arousal()
                        trend_v = self._compute_trend()

                        with self._lock:
                            self._result = {
                                "emotion": best,
                                "confidence": confidence,
                                "quality": quality,
                                "ema_vector": self._ema.copy(),
                                "valence": valence,
                                "arousal": arousal,
                                "trend_valence": trend_v,
                            }
                    elif confidence >= SOFT_MIN_CONFIDENCE:
                        self._last_face_time = time.time()

                        # Unsichere Vorhersagen nur schwach einmischen statt hart zu verwerfen.
                        soft_conf = confidence * LOW_CONFIDENCE_ALPHA_SCALE
                        self._update_ema(scores, soft_conf)

                        best, best_w = self._dominant_emotion()
                        valence, arousal = self._compute_valence_arousal()
                        trend_v = self._compute_trend()

                        with self._lock:
                            self._result = {
                                "emotion": best,
                                "confidence": confidence,
                                "quality": quality,
                                "ema_vector": self._ema.copy(),
                                "valence": valence,
                                "arousal": arousal,
                                "trend_valence": trend_v,
                            }
                    else:
                        self._maybe_fallback()
                else:
                    self._maybe_fallback()

            except Exception as exc:
                if self._telemetry is not None:
                    self._telemetry.record(
                        component="emotion",
                        code=DEEPFACE_ANALYZE_FAILED,
                        detail="DeepFace.analyze failed, fallback applied",
                        exc=exc,
                        level=logging.DEBUG,
                        cooldown_s=10.0,
                    )
                self._maybe_fallback()

    def _maybe_fallback(self):
        """Driftet EMA-Vektor sanft Richtung neutral wenn zu lange kein Gesicht erkannt."""
        if time.time() - self._last_face_time > FALLBACK_AFTER_SECONDS:
            d = FALLBACK_DECAY
            self._ema_prev = self._ema.copy()
            for e in self._EMOTIONS:
                target = 1.0 if e == "neutral" else 0.0
                self._ema[e] = (1.0 - d) * self._ema[e] + d * target
            normalize_vector_inplace(self._ema, self._EMOTIONS)

            best, _ = self._dominant_emotion()
            valence, arousal = self._compute_valence_arousal()
            with self._lock:
                self._result = {
                    "emotion": best,
                    "confidence": 0.0,
                    "quality": 0.0,
                    "ema_vector": self._ema.copy(),
                    "valence": valence,
                    "arousal": arousal,
                    "trend_valence": 0.0,
                }

    def get(self) -> dict:
        with self._lock:
            return self._result.copy()

    def stop(self):
        self._running = False
