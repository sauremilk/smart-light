#!/usr/bin/env python3
"""
Emotion-gesteuerte Philips Hue Lampe – Hauptprogramm.
Erkennt Emotionen via Webcam (+ optional Audio, Pose) und steuert Hue-Licht in Echtzeit.
Alle Verarbeitung erfolgt lokal – keine Cloud-APIs.
"""

import argparse
import hashlib
import io
import json
import logging
import os
import queue
import sys
import threading
import time
import warnings
from collections import defaultdict, deque

import cv2
import numpy as np


def _configure_third_party_runtime_logs():
    """Reduziert bekannte, nicht-kritische Laufzeitwarnungen von Drittanbieter-Libs."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")

    # TensorFlow-deprecation noise in this stack is known and not actionable at runtime.
    warnings.filterwarnings(
        "ignore",
        message=r".*sparse_softmax_cross_entropy.*",
    )

    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass

    noisy_substrings = (
        "sparse_softmax_cross_entropy is deprecated",
        "inference_feedback_manager.cc:121",
        "landmark_projection_calculator.cc:81",
        "landmark_projection_calculator.cc",
        "Using NORM_RECT without IMAGE_DIMENSIONS",
    )

    class _FilteredStderr(io.TextIOBase):
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def write(self, s):
            if any(n in s for n in noisy_substrings):
                return len(s)
            return self._wrapped.write(s)

        def flush(self):
            return self._wrapped.flush()

    if not isinstance(sys.stderr, _FilteredStderr):
        sys.stderr = _FilteredStderr(sys.stderr)


_configure_third_party_runtime_logs()

from deepface import DeepFace
from phue import Bridge

from analyzers.audio_quality import effective_audio_weight
from analyzers.breathing_analyzer import BR_REST_BPM
from config import (
    ABSENCE_LIGHT_OFF_SECONDS,
    ACTIVITY_AROUSAL_INFLUENCE,
    ACTIVITY_TRANSITION_INFLUENCE,
    ADAPTIVE_ANALYSIS,
    ADAPTIVE_AT_TARGET_THRESHOLD,
    ADAPTIVE_BLEND_ESCALATION,
    ADAPTIVE_BLEND_MAX,
    ADAPTIVE_BLEND_STRENGTH,
    ADAPTIVE_FPS_HIGH,
    ADAPTIVE_FPS_LOW,
    ADAPTIVE_PROGRESS_TIMEOUT,
    ADAPTIVE_REGULATION,
    ADAPTIVE_TARGET_AROUSAL,
    ADAPTIVE_TARGET_VALENCE,
    ALEXA_AMAZON_URL,
    ALEXA_COOLDOWN_SECONDS,
    ALEXA_DEVICE_NAME,
    ALEXA_EMAIL,
    ALEXA_MOOD_PLAYLISTS,
    ALEXA_MUSIC_PROVIDER,
    ALEXA_PASSWORD,
    ALEXA_VOLUME_CONTROL,
    ANALYSIS_EVERY_N_FRAMES,
    ANALYSIS_FRAME_SIZE,
    AUDIO_DYNAMIC_MIN_FACTOR,
    AUDIO_DYNAMIC_QUALITY_EXPONENT,
    AUDIO_WEIGHT,
    BLINK_RATE_FATIGUE_THRESHOLD,
    BLINK_RATE_FOCUS_THRESHOLD,
    BLINK_VALENCE_INFLUENCE,
    BREAK_FATIGUE_TRIGGER_S,
    BREAK_MAX_WORK_MINUTES,
    BREAK_MIN_MINUTES,
    BREAK_POMODORO_BREAK_MINUTES,
    BREAK_POMODORO_ENABLED,
    BREAK_POMODORO_LONG_BREAK_AFTER,
    BREAK_POMODORO_LONG_BREAK_MINUTES,
    BREAK_POMODORO_WORK_MINUTES,
    BREATHING_AROUSAL_INFLUENCE,
    BREATHING_BASELINE_ADAPT_ALPHA,
    BREATHING_BASELINE_SECONDS,
    BREATHING_FRAME_INTERVAL,
    BREATHING_MIN_CONFIDENCE,
    BREATHING_OFFSET_CLAMP,
    BREATHING_OFFSET_EMA_ALPHA,
    BREATHING_PACER_AMPLITUDE,
    BREATHING_PACER_BPM,
    BREATHING_PACER_BR_THRESHOLD,
    BREATHING_PACER_FADE_IN,
    BREATHING_TRANSITION_INFLUENCE,
    BREATHING_WINDOW_SECONDS,
    BURST_CONFIDENCE_DELTA,
    BURST_FRAMES,
    CALIBRATION_FILE,
    CAMERA_BUFFER_SIZE,
    CIRCADIAN_UPDATE_INTERVAL,
    CLAHE_CLIP_LIMIT,
    COGNITIVE_STABILITY_WINDOW,
    DEFAULT_MODE,
    DETECTOR_BACKEND,
    EMA_ALPHA,
    EMA_MIN_WEIGHT,
    EMOTION_MAP,
    EMOTIONS,
    FACE_FINETUNE_ONNX_PATH,
    FACE_MESH_FRAME_SIZE,
    FACE_MESH_WEIGHT,
    FALLBACK_AFTER_SECONDS,
    FALLBACK_DECAY,
    FALLBACK_LIGHT,
    FEEDBACK_COOLDOWN_S,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HEAD_POSE_CONFIDENCE_STRENGTH,
    HEAD_TILT_RELAXATION_THRESHOLD,
    HRV_AROUSAL_INFLUENCE,
    HRV_BASELINE_ADAPT_ALPHA,
    HRV_BASELINE_BPM,
    HRV_FRAME_INTERVAL,
    HRV_MIN_CONFIDENCE,
    HRV_OFFSET_CLAMP,
    HRV_OFFSET_EMA_ALPHA,
    HRV_WINDOW_SECONDS,
    HUE_BRI_QUANT,
    HUE_BRIDGE_IP,
    HUE_HUE_QUANT,
    HUE_LIGHT_IDS,
    HUE_LIGHT_ROLES,
    HUE_MIN_UPDATE_INTERVAL,
    HUE_SAT_QUANT,
    LOW_CONFIDENCE_ALPHA_SCALE,
    LOW_FPS_RECOVERY_HYSTERESIS,
    LOW_QUALITY_MIN_TRANSITION,
    LOW_QUALITY_NEUTRAL_BLEND,
    LOW_QUALITY_THRESHOLD,
    MAX_ANALYSIS_EVERY_N_FRAMES,
    MIN_ANALYSIS_EVERY_N_FRAMES,
    MIN_CONFIDENCE,
    MIN_RUNTIME_FPS,
    MODE_HYSTERESIS_S,
    POSE_FRAME_SIZE,
    POSE_WEIGHT,
    PREDICTIVE_BOOST_DURATION,
    PREDICTIVE_BOOST_FACTOR,
    PREDICTIVE_TREND_THRESHOLD,
    PREDICTIVE_TRIGGER_SECONDS,
    PROSODIC_AROUSAL_INFLUENCE,
    PROSODIC_PITCH_CALM_HZ,
    PROSODIC_PITCH_STRESS_HZ,
    PROSODIC_SPEECH_RATE_HIGH,
    PROSODIC_SPEECH_RATE_LOW,
    PUPIL_AROUSAL_INFLUENCE,
    PUPIL_DILATION_BASELINE,
    PUPIL_OFFSET_CLAMP,
    SHOULDER_DROP_VALENCE_INFLUENCE,
    SOFT_MIN_CONFIDENCE,
    STARTUP_GUARD_DELAY_SECONDS,
    TARGET_FPS,
    TORSO_LEAN_AROUSAL_INFLUENCE,
    TRANSITION_TIME,
    TREND_INFLUENCE,
    UNCERTAINTY_ENTROPY_WEIGHT,
    UNCERTAINTY_MARGIN_WEIGHT,
    USE_ACTIVITY_MONITOR,
    USE_ALEXA,
    USE_AUDIO,
    USE_BREAK_MANAGER,
    USE_BREATHING,
    USE_BREATHING_PACER,
    USE_CIRCADIAN,
    USE_COGNITIVE_CLASSIFIER,
    USE_COLOR_CONSTANCY,
    USE_EXTENDED_POSE,
    USE_FACE_FINETUNE_ONNX,
    USE_FACE_MESH,
    USE_FEEDBACK,
    USE_HEAD_POSE_CONFIDENCE,
    USE_HRV,
    USE_MODE_SYSTEM,
    USE_POSE,
    USE_PROSODIC,
    USE_PUPIL_BLINK,
    USE_VALENCE_AROUSAL,
    VA_BRI_HIGH,
    VA_BRI_LOW,
    VA_HUE_NEGATIVE,
    VA_HUE_NEUTRAL,
    VA_HUE_POSITIVE,
    VA_SAT_HIGH,
    VA_SAT_LOW,
    VALENCE_AROUSAL_MAP,
    WEBCAM_INDEX,
)
from core.break_manager import BreakManager
from core.circadian import CircadianSchedule
from core.cognitive_state import CognitiveClassifier
from core.ema_utils import normalize_vector_inplace, update_ema_vector_inplace
from core.emotion_regulator import EmotionRegulator
from core.error_taxonomy import (
    ACTIVITY_ANALYZER_INIT_FAILED,
    AUDIO_ANALYZER_INIT_FAILED,
    BREATHING_ANALYZER_INIT_FAILED,
    CALIBRATION_LOAD_FAILED,
    DEEPFACE_ANALYZE_FAILED,
    DEEPFACE_WARMUP_FAILED,
    FACEMESH_ANALYZER_INIT_FAILED,
    HRV_ANALYZER_INIT_FAILED,
    HUE_CONNECT_FAILED,
    HUE_OFF_FAILED,
    HUE_REENABLE_FAILED,
    HUE_SEND_FAILED,
    POSE_ANALYZER_INIT_FAILED,
    category_for_error_code,
)
from core.feedback import FeedbackCollector
from core.light_mapping import (
    BreathingPacer,
    blend_emotion_colors,
    compose_multi_light_scene,
    compute_va_from_ema,
    fuse_modalities,
    valence_arousal_to_light,
)
from core.mode_manager import ModeManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _configure_library_loggers() -> None:
    """Drosselt bekannte Drittanbieter-Logger fuer einen sauberen Runtime-Output."""
    noisy_loggers = (
        "speechbrain",
        "huggingface_hub",
        "transformers",
        "torch",
        "urllib3",
        "phue",
    )
    for name in noisy_loggers:
        try:
            logging.getLogger(name).setLevel(logging.WARNING)
        except Exception:
            pass


_configure_library_loggers()


class _ThirdPartyNoiseFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "Wav2Vec2Model is frozen." not in msg


for _handler in logging.getLogger().handlers:
    _handler.addFilter(_ThirdPartyNoiseFilter())


log = logging.getLogger("emotion-light")


try:
    import onnxruntime as ort
except Exception:
    ort = None


class OnnxEmotionModel:
    """ONNX-backed emotion classifier with DeepFace-compatible output shape."""

    _labels = EMOTIONS

    def __init__(self, model_path: str):
        if ort is None:
            raise RuntimeError("onnxruntime is not available")
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def analyze(self, frame: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        logits = self._session.run(None, {self._input_name: x})[0]
        probs = _softmax(logits[0])

        scores = {
            label: float(probs[i] * 100.0) for i, label in enumerate(self._labels)
        }
        dominant = max(scores, key=scores.get)
        return {"emotion": scores, "dominant_emotion": dominant}


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x)
    exp = np.exp(x)
    denom = float(np.sum(exp))
    if denom <= 1e-12:
        return np.full_like(exp, fill_value=1.0 / max(1, exp.size), dtype=np.float64)
    return exp / denom


def _init_optional_onnx_model() -> OnnxEmotionModel | None:
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
        log.warning(
            "Failed to initialize ONNX backend (%s). Falling back to DeepFace.", exc
        )
        return None


_ONNX_MODEL = _init_optional_onnx_model()


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


class RuntimeErrorTelemetry:
    """Sammelt Runtime-Fehler strukturiert und drosselt Log-Spam."""

    def __init__(self, max_events: int = 256):
        self._lock = threading.Lock()
        self._counts = defaultdict(int)
        self._last_emit = {}
        self._events = deque(maxlen=max_events)

    def record(
        self,
        component: str,
        code: str,
        detail: str,
        exc: Exception | None = None,
        level: int = logging.WARNING,
        cooldown_s: float = 5.0,
    ) -> None:
        now = time.time()
        category = category_for_error_code(code)
        key = f"{component}:{code}"

        with self._lock:
            self._counts[key] += 1
            count = self._counts[key]
            self._events.append(
                {
                    "ts": now,
                    "component": component,
                    "category": category,
                    "code": code,
                    "detail": detail,
                    "count": count,
                    "exception": repr(exc) if exc is not None else None,
                }
            )
            last = float(self._last_emit.get(key, 0.0))
            should_emit = (now - last) >= cooldown_s
            if should_emit:
                self._last_emit[key] = now

        if should_emit:
            suffix = f"; exc={exc}" if exc is not None else ""
            log.log(
                level,
                "[ERR:%s] component=%s count=%d detail=%s%s",
                code,
                f"{component}/{category}",
                count,
                detail,
                suffix,
            )

    def summary(self) -> dict:
        with self._lock:
            return dict(self._counts)


ERR_TELEMETRY = RuntimeErrorTelemetry()


# ──────────────────── Hue Controller ───────────────────────────


class HueController:
    """Steuert mehrere Philips Hue Lampen mit rollenbasierter Szenen-Komposition."""

    def __init__(self, ip: str, light_ids: list, light_roles: dict | None = None):
        self.bridge = Bridge(ip)
        self.bridge.connect()
        self.lids = light_ids
        self._roles = light_roles or {lid: "primary" for lid in light_ids}
        self._last_cmd: dict = {}
        self._last_sent_ts = 0.0
        self._send_q: queue.Queue = queue.Queue(maxsize=1)
        self._sender_running = True
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()
        for lid in self.lids:
            self.bridge.set_light(lid, "on", True)
        log.info(
            "Hue Bridge verbunden, %d Lampen aktiviert: %s", len(self.lids), self.lids
        )

    def _sender_loop(self):
        """Sendet Hue-Befehle asynchron, damit der Main-Loop nicht auf I/O wartet."""
        while self._sender_running:
            try:
                item = self._send_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            all_cmds, transition = item
            for lid in self.lids:
                try:
                    cmd = all_cmds[lid].copy()
                    cmd["transitiontime"] = transition
                    self.bridge.set_light(lid, cmd)
                except Exception as e:
                    ERR_TELEMETRY.record(
                        component="hue",
                        code=HUE_SEND_FAILED,
                        detail=f"set_light failed for lid={lid}",
                        exc=e,
                        level=logging.ERROR,
                        cooldown_s=2.0,
                    )

    def _enqueue_latest(self, all_cmds: dict, transition: int) -> None:
        """Haelt nur den neuesten ausstehenden Hue-Befehl in der Queue."""
        try:
            while not self._send_q.empty():
                self._send_q.get_nowait()
            self._send_q.put_nowait((all_cmds, transition))
        except queue.Full:
            pass

    @staticmethod
    def _quantize(value: int, step: int, min_value: int, max_value: int) -> int:
        if step <= 1:
            return max(min_value, min(max_value, int(value)))
        q = int(round(float(value) / step) * step)
        return max(min_value, min(max_value, q))

    def apply(self, params: dict, transition: int = TRANSITION_TIME):
        """Setzt Hue/Bri/Sat auf allen Lampen mit rollenbasierter Anpassung."""
        now = time.time()
        if now - self._last_sent_ts < HUE_MIN_UPDATE_INTERVAL:
            return

        primary_cmd = {
            "hue": self._quantize(params["hue"], HUE_HUE_QUANT, 0, 65535),
            "bri": self._quantize(params["bri"], HUE_BRI_QUANT, 1, 254),
            "sat": self._quantize(params["sat"], HUE_SAT_QUANT, 0, 254),
        }

        # Erzeuge ein Cache-Key aus allen Lampen-Parametern
        all_cmds = {}
        for lid in self.lids:
            role = self._roles.get(lid, "primary")
            role_params = compose_multi_light_scene(primary_cmd, role)
            all_cmds[lid] = {
                "hue": self._quantize(role_params["hue"], HUE_HUE_QUANT, 0, 65535),
                "bri": self._quantize(role_params["bri"], HUE_BRI_QUANT, 1, 254),
                "sat": self._quantize(role_params["sat"], HUE_SAT_QUANT, 0, 254),
            }

        if all_cmds == self._last_cmd:
            return
        self._last_cmd = all_cmds
        self._last_sent_ts = now
        self._enqueue_latest(all_cmds, transition)

    def off(self):
        """Schaltet alle Lampen aus."""
        for lid in self.lids:
            try:
                self.bridge.set_light(lid, "on", False)
            except Exception as e:
                ERR_TELEMETRY.record(
                    component="hue",
                    code=HUE_OFF_FAILED,
                    detail=f"off failed for lid={lid}",
                    exc=e,
                    level=logging.ERROR,
                    cooldown_s=2.0,
                )

    def shutdown(self):
        """Beendet den Sender-Thread kontrolliert."""
        self._sender_running = False
        try:
            self._send_q.put_nowait(None)
        except queue.Full:
            try:
                self._send_q.get_nowait()
                self._send_q.put_nowait(None)
            except Exception:
                pass
        try:
            self._sender_thread.join(timeout=1.5)
        except Exception:
            pass


# ───────────── Beleuchtungsnormalisierung ──────────────────────

_clahe = (
    cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    if CLAHE_CLIP_LIMIT > 0
    else None
)


def gray_world_correction(frame):
    """Gray-World Color-Constancy: korrigiert Farbstich durch die Hue-Lampen.

    Normalisiert jeden Kanal so, dass der Durchschnitt bei 128 liegt.
    Bricht den Feedback-Loop: bunte Lampen → verfaerbtes Kamerabild → falsche Emotion.
    """
    fb = frame.astype(np.float32)
    for c in range(3):
        avg = fb[:, :, c].mean()
        if avg > 1.0:
            fb[:, :, c] *= 128.0 / avg
    return np.clip(fb, 0, 255).astype(np.uint8)


def normalize_lighting(frame):
    """Color-Constancy + CLAHE im LAB-Farbraum fuer stabile Erkennung."""
    result = frame
    if USE_COLOR_CONSTANCY:
        result = gray_world_correction(result)
    if _clahe is None:
        return result
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _resize_for_width(frame, target_width: int):
    """Skaliert ein Frame nur dann herunter, wenn es breiter als target_width ist."""
    if target_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / float(w)
    return cv2.resize(
        frame, (target_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )


# ───────────── Kalibrierung laden ──────────────────────────────


def load_calibration(path: str) -> dict:
    """Laedt Kalibrierungs-Offsets aus JSON. Gibt leeres Dict zurueck bei Fehler."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log.info("Kalibrierung geladen: %s", path)
        return data
    except Exception as exc:
        ERR_TELEMETRY.record(
            component="calibration",
            code=CALIBRATION_LOAD_FAILED,
            detail=f"could not load calibration from {path}",
            exc=exc,
            level=logging.WARNING,
            cooldown_s=15.0,
        )
        return {}


# ──────────────── Emotion Analyzer (Thread) ────────────────────


class EmotionAnalyzer:
    """Asynchrone Emotionserkennung mit EMA-Smoothing, Confidence-Gewichtung,
    CLAHE-Normalisierung, Trend-Analyse und Mikro-Expressions-Burst."""

    _EMOTIONS = list(EMOTION_MAP.keys())  # 7 Emotionen

    def __init__(self, calibration: dict | None = None):
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
                ERR_TELEMETRY.record(
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


# ──────────────────────── Main Loop ────────────────────────────


def _parse_args():
    parser = argparse.ArgumentParser(description="Emotion-gesteuerte Hue-Lampe")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock-Modus: Webcam und Hue-Bridge simuliert",
    )
    parser.add_argument(
        "--bridge-ip",
        default=None,
        metavar="IP",
        help="IP der Hue-Bridge (ueberschreibt config.py)",
    )
    parser.add_argument(
        "--light-ids",
        default=None,
        metavar="IDs",
        help="Kommagetrennte Lampen-IDs, z.B. 2,3,4,6",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Audio-Emotionserkennung deaktivieren",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Koerpersprache-Analyse deaktivieren",
    )
    parser.add_argument(
        "--no-face-mesh",
        action="store_true",
        help="Face-Mesh / Action-Unit-Analyse deaktivieren",
    )
    parser.add_argument(
        "--no-head-pose-penalty",
        action="store_true",
        help="Kopfpose-Confidence-Abschwaechung deaktivieren",
    )
    parser.add_argument(
        "--no-hrv",
        action="store_true",
        help="HRV/Herzfrequenz-Messung via rPPG deaktivieren",
    )
    parser.add_argument(
        "--no-breathing",
        action="store_true",
        help="Atemfrequenz-Erkennung via Schulterbewegung deaktivieren",
    )
    parser.add_argument(
        "--no-activity",
        action="store_true",
        help="Tastatur/Maus-Aktivitaets-Monitoring deaktivieren",
    )
    parser.add_argument(
        "--no-pupil-blink",
        action="store_true",
        help="Pupillen-/Blink-Analyse deaktivieren",
    )
    parser.add_argument(
        "--no-prosodic",
        action="store_true",
        help="Prosodische Stimm-Analyse deaktivieren",
    )
    parser.add_argument(
        "--no-extended-pose",
        action="store_true",
        help="Erweiterte Koerpersprache-Signale deaktivieren",
    )
    parser.add_argument(
        "--mode",
        choices=["AUTO", "FOCUS", "ENERGY", "RELAX", "RECOVERY"],
        default=None,
        help="Optimierungsmodus setzen (ueberschreibt config.py DEFAULT_MODE)",
    )
    parser.add_argument(
        "--no-cognitive",
        action="store_true",
        help="Kognitiven Zustandsklassifikator deaktivieren",
    )
    parser.add_argument(
        "--no-breaks",
        action="store_true",
        help="Pausen-Manager deaktivieren",
    )
    parser.add_argument(
        "--pomodoro",
        action="store_true",
        help="Pomodoro-Modus aktivieren (25/5-Zyklen)",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Benutzer-Feedback-System deaktivieren",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Kalibrierungsmodus starten",
    )
    parser.add_argument(
        "--calibration-file",
        default=None,
        metavar="PATH",
        help="Pfad zur Kalibrierungsdatei (ueberschreibt config.py)",
    )
    parser.add_argument(
        "--session-log",
        default=None,
        metavar="PATH",
        help="Optionaler Pfad fuer JSONL-Session-Log (Evaluation adaptiv vs. Kontrolle)",
    )
    parser.add_argument(
        "--condition",
        choices=["adaptive", "control"],
        default=None,
        help="Bedingung fuer A/B-Studie. Default: aus ADAPTIVE_REGULATION abgeleitet.",
    )
    parser.add_argument(
        "--participant",
        default=None,
        metavar="ID",
        help="Optionale Teilnehmer-ID fuer Evaluations-Logs",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        metavar="ID",
        help="Optionale Session-ID fuer Evaluations-Logs",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Deaktiviert OpenCV-Fenster/Keyboard-UI (fuer headless Umgebungen).",
    )
    parser.add_argument(
        "--no-alexa",
        action="store_true",
        help="Alexa-Steuerung deaktivieren (auch wenn USE_ALEXA=True in config).",
    )
    parser.add_argument(
        "--pseudonymize-session",
        action="store_true",
        help="Pseudonymisiert participant/session_id im Session-Log (gesalzener Hash).",
    )
    return parser.parse_args()


def _append_session_log(path: str, payload: dict):
    """Schreibt einen JSON-Eintrag atomar als JSONL-Zeile weg."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _pseudonymize_identity(value: str | None, salt: str) -> str | None:
    """Ersetzt Identitaeten durch deterministische, gesalzene Hash-IDs."""
    if value is None:
        return None
    raw = f"{salt}:{value}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class _MockBridgeController:
    """Simuliert HueController ohne echte Hardware."""

    def __init__(self):
        log.info("[MOCK] HueController aktiv – keine echte Bridge.")

    def apply(self, params: dict, transition: int = 0):
        log.info(
            "[MOCK] Hue-Befehl: hue=%d bri=%d sat=%d",
            params["hue"],
            params["bri"],
            params["sat"],
        )

    def off(self):
        log.info("[MOCK] Lampe ausschalten simuliert.")

    def shutdown(self):
        return None


def _build_top3_text(fused_ema: dict, emotion: str, confidence: float) -> str:
    """Formats top emotion output for the overlay."""
    if fused_ema:
        top3 = sorted(fused_ema.items(), key=lambda item: item[1], reverse=True)[:3]
        return " | ".join(f"{name} {weight:.0%}" for name, weight in top3)
    return f"{emotion} ({confidence:.0%})"


def _build_status_text(
    fps_display: float,
    analysis_every_n: int,
    has_audio: bool,
    has_pose: bool,
    has_face_mesh: bool,
    has_calibration: bool,
    burst_active: bool,
    low_fps_guard: bool = False,
    has_hrv: bool = False,
    has_breathing: bool = False,
    low_quality_guardrail: bool = False,
    has_activity: bool = False,
    has_pupil_blink: bool = False,
) -> str:
    """Builds the status line shown at the bottom of the overlay."""
    modules = ["Video"]
    if has_audio:
        modules.append("Audio")
    if has_pose:
        modules.append("Pose")
    if has_face_mesh:
        modules.append("FaceMesh")
    if has_hrv:
        modules.append("HRV")
    if has_breathing:
        modules.append("Breath")
    if has_activity:
        modules.append("Activity")
    if has_pupil_blink:
        modules.append("Pupil")
    if has_calibration:
        modules.append("Cal")

    status = f"FPS:{fps_display:.0f}  1/{analysis_every_n}  [{'+'.join(modules)}]"
    if burst_active:
        status += "  BURST"
    if low_fps_guard:
        status += "  FPS-GUARD"
    if low_quality_guardrail:
        status += "  LOW-Q"
    return status


def _blend_light_params(base: dict, fallback: dict, blend: float) -> dict:
    """Mischt einen Lichtzustand mit einem neutral-sicheren Fallback-Zustand."""
    a = max(0.0, min(1.0, float(blend)))
    b = 1.0 - a
    return {
        "hue": int(round(b * float(base["hue"]) + a * float(fallback["hue"]))),
        "bri": int(round(b * float(base["bri"]) + a * float(fallback["bri"]))),
        "sat": int(round(b * float(base["sat"]) + a * float(fallback["sat"]))),
    }


def _draw_va_diagram(
    frame,
    current_v: float,
    current_a: float,
    target_v: float,
    target_a: float,
    at_target: bool,
    size: int = 126,
    margin: int = 12,
):
    """Zeichnet ein klar beschriftetes Valence-Arousal-Diagramm oben rechts.

    Valence (horizontal): links = negativ, rechts = positiv.
    Arousal  (vertical):  oben  = hoch,    unten  = niedrig.
    """
    h, w = frame.shape[:2]
    ox = w - size - margin  # linke obere Ecke des Diagramms
    oy = margin
    title_h = 20
    legend_h = 32
    box_x0 = ox - 10
    box_y0 = oy - 8
    box_x1 = ox + size + 10
    box_y1 = oy + size + title_h + legend_h

    # Halbtransparenter Hintergrund
    overlay_buf = frame.copy()
    cv2.rectangle(overlay_buf, (box_x0, box_y0), (box_x1, box_y1), (14, 14, 14), -1)
    cv2.addWeighted(overlay_buf, 0.58, frame, 0.42, 0, frame)
    cv2.rectangle(frame, (box_x0, box_y0), (box_x1, box_y1), (96, 96, 96), 1)

    # Quadranten subtil einfärben: erleichtert die Orientierung.
    cx = ox + size // 2
    cy = oy + size // 2
    tint = frame.copy()
    cv2.rectangle(
        tint, (ox, oy), (cx, cy), (20, 70, 170), -1
    )  # low valence, high arousal
    cv2.rectangle(
        tint, (cx, oy), (ox + size, cy), (50, 150, 80), -1
    )  # high valence, high arousal
    cv2.rectangle(
        tint, (ox, cy), (cx, oy + size), (80, 100, 150), -1
    )  # low valence, low arousal
    cv2.rectangle(
        tint, (cx, cy), (ox + size, oy + size), (70, 120, 95), -1
    )  # high valence, low arousal
    cv2.addWeighted(tint, 0.12, frame, 0.88, 0, frame)

    # Achsen (Kreuzlinien, grau)
    cv2.line(frame, (ox, cy), (ox + size, cy), (110, 110, 110), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, oy), (cx, oy + size), (110, 110, 110), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (ox, oy), (ox + size, oy + size), (90, 90, 90), 1)

    def _va_to_px(v: float, a: float):
        """Wandelt (valence, arousal) in Pixel-Koordinaten im Diagramm um."""
        px = int(ox + (v + 1.0) / 2.0 * size)
        py = int(oy + (1.0 - (a + 1.0) / 2.0) * size)
        return (
            max(ox, min(ox + size, px)),
            max(oy, min(oy + size, py)),
        )

    tgt_px = _va_to_px(target_v, target_a)
    cur_px = _va_to_px(current_v, current_a)

    # Pfeil IST → SOLL (nur wenn nicht am Ziel)
    if not at_target:
        cv2.arrowedLine(
            frame, cur_px, tgt_px, (30, 200, 255), 2, cv2.LINE_AA, tipLength=0.25
        )

    # Ziel-Punkt (grün, etwas kleiner)
    cv2.circle(frame, tgt_px, 5, (60, 220, 90), -1, cv2.LINE_AA)

    # Ist-Punkt (cyan)
    col_cur = (80, 220, 110) if at_target else (230, 200, 40)
    cv2.circle(frame, cur_px, 6, col_cur, -1, cv2.LINE_AA)

    # Mini-Labels
    label_scale = 0.34
    label_thickness = 1
    cv2.putText(
        frame,
        "IST",
        (cur_px[0] + 5, cur_px[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        col_cur,
        label_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "SOLL",
        (tgt_px[0] + 5, tgt_px[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        (60, 220, 90),
        label_thickness,
        cv2.LINE_AA,
    )

    # Titel + Achsenhinweise
    cv2.putText(
        frame,
        "Valence/Arousal",
        (ox, oy - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "-V",
        (ox + 2, cy - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "+V",
        (ox + size - 22, cy - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "+A",
        (cx + 4, oy + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "-A",
        (cx + 4, oy + size - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    # Mini-Legende
    ly = oy + size + 14
    cv2.circle(frame, (ox + 8, ly), 4, col_cur, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "IST",
        (ox + 16, ly + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(frame, (ox + 58, ly), 4, (60, 220, 90), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "SOLL",
        (ox + 66, ly + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Pfeil = Richtung",
        (ox, ly + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (165, 165, 165),
        1,
        cv2.LINE_AA,
    )


def _draw_overlay(
    frame,
    fused_ema: dict,
    emotion: str,
    confidence: float,
    valence: float,
    arousal: float,
    trend_v: float,
    params: dict,
    transition: int,
    fps_display: float,
    analysis_every_n: int,
    has_audio: bool,
    has_pose: bool,
    has_face_mesh: bool,
    has_calibration: bool,
    burst_active: bool = False,
    low_fps_guard: bool = False,
    low_quality_guardrail: bool = False,
    quality: float = 0.0,
    head_pose_conf: float = 1.0,
    absence_seconds: float = 0.0,
    lights_off_due_absence: bool = False,
    hrv_result: dict | None = None,
    breathing_result: dict | None = None,
    breathing_rest_bpm: float | None = None,
    breathing_offset: float = 0.0,
    reg_info: dict | None = None,
    circadian_label: str = "",
    pacer_info: dict | None = None,
    pupil_dilation: float = 0.0,
    blink_rate: float = 0.0,
    activity_result: dict | None = None,
    torso_lean: float = 0.0,
    shoulder_drop: float = 0.0,
    head_tilt: float = 0.0,
):
    """Renders runtime telemetry text on the camera frame."""
    x0 = 28
    y0 = 44
    line_h = 28

    def _draw_text(
        text: str,
        y: int,
        color: tuple[int, int, int],
        scale: float = 0.58,
        weight: int = 1,
    ):
        # Subtile Kontur + Vordergrund fuer lesbaren, aber ruhigeren Look.
        cv2.putText(
            frame,
            text,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (15, 15, 15),
            weight + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            weight,
            cv2.LINE_AA,
        )

    panel_lines = 4
    if reg_info is not None:
        panel_lines += 1
    if circadian_label:
        panel_lines += 1
    if pacer_info is not None and pacer_info.get("active"):
        panel_lines += 1
    if hrv_result is not None:
        panel_lines += 1
    if breathing_result is not None:
        panel_lines += 1
        if breathing_rest_bpm is not None:
            panel_lines += 1
    if lights_off_due_absence or absence_seconds >= FALLBACK_AFTER_SECONDS:
        panel_lines += 1
    if pupil_dilation > 0 or blink_rate > 0:
        panel_lines += 1
    if activity_result is not None:
        panel_lines += 1
    if torso_lean != 0 or shoulder_drop > 0 or head_tilt != 0:
        panel_lines += 1

    panel_h = panel_lines * line_h + 20
    panel_w = min(frame.shape[1] - 24, 760)
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 12), (14 + panel_w, 12 + panel_h), (26, 26, 26), -1)
    cv2.addWeighted(overlay, 0.36, frame, 0.64, 0, frame)
    cv2.rectangle(frame, (14, 12), (14 + panel_w, 12 + panel_h), (90, 90, 90), 1)

    # Linke Akzentlinie: gruener wenn stabil erkannt, sonst orange.
    accent = (80, 200, 110) if confidence > 0 else (0, 145, 230)
    cv2.rectangle(frame, (14, 12), (18, 12 + panel_h), accent, -1)

    color = (120, 255, 150) if confidence > 0 else (60, 185, 255)
    top3_str = _build_top3_text(fused_ema, emotion, confidence)
    _draw_text(top3_str, y0, color, scale=0.67, weight=2)

    va_str = f"V:{valence:+.2f} A:{arousal:+.2f} T:{trend_v:+.3f}"
    va_str += f" Q:{quality:.2f}"
    _draw_text(va_str, y0 + line_h, (220, 220, 160), scale=0.55, weight=1)

    hpc_str = f"HeadConf:{head_pose_conf:.0%}" if head_pose_conf < 1.0 else ""
    light_str = f"Hue:{params['hue']} Bri:{params['bri']} Sat:{params['sat']}  Trans:{transition}"
    if hpc_str:
        light_str += f"  {hpc_str}"
    _draw_text(light_str, y0 + line_h * 2, (235, 235, 235), scale=0.54, weight=1)

    status = _build_status_text(
        fps_display=fps_display,
        analysis_every_n=analysis_every_n,
        has_audio=has_audio,
        has_pose=has_pose,
        has_face_mesh=has_face_mesh,
        has_calibration=has_calibration,
        burst_active=burst_active,
        low_fps_guard=low_fps_guard,
        low_quality_guardrail=low_quality_guardrail,
        has_hrv=hrv_result is not None,
        has_breathing=breathing_result is not None,
        has_activity=activity_result is not None,
        has_pupil_blink=pupil_dilation > 0 or blink_rate > 0,
    )
    _draw_text(status, y0 + line_h * 3, (205, 205, 95), scale=0.50, weight=1)

    # Regulationszeile
    if reg_info is not None:
        if reg_info["at_target"]:
            reg_color = (80, 220, 110)  # grün = Ziel erreicht
        else:
            reg_color = (40, 185, 255)  # orange = aktive Regulierung
        blend_pct = reg_info["blend"]
        reg_label = reg_info["label"]
        if blend_pct > 0:
            reg_str = f"Reguliere: {reg_label}  (Blend {blend_pct:.0%})"
        else:
            reg_str = f"Reguliere: {reg_label}"

    # Zirkadian-Zeile anzeigen
    y_next = y0 + line_h * (5 if reg_info is not None else 4)
    if circadian_label:
        quality: float = (0.0,)
        low_quality_guardrail: bool = (False,)
        circ_str = f"Tageszeit: {circadian_label}"
        _draw_text(circ_str, y_next, (180, 200, 255), scale=0.50, weight=1)
        y_next += line_h

    # Atemfuehrungs-Zeile anzeigen
    if pacer_info is not None and pacer_info.get("active"):
        fade_pct = pacer_info.get("fade_pct", 0.0)
        pacer_str = f"Atemfuehrung 6/min ({fade_pct:.0%})"
        _draw_text(pacer_str, y_next, (160, 255, 220), scale=0.50, weight=1)
        y_next += line_h

    # HRV-Zeile anzeigen
    if hrv_result is not None:
        hr = hrv_result.get("hr_bpm", 0.0)
        rmssd = hrv_result.get("hrv_rmssd", 0.0)
        conf = hrv_result.get("confidence", 0.0)
        face_detected = bool(hrv_result.get("face_detected", True))
        if not face_detected:
            hrv_str = "HR: kein Gesicht"
            hrv_color = (145, 145, 145)
        elif hr > 0:
            hrv_str = f"HR:{hr:.0f}bpm  RMSSD:{rmssd:.0f}ms  ({conf:.0%})"
            hrv_color = (100, 220, 255)
        else:
            hrv_str = "HR: wird gemessen..."
            hrv_color = (145, 145, 145)
        _draw_text(hrv_str, y_next, hrv_color, scale=0.54, weight=1)
        y_next += line_h

    # Atemfrequenz-Zeile anzeigen
    if breathing_result is not None:
        br = breathing_result.get("br_bpm", 0.0)
        br_conf = breathing_result.get("confidence", 0.0)
        if br > 0:
            br_str = f"Atem:{br:.1f} AZ/min  ({br_conf:.0%})"
            br_color = (140, 255, 180)  # Hellgrün
        else:
            br_str = "Atem: wird gemessen..."
            br_color = (145, 145, 145)
        _draw_text(br_str, y_next, br_color, scale=0.54, weight=1)
        y_next += line_h

        if breathing_rest_bpm is not None:
            offset_str = (
                f"Atem-Basis:{breathing_rest_bpm:.1f}  Offset:{breathing_offset:+.3f}"
            )
            _draw_text(offset_str, y_next, (175, 230, 185), scale=0.5, weight=1)
            y_next += line_h

    # Abwesenheits-Warnung anzeigen
    if lights_off_due_absence:
        _draw_text(
            "LICHT AUS - kein Gesicht", y_next, (40, 70, 255), scale=0.6, weight=2
        )
        y_next += line_h
    elif absence_seconds >= FALLBACK_AFTER_SECONDS:
        remaining = max(0.0, ABSENCE_LIGHT_OFF_SECONDS - absence_seconds)
        warn_str = f"Kein Gesicht - Licht aus in {remaining:.0f}s"
        _draw_text(warn_str, y_next, (40, 175, 255), scale=0.54, weight=1)
        y_next += line_h

    # Pupillen-/Blink-Zeile anzeigen
    if pupil_dilation > 0 or blink_rate > 0:
        pb_parts = []
        if pupil_dilation > 0:
            pb_parts.append(f"Pupille:{pupil_dilation:.2f}")
        if blink_rate > 0:
            pb_parts.append(f"Blink:{blink_rate:.0f}/min")
        pb_str = "  ".join(pb_parts)
        pb_color = (200, 180, 255)  # Helles Lila
        _draw_text(pb_str, y_next, pb_color, scale=0.50, weight=1)
        y_next += line_h

    # Aktivitaets-Zeile anzeigen
    if activity_result is not None:
        kpm = activity_result.get("keys_per_minute", 0.0)
        cl = activity_result.get("cognitive_load", 0.0)
        act_str = f"Aktivitaet: {kpm:.0f} Tasten/min  Load:{cl:.0%}"
        act_color = (255, 200, 140)  # Helles Orange
        _draw_text(act_str, y_next, act_color, scale=0.50, weight=1)
        y_next += line_h

    # Erweiterte Pose-Zeile anzeigen
    if torso_lean != 0 or shoulder_drop > 0 or head_tilt != 0:
        pose_parts = []
        if torso_lean != 0:
            pose_parts.append(f"Lean:{torso_lean:+.2f}")
        if shoulder_drop > 0:
            pose_parts.append(f"Drop:{shoulder_drop:.2f}")
        if head_tilt != 0:
            pose_parts.append(f"Tilt:{head_tilt:+.2f}")
        pose_str = "Pose: " + "  ".join(pose_parts)
        _draw_text(pose_str, y_next, (180, 220, 255), scale=0.50, weight=1)
        y_next += line_h

    # VA-Diagramm: zeigt Ist-Zustand, Ziel und Regulierungsrichtung (oben rechts)
    if reg_info is not None:
        _draw_va_diagram(
            frame,
            current_v=reg_info["current_v"],
            current_a=reg_info["current_a"],
            target_v=reg_info["target_v"],
            target_a=reg_info["target_a"],
            at_target=reg_info["at_target"],
        )


def _circadian_va_to_light(
    valence: float,
    arousal: float,
    hue_neg: int,
    hue_pos: int,
    bri_max: int,
) -> dict:
    """Wie valence_arousal_to_light, aber mit zirkadianen Hue-/Bri-Overrides."""
    if valence >= 0:
        hue = VA_HUE_NEUTRAL + (hue_pos - VA_HUE_NEUTRAL) * valence
    else:
        hue = VA_HUE_NEUTRAL + (hue_neg - VA_HUE_NEUTRAL) * (-valence)
    a_norm = (arousal + 1.0) / 2.0
    bri = VA_BRI_LOW + (min(bri_max, VA_BRI_HIGH) - VA_BRI_LOW) * a_norm
    sat = VA_SAT_LOW + (VA_SAT_HIGH - VA_SAT_LOW) * a_norm
    return {
        "hue": max(0, min(65535, int(round(hue)))),
        "bri": max(1, min(254, int(round(bri)))),
        "sat": max(0, min(254, int(round(sat)))),
    }


def main():
    args = _parse_args()

    # --- Kalibrierungsmodus ---
    if args.calibrate:
        from core.calibration import run_calibration

        cal_path = args.calibration_file or CALIBRATION_FILE
        run_calibration(cal_path)
        return

    bridge_ip = args.bridge_ip or HUE_BRIDGE_IP
    if args.light_ids:
        light_ids = [int(x) for x in args.light_ids.split(",")]
    else:
        light_ids = HUE_LIGHT_IDS

    # --- Kalibrierung laden ---
    cal_path = args.calibration_file or CALIBRATION_FILE
    calibration = load_calibration(cal_path)

    # --- Webcam initialisieren ---
    if args.mock:
        cap = None
        log.info("[MOCK] Webcam simuliert – generiere Dummy-Frames.")
    else:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not args.mock:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        if not cap.isOpened():
            log.error("Webcam %d nicht verfügbar!", WEBCAM_INDEX)
            sys.exit(1)
        log.info("Webcam %d geöffnet (%dx%d)", WEBCAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT)

    # --- Hue Bridge verbinden ---
    if args.mock:
        hue = _MockBridgeController()
    else:
        try:
            hue = HueController(bridge_ip, light_ids, light_roles=HUE_LIGHT_ROLES)
        except Exception as e:
            ERR_TELEMETRY.record(
                component="startup",
                code=HUE_CONNECT_FAILED,
                detail=f"bridge_ip={bridge_ip}",
                exc=e,
                level=logging.ERROR,
                cooldown_s=0.0,
            )
            log.error("Tipp: Bridge-Button drücken und erneut starten.")
            if cap:
                cap.release()
            sys.exit(1)

    # --- Emotion-Analyse starten ---
    analyzer = EmotionAnalyzer(calibration=calibration)
    analyzer.start()
    backend_name = "onnx" if _ONNX_MODEL is not None else DETECTOR_BACKEND
    log.info(
        "Emotion-Analyse gestartet (Backend: %s). Druecke 'q' zum Beenden.",
        backend_name,
    )

    # --- Audio-Analyse (optional) ---
    audio_analyzer = None
    use_audio = USE_AUDIO and not args.no_audio
    if use_audio:
        try:
            from analyzers.audio_analyzer import AudioEmotionAnalyzer

            audio_analyzer = AudioEmotionAnalyzer()
            audio_analyzer.start()
            log.info("Audio-Emotionserkennung gestartet.")
        except Exception as exc:
            ERR_TELEMETRY.record(
                component="startup",
                code=AUDIO_ANALYZER_INIT_FAILED,
                detail="Audio analyzer disabled",
                exc=exc,
                level=logging.WARNING,
                cooldown_s=0.0,
            )
            audio_analyzer = None

    # --- Pose-Analyse (optional) ---
    pose_analyzer = None
    use_pose = USE_POSE and not args.no_pose
    if use_pose:
        try:
            from analyzers.pose_analyzer import PoseEmotionAnalyzer

            pose_analyzer = PoseEmotionAnalyzer()
            pose_analyzer.start()
            log.info("Pose-Analyse gestartet.")
        except Exception as exc:
            ERR_TELEMETRY.record(
                component="startup",
                code=POSE_ANALYZER_INIT_FAILED,
                detail="Pose analyzer disabled",
                exc=exc,
                level=logging.WARNING,
                cooldown_s=0.0,
            )
            pose_analyzer = None

    # --- Face Mesh + Action Units (optional) ---
    face_mesh_analyzer = None
    use_face_mesh = USE_FACE_MESH and not args.no_face_mesh
    if use_face_mesh:
        try:
            from analyzers.face_mesh_analyzer import FaceMeshAnalyzer

            face_mesh_analyzer = FaceMeshAnalyzer()
            face_mesh_analyzer.start()
            log.info("Face-Mesh-Analyse gestartet (AUs + Kopfpose).")
        except Exception as exc:
            ERR_TELEMETRY.record(
                component="startup",
                code=FACEMESH_ANALYZER_INIT_FAILED,
                detail="Face mesh analyzer disabled",
                exc=exc,
                level=logging.WARNING,
                cooldown_s=0.0,
            )
            face_mesh_analyzer = None

    # --- HRV-Analyse via rPPG (optional) ---
    hrv_analyzer = None
    use_hrv = USE_HRV and not args.no_hrv
    if use_hrv:
        try:
            from analyzers.hrv_analyzer import HRVAnalyzer

            hrv_analyzer = HRVAnalyzer(
                window_seconds=HRV_WINDOW_SECONDS,
                target_fps=TARGET_FPS,
            )
            hrv_analyzer.start()
            log.info(
                "HRV-Analyse (rPPG) gestartet (Fenster: %.0fs).", HRV_WINDOW_SECONDS
            )
        except Exception as exc:
            ERR_TELEMETRY.record(
                component="startup",
                code=HRV_ANALYZER_INIT_FAILED,
                detail="HRV analyzer disabled",
                exc=exc,
                level=logging.WARNING,
                cooldown_s=0.0,
            )
            hrv_analyzer = None

    # --- Atemfrequenz-Erkennung via Schulterbewegung (optional) ---
    breathing_analyzer = None
    use_breathing = USE_BREATHING and not args.no_breathing
    if use_breathing:
        try:
            from analyzers.breathing_analyzer import BreathingAnalyzer

            breathing_analyzer = BreathingAnalyzer(
                window_seconds=BREATHING_WINDOW_SECONDS,
                target_fps=TARGET_FPS / BREATHING_FRAME_INTERVAL,
            )
            breathing_analyzer.start()
            log.info(
                "Atemfrequenz-Erkennung gestartet (Fenster: %.0fs, Intervall: %d Frames).",
                BREATHING_WINDOW_SECONDS,
                BREATHING_FRAME_INTERVAL,
            )
        except Exception as exc:
            ERR_TELEMETRY.record(
                component="startup",
                code=BREATHING_ANALYZER_INIT_FAILED,
                detail="Breathing analyzer disabled",
                exc=exc,
                level=logging.WARNING,
                cooldown_s=0.0,
            )
            breathing_analyzer = None

    # --- Tastatur/Maus-Aktivitaets-Monitoring (optional) ---
    activity_analyzer = None
    use_activity = USE_ACTIVITY_MONITOR and not args.no_activity
    if use_activity:
        try:
            from analyzers.activity_analyzer import ActivityAnalyzer

            activity_analyzer = ActivityAnalyzer()
            activity_analyzer.start()
            log.info("Aktivitaets-Monitoring gestartet (Tastatur + Maus).")
        except Exception as exc:
            ERR_TELEMETRY.record(
                component="startup",
                code=ACTIVITY_ANALYZER_INIT_FAILED,
                detail="Activity analyzer disabled",
                exc=exc,
                level=logging.WARNING,
                cooldown_s=0.0,
            )
            activity_analyzer = None

    # --- Warm-up: DeepFace-Modell vorladen ---
    log.info("Emotion-Modell wird geladen (Warm-up)...")
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        analyze_emotion_frame(dummy)
    except Exception as exc:
        ERR_TELEMETRY.record(
            component="startup",
            code=DEEPFACE_WARMUP_FAILED,
            detail="emotion warm-up failed, runtime will retry on demand",
            exc=exc,
            level=logging.DEBUG,
            cooldown_s=0.0,
        )
    log.info("Warm-up abgeschlossen.")

    frame_count = 0
    fps_counter = 0
    fps_display = 0.0
    fps_timer = time.time()
    analysis_every_n = ANALYSIS_EVERY_N_FRAMES
    adapt_cooldown_until = 0.0
    low_fps_guard = False
    lights_off_due_absence = False  # True wenn Licht wegen Abwesenheit ausgeschaltet
    breathing_rest_bpm = float(BR_REST_BPM)
    breathing_baseline_start = time.time()
    breathing_baseline_samples = deque(maxlen=256)
    breathing_arousal_offset_ema = 0.0
    hrv_rest_bpm = float(HRV_BASELINE_BPM)
    hrv_arousal_offset_ema = 0.0
    runtime_started_at = time.time()

    # Adaptiver Regulations-Regler
    regulator = EmotionRegulator(
        target_v=ADAPTIVE_TARGET_VALENCE,
        target_a=ADAPTIVE_TARGET_AROUSAL,
        blend_strength=ADAPTIVE_BLEND_STRENGTH,
        blend_max=ADAPTIVE_BLEND_MAX,
        progress_timeout=ADAPTIVE_PROGRESS_TIMEOUT,
        escalation=ADAPTIVE_BLEND_ESCALATION,
        at_target_threshold=ADAPTIVE_AT_TARGET_THRESHOLD,
    )

    # Zirkadianes Licht-Modell
    circadian = CircadianSchedule() if USE_CIRCADIAN else None
    last_circadian_update = 0.0
    circadian_label = ""
    circadian_hue_neg = VA_HUE_NEGATIVE
    circadian_hue_pos = VA_HUE_POSITIVE
    circadian_bri_max = VA_BRI_HIGH

    # --- Alexa-Controller (optional) ---
    alexa_controller = None
    use_alexa = USE_ALEXA and not args.mock and not getattr(args, "no_alexa", False)
    if use_alexa:
        if not ALEXA_EMAIL or not ALEXA_PASSWORD or not ALEXA_DEVICE_NAME:
            log.warning(
                "USE_ALEXA=True, aber ALEXA_EMAIL/ALEXA_PASSWORD/ALEXA_DEVICE_NAME "
                "fehlen in config_local.py – Alexa-Steuerung deaktiviert."
            )
        else:
            try:
                from core.alexa_controller import AlexaController

                alexa_controller = AlexaController(
                    email=ALEXA_EMAIL,
                    password=ALEXA_PASSWORD,
                    device_name=ALEXA_DEVICE_NAME,
                    amazon_url=ALEXA_AMAZON_URL,
                    cooldown_seconds=ALEXA_COOLDOWN_SECONDS,
                    music_provider=ALEXA_MUSIC_PROVIDER,
                    mood_playlists=ALEXA_MOOD_PLAYLISTS,
                    volume_control=ALEXA_VOLUME_CONTROL,
                )
                alexa_controller.start()
                log.info(
                    "Alexa-Controller gestartet (Geraet: '%s').", ALEXA_DEVICE_NAME
                )
            except Exception as exc:
                log.warning("Alexa-Controller konnte nicht gestartet werden: %s", exc)
                alexa_controller = None

    # Atemfuehrungs-Entrainment
    pacer = (
        BreathingPacer(
            guide_bpm=BREATHING_PACER_BPM,
            amplitude=BREATHING_PACER_AMPLITUDE,
            fade_in_seconds=BREATHING_PACER_FADE_IN,
        )
        if USE_BREATHING_PACER
        else None
    )

    # --- Kognitiver Zustandsklassifikator (optional) ---
    cognitive_classifier = None
    use_cognitive = USE_COGNITIVE_CLASSIFIER and not args.no_cognitive
    if use_cognitive:
        cognitive_classifier = CognitiveClassifier(
            stability_window_s=COGNITIVE_STABILITY_WINDOW
        )
        log.info("Kognitiver Zustandsklassifikator aktiviert.")

    # --- Modus-System (optional) ---
    mode_manager = None
    use_modes = USE_MODE_SYSTEM and not args.no_cognitive  # benoetigt Klassifikator fuer AUTO
    if use_modes:
        initial_mode = args.mode or DEFAULT_MODE
        mode_manager = ModeManager(initial_mode=initial_mode)
        mode_manager._hysteresis_s = MODE_HYSTERESIS_S
        log.info("Modus-System aktiviert (Start: %s).", initial_mode)

    # --- Pausen-Manager (optional) ---
    break_manager = None
    use_breaks = USE_BREAK_MANAGER and not args.no_breaks
    if use_breaks:
        pomo = BREAK_POMODORO_ENABLED or args.pomodoro
        break_manager = BreakManager(
            max_work_minutes=BREAK_MAX_WORK_MINUTES,
            fatigue_trigger_s=BREAK_FATIGUE_TRIGGER_S,
            min_break_minutes=BREAK_MIN_MINUTES,
            pomodoro_enabled=pomo,
            pomodoro_work_minutes=BREAK_POMODORO_WORK_MINUTES,
            pomodoro_break_minutes=BREAK_POMODORO_BREAK_MINUTES,
            pomodoro_long_break_minutes=BREAK_POMODORO_LONG_BREAK_MINUTES,
            pomodoro_long_break_after=BREAK_POMODORO_LONG_BREAK_AFTER,
        )
        log.info("Pausen-Manager aktiviert (Pomodoro: %s).", "ja" if pomo else "nein")

    # --- Feedback-Collector (optional) ---
    feedback_collector = None
    use_feedback = USE_FEEDBACK and not args.no_feedback
    if use_feedback:
        feedback_collector = FeedbackCollector(cooldown_s=FEEDBACK_COOLDOWN_S)
        log.info("Feedback-System aktiviert (Tasten: f=positiv, d=negativ).")

    # Vorausschauende Intervention: Trend-Zaehler
    trend_negative_counter = 0.0
    trend_last_time = time.time()
    last_reg_info = {
        "reg_v": ADAPTIVE_TARGET_VALENCE,
        "reg_a": ADAPTIVE_TARGET_AROUSAL,
        "current_v": ADAPTIVE_TARGET_VALENCE,
        "current_a": ADAPTIVE_TARGET_AROUSAL,
        "target_v": ADAPTIVE_TARGET_VALENCE,
        "target_a": ADAPTIVE_TARGET_AROUSAL,
        "blend": 0.0,
        "label": "Tracking...",
        "at_target": True,
    }

    condition = args.condition or ("adaptive" if ADAPTIVE_REGULATION else "control")
    session_start_ts = time.time()
    last_session_log_ts = 0.0
    session_log_salt = os.environ.get("SESSION_LOG_SALT", "smart-light-default-salt")
    if args.session_log:
        log.info(
            "Session-Logging aktiv: %s (condition=%s)", args.session_log, condition
        )
        if args.pseudonymize_session and session_log_salt == "smart-light-default-salt":
            log.warning("SESSION_LOG_SALT nicht gesetzt - Default-Salt wird verwendet.")

    try:
        while True:
            if args.mock:
                ret, frame = (
                    True,
                    np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8),
                )
                time.sleep(1 / 10)
            else:
                ret, frame = cap.read()
            if not ret:
                log.warning("Frame-Fehler, überspringe...")
                continue

            frame_count += 1
            fps_counter += 1

            # FPS alle 1s aktualisieren
            now_fps = time.time()
            if now_fps - fps_timer >= 1.0:
                fps_display = fps_counter / (now_fps - fps_timer)
                fps_counter = 0
                fps_timer = now_fps

            # Adaptive Analysefrequenz: bei FPS-Einbruch drosseln, bei Reserve wieder erhoehen.
            startup_ready = (
                now_fps - runtime_started_at
            ) >= STARTUP_GUARD_DELAY_SECONDS
            if (
                startup_ready
                and ADAPTIVE_ANALYSIS
                and now_fps >= adapt_cooldown_until
                and fps_display > 0
            ):
                if (
                    fps_display < ADAPTIVE_FPS_LOW
                    and analysis_every_n < MAX_ANALYSIS_EVERY_N_FRAMES
                ):
                    analysis_every_n += 1
                    adapt_cooldown_until = now_fps + 0.8
                    log.info(
                        "Adaptive Analyse: 1/%d (FPS %.1f)",
                        analysis_every_n,
                        fps_display,
                    )
                elif (
                    fps_display > ADAPTIVE_FPS_HIGH
                    and analysis_every_n > MIN_ANALYSIS_EVERY_N_FRAMES
                ):
                    analysis_every_n -= 1
                    adapt_cooldown_until = now_fps + 1.2
                    log.info(
                        "Adaptive Analyse: 1/%d (FPS %.1f)",
                        analysis_every_n,
                        fps_display,
                    )

            # Harte Untergrenze: unter MIN_RUNTIME_FPS optionale Analyse-Last abwerfen.
            if startup_ready and fps_display > 0:
                if not low_fps_guard and fps_display < MIN_RUNTIME_FPS:
                    low_fps_guard = True
                    analysis_every_n = min(
                        MAX_ANALYSIS_EVERY_N_FRAMES, analysis_every_n + 2
                    )
                    log.warning(
                        "FPS-GUARD aktiv (%.1f FPS): optionale Analyse wird gedrosselt",
                        fps_display,
                    )
                elif low_fps_guard and fps_display >= (
                    MIN_RUNTIME_FPS + LOW_FPS_RECOVERY_HYSTERESIS
                ):
                    low_fps_guard = False
                    log.info("FPS-GUARD deaktiviert (%.1f FPS)", fps_display)

            # Burst nur mit FPS-Reserve erlauben, sonst kann schnelle Bewegung FPS stark einbrechen lassen.
            burst_allowed = (
                (not ADAPTIVE_ANALYSIS)
                or (fps_display <= 0)
                or (fps_display >= ADAPTIVE_FPS_LOW)
            ) and not low_fps_guard

            # Frame an Analyzer senden (normal oder Burst)
            if (
                analyzer.burst_active and burst_allowed
            ) or frame_count % analysis_every_n == 0:
                analyzer_frame = _resize_for_width(frame, ANALYSIS_FRAME_SIZE)
                analyzer.submit(analyzer_frame.copy())

            # Pose-Analyse: Frame senden (halbe Rate)
            if (
                pose_analyzer is not None
                and not low_fps_guard
                and frame_count % (analysis_every_n * 2) == 0
            ):
                pose_frame = _resize_for_width(frame, POSE_FRAME_SIZE)
                pose_analyzer.submit(pose_frame.copy())

            # Face-Mesh-Analyse: Frame senden (gleiche Rate wie Hauptanalyse)
            if (
                face_mesh_analyzer is not None
                and not low_fps_guard
                and frame_count % analysis_every_n == 0
            ):
                fm_frame = _resize_for_width(frame, FACE_MESH_FRAME_SIZE)
                face_mesh_analyzer.submit(fm_frame.copy())

            # HRV-Analyse: Frame senden (eigenes festes Intervall, unabhängig von FPS-Guard)
            if hrv_analyzer is not None and frame_count % HRV_FRAME_INTERVAL == 0:
                hrv_analyzer.submit(frame.copy())

            # Atemfrequenz: Frame senden (eigenes festes Intervall)
            if (
                breathing_analyzer is not None
                and frame_count % BREATHING_FRAME_INTERVAL == 0
            ):
                breath_frame = _resize_for_width(frame, POSE_FRAME_SIZE)
                breathing_analyzer.submit(breath_frame.copy())

            # Ergebnisse abrufen
            result = analyzer.get()
            emotion = result["emotion"]
            confidence = result["confidence"]
            quality = float(result.get("quality", confidence))
            ema_vector = result.get("ema_vector", {})
            valence = result.get("valence", 0.0)
            arousal = result.get("arousal", 0.0)
            trend_v = result.get("trend_valence", 0.0)
            low_quality_guardrail = False

            # Audio-EMA abrufen
            audio_ema = None
            audio_confidence = 0.0
            audio_quality = 0.0
            dynamic_audio_weight = 0.0
            if audio_analyzer is not None:
                audio_result = audio_analyzer.get()
                audio_ema = audio_result.get("ema_vector")
                audio_confidence = float(audio_result.get("confidence", 0.0))
                audio_quality = float(audio_result.get("quality", audio_confidence))
                dynamic_audio_weight = effective_audio_weight(
                    base_weight=AUDIO_WEIGHT,
                    audio_quality=audio_quality,
                    audio_confidence=audio_confidence,
                    min_factor=AUDIO_DYNAMIC_MIN_FACTOR,
                    quality_exponent=AUDIO_DYNAMIC_QUALITY_EXPONENT,
                )

            # Pose-Arousal-Offset abrufen
            pose_arousal_offset = 0.0
            torso_lean = 0.0
            shoulder_drop = 0.0
            head_tilt = 0.0
            if pose_analyzer is not None:
                pose_result = pose_analyzer.get()
                pose_arousal_offset = pose_result.get("arousal_offset", 0.0)
                if USE_EXTENDED_POSE and not args.no_extended_pose:
                    torso_lean = pose_result.get("torso_lean", 0.0)
                    shoulder_drop = pose_result.get("shoulder_drop", 0.0)
                    head_tilt = pose_result.get("head_tilt", 0.0)

            # Face-Mesh-Ergebnisse abrufen (AU-Scores + Kopfpose-Confidence + Pupille/Blink)
            face_mesh_scores = None
            head_pose_conf_factor = 1.0
            pupil_dilation = 0.0
            blink_rate = 0.0
            if face_mesh_analyzer is not None:
                fm_result = face_mesh_analyzer.get()
                face_mesh_scores = fm_result.get("au_emotion_scores")
                head_pose_conf_factor = fm_result.get("confidence_factor", 1.0)
                if USE_PUPIL_BLINK and not args.no_pupil_blink:
                    pupil_dilation = fm_result.get("pupil_dilation", 0.0)
                    blink_rate = fm_result.get("blink_rate", 0.0)

            # Aktivitaets-Ergebnis abrufen (Tastatur/Maus)
            activity_result = None
            cognitive_load = 0.0
            if activity_analyzer is not None:
                activity_result = activity_analyzer.get()
                cognitive_load = activity_result.get("cognitive_load", 0.0)

            # HRV-Ergebnis abrufen + optionalen Arousal-Offset berechnen
            hrv_result = None
            hrv_arousal_offset = 0.0
            if hrv_analyzer is not None:
                hrv_result = hrv_analyzer.get()
                hr_bpm = hrv_result.get("hr_bpm", 0.0)
                hrv_conf = hrv_result.get("confidence", 0.0)
                raw_hrv_offset = 0.0
                if (
                    hr_bpm > 0
                    and hrv_conf >= HRV_MIN_CONFIDENCE
                    and HRV_AROUSAL_INFLUENCE > 0
                ):
                    if HRV_BASELINE_ADAPT_ALPHA > 0:
                        a_hr = max(0.0, min(1.0, HRV_BASELINE_ADAPT_ALPHA))
                        hrv_rest_bpm = (1.0 - a_hr) * hrv_rest_bpm + a_hr * hr_bpm

                    # Abweichung von persoenlicher Ruhe-HR bestimmt Arousal-Offset.
                    raw_hrv_offset = (
                        (hr_bpm - hrv_rest_bpm) / max(40.0, hrv_rest_bpm)
                    ) * HRV_AROUSAL_INFLUENCE
                    raw_hrv_offset = max(
                        -HRV_OFFSET_CLAMP, min(HRV_OFFSET_CLAMP, raw_hrv_offset)
                    )

                # HRV-Offset zeitlich glaetten; bei fehlender Messung sanft gegen 0.
                a_h = max(0.0, min(1.0, HRV_OFFSET_EMA_ALPHA))
                hrv_arousal_offset_ema = (
                    1.0 - a_h
                ) * hrv_arousal_offset_ema + a_h * raw_hrv_offset
                hrv_arousal_offset = hrv_arousal_offset_ema

            # Atemfrequenz-Ergebnis abrufen + optionalen Arousal-Offset berechnen
            breathing_result = None
            breathing_arousal_offset = 0.0
            if breathing_analyzer is not None:
                breathing_result = breathing_analyzer.get()
                br_bpm = breathing_result.get("br_bpm", 0.0)
                br_conf = breathing_result.get("confidence", 0.0)
                raw_breathing_offset = 0.0
                if (
                    br_bpm > 0
                    and br_conf >= BREATHING_MIN_CONFIDENCE
                    and BREATHING_AROUSAL_INFLUENCE > 0
                ):
                    # 1) Erste Sekunden: persoenliche Ruhe-Baseline aus stabilen Messungen lernen.
                    if (
                        time.time() - breathing_baseline_start
                    ) <= BREATHING_BASELINE_SECONDS:
                        breathing_baseline_samples.append(br_bpm)
                        if len(breathing_baseline_samples) >= 3:
                            breathing_rest_bpm = float(
                                np.median(np.array(breathing_baseline_samples))
                            )
                    # 2) Danach Baseline langsam weiterfuehren, um Tagesform abzubilden.
                    elif BREATHING_BASELINE_ADAPT_ALPHA > 0:
                        a = BREATHING_BASELINE_ADAPT_ALPHA
                        breathing_rest_bpm = (1.0 - a) * breathing_rest_bpm + a * br_bpm

                    raw_breathing_offset = (
                        (br_bpm - breathing_rest_bpm) / max(1.0, breathing_rest_bpm)
                    ) * BREATHING_AROUSAL_INFLUENCE
                    raw_breathing_offset = max(
                        -BREATHING_OFFSET_CLAMP,
                        min(BREATHING_OFFSET_CLAMP, raw_breathing_offset),
                    )

                # 3) Atem-Offset per EMA glaetten; bei ungueltiger Messung sanft gegen 0 abklingen.
                target_offset = raw_breathing_offset
                a_off = BREATHING_OFFSET_EMA_ALPHA
                breathing_arousal_offset_ema = (
                    1.0 - a_off
                ) * breathing_arousal_offset_ema + a_off * target_offset
                breathing_arousal_offset = breathing_arousal_offset_ema

            # Kopfpose-Kompensation: optional und gedaempft anwenden,
            # damit seitliche Posen nicht unverhaeltnismaessig oft zu neutral fuehren.
            pose_strength = max(0.0, min(1.0, HEAD_POSE_CONFIDENCE_STRENGTH))
            pose_enabled = USE_HEAD_POSE_CONFIDENCE and not args.no_head_pose_penalty
            pose_scale = 1.0
            if pose_enabled:
                pose_scale = (1.0 - pose_strength) + (
                    pose_strength * head_pose_conf_factor
                )
            effective_confidence = confidence * pose_scale

            # Multimodal-Fusion
            reg_info = None
            fused_v = 0.0
            fused_a = 0.0

            # Zirkadianes Licht-Modell: Zielwerte + Hue-Bereiche periodisch aktualisieren
            if (
                circadian is not None
                and (time.time() - last_circadian_update) >= CIRCADIAN_UPDATE_INTERVAL
            ):
                cp = circadian.get_params()
                regulator.set_target(cp["target_v"], cp["target_a"])
                circadian_hue_neg = cp["hue_negative"]
                circadian_hue_pos = cp["hue_positive"]
                circadian_bri_max = cp["bri_max"]
                circadian_label = cp["label"]
                last_circadian_update = time.time()
                log.info(
                    "Zirkadian-Update: %s (V=%.2f A=%.2f)",
                    circadian_label,
                    cp["target_v"],
                    cp["target_a"],
                )

            if ema_vector and effective_confidence > 0.0:
                dynamic_face_mesh_weight = 0.0
                if face_mesh_scores:
                    # Face-Mesh nur stark einmischen, wenn die Pose-/Landmark-Qualitaet stabil ist.
                    dynamic_face_mesh_weight = FACE_MESH_WEIGHT * max(
                        0.0, min(1.0, head_pose_conf_factor)
                    )

                fused_ema = fuse_modalities(
                    ema_vector,
                    audio_ema,
                    pose_arousal_offset,
                    dynamic_audio_weight if audio_ema else 0.0,
                    face_mesh_scores=face_mesh_scores,
                    face_mesh_weight=dynamic_face_mesh_weight,
                )
                fused_v, fused_a = compute_va_from_ema(fused_ema)

                # Vorausschauende Intervention: bei anhaltendem Abwaertstrend Blend verstaerken
                now_trend = time.time()
                dt_trend = now_trend - trend_last_time
                trend_last_time = now_trend
                if trend_v < PREDICTIVE_TREND_THRESHOLD:
                    trend_negative_counter += dt_trend
                else:
                    trend_negative_counter = max(
                        0.0, trend_negative_counter - dt_trend * 2
                    )
                if trend_negative_counter >= PREDICTIVE_TRIGGER_SECONDS:
                    regulator.boost_blend(
                        factor=PREDICTIVE_BOOST_FACTOR,
                        duration_s=PREDICTIVE_BOOST_DURATION,
                    )
                    trend_negative_counter = 0.0
                    log.info(
                        "Predictive Intervention: Abwaertstrend erkannt, Blend verstaerkt"
                    )

                # Adaptive Regulation: Licht nudgt Richtung Zielzustand statt Ist-Zustand zu spiegeln
                if ADAPTIVE_REGULATION and USE_VALENCE_AROUSAL:
                    reg_info = regulator.update(fused_v, fused_a)
                    # Zirkadianes Hue-Override: lokale Hue-Bereiche fuer valence_arousal_to_light
                    reg_v, reg_a = reg_info["reg_v"], reg_info["reg_a"]
                    if circadian is not None:
                        params = _circadian_va_to_light(
                            reg_v,
                            reg_a,
                            circadian_hue_neg,
                            circadian_hue_pos,
                            circadian_bri_max,
                        )
                    else:
                        params = valence_arousal_to_light(reg_v, reg_a)
                else:
                    params = blend_emotion_colors(fused_ema)
                # Pose-Arousal-Offset auf Helligkeit/Saettigung anwenden
                if pose_arousal_offset != 0.0 and POSE_WEIGHT > 0:
                    bri_adj = int(
                        params["bri"] * (1.0 + pose_arousal_offset * POSE_WEIGHT)
                    )
                    params["bri"] = max(1, min(254, bri_adj))
                # HRV-Arousal-Offset anwenden: hohe HR → höhere Helligkeit/Sättigung
                if hrv_arousal_offset != 0.0:
                    bri_adj = int(params["bri"] * (1.0 + hrv_arousal_offset))
                    params["bri"] = max(1, min(254, bri_adj))
                # Atemfrequenz-Arousal-Offset: langsame Atmung → gedämpftes Licht
                if breathing_arousal_offset != 0.0:
                    bri_adj = int(params["bri"] * (1.0 + breathing_arousal_offset))
                    params["bri"] = max(1, min(254, bri_adj))
                    sat_adj = int(
                        params["sat"] * (1.0 + breathing_arousal_offset * 0.5)
                    )
                    params["sat"] = max(0, min(254, sat_adj))

                # Pupillen-Arousal-Offset: groessere Pupillen => hohes kognitives Arousal
                if USE_PUPIL_BLINK and not args.no_pupil_blink and pupil_dilation > 0:
                    pupil_offset = (
                        (pupil_dilation - PUPIL_DILATION_BASELINE)
                        / max(0.1, PUPIL_DILATION_BASELINE)
                    ) * PUPIL_AROUSAL_INFLUENCE
                    pupil_offset = max(
                        -PUPIL_OFFSET_CLAMP, min(PUPIL_OFFSET_CLAMP, pupil_offset)
                    )
                    if pupil_offset != 0.0:
                        bri_adj = int(params["bri"] * (1.0 + pupil_offset))
                        params["bri"] = max(1, min(254, bri_adj))

                # Blink-Rate beeinflusst Valence: hohe Blink-Rate = Muedigkeit → waermeres Licht
                if USE_PUPIL_BLINK and not args.no_pupil_blink and blink_rate > 0:
                    if blink_rate > BLINK_RATE_FATIGUE_THRESHOLD:
                        # Muedigkeit: Hue Richtung warm schieben, Helligkeit senken
                        fatigue_factor = min(
                            1.0, (blink_rate - BLINK_RATE_FATIGUE_THRESHOLD) / 15.0
                        )
                        params["bri"] = max(
                            1,
                            int(
                                params["bri"]
                                * (1.0 - fatigue_factor * BLINK_VALENCE_INFLUENCE)
                            ),
                        )
                    elif blink_rate < BLINK_RATE_FOCUS_THRESHOLD:
                        # Fokus: leicht erhoehte Helligkeit
                        focus_factor = min(
                            1.0, (BLINK_RATE_FOCUS_THRESHOLD - blink_rate) / 8.0
                        )
                        params["bri"] = min(
                            254,
                            int(
                                params["bri"]
                                * (1.0 + focus_factor * BLINK_VALENCE_INFLUENCE)
                            ),
                        )

                # Erweiterte Pose-Signale anwenden
                if USE_EXTENDED_POSE and not args.no_extended_pose:
                    # Vorneigung erhoeht Arousal (Engagement/Fokus)
                    if torso_lean > 0 and TORSO_LEAN_AROUSAL_INFLUENCE > 0:
                        lean_boost = torso_lean * TORSO_LEAN_AROUSAL_INFLUENCE
                        params["bri"] = min(
                            254, int(params["bri"] * (1.0 + lean_boost))
                        )
                    # Schulterabsenkung senkt Valence (Muedigkeit/Resignation)
                    if shoulder_drop > 0 and SHOULDER_DROP_VALENCE_INFLUENCE > 0:
                        drop_dim = shoulder_drop * SHOULDER_DROP_VALENCE_INFLUENCE
                        params["sat"] = max(0, int(params["sat"] * (1.0 - drop_dim)))

                # Aktivitaets-Offset: hohe kognitive Last → angepasstes Arousal
                if (
                    activity_analyzer is not None
                    and cognitive_load > 0
                    and ACTIVITY_AROUSAL_INFLUENCE > 0
                ):
                    # Kognitive Last erhoht die Helligkeit fuer Fokus-Unterstuetzung
                    activity_boost = cognitive_load * ACTIVITY_AROUSAL_INFLUENCE
                    params["bri"] = min(
                        254, int(params["bri"] * (1.0 + activity_boost))
                    )

                # Prosodische Stimm-Offset: Tonhoehe und Sprechrate beeinflussen Arousal
                if USE_PROSODIC and not args.no_prosodic and audio_analyzer is not None:
                    audio_res = audio_analyzer.get()
                    pitch_hz = audio_res.get("pitch_mean_hz", 0.0)
                    speech_rate = audio_res.get("speech_rate", 0.0)
                    if pitch_hz > 0 and PROSODIC_AROUSAL_INFLUENCE > 0:
                        # Hohe Tonhoehe = Stress/Aufregung, niedrige = Ruhe
                        mid_hz = (
                            PROSODIC_PITCH_STRESS_HZ + PROSODIC_PITCH_CALM_HZ
                        ) / 2.0
                        range_hz = (
                            PROSODIC_PITCH_STRESS_HZ - PROSODIC_PITCH_CALM_HZ
                        ) / 2.0
                        if range_hz > 0:
                            pitch_offset = (
                                (pitch_hz - mid_hz) / range_hz
                            ) * PROSODIC_AROUSAL_INFLUENCE
                            pitch_offset = max(-0.15, min(0.15, pitch_offset))
                            params["bri"] = max(
                                1, min(254, int(params["bri"] * (1.0 + pitch_offset)))
                            )
                    if speech_rate > 0 and PROSODIC_AROUSAL_INFLUENCE > 0:
                        # Schnelles Sprechen = hoehere Energie, langsames = ruhiger
                        mid_sr = (
                            PROSODIC_SPEECH_RATE_HIGH + PROSODIC_SPEECH_RATE_LOW
                        ) / 2.0
                        range_sr = (
                            PROSODIC_SPEECH_RATE_HIGH - PROSODIC_SPEECH_RATE_LOW
                        ) / 2.0
                        if range_sr > 0:
                            sr_offset = (
                                ((speech_rate - mid_sr) / range_sr)
                                * PROSODIC_AROUSAL_INFLUENCE
                                * 0.5
                            )
                            sr_offset = max(-0.1, min(0.1, sr_offset))
                            params["sat"] = max(
                                0, min(254, int(params["sat"] * (1.0 + sr_offset)))
                            )

                # Atemfuehrungs-Entrainment: Licht sanft pulsieren lassen
                if pacer is not None:
                    br_active = (
                        breathing_result is not None
                        and breathing_result.get("br_bpm", 0.0)
                        > BREATHING_PACER_BR_THRESHOLD
                        and breathing_result.get("confidence", 0.0)
                        >= BREATHING_MIN_CONFIDENCE
                        and (reg_info is None or not reg_info.get("at_target", False))
                    )
                    pacer.set_active(br_active)
                    pf = pacer.get_pulsation_factor()
                    params["bri"] = max(1, min(254, int(params["bri"] * pf)))
            else:
                fused_ema = ema_vector
                params = FALLBACK_LIGHT

            # Trend beeinflusst Transition-Zeit
            transition = TRANSITION_TIME
            if TREND_INFLUENCE > 0 and trend_v < -0.01:
                transition = int(
                    TRANSITION_TIME * (1.0 + TREND_INFLUENCE * abs(trend_v) * 10)
                )
                transition = min(transition, 50)  # Max 5s
            if BREATHING_TRANSITION_INFLUENCE > 0 and breathing_arousal_offset != 0.0:
                # Schnelles Atmen -> kuerzere Transition (reaktiver), langsames Atmen -> laengere (ruhiger).
                factor = 1.0 - (
                    breathing_arousal_offset * BREATHING_TRANSITION_INFLUENCE
                )
                factor = max(0.6, min(1.8, factor))
                transition = int(round(transition * factor))
                transition = max(1, min(50, transition))
            # Aktivitaet beeinflusst Transition: hohe Eingaberate → reaktiveres Licht
            if (
                activity_analyzer is not None
                and cognitive_load > 0
                and ACTIVITY_TRANSITION_INFLUENCE > 0
            ):
                act_factor = 1.0 - (cognitive_load * ACTIVITY_TRANSITION_INFLUENCE)
                act_factor = max(0.6, min(1.0, act_factor))
                transition = max(1, min(50, int(round(transition * act_factor))))

            # Unsicherheits-Guardrail: bei niedriger Modellqualitaet konservativer steuern.
            if ema_vector and quality < LOW_QUALITY_THRESHOLD:
                low_quality_guardrail = True
                params = _blend_light_params(
                    params, FALLBACK_LIGHT, LOW_QUALITY_NEUTRAL_BLEND
                )
                transition = max(transition, int(LOW_QUALITY_MIN_TRANSITION))

            # Abwesenheits-Check: Licht ausschalten wenn zu lange niemand im Bild
            absence_seconds = time.time() - analyzer.last_face_time
            if absence_seconds >= ABSENCE_LIGHT_OFF_SECONDS:
                if not lights_off_due_absence:
                    lights_off_due_absence = True
                    hue.off()
                    log.info(
                        "Niemand im Bild seit %.0f s – Licht ausgeschaltet.",
                        absence_seconds,
                    )
            else:
                if lights_off_due_absence:
                    lights_off_due_absence = False
                    for lid in hue.lids if hasattr(hue, "lids") else []:
                        try:
                            hue.bridge.set_light(lid, "on", True)
                        except Exception as exc:
                            ERR_TELEMETRY.record(
                                component="hue",
                                code=HUE_REENABLE_FAILED,
                                detail=f"re-enable failed for lid={lid}",
                                exc=exc,
                                level=logging.WARNING,
                                cooldown_s=2.0,
                            )
                    log.info("Person erkannt – Licht wieder eingeschaltet.")

            if not lights_off_due_absence:
                hue.apply(params, transition=transition)

            # Alexa: Musik/Lautstaerke emotion-adaptiv steuern
            if (
                alexa_controller is not None
                and ema_vector
                and effective_confidence > 0.0
            ):
                alexa_controller.update(fused_v, fused_a, emotion)

            reg_info_display = reg_info
            if reg_info is not None:
                last_reg_info = dict(reg_info)
            elif ADAPTIVE_REGULATION and USE_VALENCE_AROUSAL:
                # Kein frisches Regulator-Update in diesem Frame: letzte Ziel-Infos halten,
                # damit das VA-Diagramm stabil sichtbar bleibt.
                reg_info_display = dict(last_reg_info)
                reg_info_display["current_v"] = float(fused_v)
                reg_info_display["current_a"] = float(fused_a)
                dv = reg_info_display["target_v"] - reg_info_display["current_v"]
                da = reg_info_display["target_a"] - reg_info_display["current_a"]
                dist = float(np.sqrt(dv * dv + da * da))
                reg_info_display["at_target"] = dist < ADAPTIVE_AT_TARGET_THRESHOLD

            _draw_overlay(
                frame=frame,
                fused_ema=fused_ema,
                emotion=emotion,
                confidence=confidence,
                quality=quality,
                valence=valence,
                arousal=arousal,
                trend_v=trend_v,
                params=params,
                transition=transition,
                fps_display=fps_display,
                analysis_every_n=analysis_every_n,
                has_audio=audio_analyzer is not None,
                has_pose=pose_analyzer is not None,
                has_face_mesh=face_mesh_analyzer is not None,
                has_calibration=bool(calibration),
                burst_active=analyzer.burst_active,
                low_fps_guard=low_fps_guard,
                low_quality_guardrail=low_quality_guardrail,
                head_pose_conf=pose_scale,
                absence_seconds=absence_seconds,
                lights_off_due_absence=lights_off_due_absence,
                hrv_result=hrv_result,
                breathing_result=breathing_result,
                breathing_rest_bpm=breathing_rest_bpm,
                breathing_offset=breathing_arousal_offset,
                reg_info=reg_info_display,
                circadian_label=circadian_label,
                pacer_info={
                    "active": pacer is not None and pacer.is_active,
                    "fade_pct": pacer.get_fade_pct() if pacer is not None else 0.0,
                }
                if pacer is not None
                else None,
                pupil_dilation=pupil_dilation,
                blink_rate=blink_rate,
                activity_result=activity_result,
                torso_lean=torso_lean,
                shoulder_drop=shoulder_drop,
                head_tilt=head_tilt,
            )

            if args.session_log and (time.time() - last_session_log_ts) >= 1.0:
                target_v = (
                    reg_info["target_v"]
                    if reg_info is not None
                    else ADAPTIVE_TARGET_VALENCE
                )
                target_a = (
                    reg_info["target_a"]
                    if reg_info is not None
                    else ADAPTIVE_TARGET_AROUSAL
                )
                output_v = reg_info["reg_v"] if reg_info is not None else fused_v
                output_a = reg_info["reg_a"] if reg_info is not None else fused_a
                current_dist = float(
                    np.sqrt((target_v - fused_v) ** 2 + (target_a - fused_a) ** 2)
                )
                output_dist = float(
                    np.sqrt((target_v - output_v) ** 2 + (target_a - output_a) ** 2)
                )

                payload = {
                    "timestamp": time.time(),
                    "runtime_sec": time.time() - session_start_ts,
                    "participant": (
                        _pseudonymize_identity(args.participant, session_log_salt)
                        if args.pseudonymize_session
                        else args.participant
                    ),
                    "session_id": (
                        _pseudonymize_identity(args.session_id, session_log_salt)
                        if args.pseudonymize_session
                        else args.session_id
                    ),
                    "condition": condition,
                    "adaptive_enabled": bool(
                        ADAPTIVE_REGULATION and USE_VALENCE_AROUSAL
                    ),
                    "emotion": emotion,
                    "confidence": float(confidence),
                    "audio_confidence": float(audio_confidence),
                    "audio_quality": float(audio_quality),
                    "audio_weight_dynamic": float(dynamic_audio_weight),
                    "model_quality": float(quality),
                    "low_quality_guardrail": bool(low_quality_guardrail),
                    "current_valence": float(fused_v),
                    "current_arousal": float(fused_a),
                    "output_valence": float(output_v),
                    "output_arousal": float(output_a),
                    "target_valence": float(target_v),
                    "target_arousal": float(target_a),
                    "distance_current": current_dist,
                    "distance_output": output_dist,
                    "blend": float(reg_info["blend"]) if reg_info is not None else 0.0,
                    "at_target": bool(reg_info["at_target"])
                    if reg_info is not None
                    else False,
                    "label": reg_info["label"] if reg_info is not None else "n/a",
                    "pupil_dilation": float(pupil_dilation),
                    "blink_rate": float(blink_rate),
                    "cognitive_load": float(cognitive_load),
                    "torso_lean": float(torso_lean),
                    "shoulder_drop": float(shoulder_drop),
                    "head_tilt": float(head_tilt),
                }
                _append_session_log(args.session_log, payload)
                last_session_log_ts = time.time()

            if not args.mock and not args.headless:
                cv2.imshow("Emotion Light", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if cv2.getWindowProperty("Emotion Light", cv2.WND_PROP_VISIBLE) < 1:
                    break

    except KeyboardInterrupt:
        log.info("Keyboard-Interrupt empfangen.")
    finally:
        log.info("Aufräumen...")
        analyzer.stop()
        if audio_analyzer is not None:
            audio_analyzer.stop()
        if pose_analyzer is not None:
            pose_analyzer.stop()
        if face_mesh_analyzer is not None:
            face_mesh_analyzer.stop()
        if hrv_analyzer is not None:
            hrv_analyzer.stop()
        if breathing_analyzer is not None:
            breathing_analyzer.stop()
        if activity_analyzer is not None:
            activity_analyzer.stop()
        if alexa_controller is not None:
            alexa_controller.shutdown()
        hue.apply(FALLBACK_LIGHT, transition=10)
        hue.shutdown()
        if ERR_TELEMETRY.summary():
            log.info("Fehler-Telemetrie: %s", ERR_TELEMETRY.summary())
        time.sleep(1)
        if cap:
            cap.release()
        if not args.mock and not args.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        log.info("Beendet.")


if __name__ == "__main__":
    main()
