"""Atemfrequenz-Erkennung via Webcam und MediaPipe Pose.

Algorithmus:
  1. MediaPipe Pose → Schulter-Landmarken (11 = links, 12 = rechts)
  2. Vertikaler Schulter-Mittelpunkt (Y) über die Zeit in einen Ringpuffer schreiben
  3. Linearen Trend entfernen (langsame Positionsveränderungen)
  4. Butterworth-Bandpass 0.1–0.5 Hz filtert Atemzyklus heraus
  5. Peak-Detektion auf gefiltertem Signal
  6. Inter-Peak-Intervalle → Atemfrequenz in Atemzügen/Minute

Läuft in einem eigenen Daemon-Thread.
"""

import logging
import os
import threading
import queue
import time

import cv2
import numpy as np
from collections import deque
from asset_integrity import ensure_mediapipe_model

log = logging.getLogger("emotion-light.breathing")

# Physiologisch plausible Atemfrequenz-Grenzen
_BR_LOW_HZ  = 0.1   #  6 AZ/min
_BR_HIGH_HZ = 0.5   # 30 AZ/min

# IBI-Plausibilitätsgrenzen (Sekunden zwischen zwei Atemzügen)
_BR_IBI_MIN = 2.0   # ~30 AZ/min
_BR_IBI_MAX = 10.0  #  ~6 AZ/min

# Ruhe-Atemfrequenz für Arousal-Normierung (Formel: offset = (br - REST) / REST)
BR_REST_BPM = 15.0


# ─────────────────────── Signalverarbeitung ───────────────────

def _bandpass_breath(signal: np.ndarray, fs: float) -> np.ndarray:
    """Butterworth-Bandpass 2. Ordnung für den Atemfrequenzbereich (0.1–0.5 Hz)."""
    from scipy.signal import butter, filtfilt
    nyq = fs / 2.0
    low_n  = _BR_LOW_HZ  / nyq
    high_n = min(_BR_HIGH_HZ / nyq, 0.99)
    if low_n <= 0 or low_n >= high_n:
        return signal
    b, a = butter(2, [low_n, high_n], btype="band")
    min_len = 3 * max(len(a), len(b)) + 1
    if len(signal) < min_len:
        return signal
    return filtfilt(b, a, signal)


def _find_peaks_simple(signal: np.ndarray, min_dist: int) -> np.ndarray:
    """Lokale-Maximum-Detektion ohne externe Abhängigkeiten."""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)
    return np.array(peaks, dtype=np.int32)


# ─────────────────────── Haupt-Klasse ──────────────────────────

class BreathingAnalyzer:
    """
    Erkennt Atemfrequenz über Schulterbewegung in einem Hintergrund-Thread.

    Liefert per get() folgendes Dict:
      br_bpm      – Atemfrequenz in Atemzügen/Minute (0.0 = noch nicht bestimmt)
      confidence  – 0.0–1.0 (steigt mit Anzahl gültiger Atemzüge im Puffer)
    """

    def __init__(self, window_seconds: float = 30.0, target_fps: float = 4.0):
        self._window_seconds = window_seconds
        # Ringpuffer: Tupel (timestamp, shoulder_y_normalized)
        max_samples = int(window_seconds * target_fps * 2)
        self._buffer: deque = deque(maxlen=max_samples)

        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._running = False
        self._pose = None
        self._pose_mode: str | None = None

        self._result: dict = {"br_bpm": 0.0, "confidence": 0.0}

    # ──────── öffentliche API ───────────────────────────────────

    def start(self) -> None:
        self._running = True
        threading.Thread(
            target=self._loop, daemon=True, name="breathing-analyzer"
        ).start()

    def stop(self) -> None:
        self._running = False
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass

    def submit(self, frame) -> None:
        """Übergibt einen Frame non-blocking (alten Frame verwerfen)."""
        try:
            while not self._q.empty():
                self._q.get_nowait()
            self._q.put_nowait(frame)
        except queue.Full:
            pass

    def get(self) -> dict:
        with self._lock:
            return self._result.copy()

    # ──────── Initialisierung ───────────────────────────────────

    def _init_pose(self) -> None:
        if self._pose is not None:
            return

        # 1) Legacy-API (mp.solutions.pose)
        try:
            import mediapipe as mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._pose_mode = "solutions"
            log.info("BreathingAnalyzer: MediaPipe Pose initialisiert (solutions API).")
            return
        except Exception:
            pass

        # 2) Tasks-API (neuere mediapipe-Builds)
        try:
            import mediapipe as mp
            mt = mp.tasks
            BaseOptions          = mt.BaseOptions
            PoseLandmarker       = mt.vision.PoseLandmarker
            PoseLandmarkerOptions = mt.vision.PoseLandmarkerOptions
            VisionRunningMode    = mt.vision.RunningMode

            model_dir  = os.path.join("pretrained_models", "mediapipe")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "pose_landmarker_lite.task")
            model_path = ensure_mediapipe_model("pose_landmarker_lite.task", model_dir=model_dir)

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._pose = PoseLandmarker.create_from_options(options)
            self._pose_mode = "tasks"
            log.info("BreathingAnalyzer: MediaPipe Pose initialisiert (tasks API).")
        except Exception as exc:
            log.warning("BreathingAnalyzer: MediaPipe Pose nicht verfuegbar: %s", exc)

    # ──────── Signal-Extraktion ─────────────────────────────────

    def _extract_shoulder_y(self, frame) -> float | None:
        """Gibt die normalisierte Y-Position des Schulter-Mittelpunkts zurück."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._pose_mode == "solutions":
            results = self._pose.process(rgb)
            if results.pose_landmarks is None:
                return None
            lm = results.pose_landmarks.landmark
            # Y in [0, 1]: 0 = oben, 1 = unten → Einatmen senkt Y, Ausatmen hebt Y
            return (lm[11].y + lm[12].y) / 2.0

        if self._pose_mode == "tasks":
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._pose.detect(mp_image)
            if not result.pose_landmarks:
                return None
            lm = result.pose_landmarks[0]
            return (lm[11].y + lm[12].y) / 2.0

        return None

    # ──────── Atemfrequenz-Berechnung ──────────────────────────

    def _compute_br(self) -> tuple[float, float]:
        """Berechnet Atemfrequenz aus dem aktuellen Ringpuffer."""
        if len(self._buffer) < 15:
            return 0.0, 0.0

        buf = list(self._buffer)
        times  = np.array([t for t, _ in buf])
        signal = np.array([y for _, y in buf])

        duration = times[-1] - times[0]
        if duration < 8.0:
            # Mindestens 8 Sekunden Daten (~ 1 Atemzyklus sicher abdecken)
            return 0.0, 0.0

        fs = len(times) / duration  # Effektive Abtastrate in Hz

        # Linearen Trend entfernen (langsame Positionsveränderungen durch Bewegung)
        poly = np.polyfit(times - times[0], signal, 1)
        signal_detrended = signal - np.polyval(poly, times - times[0])

        # Bandpass-Filter anwenden
        filtered = _bandpass_breath(signal_detrended, fs)

        # Peak-Detektion: Mindestabstand = 2 Sekunden (~30 AZ/min Obergrenze)
        min_dist = max(1, int(_BR_IBI_MIN * fs))
        peaks = _find_peaks_simple(filtered, min_dist)

        if len(peaks) < 2:
            return 0.0, 0.0

        # Inter-Peak-Intervalle berechnen
        ibis = np.diff(times[peaks])
        valid = ibis[(ibis >= _BR_IBI_MIN) & (ibis <= _BR_IBI_MAX)]

        if len(valid) < 1:
            return 0.0, 0.0

        br_bpm = 60.0 / float(np.mean(valid))

        # Plausibilitätsprüfung (5–35 AZ/min abdecken inkl. Randfälle)
        if not (5.0 <= br_bpm <= 35.0):
            return 0.0, 0.0

        # Confidence steigt mit Anzahl auswertbarer Atemzüge (max bei 5)
        confidence = min(1.0, float(len(valid)) / 5.0)
        return br_bpm, confidence

    # ──────── Haupt-Thread ──────────────────────────────────────

    def _loop(self) -> None:
        self._init_pose()
        if self._pose is None:
            log.warning("BreathingAnalyzer: kein Pose-Modell verfuegbar – Thread beendet.")
            return

        while self._running:
            try:
                frame = self._q.get(timeout=2.0)
            except queue.Empty:
                continue

            try:
                shoulder_y = self._extract_shoulder_y(frame)
                if shoulder_y is not None:
                    self._buffer.append((time.time(), shoulder_y))
                    br_bpm, confidence = self._compute_br()
                    with self._lock:
                        self._result = {
                            "br_bpm":      br_bpm,
                            "confidence":  confidence,
                        }
            except Exception as exc:
                log.debug("Breathing-Analyse fehlgeschlagen: %s", exc)
