"""
rPPG-basierte Herzfrequenz (HR) und HRV-Analyse via Webcam.

Algorithmus: CHROM (De Haan & Jeanne, 2013)
  1. MediaPipe Face Mesh → Stirn-ROI lokalisieren
  2. Mittlere RGB-Werte im Stirn-ROI puffern
  3. CHROM-Verfahren → Blood-Volume-Pulse-Signal (BVP)
  4. Butterworth-Bandpass 0.7–3.5 Hz → Peak-Detektion
  5. Inter-Beat-Intervalle (IBI) → BPM + RMSSD (HRV)

Läuft in einem eigenen Daemon-Thread.
"""

import logging
import threading
import queue
import time
import os
import numpy as np
import cv2
from collections import deque
from asset_integrity import ensure_mediapipe_model

log = logging.getLogger("emotion-light.hrv")

# Bandpass-Grenzen für plausible Herzfrequenzen
_HR_LOW_HZ = 0.7    # 42 BPM
_HR_HIGH_HZ = 3.5   # 210 BPM

# IBI-Plausibilitätsgrenzen (Sekunden)
_IBI_MIN = 0.28   # ~214 BPM
_IBI_MAX = 2.0    # ~30 BPM

# Stabilitaetsparameter
_PEAK_PROMINENCE_STD = 0.18  # Mindest-Prominenz relativ zur BVP-Std
_IBI_REL_DEV_MAX = 0.25      # Max. relative Abweichung vom Median-IBI
_HR_SMOOTH_MAX_DELTA = 8.0   # Maximaler BPM-Sprung pro Update


# ─────────────────────── Signalverarbeitung ────────────────────

def _bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """Butterworth-Bandpass 2. Ordnung via scipy.signal."""
    from scipy.signal import butter, filtfilt
    nyq = fs / 2.0
    low_n = _HR_LOW_HZ / nyq
    high_n = min(_HR_HIGH_HZ / nyq, 0.99)
    b, a = butter(2, [low_n, high_n], btype="band")
    # filtfilt benötigt mindestens 3 * max(len(a), len(b)) Samples
    min_len = 3 * max(len(a), len(b)) + 1
    if len(signal) < min_len:
        return signal
    return filtfilt(b, a, signal)


def _find_peaks(signal: np.ndarray, min_dist: int) -> np.ndarray:
    """Robuste Peak-Detektion mit Distanz + Prominenz."""
    from scipy.signal import find_peaks

    if len(signal) < 5:
        return np.array([], dtype=np.int32)

    sig_std = float(np.std(signal))
    if sig_std < 1e-8:
        return np.array([], dtype=np.int32)

    prominence = max(1e-8, _PEAK_PROMINENCE_STD * sig_std)
    dist = max(1, int(min_dist))
    peaks, _ = find_peaks(signal, distance=dist, prominence=prominence)
    if len(peaks) >= 3:
        return peaks.astype(np.int32)

    # Fallback fuer sehr glatte Signale: Distanz bleibt, Prominenz wird gelockert.
    peaks_fallback, _ = find_peaks(signal, distance=dist)
    return peaks_fallback.astype(np.int32)


def _compute_hr_hrv(ibi_s: np.ndarray) -> tuple[float, float, float]:
    """Berechnet HR (BPM), RMSSD (ms) und SDNN (ms) aus IBI-Array in Sekunden."""
    if len(ibi_s) < 2:
        return 0.0, 0.0, 0.0
    hr = 60.0 / float(np.mean(ibi_s))
    sdnn = float(np.std(ibi_s)) * 1000.0
    rmssd = float(np.sqrt(np.mean(np.diff(ibi_s) ** 2))) * 1000.0
    return hr, rmssd, sdnn


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _resample_rgb_uniform(timestamps: np.ndarray, rgb: np.ndarray, target_fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Resampled unregelmaessige RGB-Zeitreihe auf uniforme Abtastung."""
    if len(timestamps) < 4:
        return timestamps, rgb

    fs = max(8.0, min(float(target_fs), 60.0))
    step = 1.0 / fs
    t0 = float(timestamps[0])
    t1 = float(timestamps[-1])
    if t1 <= t0:
        return timestamps, rgb

    t_uniform = np.arange(t0, t1, step, dtype=np.float64)
    if len(t_uniform) < 12:
        return timestamps, rgb

    rgb_uniform = np.empty((len(t_uniform), 3), dtype=np.float64)
    for c in range(3):
        rgb_uniform[:, c] = np.interp(t_uniform, timestamps, rgb[:, c])

    return t_uniform, rgb_uniform


def _filter_ibi_outliers(ibi: np.ndarray) -> np.ndarray:
    """Entfernt IBI-Ausreisser relativ zum Median (robuster gegen Fehlpeaks)."""
    if len(ibi) < 3:
        return ibi

    median_ibi = float(np.median(ibi))
    if median_ibi <= 0:
        return np.array([], dtype=np.float64)

    rel_dev = np.abs(ibi - median_ibi) / median_ibi
    filtered = ibi[rel_dev <= _IBI_REL_DEV_MAX]
    if len(filtered) >= 2:
        return filtered
    return ibi


# ─────────────────────── Haupt-Klasse ──────────────────────────

class HRVAnalyzer:
    """
    Analysiert Herzfrequenz und HRV via rPPG in einem Hintergrund-Thread.

    Liefert per get() folgendes Dict:
      hr_bpm        – Herzfrequenz in Schlägen pro Minute
      hrv_rmssd     – RMSSD in ms (Maß für kurzfristige HRV)
      hrv_sdnn      – SDNN in ms (Maß für Gesamtvariabilität)
      confidence    – 0.0–1.0 (steigt mit Anzahl gültiger IBIs)
      face_detected – True wenn im letzten Frame ein Gesicht erkannt wurde
    """

    def __init__(self, window_seconds: float = 30.0, target_fps: float = 24.0):
        self._window_seconds = window_seconds
        self._target_fps = target_fps
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._running = False
        self._face_mesh = None
        self._mesh_mode = None  # "solutions" | "tasks" | "haar"
        self._use_haar = False
        self._haar = None

        # Rollender Puffer: Tupel (timestamp, R_mean, G_mean, B_mean)
        max_samples = int(window_seconds * target_fps * 1.5)
        self._buffer: deque = deque(maxlen=max_samples)

        self._result: dict = {
            "hr_bpm": 0.0,
            "hrv_rmssd": 0.0,
            "hrv_sdnn": 0.0,
            "confidence": 0.0,
            "face_detected": False,
        }
        self._last_metrics = {
            "hr_bpm": 0.0,
            "hrv_rmssd": 0.0,
            "hrv_sdnn": 0.0,
            "confidence": 0.0,
        }

    # ──────── öffentliche API ───────────────────────────────────

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="hrv-analyzer").start()

    def stop(self) -> None:
        self._running = False

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

    def _decay_result(self, face_detected: bool, factor: float = 0.86) -> None:
        """Reduziert stale Confidence, wenn gerade kein brauchbares Signal vorliegt."""
        with self._lock:
            conf = float(self._result.get("confidence", 0.0))
            new_conf = conf * factor
            self._result["confidence"] = round(new_conf, 2)
            self._result["face_detected"] = bool(face_detected)

            # Alte HR/HRV-Werte ausblenden, sobald die Confidence praktisch weg ist.
            if new_conf < 0.05:
                self._result["hr_bpm"] = 0.0
                self._result["hrv_rmssd"] = 0.0
                self._result["hrv_sdnn"] = 0.0
                self._last_metrics["hr_bpm"] = 0.0
                self._last_metrics["hrv_rmssd"] = 0.0
                self._last_metrics["hrv_sdnn"] = 0.0
                self._last_metrics["confidence"] = 0.0

    # ──────── Initialisierung ───────────────────────────────────

    def _init_mesh(self) -> bool:
        if self._face_mesh is not None:
            return True
        try:
            import mediapipe as mp
            solutions = getattr(mp, "solutions", None)

            if solutions is not None:
                try:
                    self._face_mesh = solutions.face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self._mesh_mode = "solutions"
                    return True
                except Exception:
                    pass

            # Fallback auf MediaPipe Tasks API, falls solutions nicht verfuegbar ist.
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
                self._face_mesh = FaceLandmarker.create_from_options(options)
                self._mesh_mode = "tasks"
                log.info("HRV-Analyzer: mediapipe.tasks FaceLandmarker aktiv.")
                return True
            except Exception as exc:
                log.warning("HRV-Analyzer: mediapipe.tasks nicht nutzbar (%s), nutze Haar-Fallback.", exc)

            # Letzter Fallback: OpenCV Haar face detection
            self._use_haar = True
            self._mesh_mode = "haar"
            cv2_data = getattr(cv2, "data", None)
            if cv2_data is None:
                log.error("HRV-Analyzer: OpenCV data-Pfad nicht verfuegbar.")
                return False
            cascade_path = cv2_data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            if self._haar.empty():
                log.error("HRV-Analyzer: Haar-Cascade konnte nicht geladen werden: %s", cascade_path)
                return False
            log.warning("HRV-Analyzer: mediapipe.solutions fehlt, nutze OpenCV-Haar-Fallback.")
            return True
        except Exception as exc:
            log.error("HRV-Analyzer: MediaPipe FaceMesh Init fehlgeschlagen: %s", exc)
            return False

    # ──────── ROI-Extraktion ────────────────────────────────────

    def _extract_forehead_roi(self, frame: np.ndarray, landmarks) -> np.ndarray | None:
        """
        Extrahiert die Stirn-ROI aus Face-Mesh-Landmarks.

        Die Stirn entspricht dem oberen ~28 % des Gesichts-Bounding-Box,
        lateral um 20 % beschnitten um Haar- und Hintergrundrauschen zu
        reduzieren.
        """
        h, w = frame.shape[:2]
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        x_min = int(max(0, min(xs)))
        x_max = int(min(w - 1, max(xs)))
        y_min = int(max(0, min(ys)))
        y_max = int(min(h - 1, max(ys)))

        face_h = y_max - y_min
        face_w = x_max - x_min
        if face_h < 10 or face_w < 10:
            return None

        roi_y1 = y_min
        roi_y2 = y_min + int(face_h * 0.28)
        roi_x1 = x_min + int(face_w * 0.20)
        roi_x2 = x_max - int(face_w * 0.20)

        roi_y1 = max(0, roi_y1)
        roi_y2 = min(h, roi_y2)
        roi_x1 = max(0, roi_x1)
        roi_x2 = min(w, roi_x2)

        if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
            return None

        return frame[roi_y1:roi_y2, roi_x1:roi_x2]

    def _extract_forehead_roi_from_bbox(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        """Fallback: Stirn-ROI aus Face-Bounding-Box extrahieren."""
        x, y, w, h = bbox
        if w < 10 or h < 10:
            return None

        roi_y1 = y
        roi_y2 = y + int(h * 0.28)
        roi_x1 = x + int(w * 0.20)
        roi_x2 = x + w - int(w * 0.20)

        fh, fw = frame.shape[:2]
        roi_y1 = max(0, roi_y1)
        roi_y2 = min(fh, roi_y2)
        roi_x1 = max(0, roi_x1)
        roi_x2 = min(fw, roi_x2)

        if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
            return None
        return frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # ──────── CHROM-Algorithmus + HRV ──────────────────────────

    def _process_signal(self) -> bool:
        """
        Führt den CHROM-Algorithmus auf dem aktuellen Puffer durch.

        Die Verarbeitung erfolgt periodisch (alle ~2 s) statt in jedem
        Frame, um CPU-Last zu minimieren.
        """
        if len(self._buffer) < 30:
            return False

        data = np.array(list(self._buffer))   # Shape: (N, 4)
        timestamps = data[:, 0]
        rgb = data[:, 1:4]                    # Spalten: R, G, B

        # Effektive Abtastrate aus Zeitstempeln schätzen
        dt = np.diff(timestamps)
        median_dt = float(np.median(dt)) if len(dt) > 0 else 1.0 / self._target_fps
        if median_dt <= 0:
            return False
        fs = 1.0 / median_dt

        # Auf uniforme Zeitbasis resamplen, damit Filter + Peaks stabiler laufen.
        timestamps, rgb = _resample_rgb_uniform(timestamps, rgb, target_fs=fs)
        if len(timestamps) < 30:
            return False
        fs = 1.0 / max(1e-6, float(np.median(np.diff(timestamps))))

        # CHROM: Normalisierung auf mittlere Intensität
        mean_rgb = rgb.mean(axis=0)
        if np.any(mean_rgb < 1.0):
            return False

        Rn = rgb[:, 0] / mean_rgb[0]
        Gn = rgb[:, 1] / mean_rgb[1]
        Bn = rgb[:, 2] / mean_rgb[2]

        X = 3.0 * Rn - 2.0 * Gn
        Y = 1.5 * Rn + Gn - 1.5 * Bn

        try:
            Xf = _bandpass_filter(X, fs)
            Yf = _bandpass_filter(Y, fs)
        except Exception as exc:
            log.debug("Bandpass fehlgeschlagen: %s", exc)
            return False

        std_y = float(np.std(Yf))
        if std_y < 1e-8:
            return False
        alpha = float(np.std(Xf)) / std_y

        bvp = Xf - alpha * Yf
        if float(np.std(bvp)) < 1e-6:
            # Bei stark korrelierten Farbkanälen kann CHROM numerisch kollabieren.
            # Fallback auf Xf liefert dann ein noch nutzbares Pulssignal.
            bvp = Xf

        # Peak-Detektion: Mindestabstand entspricht ~210 BPM
        min_peak_dist = max(4, int(fs * 60.0 / 210.0))
        peaks = _find_peaks(bvp, min_peak_dist)

        if len(peaks) < 3:
            return False

        # IBI aus Peak-Zeitstempeln
        peak_times = timestamps[peaks]
        ibi = np.diff(peak_times)

        # Plausibilitätsprüfung
        valid = (ibi > _IBI_MIN) & (ibi < _IBI_MAX)
        ibi = ibi[valid]
        ibi = _filter_ibi_outliers(ibi)

        if len(ibi) < 2:
            return False

        hr, rmssd, sdnn = _compute_hr_hrv(ibi)

        # Plausibilitätsprüfung HR
        if not (30 <= hr <= 220):
            return False

        # Confidence kombiniert Datenmenge und Signalregelmaessigkeit,
        # damit sie im 30s-Fenster nicht permanent auf 100% klebt.
        target_ibi = max(20.0, self._window_seconds * 1.6)
        count_conf = _clip01(len(ibi) / target_ibi)
        ibi_cv = float(np.std(ibi) / max(1e-6, np.mean(ibi)))
        regularity_conf = _clip01(1.0 - (ibi_cv / 0.35))
        confidence = _clip01(0.70 * count_conf + 0.30 * regularity_conf)

        prev = self._last_metrics.copy()
        conf_blend = max(0.12, min(0.45, 0.12 + 0.35 * confidence))

        if prev["hr_bpm"] > 0.0:
            raw_delta = hr - prev["hr_bpm"]
            raw_delta = max(-_HR_SMOOTH_MAX_DELTA, min(_HR_SMOOTH_MAX_DELTA, raw_delta))
            hr = prev["hr_bpm"] + raw_delta

        hr_s = (1.0 - conf_blend) * prev["hr_bpm"] + conf_blend * hr
        rmssd_s = (1.0 - conf_blend) * prev["hrv_rmssd"] + conf_blend * rmssd
        sdnn_s = (1.0 - conf_blend) * prev["hrv_sdnn"] + conf_blend * sdnn
        conf_s = max(confidence, (1.0 - 0.25) * prev["confidence"] + 0.25 * confidence)

        self._last_metrics = {
            "hr_bpm": float(hr_s),
            "hrv_rmssd": float(rmssd_s),
            "hrv_sdnn": float(sdnn_s),
            "confidence": float(conf_s),
        }

        with self._lock:
            self._result = {
                "hr_bpm": round(hr_s, 1),
                "hrv_rmssd": round(rmssd_s, 1),
                "hrv_sdnn": round(sdnn_s, 1),
                "confidence": round(conf_s, 2),
                "face_detected": True,
            }

        log.debug(
            "rPPG: HR=%.1f BPM  RMSSD=%.1f ms  SDNN=%.1f ms  conf=%.0f%%  IBIs=%d  fs=%.1f Hz",
            hr, rmssd, sdnn, confidence * 100, len(ibi), fs,
        )
        return True

    # ──────── Hintergrund-Thread ────────────────────────────────

    def _loop(self) -> None:
        if not self._init_mesh():
            return

        last_process_time = 0.0
        process_interval = 2.0   # Sekunden zwischen HRV-Berechnungen

        while self._running:
            try:
                frame = self._q.get(timeout=1.0)
            except queue.Empty:
                continue

            # BGR → RGB für MediaPipe
            if self._use_haar and self._haar is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._haar.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(80, 80),
                )
                if len(faces) == 0:
                    self._decay_result(face_detected=False, factor=0.78)
                    continue
                # Groesstes Gesicht verwenden
                faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                x, y, w, h = [int(v) for v in faces[0]]
                roi = self._extract_forehead_roi_from_bbox(frame, (x, y, w, h))
            else:
                if self._face_mesh is None:
                    continue
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    if self._mesh_mode == "tasks":
                        import mediapipe as mp
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        mp_result = self._face_mesh.detect(mp_image)
                        if not mp_result.face_landmarks:
                            self._decay_result(face_detected=False, factor=0.78)
                            continue
                        landmarks = mp_result.face_landmarks[0]
                    else:
                        rgb_frame.flags.writeable = False
                        mp_result = self._face_mesh.process(rgb_frame)
                        if not mp_result.multi_face_landmarks:
                            self._decay_result(face_detected=False, factor=0.78)
                            continue
                        landmarks = mp_result.multi_face_landmarks[0].landmark
                except Exception as exc:
                    log.debug("Face-Mesh-Verarbeitung fehlgeschlagen: %s", exc)
                    continue

                roi = self._extract_forehead_roi(frame, landmarks)

            if roi is None or roi.size == 0:
                self._decay_result(face_detected=False, factor=0.82)
                continue

            # OpenCV liefert BGR → Kanäle explizit benennen
            B_mean = float(roi[:, :, 0].mean())
            G_mean = float(roi[:, :, 1].mean())
            R_mean = float(roi[:, :, 2].mean())

            self._buffer.append((time.time(), R_mean, G_mean, B_mean))

            now = time.time()
            if now - last_process_time >= process_interval:
                last_process_time = now
                if not self._process_signal():
                    self._decay_result(face_detected=True, factor=0.92)
