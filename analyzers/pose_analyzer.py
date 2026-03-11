"""Koerpersprache-Analyse via MediaPipe Pose.

Extrahiert Arousal-relevante Merkmale aus Koerperhaltung:
- Schulterposition (hoch = Stress, niedrig = Entspannung)
- Kopfneigung (vorne = Muedigkeit, aufrecht = Aufmerksamkeit)
- Schulter-Asymmetrie (ungleich = Unruhe)

Liefert einen Arousal-Offset (-1.0 bis +1.0) der auf die Lichtsteuerung angewendet wird.
"""

import logging
import threading
import queue
import cv2
import os
from core.asset_integrity import ensure_mediapipe_model

log = logging.getLogger("emotion-light.pose")


class PoseEmotionAnalyzer:
    """Analysiert Koerperhaltung in einem Hintergrund-Thread via MediaPipe Pose."""

    def __init__(self):
        import mediapipe as mp  # noqa: F401 — Verfuegbarkeit pruefen
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._running = False
        self._result = {"arousal_offset": 0.0}
        self._pose = None
        self._pose_mode = None  # "solutions" | "tasks"

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, frame):
        """Uebergibt einen Frame (non-blocking)."""
        try:
            while not self._q.empty():
                self._q.get_nowait()
            self._q.put_nowait(frame)
        except queue.Full:
            pass

    def _init_pose(self):
        """Initialisiert MediaPipe Pose (lazy)."""
        if self._pose is not None:
            return

        # 1) Legacy-API (mp.solutions.pose)
        try:
            import mediapipe as mp
            PoseCls = mp.solutions.pose.Pose
            self._pose = PoseCls(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._pose_mode = "solutions"
            log.info("MediaPipe Pose initialisiert (legacy solutions API).")
            return
        except Exception:
            pass

        # 2) Tasks-API (neuere/angepasste Wheels)
        try:
            import mediapipe as mp
            mt = mp.tasks
            BaseOptions = mt.BaseOptions
            PoseLandmarker = mt.vision.PoseLandmarker
            PoseLandmarkerOptions = mt.vision.PoseLandmarkerOptions
            VisionRunningMode = mt.vision.RunningMode

            model_dir = os.path.join("pretrained_models", "mediapipe")
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
            log.info("MediaPipe Pose initialisiert (tasks API).")
        except Exception as exc:
            log.warning("MediaPipe Pose konnte nicht initialisiert werden: %s", exc)

    def _loop(self):
        self._init_pose()
        if self._pose is None:
            return

        while self._running:
            try:
                frame = self._q.get(timeout=2.0)
            except queue.Empty:
                continue

            try:
                lm = self._extract_landmarks(frame)
                if lm is None:
                    continue
                arousal_offset = self._compute_arousal(lm, frame.shape)

                with self._lock:
                    self._result = {"arousal_offset": arousal_offset}

            except Exception as exc:
                log.debug("Pose-Analyse fehlgeschlagen: %s", exc)

    def _extract_landmarks(self, frame):
        """Extrahiert Pose-Landmarks aus solutions- oder tasks-API."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._pose_mode == "solutions":
            results = self._pose.process(rgb)
            if results.pose_landmarks is None:
                return None
            return results.pose_landmarks.landmark

        if self._pose_mode == "tasks":
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._pose.detect(mp_image)
            if not result.pose_landmarks:
                return None
            return result.pose_landmarks[0]

        return None

    def _compute_arousal(self, landmarks, frame_shape) -> float:
        """Berechnet Arousal-Offset aus Pose-Landmarks."""
        h, w = frame_shape[:2]

        # Relevante Landmarks (MediaPipe Pose indices)
        # 0 = Nase, 11 = linke Schulter, 12 = rechte Schulter
        # 7 = linkes Ohr, 8 = rechtes Ohr
        nose = landmarks[0]
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        signals = []

        # 1. Schulterhoehe relativ zum Bild (hoeher = angespannt = mehr Arousal)
        avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        # Normiert: 0.3 (hoch) bis 0.7 (niedrig) → mapped auf +0.5 bis -0.5
        shoulder_signal = (0.5 - avg_shoulder_y) * 2.0
        shoulder_signal = max(-1.0, min(1.0, shoulder_signal))
        signals.append(shoulder_signal * 0.4)

        # 2. Kopfneigung: Nase-Y relativ zu Schultern (vorne/unten = niedrig Arousal)
        head_drop = nose.y - avg_shoulder_y  # positiv = Kopf unter Schultern = muede
        head_signal = -head_drop * 5.0  # Skalierung
        head_signal = max(-1.0, min(1.0, head_signal))
        signals.append(head_signal * 0.35)

        # 3. Schulter-Asymmetrie (ungleich = Unruhe = erhoehtes Arousal)
        asymmetry = abs(l_shoulder.y - r_shoulder.y)
        asym_signal = min(asymmetry * 10.0, 1.0)
        signals.append(asym_signal * 0.25)

        return max(-1.0, min(1.0, sum(signals)))

    def get(self) -> dict:
        with self._lock:
            return self._result.copy()

    def stop(self):
        self._running = False
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass
