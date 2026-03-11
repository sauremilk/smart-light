"""Face-Mesh-basierte Emotionserkennung & Kopfpose-Kompensation.

Nutzt MediaPipe Face Mesh (478 Landmarks) fuer:
1. Action Unit (AU) Extraktion aus Landmark-Distanzen
2. AU-basierte Emotionsklassifikation (ergaenzt DeepFace)
3. Kopfpose-Schaetzung (Yaw/Pitch/Roll) via solvePnP
   → Confidence-Attenuation bei starker Kopfdrehung

Laeuft in einem eigenen Hintergrund-Thread.
"""

import logging
import math
import os
import queue
import threading

import cv2
import numpy as np

from config import (
    EMOTION_MAP,
    HEAD_POSE_MAX_ATTENUATION,
    HEAD_POSE_PENALTY_PITCH,
    HEAD_POSE_PENALTY_YAW,
)
from core.asset_integrity import ensure_mediapipe_model

log = logging.getLogger("emotion-light.facemesh")

_EMOTIONS = list(EMOTION_MAP.keys())

# 3D-Referenzpunkte fuer Kopfpose-Schaetzung (generisches Gesichtsmodell)
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # Nasenspitze
        (0.0, -330.0, -65.0),  # Kinn
        (-225.0, 170.0, -135.0),  # Linkes Auge, linke Ecke
        (225.0, 170.0, -135.0),  # Rechtes Auge, rechte Ecke
        (-150.0, -150.0, -125.0),  # Linker Mundwinkel
        (150.0, -150.0, -125.0),  # Rechter Mundwinkel
    ],
    dtype=np.float64,
)

# Zugehoerige Face Mesh Landmark-Indizes
_POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


class FaceMeshAnalyzer:
    """Analysiert Gesichts-Landmarks in einem Hintergrund-Thread."""

    # Iris-Landmark-Indizes (MediaPipe Face Mesh, refine_landmarks=True)
    _LEFT_IRIS = [468, 469, 470, 471]  # 4 Punkte um linke Iris
    _RIGHT_IRIS = [473, 474, 475, 476]  # 4 Punkte um rechte Iris

    # Augen-Landmarks fuer EAR (Eye Aspect Ratio) Berechnung
    _LEFT_EYE_UPPER = 159  # oberes Augenlid Mitte
    _LEFT_EYE_LOWER = 145  # unteres Augenlid Mitte
    _LEFT_EYE_OUTER = 33  # aeusserer Augenwinkel
    _LEFT_EYE_INNER = 133  # innerer Augenwinkel
    _RIGHT_EYE_UPPER = 386
    _RIGHT_EYE_LOWER = 374
    _RIGHT_EYE_OUTER = 263
    _RIGHT_EYE_INNER = 362

    _EAR_BLINK_THRESHOLD = 0.18  # EAR unter diesem Wert gilt als Blink

    def __init__(self):
        import mediapipe as mp  # noqa: F401 — Verfuegbarkeit pruefen

        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._running = False
        n = len(_EMOTIONS)
        self._result = {
            "au_emotion_scores": {e: 100.0 / n for e in _EMOTIONS},
            "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "confidence_factor": 1.0,
            "pupil_dilation": 0.0,
            "blink_rate": 0.0,
        }
        self._face_mesh = None
        self._mesh_mode = None  # "solutions" | "tasks"
        # Blink-Tracking
        self._blink_timestamps: list[float] = []
        self._last_ear_below = False  # True wenn letzter Frame ein Blink war

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

    def _init_mesh(self):
        """Initialisiert MediaPipe Face Mesh (lazy)."""
        if self._face_mesh is not None:
            return
        try:
            import mediapipe as mp

            try:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._mesh_mode = "solutions"
            except Exception:
                # Fallback fuer mediapipe-Wheels mit nur tasks-API.
                mt = mp.tasks
                BaseOptions = mt.BaseOptions
                FaceLandmarker = mt.vision.FaceLandmarker
                FaceLandmarkerOptions = mt.vision.FaceLandmarkerOptions
                VisionRunningMode = mt.vision.RunningMode

                model_dir = os.path.join("pretrained_models", "mediapipe")
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "face_landmarker.task")
                model_path = ensure_mediapipe_model(
                    "face_landmarker.task", model_dir=model_dir
                )

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

            log.info("MediaPipe Face Mesh initialisiert.")
        except Exception as exc:
            log.warning("Face Mesh konnte nicht initialisiert werden: %s", exc)

    def _extract_landmarks(self, frame):
        """Extrahiert Face-Landmarks aus solutions- oder tasks-API."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._mesh_mode == "solutions":
            results = self._face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None
            return results.multi_face_landmarks[0].landmark

        if self._mesh_mode == "tasks":
            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self._face_mesh.detect(mp_image)
            if not results.face_landmarks:
                return None
            return results.face_landmarks[0]

        return None

    def _loop(self):
        self._init_mesh()
        if self._face_mesh is None:
            return

        while self._running:
            try:
                frame = self._q.get(timeout=2.0)
            except queue.Empty:
                continue

            try:
                h, w = frame.shape[:2]
                landmarks = self._extract_landmarks(frame)
                if landmarks is None:
                    continue

                # Action Units extrahieren
                aus = self._extract_action_units(landmarks, w, h)

                # AU → Emotionsscores
                emotion_scores = self._aus_to_emotions(aus)

                # Kopfpose schaetzen
                head_pose = self._estimate_head_pose(landmarks, w, h)

                # Confidence-Faktor aus Kopfpose
                conf_factor = self._head_pose_to_confidence(head_pose)

                # Pupillengroesse relativ zur Augenoeffnung (0..1)
                pupil_dilation = self._compute_pupil_dilation(landmarks, w, h)

                # Blink-Rate (Blinks/Minute) ueber EAR
                blink_rate = self._update_blink_rate(landmarks, w, h)

                with self._lock:
                    self._result = {
                        "au_emotion_scores": emotion_scores,
                        "head_pose": head_pose,
                        "confidence_factor": conf_factor,
                        "pupil_dilation": pupil_dilation,
                        "blink_rate": blink_rate,
                    }

            except Exception as exc:
                log.debug("Face-Mesh-Analyse fehlgeschlagen: %s", exc)

    # ──────── Action Unit Extraktion ────────

    def _dist(self, lm, i, j, w, h):
        """Euklidische Distanz zwischen zwei Landmarks in Pixeln."""
        dx = (lm[i].x - lm[j].x) * w
        dy = (lm[i].y - lm[j].y) * h
        return math.sqrt(dx * dx + dy * dy)

    def _extract_action_units(self, lm, w, h) -> dict:
        """Extrahiert approximierte Action Units aus Face Mesh Landmarks.

        Alle Werte sind auf die inter-okulare Distanz normiert (0.0 – ~2.0).
        """
        # Inter-okulare Distanz als Normierungsbasis
        iod = self._dist(lm, 33, 263, w, h)
        if iod < 1.0:
            iod = 1.0

        # AU1 – Inner Brow Raise: Abstand innere Braue ↔ oberes Augenlid
        au1_l = self._dist(lm, 107, 159, w, h) / iod
        au1_r = self._dist(lm, 336, 386, w, h) / iod
        au1 = (au1_l + au1_r) / 2.0

        # AU2 – Outer Brow Raise: Abstand aeussere Braue ↔ aeusserer Augenwinkel
        au2_l = self._dist(lm, 70, 130, w, h) / iod
        au2_r = self._dist(lm, 300, 359, w, h) / iod
        au2 = (au2_l + au2_r) / 2.0

        # AU4 – Brow Lowerer: Abstand Brauen-Mitte ↔ Nasenwurzel (klein = gesenkt)
        au4_raw = self._dist(lm, 9, 168, w, h) / iod
        au4 = max(0.0, 0.35 - au4_raw) * 5.0  # Invertiert: kleiner Abstand = hoeher

        # AU6 – Cheek Raiser: Augenoeffnung verkleinert sich
        eye_open_l = self._dist(lm, 159, 145, w, h) / iod
        eye_open_r = self._dist(lm, 386, 374, w, h) / iod
        eye_open = (eye_open_l + eye_open_r) / 2.0
        au6 = max(0.0, 0.18 - eye_open) * 10.0  # Engere Augen = hoeher

        # AU12 – Lip Corner Puller (Laecheln): Mundbreite relativ zur IOD
        mouth_width = self._dist(lm, 61, 291, w, h) / iod
        au12 = max(0.0, mouth_width - 0.55) * 4.0

        # AU15 – Lip Corner Depressor: Mundwinkel-Y relativ zur Mund-Mitte
        mouth_center_y = (lm[13].y + lm[14].y) / 2.0
        corner_avg_y = (lm[61].y + lm[291].y) / 2.0
        au15 = max(0.0, (corner_avg_y - mouth_center_y) * h / iod) * 3.0

        # AU20 – Lip Stretcher: horizontale Lippenstreckung
        au20 = max(0.0, mouth_width - 0.6) * 3.0

        # AU25 – Lips Part: vertikaler Lippenabstand
        lip_sep = self._dist(lm, 13, 14, w, h) / iod
        au25 = lip_sep * 5.0

        # AU26 – Jaw Drop: Kinn-zu-Nase Abstand
        jaw_drop = self._dist(lm, 152, 1, w, h) / iod
        au26 = max(0.0, jaw_drop - 0.65) * 4.0

        # AU9 – Nose Wrinkler: Nasenfluegel-Abstand
        nose_w = self._dist(lm, 97, 326, w, h) / iod
        au9 = max(0.0, nose_w - 0.28) * 5.0

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

    # ──────── AU → Emotions-Mapping ────────

    @staticmethod
    def _aus_to_emotions(aus: dict) -> dict:
        """Regelbasiertes Mapping von Action Units auf 7 Emotionen.

        Gibt Score-Dict (0–100) zurueck, analog zu DeepFace.
        """
        scores = {}

        # Happy: Duchenne-Laecheln = AU6 (Cheek Raiser) + AU12 (Lip Corner Puller)
        scores["happy"] = min(100.0, (aus["AU6"] + aus["AU12"]) * 40.0)

        # Sad: AU1 (Inner Brow Raise) + AU4 (Brow Lowerer) + AU15 (Lip Corner Depressor)
        scores["sad"] = min(
            100.0, (aus["AU1"] * 0.3 + aus["AU4"] * 0.3 + aus["AU15"] * 0.4) * 60.0
        )

        # Angry: AU4 (Brow Lowerer) + AU9 (Nose Wrinkler) stark, wenig AU1
        scores["angry"] = min(
            100.0,
            (aus["AU4"] * 0.5 + aus["AU9"] * 0.3) * 50.0 * max(0.2, 1.0 - aus["AU1"]),
        )

        # Fear: AU1 + AU2 (Brauen hoch) + AU4 + AU20 (Lip Stretcher)
        scores["fear"] = min(
            100.0,
            (aus["AU1"] * 0.3 + aus["AU2"] * 0.2 + aus["AU4"] * 0.2 + aus["AU20"] * 0.3)
            * 55.0,
        )

        # Surprise: AU1 + AU2 (Brauen hoch) + AU25 (Lips Part) + AU26 (Jaw Drop)
        scores["surprise"] = min(
            100.0,
            (
                aus["AU1"] * 0.2
                + aus["AU2"] * 0.2
                + aus["AU25"] * 0.3
                + aus["AU26"] * 0.3
            )
            * 50.0,
        )

        # Disgust: AU9 (Nose Wrinkler) + AU15 (Lip Corner Depressor)
        scores["disgust"] = min(100.0, (aus["AU9"] * 0.5 + aus["AU15"] * 0.5) * 55.0)

        # Neutral: inverse Gesamtaktivitaet
        total_au = sum(aus.values())
        scores["neutral"] = max(0.0, 100.0 - total_au * 15.0)

        # Auf Summe 100 normieren
        total = sum(scores.values())
        if total > 0:
            scores = {e: (v / total) * 100.0 for e, v in scores.items()}

        return scores

    # ──────── Kopfpose-Schaetzung ────────

    @staticmethod
    def _estimate_head_pose(landmarks, w, h) -> dict:
        """Schaetzt Kopf-Yaw/Pitch/Roll via cv2.solvePnP."""
        image_points = np.array(
            [
                (landmarks[idx].x * w, landmarks[idx].y * h)
                for idx in _POSE_LANDMARK_IDS
            ],
            dtype=np.float64,
        )

        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, _ = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        rmat, _ = cv2.Rodrigues(rotation_vec)

        # Euler-Winkel aus Rotationsmatrix
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

    @staticmethod
    def _head_pose_to_confidence(head_pose: dict) -> float:
        """Berechnet Confidence-Faktor (0..1) aus Kopfpose.

        Bei starker Kopfdrehung wird die DeepFace-Confidence abgeschwächt,
        weil Emotionserkennung bei Seitenansicht unzuverlaessig ist.
        """
        yaw_abs = abs(head_pose["yaw"])
        pitch_abs = abs(head_pose["pitch"])

        factor = 1.0

        if yaw_abs > HEAD_POSE_PENALTY_YAW:
            excess = (yaw_abs - HEAD_POSE_PENALTY_YAW) / 30.0  # 30° bis volle Penalty
            factor *= max(HEAD_POSE_MAX_ATTENUATION, 1.0 - excess)

        if pitch_abs > HEAD_POSE_PENALTY_PITCH:
            excess = (pitch_abs - HEAD_POSE_PENALTY_PITCH) / 25.0
            factor *= max(HEAD_POSE_MAX_ATTENUATION, 1.0 - excess)

        return max(HEAD_POSE_MAX_ATTENUATION, factor)

    # ──────── Pupillengroesse & Blink-Rate ────────

    def _compute_pupil_dilation(self, lm, w, h) -> float:
        """Berechnet relative Pupillengroesse aus Iris-Landmarks.

        Gibt einen Wert ~0..1 zurueck: Iris-Durchmesser normiert auf Augenoeffnung.
        Hoher Wert = groessere Pupille relativ zum Auge (hohes kognitives Arousal).
        """
        try:
            # Iris-Durchmesser (links + rechts mitteln)
            iris_d_l = self._iris_diameter(lm, self._LEFT_IRIS, w, h)
            iris_d_r = self._iris_diameter(lm, self._RIGHT_IRIS, w, h)
            iris_d = (iris_d_l + iris_d_r) / 2.0

            # Augenoeffnung (vertikal) als Normierungsbasis
            eye_h_l = self._dist(lm, self._LEFT_EYE_UPPER, self._LEFT_EYE_LOWER, w, h)
            eye_h_r = self._dist(lm, self._RIGHT_EYE_UPPER, self._RIGHT_EYE_LOWER, w, h)
            eye_h = (eye_h_l + eye_h_r) / 2.0

            if eye_h < 1.0:
                return 0.0

            ratio = iris_d / eye_h
            return max(0.0, min(1.0, ratio))
        except (IndexError, AttributeError):
            return 0.0

    def _iris_diameter(self, lm, indices, w, h) -> float:
        """Mittlerer Durchmesser der Iris aus 4 Landmark-Punkten."""
        if len(indices) < 4:
            return 0.0
        try:
            cx = sum(lm[i].x for i in indices) / len(indices) * w
            cy = sum(lm[i].y for i in indices) / len(indices) * h
            dists = []
            for i in indices:
                dx = lm[i].x * w - cx
                dy = lm[i].y * h - cy
                dists.append(math.sqrt(dx * dx + dy * dy))
            return 2.0 * (sum(dists) / len(dists)) if dists else 0.0
        except (IndexError, AttributeError):
            return 0.0

    def _compute_ear(self, lm, w, h) -> float:
        """Eye Aspect Ratio (EAR) – Durchschnitt aus beiden Augen."""
        ear_l = self._single_eye_ear(
            lm,
            w,
            h,
            self._LEFT_EYE_OUTER,
            self._LEFT_EYE_INNER,
            self._LEFT_EYE_UPPER,
            self._LEFT_EYE_LOWER,
        )
        ear_r = self._single_eye_ear(
            lm,
            w,
            h,
            self._RIGHT_EYE_OUTER,
            self._RIGHT_EYE_INNER,
            self._RIGHT_EYE_UPPER,
            self._RIGHT_EYE_LOWER,
        )
        return (ear_l + ear_r) / 2.0

    def _single_eye_ear(self, lm, w, h, outer, inner, upper, lower) -> float:
        """EAR fuer ein einzelnes Auge: vertical / horizontal."""
        horiz = self._dist(lm, outer, inner, w, h)
        vert = self._dist(lm, upper, lower, w, h)
        if horiz < 1.0:
            return 0.3  # default open
        return vert / horiz

    def _update_blink_rate(self, lm, w, h) -> float:
        """Aktualisiert Blink-Zaehler und gibt Blinks/Minute zurueck."""
        import time as _time

        ear = self._compute_ear(lm, w, h)
        now = _time.time()

        # Blink erkannt: EAR faellt unter Schwelle und war vorher drueber
        if ear < self._EAR_BLINK_THRESHOLD:
            if not self._last_ear_below:
                self._blink_timestamps.append(now)
                self._last_ear_below = True
        else:
            self._last_ear_below = False

        # Nur Blinks der letzten 60 Sekunden behalten
        cutoff = now - 60.0
        self._blink_timestamps = [t for t in self._blink_timestamps if t >= cutoff]

        # Blink-Rate berechnen
        if len(self._blink_timestamps) < 2:
            return 0.0
        duration = now - self._blink_timestamps[0]
        if duration < 5.0:
            return 0.0  # Zu wenig Daten
        return len(self._blink_timestamps) / (duration / 60.0)

    def get(self) -> dict:
        with self._lock:
            return self._result.copy()

    def stop(self):
        self._running = False
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass
