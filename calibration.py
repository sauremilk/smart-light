"""Interaktive Benutzer-Kalibrierung fuer die Emotionserkennung.

Laeuft 7 Emotionen × CALIBRATION_SECONDS_PER_EMOTION Sekunden durch.
Der Benutzer zeigt jeweils die angezeigte Emotion, DeepFace wird aufgezeichnet
und ein Offset-Vektor berechnet, der spaeter auf die Live-Erkennung angewendet wird.
"""

import json
import logging
import time
import cv2
from deepface import DeepFace
import config

log = logging.getLogger("emotion-light.calibration")

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LABEL_DE = {
    "angry": "WUETEND",
    "disgust": "EKEL",
    "fear": "ANGST",
    "happy": "GLUECKLICH",
    "sad": "TRAURIG",
    "surprise": "UEBERRASCHT",
    "neutral": "NEUTRAL",
}


def run_calibration(path: str | None = None):
    """Fuehrt die interaktive Kalibrierung durch und speichert als JSON.

    Args:
        path: Zielpfad fuer die Kalibrierungsdatei. Standard: config.CALIBRATION_FILE
    """
    if path is None:
        path = config.CALIBRATION_FILE

    secs = config.CALIBRATION_SECONDS_PER_EMOTION
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Kamera nicht verfuegbar fuer Kalibrierung.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    calibration: dict[str, dict[str, float]] = {}

    for emotion in EMOTIONS:
        label_de = LABEL_DE.get(emotion, emotion.upper())
        log.info("Kalibrierung: %s (%d Sekunden)", label_de, secs)

        # Countdown 3 Sekunden
        for countdown in range(3, 0, -1):
            _show_countdown(cap, label_de, countdown)

        scores_sum = {e: 0.0 for e in EMOTIONS}
        n_frames = 0
        t_start = time.monotonic()

        while time.monotonic() - t_start < secs:
            ok, frame = cap.read()
            if not ok:
                continue

            # Fortschrittsanzeige
            elapsed = time.monotonic() - t_start
            progress = elapsed / secs
            _draw_overlay(frame, label_de, progress)
            cv2.imshow("Kalibrierung", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC = Abbruch
                cap.release()
                cv2.destroyAllWindows()
                log.warning("Kalibrierung abgebrochen.")
                return

            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",  # Schnell fuer Kalibrierung
                    silent=True,
                )
                face = result[0] if isinstance(result, list) else result
                for e in EMOTIONS:
                    scores_sum[e] += face["emotion"].get(e, 0.0)
                n_frames += 1
            except Exception:
                pass

        if n_frames > 0:
            avg = {e: scores_sum[e] / n_frames for e in EMOTIONS}
            # Offset = Erwartung - Beobachtung
            # Positive Werte = Emotion wurde zu schwach erkannt
            expected = {e: (100.0 if e == emotion else 0.0) for e in EMOTIONS}
            offset = {e: expected[e] - avg[e] for e in EMOTIONS}
            calibration[emotion] = offset
            log.info("  %s: %d Frames, Avg-Score=%.1f%%", emotion, n_frames, avg[emotion])

    cap.release()
    cv2.destroyAllWindows()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    log.info("Kalibrierung gespeichert: %s", path)


def _show_countdown(cap, label_de: str, seconds: int):
    """Zeigt einen Countdown mit Emotion-Hinweis."""
    t_end = time.monotonic() + 1.0
    while time.monotonic() < t_end:
        ok, frame = cap.read()
        if ok:
            cv2.putText(
                frame,
                f"Naechste Emotion: {label_de}",
                (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                str(seconds),
                (290, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (0, 255, 255),
                4,
            )
            cv2.imshow("Kalibrierung", frame)
        cv2.waitKey(30)


def _draw_overlay(frame, label_de: str, progress: float):
    """Zeichnet Emotion-Label und Fortschrittsbalken auf den Frame."""
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        f"Zeige: {label_de}",
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2,
    )
    # Fortschrittsbalken
    bar_y = h - 30
    bar_w = int(w * 0.8)
    bar_x = int(w * 0.1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (80, 80, 80), -1)
    fill_w = int(bar_w * min(progress, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 20), (0, 255, 0), -1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_calibration()
