"""Frame preprocessing: color constancy and CLAHE normalization."""

import cv2
import numpy as np

from config import CLAHE_CLIP_LIMIT, USE_COLOR_CONSTANCY

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


def resize_for_width(frame, target_width: int):
    """Skaliert ein Frame nur dann herunter, wenn es breiter als target_width ist."""
    if target_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / float(w)
    return cv2.resize(frame, (target_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
