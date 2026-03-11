"""Video-Capture-Abstraktion fuer Real- und Mock-Modus."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

log = logging.getLogger("emotion-light")


class FrameSource:
    """Einheitliche Schnittstelle fuer Webcam-Capture und Mock-Frames."""

    def __init__(
        self,
        *,
        mock: bool,
        webcam_index: int,
        width: int,
        height: int,
        target_fps: int,
        buffer_size: int,
    ):
        self._mock = mock
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None

        if mock:
            log.info("[MOCK] Webcam simuliert – generiere Dummy-Frames.")
        else:
            self._cap = cv2.VideoCapture(webcam_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, target_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
            if not self._cap.isOpened():
                raise RuntimeError(f"Webcam {webcam_index} nicht verfuegbar!")
            log.info("Webcam %d geoeffnet (%dx%d)", webcam_index, width, height)

    @property
    def is_mock(self) -> bool:
        return self._mock

    def read(self) -> tuple[bool, np.ndarray]:
        """Liefert (success, frame).  Im Mock-Modus wird ein schwarzes Bild erzeugt."""
        if self._mock:
            time.sleep(1 / 10)
            return True, np.zeros((self._height, self._width, 3), dtype=np.uint8)
        assert self._cap is not None
        return self._cap.read()

    def release(self) -> None:
        """Gibt die Kamera-Ressource frei."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
