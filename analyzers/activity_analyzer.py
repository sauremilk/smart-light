"""Tastatur- und Maus-Aktivitaets-Analyse als kognitiver Kontextgeber.

Misst passiv die Eingaberate (Tastenanschlaege/Minute und Mausbewegung)
und leitet daraus einen kognitiven Lastindikator ab:

- Hohe Tipp-Rate + schnelle Mausbewegung → hohe kognitive Last → Fokus-Modus
- Niedrige Aktivitaet → Regeneration / Pause
- Abrupte Aktivitaetsspitzen → Stress / Deadline-Druck

Laeuft in einem eigenen Daemon-Thread. Benoetigt pynput (optional).
"""

import logging
import threading
import time
from collections import deque

log = logging.getLogger("emotion-light.activity")

# Fenstergroesse fuer Rate-Berechnung
_WINDOW_SECONDS = 60.0


class ActivityAnalyzer:
    """Passives Monitoring von Tastatur- und Maus-Aktivitaet."""

    def __init__(self, window_seconds: float = _WINDOW_SECONDS):
        self._window = window_seconds
        self._lock = threading.Lock()
        self._running = False

        # Zeitstempel-Puffer
        self._keystrokes: deque = deque(maxlen=2000)
        self._mouse_moves: deque = deque(maxlen=2000)

        self._result = {
            "keys_per_minute": 0.0,
            "mouse_moves_per_minute": 0.0,
            "cognitive_load": 0.0,  # 0..1 normiert
        }

        self._kb_listener = None
        self._mouse_listener = None

    def start(self) -> None:
        self._running = True
        threading.Thread(
            target=self._start_listeners, daemon=True, name="activity-analyzer"
        ).start()
        # Update-Thread berechnet periodisch die Raten
        threading.Thread(
            target=self._update_loop, daemon=True, name="activity-calc"
        ).start()

    def _start_listeners(self) -> None:
        """Startet pynput Listener fuer Tastatur und Maus."""
        try:
            from pynput import keyboard, mouse

            def on_key_press(key):
                if self._running:
                    self._keystrokes.append(time.time())

            def on_mouse_move(x, y):
                if self._running:
                    self._mouse_moves.append(time.time())

            self._kb_listener = keyboard.Listener(on_press=on_key_press)
            self._mouse_listener = mouse.Listener(on_move=on_mouse_move)

            self._kb_listener.start()
            self._mouse_listener.start()
            log.info("Aktivitaets-Monitoring gestartet (Tastatur + Maus).")

            # Listener laufen in eigenen Threads, hier auf Ende warten
            while self._running:
                time.sleep(1.0)

        except Exception as exc:
            log.warning(
                "pynput nicht verfuegbar, Aktivitaets-Monitoring deaktiviert: %s", exc
            )
            self._running = False

    def _update_loop(self) -> None:
        """Berechnet periodisch Aktivitaetsraten."""
        while self._running:
            time.sleep(2.0)
            now = time.time()
            cutoff = now - self._window

            # Alte Eintraege entfernen
            while self._keystrokes and self._keystrokes[0] < cutoff:
                self._keystrokes.popleft()
            while self._mouse_moves and self._mouse_moves[0] < cutoff:
                self._mouse_moves.popleft()

            # Raten berechnen
            elapsed = min(
                self._window, now - (self._keystrokes[0] if self._keystrokes else now)
            )
            if elapsed < 5.0:
                elapsed = self._window  # Noch nicht genug Daten

            kpm = len(self._keystrokes) / (elapsed / 60.0) if elapsed > 0 else 0.0
            mpm = len(self._mouse_moves) / (elapsed / 60.0) if elapsed > 0 else 0.0

            # Kognitiver Lastindikator (0..1)
            # Typische Bereiche: 0-50 KPM = niedrig, 50-200 = mittel, 200+ = hoch
            key_load = min(1.0, kpm / 200.0)
            # Maus: 0-100 Bewegungen/min = niedrig, 100-500 = mittel, 500+ = hoch
            mouse_load = min(1.0, mpm / 500.0)
            # Gewichtete Kombination: Tippen ist staerkerer Indikator
            cognitive_load = 0.65 * key_load + 0.35 * mouse_load

            with self._lock:
                self._result = {
                    "keys_per_minute": round(kpm, 1),
                    "mouse_moves_per_minute": round(mpm, 1),
                    "cognitive_load": round(cognitive_load, 3),
                }

    def get(self) -> dict:
        with self._lock:
            return self._result.copy()

    def stop(self) -> None:
        self._running = False
        if self._kb_listener is not None:
            try:
                self._kb_listener.stop()
            except Exception:
                pass
        if self._mouse_listener is not None:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass
