"""Alexa-Controller: steuert Amazon Echo-Geräte basierend auf erkannter Emotion.

Nutzt die inoffizielle alexapy-Bibliothek (dieselbe die Home Assistant verwendet).
Läuft vollständig asynchron in einem Daemon-Thread – der Main-Loop wird nicht blockiert.

Setup (einmalig):
    1. pip install alexapy
    2. Credentials als Umgebungsvariablen setzen (empfohlen)::

           set SMART_LIGHT_ALEXA_EMAIL=deine@email.de
           set SMART_LIGHT_ALEXA_PASSWORD=deinPasswort

       Alternativ in config_local.py (nicht versioniert) eintragen.
    3. In config_local.py aktivieren::

           USE_ALEXA = True
           ALEXA_DEVICE_NAME = "Michs Echo"  # exakter Name aus der Alexa-App

    4. Beim ersten Start: Browser öffnen falls 2-FA verlangt wird
       (alexapy speichert den Cookie danach als alexa_session_cookies.pickle).

Hinweis:
    alexapy nutzt inoffizielle Amazon-Endpunkte. Amazon kann diese jederzeit
    ändern oder den Zugang sperren.
"""

import asyncio
import inspect
import logging
import os
import threading
import time

log = logging.getLogger("emotion-light.alexa")

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Stimmungs-Mapping
# ---------------------------------------------------------------------------


# Hysterese-Schwellen: unterschiedliche Schwellen je nach Richtung
# verhindern Mood-Flapping an den Grenzen.
_MOOD_THRESHOLDS = {
    "positive_enter": 0.25,
    "positive_leave": 0.18,
    "negative_enter": -0.20,
    "negative_leave": -0.13,
}


def _va_to_mood(valence: float, arousal: float, current_mood: str | None = None) -> str:
    """Ordnet Valence/Arousal einer von fünf Stimmungskategorien zu.

    Nutzt Hysterese: um von neutral in positiv/negativ zu wechseln gelten
    strengere Schwellen als um dort zu bleiben.
    """
    is_positive = current_mood in ("energetic_positive", "calm_positive")
    is_negative = current_mood in ("energetic_negative", "calm_negative")

    pos_thresh = (
        _MOOD_THRESHOLDS["positive_leave"] if is_positive else _MOOD_THRESHOLDS["positive_enter"]
    )
    neg_thresh = (
        _MOOD_THRESHOLDS["negative_leave"] if is_negative else _MOOD_THRESHOLDS["negative_enter"]
    )

    if valence >= pos_thresh:
        return "energetic_positive" if arousal >= 0.15 else "calm_positive"
    if valence < neg_thresh:
        return "energetic_negative" if arousal >= 0.10 else "calm_negative"
    return "neutral"


def _va_to_volume(valence: float, arousal: float) -> int:
    """Lautstärke (0–100) aus Valence/Arousal: ruhiges Licht → leiser."""
    base = 35
    adj = int(arousal * 15)
    return max(10, min(70, base + adj))


def _hide_email(email: str) -> str:
    """Maskiert E-Mail-Adresse für Log-Ausgaben."""
    if "@" not in email:
        return "****"
    local, domain = email.split("@", 1)
    return f"{local[:2]}***@{domain}"


# ---------------------------------------------------------------------------
# AlexaController
# ---------------------------------------------------------------------------


class AlexaController:
    """Steuert Amazon Echo asynchron (Thread-safe).

    Der eigene asyncio-Loop läuft in einem Daemon-Thread.
    ``update()`` kann beliebig oft vom Main-Loop aufgerufen werden –
    Neuverbindungslogik und Cooldown-Throttling werden intern verwaltet.
    """

    _MAX_RECONNECT_ATTEMPTS = 3
    _RECONNECT_BASE_DELAY = 10.0  # Sekunden; wird exponentiell erhöht

    def __init__(
        self,
        email: str,
        password: str,
        device_name: str,
        amazon_url: str = "amazon.de",
        cooldown_seconds: float = 30.0,
        music_provider: str = "AMAZON_MUSIC",
        mood_playlists: dict | None = None,
        volume_control: bool = True,
    ):
        self._email = email
        self._password = password
        self._device_name = device_name
        self._amazon_url = amazon_url
        self._cooldown = cooldown_seconds
        self._provider = music_provider
        self._playlists = mood_playlists or {}
        self._volume_control = volume_control

        # Geteilter Zustand (Zugriff mit Lock)
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._current_mood: str | None = None
        self._last_action_time: float = 0.0
        self._last_update_time: float = 0.0  # Zeitpunkt des letzten update()-Aufrufs
        self._absence_warned: bool = False

        # Async objects (nur im Loop-Thread beschreiben)
        self._login = None
        self._device: dict | None = None
        self._ready = False
        self._stop_event: asyncio.Event | None = None

        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Startet den Hintergrundthread mit eigenem asyncio-Loop."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="alexa-loop")
        self._thread.start()

    @property
    def ready(self) -> bool:
        """True wenn Login + Gerätefindung erfolgreich abgeschlossen sind."""
        return self._ready

    def update(self, valence: float, arousal: float, emotion: str) -> None:
        """Thread-safe: reicht ein Emotions-Update in den async-Loop ein.

        Wird vom Main-Loop aufgerufen. Kehrt sofort zurück.
        """
        with self._lock:
            self._last_update_time = time.time()
            self._absence_warned = False
            if self._loop is None or not self._loop.is_running():
                return
        # run_coroutine_threadsafe ist thread-safe, kein Lock nötig
        asyncio.run_coroutine_threadsafe(
            self._handle_update(valence, arousal, emotion),
            self._loop,
        )

    def shutdown(self) -> None:
        """Beendet den asyncio-Loop und wartet auf Threadende."""
        with self._lock:
            loop = self._loop
            stop = self._stop_event

        if loop is not None and loop.is_running() and stop is not None:
            loop.call_soon_threadsafe(stop.set)

        if self._thread is not None:
            self._thread.join(timeout=3.0)

        log.info("Alexa-Controller beendet.")

    # ------------------------------------------------------------------
    # Internes – Loop-Thread
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Einstiegspunkt des Daemon-Threads: erstellt und betreibt den async-Loop."""
        loop = asyncio.new_event_loop()
        with self._lock:
            self._loop = loop
        try:
            loop.run_until_complete(self._main_async())
        except Exception as exc:
            log.error("Alexa async-Loop abgestürzt: %s", exc)
        finally:
            try:
                loop.close()
            except Exception:
                pass

    async def _main_async(self) -> None:
        """Haupt-Coroutine: initialisiert Login, hält Loop am Leben bis Shutdown."""
        self._stop_event = asyncio.Event()

        try:
            import alexapy  # noqa: F401 – nur Importprüfung
        except ImportError:
            log.warning(
                "alexapy ist nicht installiert – Alexa-Steuerung deaktiviert. "
                "Aktivieren mit:  pip install alexapy"
            )
            return

        # Login mit Reconnect-Versuchen bei Fehlschlag
        for attempt in range(1, self._MAX_RECONNECT_ATTEMPTS + 1):
            try:
                await self._initialize_login()
                break
            except Exception as exc:
                if attempt == self._MAX_RECONNECT_ATTEMPTS:
                    log.error(
                        "Alexa-Initialisierung nach %d Versuchen endgültig fehlgeschlagen: %s",
                        attempt,
                        exc,
                    )
                    return
                delay = self._RECONNECT_BASE_DELAY * (2 ** (attempt - 1))
                log.warning(
                    "Alexa-Initialisierung fehlgeschlagen (Versuch %d/%d): %s – "
                    "Nächster Versuch in %.0fs.",
                    attempt,
                    self._MAX_RECONNECT_ATTEMPTS,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        # Passwort nach erfolgreichem Login aus dem Speicher entfernen
        self._password = ""

        # Loop-lebendig halten: run_coroutine_threadsafe-Tasks werden parallel ausgeführt
        await self._stop_event.wait()

        # Teardown
        if self._login is not None:
            try:
                await self._login.save_cookiefile()
            except Exception:
                pass

    async def _initialize_login(self) -> None:
        """Authentifiziert sich bei Amazon und findet das Zielgerät."""
        from alexapy import AlexaAPI, AlexaLogin

        log.info(
            "Alexa: Verbinde mit %s als %s ...",
            self._amazon_url,
            _hide_email(self._email),
        )
        # alexapy changed AlexaLogin constructor signatures across versions.
        # Build kwargs dynamically from the installed signature.
        init_params = inspect.signature(AlexaLogin.__init__).parameters
        kwargs = {
            "url": self._amazon_url,
            "email": self._email,
            "password": self._password,
        }
        if "debug" in init_params:
            kwargs["debug"] = False
        if "outputpath" in init_params:
            kwargs["outputpath"] = lambda name: os.path.join(_PROJECT_DIR, f"alexa_session_{name}")
        elif "outputfiles_prefix" in init_params:
            kwargs["outputfiles_prefix"] = "alexa_session_"

        self._login = AlexaLogin(**kwargs)

        await self._login.login()

        status = self._login.status or {}
        if not status.get("login_successful"):
            # Nur sichere Statusfelder loggen – Tokens/Cookies ausschließen.
            _safe_keys = {"login_successful", "error", "error_message", "captcha_required"}
            safe_status = {k: v for k, v in status.items() if k in _safe_keys}
            log.error(
                "Alexa-Login fehlgeschlagen (Status: %s). Prüfe Zugangsdaten in config_local.py.",
                safe_status,
            )
            return

        devices = await AlexaAPI.get_devices(self._login)
        for d in devices:
            if d.get("accountName", "").lower() == self._device_name.lower():
                self._device = d
                break

        if self._device is None:
            available = [d.get("accountName", "?") for d in devices]
            log.warning(
                "Alexa-Gerät '%s' nicht gefunden. Verfügbare Geräte: %s",
                self._device_name,
                available,
            )
            return

        self._ready = True
        log.info(
            "Alexa-Controller bereit: Gerät '%s' (S/N: %s...).",
            self._device.get("accountName"),
            str(self._device.get("serialNumber", "?"))[:6],
        )

    async def _handle_update(self, valence: float, arousal: float, emotion: str) -> None:
        """Reagiert auf ein Emotions-Update (läuft im async-Loop-Thread)."""
        if not self._ready or self._device is None:
            return

        now = time.time()

        with self._lock:
            current = self._current_mood

        new_mood = _va_to_mood(valence, arousal, current_mood=current)

        with self._lock:
            mood_changed = new_mood != self._current_mood
            cooldown_passed = (now - self._last_action_time) >= self._cooldown

        if not (mood_changed and cooldown_passed):
            return

        try:
            from alexapy import AlexaAPI

            api = AlexaAPI(self._device, self._login)

            if self._volume_control:
                vol = _va_to_volume(valence, arousal)
                await api.set_volume(vol)
                log.debug("Alexa: Lautstärke → %d (V=%.2f A=%.2f)", vol, valence, arousal)

            search_phrase = self._playlists.get(new_mood, "background music")
            await api.play_music(self._provider, search_phrase)

            log.info(
                "Alexa: Stimmung '%s' → '%s'  (V=%.2f A=%.2f Emotion=%s)",
                new_mood,
                search_phrase,
                valence,
                arousal,
                emotion,
            )

            with self._lock:
                self._current_mood = new_mood
                self._last_action_time = time.time()

        except Exception as exc:
            log.warning("Alexa-Aktion fehlgeschlagen: %s", exc)

    def check_absence(self, timeout_seconds: float = 180.0) -> None:
        """Prüft ob seit timeout_seconds kein Update kam und warnt einmalig.

        Vom Main-Loop aufgerufen wenn kein Gesicht erkannt wird.
        """
        with self._lock:
            if not self._ready or self._last_update_time == 0.0:
                return
            elapsed = time.time() - self._last_update_time
            already_warned = self._absence_warned

        if elapsed >= timeout_seconds and not already_warned:
            log.warning(
                "Alexa: Seit %.0fs kein Emotions-Update – Musik läuft ggf. unbeaufsichtigt weiter.",
                elapsed,
            )
            with self._lock:
                self._absence_warned = True
