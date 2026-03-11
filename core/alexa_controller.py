"""Alexa-Controller: steuert Amazon Echo-Geräte basierend auf erkannter Emotion.

Nutzt die inoffizielle alexapy-Bibliothek (dieselbe die Home Assistant verwendet).
Läuft vollständig asynchron in einem Daemon-Thread – der Main-Loop wird nicht blockiert.

Setup (einmalig):
    1. pip install alexapy
    2. Amazon-Konto-Daten in config_local.py eintragen:
           USE_ALEXA = True
           ALEXA_EMAIL = "deine@email.de"
           ALEXA_PASSWORD = "deinPasswort"
           ALEXA_DEVICE_NAME = "Michs Echo"  # exakter Name aus der Alexa-App
    3. Beim ersten Start: Browser öffnen falls 2-FA verlangt wird
       (alexapy speichert den Cookie danach als alexa_session_cookies.pickle).
"""

import asyncio
import inspect
import logging
import os
import threading
import time

log = logging.getLogger("emotion-light.alexa")


# ---------------------------------------------------------------------------
# Stimmungs-Mapping
# ---------------------------------------------------------------------------


def _va_to_mood(valence: float, arousal: float) -> str:
    """Ordnet Valence/Arousal einer von fünf Stimmungskategorien zu."""
    if valence >= 0.25:
        return "energetic_positive" if arousal >= 0.15 else "calm_positive"
    if valence < -0.20:
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
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="alexa-loop"
        )
        self._thread.start()

    def update(self, valence: float, arousal: float, emotion: str) -> None:
        """Thread-safe: reicht ein Emotions-Update in den async-Loop ein.

        Wird vom Main-Loop aufgerufen. Kehrt sofort zurück.
        """
        with self._lock:
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
        asyncio.set_event_loop(loop)
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

        try:
            await self._initialize_login()
        except Exception as exc:
            log.error("Alexa-Initialisierung fehlgeschlagen: %s", exc)
            return

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
            # Store alexa session artifacts locally in the project directory.
            kwargs["outputpath"] = lambda name: os.path.join(
                os.getcwd(), f"alexa_session_{name}"
            )
        elif "outputfiles_prefix" in init_params:
            kwargs["outputfiles_prefix"] = "alexa_session_"

        self._login = AlexaLogin(**kwargs)

        await self._login.login()

        status = self._login.status or {}
        if not status.get("login_successful"):
            log.error(
                "Alexa-Login fehlgeschlagen (Status: %s). "
                "Prüfe Zugangsdaten in config_local.py.",
                {k: v for k, v in status.items() if k != "cookies"},
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

    async def _handle_update(
        self, valence: float, arousal: float, emotion: str
    ) -> None:
        """Reagiert auf ein Emotions-Update (läuft im async-Loop-Thread)."""
        if not self._ready or self._device is None:
            return

        now = time.time()
        new_mood = _va_to_mood(valence, arousal)

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
                log.debug(
                    "Alexa: Lautstärke → %d (V=%.2f A=%.2f)", vol, valence, arousal
                )

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
