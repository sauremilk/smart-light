"""Modus-System fuer zielgerichtete Leistungsoptimierung.

Jeder Modus definiert:
  - V/A-Regulationsziele (ueberschreiben Circadian/Default)
  - Licht-Parameter-Overrides (Hue-Bereiche, max Helligkeit)
  - Empfohlene Konfigurationsanpassungen (z.B. Breathing-Pacer an/aus)

Verfuegbare Modi:
  FOCUS    – Konzentration maximieren (kuehles Licht, moderate Helligkeit)
  ENERGY   – Energie/Aktivitaet steigern (helles, kuehles Licht)
  RELAX    – Entspannung/Erholung (warmes, gedaempftes Licht)
  RECOVERY – Pausenmodus nach Muedigkeit (sehr warm, minimal)
  AUTO     – Automatische Modusauswahl basierend auf kognitivem Zustand
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModeProfile:
    """Immutable Profil fuer einen Optimierungsmodus."""

    name: str
    label: str  # Menschenlesbar (deutsch)
    target_v: float  # Regulations-Ziel Valence
    target_a: float  # Regulations-Ziel Arousal
    hue_negative: int  # Hue fuer negative Valence
    hue_positive: int  # Hue fuer positive Valence
    bri_max: int  # Maximale Helligkeit
    pacer_active: bool  # Atem-Pacer empfohlen
    blend_strength: float  # Regulationsstaerke (hoeher = aggressiver)


# ── Vordefinierte Modi ──

MODE_FOCUS = ModeProfile(
    name="FOCUS",
    label="Fokus",
    target_v=0.65,
    target_a=0.35,
    hue_negative=9000,
    hue_positive=11000,  # Kuehler als normal
    bri_max=230,
    pacer_active=False,
    blend_strength=0.50,
)

MODE_ENERGY = ModeProfile(
    name="ENERGY",
    label="Energie",
    target_v=0.70,
    target_a=0.60,  # Hoehere Arousal
    hue_negative=10000,
    hue_positive=13000,
    bri_max=254,  # Maximale Helligkeit
    pacer_active=False,
    blend_strength=0.55,
)

MODE_RELAX = ModeProfile(
    name="RELAX",
    label="Entspannung",
    target_v=0.60,
    target_a=-0.10,  # Niedrige Arousal
    hue_negative=7500,
    hue_positive=10000,  # Sehr warm
    bri_max=160,  # Gedaempft
    pacer_active=True,  # Atemfuehrung aktiv
    blend_strength=0.40,
)

MODE_RECOVERY = ModeProfile(
    name="RECOVERY",
    label="Erholung",
    target_v=0.55,
    target_a=-0.25,  # Sehr niedrige Arousal
    hue_negative=7000,
    hue_positive=9000,  # Sehr warm/amber
    bri_max=120,  # Minimal
    pacer_active=True,
    blend_strength=0.35,
)

# Lookup-Tabelle
MODES: dict[str, ModeProfile] = {
    "FOCUS": MODE_FOCUS,
    "ENERGY": MODE_ENERGY,
    "RELAX": MODE_RELAX,
    "RECOVERY": MODE_RECOVERY,
}


class ModeManager:
    """Verwaltet den aktiven Optimierungsmodus.

    Im AUTO-Modus waehlt der Manager basierend auf dem kognitiven Zustand
    automatisch den passenden Modus.
    """

    # Mapping: kognitiver Zustand → empfohlener Modus
    _AUTO_MAP = {
        "FOCUS": "FOCUS",
        "FLOW": "FOCUS",  # Im Flow bleiben wir bei Focus-Licht
        "FATIGUE": "RECOVERY",
        "STRESS": "RELAX",
        "NEUTRAL": "FOCUS",  # Default bei unklarem Zustand
    }

    def __init__(self, initial_mode: str = "AUTO"):
        self._manual_mode: str | None = None if initial_mode == "AUTO" else initial_mode
        self._auto_mode: str = "FOCUS"
        self._mode_since: float = 0.0

        # Hysterese: Modus erst wechseln nach mind. N Sekunden stabilen Zustands
        self._pending_mode: str | None = None
        self._pending_since: float = 0.0
        self._hysteresis_s: float = 15.0

    @property
    def is_auto(self) -> bool:
        return self._manual_mode is None

    @property
    def active_mode(self) -> str:
        return self._manual_mode if self._manual_mode is not None else self._auto_mode

    @property
    def active_profile(self) -> ModeProfile:
        return MODES.get(self.active_mode, MODE_FOCUS)

    def set_mode(self, mode: str) -> None:
        """Setzt den Modus manuell. 'AUTO' wechselt zu automatischer Auswahl."""
        if mode == "AUTO":
            self._manual_mode = None
        elif mode in MODES:
            self._manual_mode = mode

    def update_auto(self, cognitive_state: str, now: float) -> str:
        """Aktualisiert den Auto-Modus basierend auf kognitivem Zustand.

        Returns den alten Modus-Namen (fuer Wechsel-Erkennung).
        """
        if self._manual_mode is not None:
            return self.active_mode

        suggested = self._AUTO_MAP.get(cognitive_state, "FOCUS")

        if suggested == self._auto_mode:
            # Stabiler Zustand, Pending zuruecksetzen
            self._pending_mode = None
            return self._auto_mode

        # Neuer Vorschlag: Hysterese starten oder pruefen
        if suggested != self._pending_mode:
            self._pending_mode = suggested
            self._pending_since = now
            return self._auto_mode

        # Pending-Modus lang genug stabil?
        if (now - self._pending_since) >= self._hysteresis_s:
            old = self._auto_mode
            self._auto_mode = suggested
            self._mode_since = now
            self._pending_mode = None
            return old

        return self._auto_mode

    def cycle_mode(self) -> str:
        """Wechselt zum naechsten Modus (fuer Hotkey-Steuerung).

        Reihenfolge: AUTO → FOCUS → ENERGY → RELAX → RECOVERY → AUTO
        """
        cycle = ["AUTO", "FOCUS", "ENERGY", "RELAX", "RECOVERY"]
        current = "AUTO" if self._manual_mode is None else self._manual_mode
        idx = cycle.index(current) if current in cycle else 0
        next_mode = cycle[(idx + 1) % len(cycle)]
        self.set_mode(next_mode)
        return next_mode
