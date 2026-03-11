"""Zentrale Konfiguration für das Emotion-Light-System."""

# === Hardware ===
WEBCAM_INDEX = 0  # USB-Webcam Device-Index (1920x1080 erkannt)
HUE_BRIDGE_IP = "192.168.178.20"  # IP der Philips Hue Bridge
HUE_LIGHT_IDS = [
    2,
    3,
    4,
    6,
]  # Mick Zimmer 1, Mick Zimmer 3, Mick Zimmer 2, Hue lightstrip 1

# === Capture ===
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 24
CAMERA_BUFFER_SIZE = 1  # Puffergröße der Kamera (1 = immer aktuellster Frame)
ANALYSIS_EVERY_N_FRAMES = 8  # Nur jeder n-te Frame → weniger CPU-Last
MIN_ANALYSIS_EVERY_N_FRAMES = 6  # Untere Grenze fuer adaptive Analysefrequenz
MAX_ANALYSIS_EVERY_N_FRAMES = 20  # Obere Grenze fuer adaptive Analysefrequenz
ADAPTIVE_ANALYSIS = True  # Analysefrequenz dynamisch an FPS anpassen
ADAPTIVE_FPS_LOW = 14.0  # Unterhalb: Analyse drosseln
ADAPTIVE_FPS_HIGH = 22.0  # Oberhalb: Analyse wieder erhoehen
MIN_RUNTIME_FPS = 10.0  # Zieluntergrenze im Livebetrieb
LOW_FPS_RECOVERY_HYSTERESIS = 2.0  # Guard erst bei MIN_RUNTIME_FPS + X wieder loesen
STARTUP_GUARD_DELAY_SECONDS = (
    4.0  # FPS-Regelung erst nach Initialisierungsphase aktivieren
)

# === DeepFace ===
DETECTOR_BACKEND = "opencv"  # opencv ist deutlich leichter und haelt FPS stabiler
MIN_CONFIDENCE = (
    0.55  # Mindest-Confidence fuer Emotion (von 0.45 erhoeht → weniger Rauschen)
)
SOFT_MIN_CONFIDENCE = (
    0.30  # Untere Grenze: darunter wird ein Frame als zu unsicher verworfen
)
LOW_CONFIDENCE_ALPHA_SCALE = (
    0.35  # Wie stark unsichere Frames (zwischen SOFT_MIN und MIN) EMA beeinflussen
)
ANALYSIS_FRAME_SIZE = 192  # Kleinere Analysebreite fuer stabilere FPS

# === Unsicherheitsbewertung / Guardrails ===
# Modellqualitaet aus Margin (Top1-Top2) und normierter Entropie.
UNCERTAINTY_MARGIN_WEIGHT = (
    0.60  # Hoeher => klare Klassenabstaende werden staerker belohnt
)
UNCERTAINTY_ENTROPY_WEIGHT = (
    0.40  # Hoeher => flache Verteilungen werden staerker abgestraft
)
LOW_QUALITY_THRESHOLD = 0.35  # Unterhalb wird konservativer Lichtausgang aktiviert
LOW_QUALITY_NEUTRAL_BLEND = 0.45  # Anteil neutral-safe Light bei LOW-Q-Guardrail
LOW_QUALITY_MIN_TRANSITION = 30  # Mindest-Transition (1/10s) bei LOW-Q-Guardrail

# === Beleuchtungsnormalisierung ===
USE_COLOR_CONSTANCY = (
    True  # Gray-World-Korrektur vor CLAHE (bricht Licht→Kamera-Feedback-Loop)
)
CLAHE_CLIP_LIMIT = (
    3.0  # CLAHE Contrast-Limit (0=aus). Normalisiert ungleichmaessige Beleuchtung.
)

# === Emotion Smoothing (EMA-basiert) ===
EMA_ALPHA = 0.15  # Basis-Gewicht neuer Messung (0.0–1.0). Wird mit Confidence skaliert.
# Effectives Alpha = EMA_ALPHA * confidence
EMA_MIN_WEIGHT = (
    0.05  # Emotionen unter 5% EMA-Gewicht werden beim Farbblending ignoriert
)
FALLBACK_DECAY = (
    0.08  # Wie schnell EMA-Vektor Richtung neutral driftet wenn kein Gesicht erkannt
)

# === Trend-Analyse ===
TREND_INFLUENCE = 0.3  # Gewicht des Emotions-Trends auf Transition-Zeit (0.0 = aus)

# === Mikro-Expressions-Burst ===
BURST_CONFIDENCE_DELTA = 0.25  # Schwelle: Abweichung der aktuellen Confidence vom Durchschnitt loest Burst aus
BURST_FRAMES = 3  # Anzahl aufeinanderfolgender Frames die im Burst analysiert werden

# === Hue Transition ===
TRANSITION_TIME = 20  # In 1/10 Sekunden → 2.0s Fade (sanfterer Uebergang)
FALLBACK_AFTER_SECONDS = 8  # Sekunden ohne Gesicht → Fallback
ABSENCE_LIGHT_OFF_SECONDS = (
    180  # Sekunden ohne Gesicht im Bild → Licht ausschalten (3 Minuten)
)
HUE_MIN_UPDATE_INTERVAL = 0.20  # Max. 5 Hue-Updates/s (entlastet Netzwerk + Main-Loop)
HUE_HUE_QUANT = 512  # Hue-Werte grob rastern, um Update-Flut zu vermeiden
HUE_BRI_QUANT = 8  # Helligkeit rastern
HUE_SAT_QUANT = 8  # Saettigung rastern

# === Valence-Arousal-Modell ===
USE_VALENCE_AROUSAL = (
    True  # True: Licht ueber Valence/Arousal steuern. False: direktes Emotion-Mapping.
)

VALENCE_AROUSAL_MAP = {
    "happy": {"valence": 0.9, "arousal": 0.5},
    "sad": {"valence": -0.8, "arousal": -0.7},
    "angry": {"valence": -0.9, "arousal": 1.0},
    "fear": {"valence": -0.7, "arousal": 0.8},
    "surprise": {"valence": 0.3, "arousal": 0.9},
    "disgust": {"valence": -0.6, "arousal": 0.2},
    "neutral": {"valence": 0.0, "arousal": 0.0},
}

# Valence/Arousal → Hue-Bereich (fuer Interpolation)
# Wissenschaftliche Korrektur: Blaulicht aktiviert Melanopsin → verstärkt Kortisol
# bei negativen Emotionen kontraproduktiv.  Warm-Amber (2200 K) beruhigt nachweislich.
VA_HUE_NEGATIVE = 9000  # Warmes Bernstein / Amber (statt Blau) bei negativer Valence
VA_HUE_NEUTRAL = 14000  # Warmweiss (neutrale Valence)
VA_HUE_POSITIVE = 14500  # Helles Warm-Weiss ~3500 K (Fokus-optimiert)
VA_BRI_LOW = 70  # Helligkeit bei niedrigem Arousal (ruhiger)
VA_BRI_HIGH = 240  # Helligkeit bei hohem Arousal
VA_SAT_LOW = 50  # Saettigung bei niedrigem Arousal (saubereres Amber)
VA_SAT_HIGH = 240  # Saettigung bei hohem Arousal

# === Emotion → Hue Mapping (Fallback wenn USE_VALENCE_AROUSAL = False) ===
# Hue: 0–65535 (Farbkreis), Bri: 1–254, Sat: 0–254
# Korrigiert: negative Emotionen verwenden Warm-Amber statt Blau
EMOTION_MAP = {
    "happy": {"hue": 14500, "bri": 200, "sat": 200},
    "sad": {"hue": 8500, "bri": 100, "sat": 120},
    "angry": {"hue": 65535, "bri": 220, "sat": 254},
    "fear": {"hue": 9000, "bri": 110, "sat": 140},
    "surprise": {"hue": 10000, "bri": 254, "sat": 230},
    "disgust": {"hue": 10500, "bri": 120, "sat": 160},
    "neutral": {"hue": 14000, "bri": 160, "sat": 120},
}

# Fallback wenn kein Gesicht erkannt wird
FALLBACK_LIGHT = {"hue": 14000, "bri": 140, "sat": 80}

# === Audio-Emotion ===
USE_AUDIO = True  # Audio-Emotionserkennung aktivieren
AUDIO_DEVICE_INDEX = None  # None = System-Standard-Mikrofon
AUDIO_SAMPLE_RATE = 16000  # Abtastrate fuer Audio
AUDIO_CHUNK_SECONDS = 2.0  # Laenge eines Audio-Chunks fuer Analyse
AUDIO_INFERENCE_COOLDOWN = 1.0  # Pause nach jeder Audio-Inferenz (entlastet CPU/FPS)
AUDIO_EMA_ALPHA = 0.12  # EMA-Alpha fuer Audio (etwas traeger als Video)
AUDIO_WEIGHT = 0.35  # Fusion: 35% Audio, 65% Video
AUDIO_SNR_DB_FLOOR = 0.0  # Unterhalb davon gilt Audio als stark verrauscht
AUDIO_SNR_DB_CEIL = 20.0  # Ab hier gilt Audio als robust/sauber
AUDIO_DYNAMIC_MIN_FACTOR = (
    0.20  # Minimaler Anteil vom AUDIO_WEIGHT bei schlechter Qualitaet
)
AUDIO_DYNAMIC_QUALITY_EXPONENT = 1.2  # >1 bestraft mittlere Qualitaet etwas staerker

# === Koerpersprache (MediaPipe Pose) ===
USE_POSE = True  # Pose-Analyse aktivieren
POSE_WEIGHT = 0.2  # Gewicht des Pose-Arousal-Offsets
POSE_FRAME_SIZE = 256  # Pose-Eingangsbreite (kleiner = schneller)

# === Face Mesh + Action Units ===
USE_FACE_MESH = True  # MediaPipe Face Mesh fuer Action-Unit-basierte Emotionserkennung
FACE_MESH_WEIGHT = 0.15  # Gewicht der Face-Mesh-Erkennung in der Fusion (0.0–1.0)
FACE_MESH_FRAME_SIZE = 256  # Eingabebreite fuer Face Mesh
HEAD_POSE_PENALTY_YAW = 30.0  # Ab diesem Yaw (Grad) wird Confidence reduziert
HEAD_POSE_PENALTY_PITCH = 25.0  # Ab diesem Pitch (Grad) wird Confidence reduziert
HEAD_POSE_MAX_ATTENUATION = 0.5  # Minimaler Confidence-Faktor bei extremer Kopfdrehung
USE_HEAD_POSE_CONFIDENCE = True  # Kopfpose-Faktor auf Video-Confidence anwenden
HEAD_POSE_CONFIDENCE_STRENGTH = 0.35  # 0.0=aus, 1.0=voller Faktor

# === HRV via rPPG ===
USE_HRV = True  # Herzfrequenz- und HRV-Messung via Webcam aktivieren
HRV_WINDOW_SECONDS = 30.0  # Rollendes Zeitfenster für HRV-Berechnung in Sekunden
HRV_FRAME_INTERVAL = 3  # Alle N Frames einen Frame an den HRV-Analyzer schicken
HRV_AROUSAL_INFLUENCE = 0.15  # Gewicht der HR auf den Arousal-Offset (0.0 = aus)
# Formel: offset = (hr_bpm - 72) / 72 * HRV_AROUSAL_INFLUENCE
HRV_MIN_CONFIDENCE = 0.45  # Mindest-Confidence fuer HRV-Offset im Licht
HRV_BASELINE_BPM = 72.0  # Start-Baseline fuer Ruhepuls
HRV_BASELINE_ADAPT_ALPHA = 0.015  # Langsame Nachfuehrung der HR-Baseline im Betrieb
HRV_OFFSET_EMA_ALPHA = 0.18  # Glaettung des HRV-Arousal-Offsets
HRV_OFFSET_CLAMP = 0.18  # Harte Begrenzung des HRV-Offsets (symmetrisch)

# === Atemfrequenz via Webcam ===
USE_BREATHING = True  # Atemfrequenz-Erkennung via Schulterbewegung aktivieren
BREATHING_WINDOW_SECONDS = (
    30.0  # Rollendes Zeitfenster in Sekunden (mind. 2–3 Atemzyklen)
)
BREATHING_FRAME_INTERVAL = (
    6  # Alle N Frames einen Frame schicken (24fps → ~4 Hz Abtastrate)
)
BREATHING_AROUSAL_INFLUENCE = 0.12  # Gewicht der Atemfrequenz auf Arousal (0.0 = aus)
# Formel: offset = (br_bpm - 15) / 15 * BREATHING_AROUSAL_INFLUENCE
# Langsames Atmen → −Arousal (ruhigeres Licht), schnelles → +Arousal
BREATHING_MIN_CONFIDENCE = 0.5  # Mindest-Confidence fuer Atmungssteuerung
BREATHING_BASELINE_SECONDS = 90  # Dauer der initialen Ruhe-Baseline in Sekunden
BREATHING_BASELINE_ADAPT_ALPHA = 0.02  # Langsame Nachfuehrung der Baseline im Betrieb
BREATHING_OFFSET_EMA_ALPHA = 0.15  # Glaettung fuer Atem-Arousal-Offset
BREATHING_OFFSET_CLAMP = 0.2  # Harte Begrenzung des Atem-Offsets (symmetrisch)
BREATHING_TRANSITION_INFLUENCE = (
    0.8  # Positives Offset beschleunigt, negatives verlangsamt Transition
)

# === Adaptive Regulation ===
# Das Licht steuert NICHT mehr nur den erkannten Ist-Zustand, sondern nudgt ihn adaptiv
# Richtung eines therapeutischen Zielzustands (positiv + ruhig).
ADAPTIVE_REGULATION = (
    True  # True: adaptiver Modus aktiv; False: klassischer Spiegel-Modus
)
ADAPTIVE_TARGET_VALENCE = 0.65  # Ziel-Valence (Fokus: positiv, aber nicht euphorisch)
ADAPTIVE_TARGET_AROUSAL = 0.35  # Ziel-Arousal (leicht erhoeht fuer Konzentration)
ADAPTIVE_BLEND_STRENGTH = 0.45  # Startstärke: 45% Richtung Ziel, 55% Ist-Zustand
ADAPTIVE_BLEND_MAX = 0.80  # Obere Grenze der Blend-Stärke (Eskalation)
ADAPTIVE_PROGRESS_TIMEOUT = (
    30.0  # Sekunden ohne Verbesserung → blend um ESCALATION erhöhen
)
ADAPTIVE_BLEND_ESCALATION = 0.10  # Schrittweite je Eskalationsstufe
ADAPTIVE_AT_TARGET_THRESHOLD = (
    0.18  # Distanz zum Ziel < Schwelle → "Stabil", kein Eingriff
)

# === Nutzer-Kalibrierung ===
CALIBRATION_FILE = "calibration_default.json"  # Pfad zur Kalibrierungsdatei
CALIBRATION_SECONDS_PER_EMOTION = 10  # Sekunden pro Emotion bei Kalibrierung

# === Zirkadianes Licht-Modell ===
USE_CIRCADIAN = True  # Tageszeit-adaptives Zielprofil aktivieren
CIRCADIAN_UPDATE_INTERVAL = 300  # Sekunden zwischen Zirkadian-Updates (5 Minuten)

# === Atemfuehrungs-Entrainment ===
USE_BREATHING_PACER = True  # Licht-Pulsation zur Atemfuehrung aktivieren
BREATHING_PACER_BPM = 6.0  # Ziel-Atemfrequenz (0.1 Hz Resonanzfrequenz)
BREATHING_PACER_AMPLITUDE = 0.08  # ±8% Helligkeits-Modulation
BREATHING_PACER_FADE_IN = 30.0  # Sekunden zum langsamen Einblenden
BREATHING_PACER_BR_THRESHOLD = 18.0  # Pacer aktiviert wenn BR > dieser Wert

# === Multi-Licht-Szenen-Komposition ===
# Rollenbasierte Lichtsteuerung: verschiedene Lichter erhalten angepasste Parameter
HUE_LIGHT_ROLES = {
    2: "primary",  # Mick Zimmer 1 – Hauptfarbe (unveraendert)
    3: "accent",  # Mick Zimmer 3 – Akzent (waermer, leicht gedaempfte Saettigung)
    4: "accent",  # Mick Zimmer 2 – Akzent
    6: "ambient",  # Hue Lightstrip – Ambient (deutlich waermer, gedaempft)
}

# === Vorausschauende Intervention ===
PREDICTIVE_TREND_THRESHOLD = -0.04  # Valence-Trend-Schwelle fuer Abwaertserkennung
PREDICTIVE_TRIGGER_SECONDS = 4.0  # Sekunden konstanter Abwaertstrend bis Eingriff
PREDICTIVE_BOOST_FACTOR = 1.5  # Blend-Verstaerkungsfaktor
PREDICTIVE_BOOST_DURATION = 15.0  # Dauer des verstaerkten Blend in Sekunden

# === Pupillen- und Blink-Analyse (aus FaceMesh) ===
USE_PUPIL_BLINK = True  # Pupillengroesse und Blink-Rate aus Iris-Landmarks
PUPIL_AROUSAL_INFLUENCE = 0.10  # Gewicht der Pupillengroesse auf Arousal-Offset
PUPIL_DILATION_BASELINE = 0.45  # Ruhe-Baseline der relativen Pupillengroesse
PUPIL_OFFSET_CLAMP = 0.12  # Harte Begrenzung des Pupillen-Offsets
BLINK_RATE_FATIGUE_THRESHOLD = 25.0  # Blinks/Min oberhalb = Muedigkeitsindikator
BLINK_RATE_FOCUS_THRESHOLD = 10.0  # Blinks/Min unterhalb = Fokus-Indikator
BLINK_VALENCE_INFLUENCE = 0.05  # Wie stark Blink-Rate Valence beeinflusst

# === Erweiterte Koerpersprache-Analyse ===
USE_EXTENDED_POSE = True  # Erweiterte Pose-Signale nutzen
TORSO_LEAN_AROUSAL_INFLUENCE = 0.08  # Vorneigung → Arousal (+vorne = +Arousal)
SHOULDER_DROP_VALENCE_INFLUENCE = 0.06  # Schulterabsenkung → Valence (-)
HEAD_TILT_RELAXATION_THRESHOLD = 0.15  # Ab dieser Seitneigung: Entspannungsindikator

# === Tastatur/Maus-Aktivitaets-Monitoring ===
USE_ACTIVITY_MONITOR = True  # Passives Input-Monitoring aktivieren
ACTIVITY_AROUSAL_INFLUENCE = 0.10  # Wie stark kognitive Last auf Arousal wirkt
ACTIVITY_TRANSITION_INFLUENCE = 0.4  # Hohe Aktivitaet → schnellere Licht-Transition

# === Prosodische Stimm-Analyse ===
USE_PROSODIC = True  # Prosodische Merkmale aus Audio extrahieren
PROSODIC_PITCH_STRESS_HZ = 200.0  # Mittlere F0 ueber diesem Wert = erhoehter Stress
PROSODIC_PITCH_CALM_HZ = 120.0  # Mittlere F0 unter diesem Wert = ruhiger Zustand
PROSODIC_AROUSAL_INFLUENCE = 0.08  # Gewicht prosodischer Merkmale auf Arousal
PROSODIC_SPEECH_RATE_HIGH = 6.0  # Onsets/s ueber diesem Wert = schnelles Sprechen
PROSODIC_SPEECH_RATE_LOW = 2.0  # Onsets/s unter diesem Wert = langsames Sprechen

# === Agentic Face Fine-Tune ===
USE_FACE_FINETUNE_ONNX = True
FACE_FINETUNE_ONNX_PATH = "artifacts/face_finetune/face_finetuned.onnx"

# === Alexa-Steuerung (optional) ===
# Steuert Amazon Echo basierend auf erkannter Emotion:
# Musik passend zur Stimmung abspielen und Lautstaerke anpassen.
# Zugangsdaten NICHT hier eintragen – stattdessen in config_local.py (nicht versioniert).
USE_ALEXA = False  # True: Alexa-Steuerung aktivieren
ALEXA_EMAIL = ""  # Amazon-Konto E-Mail (in config_local.py setzen)
ALEXA_PASSWORD = ""  # Amazon-Konto Passwort (in config_local.py setzen)
ALEXA_DEVICE_NAME = ""  # Geraetename exakt wie in der Alexa-App
ALEXA_AMAZON_URL = "amazon.de"  # Laender-Suffix (amazon.de, amazon.com, ...)
ALEXA_COOLDOWN_SECONDS = 30.0  # Mindestabstand zwischen Alexa-Aktionen (Sekunden)
ALEXA_MUSIC_PROVIDER = "AMAZON_MUSIC"  # AMAZON_MUSIC | TUNEIN | SPOTIFY
ALEXA_VOLUME_CONTROL = True  # Lautstaerke ebenfalls emotion-adaptiv anpassen

# Stimmungskategorie → Amazon-Music-Suchanfrage
# Schluessel: energetic_positive | calm_positive | calm_negative |
#             energetic_negative | neutral
ALEXA_MOOD_PLAYLISTS = {
    "energetic_positive": "upbeat happy pop music",
    "calm_positive": "focus concentration background music",
    "calm_negative": "calming relaxing ambient music",
    "energetic_negative": "stress relief calming music",
    "neutral": "lo-fi background music",
}


# === Lokale Overrides (optional, nicht versioniert) ===
# Lege bei Bedarf eine `config_local.py` im Projektroot an, um sensible lokale
# Werte wie Bridge-IP/Lampenrollen zu ueberschreiben, ohne `config.py` zu aendern.
try:
    import config_local as _config_local  # type: ignore

    _LOCAL_KEYS = (
        "HUE_BRIDGE_IP",
        "HUE_LIGHT_IDS",
        "HUE_LIGHT_ROLES",
        "USE_ALEXA",
        "ALEXA_EMAIL",
        "ALEXA_PASSWORD",
        "ALEXA_DEVICE_NAME",
        "ALEXA_AMAZON_URL",
        "ALEXA_COOLDOWN_SECONDS",
        "ALEXA_MUSIC_PROVIDER",
        "ALEXA_VOLUME_CONTROL",
        "ALEXA_MOOD_PLAYLISTS",
    )
    for _name in _LOCAL_KEYS:
        if hasattr(_config_local, _name):
            globals()[_name] = getattr(_config_local, _name)
except Exception:
    pass
