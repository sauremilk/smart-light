"""Zentrale Konfiguration fuer das Emotion-Light-System."""

from __future__ import annotations

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Validierte Konfiguration -- Werte per Umgebungsvariable ueberschreibbar.

    Beispiel: ``SL_MIN_CONFIDENCE=0.6 python main.py``
    """

    model_config = SettingsConfigDict(
        env_prefix="SL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === Kanonische Emotionsliste (Single Source of Truth) ===
    EMOTIONS: tuple[str, ...] = (
        "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
    )

    # === Hardware ===
    WEBCAM_INDEX: int = 0
    HUE_BRIDGE_IP: str = "192.168.178.20"
    HUE_LIGHT_IDS: list[int] = [2, 3, 4, 6]

    # === Capture ===
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    TARGET_FPS: int = 24
    CAMERA_BUFFER_SIZE: int = 1
    ANALYSIS_EVERY_N_FRAMES: int = 8
    MIN_ANALYSIS_EVERY_N_FRAMES: int = 6
    MAX_ANALYSIS_EVERY_N_FRAMES: int = 20
    ADAPTIVE_ANALYSIS: bool = True
    ADAPTIVE_FPS_LOW: float = 14.0
    ADAPTIVE_FPS_HIGH: float = 22.0
    MIN_RUNTIME_FPS: float = 10.0
    LOW_FPS_RECOVERY_HYSTERESIS: float = 2.0
    STARTUP_GUARD_DELAY_SECONDS: float = 4.0

    # === DeepFace ===
    DETECTOR_BACKEND: str = "opencv"
    MIN_CONFIDENCE: float = 0.55
    SOFT_MIN_CONFIDENCE: float = 0.30
    LOW_CONFIDENCE_ALPHA_SCALE: float = 0.35
    ANALYSIS_FRAME_SIZE: int = 192

    # === Unsicherheitsbewertung / Guardrails ===
    UNCERTAINTY_MARGIN_WEIGHT: float = 0.60
    UNCERTAINTY_ENTROPY_WEIGHT: float = 0.40
    LOW_QUALITY_THRESHOLD: float = 0.35
    LOW_QUALITY_NEUTRAL_BLEND: float = 0.45
    LOW_QUALITY_MIN_TRANSITION: int = 30

    # === Beleuchtungsnormalisierung ===
    USE_COLOR_CONSTANCY: bool = True
    CLAHE_CLIP_LIMIT: float = 3.0

    # === Emotion Smoothing (EMA-basiert) ===
    EMA_ALPHA: float = 0.15
    EMA_MIN_WEIGHT: float = 0.05
    FALLBACK_DECAY: float = 0.08

    # === Trend-Analyse ===
    TREND_INFLUENCE: float = 0.3

    # === Mikro-Expressions-Burst ===
    BURST_CONFIDENCE_DELTA: float = 0.25
    BURST_FRAMES: int = 3

    # === Hue Transition ===
    TRANSITION_TIME: int = 20
    FALLBACK_AFTER_SECONDS: int = 8
    ABSENCE_LIGHT_OFF_SECONDS: int = 180
    HUE_MIN_UPDATE_INTERVAL: float = 0.20
    HUE_HUE_QUANT: int = 512
    HUE_BRI_QUANT: int = 8
    HUE_SAT_QUANT: int = 8
    HUE_CT_QUANT: int = 4

    # === Valence-Arousal-Modell ===
    USE_VALENCE_AROUSAL: bool = True

    VALENCE_AROUSAL_MAP: dict[str, dict[str, float]] = {
        "happy": {"valence": 0.9, "arousal": 0.5},
        "sad": {"valence": -0.8, "arousal": -0.7},
        "angry": {"valence": -0.9, "arousal": 1.0},
        "fear": {"valence": -0.7, "arousal": 0.8},
        "surprise": {"valence": 0.3, "arousal": 0.9},
        "disgust": {"valence": -0.6, "arousal": 0.2},
        "neutral": {"valence": 0.0, "arousal": 0.0},
    }

    VA_HUE_NEGATIVE: int = 9000
    VA_HUE_NEUTRAL: int = 14000
    VA_HUE_POSITIVE: int = 14500
    VA_BRI_LOW: int = 70
    VA_BRI_HIGH: int = 240
    VA_SAT_LOW: int = 50
    VA_SAT_HIGH: int = 240

    VA_CT_NEGATIVE: int = 450
    VA_CT_NEUTRAL: int = 333
    VA_CT_POSITIVE: int = 222
    VA_CT_AROUSAL_SHIFT: int = 30

    # === Emotion -> Hue Mapping (Fallback) ===
    EMOTION_MAP: dict[str, dict[str, int]] = {
        "happy": {"hue": 14500, "bri": 200, "sat": 200},
        "sad": {"hue": 8500, "bri": 100, "sat": 120},
        "angry": {"hue": 65535, "bri": 220, "sat": 254},
        "fear": {"hue": 9000, "bri": 110, "sat": 140},
        "surprise": {"hue": 10000, "bri": 254, "sat": 230},
        "disgust": {"hue": 10500, "bri": 120, "sat": 160},
        "neutral": {"hue": 14000, "bri": 160, "sat": 120},
    }

    FALLBACK_LIGHT: dict[str, int] = {"hue": 14000, "bri": 140, "sat": 80, "ct": 333}

    # === Audio-Emotion ===
    USE_AUDIO: bool = True
    AUDIO_DEVICE_INDEX: int | None = None
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHUNK_SECONDS: float = 2.0
    AUDIO_INFERENCE_COOLDOWN: float = 1.0
    AUDIO_EMA_ALPHA: float = 0.12
    AUDIO_WEIGHT: float = 0.35
    AUDIO_SNR_DB_FLOOR: float = 0.0
    AUDIO_SNR_DB_CEIL: float = 20.0
    AUDIO_DYNAMIC_MIN_FACTOR: float = 0.20
    AUDIO_DYNAMIC_QUALITY_EXPONENT: float = 1.2

    # === Koerpersprache (MediaPipe Pose) ===
    USE_POSE: bool = True
    POSE_WEIGHT: float = 0.2
    POSE_FRAME_SIZE: int = 256

    # === Face Mesh + Action Units ===
    USE_FACE_MESH: bool = True
    FACE_MESH_WEIGHT: float = 0.15
    FACE_MESH_FRAME_SIZE: int = 256
    HEAD_POSE_PENALTY_YAW: float = 30.0
    HEAD_POSE_PENALTY_PITCH: float = 25.0
    HEAD_POSE_MAX_ATTENUATION: float = 0.5
    USE_HEAD_POSE_CONFIDENCE: bool = True
    HEAD_POSE_CONFIDENCE_STRENGTH: float = 0.35

    # === HRV via rPPG ===
    USE_HRV: bool = True
    HRV_WINDOW_SECONDS: float = 30.0
    HRV_FRAME_INTERVAL: int = 3
    HRV_AROUSAL_INFLUENCE: float = 0.15
    HRV_MIN_CONFIDENCE: float = 0.45
    HRV_BASELINE_BPM: float = 72.0
    HRV_BASELINE_ADAPT_ALPHA: float = 0.015
    HRV_OFFSET_EMA_ALPHA: float = 0.18
    HRV_OFFSET_CLAMP: float = 0.18

    # === Atemfrequenz via Webcam ===
    USE_BREATHING: bool = True
    BREATHING_WINDOW_SECONDS: float = 30.0
    BREATHING_FRAME_INTERVAL: int = 6
    BREATHING_AROUSAL_INFLUENCE: float = 0.12
    BREATHING_MIN_CONFIDENCE: float = 0.5
    BREATHING_BASELINE_SECONDS: int = 90
    BREATHING_BASELINE_ADAPT_ALPHA: float = 0.02
    BREATHING_OFFSET_EMA_ALPHA: float = 0.15
    BREATHING_OFFSET_CLAMP: float = 0.2
    BREATHING_TRANSITION_INFLUENCE: float = 0.8

    # === Adaptive Regulation ===
    ADAPTIVE_REGULATION: bool = True
    ADAPTIVE_TARGET_VALENCE: float = 0.65
    ADAPTIVE_TARGET_AROUSAL: float = 0.35
    ADAPTIVE_BLEND_STRENGTH: float = 0.45
    ADAPTIVE_BLEND_MAX: float = 0.80
    ADAPTIVE_PROGRESS_TIMEOUT: float = 30.0
    ADAPTIVE_BLEND_ESCALATION: float = 0.10
    ADAPTIVE_AT_TARGET_THRESHOLD: float = 0.18

    # === Nutzer-Kalibrierung ===
    CALIBRATION_FILE: str = "calibration_default.json"
    CALIBRATION_SECONDS_PER_EMOTION: int = 10

    # === Zirkadianes Licht-Modell ===
    USE_CIRCADIAN: bool = True
    CIRCADIAN_UPDATE_INTERVAL: int = 300

    # === Atemfuehrungs-Entrainment ===
    USE_BREATHING_PACER: bool = True
    BREATHING_PACER_BPM: float = 6.0
    BREATHING_PACER_AMPLITUDE: float = 0.08
    BREATHING_PACER_FADE_IN: float = 30.0
    BREATHING_PACER_BR_THRESHOLD: float = 18.0

    # === Multi-Licht-Szenen-Komposition ===
    HUE_LIGHT_ROLES: dict[int, str] = {
        2: "primary",
        3: "accent",
        4: "accent",
        6: "ambient",
    }

    # === Vorausschauende Intervention ===
    PREDICTIVE_TREND_THRESHOLD: float = -0.04
    PREDICTIVE_TRIGGER_SECONDS: float = 4.0
    PREDICTIVE_BOOST_FACTOR: float = 1.5
    PREDICTIVE_BOOST_DURATION: float = 15.0

    # === Pupillen- und Blink-Analyse ===
    USE_PUPIL_BLINK: bool = True
    PUPIL_AROUSAL_INFLUENCE: float = 0.10
    PUPIL_DILATION_BASELINE: float = 0.45
    PUPIL_OFFSET_CLAMP: float = 0.12
    BLINK_RATE_FATIGUE_THRESHOLD: float = 25.0
    BLINK_RATE_FOCUS_THRESHOLD: float = 10.0
    BLINK_VALENCE_INFLUENCE: float = 0.05

    # === Erweiterte Koerpersprache-Analyse ===
    USE_EXTENDED_POSE: bool = True
    TORSO_LEAN_AROUSAL_INFLUENCE: float = 0.08
    SHOULDER_DROP_VALENCE_INFLUENCE: float = 0.06
    HEAD_TILT_RELAXATION_THRESHOLD: float = 0.15

    # === Tastatur/Maus-Aktivitaets-Monitoring ===
    USE_ACTIVITY_MONITOR: bool = True
    ACTIVITY_AROUSAL_INFLUENCE: float = 0.10
    ACTIVITY_TRANSITION_INFLUENCE: float = 0.4

    # === Prosodische Stimm-Analyse ===
    USE_PROSODIC: bool = True
    PROSODIC_PITCH_STRESS_HZ: float = 200.0
    PROSODIC_PITCH_CALM_HZ: float = 120.0
    PROSODIC_AROUSAL_INFLUENCE: float = 0.08
    PROSODIC_SPEECH_RATE_HIGH: float = 6.0
    PROSODIC_SPEECH_RATE_LOW: float = 2.0

    # === Kognitiver Zustandsklassifikator ===
    USE_COGNITIVE_CLASSIFIER: bool = True
    COGNITIVE_STABILITY_WINDOW: float = 30.0

    # === Modus-System ===
    USE_MODE_SYSTEM: bool = True
    DEFAULT_MODE: str = "AUTO"
    MODE_HYSTERESIS_S: float = 15.0

    # === Pausen-Manager ===
    USE_BREAK_MANAGER: bool = True
    BREAK_MAX_WORK_MINUTES: float = 50.0
    BREAK_FATIGUE_TRIGGER_S: float = 120.0
    BREAK_MIN_MINUTES: float = 5.0
    BREAK_POMODORO_ENABLED: bool = False
    BREAK_POMODORO_WORK_MINUTES: float = 25.0
    BREAK_POMODORO_BREAK_MINUTES: float = 5.0
    BREAK_POMODORO_LONG_BREAK_MINUTES: float = 15.0
    BREAK_POMODORO_LONG_BREAK_AFTER: int = 4

    # === Feedback-System ===
    USE_FEEDBACK: bool = True
    FEEDBACK_COOLDOWN_S: float = 3.0

    # === Agentic Face Fine-Tune ===
    USE_FACE_FINETUNE_ONNX: bool = True
    FACE_FINETUNE_ONNX_PATH: str = "artifacts/face_finetune/face_finetuned.onnx"

    # === Alexa-Steuerung ===
    USE_ALEXA: bool = False
    ALEXA_EMAIL: str = ""
    ALEXA_PASSWORD: str = ""
    ALEXA_DEVICE_NAME: str = ""
    ALEXA_AMAZON_URL: str = "amazon.de"
    ALEXA_COOLDOWN_SECONDS: float = 30.0
    ALEXA_MUSIC_PROVIDER: str = "AMAZON_MUSIC"
    ALEXA_VOLUME_CONTROL: bool = True
    ALEXA_MOOD_PLAYLISTS: dict[str, str] = {
        "energetic_positive": "upbeat happy pop music",
        "calm_positive": "focus concentration background music",
        "calm_negative": "calming relaxing ambient music",
        "energetic_negative": "stress relief calming music",
        "neutral": "lo-fi background music",
    }

    # --- Validatoren ---

    @field_validator("EMA_ALPHA", "AUDIO_EMA_ALPHA")
    @classmethod
    def _alpha_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"Alpha muss zwischen 0 und 1 liegen, ist {v}")
        return v

    @field_validator("MIN_CONFIDENCE", "SOFT_MIN_CONFIDENCE")
    @classmethod
    def _confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence muss zwischen 0 und 1 liegen, ist {v}")
        return v

    @model_validator(mode="after")
    def _soft_lt_min(self) -> "AppSettings":
        if self.SOFT_MIN_CONFIDENCE >= self.MIN_CONFIDENCE:
            raise ValueError(
                f"SOFT_MIN_CONFIDENCE ({self.SOFT_MIN_CONFIDENCE}) "
                f"muss kleiner als MIN_CONFIDENCE ({self.MIN_CONFIDENCE}) sein"
            )
        return self


# -- Singleton-Instanz --------------------------------------------------------
settings = AppSettings()


# -- Rueckwaertskompatible Module-Level-Konstanten ----------------------------
# Alle bestehenden ``from config import X``-Stellen bleiben unveraendert.

EMOTIONS = settings.EMOTIONS
WEBCAM_INDEX = settings.WEBCAM_INDEX
HUE_BRIDGE_IP = settings.HUE_BRIDGE_IP
HUE_LIGHT_IDS = settings.HUE_LIGHT_IDS
FRAME_WIDTH = settings.FRAME_WIDTH
FRAME_HEIGHT = settings.FRAME_HEIGHT
TARGET_FPS = settings.TARGET_FPS
CAMERA_BUFFER_SIZE = settings.CAMERA_BUFFER_SIZE
ANALYSIS_EVERY_N_FRAMES = settings.ANALYSIS_EVERY_N_FRAMES
MIN_ANALYSIS_EVERY_N_FRAMES = settings.MIN_ANALYSIS_EVERY_N_FRAMES
MAX_ANALYSIS_EVERY_N_FRAMES = settings.MAX_ANALYSIS_EVERY_N_FRAMES
ADAPTIVE_ANALYSIS = settings.ADAPTIVE_ANALYSIS
ADAPTIVE_FPS_LOW = settings.ADAPTIVE_FPS_LOW
ADAPTIVE_FPS_HIGH = settings.ADAPTIVE_FPS_HIGH
MIN_RUNTIME_FPS = settings.MIN_RUNTIME_FPS
LOW_FPS_RECOVERY_HYSTERESIS = settings.LOW_FPS_RECOVERY_HYSTERESIS
STARTUP_GUARD_DELAY_SECONDS = settings.STARTUP_GUARD_DELAY_SECONDS
DETECTOR_BACKEND = settings.DETECTOR_BACKEND
MIN_CONFIDENCE = settings.MIN_CONFIDENCE
SOFT_MIN_CONFIDENCE = settings.SOFT_MIN_CONFIDENCE
LOW_CONFIDENCE_ALPHA_SCALE = settings.LOW_CONFIDENCE_ALPHA_SCALE
ANALYSIS_FRAME_SIZE = settings.ANALYSIS_FRAME_SIZE
UNCERTAINTY_MARGIN_WEIGHT = settings.UNCERTAINTY_MARGIN_WEIGHT
UNCERTAINTY_ENTROPY_WEIGHT = settings.UNCERTAINTY_ENTROPY_WEIGHT
LOW_QUALITY_THRESHOLD = settings.LOW_QUALITY_THRESHOLD
LOW_QUALITY_NEUTRAL_BLEND = settings.LOW_QUALITY_NEUTRAL_BLEND
LOW_QUALITY_MIN_TRANSITION = settings.LOW_QUALITY_MIN_TRANSITION
USE_COLOR_CONSTANCY = settings.USE_COLOR_CONSTANCY
CLAHE_CLIP_LIMIT = settings.CLAHE_CLIP_LIMIT
EMA_ALPHA = settings.EMA_ALPHA
EMA_MIN_WEIGHT = settings.EMA_MIN_WEIGHT
FALLBACK_DECAY = settings.FALLBACK_DECAY
TREND_INFLUENCE = settings.TREND_INFLUENCE
BURST_CONFIDENCE_DELTA = settings.BURST_CONFIDENCE_DELTA
BURST_FRAMES = settings.BURST_FRAMES
TRANSITION_TIME = settings.TRANSITION_TIME
FALLBACK_AFTER_SECONDS = settings.FALLBACK_AFTER_SECONDS
ABSENCE_LIGHT_OFF_SECONDS = settings.ABSENCE_LIGHT_OFF_SECONDS
HUE_MIN_UPDATE_INTERVAL = settings.HUE_MIN_UPDATE_INTERVAL
HUE_HUE_QUANT = settings.HUE_HUE_QUANT
HUE_BRI_QUANT = settings.HUE_BRI_QUANT
HUE_SAT_QUANT = settings.HUE_SAT_QUANT
HUE_CT_QUANT = settings.HUE_CT_QUANT
USE_VALENCE_AROUSAL = settings.USE_VALENCE_AROUSAL
VALENCE_AROUSAL_MAP = settings.VALENCE_AROUSAL_MAP
VA_HUE_NEGATIVE = settings.VA_HUE_NEGATIVE
VA_HUE_NEUTRAL = settings.VA_HUE_NEUTRAL
VA_HUE_POSITIVE = settings.VA_HUE_POSITIVE
VA_BRI_LOW = settings.VA_BRI_LOW
VA_BRI_HIGH = settings.VA_BRI_HIGH
VA_SAT_LOW = settings.VA_SAT_LOW
VA_SAT_HIGH = settings.VA_SAT_HIGH
VA_CT_NEGATIVE = settings.VA_CT_NEGATIVE
VA_CT_NEUTRAL = settings.VA_CT_NEUTRAL
VA_CT_POSITIVE = settings.VA_CT_POSITIVE
VA_CT_AROUSAL_SHIFT = settings.VA_CT_AROUSAL_SHIFT
EMOTION_MAP = settings.EMOTION_MAP
FALLBACK_LIGHT = settings.FALLBACK_LIGHT
USE_AUDIO = settings.USE_AUDIO
AUDIO_DEVICE_INDEX = settings.AUDIO_DEVICE_INDEX
AUDIO_SAMPLE_RATE = settings.AUDIO_SAMPLE_RATE
AUDIO_CHUNK_SECONDS = settings.AUDIO_CHUNK_SECONDS
AUDIO_INFERENCE_COOLDOWN = settings.AUDIO_INFERENCE_COOLDOWN
AUDIO_EMA_ALPHA = settings.AUDIO_EMA_ALPHA
AUDIO_WEIGHT = settings.AUDIO_WEIGHT
AUDIO_SNR_DB_FLOOR = settings.AUDIO_SNR_DB_FLOOR
AUDIO_SNR_DB_CEIL = settings.AUDIO_SNR_DB_CEIL
AUDIO_DYNAMIC_MIN_FACTOR = settings.AUDIO_DYNAMIC_MIN_FACTOR
AUDIO_DYNAMIC_QUALITY_EXPONENT = settings.AUDIO_DYNAMIC_QUALITY_EXPONENT
USE_POSE = settings.USE_POSE
POSE_WEIGHT = settings.POSE_WEIGHT
POSE_FRAME_SIZE = settings.POSE_FRAME_SIZE
USE_FACE_MESH = settings.USE_FACE_MESH
FACE_MESH_WEIGHT = settings.FACE_MESH_WEIGHT
FACE_MESH_FRAME_SIZE = settings.FACE_MESH_FRAME_SIZE
HEAD_POSE_PENALTY_YAW = settings.HEAD_POSE_PENALTY_YAW
HEAD_POSE_PENALTY_PITCH = settings.HEAD_POSE_PENALTY_PITCH
HEAD_POSE_MAX_ATTENUATION = settings.HEAD_POSE_MAX_ATTENUATION
USE_HEAD_POSE_CONFIDENCE = settings.USE_HEAD_POSE_CONFIDENCE
HEAD_POSE_CONFIDENCE_STRENGTH = settings.HEAD_POSE_CONFIDENCE_STRENGTH
USE_HRV = settings.USE_HRV
HRV_WINDOW_SECONDS = settings.HRV_WINDOW_SECONDS
HRV_FRAME_INTERVAL = settings.HRV_FRAME_INTERVAL
HRV_AROUSAL_INFLUENCE = settings.HRV_AROUSAL_INFLUENCE
HRV_MIN_CONFIDENCE = settings.HRV_MIN_CONFIDENCE
HRV_BASELINE_BPM = settings.HRV_BASELINE_BPM
HRV_BASELINE_ADAPT_ALPHA = settings.HRV_BASELINE_ADAPT_ALPHA
HRV_OFFSET_EMA_ALPHA = settings.HRV_OFFSET_EMA_ALPHA
HRV_OFFSET_CLAMP = settings.HRV_OFFSET_CLAMP
USE_BREATHING = settings.USE_BREATHING
BREATHING_WINDOW_SECONDS = settings.BREATHING_WINDOW_SECONDS
BREATHING_FRAME_INTERVAL = settings.BREATHING_FRAME_INTERVAL
BREATHING_AROUSAL_INFLUENCE = settings.BREATHING_AROUSAL_INFLUENCE
BREATHING_MIN_CONFIDENCE = settings.BREATHING_MIN_CONFIDENCE
BREATHING_BASELINE_SECONDS = settings.BREATHING_BASELINE_SECONDS
BREATHING_BASELINE_ADAPT_ALPHA = settings.BREATHING_BASELINE_ADAPT_ALPHA
BREATHING_OFFSET_EMA_ALPHA = settings.BREATHING_OFFSET_EMA_ALPHA
BREATHING_OFFSET_CLAMP = settings.BREATHING_OFFSET_CLAMP
BREATHING_TRANSITION_INFLUENCE = settings.BREATHING_TRANSITION_INFLUENCE
ADAPTIVE_REGULATION = settings.ADAPTIVE_REGULATION
ADAPTIVE_TARGET_VALENCE = settings.ADAPTIVE_TARGET_VALENCE
ADAPTIVE_TARGET_AROUSAL = settings.ADAPTIVE_TARGET_AROUSAL
ADAPTIVE_BLEND_STRENGTH = settings.ADAPTIVE_BLEND_STRENGTH
ADAPTIVE_BLEND_MAX = settings.ADAPTIVE_BLEND_MAX
ADAPTIVE_PROGRESS_TIMEOUT = settings.ADAPTIVE_PROGRESS_TIMEOUT
ADAPTIVE_BLEND_ESCALATION = settings.ADAPTIVE_BLEND_ESCALATION
ADAPTIVE_AT_TARGET_THRESHOLD = settings.ADAPTIVE_AT_TARGET_THRESHOLD
CALIBRATION_FILE = settings.CALIBRATION_FILE
CALIBRATION_SECONDS_PER_EMOTION = settings.CALIBRATION_SECONDS_PER_EMOTION
USE_CIRCADIAN = settings.USE_CIRCADIAN
CIRCADIAN_UPDATE_INTERVAL = settings.CIRCADIAN_UPDATE_INTERVAL
USE_BREATHING_PACER = settings.USE_BREATHING_PACER
BREATHING_PACER_BPM = settings.BREATHING_PACER_BPM
BREATHING_PACER_AMPLITUDE = settings.BREATHING_PACER_AMPLITUDE
BREATHING_PACER_FADE_IN = settings.BREATHING_PACER_FADE_IN
BREATHING_PACER_BR_THRESHOLD = settings.BREATHING_PACER_BR_THRESHOLD
HUE_LIGHT_ROLES = settings.HUE_LIGHT_ROLES
PREDICTIVE_TREND_THRESHOLD = settings.PREDICTIVE_TREND_THRESHOLD
PREDICTIVE_TRIGGER_SECONDS = settings.PREDICTIVE_TRIGGER_SECONDS
PREDICTIVE_BOOST_FACTOR = settings.PREDICTIVE_BOOST_FACTOR
PREDICTIVE_BOOST_DURATION = settings.PREDICTIVE_BOOST_DURATION
USE_PUPIL_BLINK = settings.USE_PUPIL_BLINK
PUPIL_AROUSAL_INFLUENCE = settings.PUPIL_AROUSAL_INFLUENCE
PUPIL_DILATION_BASELINE = settings.PUPIL_DILATION_BASELINE
PUPIL_OFFSET_CLAMP = settings.PUPIL_OFFSET_CLAMP
BLINK_RATE_FATIGUE_THRESHOLD = settings.BLINK_RATE_FATIGUE_THRESHOLD
BLINK_RATE_FOCUS_THRESHOLD = settings.BLINK_RATE_FOCUS_THRESHOLD
BLINK_VALENCE_INFLUENCE = settings.BLINK_VALENCE_INFLUENCE
USE_EXTENDED_POSE = settings.USE_EXTENDED_POSE
TORSO_LEAN_AROUSAL_INFLUENCE = settings.TORSO_LEAN_AROUSAL_INFLUENCE
SHOULDER_DROP_VALENCE_INFLUENCE = settings.SHOULDER_DROP_VALENCE_INFLUENCE
HEAD_TILT_RELAXATION_THRESHOLD = settings.HEAD_TILT_RELAXATION_THRESHOLD
USE_ACTIVITY_MONITOR = settings.USE_ACTIVITY_MONITOR
ACTIVITY_AROUSAL_INFLUENCE = settings.ACTIVITY_AROUSAL_INFLUENCE
ACTIVITY_TRANSITION_INFLUENCE = settings.ACTIVITY_TRANSITION_INFLUENCE
USE_PROSODIC = settings.USE_PROSODIC
PROSODIC_PITCH_STRESS_HZ = settings.PROSODIC_PITCH_STRESS_HZ
PROSODIC_PITCH_CALM_HZ = settings.PROSODIC_PITCH_CALM_HZ
PROSODIC_AROUSAL_INFLUENCE = settings.PROSODIC_AROUSAL_INFLUENCE
PROSODIC_SPEECH_RATE_HIGH = settings.PROSODIC_SPEECH_RATE_HIGH
PROSODIC_SPEECH_RATE_LOW = settings.PROSODIC_SPEECH_RATE_LOW
USE_COGNITIVE_CLASSIFIER = settings.USE_COGNITIVE_CLASSIFIER
COGNITIVE_STABILITY_WINDOW = settings.COGNITIVE_STABILITY_WINDOW
USE_MODE_SYSTEM = settings.USE_MODE_SYSTEM
DEFAULT_MODE = settings.DEFAULT_MODE
MODE_HYSTERESIS_S = settings.MODE_HYSTERESIS_S
USE_BREAK_MANAGER = settings.USE_BREAK_MANAGER
BREAK_MAX_WORK_MINUTES = settings.BREAK_MAX_WORK_MINUTES
BREAK_FATIGUE_TRIGGER_S = settings.BREAK_FATIGUE_TRIGGER_S
BREAK_MIN_MINUTES = settings.BREAK_MIN_MINUTES
BREAK_POMODORO_ENABLED = settings.BREAK_POMODORO_ENABLED
BREAK_POMODORO_WORK_MINUTES = settings.BREAK_POMODORO_WORK_MINUTES
BREAK_POMODORO_BREAK_MINUTES = settings.BREAK_POMODORO_BREAK_MINUTES
BREAK_POMODORO_LONG_BREAK_MINUTES = settings.BREAK_POMODORO_LONG_BREAK_MINUTES
BREAK_POMODORO_LONG_BREAK_AFTER = settings.BREAK_POMODORO_LONG_BREAK_AFTER
USE_FEEDBACK = settings.USE_FEEDBACK
FEEDBACK_COOLDOWN_S = settings.FEEDBACK_COOLDOWN_S
USE_FACE_FINETUNE_ONNX = settings.USE_FACE_FINETUNE_ONNX
FACE_FINETUNE_ONNX_PATH = settings.FACE_FINETUNE_ONNX_PATH
USE_ALEXA = settings.USE_ALEXA
ALEXA_EMAIL = settings.ALEXA_EMAIL
ALEXA_PASSWORD = settings.ALEXA_PASSWORD
ALEXA_DEVICE_NAME = settings.ALEXA_DEVICE_NAME
ALEXA_AMAZON_URL = settings.ALEXA_AMAZON_URL
ALEXA_COOLDOWN_SECONDS = settings.ALEXA_COOLDOWN_SECONDS
ALEXA_MUSIC_PROVIDER = settings.ALEXA_MUSIC_PROVIDER
ALEXA_VOLUME_CONTROL = settings.ALEXA_VOLUME_CONTROL
ALEXA_MOOD_PLAYLISTS = settings.ALEXA_MOOD_PLAYLISTS


# === Lokale Overrides (optional, nicht versioniert) ===
# config_local.py kann weiterhin sensible Werte ueberschreiben.
# Die Overrides werden auf die Singleton-Instanz UND die Module-Level-Konstanten angewandt.
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
try:
    import config_local as _config_local  # type: ignore

    _overrides: dict = {}
    for _name in _LOCAL_KEYS:
        if hasattr(_config_local, _name):
            _overrides[_name] = getattr(_config_local, _name)
    if _overrides:
        settings = settings.model_copy(update=_overrides)
        # Module-Level-Konstanten ebenfalls aktualisieren
        import sys as _sys
        _this = _sys.modules[__name__]
        for _k, _v in _overrides.items():
            setattr(_this, _k, _v)
except Exception:
    pass
