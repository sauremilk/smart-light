#!/usr/bin/env python3
"""
Emotion-gesteuerte Philips Hue Lampe – Hauptprogramm.
Erkennt Emotionen via Webcam (+ optional Audio, Pose) und steuert Hue-Licht in Echtzeit.
Alle Verarbeitung erfolgt lokal – keine Cloud-APIs.
"""

import argparse
import io
import json
import logging
import os
import sys
import time
import warnings
from collections import deque

import cv2
import numpy as np


def _configure_third_party_runtime_logs():
    """Reduziert bekannte, nicht-kritische Laufzeitwarnungen von Drittanbieter-Libs."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")

    # TensorFlow-deprecation noise in this stack is known and not actionable at runtime.
    warnings.filterwarnings(
        "ignore",
        message=r".*sparse_softmax_cross_entropy.*",
    )

    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass

    noisy_substrings = (
        "sparse_softmax_cross_entropy is deprecated",
        "inference_feedback_manager.cc:121",
        "landmark_projection_calculator.cc:81",
        "landmark_projection_calculator.cc",
        "Using NORM_RECT without IMAGE_DIMENSIONS",
    )

    class _FilteredStderr(io.TextIOBase):
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def write(self, s):
            if any(n in s for n in noisy_substrings):
                return len(s)
            return self._wrapped.write(s)

        def flush(self):
            return self._wrapped.flush()

    if not isinstance(sys.stderr, _FilteredStderr):
        sys.stderr = _FilteredStderr(sys.stderr)


_configure_third_party_runtime_logs()

from analyzers.audio_quality import effective_audio_weight
from analyzers.breathing_analyzer import BR_REST_BPM
from config import (
    ABSENCE_LIGHT_OFF_SECONDS,
    ADAPTIVE_ANALYSIS,
    ADAPTIVE_AT_TARGET_THRESHOLD,
    ADAPTIVE_BLEND_ESCALATION,
    ADAPTIVE_BLEND_MAX,
    ADAPTIVE_BLEND_STRENGTH,
    ADAPTIVE_FPS_HIGH,
    ADAPTIVE_FPS_LOW,
    ADAPTIVE_PROGRESS_TIMEOUT,
    ADAPTIVE_REGULATION,
    ADAPTIVE_TARGET_AROUSAL,
    ADAPTIVE_TARGET_VALENCE,
    ALEXA_AMAZON_URL,
    ALEXA_COOLDOWN_SECONDS,
    ALEXA_DEVICE_NAME,
    ALEXA_EMAIL,
    ALEXA_MOOD_PLAYLISTS,
    ALEXA_MUSIC_PROVIDER,
    ALEXA_PASSWORD,
    ALEXA_VOLUME_CONTROL,
    ANALYSIS_EVERY_N_FRAMES,
    ANALYSIS_FRAME_SIZE,
    AUDIO_DYNAMIC_MIN_FACTOR,
    AUDIO_DYNAMIC_QUALITY_EXPONENT,
    AUDIO_WEIGHT,
    BREAK_FATIGUE_TRIGGER_S,
    BREAK_MAX_WORK_MINUTES,
    BREAK_MIN_MINUTES,
    BREAK_POMODORO_BREAK_MINUTES,
    BREAK_POMODORO_ENABLED,
    BREAK_POMODORO_LONG_BREAK_AFTER,
    BREAK_POMODORO_LONG_BREAK_MINUTES,
    BREAK_POMODORO_WORK_MINUTES,
    BREATHING_AROUSAL_INFLUENCE,
    BREATHING_BASELINE_ADAPT_ALPHA,
    BREATHING_BASELINE_SECONDS,
    BREATHING_FRAME_INTERVAL,
    BREATHING_MIN_CONFIDENCE,
    BREATHING_OFFSET_CLAMP,
    BREATHING_OFFSET_EMA_ALPHA,
    BREATHING_PACER_AMPLITUDE,
    BREATHING_PACER_BPM,
    BREATHING_PACER_BR_THRESHOLD,
    BREATHING_PACER_FADE_IN,
    BREATHING_WINDOW_SECONDS,
    CALIBRATION_FILE,
    CAMERA_BUFFER_SIZE,
    CIRCADIAN_UPDATE_INTERVAL,
    COGNITIVE_STABILITY_WINDOW,
    DEFAULT_MODE,
    DETECTOR_BACKEND,
    FACE_MESH_FRAME_SIZE,
    FACE_MESH_WEIGHT,
    FALLBACK_LIGHT,
    FEEDBACK_COOLDOWN_S,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HEAD_POSE_CONFIDENCE_STRENGTH,
    HRV_AROUSAL_INFLUENCE,
    HRV_BASELINE_ADAPT_ALPHA,
    HRV_BASELINE_BPM,
    HRV_FRAME_INTERVAL,
    HRV_MIN_CONFIDENCE,
    HRV_OFFSET_CLAMP,
    HRV_OFFSET_EMA_ALPHA,
    HRV_WINDOW_SECONDS,
    HUE_BRIDGE_IP,
    HUE_LIGHT_IDS,
    HUE_LIGHT_ROLES,
    LOW_FPS_RECOVERY_HYSTERESIS,
    LOW_QUALITY_MIN_TRANSITION,
    LOW_QUALITY_NEUTRAL_BLEND,
    LOW_QUALITY_THRESHOLD,
    MAX_ANALYSIS_EVERY_N_FRAMES,
    MIN_ANALYSIS_EVERY_N_FRAMES,
    MIN_RUNTIME_FPS,
    MODE_HYSTERESIS_S,
    POSE_FRAME_SIZE,
    PREDICTIVE_BOOST_DURATION,
    PREDICTIVE_BOOST_FACTOR,
    PREDICTIVE_TREND_THRESHOLD,
    PREDICTIVE_TRIGGER_SECONDS,
    STARTUP_GUARD_DELAY_SECONDS,
    TARGET_FPS,
    USE_ACTIVITY_MONITOR,
    USE_ALEXA,
    USE_AUDIO,
    USE_BREAK_MANAGER,
    USE_BREATHING,
    USE_BREATHING_PACER,
    USE_CIRCADIAN,
    USE_COGNITIVE_CLASSIFIER,
    USE_EXTENDED_POSE,
    USE_FACE_MESH,
    USE_FEEDBACK,
    USE_HEAD_POSE_CONFIDENCE,
    USE_HRV,
    USE_MODE_SYSTEM,
    USE_POSE,
    USE_PROSODIC,
    USE_PUPIL_BLINK,
    USE_VALENCE_AROUSAL,
    VA_BRI_HIGH,
    VA_CT_NEGATIVE,
    VA_CT_POSITIVE,
    VA_HUE_NEGATIVE,
    VA_HUE_POSITIVE,
    WEBCAM_INDEX,
)
from core.break_manager import BreakManager
from core.circadian import CircadianSchedule
from core.cognitive_state import CognitiveClassifier
from core.emotion_regulator import EmotionRegulator
from core.error_taxonomy import (
    ACTIVITY_ANALYZER_INIT_FAILED,
    AUDIO_ANALYZER_INIT_FAILED,
    BREATHING_ANALYZER_INIT_FAILED,
    CALIBRATION_LOAD_FAILED,
    DEEPFACE_WARMUP_FAILED,
    FACEMESH_ANALYZER_INIT_FAILED,
    HRV_ANALYZER_INIT_FAILED,
    HUE_CONNECT_FAILED,
    HUE_REENABLE_FAILED,
    POSE_ANALYZER_INIT_FAILED,
)
from core.feedback import FeedbackCollector
from core.light_mapping import (
    BreathingPacer,
    blend_emotion_colors,
    compute_va_from_ema,
    fuse_modalities,
    valence_arousal_to_light,
)
from core.mode_manager import ModeManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _configure_library_loggers() -> None:
    """Drosselt bekannte Drittanbieter-Logger fuer einen sauberen Runtime-Output."""
    noisy_loggers = (
        "speechbrain",
        "huggingface_hub",
        "transformers",
        "torch",
        "urllib3",
        "phue",
    )
    for name in noisy_loggers:
        try:
            logging.getLogger(name).setLevel(logging.WARNING)
        except Exception:
            pass


_configure_library_loggers()


class _ThirdPartyNoiseFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "Wav2Vec2Model is frozen." not in msg


for _handler in logging.getLogger().handlers:
    _handler.addFilter(_ThirdPartyNoiseFilter())


log = logging.getLogger("emotion-light")


# ─── Extracted modules ────────────────────────────────────────
from analyzers.emotion_analyzer import (  # noqa: E402
    _ONNX_MODEL,
    EmotionAnalyzer,
    analyze_emotion_frame,
)
from core.capture import FrameSource  # noqa: E402
from core.fusion import (  # noqa: E402
    apply_modality_offsets,
    circadian_va_to_light,
    compute_transition,
)
from core.hue_controller import HueController, MockBridgeController  # noqa: E402
from core.preprocessing import resize_for_width  # noqa: E402
from core.session_log import (  # noqa: E402
    append_session_log,
    build_session_payload,
)
from core.telemetry import ERR_TELEMETRY  # noqa: E402

# ───────────── Kalibrierung laden ──────────────────────────────


def load_calibration(path: str) -> dict:
    """Laedt Kalibrierungs-Offsets aus JSON. Gibt leeres Dict zurueck bei Fehler."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log.info("Kalibrierung geladen: %s", path)
        return data
    except Exception as exc:
        ERR_TELEMETRY.record(
            component="calibration",
            code=CALIBRATION_LOAD_FAILED,
            detail=f"could not load calibration from {path}",
            exc=exc,
            level=logging.WARNING,
            cooldown_s=15.0,
        )
        return {}


# ──────────────────────── Main Loop ────────────────────────────


def _parse_args():
    parser = argparse.ArgumentParser(description="Emotion-gesteuerte Hue-Lampe")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock-Modus: Webcam und Hue-Bridge simuliert",
    )
    parser.add_argument(
        "--bridge-ip",
        default=None,
        metavar="IP",
        help="IP der Hue-Bridge (ueberschreibt config.py)",
    )
    parser.add_argument(
        "--light-ids",
        default=None,
        metavar="IDs",
        help="Kommagetrennte Lampen-IDs, z.B. 2,3,4,6",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Audio-Emotionserkennung deaktivieren",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Koerpersprache-Analyse deaktivieren",
    )
    parser.add_argument(
        "--no-face-mesh",
        action="store_true",
        help="Face-Mesh / Action-Unit-Analyse deaktivieren",
    )
    parser.add_argument(
        "--no-head-pose-penalty",
        action="store_true",
        help="Kopfpose-Confidence-Abschwaechung deaktivieren",
    )
    parser.add_argument(
        "--no-hrv",
        action="store_true",
        help="HRV/Herzfrequenz-Messung via rPPG deaktivieren",
    )
    parser.add_argument(
        "--no-breathing",
        action="store_true",
        help="Atemfrequenz-Erkennung via Schulterbewegung deaktivieren",
    )
    parser.add_argument(
        "--no-activity",
        action="store_true",
        help="Tastatur/Maus-Aktivitaets-Monitoring deaktivieren",
    )
    parser.add_argument(
        "--no-pupil-blink",
        action="store_true",
        help="Pupillen-/Blink-Analyse deaktivieren",
    )
    parser.add_argument(
        "--no-prosodic",
        action="store_true",
        help="Prosodische Stimm-Analyse deaktivieren",
    )
    parser.add_argument(
        "--no-extended-pose",
        action="store_true",
        help="Erweiterte Koerpersprache-Signale deaktivieren",
    )
    parser.add_argument(
        "--mode",
        choices=["AUTO", "FOCUS", "ENERGY", "RELAX", "RECOVERY"],
        default=None,
        help="Optimierungsmodus setzen (ueberschreibt config.py DEFAULT_MODE)",
    )
    parser.add_argument(
        "--no-cognitive",
        action="store_true",
        help="Kognitiven Zustandsklassifikator deaktivieren",
    )
    parser.add_argument(
        "--no-breaks",
        action="store_true",
        help="Pausen-Manager deaktivieren",
    )
    parser.add_argument(
        "--pomodoro",
        action="store_true",
        help="Pomodoro-Modus aktivieren (25/5-Zyklen)",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Benutzer-Feedback-System deaktivieren",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Kalibrierungsmodus starten",
    )
    parser.add_argument(
        "--calibration-file",
        default=None,
        metavar="PATH",
        help="Pfad zur Kalibrierungsdatei (ueberschreibt config.py)",
    )
    parser.add_argument(
        "--session-log",
        default=None,
        metavar="PATH",
        help="Optionaler Pfad fuer JSONL-Session-Log (Evaluation adaptiv vs. Kontrolle)",
    )
    parser.add_argument(
        "--condition",
        choices=["adaptive", "control"],
        default=None,
        help="Bedingung fuer A/B-Studie. Default: aus ADAPTIVE_REGULATION abgeleitet.",
    )
    parser.add_argument(
        "--participant",
        default=None,
        metavar="ID",
        help="Optionale Teilnehmer-ID fuer Evaluations-Logs",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        metavar="ID",
        help="Optionale Session-ID fuer Evaluations-Logs",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Deaktiviert OpenCV-Fenster/Keyboard-UI (fuer headless Umgebungen).",
    )
    parser.add_argument(
        "--no-alexa",
        action="store_true",
        help="Alexa-Steuerung deaktivieren (auch wenn USE_ALEXA=True in config).",
    )
    parser.add_argument(
        "--pseudonymize-session",
        action="store_true",
        help="Pseudonymisiert participant/session_id im Session-Log (gesalzener Hash).",
    )
    return parser.parse_args()


# ─── Overlay (ausgelagert in core/overlay.py) ──────────────────────────
from core.overlay import (  # noqa: E402
    _blend_light_params,
    _draw_overlay,
)


def main():
    args = _parse_args()

    # --- Kalibrierungsmodus ---
    if args.calibrate:
        from core.calibration import run_calibration

        cal_path = args.calibration_file or CALIBRATION_FILE
        run_calibration(cal_path)
        return

    bridge_ip = args.bridge_ip or HUE_BRIDGE_IP
    if not args.mock and not bridge_ip:
        sys.exit(
            "FEHLER: Keine Hue-Bridge-IP konfiguriert.\n"
            "  Setze HUE_BRIDGE_IP in config_local.py oder nutze --bridge-ip <ip>.\n"
            "  Vorlage: copy config_local.example.py config_local.py"
        )
    if args.light_ids:
        light_ids = [int(x) for x in args.light_ids.split(",")]
    else:
        light_ids = HUE_LIGHT_IDS

    # --- Kalibrierung laden ---
    cal_path = args.calibration_file or CALIBRATION_FILE
    calibration = load_calibration(cal_path)

    # --- Webcam initialisieren ---
    try:
        source = FrameSource(
            mock=args.mock,
            webcam_index=WEBCAM_INDEX,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            target_fps=TARGET_FPS,
            buffer_size=CAMERA_BUFFER_SIZE,
        )
    except RuntimeError as e:
        log.error("%s", e)
        sys.exit(1)

    # --- Hue Bridge verbinden ---
    if args.mock:
        hue = MockBridgeController()
    else:
        try:
            hue = HueController(bridge_ip, light_ids, light_roles=HUE_LIGHT_ROLES)
        except Exception as e:
            ERR_TELEMETRY.record(
                component="startup",
                code=HUE_CONNECT_FAILED,
                detail=f"bridge_ip={bridge_ip}",
                exc=e,
                level=logging.ERROR,
                cooldown_s=0.0,
            )
            log.error("Tipp: Bridge-Button drücken und erneut starten.")
            source.release()
            sys.exit(1)

    # --- Emotion-Analyse starten ---
    analyzer = EmotionAnalyzer(calibration=calibration)
    analyzer.start()
    backend_name = "onnx" if _ONNX_MODEL is not None else DETECTOR_BACKEND
    log.info(
        "Emotion-Analyse gestartet (Backend: %s). Druecke 'q' zum Beenden.",
        backend_name,
    )

    # --- Audio-Analyse (optional) ---
    audio_analyzer = None
    use_audio = USE_AUDIO and not args.no_audio
    if use_audio:
        try:
            from analyzers.audio_analyzer import AudioEmotionAnalyzer

            audio_analyzer = AudioEmotionAnalyzer()
            audio_analyzer.start()
            log.info("Audio-Emotionserkennung gestartet.")
        except Exception as exc:
            ERR_TELEMETRY.record(
                component="startup",
                code=AUDIO_ANALYZER_INIT_FAILED,
                detail="Audio analyzer disabled",
                exc=exc,
                level=logging.WARNING,
                cooldown_s=0.0,
            )
            audio_analyzer = None

    # --- Pose-Analyse (optional) ---
    pose_analyzer = None
    use_pose = USE_POSE and not args.no_pose
    if use_pose:
        try:
            from analyzers.pose_analyzer import PoseEmotionAnalyzer

            pose_analyzer = PoseEmotionAnalyzer()
            pose_analyzer.start()
            log.info("Pose-Analyse gestartet.")
        except Exception as exc:
            ERR_TELEMETRY.record(
                component="startup",
                code=POSE_ANALYZER_INIT_FAILED,
                detail="Pose analyzer disabled",
                exc=exc,
                level=logging.WARNING,
                cooldown_s=0.0,
            )
            pose_analyzer = None

    # --- Face Mesh + Action Units (optional) ---
    face_mesh_analyzer = None
    use_face_mesh = USE_FACE_MESH and not args.no_face_mesh
    if use_face_mesh:
        try:
            from analyzers.face_mesh_analyzer import FaceMeshAnalyzer

            face_mesh_analyzer = FaceMeshAnalyzer()
            face_mesh_analyzer.start()
            log.info("Face-Mesh-Analyse gestartet (AUs + Kopfpose).")
        except Exception as exc:
            ERR_TELEMETRY.record(
                component="startup",
                code=FACEMESH_ANALYZER_INIT_FAILED,
                detail="Face mesh analyzer disabled",
                exc=exc,
                level=logging.WARNING,
                cooldown_s=0.0,
            )
            face_mesh_analyzer = None

    # --- HRV-Analyse via rPPG (optional) ---
    hrv_analyzer = None
    use_hrv = USE_HRV and not args.no_hrv
    if use_hrv:
        try:
            from analyzers.hrv_analyzer import HRVAnalyzer

            hrv_analyzer = HRVAnalyzer(
                window_seconds=HRV_WINDOW_SECONDS,
                target_fps=TARGET_FPS,
            )
            hrv_analyzer.start()
            log.info("HRV-Analyse (rPPG) gestartet (Fenster: %.0fs).", HRV_WINDOW_SECONDS)
        except Exception as exc:
            ERR_TELEMETRY.record(
                component="startup",
                code=HRV_ANALYZER_INIT_FAILED,
                detail="HRV analyzer disabled",
                exc=exc,
                level=logging.WARNING,
                cooldown_s=0.0,
            )
            hrv_analyzer = None

    # --- Atemfrequenz-Erkennung via Schulterbewegung (optional) ---
    breathing_analyzer = None
    use_breathing = USE_BREATHING and not args.no_breathing
    if use_breathing:
        try:
            from analyzers.breathing_analyzer import BreathingAnalyzer

            breathing_analyzer = BreathingAnalyzer(
                window_seconds=BREATHING_WINDOW_SECONDS,
                target_fps=TARGET_FPS / BREATHING_FRAME_INTERVAL,
            )
            breathing_analyzer.start()
            log.info(
                "Atemfrequenz-Erkennung gestartet (Fenster: %.0fs, Intervall: %d Frames).",
                BREATHING_WINDOW_SECONDS,
                BREATHING_FRAME_INTERVAL,
            )
        except Exception as exc:
            ERR_TELEMETRY.record(
                component="startup",
                code=BREATHING_ANALYZER_INIT_FAILED,
                detail="Breathing analyzer disabled",
                exc=exc,
                level=logging.WARNING,
                cooldown_s=0.0,
            )
            breathing_analyzer = None

    # --- Tastatur/Maus-Aktivitaets-Monitoring (optional) ---
    activity_analyzer = None
    use_activity = USE_ACTIVITY_MONITOR and not args.no_activity
    if use_activity:
        try:
            from analyzers.activity_analyzer import ActivityAnalyzer

            activity_analyzer = ActivityAnalyzer()
            activity_analyzer.start()
            log.info("Aktivitaets-Monitoring gestartet (Tastatur + Maus).")
        except Exception as exc:
            ERR_TELEMETRY.record(
                component="startup",
                code=ACTIVITY_ANALYZER_INIT_FAILED,
                detail="Activity analyzer disabled",
                exc=exc,
                level=logging.WARNING,
                cooldown_s=0.0,
            )
            activity_analyzer = None

    # --- Warm-up: DeepFace-Modell vorladen ---
    log.info("Emotion-Modell wird geladen (Warm-up)...")
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        analyze_emotion_frame(dummy)
    except Exception as exc:
        ERR_TELEMETRY.record(
            component="startup",
            code=DEEPFACE_WARMUP_FAILED,
            detail="emotion warm-up failed, runtime will retry on demand",
            exc=exc,
            level=logging.DEBUG,
            cooldown_s=0.0,
        )
    log.info("Warm-up abgeschlossen.")

    frame_count = 0
    fps_counter = 0
    fps_display = 0.0
    fps_timer = time.time()
    analysis_every_n = ANALYSIS_EVERY_N_FRAMES
    adapt_cooldown_until = 0.0
    low_fps_guard = False
    lights_off_due_absence = False  # True wenn Licht wegen Abwesenheit ausgeschaltet
    breathing_rest_bpm = float(BR_REST_BPM)
    breathing_baseline_start = time.time()
    breathing_baseline_samples = deque(maxlen=256)
    breathing_arousal_offset_ema = 0.0
    hrv_rest_bpm = float(HRV_BASELINE_BPM)
    hrv_arousal_offset_ema = 0.0
    runtime_started_at = time.time()

    # Adaptiver Regulations-Regler
    regulator = EmotionRegulator(
        target_v=ADAPTIVE_TARGET_VALENCE,
        target_a=ADAPTIVE_TARGET_AROUSAL,
        blend_strength=ADAPTIVE_BLEND_STRENGTH,
        blend_max=ADAPTIVE_BLEND_MAX,
        progress_timeout=ADAPTIVE_PROGRESS_TIMEOUT,
        escalation=ADAPTIVE_BLEND_ESCALATION,
        at_target_threshold=ADAPTIVE_AT_TARGET_THRESHOLD,
    )

    # Zirkadianes Licht-Modell
    circadian = CircadianSchedule() if USE_CIRCADIAN else None
    last_circadian_update = 0.0
    circadian_label = ""
    circadian_hue_neg = VA_HUE_NEGATIVE
    circadian_hue_pos = VA_HUE_POSITIVE
    circadian_bri_max = VA_BRI_HIGH
    circadian_ct_neg = VA_CT_NEGATIVE
    circadian_ct_pos = VA_CT_POSITIVE

    # --- Alexa-Controller (optional) ---
    alexa_controller = None
    use_alexa = USE_ALEXA and not getattr(args, "no_alexa", False)
    if USE_ALEXA and args.mock and not getattr(args, "no_alexa", False):
        log.info("Alexa bleibt trotz --mock aktiv (nur Kamera/Hue sind simuliert).")
    if getattr(args, "no_alexa", False):
        log.info("Alexa per CLI-Flag --no-alexa deaktiviert.")
    if use_alexa:
        if not ALEXA_EMAIL or not ALEXA_PASSWORD or not ALEXA_DEVICE_NAME:
            log.warning(
                "USE_ALEXA=True, aber ALEXA_EMAIL/ALEXA_PASSWORD/ALEXA_DEVICE_NAME "
                "fehlen in config_local.py oder Umgebungsvariablen – Alexa-Steuerung deaktiviert."
            )
        else:
            try:
                from core.alexa_controller import AlexaController

                alexa_controller = AlexaController(
                    email=ALEXA_EMAIL,
                    password=ALEXA_PASSWORD,
                    device_name=ALEXA_DEVICE_NAME,
                    amazon_url=ALEXA_AMAZON_URL,
                    cooldown_seconds=ALEXA_COOLDOWN_SECONDS,
                    music_provider=ALEXA_MUSIC_PROVIDER,
                    mood_playlists=ALEXA_MOOD_PLAYLISTS,
                    volume_control=ALEXA_VOLUME_CONTROL,
                )
                alexa_controller.start()
                log.info("Alexa-Controller gestartet (Geraet: '%s').", ALEXA_DEVICE_NAME)
            except Exception as exc:
                log.warning("Alexa-Controller konnte nicht gestartet werden: %s", exc)
                alexa_controller = None

    # Atemfuehrungs-Entrainment
    pacer = (
        BreathingPacer(
            guide_bpm=BREATHING_PACER_BPM,
            amplitude=BREATHING_PACER_AMPLITUDE,
            fade_in_seconds=BREATHING_PACER_FADE_IN,
        )
        if USE_BREATHING_PACER
        else None
    )

    # --- Kognitiver Zustandsklassifikator (optional) ---
    cognitive_classifier = None
    use_cognitive = USE_COGNITIVE_CLASSIFIER and not args.no_cognitive
    if use_cognitive:
        cognitive_classifier = CognitiveClassifier(stability_window_s=COGNITIVE_STABILITY_WINDOW)
        log.info("Kognitiver Zustandsklassifikator aktiviert.")

    # --- Modus-System (optional) ---
    mode_manager = None
    use_modes = USE_MODE_SYSTEM and not args.no_cognitive  # benoetigt Klassifikator fuer AUTO
    if use_modes:
        initial_mode = args.mode or DEFAULT_MODE
        mode_manager = ModeManager(initial_mode=initial_mode)
        mode_manager._hysteresis_s = MODE_HYSTERESIS_S
        log.info("Modus-System aktiviert (Start: %s).", initial_mode)

    # --- Pausen-Manager (optional) ---
    break_manager = None
    use_breaks = USE_BREAK_MANAGER and not args.no_breaks
    if use_breaks:
        pomo = BREAK_POMODORO_ENABLED or args.pomodoro
        break_manager = BreakManager(
            max_work_minutes=BREAK_MAX_WORK_MINUTES,
            fatigue_trigger_s=BREAK_FATIGUE_TRIGGER_S,
            min_break_minutes=BREAK_MIN_MINUTES,
            pomodoro_enabled=pomo,
            pomodoro_work_minutes=BREAK_POMODORO_WORK_MINUTES,
            pomodoro_break_minutes=BREAK_POMODORO_BREAK_MINUTES,
            pomodoro_long_break_minutes=BREAK_POMODORO_LONG_BREAK_MINUTES,
            pomodoro_long_break_after=BREAK_POMODORO_LONG_BREAK_AFTER,
        )
        log.info("Pausen-Manager aktiviert (Pomodoro: %s).", "ja" if pomo else "nein")

    # --- Feedback-Collector (optional) ---
    feedback_collector = None
    use_feedback = USE_FEEDBACK and not args.no_feedback
    if use_feedback:
        feedback_collector = FeedbackCollector(cooldown_s=FEEDBACK_COOLDOWN_S)
        log.info("Feedback-System aktiviert (Tasten: f=positiv, d=negativ).")

    # Vorausschauende Intervention: Trend-Zaehler
    trend_negative_counter = 0.0
    trend_last_time = time.time()
    last_reg_info = {
        "reg_v": ADAPTIVE_TARGET_VALENCE,
        "reg_a": ADAPTIVE_TARGET_AROUSAL,
        "current_v": ADAPTIVE_TARGET_VALENCE,
        "current_a": ADAPTIVE_TARGET_AROUSAL,
        "target_v": ADAPTIVE_TARGET_VALENCE,
        "target_a": ADAPTIVE_TARGET_AROUSAL,
        "blend": 0.0,
        "label": "Tracking...",
        "at_target": True,
    }

    condition = args.condition or ("adaptive" if ADAPTIVE_REGULATION else "control")
    session_start_ts = time.time()
    last_session_log_ts = 0.0
    session_log_salt = os.environ.get("SESSION_LOG_SALT", "smart-light-default-salt")
    if args.session_log:
        log.info("Session-Logging aktiv: %s (condition=%s)", args.session_log, condition)
        if args.pseudonymize_session and session_log_salt == "smart-light-default-salt":
            log.warning("SESSION_LOG_SALT nicht gesetzt - Default-Salt wird verwendet.")

    try:
        while True:
            ret, frame = source.read()
            if not ret:
                log.warning("Frame-Fehler, überspringe...")
                continue

            frame_count += 1
            fps_counter += 1

            # FPS alle 1s aktualisieren
            now_fps = time.time()
            if now_fps - fps_timer >= 1.0:
                fps_display = fps_counter / (now_fps - fps_timer)
                fps_counter = 0
                fps_timer = now_fps

            # Adaptive Analysefrequenz: bei FPS-Einbruch drosseln, bei Reserve wieder erhoehen.
            startup_ready = (now_fps - runtime_started_at) >= STARTUP_GUARD_DELAY_SECONDS
            if (
                startup_ready
                and ADAPTIVE_ANALYSIS
                and now_fps >= adapt_cooldown_until
                and fps_display > 0
            ):
                if (
                    fps_display < ADAPTIVE_FPS_LOW
                    and analysis_every_n < MAX_ANALYSIS_EVERY_N_FRAMES
                ):
                    analysis_every_n += 1
                    adapt_cooldown_until = now_fps + 0.8
                    log.info(
                        "Adaptive Analyse: 1/%d (FPS %.1f)",
                        analysis_every_n,
                        fps_display,
                    )
                elif (
                    fps_display > ADAPTIVE_FPS_HIGH
                    and analysis_every_n > MIN_ANALYSIS_EVERY_N_FRAMES
                ):
                    analysis_every_n -= 1
                    adapt_cooldown_until = now_fps + 1.2
                    log.info(
                        "Adaptive Analyse: 1/%d (FPS %.1f)",
                        analysis_every_n,
                        fps_display,
                    )

            # Harte Untergrenze: unter MIN_RUNTIME_FPS optionale Analyse-Last abwerfen.
            if startup_ready and fps_display > 0:
                if not low_fps_guard and fps_display < MIN_RUNTIME_FPS:
                    low_fps_guard = True
                    analysis_every_n = min(MAX_ANALYSIS_EVERY_N_FRAMES, analysis_every_n + 2)
                    log.warning(
                        "FPS-GUARD aktiv (%.1f FPS): optionale Analyse wird gedrosselt",
                        fps_display,
                    )
                elif low_fps_guard and fps_display >= (
                    MIN_RUNTIME_FPS + LOW_FPS_RECOVERY_HYSTERESIS
                ):
                    low_fps_guard = False
                    log.info("FPS-GUARD deaktiviert (%.1f FPS)", fps_display)

            # Burst nur mit FPS-Reserve erlauben, sonst kann schnelle Bewegung FPS stark einbrechen lassen.
            burst_allowed = (
                (not ADAPTIVE_ANALYSIS) or (fps_display <= 0) or (fps_display >= ADAPTIVE_FPS_LOW)
            ) and not low_fps_guard

            # Frame an Analyzer senden (normal oder Burst)
            if (analyzer.burst_active and burst_allowed) or frame_count % analysis_every_n == 0:
                analyzer_frame = resize_for_width(frame, ANALYSIS_FRAME_SIZE)
                analyzer.submit(analyzer_frame.copy())

            # Pose-Analyse: Frame senden (halbe Rate)
            if (
                pose_analyzer is not None
                and not low_fps_guard
                and frame_count % (analysis_every_n * 2) == 0
            ):
                pose_frame = resize_for_width(frame, POSE_FRAME_SIZE)
                pose_analyzer.submit(pose_frame.copy())

            # Face-Mesh-Analyse: Frame senden (gleiche Rate wie Hauptanalyse)
            if (
                face_mesh_analyzer is not None
                and not low_fps_guard
                and frame_count % analysis_every_n == 0
            ):
                fm_frame = resize_for_width(frame, FACE_MESH_FRAME_SIZE)
                face_mesh_analyzer.submit(fm_frame.copy())

            # HRV-Analyse: Frame senden (eigenes festes Intervall, unabhängig von FPS-Guard)
            if hrv_analyzer is not None and frame_count % HRV_FRAME_INTERVAL == 0:
                hrv_analyzer.submit(frame.copy())

            # Atemfrequenz: Frame senden (eigenes festes Intervall)
            if breathing_analyzer is not None and frame_count % BREATHING_FRAME_INTERVAL == 0:
                breath_frame = resize_for_width(frame, POSE_FRAME_SIZE)
                breathing_analyzer.submit(breath_frame.copy())

            # Ergebnisse abrufen
            result = analyzer.get()
            emotion = result["emotion"]
            confidence = result["confidence"]
            quality = float(result.get("quality", confidence))
            ema_vector = result.get("ema_vector", {})
            valence = result.get("valence", 0.0)
            arousal = result.get("arousal", 0.0)
            trend_v = result.get("trend_valence", 0.0)
            low_quality_guardrail = False

            # Audio-EMA abrufen
            audio_ema = None
            audio_confidence = 0.0
            audio_quality = 0.0
            dynamic_audio_weight = 0.0
            if audio_analyzer is not None:
                audio_result = audio_analyzer.get()
                audio_ema = audio_result.get("ema_vector")
                audio_confidence = float(audio_result.get("confidence", 0.0))
                audio_quality = float(audio_result.get("quality", audio_confidence))
                dynamic_audio_weight = effective_audio_weight(
                    base_weight=AUDIO_WEIGHT,
                    audio_quality=audio_quality,
                    audio_confidence=audio_confidence,
                    min_factor=AUDIO_DYNAMIC_MIN_FACTOR,
                    quality_exponent=AUDIO_DYNAMIC_QUALITY_EXPONENT,
                )

            # Pose-Arousal-Offset abrufen
            pose_arousal_offset = 0.0
            torso_lean = 0.0
            shoulder_drop = 0.0
            head_tilt = 0.0
            if pose_analyzer is not None:
                pose_result = pose_analyzer.get()
                pose_arousal_offset = pose_result.get("arousal_offset", 0.0)
                if USE_EXTENDED_POSE and not args.no_extended_pose:
                    torso_lean = pose_result.get("torso_lean", 0.0)
                    shoulder_drop = pose_result.get("shoulder_drop", 0.0)
                    head_tilt = pose_result.get("head_tilt", 0.0)

            # Face-Mesh-Ergebnisse abrufen (AU-Scores + Kopfpose-Confidence + Pupille/Blink)
            face_mesh_scores = None
            head_pose_conf_factor = 1.0
            pupil_dilation = 0.0
            blink_rate = 0.0
            if face_mesh_analyzer is not None:
                fm_result = face_mesh_analyzer.get()
                face_mesh_scores = fm_result.get("au_emotion_scores")
                head_pose_conf_factor = fm_result.get("confidence_factor", 1.0)
                if USE_PUPIL_BLINK and not args.no_pupil_blink:
                    pupil_dilation = fm_result.get("pupil_dilation", 0.0)
                    blink_rate = fm_result.get("blink_rate", 0.0)

            # Aktivitaets-Ergebnis abrufen (Tastatur/Maus)
            activity_result = None
            cognitive_load = 0.0
            if activity_analyzer is not None:
                activity_result = activity_analyzer.get()
                cognitive_load = activity_result.get("cognitive_load", 0.0)

            # HRV-Ergebnis abrufen + optionalen Arousal-Offset berechnen
            hrv_result = None
            hrv_arousal_offset = 0.0
            if hrv_analyzer is not None:
                hrv_result = hrv_analyzer.get()
                hr_bpm = hrv_result.get("hr_bpm", 0.0)
                hrv_conf = hrv_result.get("confidence", 0.0)
                raw_hrv_offset = 0.0
                if hr_bpm > 0 and hrv_conf >= HRV_MIN_CONFIDENCE and HRV_AROUSAL_INFLUENCE > 0:
                    if HRV_BASELINE_ADAPT_ALPHA > 0:
                        a_hr = max(0.0, min(1.0, HRV_BASELINE_ADAPT_ALPHA))
                        hrv_rest_bpm = (1.0 - a_hr) * hrv_rest_bpm + a_hr * hr_bpm

                    # Abweichung von persoenlicher Ruhe-HR bestimmt Arousal-Offset.
                    raw_hrv_offset = (
                        (hr_bpm - hrv_rest_bpm) / max(40.0, hrv_rest_bpm)
                    ) * HRV_AROUSAL_INFLUENCE
                    raw_hrv_offset = max(-HRV_OFFSET_CLAMP, min(HRV_OFFSET_CLAMP, raw_hrv_offset))

                # HRV-Offset zeitlich glaetten; bei fehlender Messung sanft gegen 0.
                a_h = max(0.0, min(1.0, HRV_OFFSET_EMA_ALPHA))
                hrv_arousal_offset_ema = (1.0 - a_h) * hrv_arousal_offset_ema + a_h * raw_hrv_offset
                hrv_arousal_offset = hrv_arousal_offset_ema

            # Atemfrequenz-Ergebnis abrufen + optionalen Arousal-Offset berechnen
            breathing_result = None
            breathing_arousal_offset = 0.0
            if breathing_analyzer is not None:
                breathing_result = breathing_analyzer.get()
                br_bpm = breathing_result.get("br_bpm", 0.0)
                br_conf = breathing_result.get("confidence", 0.0)
                raw_breathing_offset = 0.0
                if (
                    br_bpm > 0
                    and br_conf >= BREATHING_MIN_CONFIDENCE
                    and BREATHING_AROUSAL_INFLUENCE > 0
                ):
                    # 1) Erste Sekunden: persoenliche Ruhe-Baseline aus stabilen Messungen lernen.
                    if (time.time() - breathing_baseline_start) <= BREATHING_BASELINE_SECONDS:
                        breathing_baseline_samples.append(br_bpm)
                        if len(breathing_baseline_samples) >= 3:
                            breathing_rest_bpm = float(
                                np.median(np.array(breathing_baseline_samples))
                            )
                    # 2) Danach Baseline langsam weiterfuehren, um Tagesform abzubilden.
                    elif BREATHING_BASELINE_ADAPT_ALPHA > 0:
                        a = BREATHING_BASELINE_ADAPT_ALPHA
                        breathing_rest_bpm = (1.0 - a) * breathing_rest_bpm + a * br_bpm

                    raw_breathing_offset = (
                        (br_bpm - breathing_rest_bpm) / max(1.0, breathing_rest_bpm)
                    ) * BREATHING_AROUSAL_INFLUENCE
                    raw_breathing_offset = max(
                        -BREATHING_OFFSET_CLAMP,
                        min(BREATHING_OFFSET_CLAMP, raw_breathing_offset),
                    )

                # 3) Atem-Offset per EMA glaetten; bei ungueltiger Messung sanft gegen 0 abklingen.
                target_offset = raw_breathing_offset
                a_off = BREATHING_OFFSET_EMA_ALPHA
                breathing_arousal_offset_ema = (
                    1.0 - a_off
                ) * breathing_arousal_offset_ema + a_off * target_offset
                breathing_arousal_offset = breathing_arousal_offset_ema

            # Kopfpose-Kompensation: optional und gedaempft anwenden,
            # damit seitliche Posen nicht unverhaeltnismaessig oft zu neutral fuehren.
            pose_strength = max(0.0, min(1.0, HEAD_POSE_CONFIDENCE_STRENGTH))
            pose_enabled = USE_HEAD_POSE_CONFIDENCE and not args.no_head_pose_penalty
            pose_scale = 1.0
            if pose_enabled:
                pose_scale = (1.0 - pose_strength) + (pose_strength * head_pose_conf_factor)
            effective_confidence = confidence * pose_scale

            # Multimodal-Fusion
            reg_info = None
            fused_v = 0.0
            fused_a = 0.0

            # Zirkadianes Licht-Modell: Zielwerte + Hue-Bereiche periodisch aktualisieren
            if (
                circadian is not None
                and (time.time() - last_circadian_update) >= CIRCADIAN_UPDATE_INTERVAL
            ):
                cp = circadian.get_params()
                regulator.set_target(cp["target_v"], cp["target_a"])
                circadian_hue_neg = cp["hue_negative"]
                circadian_hue_pos = cp["hue_positive"]
                circadian_bri_max = cp["bri_max"]
                circadian_ct_neg = cp.get("ct_negative", VA_CT_NEGATIVE)
                circadian_ct_pos = cp.get("ct_positive", VA_CT_POSITIVE)
                circadian_label = cp["label"]
                last_circadian_update = time.time()
                log.info(
                    "Zirkadian-Update: %s (V=%.2f A=%.2f)",
                    circadian_label,
                    cp["target_v"],
                    cp["target_a"],
                )

            # --- Kognitiver Zustand aktualisieren ---
            cognitive_state_str = "NEUTRAL"
            cognitive_confidence = 0.0
            cognitive_result = None
            if cognitive_classifier is not None:
                hr_bpm_for_cog = 0.0
                hrv_rmssd_for_cog = 0.0
                br_bpm_for_cog = 0.0
                if hrv_result is not None:
                    hr_bpm_for_cog = hrv_result.get("hr_bpm", 0.0)
                    hrv_rmssd_for_cog = hrv_result.get("hrv_rmssd", 0.0)
                if breathing_result is not None:
                    br_bpm_for_cog = breathing_result.get("br_bpm", 0.0)
                at_target_cog = (
                    (reg_info is not None and reg_info.get("at_target", False))
                    if reg_info is not None
                    else (last_reg_info.get("at_target", False) if last_reg_info else False)
                )
                cognitive_result = cognitive_classifier.update(
                    hr_bpm=hr_bpm_for_cog,
                    hrv_rmssd=hrv_rmssd_for_cog,
                    blink_rate=blink_rate,
                    cognitive_load=cognitive_load,
                    torso_lean=torso_lean,
                    shoulder_drop=shoulder_drop,
                    valence=valence,
                    arousal=arousal,
                    br_bpm=br_bpm_for_cog,
                    at_target=at_target_cog,
                )
                cognitive_state_str = cognitive_result.state
                cognitive_confidence = cognitive_result.confidence

            # --- Modus-System aktualisieren ---
            active_mode_str = "AUTO"
            if mode_manager is not None:
                old_mode = mode_manager.update_auto(cognitive_state_str, time.monotonic())
                active_mode_str = mode_manager.active_mode
                if old_mode != active_mode_str:
                    log.info("Modus gewechselt: %s → %s", old_mode, active_mode_str)
                # Modus-Profil auf Regulator und Licht-Parameter anwenden
                profile = mode_manager.active_profile
                if not mode_manager.is_auto or circadian is None:
                    regulator.set_target(profile.target_v, profile.target_a)
                    regulator._blend_strength = profile.blend_strength
                    circadian_hue_neg = profile.hue_negative
                    circadian_hue_pos = profile.hue_positive
                    circadian_bri_max = profile.bri_max
                # Pacer-Steuerung durch Modus
                if pacer is not None and mode_manager is not None:
                    if profile.pacer_active and not pacer.is_active:
                        pacer.set_active(True)

            # --- Pausen-Manager aktualisieren ---
            break_event = None
            if break_manager is not None:
                break_event = break_manager.update(cognitive_state_str)
                if break_event.break_recommended and not break_event.break_active:
                    log.info(
                        "Pausenempfehlung: %s (Arbeitszeit: %.0f min)",
                        break_event.reason,
                        break_event.work_duration_s / 60.0,
                    )

            if ema_vector and effective_confidence > 0.0:
                dynamic_face_mesh_weight = 0.0
                if face_mesh_scores:
                    # Face-Mesh nur stark einmischen, wenn die Pose-/Landmark-Qualitaet stabil ist.
                    dynamic_face_mesh_weight = FACE_MESH_WEIGHT * max(
                        0.0, min(1.0, head_pose_conf_factor)
                    )

                fused_ema = fuse_modalities(
                    ema_vector,
                    audio_ema,
                    pose_arousal_offset,
                    dynamic_audio_weight if audio_ema else 0.0,
                    face_mesh_scores=face_mesh_scores,
                    face_mesh_weight=dynamic_face_mesh_weight,
                )
                fused_v, fused_a = compute_va_from_ema(fused_ema)

                # Vorausschauende Intervention: bei anhaltendem Abwaertstrend Blend verstaerken
                now_trend = time.time()
                dt_trend = now_trend - trend_last_time
                trend_last_time = now_trend
                if trend_v < PREDICTIVE_TREND_THRESHOLD:
                    trend_negative_counter += dt_trend
                else:
                    trend_negative_counter = max(0.0, trend_negative_counter - dt_trend * 2)
                if trend_negative_counter >= PREDICTIVE_TRIGGER_SECONDS:
                    regulator.boost_blend(
                        factor=PREDICTIVE_BOOST_FACTOR,
                        duration_s=PREDICTIVE_BOOST_DURATION,
                    )
                    trend_negative_counter = 0.0
                    log.info("Predictive Intervention: Abwaertstrend erkannt, Blend verstaerkt")

                # Adaptive Regulation: Licht nudgt Richtung Zielzustand statt Ist-Zustand zu spiegeln
                if ADAPTIVE_REGULATION and USE_VALENCE_AROUSAL:
                    reg_info = regulator.update(fused_v, fused_a)
                    # Zirkadianes Hue-Override: lokale Hue-Bereiche fuer valence_arousal_to_light
                    reg_v, reg_a = reg_info["reg_v"], reg_info["reg_a"]
                    if circadian is not None:
                        params = circadian_va_to_light(
                            reg_v,
                            reg_a,
                            circadian_hue_neg,
                            circadian_hue_pos,
                            circadian_bri_max,
                            ct_neg=circadian_ct_neg,
                            ct_pos=circadian_ct_pos,
                        )
                    else:
                        params = valence_arousal_to_light(reg_v, reg_a)
                else:
                    params = blend_emotion_colors(fused_ema)

                # Prosodische Werte fuer Offset-Anwendung vorab lesen
                pitch_hz = 0.0
                speech_rate_val = 0.0
                if USE_PROSODIC and not args.no_prosodic and audio_analyzer is not None:
                    audio_res = audio_analyzer.get()
                    pitch_hz = audio_res.get("pitch_mean_hz", 0.0)
                    speech_rate_val = audio_res.get("speech_rate", 0.0)

                # Atemfuehrungs-Entrainment: Pulsationsfaktor berechnen
                pacer_factor = 1.0
                if pacer is not None:
                    br_active = (
                        breathing_result is not None
                        and breathing_result.get("br_bpm", 0.0) > BREATHING_PACER_BR_THRESHOLD
                        and breathing_result.get("confidence", 0.0) >= BREATHING_MIN_CONFIDENCE
                        and (reg_info is None or not reg_info.get("at_target", False))
                    )
                    pacer.set_active(br_active)
                    pacer_factor = pacer.get_pulsation_factor()

                # Alle Sensor-Offsets auf Lichtparameter anwenden
                apply_modality_offsets(
                    params,
                    pose_arousal_offset=pose_arousal_offset,
                    hrv_arousal_offset=hrv_arousal_offset,
                    breathing_arousal_offset=breathing_arousal_offset,
                    pupil_dilation=pupil_dilation,
                    blink_rate=blink_rate,
                    torso_lean=torso_lean,
                    shoulder_drop=shoulder_drop,
                    cognitive_load=cognitive_load,
                    pitch_mean_hz=pitch_hz,
                    speech_rate=speech_rate_val,
                    pacer_factor=pacer_factor,
                    use_pupil_blink=USE_PUPIL_BLINK and not args.no_pupil_blink,
                    use_extended_pose=USE_EXTENDED_POSE and not args.no_extended_pose,
                    use_prosodic=USE_PROSODIC and not args.no_prosodic,
                    has_activity=activity_analyzer is not None,
                )
            else:
                fused_ema = ema_vector
                params = FALLBACK_LIGHT

            # Transition-Zeit berechnen
            transition = compute_transition(
                trend_v=trend_v,
                breathing_arousal_offset=breathing_arousal_offset,
                cognitive_load=cognitive_load,
                has_activity=activity_analyzer is not None,
            )

            # Unsicherheits-Guardrail: bei niedriger Modellqualitaet konservativer steuern.
            if ema_vector and quality < LOW_QUALITY_THRESHOLD:
                low_quality_guardrail = True
                params = _blend_light_params(params, FALLBACK_LIGHT, LOW_QUALITY_NEUTRAL_BLEND)
                transition = max(transition, int(LOW_QUALITY_MIN_TRANSITION))

            # Abwesenheits-Check: Licht ausschalten wenn zu lange niemand im Bild
            absence_seconds = time.time() - analyzer.last_face_time
            if absence_seconds >= ABSENCE_LIGHT_OFF_SECONDS:
                if not lights_off_due_absence:
                    lights_off_due_absence = True
                    hue.off()
                    log.info(
                        "Niemand im Bild seit %.0f s – Licht ausgeschaltet.",
                        absence_seconds,
                    )
            else:
                if lights_off_due_absence:
                    lights_off_due_absence = False
                    for lid in hue.lids if hasattr(hue, "lids") else []:
                        try:
                            hue.bridge.set_light(lid, "on", True)
                        except Exception as exc:
                            ERR_TELEMETRY.record(
                                component="hue",
                                code=HUE_REENABLE_FAILED,
                                detail=f"re-enable failed for lid={lid}",
                                exc=exc,
                                level=logging.WARNING,
                                cooldown_s=2.0,
                            )
                    log.info("Person erkannt – Licht wieder eingeschaltet.")

            if not lights_off_due_absence:
                hue.apply(params, transition=transition)

            # Alexa: Musik/Lautstaerke emotion-adaptiv steuern
            if alexa_controller is not None and ema_vector and effective_confidence > 0.0:
                alexa_controller.update(fused_v, fused_a, emotion)

            reg_info_display = reg_info
            if reg_info is not None:
                last_reg_info = dict(reg_info)
            elif ADAPTIVE_REGULATION and USE_VALENCE_AROUSAL:
                # Kein frisches Regulator-Update in diesem Frame: letzte Ziel-Infos halten,
                # damit das VA-Diagramm stabil sichtbar bleibt.
                reg_info_display = dict(last_reg_info)
                reg_info_display["current_v"] = float(fused_v)
                reg_info_display["current_a"] = float(fused_a)
                dv = reg_info_display["target_v"] - reg_info_display["current_v"]
                da = reg_info_display["target_a"] - reg_info_display["current_a"]
                dist = float(np.sqrt(dv * dv + da * da))
                reg_info_display["at_target"] = dist < ADAPTIVE_AT_TARGET_THRESHOLD

            _draw_overlay(
                frame=frame,
                fused_ema=fused_ema,
                emotion=emotion,
                confidence=confidence,
                quality=quality,
                valence=valence,
                arousal=arousal,
                trend_v=trend_v,
                params=params,
                transition=transition,
                fps_display=fps_display,
                analysis_every_n=analysis_every_n,
                has_audio=audio_analyzer is not None,
                has_pose=pose_analyzer is not None,
                has_face_mesh=face_mesh_analyzer is not None,
                has_calibration=bool(calibration),
                burst_active=analyzer.burst_active,
                low_fps_guard=low_fps_guard,
                low_quality_guardrail=low_quality_guardrail,
                head_pose_conf=pose_scale,
                absence_seconds=absence_seconds,
                lights_off_due_absence=lights_off_due_absence,
                hrv_result=hrv_result,
                breathing_result=breathing_result,
                breathing_rest_bpm=breathing_rest_bpm,
                breathing_offset=breathing_arousal_offset,
                reg_info=reg_info_display,
                circadian_label=circadian_label,
                pacer_info={
                    "active": pacer is not None and pacer.is_active,
                    "fade_pct": pacer.get_fade_pct() if pacer is not None else 0.0,
                }
                if pacer is not None
                else None,
                pupil_dilation=pupil_dilation,
                blink_rate=blink_rate,
                activity_result=activity_result,
                torso_lean=torso_lean,
                shoulder_drop=shoulder_drop,
                head_tilt=head_tilt,
                cognitive_state=cognitive_state_str,
                cognitive_confidence=cognitive_confidence,
                active_mode=active_mode_str,
                break_event=break_event,
                feedback_flash=feedback_collector.flash_active
                if feedback_collector
                else (False, ""),
            )

            if args.session_log and (time.time() - last_session_log_ts) >= 1.0:
                fb_data = None
                if feedback_collector is not None:
                    fb_data = feedback_collector.get_feedback_for_log() or None
                payload = build_session_payload(
                    session_start_ts=session_start_ts,
                    participant=args.participant,
                    session_id=args.session_id,
                    condition=condition,
                    pseudonymize=args.pseudonymize_session,
                    salt=session_log_salt,
                    emotion=emotion,
                    confidence=confidence,
                    audio_confidence=audio_confidence,
                    audio_quality=audio_quality,
                    dynamic_audio_weight=dynamic_audio_weight,
                    quality=quality,
                    low_quality_guardrail=low_quality_guardrail,
                    fused_v=fused_v,
                    fused_a=fused_a,
                    reg_info=reg_info,
                    adaptive_enabled=bool(ADAPTIVE_REGULATION and USE_VALENCE_AROUSAL),
                    pupil_dilation=pupil_dilation,
                    blink_rate=blink_rate,
                    cognitive_load=cognitive_load,
                    torso_lean=torso_lean,
                    shoulder_drop=shoulder_drop,
                    head_tilt=head_tilt,
                    cognitive_state=cognitive_state_str,
                    cognitive_confidence=cognitive_confidence,
                    active_mode=active_mode_str,
                    break_event=break_event,
                    feedback_data=fb_data,
                )
                append_session_log(args.session_log, payload)
                last_session_log_ts = time.time()

            if not args.mock and not args.headless:
                cv2.imshow("Smart Light", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m") and mode_manager is not None:
                    mode_manager.cycle_mode()
                    log.info("Modus manuell gewechselt: %s", mode_manager.active_mode)
                if key == ord("f") and feedback_collector is not None:
                    if feedback_collector.record(
                        "positive",
                        cognitive_state=cognitive_state_str,
                        active_mode=active_mode_str,
                        valence=fused_v,
                        arousal=fused_a,
                    ):
                        log.info("Feedback: positiv")
                if key == ord("d") and feedback_collector is not None:
                    if feedback_collector.record(
                        "negative",
                        cognitive_state=cognitive_state_str,
                        active_mode=active_mode_str,
                        valence=fused_v,
                        arousal=fused_a,
                    ):
                        log.info("Feedback: negativ")
                if key == ord("b") and break_manager is not None:
                    if break_manager.is_break_active:
                        break_manager.skip_break()
                        log.info("Pause manuell beendet.")
                    else:
                        break_manager.start_break()
                        log.info("Pause manuell gestartet.")
                if key == ord("n") and break_manager is not None and break_event is not None:
                    if break_event.break_recommended and not break_event.break_active:
                        break_manager.dismiss_break()
                        log.info("Pausenempfehlung abgelehnt.")
                    break
                if cv2.getWindowProperty("Smart Light", cv2.WND_PROP_VISIBLE) < 1:
                    break

    except KeyboardInterrupt:
        log.info("Keyboard-Interrupt empfangen.")
    finally:
        log.info("Aufräumen...")
        analyzer.stop()
        if audio_analyzer is not None:
            audio_analyzer.stop()
        if pose_analyzer is not None:
            pose_analyzer.stop()
        if face_mesh_analyzer is not None:
            face_mesh_analyzer.stop()
        if hrv_analyzer is not None:
            hrv_analyzer.stop()
        if breathing_analyzer is not None:
            breathing_analyzer.stop()
        if activity_analyzer is not None:
            activity_analyzer.stop()
        if alexa_controller is not None:
            alexa_controller.shutdown()
        hue.apply(FALLBACK_LIGHT, transition=10)
        hue.shutdown()
        if ERR_TELEMETRY.summary():
            log.info("Fehler-Telemetrie: %s", ERR_TELEMETRY.summary())
        time.sleep(1)
        source.release()
        if not args.mock and not args.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        log.info("Beendet.")


if __name__ == "__main__":
    main()
