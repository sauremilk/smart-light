"""Shared runtime error taxonomy for consistent telemetry codes."""

from __future__ import annotations

# Startup / initialization
HUE_CONNECT_FAILED = "HUE_CONNECT_FAILED"
AUDIO_ANALYZER_INIT_FAILED = "AUDIO_ANALYZER_INIT_FAILED"
POSE_ANALYZER_INIT_FAILED = "POSE_ANALYZER_INIT_FAILED"
FACEMESH_ANALYZER_INIT_FAILED = "FACEMESH_ANALYZER_INIT_FAILED"
HRV_ANALYZER_INIT_FAILED = "HRV_ANALYZER_INIT_FAILED"
BREATHING_ANALYZER_INIT_FAILED = "BREATHING_ANALYZER_INIT_FAILED"
ACTIVITY_ANALYZER_INIT_FAILED = "ACTIVITY_ANALYZER_INIT_FAILED"
DEEPFACE_WARMUP_FAILED = "DEEPFACE_WARMUP_FAILED"

# Runtime inference / processing
DEEPFACE_ANALYZE_FAILED = "DEEPFACE_ANALYZE_FAILED"
CALIBRATION_LOAD_FAILED = "CALIBRATION_LOAD_FAILED"

# Hue transport / control
HUE_SEND_FAILED = "HUE_SEND_FAILED"
HUE_OFF_FAILED = "HUE_OFF_FAILED"
HUE_REENABLE_FAILED = "HUE_REENABLE_FAILED"


ERROR_CODE_CATEGORIES = {
    HUE_CONNECT_FAILED: "startup.hue",
    AUDIO_ANALYZER_INIT_FAILED: "startup.audio",
    POSE_ANALYZER_INIT_FAILED: "startup.pose",
    FACEMESH_ANALYZER_INIT_FAILED: "startup.facemesh",
    HRV_ANALYZER_INIT_FAILED: "startup.hrv",
    BREATHING_ANALYZER_INIT_FAILED: "startup.breathing",
    ACTIVITY_ANALYZER_INIT_FAILED: "startup.activity",
    DEEPFACE_WARMUP_FAILED: "startup.emotion",
    DEEPFACE_ANALYZE_FAILED: "runtime.emotion",
    CALIBRATION_LOAD_FAILED: "runtime.calibration",
    HUE_SEND_FAILED: "runtime.hue",
    HUE_OFF_FAILED: "runtime.hue",
    HUE_REENABLE_FAILED: "runtime.hue",
}


def category_for_error_code(code: str) -> str:
    """Returns the canonical category for a telemetry error code."""
    return ERROR_CODE_CATEGORIES.get(code, "runtime.unknown")
