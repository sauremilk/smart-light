"""Overlay-Rendering fuer das Kamera-Bild.

Enthaelt alle UI-Zeichenroutinen, die in der Hauptschleife auf den
OpenCV-Frame gerendert werden. Ausgelagert aus main.py fuer bessere
Modularitaet.
"""

from __future__ import annotations

import cv2

from config import (
    ABSENCE_LIGHT_OFF_SECONDS,
    FALLBACK_AFTER_SECONDS,
)

# ─── UI Drawing Helpers ────────────────────────────────────────────────

_UI_EMOTION_COLORS = {
    "happy": (80, 230, 255),
    "sad": (200, 150, 100),
    "angry": (80, 80, 230),
    "fear": (160, 120, 200),
    "surprise": (80, 200, 255),
    "disgust": (120, 160, 100),
    "neutral": (170, 170, 180),
}


def _draw_rounded_rect(img, x1, y1, x2, y2, color, radius=10):
    """Draws a filled rounded rectangle."""
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ]:
        cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)


def _draw_rounded_rect_border(img, x1, y1, x2, y2, color, radius=10, thickness=1):
    """Draws a rounded rectangle outline."""
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(
        img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA
    )
    cv2.ellipse(
        img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA
    )
    cv2.ellipse(
        img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA
    )
    cv2.ellipse(
        img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA
    )


def _draw_mini_bar(frame, x, y, w, h, value, max_val=1.0, color=(120, 220, 150)):
    """Draws a small progress/gauge bar."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (35, 35, 42), -1)
    fill_w = max(0, min(w, int(w * value / max_val)))
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 58), 1)


def _draw_bipolar_bar(
    frame, x, y, w, h, value, color_neg=(130, 120, 200), color_pos=(120, 210, 160)
):
    """Draws a bar centered at 0 for values in [-1, +1]."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (35, 35, 42), -1)
    mid = x + w // 2
    clamped = max(-1.0, min(1.0, value))
    fill_w = int(abs(clamped) * (w // 2))
    if clamped >= 0 and fill_w > 0:
        cv2.rectangle(frame, (mid, y), (mid + fill_w, y + h), color_pos, -1)
    elif fill_w > 0:
        cv2.rectangle(frame, (mid - fill_w, y), (mid, y + h), color_neg, -1)
    cv2.line(frame, (mid, y), (mid, y + h), (80, 80, 90), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 58), 1)


def _draw_emotion_distribution(frame, x, y, w, h, fused_ema, radius=4):
    """Draws a horizontal bar chart showing emotion distribution."""
    if not fused_ema:
        return
    top3 = sorted(fused_ema.items(), key=lambda item: item[1], reverse=True)[:3]
    total = sum(v for _, v in top3)
    if total <= 0:
        return
    # Background with rounded corners
    _draw_rounded_rect(frame, x, y, x + w, y + h, (35, 35, 42), radius=radius)
    bar_x = x + 1
    for i, (name, val) in enumerate(top3):
        seg_w = max(1, int((w - 2) * val / total))
        if i == len(top3) - 1:
            seg_w = (x + w - 1) - bar_x
        color = _UI_EMOTION_COLORS.get(name, (170, 170, 180))
        cv2.rectangle(frame, (bar_x, y + 1), (bar_x + seg_w, y + h - 1), color, -1)
        bar_x += seg_w
    _draw_rounded_rect_border(frame, x, y, x + w, y + h, (50, 50, 58), radius=radius)


def _draw_section_divider(frame, x, y, w, color=(45, 45, 55)):
    """Draws a subtle section divider line."""
    cv2.line(frame, (x, y), (x + w, y), color, 1, cv2.LINE_AA)


def _build_top3_text(fused_ema: dict, emotion: str, confidence: float) -> str:
    """Formats top emotion output for the overlay."""
    if fused_ema:
        top3 = sorted(fused_ema.items(), key=lambda item: item[1], reverse=True)[:3]
        return " | ".join(f"{name} {weight:.0%}" for name, weight in top3)
    return f"{emotion} ({confidence:.0%})"


def _build_status_text(
    fps_display: float,
    analysis_every_n: int,
    has_audio: bool,
    has_pose: bool,
    has_face_mesh: bool,
    has_calibration: bool,
    burst_active: bool,
    low_fps_guard: bool = False,
    has_hrv: bool = False,
    has_breathing: bool = False,
    low_quality_guardrail: bool = False,
    has_activity: bool = False,
    has_pupil_blink: bool = False,
) -> str:
    """Builds the status line shown at the bottom of the overlay."""
    modules = ["Video"]
    if has_audio:
        modules.append("Audio")
    if has_pose:
        modules.append("Pose")
    if has_face_mesh:
        modules.append("FaceMesh")
    if has_hrv:
        modules.append("HRV")
    if has_breathing:
        modules.append("Breath")
    if has_activity:
        modules.append("Activity")
    if has_pupil_blink:
        modules.append("Pupil")
    if has_calibration:
        modules.append("Cal")

    status = f"FPS:{fps_display:.0f}  1/{analysis_every_n}  [{'+'.join(modules)}]"
    if burst_active:
        status += "  BURST"
    if low_fps_guard:
        status += "  FPS-GUARD"
    if low_quality_guardrail:
        status += "  LOW-Q"
    return status


def _blend_light_params(base: dict, fallback: dict, blend: float) -> dict:
    """Mischt einen Lichtzustand mit einem neutral-sicheren Fallback-Zustand."""
    a = max(0.0, min(1.0, float(blend)))
    b = 1.0 - a
    return {
        "hue": int(round(b * float(base["hue"]) + a * float(fallback["hue"]))),
        "bri": int(round(b * float(base["bri"]) + a * float(fallback["bri"]))),
        "sat": int(round(b * float(base["sat"]) + a * float(fallback["sat"]))),
    }


def _draw_va_diagram(
    frame,
    current_v: float,
    current_a: float,
    target_v: float,
    target_a: float,
    at_target: bool,
    size: int = 126,
    margin: int = 12,
):
    """Zeichnet ein klar beschriftetes Valence-Arousal-Diagramm oben rechts.

    Valence (horizontal): links = negativ, rechts = positiv.
    Arousal  (vertical):  oben  = hoch,    unten  = niedrig.
    """
    h, w = frame.shape[:2]
    ox = w - size - margin
    oy = margin
    title_h = 20
    legend_h = 32
    box_x0 = ox - 10
    box_y0 = oy - 8
    box_x1 = ox + size + 10
    box_y1 = oy + size + title_h + legend_h

    # Halbtransparenter, abgerundeter Hintergrund
    overlay_buf = frame.copy()
    _draw_rounded_rect(overlay_buf, box_x0, box_y0, box_x1, box_y1, (16, 16, 20), radius=8)
    cv2.addWeighted(overlay_buf, 0.62, frame, 0.38, 0, frame)
    _draw_rounded_rect_border(frame, box_x0, box_y0, box_x1, box_y1, (60, 60, 70), radius=8)

    # Quadranten subtil einfaerben.
    cx = ox + size // 2
    cy = oy + size // 2
    tint = frame.copy()
    cv2.rectangle(tint, (ox, oy), (cx, cy), (20, 55, 140), -1)
    cv2.rectangle(tint, (cx, oy), (ox + size, cy), (50, 135, 70), -1)
    cv2.rectangle(tint, (ox, cy), (cx, oy + size), (70, 85, 130), -1)
    cv2.rectangle(tint, (cx, cy), (ox + size, oy + size), (60, 110, 80), -1)
    cv2.addWeighted(tint, 0.14, frame, 0.86, 0, frame)

    # Achsen
    cv2.line(frame, (ox, cy), (ox + size, cy), (80, 80, 90), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, oy), (cx, oy + size), (80, 80, 90), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (ox, oy), (ox + size, oy + size), (60, 60, 70), 1)

    def _va_to_px(v: float, a: float):
        """Wandelt (valence, arousal) in Pixel-Koordinaten im Diagramm um."""
        px = int(ox + (v + 1.0) / 2.0 * size)
        py = int(oy + (1.0 - (a + 1.0) / 2.0) * size)
        return (
            max(ox, min(ox + size, px)),
            max(oy, min(oy + size, py)),
        )

    tgt_px = _va_to_px(target_v, target_a)
    cur_px = _va_to_px(current_v, current_a)

    # Pfeil IST -> SOLL (nur wenn nicht am Ziel)
    if not at_target:
        cv2.arrowedLine(frame, cur_px, tgt_px, (30, 200, 255), 2, cv2.LINE_AA, tipLength=0.25)

    # Ziel-Punkt (gruen, etwas kleiner)
    cv2.circle(frame, tgt_px, 5, (60, 220, 90), -1, cv2.LINE_AA)

    # Ist-Punkt (cyan)
    col_cur = (80, 220, 110) if at_target else (230, 200, 40)
    cv2.circle(frame, cur_px, 6, col_cur, -1, cv2.LINE_AA)

    # Mini-Labels
    label_scale = 0.34
    label_thickness = 1
    cv2.putText(
        frame,
        "IST",
        (cur_px[0] + 5, cur_px[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        col_cur,
        label_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "SOLL",
        (tgt_px[0] + 5, tgt_px[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        (60, 220, 90),
        label_thickness,
        cv2.LINE_AA,
    )

    # Titel + Achsenhinweise
    cv2.putText(
        frame,
        "Valence/Arousal",
        (ox, oy - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "-V",
        (ox + 2, cy - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "+V",
        (ox + size - 22, cy - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "+A",
        (cx + 4, oy + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "-A",
        (cx + 4, oy + size - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    # Mini-Legende
    ly = oy + size + 14
    cv2.circle(frame, (ox + 8, ly), 4, col_cur, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "IST",
        (ox + 16, ly + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(frame, (ox + 58, ly), 4, (60, 220, 90), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "SOLL",
        (ox + 66, ly + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Pfeil = Richtung",
        (ox, ly + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (165, 165, 165),
        1,
        cv2.LINE_AA,
    )


def _draw_overlay(
    frame,
    fused_ema: dict,
    emotion: str,
    confidence: float,
    valence: float,
    arousal: float,
    trend_v: float,
    params: dict,
    transition: int,
    fps_display: float,
    analysis_every_n: int,
    has_audio: bool,
    has_pose: bool,
    has_face_mesh: bool,
    has_calibration: bool,
    burst_active: bool = False,
    low_fps_guard: bool = False,
    low_quality_guardrail: bool = False,
    quality: float = 0.0,
    head_pose_conf: float = 1.0,
    absence_seconds: float = 0.0,
    lights_off_due_absence: bool = False,
    hrv_result: dict | None = None,
    breathing_result: dict | None = None,
    breathing_rest_bpm: float | None = None,
    breathing_offset: float = 0.0,
    reg_info: dict | None = None,
    circadian_label: str = "",
    pacer_info: dict | None = None,
    pupil_dilation: float = 0.0,
    blink_rate: float = 0.0,
    activity_result: dict | None = None,
    torso_lean: float = 0.0,
    shoulder_drop: float = 0.0,
    head_tilt: float = 0.0,
    *,
    cognitive_state: str = "",
    cognitive_confidence: float = 0.0,
    active_mode: str = "",
    break_event=None,
    feedback_flash: tuple[bool, str] = (False, ""),
):
    """Renders runtime telemetry text on the camera frame."""
    panel_x = 14
    panel_y = 12
    x0 = 32
    line_h = 30
    header_h = 24
    emo_bar_h = 8
    y0 = panel_y + header_h + 16

    def _draw_text(
        text: str,
        y: int,
        color: tuple[int, int, int],
        scale: float = 0.58,
        weight: int = 1,
        x_override: int | None = None,
    ):
        tx = x_override if x_override is not None else x0
        cv2.putText(
            frame,
            text,
            (tx, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (10, 10, 12),
            weight + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (tx, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            weight,
            cv2.LINE_AA,
        )

    # --- Panel-Groesse berechnen ---
    content_lines = 4  # emotion, VA, light, status
    if fused_ema:
        content_lines += 1  # emotion bar + spacing
    if reg_info is not None:
        content_lines += 1
    if circadian_label:
        content_lines += 1
    if pacer_info is not None and pacer_info.get("active"):
        content_lines += 1
    if hrv_result is not None:
        content_lines += 1
    if breathing_result is not None:
        content_lines += 1
        if breathing_rest_bpm is not None:
            content_lines += 1
    if lights_off_due_absence or absence_seconds >= FALLBACK_AFTER_SECONDS:
        content_lines += 1
    if pupil_dilation > 0 or blink_rate > 0:
        content_lines += 1
    if activity_result is not None:
        content_lines += 1
    if torso_lean != 0 or shoulder_drop > 0 or head_tilt != 0:
        content_lines += 1
    if cognitive_state or active_mode:
        content_lines += 1
    if break_event is not None:
        content_lines += 1
    if feedback_flash[0]:
        content_lines += 1

    panel_h = header_h + content_lines * line_h + 28
    panel_w = min(frame.shape[1] - 24, 620)
    panel_x2 = panel_x + panel_w
    panel_y2 = panel_y + panel_h

    # --- Abgerundetes Panel mit sanfter Transparenz ---
    overlay = frame.copy()
    _draw_rounded_rect(overlay, panel_x, panel_y, panel_x2, panel_y2, (18, 18, 22), radius=12)
    cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, frame)
    _draw_rounded_rect_border(frame, panel_x, panel_y, panel_x2, panel_y2, (50, 50, 60), radius=12)

    # Linke Akzentlinie: breiter, weicher.
    accent = (110, 210, 140) if confidence > 0 else (60, 165, 235)
    _draw_rounded_rect(
        frame,
        panel_x,
        panel_y,
        panel_x + 6,
        panel_y2,
        accent,
        radius=3,
    )

    # --- Header ---
    _draw_text("Smart Light", panel_y + 18, (130, 135, 145), scale=0.42, weight=1)
    _draw_section_divider(frame, x0, panel_y + header_h + 2, panel_w - 36)

    # --- Emotion ---
    color = (140, 250, 170) if confidence > 0 else (80, 190, 245)
    top3_str = _build_top3_text(fused_ema, emotion, confidence)
    _draw_text(top3_str, y0, color, scale=0.62, weight=2)

    # Emotion-Verteilungsbalken
    emo_bar_y = y0 + 6
    if fused_ema:
        _draw_emotion_distribution(frame, x0, emo_bar_y, panel_w - 42, emo_bar_h, fused_ema)
        y_cursor = emo_bar_y + emo_bar_h + 10
    else:
        y_cursor = y0 + line_h

    # --- Stimmung & Energie (Valence / Arousal) ---
    va_str = f"V:{valence:+.2f} A:{arousal:+.2f} T:{trend_v:+.3f}"
    va_str += f" Q:{quality:.2f}"
    _draw_text(va_str, y_cursor, (210, 210, 155), scale=0.48, weight=1)

    # Visuelle Mini-Bars fuer Valence und Arousal
    bar_x = x0 + panel_w - 190
    _draw_bipolar_bar(
        frame,
        bar_x,
        y_cursor - 12,
        70,
        8,
        valence,
        color_neg=(150, 130, 200),
        color_pos=(120, 210, 150),
    )
    _draw_bipolar_bar(
        frame,
        bar_x + 78,
        y_cursor - 12,
        70,
        8,
        arousal,
        color_neg=(140, 140, 160),
        color_pos=(100, 200, 240),
    )
    # Quality-Balken
    _draw_mini_bar(
        frame, bar_x + 156, y_cursor - 12, 30, 8, quality, max_val=1.0, color=(180, 200, 120)
    )
    y_cursor += line_h

    # --- Licht-Parameter ---
    hpc_str = f"HeadConf:{head_pose_conf:.0%}" if head_pose_conf < 1.0 else ""
    light_str = f"Hue:{params['hue']} Bri:{params['bri']} Sat:{params['sat']}  Trans:{transition}"
    if hpc_str:
        light_str += f"  {hpc_str}"
    _draw_text(light_str, y_cursor, (215, 215, 225), scale=0.46, weight=1)
    y_cursor += line_h

    # --- Status-Footer ---
    status = _build_status_text(
        fps_display=fps_display,
        analysis_every_n=analysis_every_n,
        has_audio=has_audio,
        has_pose=has_pose,
        has_face_mesh=has_face_mesh,
        has_calibration=has_calibration,
        burst_active=burst_active,
        low_fps_guard=low_fps_guard,
        low_quality_guardrail=low_quality_guardrail,
        has_hrv=hrv_result is not None,
        has_breathing=breathing_result is not None,
        has_activity=activity_result is not None,
        has_pupil_blink=pupil_dilation > 0 or blink_rate > 0,
    )
    _draw_text(status, y_cursor, (155, 160, 120), scale=0.42, weight=1)
    y_cursor += line_h

    # Trennlinie vor optionalen Sektionen
    _draw_section_divider(frame, x0, y_cursor - line_h + 18, panel_w - 42)

    # Regulationszeile
    if reg_info is not None:
        if reg_info["at_target"]:
            reg_color = (110, 220, 140)
        else:
            reg_color = (60, 195, 245)
        blend_pct = reg_info["blend"]
        reg_label = reg_info["label"]
        if blend_pct > 0:
            reg_str = f"Reguliere: {reg_label}  (Blend {blend_pct:.0%})"
        else:
            reg_str = f"Reguliere: {reg_label}"
        _draw_text(reg_str, y_cursor, reg_color, scale=0.48, weight=1)
        y_cursor += line_h

    # Zirkadian-Zeile anzeigen
    y_next = y_cursor
    if circadian_label:
        circ_str = f"Tageszeit: {circadian_label}"
        _draw_text(circ_str, y_next, (175, 195, 245), scale=0.48, weight=1)
        y_next += line_h

    # Atemfuehrungs-Zeile anzeigen
    if pacer_info is not None and pacer_info.get("active"):
        fade_pct = pacer_info.get("fade_pct", 0.0)
        pacer_str = f"Atemfuehrung 6/min ({fade_pct:.0%})"
        _draw_text(pacer_str, y_next, (150, 240, 210), scale=0.48, weight=1)
        if fade_pct > 0:
            _draw_mini_bar(frame, x0 + 240, y_next - 11, 60, 7, fade_pct, color=(150, 240, 210))
        y_next += line_h

    # HRV-Zeile anzeigen
    if hrv_result is not None:
        hr = hrv_result.get("hr_bpm", 0.0)
        rmssd = hrv_result.get("hrv_rmssd", 0.0)
        conf = hrv_result.get("confidence", 0.0)
        face_detected = bool(hrv_result.get("face_detected", True))
        if not face_detected:
            hrv_str = "HR: kein Gesicht"
            hrv_color = (130, 130, 135)
        elif hr > 0:
            hrv_str = f"HR:{hr:.0f}bpm  RMSSD:{rmssd:.0f}ms  ({conf:.0%})"
            hrv_color = (100, 210, 245)
            # Confidence-Balken
            _draw_mini_bar(frame, x0 + 320, y_next - 11, 50, 7, conf, color=(100, 210, 245))
        else:
            hrv_str = "HR: wird gemessen..."
            hrv_color = (130, 130, 135)
        _draw_text(hrv_str, y_next, hrv_color, scale=0.48, weight=1)
        y_next += line_h

    # Atemfrequenz-Zeile anzeigen
    if breathing_result is not None:
        br = breathing_result.get("br_bpm", 0.0)
        br_conf = breathing_result.get("confidence", 0.0)
        if br > 0:
            br_str = f"Atem:{br:.1f} AZ/min  ({br_conf:.0%})"
            br_color = (130, 240, 175)
            _draw_mini_bar(frame, x0 + 240, y_next - 11, 50, 7, br_conf, color=(130, 240, 175))
        else:
            br_str = "Atem: wird gemessen..."
            br_color = (130, 130, 135)
        _draw_text(br_str, y_next, br_color, scale=0.48, weight=1)
        y_next += line_h

        if breathing_rest_bpm is not None:
            offset_str = f"Atem-Basis:{breathing_rest_bpm:.1f}  Offset:{breathing_offset:+.3f}"
            _draw_text(offset_str, y_next, (165, 220, 180), scale=0.44, weight=1)
            y_next += line_h

    # Abwesenheits-Warnung anzeigen
    if lights_off_due_absence:
        _draw_text("LICHT AUS - kein Gesicht", y_next, (40, 70, 255), scale=0.55, weight=2)
        y_next += line_h
    elif absence_seconds >= FALLBACK_AFTER_SECONDS:
        remaining = max(0.0, ABSENCE_LIGHT_OFF_SECONDS - absence_seconds)
        warn_str = f"Kein Gesicht - Licht aus in {remaining:.0f}s"
        _draw_text(warn_str, y_next, (50, 175, 245), scale=0.48, weight=1)
        y_next += line_h

    # Pupillen-/Blink-Zeile anzeigen
    if pupil_dilation > 0 or blink_rate > 0:
        pb_parts = []
        if pupil_dilation > 0:
            pb_parts.append(f"Pupille:{pupil_dilation:.2f}")
        if blink_rate > 0:
            pb_parts.append(f"Blink:{blink_rate:.0f}/min")
        pb_str = "  ".join(pb_parts)
        pb_color = (190, 175, 240)
        _draw_text(pb_str, y_next, pb_color, scale=0.46, weight=1)
        y_next += line_h

    # Aktivitaets-Zeile anzeigen
    if activity_result is not None:
        kpm = activity_result.get("keys_per_minute", 0.0)
        cl = activity_result.get("cognitive_load", 0.0)
        act_str = f"Aktivitaet: {kpm:.0f} Tasten/min  Load:{cl:.0%}"
        act_color = (240, 195, 140)
        _draw_text(act_str, y_next, act_color, scale=0.46, weight=1)
        _draw_mini_bar(frame, x0 + 320, y_next - 11, 50, 7, cl, color=(240, 195, 140))
        y_next += line_h

    # Erweiterte Pose-Zeile anzeigen
    if torso_lean != 0 or shoulder_drop > 0 or head_tilt != 0:
        pose_parts = []
        if torso_lean != 0:
            pose_parts.append(f"Lean:{torso_lean:+.2f}")
        if shoulder_drop > 0:
            pose_parts.append(f"Drop:{shoulder_drop:.2f}")
        if head_tilt != 0:
            pose_parts.append(f"Tilt:{head_tilt:+.2f}")
        pose_str = "Pose: " + "  ".join(pose_parts)
        _draw_text(pose_str, y_next, (175, 210, 245), scale=0.46, weight=1)
        y_next += line_h

    # Kognitiver Zustand + Modus anzeigen
    if cognitive_state or active_mode:
        cog_parts = []
        if cognitive_state:
            cog_parts.append(f"Zustand:{cognitive_state}({cognitive_confidence:.0%})")
        if active_mode:
            cog_parts.append(f"Modus:{active_mode}")
        cog_str = "  ".join(cog_parts)
        _cog_colors = {
            "FOCUS": (245, 215, 110),
            "FLOW": (110, 240, 200),
            "FATIGUE": (110, 145, 240),
            "STRESS": (90, 90, 240),
            "NEUTRAL": (190, 190, 195),
        }
        cog_color = _cog_colors.get(cognitive_state, (190, 190, 195))
        _draw_text(cog_str, y_next, cog_color, scale=0.46, weight=1)
        y_next += line_h

    # Pausen-Info anzeigen
    if break_event is not None:
        if break_event.break_active:
            b_dur = break_event.break_duration_s
            b_str = f"PAUSE ({b_dur:.0f}s)  [b]=beenden"
            _draw_text(b_str, y_next, (100, 240, 200), scale=0.50, weight=2)
            y_next += line_h
        elif break_event.break_recommended:
            b_str = f"Pause empfohlen: {break_event.reason}  [b]=starten [n]=spaeter"
            _draw_text(b_str, y_next, (70, 200, 245), scale=0.46, weight=2)
            y_next += line_h
        elif break_event.work_duration_s > 0:
            work_min = break_event.work_duration_s / 60.0
            pomo_str = (
                f"  Pomodoro #{break_event.pomodoro_cycle}"
                if break_event.pomodoro_cycle > 0
                else ""
            )
            b_str = f"Arbeit: {work_min:.0f}min{pomo_str}"
            _draw_text(b_str, y_next, (165, 165, 170), scale=0.42, weight=1)
            y_next += line_h

    # Feedback-Flash anzeigen
    if feedback_flash[0]:
        if feedback_flash[1] == "positive":
            fb_str = "Feedback: Positiv"
            fb_color = (100, 240, 140)
        else:
            fb_str = "Feedback: Negativ"
            fb_color = (100, 130, 240)
        _draw_text(fb_str, y_next, fb_color, scale=0.50, weight=2)
        y_next += line_h

    # VA-Diagramm: zeigt Ist-Zustand, Ziel und Regulierungsrichtung (oben rechts)
    if reg_info is not None:
        _draw_va_diagram(
            frame,
            current_v=reg_info["current_v"],
            current_a=reg_info["current_a"],
            target_v=reg_info["target_v"],
            target_a=reg_info["target_a"],
            at_target=reg_info["at_target"],
        )
