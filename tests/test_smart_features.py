"""Schnelltest fuer alle neuen Features der smarten Lichtsteuerung."""

import time


def test_circadian_24h_coverage():
    from circadian import CircadianSchedule
    c = CircadianSchedule()
    for h in range(24):
        p = c.get_params(h)
        assert "target_v" in p
        assert "target_a" in p
        assert "hue_negative" in p
        assert "hue_positive" in p
        assert "bri_max" in p
        assert "label" in p
        assert -1.0 <= p["target_v"] <= 1.0
        assert -1.0 <= p["target_a"] <= 1.0
        assert 0 < p["bri_max"] <= 254


def test_circadian_no_jumps():
    from circadian import CircadianSchedule
    c = CircadianSchedule()
    prev = c.get_params(0)
    for h in range(1, 24):
        cur = c.get_params(h)
        # Arousal-Sprung zwischen benachbarten Stunden max 0.65 (Nacht→Morgen erlaubt)
        assert abs(cur["target_a"] - prev["target_a"]) <= 0.65, (
            f"Arousal-Sprung h={h}: {prev['target_a']} -> {cur['target_a']}"
        )
        prev = cur


def test_breathing_pacer_inactive():
    from light_mapping import BreathingPacer
    p = BreathingPacer()
    assert not p.is_active
    assert p.get_pulsation_factor() == 1.0
    assert p.get_fade_pct() == 0.0


def test_breathing_pacer_active_pulsation():
    from light_mapping import BreathingPacer
    p = BreathingPacer(guide_bpm=6.0, amplitude=0.08, fade_in_seconds=0.01)
    p.set_active(True)
    time.sleep(0.02)
    pf = p.get_pulsation_factor()
    assert 0.90 <= pf <= 1.10, f"Unexpected: {pf}"
    assert p.is_active
    assert p.get_fade_pct() > 0.0

    p.set_active(False)
    assert not p.is_active
    assert p.get_pulsation_factor() == 1.0


def test_compose_multi_light_scene_primary():
    from light_mapping import compose_multi_light_scene
    primary = {"hue": 14000, "bri": 200, "sat": 180}
    result = compose_multi_light_scene(primary, "primary")
    assert result == primary


def test_compose_multi_light_scene_accent():
    from light_mapping import compose_multi_light_scene
    primary = {"hue": 14000, "bri": 200, "sat": 180}
    result = compose_multi_light_scene(primary, "accent")
    assert result["hue"] == 15500   # +1500
    assert result["sat"] == 162     # 180 * 0.9
    assert result["bri"] == 200     # unchanged


def test_compose_multi_light_scene_ambient():
    from light_mapping import compose_multi_light_scene
    primary = {"hue": 14000, "bri": 200, "sat": 180}
    result = compose_multi_light_scene(primary, "ambient")
    assert result["hue"] == 16000   # +2000
    assert result["sat"] == 108     # 180 * 0.6
    assert result["bri"] == 140     # 200 * 0.7


def test_regulator_set_target():
    from emotion_regulator import EmotionRegulator
    r = EmotionRegulator(0.65, 0.35, 0.45, 0.8, 30, 0.1, 0.18)
    r.set_target(0.5, -0.1)
    info = r.update(-0.5, 0.5)
    assert info["target_v"] == 0.5
    assert info["target_a"] == -0.1


def test_regulator_boost_blend():
    from emotion_regulator import EmotionRegulator
    r = EmotionRegulator(0.65, 0.35, 0.45, 0.8, 30, 0.1, 0.18)
    r.update(-0.8, 0.9)  # init
    r.boost_blend(factor=1.5, duration_s=10.0)
    info = r.update(-0.8, 0.9)
    # Boosted blend should be > normal Q3 blend_start (0.55)
    assert info["blend"] > 0.55


def test_regulator_quadrant_q3_aggressive():
    from emotion_regulator import EmotionRegulator
    r = EmotionRegulator(0.65, 0.35, 0.45, 0.8, 30, 0.1, 0.18)
    # Q3: Stress (-v, +a) -> should use aggressive params
    info = r.update(-0.9, 1.0)
    # First call has no progress tracking yet, but should still regulate
    assert info["reg_v"] > -0.9
    assert info["reg_a"] < 1.0
    assert not info["at_target"]


def test_regulator_quadrant_q2_minimal():
    from emotion_regulator import EmotionRegulator
    r = EmotionRegulator(0.65, 0.35, 0.45, 0.8, 30, 0.1, 0.18)
    # Q2: Relaxed (+v, -a) -> minimal intervention
    info = r.update(0.5, -0.3)
    # Should still regulate toward target, but with lower blend
    assert info["at_target"] is False


def test_warm_amber_no_blue():
    """Bei negativer Valence (Trauer/Angst) darf kein Blau mehr kommen."""
    from light_mapping import valence_arousal_to_light
    # Negative Valence = Trauer
    params = valence_arousal_to_light(valence=-1.0, arousal=-0.5)
    # Hue 47000 war vorher Blau; jetzt sollte es Warm-Amber (~9000) sein
    assert params["hue"] < 15000, f"Erwartet Warm-Amber, bekam hue={params['hue']}"
    # Positive Valence = Freude
    params_pos = valence_arousal_to_light(valence=1.0, arousal=0.5)
    assert params_pos["hue"] < 20000, f"Erwartet Warm-Weiss, bekam hue={params_pos['hue']}"
