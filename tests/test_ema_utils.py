from core.ema_utils import normalize_vector_inplace, update_ema_vector_inplace


def test_update_ema_vector_inplace_keeps_distribution_normalized():
    emotions = ["happy", "sad", "neutral"]
    ema = {"happy": 1 / 3, "sad": 1 / 3, "neutral": 1 / 3}
    scores = {"happy": 100.0, "sad": 0.0, "neutral": 0.0}

    update_ema_vector_inplace(ema, scores, alpha=0.2, emotions=emotions)

    assert abs(sum(ema.values()) - 1.0) < 1e-9
    assert ema["happy"] > ema["sad"]
    assert ema["happy"] > ema["neutral"]


def test_update_ema_vector_inplace_alpha_zero_leaves_values_unchanged():
    emotions = ["happy", "sad", "neutral"]
    ema = {"happy": 0.2, "sad": 0.3, "neutral": 0.5}
    original = ema.copy()
    scores = {"happy": 100.0, "sad": 0.0, "neutral": 0.0}

    update_ema_vector_inplace(ema, scores, alpha=0.0, emotions=emotions)

    assert ema == original


def test_normalize_vector_inplace_noop_for_zero_total():
    emotions = ["happy", "sad", "neutral"]
    vector = {"happy": 0.0, "sad": 0.0, "neutral": 0.0}

    normalize_vector_inplace(vector, emotions)

    assert vector == {"happy": 0.0, "sad": 0.0, "neutral": 0.0}
