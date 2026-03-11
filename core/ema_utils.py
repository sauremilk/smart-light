"""Shared EMA utilities for emotion probability vectors."""


def normalize_vector_inplace(vector: dict, emotions: list[str]) -> None:
    """Normalize vector values to a sum of 1.0 in-place."""
    total = sum(vector.values())
    if total > 0:
        for emotion in emotions:
            vector[emotion] /= total


def update_ema_vector_inplace(
    ema: dict,
    scores_percent: dict,
    alpha: float,
    emotions: list[str],
) -> None:
    """Update EMA in-place using score percentages and normalize afterwards."""
    for emotion in emotions:
        new_val = scores_percent.get(emotion, 0.0) / 100.0
        ema[emotion] = (1.0 - alpha) * ema[emotion] + alpha * new_val

    normalize_vector_inplace(ema, emotions)
