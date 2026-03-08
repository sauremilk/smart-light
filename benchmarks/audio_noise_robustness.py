#!/usr/bin/env python3
"""Synthetic SNR sweep for audio quality and dynamic fusion weighting."""

from __future__ import annotations

import json
import os
import sys
import time

try:
    from audio_quality import effective_audio_weight, quality_from_snr_db
    from config import (
        AUDIO_WEIGHT,
        AUDIO_SNR_DB_FLOOR,
        AUDIO_SNR_DB_CEIL,
        AUDIO_DYNAMIC_MIN_FACTOR,
        AUDIO_DYNAMIC_QUALITY_EXPONENT,
    )
except ModuleNotFoundError:
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    from audio_quality import effective_audio_weight, quality_from_snr_db
    from config import (
        AUDIO_WEIGHT,
        AUDIO_SNR_DB_FLOOR,
        AUDIO_SNR_DB_CEIL,
        AUDIO_DYNAMIC_MIN_FACTOR,
        AUDIO_DYNAMIC_QUALITY_EXPONENT,
    )


def run_snr_sweep() -> dict:
    snr_levels = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    rows = []
    for snr in snr_levels:
        quality = quality_from_snr_db(snr, AUDIO_SNR_DB_FLOOR, AUDIO_SNR_DB_CEIL)
        # Use quality as confidence proxy for synthetic sweep.
        dynamic_weight = effective_audio_weight(
            base_weight=AUDIO_WEIGHT,
            audio_quality=quality,
            audio_confidence=quality,
            min_factor=AUDIO_DYNAMIC_MIN_FACTOR,
            quality_exponent=AUDIO_DYNAMIC_QUALITY_EXPONENT,
        )
        rows.append(
            {
                "snr_db": snr,
                "quality": quality,
                "audio_weight_dynamic": dynamic_weight,
            }
        )

    monotonic = all(
        rows[i + 1]["audio_weight_dynamic"] >= rows[i]["audio_weight_dynamic"]
        for i in range(len(rows) - 1)
    )

    return {
        "benchmark": "audio_noise_robustness_v1",
        "generated_at": time.time(),
        "config": {
            "audio_weight": AUDIO_WEIGHT,
            "snr_floor_db": AUDIO_SNR_DB_FLOOR,
            "snr_ceil_db": AUDIO_SNR_DB_CEIL,
            "dynamic_min_factor": AUDIO_DYNAMIC_MIN_FACTOR,
            "quality_exponent": AUDIO_DYNAMIC_QUALITY_EXPONENT,
        },
        "rows": rows,
        "monotonic_dynamic_weight": monotonic,
    }


def main() -> int:
    out = run_snr_sweep()
    path = os.path.join("benchmarks", "results", "audio_noise_robustness.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== Audio Noise Robustness ===")
    for row in out["rows"]:
        print(
            f"SNR {row['snr_db']:>5.1f} dB  | quality={row['quality']:.3f} "
            f"| dyn_weight={row['audio_weight_dynamic']:.3f}"
        )
    print(f"Monotonic dynamic weight: {out['monotonic_dynamic_weight']}")
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
