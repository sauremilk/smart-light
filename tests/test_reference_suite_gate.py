from benchmarks.reference_suite import _gate


def _current_report(index: int, score01: float = 0.5, benchmark: str = "reference_suite_v2") -> dict:
    return {
        "benchmark": benchmark,
        "composite": {"index": index},
        "components": {
            "extreme_visual_robustness": {"score01": score01},
        },
    }


def _baseline_report(index: int, score01: float = 0.5, benchmark: str = "reference_suite_v2") -> dict:
    return {
        "benchmark": benchmark,
        "composite": {"index": index},
        "components": {
            "extreme_visual_robustness": {"score01": score01},
        },
    }


def test_gate_fails_on_benchmark_mismatch_by_default():
    gate = _gate(
        current=_current_report(index=520, benchmark="reference_suite_v2"),
        baseline=_baseline_report(index=520, benchmark="reference_suite_v1"),
        max_composite_drop=15,
        max_component_drop=0.04,
        fail_on_benchmark_mismatch=True,
    )

    assert gate["pass"] is False
    assert gate["benchmark_compatible"] is False
    assert any("mismatch" in line.lower() for line in gate["failures"])


def test_gate_can_warn_on_mismatch_when_override_enabled():
    gate = _gate(
        current=_current_report(index=520, benchmark="reference_suite_v2"),
        baseline=_baseline_report(index=520, benchmark="reference_suite_v1"),
        max_composite_drop=15,
        max_component_drop=0.04,
        fail_on_benchmark_mismatch=False,
    )

    assert gate["benchmark_compatible"] is False
    assert gate["pass"] is True
    assert gate["failures"] == []
    assert any("mismatch" in line.lower() for line in gate["warnings"])


def test_gate_fails_on_component_regression():
    gate = _gate(
        current=_current_report(index=520, score01=0.10),
        baseline=_baseline_report(index=520, score01=0.30),
        max_composite_drop=15,
        max_component_drop=0.04,
        fail_on_benchmark_mismatch=True,
    )

    assert gate["pass"] is False
    assert any("component regression" in line.lower() for line in gate["failures"])


def test_gate_ignores_regression_for_skipped_component():
    gate = _gate(
        current=_current_report(index=400, score01=0.10),
        baseline=_baseline_report(index=520, score01=0.30),
        max_composite_drop=15,
        max_component_drop=0.04,
        fail_on_benchmark_mismatch=True,
        skipped_components=["extreme_visual_robustness"],
    )

    assert gate["pass"] is True
    assert gate["failures"] == []
    assert any("partial-check" in line.lower() for line in gate["warnings"])


def test_gate_skips_composite_drop_when_any_component_skipped():
    gate = _gate(
        current=_current_report(index=100, score01=0.50),
        baseline=_baseline_report(index=520, score01=0.50),
        max_composite_drop=15,
        max_component_drop=0.04,
        fail_on_benchmark_mismatch=True,
        skipped_components=["test_quality"],
    )

    assert gate["pass"] is True
    assert gate["failures"] == []
    assert any("composite drop check disabled" in line.lower() for line in gate["warnings"])
