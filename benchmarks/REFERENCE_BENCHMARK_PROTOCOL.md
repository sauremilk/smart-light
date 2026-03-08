# Reference Benchmark Protocol

## Purpose

This protocol defines how benchmark evidence is collected, compared, and accepted.
The goal is to improve the full system, not isolated metrics.

## Benchmark Scope

The reference suite is `benchmarks/reference_suite.py`.
It evaluates five components:

1. `extreme_visual_robustness`
- Very hard perturbation benchmark using `benchmarks/extreme_reference_benchmark.py`.
- Covers low light, blur, noise, occlusion, compression artifacts, geometry, and mixed stress.

2. `multi_seed_stability`
- Seed-based quality and variance check using `benchmarks/accuracy_benchmark.py`.
- Penalizes instability explicitly.

3. `test_quality`
- Executes local project tests.
- Guards against functional regressions in core logic.

4. `module_sanity`
- Contract-level checks for multimodal fusion, regulation, HRV math, breathing signal logic,
  and face-mesh emotion mapping.

5. `e2e_runtime`
- Runs `main.py` in deterministic mock scenarios and measures control-loop cadence,
  tail latency, process CPU, and memory drift.

## Scoring Model

Composite score is a weighted sum of component scores in [0, 1]:

- `extreme_visual_robustness`: 45%
- `multi_seed_stability`: 23%
- `test_quality`: 17%
- `module_sanity`: 10%
- `e2e_runtime`: 10%

Final index = `round(composite * 1000)`.

Formula details:

```text
extreme_weighted_score = 0.4 * accuracy + 0.6 * macro_f1
extreme_index = round(clamp01(extreme_weighted_score) * 1000)

stability_seed_score = 0.4 * enhanced_accuracy + 0.6 * enhanced_macro_f1
stability_score = clamp01(mean(stability_seed_score) - 0.5 * std(stability_seed_score))

composite =
    0.40 * extreme_visual_robustness
    + 0.23 * multi_seed_stability
    + 0.17 * test_quality
    + 0.10 * module_sanity
    + 0.10 * e2e_runtime
composite_index = round(clamp01(composite) * 1000)
```

Rationale:
- Strong emphasis on hard robustness and stable generalization.
- Tests and sanity checks prevent brittle overfitting to one benchmark.

## Run Profiles

- `quick`: fast feedback during development.
- `standard`: mandatory before merge/handover.
- `strict`: release-grade verification.

Commands:

```powershell
# quick
.\venv\Scripts\python benchmarks\reference_suite.py --profile quick

# standard + gate
.\venv\Scripts\python benchmarks\reference_suite.py --profile standard --enforce-gate

# strict + gate
.\venv\Scripts\python benchmarks\reference_suite.py --profile strict --enforce-gate
```

## Gate Policy

Baseline file:
- `benchmarks/results/reference_suite_baseline.json`

Gate default constraints:
- Composite index drop must be <= 15 points.
- Any component score drop must be <= 0.04.
- Baseline benchmark schema/version must match current run (`benchmark` field).

Schema mismatch policy:
- A schema/version mismatch is treated as **gate failure by default**.
- Rationale: mismatch can hide regressions and invalidate direct comparisons.
- One-time diagnostic override exists, but must not be used for merge/release decisions:

```powershell
.\venv\Scripts\python benchmarks\reference_suite.py --profile quick --allow-incompatible-baseline
```

Failure means the change is not benchmark-safe yet.

## Baseline Management

Baseline updates are controlled and rare.
Only update when all conditions are true:

1. Strict profile passes gate.
2. Composite and critical components improve or remain stable.
3. Change notes include clear causal explanation.
4. No unresolved test regressions.

When to refresh baseline intentionally:

1. Benchmark schema changed (for example `reference_suite_v1` -> `reference_suite_v2`).
2. Strict run completed and reviewed with full evidence.
3. Explicit approval exists for baseline rollover.

Recommended rollover sequence:

```powershell
# 1) verify with strict gate first
.\venv\Scripts\python benchmarks\reference_suite.py --profile strict --enforce-gate

# 2) only after approval, write new baseline
.\venv\Scripts\python benchmarks\reference_suite.py --profile strict --write-baseline

# 3) re-run to validate new baseline compatibility
.\venv\Scripts\python benchmarks\reference_suite.py --profile standard --enforce-gate
```

Update command:

```powershell
.\venv\Scripts\python benchmarks\reference_suite.py --profile strict --write-baseline
```

## Evidence Requirements

Every benchmark claim must include:

- profile used (`quick`, `standard`, `strict`)
- report path
- composite index before/after
- per-component before/after
- gate result and reasons
- sample sizes (`n`) for seed-based statistics
- 95% confidence interval context for multi-seed metrics

The report artifact is:
- `benchmarks/results/reference_suite_latest.json`

Run history artifact:
- `benchmarks/results/reference_suite_history.jsonl`

Each run appends one JSON line with timestamp, profile, detector, composite index,
component indices, and gate status. The latest report includes a `trend` section
with deltas against the last comparable run.

Latest report also includes:
- `environment`: OS/Python/lib versions and CUDA/GPU metadata when available
- `runtime`: total run time and per-component durations
- `components.multi_seed_stability.details.*_stats`: mean/std/sem/95%-CI

Interpretation policy:
- Treat improvements smaller than CI half-width as likely noise.
- Escalate to stricter profile or larger sample count when deltas are marginal.

## Perturbation Realism Disclosure

The extreme benchmark report includes `profile_specs` with parameter ranges and transformation semantics
for all perturbation profiles (`low_light_noise`, `motion_blur`, `jpeg_artifacts`, `occlusion`,
`rotation_scale`, `color_cast_shadow`, `mixed_extreme`).

This disclosure makes stress scenarios auditable and easier to compare to real webcam usage patterns.

## Anti-Gaming Rules

- No acceptance if one metric improves while another critical component regresses.
- No cherry-picked seed reporting; profile seed set is fixed by preset.
- No benchmark-only tuning without test stability.

## Recommended Cadence

- Every meaningful code change: `quick`.
- Before merge/handover: `standard --enforce-gate`.
- Daily or release candidate: `strict --enforce-gate`.

## Windows Convenience Runner

Use:

```powershell
.\benchmarks\run_reference_suite.ps1 -Mode standard -EnforceGate
```

This wrapper sets stable TensorFlow log/runtime environment variables before execution.
