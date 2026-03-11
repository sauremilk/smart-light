# Agent Benchmark Governance

This repository uses a benchmark-first workflow. Every coding agent must use the reference suite regularly and optimize for measurable improvements.

## Required Workflow

1. Before implementation:

- Run a quick baseline check:
- `c:/Users/mickg/smart-light/.venv/Scripts/python.exe benchmarks/reference_suite.py --profile quick --preflight`

2. During implementation:

- After meaningful changes, rerun quick profile.
- If composite index or any component regresses, fix before continuing.

3. Before handoff / merge:

- Run standard profile with gate enforcement:
- `c:/Users/mickg/smart-light/.venv/Scripts/python.exe benchmarks/reference_suite.py --profile standard --enforce-gate`

4. Release-quality verification:

- Run strict profile:
- `c:/Users/mickg/smart-light/.venv/Scripts/python.exe benchmarks/reference_suite.py --profile strict --enforce-gate`

5. Baseline management:

- Update baseline only with explicit approval and evidence:
- `c:/Users/mickg/smart-light/.venv/Scripts/python.exe benchmarks/reference_suite.py --profile strict --write-baseline`

## Non-Negotiable Rules

- Do not claim improvements without a new JSON report in `benchmarks/results/reference_suite_latest.json`.
- Do not optimize only a single metric. The target is the composite index and all components.
- Do not accept regressions masked by variance. Multi-seed stability is part of the score.
- Do not skip tests when changing core logic. Test quality is a weighted benchmark component.

## Evidence Standard

Every agent change summary should include:

- reference suite profile used (`quick`, `standard`, or `strict`)
- previous and current composite index
- per-component index deltas
- gate result (`PASS` or `FAIL`)
- report path(s)
- history path (`benchmarks/results/reference_suite_history.jsonl`) and trend summary
