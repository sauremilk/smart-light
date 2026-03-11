# Copilot Instructions for smart-light

## Goal
Keep changes measurable and regression-safe. Prefer minimal diffs and preserve runtime stability.

## Fast Start Checklist
1. Use the project venv interpreter: `c:/Users/mickg/smart-light/.venv/Scripts/python.exe`
2. Run baseline before meaningful edits:
  - `c:/Users/mickg/smart-light/.venv/Scripts/python.exe benchmarks/reference_suite.py --profile quick --preflight`
3. After meaningful changes, rerun quick profile and compare component deltas.
4. Before handoff, run:
   - `c:/Users/mickg/smart-light/.venv/Scripts/python.exe benchmarks/reference_suite.py --profile standard --enforce-gate`

## Codebase Conventions
- Use package imports from `core` and `analyzers`:
  - `from core.<module> import ...`
  - `from analyzers.<module> import ...`
- Avoid bare root-module imports like `from light_mapping import ...`.
- Keep optional CLI/runtime flags explicit and backwards compatible.

## Testing and Validation
- For quick checks: `c:/Users/mickg/smart-light/.venv/Scripts/python.exe -m pytest tests -q`
- Do not claim performance/quality improvements without updated report:
  - `benchmarks/results/reference_suite_latest.json`
- Include benchmark evidence in summaries (composite and per-component deltas).

## Safety
- Do not weaken or bypass benchmark gate logic to make runs pass.
- Avoid destructive git commands unless explicitly requested.
