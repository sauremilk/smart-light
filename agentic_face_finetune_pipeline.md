# Agentic Face Fine-Tune Pipeline

This file is the single master prompt and execution contract for fully automated face fine-tuning in this repository.

## Master Prompt

Use this in Copilot Chat / Claude Agent / Workspace Agent:

```text
You are the MLOps agent for smart-light. Execute execute_agentic_face_finetune_pipeline.py end-to-end with no manual steps.

Constraints:
- Use Python: c:/Users/mickg/smart-light/.venv/Scripts/python.exe
- Run baseline benchmark first: benchmarks/reference_suite.py --profile quick
- Generate dataset with >=1000 samples into dataset/face_finetune
- Train emotion classifier for 7 classes and stop early when val_accuracy >= 0.82
- Export ONNX artifact to artifacts/face_finetune/face_finetuned.onnx
- Auto-update config.py with ONNX path
- Run reference suite with gate enforcement using strict profile
- If gate fails, retrain with lr/2 and rerun benchmark once
- Produce pipeline summary JSON in artifacts/face_finetune/pipeline_summary.json
- If final gate passes, create commit and PR title: Agentic Face-Finetune v2.0
```

## One-Command Execution

```powershell
c:/Users/mickg/smart-light/.venv/Scripts/python.exe execute_agentic_face_finetune_pipeline.py --benchmark-profile strict --retry-on-fail
```

## Script Roles

- `agentic_dataset_gen.py`
  - Captures webcam samples for 7 emotions
  - Optional voice guidance
  - Self-supervision + clustering refinement
  - Augmentation and train/val split
  - Gate: dataset size >= `--min-total-samples` (default: 1000)

- `finetune_face_agentic.py`
  - Tries AutoTrain CLI first (if available)
  - Falls back to torchvision training
  - Exports ONNX and updates `config.py`
  - Gate: `val_accuracy >= --target-val-acc` (default: 0.82)

- `execute_agentic_face_finetune_pipeline.py`
  - Orchestrates dataset -> train -> benchmark gate
  - Optional automatic retry with lower learning rate
  - Writes final summary JSON and exits non-zero on gate failure

## Notes

- This pipeline is autonomous, but hardware dependencies still apply (webcam, optional GPU).
- Current benchmark governance in `AGENTS.md` remains mandatory.
- If AutoTrain is not installed, fallback training is used automatically.
