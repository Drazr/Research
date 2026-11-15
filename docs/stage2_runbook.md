# Integrated Stage 2 notebook order

Use a clean integrated phase1 checkout. All four notebooks are under scripts/stage2; shared implementations and workers are under src/stage2. Open the notebooks in Colab and run cells in order. The first setup cell mounts the intended Drive, obtains the checkout, fixes imports and installs CPU or GPU dependencies. Configure paths there if using another environment. No existing checkout is silently reset/pulled. Update between jobs if it predates the worker integration; never update an active producer.

## 1. Prepare plans on CPU

Run prepare_stage2.ipynb with PHASE1_ROOT=/content/drive/MyDrive/phase1. The existing dataset-manifest suffix is prefilled. Default BINDINGS={} produces the shared CPU plan independently. Retain its printed completion path as SHARED_PLAN.

To add model plans, reuse that SHARED_PLAN in the same preparation notebook and supply BINDINGS for qwen and gemma. Each binding needs configuration_manifest pointing to the _COMPLETE.json in the successful configuration artifact folder, not its models.json payload. The currently recorded models.json paths are in AGENTS.md: use each file's sibling _COMPLETE.json. Leave model_validation and smoke null to use the typed, compatible discovery described in stage2_interfaces.md, or supply exact completion paths explicitly. Never enter HF tokens in bindings. Receipt verification and model binding load no model weights. Check each model's reported status; a published shared plan does not mean both model bindings passed.

Keep the printed shared plan for E03 and the separate printed per-model plan for extraction. Workers resolve dataset/config/readiness paths from those plans. Do not copy stale model paths from older runs into worker settings.

## 2. E03 on CPU, once for both models

Open prepare_e03.ipynb, run setup, set SHARED_PLAN to the accepted shared completion manifest and leave RESUME_INDEX=None for the first run. Default THROUGH_SPLIT=DEV-S publishes all 13 families' FIT/DEV-S inputs promptly. Save the printed checkpoint completion path. To finish, set RESUME_INDEX to that checkpoint and THROUGH_SPLIT=SCREEN and rerun execution. Completed FIT preprocessing is reused without refitting; DEV-C/SCREEN are only transformed. A fresh runtime must install the same profile and reproduce the recorded preprocessing versions.

This is model-independent work. It can run alongside GPU extraction after shared data acceptance. No base model, new human annotation, classifier fitting, calibration or scientific outcome scoring is involved. Preserve checkpoint references on failure; never use an arbitrary newest directory for resume.

## 3. Extract per model on L4

Open extract_features.ipynb, select MODEL=qwen or gemma and its matching MODEL_PLAN completion path, then run setup/preflight. For Gemma enable that notebook's access to the authorized HF_TOKEN secret and run authentication. Set RESUME_INDEXES only to explicit prior progress completion paths when resuming. Execution repeats preflight and compares actual loaded runtime identity to the bound plan before any production forward.

The worker runs all three conditions with shared E01/E04 forwards, required token/visual/repeat audits, first-chunk resources, immutable progress and complete condition/split releases. Preserve printed progress paths, resource_report and feature_releases. On one physical L4 queue models; distinct L4s can run separate model jobs. CPU E03 can overlap either case. Runtime failures do not authorize truncation, changed precision/sites, cohort drops or fabricated vectors.

## 4. Validate the suite on CPU

Run validate_stage2.ipynb with an explicit INVENTORY containing the shared data_plan, both model_plans, 12 condition/split feature_releases per model, one resource_report per model and all 13 complete text_families. Use final E03 family indexes after SCREEN transformation, not the earlier partial indexes. Missing or incompatible entries produce blocked status. A passed result is Stage 2 artifact acceptance, not full Stage 3 G0 completion or research findings.

See stage2_interfaces.md for exact callable/payload contracts, stage2_contracts.md for the frozen scientific/operational scope, and reports/stage2_integration.md for what was actually verified locally and what still requires execution.
