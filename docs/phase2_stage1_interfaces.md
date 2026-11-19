# Phase 2 Stage 1 executable interfaces and ownership

Interface freeze: `phase2-stage1.v1`, 19 November 2026. Implements [phase2.md §1](phase2.md#stage-1-finalized-specification--19-november-2026). Import shared constants and callables from `src.phase2.interfaces`.

Chats 2, 3, and 4 branch from the immutable foundation commit (`STAGE1_BASE_SHA`) that first adds this document and `src/phase2/interfaces.py`. Do not modify Coordinator-owned `src/phase2/` modules or any Phase 1 files in worker branches.

## Ownership

| Chat | Exclusive owned files |
|---|---|
| Coordinator (Chat 1) | `src/phase2/{__init__,schema,inputs,interfaces}.py`, `docs/phase2_stage1_interfaces.md`, `tests/test_phase2_foundation.py` |
| E07 audit worker (Chat 2) | `src/phase2/e07_audit.py`, `scripts/phase2/audit_e07.ipynb`, `tests/test_phase2_e07_audit.py` |
| E17 DEV-C preflight worker (Chat 3) | `src/phase2/e17_auth.py`, `scripts/phase2/preflight_e17.ipynb`, `tests/test_phase2_e17_auth.py` |
| E29 gate & Stage 1 plan worker (Chat 4) | `src/phase2/{gates,plan,operations}.py`, `scripts/phase2/prepare_phase2.ipynb`, `tests/test_phase2_gates_plan.py` |
| Integration (Chat 5) | Merge commits, `tests/test_phase2_stage1_integration.py`, Stage 1 handoff note in `phase2.md` |

## Public imports (`src.phase2.interfaces`)

- **Constants:** `PHASE2_VERSION`, `PHASE2_NAMESPACE`, `MODELS`, `POOLINGS`, `CONDITIONS`, `ALLOWED_SPLITS`, `CLOSED_SPLITS`, `PHASE2_PARENT_DIRS`, `STATUSES`, `SUPERVISION_LABELS`, `E07_VERDICTS`, `GATE_DECISIONS`, `PHASE1_STAGE4_RELEASE`, `PHASE1_STAGE5_SYNTHESIS`, plus all `KIND_*` and `SCHEMA_*` identifiers.
- **Storage & guards:**
  - `require_open_split(split)` — rejects `"CONFIRM"` and unknown splits before IO.
  - `require_phase2_namespace(namespace)` — enforces writes strictly under `PHASE1_ROOT/phase2/`.
  - `init_phase2_dirs(root)` — pre-creates all `PHASE2_PARENT_DIRS` once before concurrent writers.
  - `publish_phase2(root, protocol, dataset_fingerprint, namespace, kind, body, producer, *, arrays=None, files=None, upstream=(), model_fingerprint=None, seeds=None, run_id=None)` — writes `body.json` + optional `.npy`/raw files under `phase2/.../<run_id>/`, verifies completion marker/hashes on disk, returns immutable `{"path", "sha256"}`.
  - `load_phase2_artifact(root, protocol, reference, expected_kind, *, expected_schema=None, expected_dataset=None, expected_plan_identity=None, expected_owner=None)` — returns `(ref, frozen_body, frozen_marker)`.
  - `verify_resume_ref(..., expected_inputs=None)` — explicit-reference resume check; rejects missing, corrupt, foreign-owner, plan-mismatched, or input-mismatched artifacts.
  - `storage_probe(root, protocol, dataset_fingerprint, producer)` and `verify_storage_probe(root, protocol, probe_ref, dataset_fingerprint)` — two-process readback receipt under `phase2/probes/`.
  - `current_producer(scope="local-development")` — delegates to Stage 3's pinned CPU runtime/git producer metadata helper.
- **Read-only Phase 1 binder:**
  - `bind_phase1(root, protocol, stage4_release_ref=None) -> Phase1Binding` — calls `src.stage5.interfaces.open_release` and `load_bootstrap`, verifies unflagged G0, confirms CONFIRM absence, and resolves Stage 1 dataset manifest without opening raw images.
  - `midpoint_depth(binding, model)` / `midpoint_groups(binding, model, *, condition="image_caption", pooling="last_caption")` — resolves midpoint depth from `src.stage4.plan.DEPTHS[model][1]` (Qwen 18, Gemma 17 in production; supports patched depths in fixtures).
  - `e29_screen_blocks(binding, model)` — resolves all 16 accepted Stage 4 SCREEN midpoint blocks per model (`image_caption` and `caption_only`, both poolings).
  - `e17_inputs(binding, model)` — resolves 4 reused SCREEN blocks, 4 reused DEV-C diagonals, 4 Stage 2 DEV-C feature releases, 4 Stage 3 selected source systems (`selection_ref`, `candidate_ref`, `C`, `statistics_ref`), and the 12 off-diagonal `(target_lang, source_lang)` targets.
  - `e11_fit_statistics_refs(binding, model)` — resolves the 4 reused Stage 3 FIT `statistics_ref` pointers and SCREEN `caption_mean` refs per model.
  - `stage1_caption_paths(binding)` — verifies the `stage1_dataset` marker and returns verified relative paths + SHA-256s for `dataset.json`, `records.jsonl`, `targets.jsonl`, `splits.json`, and `captions/{en,hi,bn,te}.jsonl` (no image IO).
  - `read_verified_midpoint_features(binding, model, condition, pooling, target_lang, split)` — enforces `require_open_split` and reads verified numeric features via Stage 3's `_read_verified_group`.
  - `load_frozen_source_head(binding, model, condition, pooling, source_lang)` — loads `(coef, intercept, estimable, prevalence, stats)` via Stage 4's `_selected_parameters` without opening FIT arrays.

## Frozen cross-worker JSON schemas

Workers publish and validate artifacts through these shared top-level body keys so Chat 4 (`plan.py`, `operations.py`) validates Chat 2 and Chat 3 outputs purely by schema without cross-importing worker modules.

### `phase2_e07_audit` (`phase2/audits/e07/<run_id>/`)
- `kind`: `KIND_E07_AUDIT` (`"phase2_e07_audit"`), `body["schema"]`: `SCHEMA_E07_AUDIT` (`"phase2.e07_audit.v1"`).
- Required body keys: `schema`, `stage4_release`, `lineage_identity`, `dataset_ref`, `dataset_fingerprint`, `draw0_verified` (`True`), `confirm_excluded` (`True`), `verdict` (`"passed_full" | "passed_restricted" | "blocked"`), `verdict_reasons` (list of strings), `screen_multiplicity`, `fit_dev_s_multiplicity`, `draw_buckets` (per language & draw 1/2 counts of `same_id_as_draw0`, `distinct_id_same_text_as_draw0`, `changed_text_vs_draw0`), `support_tables` (full 360 SCREEN, per-language `S_changed`, 4-language intersection), `unique_changed_views` (counts per language & total), `extraction_plan` (32-image SCREEN-only chunks if `verdict != "blocked"`, else `None`).

### `phase2_e17_auth` (`phase2/plans/<run_id>/`)
- `kind`: `KIND_E17_AUTH` (`"phase2_e17_auth"`), `body["schema"]`: `SCHEMA_E17_AUTH` (`"phase2.e17_auth.v1"`).
- Required body keys: `schema`, `stage4_release`, `lineage_identity`, `dataset_fingerprint`, `permission` (`"phase2.e17_dev_c_transfer"`), `confirm_excluded` (`True`), `models` (mapping `qwen`/`gemma` -> `{midpoint_depth, screen_blocks, dev_c_diagonals, off_diagonal_blocks, source_systems, diagonal_reproduction_check}` where `diagonal_reproduction_check` records `passed: True`, `max_abs_diff: 0.0`, `atol: 0.0`, `rtol: 0.0`), `calibrator_recipe` (forms, equal-weight 14-primary macro BCE, BFGS settings, `invariance_tol: 1e-12`, supervision labels).

### `phase2_e29_gate` (`phase2/results/e29_gates/<run_id>/`)
- `kind`: `KIND_E29_GATE` (`"phase2_e29_gate"`), `body["schema"]`: `SCHEMA_E29_GATE` (`"phase2.e29_gate.v1"`).
- Required body keys: `schema`, `plan`, `plan_identity`, `model`, `e29_analysis_ref`, `thresholds`, `observed` (`diagonal_min_auroc`, `valid_draws`, `delta_g_mean`, `delta_g_ci`, `g_res_mean`, `g_res_ci`, `directional_routes_triggered`), `decisions` (`{"E30": {"decision": ..., "reason": ...}, "E11": {"decision": ..., "reason": ..., "trigger": ...}}`).

### `phase2_plan` (`phase2/plans/<run_id>/`)
- `kind`: `KIND_PLAN` (`"phase2_plan"`), `body["schema"]`: `SCHEMA_PLAN` (`"phase2.plan.v1"`).
- Required body keys: `schema`, `phase2_version`, `plan_identity`, `lineage_identity`, `dataset_fingerprint`, `protocol_fingerprint`, `stage4_release`, `stage4_plan`, `stage3_registry`, `g0_gate`, `g0_p_value`, `stage2_suite`, `stage1_dataset`, `bootstrap` (`shape`, `sha256`), `midpoints`, `storage_probe`, `initialized_dirs`, `e29_blocks`, `e11_fit_statistics`, `gate_thresholds`, `stage5_decision_ranges`, `e07_audit` (optional ref or `None`), `e17_auth` (optional ref or `None`), `confirm_excluded` (`True`).
