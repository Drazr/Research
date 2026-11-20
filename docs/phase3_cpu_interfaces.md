# Phase 3 CPU Track Stage A1 interfaces and ownership

Interface freeze: `phase3-cpu-stageA1.v1`, 20 November 2026. Implements [phase3.md §A1](phase3.md#stage-a1-finalized-cpu-foundation-specification--20-november-2026). Import shared symbols from `src.phase3.interfaces`.

Stage A2 parallel workers (`E40`, `E41`, `E44`, `E12`) branch from the immutable Stage A1 foundation commit (`PHASE3_A1_SHA`) and must NOT modify `src/phase3/{__init__,schema,cpu_plan,interfaces}.py`, any Phase 1/2 files, or `CONFIRM`.

## Public imports (`src.phase3.interfaces`)

- **Constants:** `PHASE3_CPU_VERSION`, `PHASE3_NAMESPACE`, `MODELS`, `POOLINGS`, `CONDITIONS`, `ALLOWED_SPLITS`, `CLOSED_SPLITS`, `PHASE3_PARENT_DIRS`, `CERTIFIED_PHASE2_FOR_PHASE3_CPU`, `PHASE3_CPU_DECISION_RANGES`, plus `KIND_*` / `SCHEMA_*`.
- **Guards & storage:** `require_open_split`, `require_phase3_namespace`, `init_phase3_dirs`, `publish_phase3`, `load_phase3_artifact`, `verify_resume_ref`, `storage_probe`, `verify_storage_probe`, `normalize_ref`, `current_producer`.
- **Plan loader:** `prepare_phase3_cpu`, `load_phase3_cpu_plan(root, protocol, reference, *, bind=True) -> Phase3CpuPlan` (note `plan.binding` is already bound when `bind=True`).

## Frozen `phase3_cpu_plan` body keys

- `schema` (`"phase3.cpu_plan.v1"`), `phase3_version`, `plan_identity`, `lineage_identity`, `dataset_fingerprint`, `protocol_fingerprint`, `phase1_stage4_release`, `phase2_refs`, `bootstrap`, `depths`, `midpoints`, `storage_probe`, `initialized_dirs`, `e40_inventory`, `e41_inventory`, `e44_inventory`, `e12_inventory`, `preflight_checks`, `decision_ranges`, `confirm_excluded` (`True`).
