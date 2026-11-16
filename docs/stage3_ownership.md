# Stage 3 implementation handoff and reserved interfaces

16 November 2026; planning only. Read AGENTS.md, plan.md, prelim.md, stage3_contracts.md and accepted Stage 2 interfaces. This document creates no code, branch, chat or executable API. The current task authorizes planning; implementation is separate.

## Ownership and sequencing

| Future workstream | Proposed branch | Owned area |
|---|---|---|
| Coordinator foundation | phase1 | src/stage3 shared schemas/plan compiler, verified numeric sessions, source statistics, canonical logistic fitting engine, numeric serialization/cache readers, selection primitives and artifact registries; prepare_stage3.ipynb and validate_stage3.ipynb; shared numerical/identity/role tests; integration docs |
| Ordinary-fit worker | phase1-stage3-fits | src/stage3/fit_readouts.py, scripts/stage3/fit_readouts.ipynb, ordinary-fit orchestration/resume tests and implementation notes; no private second solver/scaler |
| G0 worker | phase1-stage3-g0 | src/stage3/run_g0.py, scripts/stage3/run_g0.ipynb, assignment/null-family/gate orchestration and tests; uses the identical shared fitting/selection engine |

The coordinator implements and tests the numerical engine first, freezes public callable semantics at a clean committed phase1 base and supplies its exact SHA. Only then do ordinary/G0 workers implement on separate branches/worktrees from that base. A planning commit is not that executable foundation. Both workers can proceed in parallel after shared source-statistics/assignment ownership is agreed; CPU resources, input memory and Drive traffic determine production concurrency. No worker alters a frozen common helper, Stage 2 reader, scientific protocol or objective independently. Report needed changes for coordination; final integration review accepts actual evidence before stage completion.

Ordinary and control orchestration differ in their target-assignment/reader permissions, not their solver, C grid or weighting. There is no separate Qwen/Gemma fitting implementation and no separate E03 solver. A FEATURE_SOURCE selector distinguishes vlm from e03; MODEL is required only for VLM input groups and source/E03 identity is explicit. E03 is run once, not once per model. G0 worker settings select an explicit source-group list and seed subset only for distributed ownership; the final registry still requires all fixed groups/seeds. No configuration switch permits CONFIRM or ordinary SCREEN fitting.

## Reserved shared interfaces

These names/semantics are specified for a future src/stage3/interfaces.py. The coordinator publishes their actual types and exact payload fields with implementation, before worker coding:

| Planned callable | Contract |
|---|---|
| compile_fit_plan(suite_manifest, root, protocol, producer) | Verify passed complete Stage 2 suite and immutable children, fix 145 expected source groups, roles, all C/true/control blocks, target/scaler/solver/null recipes and dependencies; publish Stage 3 plan without fitting |
| open_numeric_session(plan, root, groups, role) | Verify permitted complete numeric input releases and target/mask identities at open; return immutable bounded cached matrices/rows with cheap role checks on every access; no raw images or unsafe path-only cache |
| load_or_publish_statistics(plan, group_id, session, producer, supplied_refs) | Reuse an exactly matching completed source-FIT statistics artifact, otherwise compute/write under explicit single-producer ownership; return typed centering/means/variance/scales/weight identity |
| publish_target_assignments(plan, root, producer) | Publish deterministic true/control image-vector/mask donor maps per eligible role/partition, with no model/language dependence or CONFIRM rows |
| fit_candidate_block(plan, group_id, assignment_id, C, session, statistics, producer) | Fit all estimable category heads using the one canonical engine, compute required source-DEV-S diagnostics, validate/reload finite arrays and publish one complete C block; blocked attempts are not fitted systems |
| select_system(plan, group_id, assignment_id, candidate_refs, producer) | Verify all four exact C blocks, fixed-cohort DEV-S scores and tie rule; publish immutable selected-system reference without refitting |
| load_fitted_system(selection_manifest, root, plan, role, expected) | Verify full identities, original candidate/statistics/target/column order/flags and role; return an immutable numeric scorer that applies the frozen source transform and head |
| score_g0(plan, selected_control_refs, session, producer) | Generate only the declared control SCREEN scores after selection; preserve grouping, masks, true/control separation and priors |
| evaluate_g0(plan, control_score_refs, assignment_refs, resume_refs, producer) | Complete structural checks and the frozen conditional global null calculation in 100-draw blocks; publish diagnostics and a gate receipt with unresolved flags blocking acceptance |
| validate_stage3(plan, candidate_registry, selection_registry, g0_receipt, producer) | Check exact inventory/compatibility/convergence/masks/selection/gate/readback; publish a passed Stage 3 system registry only for complete valid inputs |

No callable is implemented by writing this table. The shared implementation must expose reliable error/status types, exact little-endian numeric layouts and clear immutable artifact/reference return values. Workers import these functions rather than duplicating objective/scaler math or bypassing Stage 2 verification through private APIs.

## Notebook settings and handoff

All planned notebooks start with PHASE1_ROOT, a single explicit accepted plan/suite reference, declared group/model ownership, storage scope and optional resume-index list. Preparation resolves compatible Stage 2 features/targets and full manifests; workers do not repeatedly copy raw config/image/feature paths. Include a CPU-only reproducible bootstrap and actual pinned package/thread settings; no L4/Torch installation or HF token is needed. Default worker concurrency is one process/one BLAS thread. Any distributed group/C ownership is explicit and disjoint; validate the complete registry, not just the local shard.

Preparation emits the complete source-group/job inventory, counts and missing dependencies. Work loops keep one group's input/scaled working arrays in memory and reuse its persisted small statistics. Completed C-block/scaler/control artifacts are read through typed cache interfaces. No per-category checkpoint flood, mutable latest folder or repeated growing full-job inventory. Report a final reference and counts for every complete group; print failure/partial diagnostics without claiming successful fits. Do not copy selected weights into every consumer's directory.

Each handoff gives immutable base/head, changed owned files, exact environment/threads/objective settings, meaningful checks, actual versus fixture artifacts, fit/selection/control counts, failure/retry details, independent readback and unresolved prerequisites. Use explicit artifact refs, never raw tokens or user-specific machine details in public reports. No worker merges into phase1 or marks Stage 3 complete without coordinator integration and the full accepted registry/gate.

## Acceptance sequence

1. Bind the passed Stage 2 suite, preserving all original source provenance; compile and review Stage 3 numerical/role/identity inventory before outcomes.
2. Freeze and publish tested shared numerical/cache interfaces. Verify dense/sparse objective, fractional image weighting and mask/role invariants independently.
3. Obtain small actual first-source C=0.01 true/seed-0 production units and measurements, reuse those units and publish resource projections.
4. Run ordinary and matched G0 blocks with bounded checkpointing; consume/reuse common statistics and explicit completed artifacts across scripts/models.
5. Freeze all selections, complete G0 conditional randomization/structural diagnostics and resolve any validity flag without seed rerolling or silent exclusions.
6. Validate complete records/readback and publish the registry for Stage 4. Stage 4 ordinary scoring, E19/E25 and final scientific interpretation are not run by this stage.

All implementation/runtime items are pending. CPU component successes reported for Stage 2 are not a substitute for its final accepted suite or for actual Stage 3 fitting/controls.
