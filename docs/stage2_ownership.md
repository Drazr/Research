# Stage 2 ownership and implementation handoff

Dated 15 November 2026. Planning only: no chats, branches, worktrees, notebooks, modules or runtime artifacts are created by this document. Read AGENTS.md, plan.md, prelim.md and stage2_contracts.md. Stage 1 launch gates remain in force. The user requested specification without code in this turn.

## Shared base and ownership

Use phase1 as the integration branch and the coordinator's eventual immutable shared-interface implementation commit as the worker base. This planning commit is a reading/design base, not a frozen executable interface commit. Do not have workers branch from the old Stage 1 ea06d93 foundation. Before parallel implementation, the coordinator must supply and commit the Stage 2 schemas/readers/public interfaces consistent with the specification and publish their exact base SHA.

The following are future ownership assignments, not active agents:

| Owner/workstream | Proposed branch | Owned implementation area | Reviewable output |
|---|---|---|---|
| Coordinator | phase1 | Stage 2 plan/compiler, readiness receipt, row/site/job schemas, same-forward evidence bridge, release-index readers/validators, storage adapters using existing common helpers, integration notebook/report, shared tests and plan/AGENTS updates | Frozen interfaces first; final combined coverage/readback report later |
| VLM extraction chat | phase1-stage2-features | src/stage2/extract_features.py and feature-specific helpers; scripts/stage2/extract_features.ipynb; extraction tests and implementation notes | One MODEL-variable notebook; three conditions, pooled block hooks, visual audits, bounded shards/restart; per-model evidence |
| E03 CPU chat | phase1-stage2-e03 | src/stage2/prepare_e03.py and text-specific helpers; scripts/stage2/prepare_e03.ipynb; E03 tests and implementation notes | One model-independent notebook; 13 preprocessing families, FIT-only fitting, sparse/dense reloadable artifacts |

The coordinator's proposed entry points are scripts/stage2/prepare_stage2.ipynb and validate_stage2.ipynb, backed by reusable src/stage2 modules. They prepare/check metadata and indexes; they do not independently reimplement GPU or text preprocessing. A file path listed here is reserved ownership, not an implemented command. Do not create duplicated per-model/per-experiment notebooks where one MODEL selector suffices.

Changes to src/common/contracts.py, cache_keys.py, storage.py, configs/protocol.json, src/models or Stage 1 acceptance rules require coordinator integration and affected compatibility checks. Do not change them independently to fit a worker's own serialization. The VLM worker may propose a narrowly scoped adapter improvement with test evidence, but bulk runs must wait for affected actual-model revalidation. research.md, prelim.md, M1.md and M2.md remain scientific context; workers do not edit them to excuse implementation shortcuts.

## Interfaces to freeze before worker implementation

The coordinator implements a job-plan reader/validator with immutable expected row/chunk/site inventories, a readiness-receipt verifier, a feature-shard writer/reader and a complete-release-index reader. These preserve the common storage manifest and existing extraction/cache key semantics. The specification in stage2_contracts.md defines payload contents. The following public interfaces are reserved for the future coordinator-owned src/stage2/interfaces.py; annotations here are a specification, not existing code:

| Planned interface | Inputs and result |
|---|---|
| load_job_plan(plan_manifest, root, protocol) | Verify a completed plan artifact and common identities; return the immutable resolved plan mapping with ordered work units and required sites |
| verify_readiness(receipt_manifest, root, protocol, model_config) | Verify the referenced passed upstream data/config/model/smoke artifacts and their actual execution/site compatibility; return the resolved readiness mapping; blocked/missing evidence raises an explicit contract failure |
| extract_with_evidence(adapter, request) | Validate the request; obtain FeatureBatch and ordered token/processing evidence from the same existing adapter forward; preserve numerical semantics and validate rows/views/sites; only the coordinator bridge accesses the internal adapter primitive, with initial public-path equality checks |
| publish_feature_chunk(root, plan, chunk_id, condition, attempt_id, payload_paths, metadata) | Validate exact planned rows/sites/dtypes/audits and full identities, then use storage.publish for a unique immutable completed chunk; return its completion-manifest Path, not an unchecked folder |
| publish_release_index(root, plan, condition, split, chunk_manifests, run_id) | Independently verify exact full required coverage for one condition/partition and publish a passed immutable index; return its completion-manifest Path |
| load_feature_release(index_manifest, root, protocol, role, expected) | Verify completion, schemas, hashes, frozen assignments and full expected model/data/plan/condition/site identities; return a FeatureRelease reader with ordered rows and site catalog |
| FeatureRelease.read_matrix(site_key, language=None) | Read the release's already-fixed split for one site and requested language, retaining canonical row identities; return rows plus an N-by-D matrix. image_only rejects a language filter. No implicit concatenation of partitions or normalization |
| publish_text_family(root, recipe, split_payloads, run_id, metadata) | Publish an immutable fitted FIT-preprocessor/family identity, verified supplied split releases and their expected coverage; return its completion-manifest Path. Later split releases extend coverage through a new immutable index, not mutation; family completion requires all four eligible partitions |
| load_text_family(family_manifest, root, protocol, role, expected) | Verify recipe/schema/FIT-fit identity, hashes and row/partition coverage; return a TextFamily reader with a read_matrix(split, language=None) operation returning rows plus sparse/dense values under the fixed recipe |

Readers enforce role/split restrictions: source_fit permits FIT, source_select permits DEV-S, calibration permits DEV-C, exploratory_score permits SCREEN. Each release is fixed to its split and cannot broaden that role through a language/site filter. Labels and source/target language permissions still follow the specific scientific task; a source_fit reader does not authorize target-context adaptation. GPU/text transformation producers use the extraction-purpose guard and immutable job inventory rather than a downstream fitting role. CONFIRM is absent from every preliminary role. Writers and readers must reject schema/identity/coverage mismatches instead of accepting coercions.

Exact serialized field validators, error messages and implementation object types are finalized in the coordinator's executable-interface commit before workers code against them. These reserved names/semantics must not change independently across branches. Workers report a missing capability to the coordinator rather than duplicating a loader or silently expanding the interface.

Worker configuration must expose PHASE1_ROOT, dataset-manifest path, extraction-plan/readiness paths, MODEL and its generated config where applicable, optional resume-index references and unique producer/run IDs. All authoritative artifact references are explicit. No hidden global model/Drive account, implicit newest directory, hard-coded historical run ID, score-selected site or CONFIRM include switch is allowed. Front-end notebooks show validated settings at the top and fail early on placeholder/missing paths; authentication never prints secrets.

User-facing worker settings are PHASE1_ROOT, the printed shared/model plan manifest path, MODEL when applicable, and an optional explicit resume index. Other paths are resolved from that plan, not copied repeatedly across cells. The coordinator preparation notebook binds supplied evidence/config paths once and verifies typed artifacts. The shared data/E03 plan can pass independently; each model gets a separate readiness/model binding and the suite gets a final integration index. Do not accidentally make E03 wait for both GPU receipts or make one ready model wait for another. Requirements for per-runtime secrets, resource estimates and same-forward token evidence are in stage2_contracts.md, review revision 2.

## Execution dependencies and parallel work

1. Freeze shared interfaces and agreed file ownership; workers implement on separate task branches from that exact commit. Run CPU contract/serialization/role/restart checks before a bulk job is accepted.
2. E03 CPU preprocessing may run after common data acceptance, alongside a ready model's GPU extraction. It runs once, not once per model. Image/partition/target references remain identical across both workstreams.
3. A model's GPU launch requires its real passed Stage 1 validation and final smoke evidence. Models can overlap only with distinct available L4s; on a single L4 run sequentially. First production FIT work includes repeat/readback checks before continuing.
4. Publish complete compatible FIT/DEV-S releases to enable dependent Stage 3 work while remaining extraction continues. A release is a verified index, not the existence of a shard directory. Do not start Stage 3 scientific interpretation without its later full G0 controls.
5. Coordinator reviews/integrates worker branches, checks E03 and both models' full coverage/identity/Drive readback and missing/error logs, and updates verified acceptance. No worker marks whole-stage completion or changes main.

Every worker handoff includes base/head commit, owned/changed files, exact dependency versions, deterministic settings, tests and actual runtime commands/notebook versions, artifact completion paths/hashes, conditions/sites/dimensions, missing prerequisites and an honest pass/blocked/failed status. Fixture passes, metadata resolution and GPU inference are separate evidence. No new human semantic annotation/verification is an acceptance step.

## Implementation and acceptance checklist

- [ ] Coordinator shared-interface code/configuration committed and immutable worker base published.
- [ ] Stage 1 readiness receipts verified for each model; data/secret/mounted-root settings valid.
- [ ] VLM extraction and E03 implementations/notebooks written and relevant checks passed.
- [ ] Unique producer namespaces, incomplete/corrupt artifact rejection and resume demonstrated.
- [ ] Every eligible row/site covered with no silent cohort change and no CONFIRM processing.
- [ ] Model-specific first FIT/repeat and persistent mounted-path readbacks passed.
- [ ] Both model feature inventories, E03 family inventories and role-specific release indexes accepted.
- [ ] Integration report records actual evidence and explicit optional-site limits; plan/AGENTS updated without completing later stages.

All items are currently uncompleted. No executable interfaces, Stage 2 outputs or measured scientific results are implied by finalized planning.
