# Stage 3 executable foundation

Implements stage3-plan.v1 review revision 2. Import worker callables from
`src.stage3.interfaces`. The immutable worker base is the commit first adding this
document and the Stage 3 modules to phase1; recover it with:

```sh
git log --diff-filter=A --format=%H -- docs/stage3_interfaces.md
```

Both workers branch from that exact implementation commit. Earlier planning
commit 0ed0f79 is not an executable base. Do not edit frozen common files or
scientific settings independently; report necessary changes to the coordinator.

## Ownership

| Owner | Files |
|---|---|
| Coordinator | src/stage3/{artifacts,data,engine,readouts,controls,recovery,validation,operations,interfaces,__init__}.py; scripts/stage3/{prepare_stage3,validate_stage3}.ipynb; tests/test_stage3_foundation.py; this interface document and shared integration reports/plan updates |
| Ordinary worker | src/stage3/fit_readouts.py; scripts/stage3/fit_readouts.ipynb; tests/test_stage3_fits.py; docs/stage3_fits.md |
| G0 worker | src/stage3/run_g0.py; scripts/stage3/run_g0.ipynb; tests/test_stage3_g0.py; docs/stage3_g0.md |

controls.py supplies shared scoring/randomization/reduction primitives; it is not
the worker's scheduling, distributed ownership, checkpoint discovery or resume
loop. Workers must not replace it with a second numerical implementation.

## Environment and references

Use requirements-stage2-cpu.txt unchanged, Python 3.11/3.12 and one BLAS/OpenMP
thread set before numerical imports. runtime() verifies the three numerical
pins and actual thread pools. current_producer(scope='persistent-drive') refuses
a dirty checkout and records actual commit/runtime. No Torch, L4 or HF token is
needed. PHASE1_ROOT is /content/drive/MyDrive/phase1 in the user's Colab runtime.

Coordinator runtime-order correction: threadpool records are sorted by canonical
record bytes before creating the runtime identity. Discovery/import order alone
must not cause fresh-process rejection; library/version/thread changes still do.
Use the coordinator correction for production preparation and worker execution;
do not patch shared files privately. Old fixture plans may be recreated, never
rewritten. No production artifact rerun is established as necessary by this fix.

Artifact references are immutable mappings `{path, sha256}` relative to that root.
Use `plain(value)` from src.stage2.schema for JSON conversion. A FitPlan is an
immutable Mapping with .root, .protocol and .reference. Keep it and bounded
sessions alive inside a worker process. No latest-file/tree discovery, implicit
root, mutable shared checkpoint, pickle or overwrite is supported.

The accepted Stage 2 suite is explicitly:
results/stage2-validation/20261116T010841Z-17d614fe5da340b4b6c036a431fc6be8/_COMPLETE.json.
The verdict is user-reported; actual mounted consumption verifies it. Preparation
authenticates metadata/targets/splits/record bytes, not images/archives. Sessions
authenticate only their consumed numeric bytes and exact semantic row/site joins.
Stage 2 readers/producers/protocol remain unchanged. A cached byte snapshot is
immutable and hash-bound; a new process checks disk again. No pathname-only trust.

## Public callables

| Callable | Return and required semantics |
|---|---|
| compile_fit_plan(suite_manifest, root, protocol, producer) | FitPlan; writes one complete plan, validates exact 24 releases/13 families/145 groups, binds runtime/numerical recipes, profiling group IDs and eligible targets; no fitting or raw-image access |
| load_fit_plan(manifest, root, protocol) | FitPlan; authenticates the saved plan, accepted metadata graph and numerical/runtime identity, without rerunning Stage 2 validation |
| open_numeric_session(plan, root, groups, role) | NumericSession; groups is an explicit iterable of group IDs, role is source_fit, source_select or g0_score |
| session.read(group_id, role=None) | (immutable ordered rows, read-only float32 ndarray/CSR); the role, if supplied, must equal session.role; one current group matrix is retained, payload bytes use a bounded 512 MiB LRU; shared chunks can serve sites/languages without repeated reads while resident |
| publish_target_assignments(plan, root, producer) | One assignments reference for true/seed0/seed1/seed2 in FIT/DEV-S/SCREEN; preparation owns this single publication |
| load_or_publish_statistics(plan, group_id, session, producer, supplied_refs=()) | dict with mean/variance/scale read-only float64 vectors, center bool and reference; FIT session only; exact compatible supplied statistics are reused without computing new statistics |
| fit_candidate_block(plan, group_id, assignment_id, C, session, statistics, producer) | One completed candidate reference; session is a dict with source_fit and source_select sessions; assignment_id is true, seed0, seed1 or seed2; C is one of CS=(.01,.1,1.,10.) |
| load_candidate_block(plan, ref, store=None) | (body, coefficients, intercepts, estimable); full candidate payload/status/target/DEV-S metric and source-statistics checks; no refit |
| select_system(plan, group_id, assignment_id, candidate_refs, producer) | One selection reference; exactly four distinct C blocks, same source statistics, defined fixed-cohort metrics, smallest-C tie rule; no copied selected weights or refit |
| load_fitted_system(selection_manifest, root, plan, role, expected) | FittedSystem; expected must contain group_id and assignment_id; .score(session) returns (rows, logits, estimable); true systems cannot use g0_score |
| score_g0(plan, selected_control_refs, session, producer) | List of control_scores references; g0_score session only; selected randomized systems and within-source-context SCREEN scores, full metrics/status arrays |
| evaluate_g0(plan, control_score_refs, assignment_refs, resume_refs, producer) | dict complete/groups/gate; one assignment reference, complete three-seed score triples per requested group, optional completed null-group refs; one all-1999-draw summary per group, 100-draw RAM batches; gate remains None until all 145 groups are present |
| read_completed(plan, ref, kind, expected) | Immutable body; verifies completion/payload bytes and expected identity fields before reuse; candidate/selection/statistics kinds additionally use their semantic loaders |
| publish_run_index(plan, entries, producer) | Small immutable recovery-index reference; entries maps supported categories to explicit completed references; use at group/partition boundaries or handled failure, never per head/iteration |
| load_run_index(plan, ref) | Immutable entries mapping; verifies index/child marker identities; call the semantic loader/read_completed for actual units being resumed, rather than rereading every unused coefficient block |
| validate_stage3(plan, candidate_registry, selection_registry, g0_receipt, producer) | Accepted system_registry reference only after exact 2320 candidates/580 selections, selected-system/G0 agreement, complete unflagged global gate and fresh-subprocess payload readback; missing/invalid inputs raise ContractError |

Supported run-index entry names: candidates, selections, statistics, assignments,
control_scores, null_groups, gates. Their values are lists of references. An
index is recovery metadata, not whole-stage acceptance. The final validation
inventory JSON has candidates (all references), selections (all references),
g0_gate (one explicit gate reference). Conflicting/duplicate coverage blocks.

## Payload and ownership rules

Each completed Stage 3 bundle has body.json and a small fixed set of .npy files,
with the original stage1.v1 completion-marker wrapper. Bodies bind plan and
fit_identity, plus group/assignment/C where applicable. Source statistics use
input_identity independent of C/control seed and bind source FIT rows/weights,
feature/site/context, statistics implementation and numerical runtime. Producer
commits remain provenance; unrelated documentation/notebook changes do not enter
the numerical identity. Changed numerical engine/target/weight semantics do.

Candidate arrays: coefficients (25,D), intercepts (25,), estimable (25,) and
dev_logits (N_DEV-S,25). Float arrays are little-endian float64, estimable is bool.
body.json retains exact DEV-S rows, source FIT row hash, statistics reference,
per-category support/convergence/objective/gradient records and metrics. Invalid
heads have explicit status and finite zero padding; consumers must use estimable.
No FIT logits or optimizer iterations are saved. Missing-class secondary metrics
remain null with a reason. Primary undefined metrics prevent selection.

Selection bodies contain all four candidates/scores, selected_C and selected
reference. No extra selected coefficient copy exists. G0 score bundles store
logits and estimable plus rows/selection/assignment/metrics. Null-group bundles
hold 1999 draw-wise maxima and the observed statistic, exact score refs and fixed
family identity across all three seeds. Missing groups/seeds cannot be silently
averaged away. No per-draw prediction tensors or routine 100-draw Drive files.

One ordinary worker owns a group's statistics and supplies its immutable reference
to G0. G0 waits only for that group, never creates another scaler. Both workers
may own disjoint group lists on separate CPU runtimes; E03 is model-independent
and runs once. Use FEATURE_SOURCE=vlm/e03 and MODEL=qwen/gemma only for VLM groups.
No concurrent producer may append to another worker's index or repeat its block.

## Execution boundary and reporting

Run prepare_stage3.ipynb once on CPU after publication, then implement the two
workers against this base. The coordinator notebook streams child output and
prints plan/assignment refs and profiling group IDs. Worker notebooks must do
the same and report read/hash/compute/write timing. Run the scheduled metadata-
selected representative blocks before bulk scheduling and reuse them afterward.
Do not infer production timing from synthetic fixtures.

This foundation does not run actual fits, worker scheduling or G0 production.
validate_stage3.ipynb waits for the integrated explicit worker inventory. Stage 4
ordinary DEV-C/SCREEN readers will require the accepted Stage 3 registry; those
permissions are not exposed by Stage 3's role enum. No CONFIRM access or scientific
interpretation is enabled. Bring worker heads back for coordinator integration.
