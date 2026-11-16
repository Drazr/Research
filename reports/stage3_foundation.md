# Stage 3 coordinator foundation — 16 November 2026

Implemented the shared CPU numerical/data/storage foundation and two coordinator
notebooks. This is the executable base for worker implementation, not completed
Stage 3 production work. Read docs/stage3_interfaces.md for signatures, payloads,
ownership and recovery rules. Its first-add commit is the immutable worker base;
the earlier planning commit is not usable as an executable interface freeze.

## Implemented scope

- Exact passed Stage 2 suite binding; metadata/record/target/split provenance,
  24 feature releases, 13 E03 families, 145 source systems and representative IDs.
- Numeric-only read-only sessions with checksum-bound byte snapshots, bounded
  reuse, exact Stage 2 planned_rows identities, dtype/shape/finite/CSR validation,
  explicit target joins and guards on every role/cache access. No images,
  archives, base weights, caption encoding or recursive Stage 2 validation.
- Source-only weighted float64 dense/sparse statistics and in-memory transformed
  input reuse across categories, Cs and target assignments.
- One binary logistic engine with the frozen sum-loss/L2 convention, unpenalized
  intercept, fractional pooled-image weights, fixed convergence retry, all
  category statuses and source-DEV-S C selection without refitting.
- Immutable assignments/statistics/candidate/selection/control-score/null-group
  bundles and explicit recovery indexes. Completed 25-head C blocks are saved;
  optimizer iterations, FIT logits and standardized feature copies are not.
- Shared G0 scoring and vectorized tie/mask-aware AUROC, 100-draw RAM batches,
  one all-draw result per group, common three-seed assignments and full-family
  gate reduction. Incomplete/flagged coverage cannot pass final validation.
- Full candidate/selection/gate coverage checks and independent subprocess
  payload readback before publishing an accepted system registry.
- CPU prepare_stage3/validate_stage3 notebooks: clean fixed checkout, existing
  numerical pins, thread limits, explicit settings and streamed child output.

The ordinary-fit and G0 worker loops/notebooks were intentionally not implemented.
Shared controls.py is a numerical/scoring primitive library, not run_g0.py.
No protocol/model/dependency/Stage 1/2 source or Stage 2 specification change was
required. Read-only producer artifacts remain at their original identities.

## Verification and limits

The selected local Python 3.12 CPU environment uses numpy 2.1.3, scipy 1.18.1,
scikit-learn 1.6.1 and one BLAS/OpenMP thread. Foundation fixture tests cover:
objective finite-difference gradients, intercept and fractional replication;
dense/sparse coordinate equivalence and constant columns; convergence/retry and
unestimable cases; exact native row joins and reference Stage 2 reader equality;
all 13 text families; role guards, immutable arrays, corruption, dependency
changes and unique payload-read counts; source-scaler/transformed-array reuse;
actual tiny true/control C fits, selection/reload and explicit resume records;
shared image-vector/mask assignment, moving unknown masks, vectorized AUC versus
the reference metric, complete three-seed group null calculation/resume and
missing/duplicate coverage rejection; fresh-process readback/corruption handling.

Fixture suite construction is synthetic and deliberately small. Its accepted
markers are software fixtures, not independently verified research artifacts.
The Stage 2 reference-reader comparison isolates upstream acceptance with fixture
mocks while using real publication, hashes, row identities and payload readers.
No production runtime projection or scientific result is inferred from timings.

Final combined suite: **92 tests passed** in 209.603 seconds (78 existing Stage
1/2 regressions plus 14 Stage 3 tests). All Stage 3 modules and both notebook
code-cell sequences compile; notebook outputs/execution counts are empty.
Protected protocol/dependency/Stage 2 specification bytes are unchanged. The
entire diff consists of coordinator-owned implementation/tests/notebooks and
handoff/status documentation. No actual mounted artifacts or GPU were required.

The final 145-group accepted-registry path has not been exercised against a
production worker inventory: those workers and inputs are absent. Available
checks include complete tiny true/control C fits, a complete three-seed group
null result and its reuse, rejection of incomplete whole-stage inventory, and
fresh-process payload readback/corruption. Do not promote these into a completed
full G0 gate or whole-stage acceptance claim.

## Runtime evidence and remaining gates

The user's final Stage 2 passed pointer is
results/stage2-validation/20261116T010841Z-17d614fe5da340b4b6c036a431fc6be8/_COMPLETE.json
under /content/drive/MyDrive/phase1. It was not independently opened through a
mounted Drive filesystem in this foundation task. That suite and its numeric
inputs must be authenticated by the first actual prepare_stage3 runtime; no
expensive Stage 2 rerun is requested. No new production artifacts were created.

Both exact model choices remain those already accepted in Stage 2; Stage 3
derives their actual sites/dimensions/configurations from that suite. No GPU or
model access is needed for this stage. Production matrices are unavailable in
this local checkout, so actual representative FIT/C/control measurements, Drive
performance/readback and memory projections remain pending.

Next, both workers branch from the immutable implementation base. Ordinary-fit
publishes each group's single source-statistics reference; G0 consumes it and
the preparation-owned assignments, with disjoint source-group ownership.
Integrate both worker branches and review their tests before production launch.
Measure and reuse the scheduled representative units, then finish all 2,320 C
blocks, 580 selections and the full unflagged G0 gate. Only final accepted Stage 3
registry/readback permits Stage 4 ordinary scoring. Stage 4 permissions and
scientific E19/E25 analysis remain future work; CONFIRM stays closed.
