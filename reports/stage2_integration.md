# Stage 2 software integration — 15 November 2026

Status: shared foundation and both workers reviewed and integrated; production Stage 2 completion remains pending. No scientific results or actual local L4 inference are claimed.

## Reviewed sources and preserved ownership

- Shared foundation: 986f812ac6e3811a035391738e295bf55043369a, 20 files, including public interfaces, schemas, plans, readiness, bridge, artifact/release readers/writers, coordinator notebooks, requirements and tests.
- E03: 388579ff6b2020d5a6db6c23f162d956d536923a, based exactly on that foundation, five owned files. Reviewed analyzers, FIT-only statistics/cap/ties/IDF, sparse arithmetic/serialization, source identity checks, checkpoints and independent-process transform reconstruction.
- Extraction: ab446367df7caea902a5165ff2f90de01b3d154f, based exactly on that foundation, four owned files. Reviewed input binding/CONFIRM exclusion, same-forward bridge/public comparisons, three conditions and site coverage, visual/repeat audits, first-chunk measurements, runtime identity, immutable progress/resume and complete partition releases.

Both branches merged cleanly without overlapping worker edits. Their commits are retained as integration parents. Stage 1 code/adapters, scientific source documents, model/protocol configurations and dependency pins are unchanged. The shared public interfaces remain those frozen at 986f812; no new feature arithmetic or scientific design was introduced during integration.

## Integration corrections

Three notebooks required an already mounted Drive, a prepared checkout and dependencies; they also relied on changing cwd without explicitly adding the repository to Python's import path. The extraction notebook defaulted to its development branch. All four Stage 2 notebooks now have an explicit first setup cell that mounts Drive in Colab, checks PHASE1_ROOT, clones phase1 when needed, rejects dirty/wrong-branch/stale checkouts without resetting them, adds the repository import path and installs the appropriate requirements profile. Root/repository/scope settings live in this cell rather than being reset in later cells. CPU notebooks install requirements-stage2-cpu.txt; extraction installs requirements-l4.txt. Notebook files use UTF-8/LF and contain no saved outputs or tokens.

This correction affects entry-point setup only. It does not bypass readiness, perform notebook authentication on the user's behalf, alter the frozen execution identity or claim that model memory fits every production input. Existing running Colab jobs should keep their frozen checkout; update between jobs.

The final suite-index composer also incorrectly placed resource-report references in its feature_releases list. They now appear in a separate resource_reports mapping keyed by model, while feature_releases contains only actual feature release references; both remain in upstream_artifacts. A new orchestration regression covers a complete synthetic two-model/24-release/13-family inventory, checks reference separation, and confirms missing release/resource inputs block acceptance. Payload correctness remains covered by the existing shared/worker tests. This correction changes no callable signature, input protocol, adapter arithmetic or feature payload.

## Independent verification performed here

The final integrated suite passed all **78 tests**: 40 Stage 1 regressions, 17 shared-foundation tests, 8 E03 tests, 12 extraction tests and one new suite-orchestration regression. Runtime: the foundation's Python 3.12 virtual environment with the pinned CPU profile. Windows fixtures used the explicit short TEMP/TMP root documented by the foundation. The first combined 77-test run passed in approximately 165 seconds; the final 78-test run after both integration corrections passed in approximately 160 seconds. The two existing notebook validation tests were additionally rerun after setup changes and passed; every Stage 2 module and code cell in all four notebooks compiled, with no saved outputs. All three module CLI help paths loaded. Whitespace checks passed after LF normalization.

A separate read-only check loaded and revalidated the actual available local dataset artifact and compiled its Stage 2 inventory: 14,400 original selected records, unchanged dataset fingerprint 54825d236efcd88cc5252918ebce6d1dad6109ef10a244281f262204bc6cc5bd, 92 image chunks, 11,520 image_caption rows, 11,520 caption_only rows and 2,880 image_only rows. CONFIRM was absent from every compiled processing row. This created no production plan, features or text-preprocessing artifacts and did not access a Colab mount.

Software fixtures cover real serialization/reader paths, sparse/dense reload, all 13 E03 families, resume without refitting, complete synthetic condition/split releases, numerical conflicts, corruption, first-chunk recovery, runtime mismatch, token/visual/repeat failures, secret redaction and immutable failure preservation. Fake adapters are not checkpoint validation. The source review also checked typed historical evidence/source compatibility and the actual backend assertion; it does not independently re-fetch the remote Drive objects reported by the foundation.

## Remaining runtime gates

Run prepare_stage2.ipynb against the intended mounted root to verify actual data/configuration/evidence and publish shared and per-model plans. Historical Stage 1 passes are reused only when the typed hashes/source/runtime identities match; do not rerun models merely because the consumer commit changed. The foundation's evidence report records its earlier Drive checksum verification; this integration does not replace live mounted receipt verification.

Next run the E03 worker and actual model extraction as described in docs/stage2_runbook.md. Actual first-FIT bridge/public equality, repeated vectors, measured resources, mounted-path readback, complete 12-release inventories per model and all 13 complete E03 families remain pending. Only the final validation notebook can accept the full supplied suite inventory after its checks. Full G0 controls, fitting and scientific scoring remain Stages 3/4; Stage 2 software integration is not whole-stage completion.

Automatic publication review initially rejected the full bundle because inherited documentation included a personal account email and user-specific Windows paths. The integration update removes the personal email/name and uses generic short-temp examples. No tokens or scientific/runtime settings were changed.
