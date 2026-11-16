# Stage 3 worker integration review — 16 November 2026

Reviewed ordinary worker 60f36fb93cd7542e36e1638c34d17570a955e2a0 and G0 worker
732ea46855c3b334667995dd42455a701cdc8e9e, both based on the frozen foundation
3ba1acdc9cc429cf8086ee87ca9f4058bbc83fb7. Each changed only its four owned files.
Both are integrated with coordinator runtime-order correction a3ae741.

The workers share the same statistics/solver/selection/target interfaces and
preserve model-independent E03, exact group ownership, all categories/masks,
source-only fitting, C grid and fixed G0 controls. Profile units are reused;
ordinary bulk requires the full representative evidence, including the G0 seed0
units. G0 never creates missing ordinary statistics or preparation assignments.
Interrupted groups preserve completed block references, without per-head or
per-null-draw checkpoint files. Scientific settings and Stage 1/2 are unchanged.

Integration changes: both worker notebooks default to integrated phase1 rather
than their original worker branches, so preparation and workers use the same
canonical runtime correction. Completed G0 score recovery now authenticates
its exact rows through the coordinator's role-scoped expected_group_rows
metadata helper. Saved score/candidate/statistic payloads and selection/target
identities are still verified; unused SCREEN feature matrices are not reopened
merely to reconstruct row IDs. A strengthened resume regression forbids opening
numeric sessions when the complete score/null dependency closure already exists.

All four notebook code-cell sequences compile with empty saved outputs. Local
tests use the existing pinned Python 3.12 numerical environment and one thread,
with a short workspace temporary root to avoid Windows path-length failures.
Combined regression discovery passed all **109 tests** in 389.481 seconds,
without the workers' previously reported Windows path failures. After the
metadata-only recovery fix, all **9 G0 tests** passed again in 63.353 seconds,
including complete resume with numeric-session construction forbidden. Notebook
syntax/empty-output and whitespace checks passed. Software fixtures are not
production artifact acceptance.

Actual mounted preparation, complete ownership configuration, representative
measurements, all ordinary/control fits, full G0 coverage/gate and accepted Stage
3 registry remain pending. No production artifacts, GPU execution, new human
work or scientific findings are claimed. See docs/stage3_runbook.md for exact
notebook modes/order and required handoffs; all Stage 3 work is CPU-only.
