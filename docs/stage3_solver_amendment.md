# Stage 3 computational amendment — 17 November 2026

The pooled character E03 seed0 C=10 building.n.01 head exhausted the original
1000/5000 cold-start retry ladder. The user-supplied diagnostic at
results/stage3-solver-diagnostics/20261117T010451Z-ce2afa1e3d084eabb8ef22a02e9d9396
converged at 5511 with max_iter=20000 (67.93 seconds). It is diagnostic_only=true,
accepted_fit=false. No scientific score was used to choose a budget. The incidental
OptimizeWarning about iprint is not a ConvergenceWarning and is not this blocker.

The sole engine change adds a third cold-start 20000 attempt after the unchanged
1000/5000 attempts fail. C, regularization, loss, tolerance, weights, scaler,
targets, class order, zero initialization, dtype and numerical dependencies stay
fixed for every ordinary/control head. A remaining failure still blocks its
complete 25-head C block. The diagnostic does not supply accepted coefficients;
the production worker must execute and publish its own missing complete block.

Migration is explicit: operations amend-budget authenticates the original plan,
accepted suite and passed diagnostic in the same actual Python/package/threadpool
runtime, then publishes a new immutable plan/identity. Its compatibility receipt
binds the exact original plan reference and identity. Source proof compares the
original engine at 91c5131c32e7a675c5ef2fbaf1353da2c693eff3 against the current
engine and permits exactly the retry-tuple change. All other plan fields match.
Original successful units necessarily stopped during the unchanged first two
attempts, so their coefficients and downstream selections/control scores/null
summaries remain valid. Original bytes and producer provenance are retained;
the new registry may reference both compatible original and new artifacts.
Old candidates retain their convergence, target, row, metric and checksum checks.
Statistics reuse retains its original FIT-input identity checks. Foreign plans,
changed recipes/runtime/targets and edited payloads are rejected.

Run scripts/stage3/amend_solver_budget.ipynb once in the pinned CPU runtime.
It publishes only a plan and small handoff receipt and initializes new output
parent folders before parallel writers. No feature extraction, ordinary fitting,
selection or G0 scoring runs. It preserves the old handoff. Flush/unmount after
this one initializer; remount other runtimes to see the shared folders. Retain all
old ordinary/full-control index references for later coverage. The supplied E03
failure index retains its three completed C blocks; the unfinished C=10 block
must rerun its internal heads. Do not reuse the diagnostic as a fitted candidate.

Resume the E03 representative using the new plan, original assignments/statistics,
original E03 profile indexes and its explicit failure index. PROFILE=False,
SEEDS=[0,1,2], and the single pooled-character representative group remain fixed.
Keep completed Gemma/Qwen references; do not rerun their completed groups. Before
bulk G0 scheduling, integrate the complete ordinary statistics indexes and inspect
the first complete representative costs. Final global G0 reduction and Stage 3
validation remain necessary; no Stage 4 scoring or scientific claim is enabled.
