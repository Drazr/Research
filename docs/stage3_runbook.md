# Integrated Stage 3 execution

All four notebooks run on CPU with the pinned numerical dependencies. No L4,
model reload or HF token is required. Use integrated phase1, including the
runtime-order correction, rather than either original worker branch. Notebook
setup resolves a fixed clean commit; never pull midway through a run.

1. Run prepare_stage3.ipynb once. Retain its plan marker, shared assignment
   reference and profiling group IDs. This first mounted check authenticates the
   already passed Stage 2 suite; do not repeat extraction or Stage 2 validation.
2. Define one explicit ownership JSON covering all 145 groups exactly once.
   The standard three partitions are qwen-cpu-1 (66 Qwen VLM groups),
   gemma-cpu-1 (66 Gemma VLM groups), shared-e03-cpu-1 (13 E03 groups).
   Use the exact plan group IDs, not hand-invented IDs. Ownership is a small
   execution configuration, not a scientific cache or a distributed lock.
3. Run fit_readouts.ipynb with MODE='profile' for representative-owning
   partitions. Use FEATURE_SOURCE='vlm'/MODEL='qwen' or 'gemma', then
   FEATURE_SOURCE='e03'/MODEL=None once. Each owner produces its selected true
   C=.01/C=10 blocks and ordinary-owned statistics. Keep its index references.
4. Run run_g0.ipynb with PROFILE=True, SEEDS=[0] and only the matching plan
   representative group IDs. Supply preparation assignments and the ordinary
   statistics/index references. It produces seed0 C=.01 blocks. Retain these
   indexes and review actual timing/memory across all six representatives.
5. Switch fit_readouts to MODE='ordinary', supplying the combined ordinary/G0
   PROFILE_INDEXES and owned RESUME_INDEXES. Switch run_g0 to PROFILE=False,
   SEEDS=[0,1,2] and the explicitly owned groups, supplying ordinary statistics
   and G0 RESUME_INDEXES. Reuse the profiling work; do not start from scratch.
   Ordinary and G0 jobs can overlap on separate CPU runtimes once each group's
   statistics exist. Measure the first complete G0 group/null calculation before
   bulk control scheduling. A missing statistic defers its group (exit 3), not
   permission for G0 to create a second scaler.
6. After group jobs finish, run run_g0 in REDUCE_ONLY mode with empty GROUP_IDS
   and SEEDS, PROFILE=False and all explicit G0 indexes. This combines complete
   coverage into one global gate; do not average worker p-values. A flagged
   result (exit 4) blocks interpretation and final acceptance.
7. Run validate_stage3.ipynb once with the prepared plan and integrated explicit
   inventory JSON: all 2,320 candidate refs, 580 selection refs and one g0_gate
   ref. Deduplicate identical path/hash references from overlapping recovery
   indexes; conflicting coverage is an error. Final acceptance performs its own
   integrity/coverage and fresh-process readback checks.

Keep completed unit/group indexes for reuse across notebooks. Profile indexes
are reusable production units, not another dataset or extra experiments. E03
is not rerun for each model. Ordinary scoring, E19/E25, interpretation and
Stage 4 are not enabled until the accepted Stage 3 registry exists. CONFIRM
remains closed. Local fixture passes do not establish mounted production success.

For the original 6749bd9f53148def run's E03 iteration exhaustion, use the one-time
CPU amend_solver_budget.ipynb and the explicit recovery instructions in
docs/stage3_solver_amendment.md. Do not rerun preparation/ordinary fitting or edit
engine.py inside a running checkout. Completed original units remain reusable
only through the new plan's authenticated compatibility receipt.
