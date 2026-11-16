# Ordinary Stage 3 worker

Implemented from foundation `3ba1acdc9cc429cf8086ee87ca9f4058bbc83fb7` on
`phase1-stage3-fits`. The four owned files are `src/stage3/fit_readouts.py`,
`scripts/stage3/fit_readouts.ipynb`, `tests/test_stage3_fits.py` and this document.
No shared numerical interface, scientific setting, protocol, dependency or
Stage 1/2 file is changed. This is an implementation handoff, not Stage 3
acceptance or production execution.

## Inventory and numerical boundary

The worker consumes the preparation-owned typed FitPlan through
`src.stage3.interfaces`. Its accepted inventory contains 66 systems per VLM
and 13 shared E03 systems: 145 ordinary selections, 580 complete four-C
candidate blocks and nominally 14,500 binary fits before retries. All fixed
depths/poolings and image-only systems remain in the inventory. E03 runs once
across both model comparisons. The CLI requires `--feature-source vlm/e03`;
`--model qwen/gemma` applies only to VLM.

The coordinator engine alone implements source-FIT statistics, dense centering,
sparse scaling, variance guards, known masks, float64 solves, image weights
(including pooled .25), fixed LBFGS settings/retry, all 25 category statuses,
fixed-primary-cohort DEV-S metrics and smallest-C ties. The worker schedules the
four frozen Cs and references the selected original FIT-only block without a
refit. It does not implement another solver/scaler or introduce tuning settings.
It opens only `source_fit` and `source_select` numeric sessions. No raw images,
VLM execution, ordinary SCREEN/DEV-C/CONFIRM reads, scientific interpretation or
Stage 3 validation are part of this worker.

## Explicit ownership and references

Preparation supplies the accepted plan and shared target assignments. Before
launch, agree on one JSON ownership input with this shape:

```json
{"owners": {"qwen-cpu-1": ["exact group IDs from the plan"], "gemma-cpu-1": ["..."], "shared-e03-cpu-1": ["..."]}}
```

The example is schematic; actual lists must cover all 145 IDs exactly once.
Each owner has one feature source/model. Split larger model/E03 partitions into
additional explicit owners for separate runtimes. Unknown, overlapping, missing,
empty or selector-incompatible partitions fail before fitting. The ownership
file is an agreed input, never an appendable/shared worker checkpoint. This
validation detects declared overlap; it is not a distributed lock. Do not launch
the same owner twice or change ownership during active jobs.

For each owned group, ordinary alone publishes its label-free source statistics.
`statistics_ready` streams the exact immutable `{path, sha256}` reference
immediately, so G0 need not wait for other groups or ordinary C selection. Each
group/failure recovery index retains that same reference. G0 must consume it;
it must not publish a second scaler. G0 owns randomized candidates, including
seed0 C=.01 profiling blocks; this worker fits only `assignment_id=true`.

Completed references are relative to explicit PHASE1_ROOT. No latest-file/tree
discovery, implicit local root, mutable checkpoint, overwrite, pickle or extra
persistent input/standardized-feature cache is used. Two role-specific bounded
numeric sessions retain current-group raw/transformed matrices and bounded
512-MiB byte LRUs, reusing chunks across owned sites/languages while resident.
Only one group and one numerical fit process run at a time. Memory includes both
role caches and active numeric working arrays; production RSS must be measured.

## Production sequence

1. Run coordinator preparation once against the exact accepted Stage 2 suite.
   Use the resulting plan marker and agreed complete ownership file. No Stage 2
   extraction/validation rerun is requested. Keep assignments under preparation
   ownership.
2. On each representative-owning ordinary partition, run `MODE='profile'`. This
   produces its metadata-selected representatives' true C=.01/C=10 blocks,
   source statistics, timings and per-group indexes. It skips other owned groups.
   Share statistics with G0, which runs the prescribed seed0 C=.01 units and
   supplies their candidate recovery indexes. All six representatives cover
   Qwen, Gemma, sparse word, sparse character, pooled character and dense length;
   neither worker chooses representatives using scores.
3. Retain/review the streamed actual read/hash/write/readback/fit/RSS measurements
   before bulk launch. `MODE='ordinary'` requires `PROFILE_INDEXES` covering true
   .01/10 and seed0 .01 for every representative, with the exact same statistics
   reference per group. A missing/changed/conflicting block prevents bulk work.
   Fixture measurements cannot substitute for these mounted production units.
4. Pass owned previous `RESUME_INDEXES` and the combined `PROFILE_INDEXES`.
   Ordinary automatically reuses its owned true profiling blocks/statistics
   from the latter, even when they are not repeated in RESUME_INDEXES. G0 units
   are evidence for the profile gate, never scheduled again by ordinary.
5. Run each disjoint owner on CPU. Capture `group_complete.index` references and
   return all ordinary group indexes to coordinator integration. G0 independently
   resumes its assigned controls from its explicit references. This worker never
   merges into phase1 or publishes the accepted registry.

The notebook resolves the worker branch once to an immutable commit, checks a
clean checkout, installs unchanged `requirements-stage2-cpu.txt`, mounts explicit
PHASE1_ROOT and streams child output with `-u`. Use Python 3.11/3.12 and one
BLAS/OpenMP thread set before numerical imports. The producer records the actual
commit/runtime; persistent-drive production refuses a dirty checkout.

Equivalent CLI, with actual reference JSON supplied as single arguments:

```sh
python -u -m src.stage3.fit_readouts --phase1-root /content/drive/MyDrive/phase1 \
  --manifest /content/drive/MyDrive/phase1/EXACT_PLAN/_COMPLETE.json \
  --ownership /content/drive/MyDrive/phase1/EXACT_OWNERSHIP.json \
  --owner qwen-cpu-1 --feature-source vlm --model qwen --mode profile
```

For bulk, change `--mode ordinary` and repeat `--profile-index '{"path":...,"sha256":...}'`
and `--resume-index '{"path":...,"sha256":...}'` as needed. E03 omits `--model`.
The notebook passes arguments as a list, avoiding shell quoting pitfalls.

## Resume and progress

Recovery authenticates small index/child metadata, validates used owned units
through coordinator semantic loaders and rejects wrong groups/assignments,
duplicate coverage, corrupt payloads, differing scalers or a selection lacking
the exact four original C references. Repeated identical immutable references
across snapshots are deduplicated. Shared profile indexes contribute only owned
ordinary units to recovery; unrelated/control units are not fitted by ordinary.
No completed candidate is refitted. A fully completed group resumes without
opening source numeric sessions. An incomplete group reads its permitted inputs
once per session and computes only missing Cs/selection.

Each new C block is published with all 25 outcomes. A selection is one small
reference index, without a selected weight copy. A small per-group run index is
published at a group boundary or handled failure, never per head/iteration or as
a repeatedly growing whole-job file. Newly completed references stream before
subsequent readback/reporting, and handled exceptions/interrupts preserve useful
completed work in a failure index. If index publication itself fails, the event
prints the completed references for explicit reconstruction. Abrupt process loss
can lose the current unfinished C and references not captured by the caller;
there is no zero-loss guarantee.

Progress includes start/end events, 15-second heartbeats during long public
calls, exact group/C/assignment, shape/nonzeros, references, all head statuses,
retry counts, engine-recorded fit seconds and process high-water RSS on Linux.
Scoped cProfile records exclusive file-read, hash and file-write seconds; these
are components inside the enclosing elapsed phase, not additive phase estimates.
Candidate publication includes coordinator write/readback; a separate semantic
candidate-readback phase verifies every new block. First numeric-read/verify and
selection-numeric-read/verify phases expose input overhead separately. Cache-hit
and reuse events report their actual path; no synthetic Drive/runtime projection
is emitted. RSS is null on unsupported platforms rather than fabricated.

## Verification and execution evidence

Local fixtures use Python 3.12, numpy 2.1.3, scipy 1.18.1, scikit-learn 1.6.1 and
one BLAS/OpenMP thread. The ordinary tests cover exact disjoint inventory and
selectors; mocked full 145-group/four-C scheduling; actual tiny dense, sparse
word, length and pooled fits through coordinator interfaces; all 25 statuses,
ties/selection; statistics handoff; interruption checkpoints; missing-only resume;
profile reuse/gating/scaler agreement; no input reads or refit on complete resume;
conflicting/corrupt/wrong-owner/incomplete-selection rejection; timing events and
empty-output notebook code-cell compilation. Full 145-group scheduling is mocked
software coverage, not 145 actual source systems fitted on research data.

Validation command: `python -m unittest discover -s tests -v` with all four
BLAS/OpenMP environment variables set to 1. Full discovery ran 99 tests: 96
passed and three pre-existing Stage 1 cache tests encountered Windows WinError
206 at this long checkout path. All three passed unchanged on retry with TMP/TEMP
pointing to a shorter workspace directory. No path workaround changes repository
source or production settings. Python/module and notebook code-cell compilation,
empty notebook outputs and Git whitespace checks also passed.
After the final selection-score consistency review, all seven ordinary-worker
tests passed again in 181.014 seconds, including rejection of a hash-valid
selection with changed nonselected scores. These elapsed times describe local
software fixtures only.

Production artifacts created here: **none**. Fixture bundles live only in test
temporary directories and are removed after the checks. No actual Drive mount,
accepted production FitPlan, ownership file, representative measurement, ordinary
selection or G0 outcome was available to this local task. Those are launch/runtime
prerequisites, not reasons to invent or rerun upstream scientific inputs. Complete
ordinary inventory, G0 controls/gate, coordinator integration, independent mounted
readback and final accepted Stage 3 registry remain pending.
