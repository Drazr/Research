# Stage 3 G0 worker

Implementation base: `3ba1acdc9cc429cf8086ee87ca9f4058bbc83fb7`.
Owned files are `src/stage3/run_g0.py`, `scripts/stage3/run_g0.ipynb`,
`tests/test_stage3_g0.py` and this note. Shared modules, numerical identities,
protocol, Stage 2 artifacts and dependencies remain unchanged. This worker is
software implementation, not Stage 3 production acceptance.

## Execution and ownership

Run the notebook on Python 3.11/3.12 CPU with the unchanged
`requirements-stage2-cpu.txt`: numpy 2.1.3, scipy 1.18.1, scikit-learn 1.6.1,
Pillow 11.3.0. Set OPENBLAS/OMP/MKL/NUMEXPR threads to one before imports.
The notebook resolves its worker code ref once, detaches at that commit and
refuses dirty/changed checkouts. No GPU, Torch or model token is needed.

Use the preparation-owned plan and assignment reference. Ordinary workers own
source statistics; pass explicit statistics refs or their run indexes. Only the
statistics entries of ordinary indexes are consumed. G0 authenticates each
supplied statistic's group, input/row/weight/runtime identity and numeric arrays
before invoking the shared reuse API with that exact reference. No call is made
when a group's reference is missing, so the fallback statistics producer cannot
run. Ready groups continue independently; missing groups are listed as waiting.
G0 never calls the assignment publisher.

Declare disjoint group/seed ownership. FEATURE_SOURCE=vlm uses one MODEL selector
qwen/gemma; FEATURE_SOURCE=e03 has MODEL=None and is run once across both models.
The notebook checks selectors against the authenticated group metadata. The CLI
uses exact group IDs from the same plan. There is one process and one numerical
thread; a progress heartbeat thread performs no numerical work or artifact I/O.

The shared fitter receives every group/seed's four Cs (.01,.1,1,10), the same
source statistic and FIT/DEV-S sessions. It retains all 25 target statuses and
the fixed 14-category selection cohort. Whole-image assignments, unknown masks,
four-language pooled weights, sparse/dense transforms, convergence retry,
selection/tie rules and no-refit behavior are entirely shared with ordinary
fitting. True candidates/selections are rejected as G0 recovery inputs.

Profile mode runs only metadata-selected representatives' seed0 C=.01 blocks.
It publishes no selection or SCREEN scores. Ordinary workers provide the
corresponding true C=.01/C=10 measurements. Review these actual production
measurements before full scheduling and retain their references. Full mode
always schedules the full four-C grid and reuses representative blocks.

## References, recovery and null reduction

CLI example (all paths/references must be supplied explicitly):

```sh
python -u -m src.stage3.run_g0 --phase1-root "$PHASE1_ROOT" \
  --manifest /path/to/preparation/_COMPLETE.json \
  --settings /path/to/g0-settings.json --storage-scope persistent-drive
```

Settings JSON fields:

```json
{
  "groups": ["EXACT_GROUP_ID_FROM_PLAN"],
  "seeds": ["seed0", "seed1", "seed2"],
  "assignments": {"path": "PREPARATION/_COMPLETE.json", "sha256": "FULL_SHA256"},
  "supplied": {"statistics": [{"path": "ORDINARY/_COMPLETE.json", "sha256": "FULL_SHA256"}]},
  "statistics_indexes": [],
  "resume_indexes": [],
  "profile": false
}
```

References are root-relative immutable path/hash mappings, not mutable latest
files. Resume accepts G0 indexes or explicit supplied candidates, selections,
control_scores, null_groups and gates. Identical references repeated across
indexes are deduplicated. Distinct references claiming the same group/seed/C,
selection, score, null result or assignment block execution as conflicting
coverage. There is no silent last-writer choice.

Selections close over all four candidates; score refs close over their exact
selections; null refs close over all three seed scores. Completed candidate
loaders authenticate solver/row/target/scaler/DEV-S evidence. Resumed scores
also check exact SCREEN rows through the role-scoped numeric reader. Shared
evaluation checks assignment maps, fixed draw/family identities and exact score
references before using a completed group summary. Missing seed triples wait;
they never become partial-seed null summaries.

Every new C block is independently complete in the shared store and its reference
is streamed immediately. At group completion or handled failure/interruption the
worker publishes a small group index, retaining completed candidates, selections
and scores. Keep the streamed index; Ctrl-C in the notebook forwards SIGINT so
the worker can publish this index. A killed process may require explicitly
reusing the last printed candidate ref. Unfinished C blocks and unfinished null
calculations restart; there is no per-head/iteration checkpoint.

After each seed's four-C selection freezes, score its control-only within-context
SCREEN system exactly once. Existing score refs prevent rescoring. The shared
evaluator handles every draw 1..1999 in 100-draw RAM batches and publishes only
one `body.json`/`maxima.npy` bundle per group across all three seeds. No routine
per-batch Drive writes, per-draw predictions, cached ranks, copied selected
weights or standardized input caches are added.

Use groups=[], seeds=[] for reduction-only work with explicit G0 indexes/refs.
All complete groups can reduce independently of fitting ownership. The global
gate is produced only at exact complete plan coverage; draw-wise maxima are
reduced, never per-worker p-values. Existing gate refs are reused with matching
group/assignment coverage. A gate flag preserves the original result and exits
4; it does not reroll seeds, delete groups or authorize interpretation. The
coordinator's separate validator remains responsible for Stage 3 acceptance.

Exit 0 means declared work finished (possibly partial coverage), 2 means blocked,
3 means some groups await ordinary statistics, and 4 means the full diagnostic
is flagged. Every worker result explicitly records stage3_complete=false.

## Costs and evidence

JSON progress includes phase/group/seed/C, elapsed wall and process CPU time,
30-second heartbeats, candidate refs, actual solver attempt seconds/retry-head
counts, numeric session payload read counts/bytes and OS process peak memory.
Read/hash preloads are timed separately; resident matrices/transforms are reused
across Cs/seeds and verified-byte sessions across groups. Recovery verification,
serialization, selection and shared publication/readback also consume time.
Fit/score/null phase timings include shared bundle publication and readback;
the frozen API exposes no distinct write versus readback timer. These combined
measurements must not be mislabeled as pure numerical time or Drive-only cost.
Peak memory is a process high-water mark including preparation/recovery, not
the allocation of an individual head. Read counts are for numeric sessions;
they do not claim to count every small recovery/store publication read.

Local checks use Python 3.12.14 and the pinned numerical packages with one thread.
The fixtures build an authenticated synthetic 145-group plan, fit real tiny
control blocks, exercise interrupted-C recovery and distributed seed handoffs,
reuse scores/null summaries without repeating solves, and verify payloads in a
fresh process. Mathematical checks compare moving primary unknown masks/tied
scores against sklearn AUROC, preserve all four image views, and compare all-draw
group-max reduction against a monolithic synthetic family. Incomplete/conflicting
coverage, corrupt statistics, invalid ownership and role violations block.

Actual mounted Stage 3 preparation references, ordinary statistics, representative
production fits and full production controls were not available in this local
workspace. No production artifacts or timings, ordinary SCREEN predictions,
CONFIRM access, new human work or scientific conclusions are claimed. Fixture
gate artifacts are software evidence only; no accepted registry is produced.

Verification performed on 16 November 2026: combined Stage 3 foundation/worker
suite passed 23 tests in 210.312 seconds. After the final null-seed coverage
guard, all 9 worker tests passed again in 40.747 seconds, including fresh-process
candidate/control-score/null payload readback. Notebook code cells compile with
empty outputs; CLI help, module compilation and staged whitespace checks pass.
The final fixture's seed1/seed2 resume leg took 5.652 seconds, ending at 12 C
blocks, 3 selections, 3 scores and 1 all-draw group summary. Its numeric sessions
recorded 29 payload reads / 476,122 bytes, including metadata. These local
four-image/two-feature fixture costs are not production estimates. Preparation
and worker CLI import paths produced equal pinned runtime records locally.

Shared-interface coordination: runtime() currently records discovered threadpools
in discovery order. A fixture plan created after importing sklearn.metrics failed
fresh-process load when only interfaces were imported; matching imports passed.
The verified foundation chat was notified to consider canonical ordering and an
import-order regression check before production binding. Shared files were not
edited. Notebook inspection and worker execution use isolated children with the
same import path; incompatible runtime checks continue to fail closed.
