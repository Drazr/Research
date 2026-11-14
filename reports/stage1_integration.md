# Stage 1 foundation review — 15 November 2026

**Stage 1 is not complete.** This report covers the coordinator foundation before the data/model branches have been implemented or integrated. No experiment result, real-model feature extraction or GPU validation is claimed.

## Verified now

- Read repository instructions, plan, current research scope and the preliminary shared protocol. The scheduled questions/experiments and frozen split/category targets are preserved.
- Shared `stage1.v1` data, model-adapter, storage and run-manifest interfaces are defined in docs/stage1_contracts.md and implemented in standard-library helpers.
- Exact deterministic caption-selection hashing is frozen before any model results. The original native-caption source IDs and actual selected captions still require the data branch.
- File ownership and task-branch responsibilities are recorded in docs/stage1_ownership.md.
- `python -m unittest discover -s tests -p test_stage1_foundation.py -v`: **11 tests passed** on Python 3.9. Software fixtures test Unicode preservation, deterministic selection, unknown masks, FIT-only gating, shared four-language identities, row swaps/nonfinite features, model fingerprints, decoder depth resolution, incomplete/corrupt artifact rejection, duplicate writers and path containment. These fixtures are not dataset/model evidence.
- Qwen and Gemma smoke CLI preflights were run separately under explicit `.local/phase1` storage. Each returned **blocked (exit 2)** for `Unresolved model configuration: checkpoint_id`, rather than passing or fabricating model output. Local completion reports preserve the blocked status and unknown model identity. Their model/configuration namespaces are distinct.

Actual development commands, from repository root:

```text
python -m unittest discover -s tests -p test_stage1_foundation.py -v
python scripts/stage1/run_validity_smoke.py --model qwen --phase1-root .local/phase1 --storage-scope local-development
python scripts/stage1/run_validity_smoke.py --model gemma --phase1-root .local/phase1 --storage-scope local-development
```

## Pending or unavailable

| Item | Observed status |
|---|---|
| S1-A/S1-B data preparation and validation implementation | Worker branch not integrated; complete native four-language image/caption dataset not established here |
| S1-C real model registry/adapters | Worker branch not integrated; Qwen/Gemma checkpoint, immutable revisions and processor/runtime details remain unselected |
| ML runtime | numpy, sklearn, torch and transformers are not installed in this checked Python environment |
| GPU access | Not validated; missing torch alone does not establish that the machine has no GPU |
| Real extraction/repeat/optimizer smoke positive path | Not exercised; needs the real worker implementations, data, selected checkpoints and runtime |
| Persistent Drive mount or Windows synced Drive path | Not verified; successful local-development writes do not certify Drive persistence |
| Full G0 shuffled-label controls | Pending Stage 3, after real feature caches exist |
| Bulk extraction, E01-E04/E19/E25 scientific results | Not performed |

The smoke runner can report blocked/failed prerequisites and, after integration, consume the common data and adapter APIs. Its toy pooled-view optimizer/permutation plumbing is not an E01 transfer experiment, a confidence result, or the full G0 random-target study. Backend/inspection correctness must be reviewed against each actual architecture; a boolean report alone is not independent causal evidence.

## Next handoff

The user receives the foundation commit SHA on phase1. Both worker chats branch from that exact SHA, follow their owned paths and preserve the shared interfaces. The coordinator reviews their changes and actual artifacts before merging and running real FIT-only end-to-end checks. Resolve any remaining data, model access, checkpoint selection or runtime gaps explicitly. Only update Stage 1 completion after the relevant real checks have passed.
