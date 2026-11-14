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


## Worker integration and L4 preparation — 15 November 2026

Integrated data head 2f42b4a81c404b18ff5a3f451f618593d56eec1a and model head 2f374d2c171c8b4d27099c00fe1214a476c13707, both based on ea06d93. File ownership/scientific-document boundaries were verified. The prior independent review reloaded and validated all 3,600 real images/14,400 selected records, preserving category order, split assignment, 14-category primary cohort and one unknown cell. This is local data evidence, not model or Drive evidence.

Integration fixes: removed hard-link source-cache publication; added streamed source-Path payloads and incremental verification; added cache tests for missing hard-link support, reuse, partial artifacts, corruption and legacy cache migration. Selected immutable Qwen2.5-VL-3B/Gemma-3-4B revisions and a direct-pinned L4 runtime. Added metadata-only L4 configuration resolution with explicit gated-access handling, hardware/version guard fixtures, safetensors-only model loading and dirty/unknown-checkout validation guards.

The combined suite has 27 passing CPU tests (11 foundation, 4 data, 7 model, 3 source-cache, 2 L4 guard tests). No real L4, weight load, metadata resolution through authorized Gemma access, or positive actual-model validation was executed here. The local environment lacks a mounted Drive root and nvidia-smi on PATH; this does not prove the user's L4 host lacks a GPU. Model pin files were obtained from the publisher's public APIs; Gemma metadata files returned HTTP 401 and its template/token fields remain unresolved. The user has chosen to reconnect a different Google Drive; no writes were made to the old account.

Stage 1 remains incomplete pending intended Drive access/mount, real filesystem/cache readbacks, runtime setup and actual FIT-only validation/smoke checks for both models. Full G0 and the scientific first-set experiments remain pending later stages. See docs/stage1_l4.md for the handoff commands and exact selected profile.


### Committed-code local continuation

Worker integration merge: 4030434c90183abd3c2b8416bbd0dec22b632335, preserving both worker heads as parents. The real data prepare/resume command passed on that committed source using the existing explicit local root. All four legacy source files were verified and migrated into completion-verified cache artifacts, preserving dataset fingerprint 54825d236efcd88cc5252918ebce6d1dad6109ef10a244281f262204bc6cc5bd. This is local-development evidence, not Drive durability evidence.

A runtime-configuration preflight initially failed to save its blocked report because nested hash directories exceeded the Windows path limit. Its configuration-cache namespace is now compact and covered by an end-to-end blocked-report regression. Early CUDA/bfloat16 guards prevent weight downloads when the requested device is absent. Slow/PIL image processing is explicit, matching torch/torchvision versions are pinned, all selected package versions are checked, and actual processor class is recorded/checked.

The combined software suite now has **29 passing CPU tests**, including early-device and blocked-report persistence regressions. The checked Python 3.12 environment contains numpy 2.3.5/Pillow 13.0.0 but lacks torch, transformers and scikit-learn; it is not the selected L4 runtime. Real configuration/model checks remain blocked here. The intended Google Drive account/mount and actual L4 execution are still pending. No Stage 1 completion is claimed.
