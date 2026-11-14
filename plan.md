# Phase 1 execution plan: G9-Q6

Created 14 November 2026. Status: planning; experiment scripts and model runs have not been executed by creating this file. Only Stage 1 is specified in detail. Stages 2-5 are an execution overview and will be specified one at a time.

## Scope and authoritative context

This phase starts with **G0, E01, E02, E03, E04, E19 and E25**. G0 is a validity gate, so these are seven scheduled tasks containing six experiment families. Other initial candidates and conditional follow-ups in `prelim.md` are not automatically scheduled by this plan.

Read [research.md](research.md) for the research questions and data boundaries, [prelim.md](prelim.md) for experiment definitions and the shared scientific protocol, and [M1.md](M1.md)/[M2.md](M2.md) for methodological context. [paper.md](paper.md) is the bibliography; [analysis.md](references/analysis.md) is historical literature analysis whose older proposals do not override the current constraints. This plan coordinates execution; it does not silently change a question or experiment. Log any necessary protocol change before affected model outcomes are inspected.

The scientific scope is category accessibility, cross-context readout compatibility and diagnostic probability/correctness reliability in frozen Qwen and Gemma VLMs, across English, Hindi, Bengali and Telugu. Existing image-presence labels are shared across languages; independently written captions are not controlled translations. No new human annotation, semantic verification, native terminology inventory, manual dataset enrichment, base-model training, activation patching or generated-caption judging is required. Automatic preparation and checks remain necessary.

## Five-stage overview

| Stage | Purpose and shared work | Parallelism and handoff |
|---|---|---|
| **1. Shared setup and initial validity checks** | Prepare the common four-language data manifest, frozen image partitions, labels/masks, model configurations, storage conventions and small extraction/optimizer checks. | Model-independent preparation happens once. Model-specific checks use the same entry point with different model keys. Freeze shared interfaces before chats implement dependent scripts. |
| **2. Shared feature extraction and E03** | Extract the representations needed by E01/E04 together wherever serialized inputs and masks match. Reuse E02 visual-stage features when genuinely identical; its caption-only condition needs additional extraction. E03 uses the same captions/partitions on CPU. | E03 can overlap GPU extraction once its data inputs are frozen. Independent model jobs can run on separate available GPUs; on one GPU, queue extraction while CPU work proceeds. |
| **3. Probe fitting and remaining validity controls** | Fit E01/E02/E04 readouts from validated caches; complete the full G0 shuffled-label controls on the actual feature pipeline. | Independent fits can overlap as their FIT and DEV-S caches become ready. No substantive interpretation proceeds past a failed G0 check. |
| **4. Shared scoring and downstream analyses** | Save E01 DEV-C/SCREEN predictions once. Reuse those predictions for E01 metrics, E19 confidence policies and E25 paired-error analysis; produce E02/E04 scores from their fitted systems. | E19 and E25 can run together as soon as their required E01 predictions exist; they do not wait for unrelated E02/E04 jobs. Final comparative analyses wait for their actual inputs. |
| **5. Integration and preliminary findings** | Verify matching cohorts, model/config identities, masks and selection histories; assemble the complete results and unresolved questions. | Review each model separately before comparing patterns. Decide later preliminary work from actual evidence while keeping CONFIRM closed. |

These stages express artifact dependencies, not five global barriers. For example, a validated model's Stage 3 fitting can overlap another model's Stage 2 extraction. Do not wait for an unrelated experiment to finish. Do not consume a partially written or unvalidated artifact.

## Stage 1: detailed plan

### 1.1 Goal and completion boundary

Make every subsequent chat use the same data, scientific rules, model identity and artifact conventions. Stage 1 produces preparation/configuration artifacts and diagnostic smoke reports. It does not choose promising languages, categories, layers or directions using experimental performance.

G0 is a staged gate. Its data/extraction checks begin here. Its full shuffled-label controls require the real feature caches and therefore finish in Stage 3. Passing Stage 1 means ready for bulk extraction and CPU baselines; it does not mean the full G0 controls or scientific experiments have passed.

### 1.2 One model variable and model-specific configuration

Every script that performs model-specific work must accept a single model selector, provisionally `--model qwen` or `--model gemma`. These are planned interfaces, not existing commands. A shared registry resolves the selector to the exact configuration; changing the selector must not require editing experiment logic.

Each registry entry records at least:

- Checkpoint identifier and immutable revision, plus processor/tokenizer and chat-template revisions or hashes.
- VLM adapter, exposed feature sites, block-output hook convention and actual modality/attention visibility.
- Precision/quantization policy, image resize/crop policy, maximum-input handling and relevant software versions. These choices are explicit, not silently inferred from whichever environment runs a chat's code.
- The four fixed decoder depths and midpoint resolved from that checkpoint's actual decoder block count.
- A model-configuration fingerprint. The aliases `qwen` and `gemma` alone are not enough to identify cached results.

The selected starting pair is Qwen/Qwen2.5-VL-3B-Instruct and google/gemma-3-4b-it, with immutable revisions in configs/models.json. See docs/stage1_l4.md for the selected L4 runtime profile. Authorized processor-metadata resolution and actual model checks remain prerequisites before bulk inference. Different revisions/configurations produce different fingerprints and new output namespaces; they must not overwrite an earlier run under the same alias.

Use one generic experiment entry point with model adapters for architecture-specific loading, token spans and hooks. Model-independent preparation/E03 should not be repeated separately for each model. If a genuinely different model-specific script is necessary, a name such as `e04_positions_gemma.py` is acceptable, but it must use the same data/scoring contracts and document the reason. Prefer adapters over copying a whole experiment pipeline.

Different model jobs can run concurrently when hardware permits. All outputs are separated by model fingerprint and run ID. A single GPU does not imply several large VLM processes should be launched simultaneously; queue extraction and overlap it with CPU work.

### 1.3 Shared storage: a Drive folder named phase1

Set a configurable `PHASE1_ROOT` to the persistent Drive folder named `phase1`. This is a storage proposal; creating this document has not created a remote folder, uploaded data or downloaded model weights. No absolute machine-specific Drive path is hard-coded in scripts.

Suggested layout:

```text
<DRIVE_ROOT>/phase1/
  datasets/xm3600/<dataset_fingerprint>/
    images/                         # One common image collection
    captions/
      en.jsonl
      hi.jsonl
      bn.jsonl
      te.jsonl
    annotations/                    # Original shared image labels
    provenance.json
  shared/<protocol_fingerprint>/
    manifests/
    splits/
    labels_and_masks/
    validation/
  cache/shared/<dataset_fingerprint>/  # Persistent model-independent caches
  cache/<model_fingerprint>/<protocol_fingerprint>/
  features/<model_fingerprint>/<protocol_fingerprint>/shards/
  checkpoints/<model_fingerprint>/<protocol_fingerprint>/<run_id>/
  predictions/<model_fingerprint>/<protocol_fingerprint>/<run_id>/
  results/<model_fingerprint>/<protocol_fingerprint>/<run_id>/
  logs/<model_fingerprint>/<run_id>/
  smoke/<model_fingerprint>/<run_id>/
  model_cache/                      # Optional reusable base-weight cache
```

The four caption files are automatically prepared views of the original release, preserving original text and provenance. They are not four copies of the images or four separate annotations. A single equivalent caption table with a language column is also possible, but Stage 1 must select and freeze one reader contract for all chats.

Checkpoints include diagnostic readout weights, scalers, selected hyperparameters and any needed restart state. Frozen base-model weights are reusable inputs, not new trained outputs; do not duplicate them in every experiment/run directory. Shared E01 predictions are referenced by E19/E25 rather than copied into separately drifting versions.

Persistent caches and resumable shards also live under this phase1 root. Common dataset/index caches are shared; tokenizer/processor-dependent caches and feature shards are model/configuration-specific. Temporary scratch may remain local and is not an authoritative artifact.

One producer owns each artifact namespace/shard. Concurrent jobs write unique run/shard paths and finish with a completion manifest containing hashes and provenance. Consumers check completion and file integrity; they do not read partially synced Drive files. Live work may use local scratch, with verified persistent checkpoints on Drive. Never have multiple jobs append to one shared result file or overwrite the same checkpoint.

### 1.4 What belongs in the repository and main

The Research repository should version the documentation, scripts, model/protocol configurations, environment specification and small reproducibility metadata: split IDs, category order/cohort, mask rules, source revisions/hashes and data-fetch instructions. Carry the existing `g9q6_split_planning_audit.json` into the repository's metadata area as the starting partition/support record. The reviewed experiment-card/phase manifests can accompany `prelim.md` for traceability.

The full image/caption dataset does **not** need to be added to main. Put it in the shared Drive dataset folder and make it recoverable from pinned source/provenance instructions. Raw images, model weights, feature arrays and generated checkpoints/predictions stay out of ordinary Git history. A small pinned annotation snapshot can accompany its provenance if desired; it does not replace a deterministic loader or the mask specification. Configure ignore rules before adding runtime outputs.

Main holds reviewed shared context and interfaces. `phase1` can be the integration branch for this work. Five stage directories are useful, but directory separation alone is not separate Git working state. Recommended parallel development uses task branches/worktrees based on `phase1`, for example `phase1-stage1-data` and `phase1-stage1-models`, followed by integration into `phase1`. [Git's worktree documentation](https://git-scm.com/docs/git-worktree) describes separate working trees and their branch checkout rules.

Use a structure such as:

```text
Research/
  plan.md
  AGENTS.md
  research.md
  prelim.md
  M1.md
  M2.md
  paper.md
  references/analysis.md
  configs/
  data/manifests/                    # Small versioned metadata only
  src/common/
  src/models/
  scripts/stage1/
  scripts/stage2/
  scripts/stage3/
  scripts/stage4/
  scripts/stage5/
```

Experiment-specific script names can include their ID and readable name, e.g. `e01_accessibility.py`; the containing stage identifies its role. Shared work should have one shared script, such as an extraction entry point, rather than separate copies under every experiment. Python scripts are the reproducible execution entry points; notebooks may import them for inspection/plots. Names and directories here are proposed, not files already implemented.

### 1.5 Data preparation and frozen scientific rules

Prepare one common image/caption/label manifest before model-dependent work:

1. Recover the original XM3600 images and native English/Hindi/Bengali/Telugu captions with pinned provenance. Do not accidentally substitute English translations stored in a derived non-English `caption` field; the derived records' original-text field is `orig`. Prefer the original release. Native captions are independent descriptions, not parallel sentence translations. Do not use Bengali/Telugu mention-extraction proxies.
2. Canonicalize image IDs losslessly as 16-character lowercase hexadecimal strings. Join by IDs, not array position or floating-point-converted IDs. Check duplicates, file availability, original-text field provenance, language coverage and caption multiplicity automatically. Define the common valid-image population once for both model families and all languages.
3. Keep one image-level vector of all 25 category bits plus an explicit known-cell mask and immutable category order. Shared labels apply to every language. Under the pinned release, blank category cells retain its absent convention. The invalid `plant.n.02 = 4` cell for image `6a20677eee04fd08` is unknown; do not repair it manually or turn it into a count/negative.
4. Start from the existing frozen image partitions and support audit. With all 3,600 images available, use FIT 2,160 / DEV-S 180 / DEV-C 180 / SCREEN 360 / CONFIRM 720. Assignment is sorted `SHA256("G9Q6-first-set-20261114-v1" + "|" + canonical_image_id)`, then contiguous blocks in that order. Keep every caption/language/view of an image together.
5. If the actual release intersection is smaller, version a deterministic replacement split and recompute supports **before model scores**; document exclusions and the exact size-allocation rule. Never silently reuse the full-dataset counts or reroll for an attractive effect. Model-specific input errors do not silently redefine separate four-language cohorts.
6. Preserve all 25 targets, while keeping the primary macro cohort fixed from label support: at least 20 known positives and negatives in FIT/SCREEN, and at least 10 of each in DEV-S/DEV-C. The current full-population audit has 14 primary categories. Low-support categories remain secondary or unestimable, not zero-valued scores.
7. Finalize the exact deterministic caption-draw rule and stable source caption IDs, and save the chosen draw-0 IDs. Both model families and all relevant experiments reuse this selection. Captions selected for the same image in different languages are not treated as equivalent sentences.

Pinned annotation Git blob: `6b99def06a621c570edf074c26741868a4df7353`, from the [annotation repository CSV](https://github.com/uriberger/cross_lingual_diff_in_descs/blob/main/data/xm3600_annotation.csv). Record a content checksum alongside source/revision metadata. The existing audit is reproducibility metadata, not a new human check of annotation truth.

Metadata completeness and frozen support counts may be checked across the released IDs. Preliminary model input processing, feature extraction, prediction and outcome inspection must exclude CONFIRM. Stage 1 model smoke inputs come only from FIT, with an inner image-group validation split if a toy readout is fitted.

### 1.6 Minimum contracts to agree before parallel implementation

Freeze logical identities and reader/writer responsibilities now; detailed Stage 2-5 tensor formats and numerical implementation choices will be specified when those stages are planned. No chat independently invents a conflicting schema.

| Contract | Required identity or meaning |
|---|---|
| Dataset record | Canonical image ID, shared image location/checksum, language, original caption ID/text and source provenance |
| Target record | Image ID, fixed category IDs/order, binary values, known-cell mask and split; labels are not part of model input text |
| Model identity | Model selector plus exact checkpoint/processor/template configuration fingerprint |
| Feature identity | Image ID, language/caption draw, input condition, decoder depth or visual stage, readout site, model/protocol fingerprints; row IDs accompany arrays |
| Prediction identity | Image/category, source-head language, receiving-context language, site/depth, fitted-system ID, probability/logit and known-cell mask |
| Run record | Run ID, code commit, resolved configuration, data/split/mask hashes, seeds, versions, upstream artifact IDs, completion state and output checksums |

Shapes and compatibility are per model/site, not globally assumed equal between Qwen and Gemma. Target-fitted systems and unchanged-source transfer must remain distinguishable. Independently running scripts must reuse the same target order, caption selection and source-fitted preprocessing. Consult `prelim.md` for the readout/scoring protocol rather than choosing a new loss, threshold, class weighting or selection split within a chat.

### 1.7 Stage 1 implementation tasks and ownership

Agree the common manifest/config interface first. Then parallelize the following work with explicitly owned files; another chat may review them but should not simultaneously rewrite them.

| Work item | Proposed script or area | Dependencies and execution |
|---|---|---|
| S1-A: dataset preparation | `scripts/stage1/prepare_dataset.py`, data-reader module | Model-independent; builds the common manifest, original caption views, selections, labels/masks and versioned split metadata once |
| S1-B: automatic data validation | `scripts/stage1/validate_data.py` | Depends on S1-A outputs; verifies joins, support, partition disjointness, masks, checksums and deterministic reload |
| S1-C: model registry and adapters | `configs/`, `src/models/`, `scripts/stage1/validate_model.py` | Can be implemented alongside S1-A after the shared contract is fixed; execute separately with each model selector after its configuration and FIT smoke inputs are ready |
| S1-D: end-to-end smoke check | `scripts/stage1/run_validity_smoke.py` | Depends on validated data/adapter interfaces; small FIT-only extraction and readout-optimizer/label-permutation plumbing checks, model-namespaced outputs |
| S1-E: integration/handoff review | Stage 1 report plus owned updates to `plan.md` | Verifies both model smoke reports and the shared artifact/config identities before bulk extraction is enabled |

These names are implementation proposals, not existing scripts. Keep a single owner for shared helpers, registry schemas and `plan.md`; model-adapter files can have separate owners. Every chat must read the common constraints before modifying its owned part. Use unique run outputs even if two chats run the same entry point with different model variables.

S1-C/S1-D must check actual processed token spans, attention visibility and repeatable evaluation-mode extraction. Verify the last actual caption token, caption-only pooling, optional post-caption boundary and exposed visual encoder/projector interfaces. Resolve the four depths `ceil(L/4), ceil(L/2), ceil(3L/4), L`, deduplicate, and record a consistent block-output residual-stream convention. Do not silently mix a final-normalized last state with earlier unnormalized states. Log unsupported sites as unsupported; do not force comparability through activation editing.

The default native chat input is an image before the unchanged original caption with a shared neutral field prefix such as `Caption:\n`; no category label, target answer or image-label filename is inserted into the input. Build the template once per configured model and reuse it across languages. Perform a small deterministic repeat check, recording precision/backend tolerances rather than promising bitwise equality across all devices.

Toy readout checks can verify dimensions, finite values, loss convergence, prior/constant baselines and correct permutation/mask handling on FIT-only smoke data. Unsupported toy categories are skipped explicitly. Such a small run cannot certify population-level chance performance or scientific signal. Full G0 controls retain the three fixed within-partition whole-vector label permutations shared across languages, with masks traveling with their vectors; these use the validated real caches in Stage 3. Smoke checkpoints stay labeled as smoke and never masquerade as final scientific fits.

### 1.8 Stage 1 deliverables and acceptance checklist

- [ ] Common four-language dataset/caption reader contract, source revisions/hashes and automatically logged exclusions.
- [ ] Exact image-ID partitions, draw-0 caption selection, fixed category order/cohort and shared label/mask artifacts with checksums.
- [ ] Concrete selected Qwen and Gemma registry entries, model fingerprints and resolved extraction-site/depth reports.
- [ ] Configurable Drive `phase1` root and documented namespace/completion conventions; a small write/read check succeeds before runtime artifacts rely on it.
- [ ] CPU data-validation report and FIT-only smoke reports for each selected model, with unavailable sites and numerical limitations explicit.
- [ ] Evidence that model inputs cannot receive gold labels through text/filenames and preliminary model work excludes CONFIRM.
- [ ] Documented script ownership, small shared interfaces, environment/revision records and reproducible command/config conventions.
- [ ] Stage 1 integration review records pass, blocked or unavailable items accurately; full G0 shuffled-label completion is explicitly pending Stage 3.

Only then specify the detailed Stage 2 extraction and E03 implementation plan. The integration reviewer records completion in this file; individual chats report their own artifacts and checks without independently declaring the entire stage finished. No Stage 1 model check or experiment is marked complete merely because this plan has been written.

## Current status

Updated 15 November 2026 during coordinator integration.

- Data branch 2f42b4a and model branch 2f374d2 are integrated into this phase1 snapshot with their histories preserved. Main remains unchanged.
- Original data independently revalidated locally: 3,600 images, 14,400 selected four-language records, 14 primary categories and one preserved unknown cell. No scientific results were produced.
- Shared stage1.v1 public APIs remain compatible; source-cache/storage helpers gained streaming, completion-verified publication without hard links. Cache corruption/partial-publication checks passed locally.
- Combined CPU foundation/data/model/cache/L4-guard suite: 29 tests passed. Tests using fake devices/modules are software checks, not actual GPU/model evidence.
- Selected L4 starting pair: Qwen2.5-VL-3B-Instruct and Gemma 3 4B IT, sequential dense bfloat16, batch one. Immutable revisions and the direct runtime pins are recorded; actual L4 execution/memory fit remains untested. Gemma gated processor metadata remains unresolved pending authorized access.
- The user chose a different Google Drive account. No folder/artifact was created in the previously connected account. PHASE1_ROOT and actual Drive-mounted read/write/sync checks remain pending in the intended runtime.
- Use docs/stage1_l4.md for actual storage/configuration/model-validation commands. Detailed Stage 2-5 planning remains pending. Stage 1 and full G0 are not complete.
