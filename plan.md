# Phase 1 execution plan: G9-Q6

Stage 3 worker integration update, 16 November 2026: ordinary-fit and G0 worker
modules/notebooks are now present alongside the coordinator. Follow
[the integrated runbook](docs/stage3_runbook.md): preparation, agreed ownership,
ordinary profiles, G0 seed0 profiles, measured full runs with profile reuse,
complete G0 reduction, then final validation. Runtime-order correction and
metadata-only completed-score recovery are included. Production acceptance and
scientific outcomes remain pending; older worker-pending notes describe history.

Implementation update, 16 November 2026: the Stage 3 coordinator foundation now exists. Read [executable interfaces and ownership](docs/stage3_interfaces.md) and [foundation evidence](reports/stage3_foundation.md). CPU numeric-only plan/session/statistics/fit/selection/control/recovery/validation primitives and prepare_stage3/validate_stage3 notebooks are implemented. Ordinary-fit and G0 scheduling workers are not implemented; mounted suite binding, representative production measurements and full Stage 3 acceptance remain pending. Use the first-add commit of docs/stage3_interfaces.md as the immutable worker base, not the earlier planning commit. The planning-only status below describes the prior design review.

Created 14 November 2026; Stages 1-3 are now specified in detail as of 16 November 2026. The user supplied the final Stage 2 validation status=passed at results/stage2-validation/20261116T010841Z-17d614fe5da340b4b6c036a431fc6be8/_COMPLETE.json under PHASE1_ROOT. This is user-reported runtime evidence; first Stage 3 preparation verifies the exact accepted suite and consumed numeric bytes. Stage 3 review revision 2 is documentation/planning only, with no implementation or fits. See the stage-specific contracts, integration and planning reports. Stages 4/5 remain an execution overview and will be specified separately. Software implementation or planning does not establish experimental completion.

## Scope and authoritative context

This phase starts with **G0, E01, E02, E03, E04, E19 and E25**. G0 is a validity gate, so these are seven scheduled tasks containing six experiment families. Other initial candidates and conditional follow-ups in `prelim.md` are not automatically scheduled by this plan.

Read [research.md](research.md) for the research questions and data boundaries, [prelim.md](prelim.md) for experiment definitions and the shared scientific protocol, and [M1.md](M1.md)/[M2.md](M2.md) for methodological context. [paper.md](paper.md) is the bibliography; [analysis.md](references/analysis.md) is historical literature analysis whose older proposals do not override the current constraints. This plan coordinates execution; it does not silently change a question or experiment. Log any necessary protocol change before affected model outcomes are inspected.

The scientific scope is category accessibility, cross-context readout compatibility and diagnostic probability/correctness reliability in frozen Qwen and Gemma VLMs, across English, Hindi, Bengali and Telugu. Existing image-presence labels are shared across languages; independently written captions are not controlled translations. No new human annotation, semantic verification, native terminology inventory, manual dataset enrichment, base-model training, activation patching or generated-caption judging is required. Automatic preparation and checks remain necessary.

## Five-stage overview

| Stage | Purpose and shared work | Parallelism and handoff |
|---|---|---|
| **1. Shared setup and initial validity checks** | Prepare the common four-language data manifest, frozen image partitions, labels/masks, model configurations, storage conventions and small extraction/optimizer checks. | Model-independent preparation happens once. Model-specific checks use the same entry point with different model keys. Freeze shared interfaces before chats implement dependent scripts. |
| **2. Shared feature extraction and E03 inputs** | Extract shared E01/E04 decoder features, E02 caption-only and once-per-image visual references, with explicit visual-reuse audits. Fit E03 vocabulary/IDF on FIT captions and transform eligible partitions on CPU; no category heads or scores yet. | E03 preprocessing can overlap a ready model's GPU extraction. Independent models need separate GPUs to overlap; on one L4 queue model processes. Publish verified condition/partition releases as they complete. |
| **3. Probe fitting and remaining validity controls** | Fit E01/E02/E03/E04 readouts from validated caches; complete full G0 shuffled-label controls on the actual feature pipeline. | Current v1 first binds the complete passed Stage 2 suite. Then disjoint source-group fits can overlap on separate CPU runtimes; G0 reuses each group's published statistics without waiting for unrelated ordinary groups. No substantive interpretation proceeds past a failed G0 check. |
| **4. Shared scoring and downstream analyses** | Save E01 DEV-C/SCREEN predictions once. Reuse those predictions for E01 metrics, E19 confidence policies and E25 paired-error analysis; produce E02/E03/E04 scores from their fitted systems. | E19 and E25 can run together as soon as their required E01 predictions exist; they do not wait for unrelated E02/E03/E04 jobs. Final comparative analyses wait for their actual inputs. |
| **5. Integration and preliminary findings** | Verify matching cohorts, model/config identities, masks and selection histories; assemble the complete results and unresolved questions. | Review each model separately before comparing patterns. Decide later preliminary work from actual evidence while keeping CONFIRM closed. |

These stages express artifact dependencies with explicit validity gates. The finalized v1 Stage 3 entry gate is a complete passed Stage 2 suite; the earlier proposal to fit from partial Stage 2 releases is not enabled by this version. Within Stage 3, disjoint groups and ordinary/G0 workers need only their own accepted inputs/shared statistics. Stage 4 scientific scoring waits for the full Stage 3 registry/G0 gate; E19/E25 can then overlap as their canonical predictions become ready. Do not consume a partially written or unvalidated artifact.

### Stage 3-5 reuse requirement (user decision, 15 November 2026)

This records the cross-stage reuse requirement; the concrete Stage 3 numerical/cache plan is now in docs/stage3_contracts.md. No implementation is created by either note. Cache expensive reusable results once, with shared readers and immutable compatibility identities, so different scripts consume the same outputs. Store them under PHASE1_ROOT; do not duplicate artifacts merely to give each experiment a separate copy.

| Artifact to retain | Purpose and reuse boundary |
|---|---|
| Stage 2 raw features and E03 fitted preprocessing | Already-produced inputs; reuse across heads, C candidates and relevant experiments without repeating VLM extraction or vocabulary/IDF fitting |
| Source-FIT scalers/statistics | Share only when training rows, weighting and preprocessing recipe are identical; never estimate them from DEV/SCREEN |
| Completed readout fits, convergence and DEV-S selection records | Resume fitting/search and reuse frozen systems across scoring scripts; checkpoint a meaningful fit block, not every optimizer iteration |
| G0 image-vector permutations, traveling masks and completed controls | Keep fixed shared control identities across languages/methods and reuse completed expensive fits/results; control targets remain separate from true labels |
| Canonical logits/probabilities and prediction row/system identities | Produce at the permitted later scoring stage once; E01/E19/E25 and other declared metrics consume them without refitting or repeated inference |
| E19 source-DEV-C cutoffs/policy specification | Reuse exactly the frozen source policy; preserve source-only fitting, confidence event and deterministic tie handling |
| Shared bootstrap membership/recipe and completed expensive metric summaries | Preserve paired image resampling and resume analysis blocks; reuse summaries across reports/plots when predictions/cohorts/analysis settings match |
| Final report tables, support/undefined flags and provenance | Stage 5 assembles verified results without recomputing expensive fitting/scoring; cheap derivations need not create standalone caches |

Raw-feature, fitted-system, prediction and analysis identities remain separate: changing a metric/bootstrap/plot must not invalidate raw features or unchanged heads. Additional calibrators/adaptation maps are retained only if a later declared experiment requires them. No such experiment is added here. Model outputs remain separated by MODEL and full identity; all known masks/splits/source supervision rules remain enforced on cache hits.

Choose bounded checkpoint bundles and frequency during Stage 3 implementation planning, based on meaningful completed work and observed I/O/recovery cost. Keep producer/consumer/resume integrity checks and cheap role guards, while reducing repeated whole-dataset verification against the same immutable in-process inputs. Do not cache every temporary tensor, save frozen base weights repeatedly, or change an active Stage 2 run under this preference.

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
| S1-A: dataset preparation | `src/stage1/prepare_dataset.py`, data-reader module | Model-independent; builds the common manifest, original caption views, selections, labels/masks and versioned split metadata once |
| S1-B: automatic data validation | `src/stage1/validate_data.py` | Depends on S1-A outputs; verifies joins, support, partition disjointness, masks, checksums and deterministic reload |
| S1-C: model registry and adapters | `configs/`, `src/models/`, `src/stage1/validate_model.py` | Can be implemented alongside S1-A after the shared contract is fixed; execute separately with each model selector after its configuration and FIT smoke inputs are ready |
| S1-D: end-to-end smoke check | `src/stage1/run_validity_smoke.py` | Depends on validated data/adapter interfaces; small FIT-only extraction and readout-optimizer/label-permutation plumbing checks, model-namespaced outputs |
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

At the user's explicit request, Stage 2 planning may be finalized before all Stage 1 runs finish; its execution remains gated on the applicable real acceptance evidence. The integration reviewer records completion in this file; individual chats report their own artifacts and checks without independently declaring the entire stage finished. No Stage 1 model check or experiment is marked complete merely because a plan has been written.

## Stage 2: finalized implementation specification

Planning version stage2-plan.v1, 15 November 2026. Full operational details and payload/role contracts are in [docs/stage2_contracts.md](docs/stage2_contracts.md); implementation ownership and parallel handoff are in [docs/stage2_ownership.md](docs/stage2_ownership.md). These documents are required reading before writing Stage 2 code. No Stage 2 notebooks, modules, configuration files or runtime outputs are created by this planning change.

Final review revision 2 incorporates the user-confirmed final smoke passes for both models and closes implementation handoff gaps: typed evidence verification and deterministic compatible-artifact discovery; independent shared data/E03 and per-model plans; one source of notebook settings; a same-forward token-evidence bridge; partial E03 split releases; and resource projections based on actual first-chunk measurements. These are operational details, not changes to the scientific input/feature recipes or authorization to skip acceptance.

### 2.1 Purpose and launch boundary

Create reusable raw features and E03 preprocessing without looking for a winning model/language/site. GPU launch is per model: require passed common data, matching generated configuration, actual model-validation and final smoke artifacts, plus a verified readiness receipt. A ready model need not wait for another blocked one. E03 preprocessing needs common data acceptance and runs once on CPU. Whole-stage acceptance requires both models and all required E03 outputs. The latest evidence does not yet authorize claiming these gates passed for both models.

### 2.2 Frozen workload and feature inventory

- Reuse the existing dataset/protocol fingerprints, draw-0 caption selection and image partitions. No data rebuilding, extra captions, native semantic verification, translation or CONFIRM preprocessing.
- Eligible population: 2,880 images from FIT/DEV-S/DEV-C/SCREEN; 11,520 caption rows per condition across en/hi/bn/te. Keep all 25 targets/masks and the 14-category reporting cohort unchanged.
- For each model extract image_caption and caption_only, each with last_caption and caption_mean at every validated fixed depth. Capture both poolings/required depths in the same forward; E01/E04 share these values. Obtain one actual image_only reference per image at every validated exposed visual encoder/projector site, at least one stage required. Image-only is not four language observations.
- post_caption and decoder visual-token pooling remain explicitly unsupported/not scheduled in v1. E04's first implementation covers the supported caption-pooling contrast and documents its limits. Adding optional sites requires a pre-outcome versioned plan and actual span/visibility validation, not a silent implementation choice.
- Preserve the L4 dense-bfloat16/eager/batch-one/1,024-token/no-truncation policy and actual physical site conventions. Store raw unscaled float32 vectors, never model-family-agnostic coordinates. Capture/verify combined visual hooks against explicit image-only references across the four actual contexts before any visual alias/reuse claim.

### 2.3 Chunking, ordering, storage and resume

Use 32-image storage chunks, with all four language views together and no cross-partition chunk. This gives 92 image chunks and 276 condition-chunks per model: FIT 68, DEV-S 6, DEV-C 6, SCREEN 12, each with three conditions. Storage chunks do not change actual tensor batch size one. The core workload is 25,920 unique forward inputs per model, plus deterministic repeat checks.

Order partitions FIT, DEV-S, DEV-C, SCREEN and image IDs lexically within their already frozen partitions. Process image-only then four combined and four caption-only views per image; retain bounded CPU chunk buffers. One producer owns each unique run/chunk namespace under PHASE1_ROOT. Use existing checksum/completion helpers, portable numeric payloads, immutable job/release indexes and layered feature keys. Resume only exact compatible completed row/site coverage, leaving partial/conflicting attempts immutable and explicitly diagnosed. No retry may silently truncate, change precision/processor, replace captions or drop images.

Ready condition/partition releases enable downstream implementation and input checks. The earlier streaming proposal allowed FIT/DEV-S fitting while unrelated extraction continued; finalized Stage 3 v1 instead requires the complete accepted suite, now user-reported passed. Incremental production launch would need an explicit future interface/gate change. No consumer discovers arbitrary shards by directory existence. Actual mounted-path readback and deterministic initial/repeat checks precede release. Exact schema, source hashes, backend identity, row/site arrays, token evidence, conflict handling and release validation are fixed in stage2_contracts.md.

### 2.4 E03 preprocessing specification

Create 13 model-independent families: per-language whitespace-token word TF-IDF, literal-character TF-IDF and two-column log-length features (12), plus one pooled four-language character TF-IDF (1). Word ngrams are 1-2, min_df=2, cap 20,000; character ngrams are 3-5, min_df=2, cap 50,000. Keep case, accents, Unicode and original whitespace under the declared literal character analyzer; do not introduce language-specific NLP resources. The contract fixes vocabulary tie/order rules, sublinear tf, smoothed idf, row L2 normalization, literal analyzer behavior and finite float32 sparse/dense storage.

All vocabulary/IDF fitting uses the relevant FIT rows only. Transform DEV-S/DEV-C/SCREEN with the frozen FIT recipe; do not process CONFIRM. Preserve zero-feature rows and row identities. Per-language vocabularies are independent; pooled character preprocessing is explicitly multilingual, not source-only transfer. No label heads, C/layer selection, calibration or scientific scores occur in Stage 2. Stage 3 later handles shared source-FIT scaling, logistic readouts and pooled-view equal image weighting; Stage 4 handles scoring.

### 2.5 Ownership, integration and completion

The coordinator first implements/commits common Stage 2 interfaces, readiness/job/release validators, the same-forward evidence bridge and ownership at an immutable shared base. Then the extraction and E03 workstreams can implement separately against that base; no chat independently invents formats or edits frozen shared scientific rules. Future notebook entry points use scripts/stage2 with reusable src/stage2 modules, a single MODEL variable where applicable and secret-safe per-runtime authentication. Worker notebook data/config paths resolve from the explicit verified plan instead of repeated placeholder/path cells. Shared data/E03 planning is independent of each model's launch binding. Listed files in the ownership document are proposed future ownership, not present code.

Acceptance requires all expected rows/sites for both models, all E03 families, exact common masks/cohorts and source identities, no CONFIRM processing, passed relevant software/actual runtime/restart/readback checks, explicit optional-site limitations and a coordinator report referencing verified release indexes. Full G0 random-target controls remain Stage 3. Detail and status checklist: [reports/stage2_planning.md](reports/stage2_planning.md).

Implementation update: the coordinator executable shared interfaces, evidence bridge, schemas, readers/writers, preparation/validation notebooks and software tests are implemented. See docs/stage2_interfaces.md for the frozen API/ownership and reports/stage2_implementation.md for verified historical evidence and remaining gates. Branch both workers from the exact coordinator publication SHA. Extraction/E03 implementations, mounted readiness receipts and production artifacts remain pending; Stage 2 is not complete. Stages 3-5 are not finalized by these handoff requirements.

## Stage 3: finalized fitting/control specification — 16 November 2026

Stage 3 review revision 2 is specified in [docs/stage3_contracts.md](docs/stage3_contracts.md), with [implementation ownership](docs/stage3_ownership.md) and a [planning review](reports/stage3_planning.md). This update is documentation only. Stages 4/5 remain overview/handoff requirements, not finalized implementations. The final Stage 2 suite now has a user-reported passed verdict at the pointer above; Stage 3 first-consumer verification binds that exact artifact without rerunning the entire Stage 2 validator.

The Stage 3 workload is derived from accepted site/recipe inventories: 145 ordinary systems, 4 C candidates and all 25 category statuses; the same 3-seed randomized-target fitting/selection pipeline adds matched controls. This gives 14,500 ordinary and 43,500 control nominal binary fits before fixed retries. Reuse cached raw features/E03 preprocessing and label-free source statistics. Fit independent float64 logistic heads with the documented source-only scaler, image weights, exact objective/solver/retry and fixed-primary-cohort C-selection/tie rules. Selected systems reference FIT-only weights; no refit on development data or real SCREEN tuning. The 17 November computational amendment adds a third 20000 cold-start budget after unchanged 1000/5000 failures, based on a diagnostic-only head converging at 5511. Use a new plan identity with authenticated budget-only compatibility for completed original units; see docs/stage3_solver_amendment.md. Production recovery/global acceptance remain pending.

The shared numeric-only reader verifies accepted metadata and consumed numeric bytes once per bounded session; it must not call recursive Stage 2 loaders that recheck images. Reuse inputs across heads/Cs/seeds/sites/languages where memory permits and retain cheap role guards. Checkpoint complete 25-head C blocks, source statistics, selections and control scores. G0 computes in 100-draw RAM batches but saves one all-draw null summary per source group, then the final gate; no routine per-batch Drive checkpoints. Stream progress and profile predeclared representative production blocks before bulk work, reusing them afterward. The coordinator first freezes tested numerical/cache APIs; ordinary-fit and G0 workers reuse that engine. Future CPU notebooks are prepare_stage3, fit_readouts, run_g0 and validate_stage3, with shared modules under src/stage3; these are planned paths, not existing commands.

G0 uses three fixed whole-image target-vector/mask permutations and source-DEV-S selection. Control-only SCREEN scoring and a predeclared conditional global randomization diagnostic complete its validity gate; ordinary scientific predictions, source DEV-C confidence policies and E19/E25 reuse remain Stage 4. A flagged/failed control or structural/solver/identity violation blocks substantive interpretation until resolved. No GPU, new dataset/manual annotation or extra scientific experiment is required. Stage 3 implementation/runtime acceptance remains pending.

## Current status

Latest runtime evidence, 16 November 2026: the final Stage 2 validator reported passed at results/stage2-validation/20261116T010841Z-17d614fe5da340b4b6c036a431fc6be8/_COMPLETE.json. This supersedes pending-production/suite statements in the historical integration notes below. This review did not independently inspect Drive or produce Stage 3 code/runs. Its finalized design uses numeric-only consumer verification and limited meaningful checkpoints; implementation interfaces and runtime acceptance remain pending.

Software integration update, 15 November 2026: the shared foundation and both Stage 2 workers are now integrated, with all four entry notebooks under scripts/stage2. The combined 78-test suite passed; read-only actual-data compilation confirmed the unchanged fingerprint, 92 chunks and exact three-condition population, excluding CONFIRM. Colab setup/import/dependency and development-branch defaults were corrected without changing scientific configurations, shared APIs or Stage 1 adapters. See docs/stage2_runbook.md and reports/stage2_integration.md. Actual mounted plan/receipt binding, first-FIT L4 checks, E03 production outputs, full feature releases and suite acceptance remain pending. Older planning-only statements below describe the earlier handoff, not a completed runtime stage.

Updated 15 November 2026 during coordinator integration.

- Data branch 2f42b4a and model branch 2f374d2 are integrated into this phase1 snapshot with their histories preserved. Main remains unchanged.
- Original data independently revalidated locally: 3,600 images, 14,400 selected four-language records, 14 primary categories and one preserved unknown cell. No scientific results were produced.
- Shared stage1.v1 public APIs remain compatible; source-cache/storage helpers gained streaming, completion-verified publication without hard links. Cache corruption/partial-publication checks passed locally.
- Latest implemented CPU suite: 40 tests passed after the Qwen caption-alignment correction; 16 actual pinned-tokenizer checks passed on the selected four-language smoke rows. Those checks are not actual local GPU inference.
- Selected L4 pair: Qwen2.5-VL-3B-Instruct and Gemma 3 4B IT, dense bfloat16, batch one. Immutable revisions and direct runtime pins are recorded. Authorized metadata resolution passed for both models and actual final smoke passes were supplied; this does not establish bulk-extraction memory/time for every eligible input. Queue models on one physical L4; distinct L4 runtimes may run independently.
- The user supplied passed CPU preparation/validation logs under `/content/drive/MyDrive/phase1` and passed generated runtime configurations for both models. A subsequent Gemma model-validation log reported passed with no blockers. These are user-supplied runtime results; this chat has not independently inspected all payloads or certified every Drive sync/remount guarantee.
- Both final smoke outputs were subsequently supplied and explicitly identified by the user as run_validity_smoke passes, with no blockers. The separate post-fix Qwen model-validation artifact pointer remains to be located/verified, as do the complete readiness receipts. Do not request reruns merely for a missing pointer or documentation-only update. Stage 1 overall artifact acceptance and full G0 remain distinct pending gates. Latest paths and authentication/provenance lessons are in AGENTS.md.
- Stage 2 is now fully specified for implementation in docs/stage2_contracts.md and docs/stage2_ownership.md, at the user's request before remaining Stage 1 checks finish. No Stage 2 code or run is claimed. Stages 3-5 remain overview only.

The design-review extraction-condition/split and cache-key gaps are addressed by the additive extraction.v2 request interface and layered execution/feature/readout/analysis identities; see docs/stage1_contracts.md. Legacy smoke calls stay FIT-only. The validator now requires repeatable caption-only and single-per-image visual extraction as well as image-plus-caption, and records actual backend/site identities when executed successfully. No new model outcome was inspected to choose these rules. No scientific question, split, data identity or confirmed model/GPU pass is changed by this implementation.


Notebook entry points (user-requested conversion): `scripts/stage1/stage1_cpu.ipynb` runs preparation then validation; `configure_l4.ipynb`, `validate_model.ipynb` and `run_validity_smoke.ipynb` run in that order per MODEL. The five former scripts/stage1 Python entry files are removed. Their shared implementations now live in src/stage1; scientific rules, artifacts and completion gates are unchanged. Notebook syntax and CPU fixture checks do not establish actual Colab/L4/Drive completion.
