# Stage 1 shared interfaces — stage1.v1

Frozen for the initial data/model worker branches on 15 November 2026. Read alongside AGENTS.md, plan.md and prelim.md. This freezes implementation interfaces, not unchosen checkpoint identities or later-stage tensor formats. Coordinator owns changes to this contract and src/common/contracts.py/storage.py.

## Data worker API and artifact

Implement `src.common.data.load_dataset(manifest_path: Path, phase1_root: Path, protocol: Mapping) -> DatasetBundle`. `manifest_path` points to the dataset artifact's `_COMPLETE.json` file under PHASE1_ROOT. Use storage.verify on its parent, requiring status passed and matching protocol fingerprint. Verify payload checksums before reading records; do not trust a filename as evidence of compatibility. Source image existence/checksums and source provenance belong in data validation. Loading metadata for CONFIRM is allowed; passing those records to a preliminary model is not.

The completed dataset artifact contains:

- `records.jsonl`: one draw-0 record per image/language, with exactly the ModelInput field names in contracts.py. Four languages per common image. `image_path` is a portable POSIX path relative to PHASE1_ROOT, never a machine-specific absolute path. Preserve the original text; no Unicode normalization or translation.
- `targets.jsonl`: one record per image, fields image_id, split, values. Values follow configs/protocol.json category_order, length 25, each 0/1/null. Derive known_mask from null; there is no separate contradictory mask field. Invalid released values map to null with logged provenance, including the existing plant=4 cell.
- `dataset.json`: schema_version, category_order, dataset_fingerprint, protocol_fingerprint, source revisions/hashes, split manifest hash, caption-selection recipe and selected record/target file hashes.
- `validation.json`: status, dataset_fingerprint, protocol_fingerprint, checks and counts/exclusions. Passed requires actual data checks, not just schema conformance.

The DatasetBundle returned by the loader contains ModelInput objects, image-ID-keyed TargetRecord objects, category order, dataset/protocol fingerprints and the validation report. It must pass DatasetBundle.validate(protocol). The loader also performs the actual source/split/support checks assigned in plan.md; this common shape validator is not a substitute for those checks.

Dataset fingerprint: hash the canonical JSON identity containing source revisions/checksums, category_order, split-manifest hash, caption recipe, records.jsonl SHA256 and targets.jsonl SHA256. Exclude dataset_fingerprint itself and the validation/completion reports to avoid circular hashing. Protocol fingerprint is fingerprint(the entire resolved protocol JSON); record any versioned replacement protocol explicitly. Original audit labels/splits remain unchanged. If incomplete real data needs a replacement split, report the proposed deterministic version and changed support to the coordinator before emitting a passed artifact under a new protocol; never silently claim the 3,600-image counts.

Caption selection is now fixed by select_caption: within one image/language/split, select the minimum SHA256 of canonical JSON `["caption-draw-v1", image_id, language, draw_key, caption_id]`, lexical caption-ID tie break. Draw 0 is the reference. Source caption IDs must be unique, stable and derived from the pinned source release; persist their source mapping. Reordering candidates must not change the draw. No text-based semantic preference is used. Original captions can be stored separately as the four language views in plan.md; records.jsonl is the common selected-view reader contract.

## Model worker API

Implement `src.models.registry.create_adapter(model_key: str, config: Mapping, phase1_root: Path) -> ModelAdapter`. The coordinator resolves configs/models.json with resolve_model first. The adapter receives ModelInput objects only, never TargetRecord or a label-containing dataset bundle. Load image pixels from phase1_root/image_path; do not serialize paths, IDs, splits or other metadata into the model's chat input. Only the unchanged caption, image and declared common template/prefix belong there.

Adapter methods:

1. `describe()` returns JSON-compatible decoder_blocks, model_fingerprint, hook_convention, supported_sites (FeatureSite.key strings), unsupported-site reasons, repeat_atol and repeat_rtol. Tolerances must be explicit finite numbers in [0,1), justified for the recorded backend/precision. Use decoder_depths to resolve 1-based block outputs. No claim of normalized/unnormalized equivalence.
2. `inspect(inputs)` returns `checks` with every INSPECTION_CHECKS name mapped to a real boolean, plus `records` in input order, each containing row_key `[image_id, language, caption_id]`, actual token-span/mask and visibility evidence. Record actual native serialization, caption-token boundaries and relevant architecture visibility; unsupported E04 sites have reasons rather than fabricated checks.
3. `extract(inputs, sites)` returns FeatureBatch. row_keys exactly match input order. vectors maps every requested FeatureSite.key to an N-by-D finite CPU numeric matrix/list. D can differ by model/site, but must stay consistent for a given site across batches and repeats. No gold answers or generation are involved. The caller bounds batch size; the adapter may use a smaller internal batch as needed without changing row ordering.

Implement one validate_model.py CLI using --model qwen/gemma, the shared registry and PHASE1_ROOT. Exact checkpoint, processor and tokenizer revisions must be immutable 40-character commit IDs; template hash, precision, image policy, max-input policy and actual model software versions are mandatory. `quantization: null` explicitly means no quantization, not an unknown setting. Model fingerprint hashes every resolved model field except config_fingerprint. If one is declared, it must match; do not change shared fingerprint semantics. Device/backend choices affecting outputs must be recorded in the resolved configuration or provenance and cannot silently reuse incompatible caches.

Null checkpoint fields remain unresolved. No worker should choose a checkpoint by guessing what the user intended. Implement loading/configuration support and report any still-required checkpoint choice or gated-model access.

## Storage and run manifests

PHASE1_ROOT has no implicit fallback. In Colab, after Drive mounting, it is normally `/content/drive/MyDrive/phase1`. Windows uses the actual verified synced folder. Development checks may use an explicit `.local/phase1` with storage_scope local-development. A successful local file check is not proof of remote Drive durability.

Use storage.publish(root, namespace, manifest, files) for a unique artifact directory. The run manifest must contain kind, status, run_id, code_commit, protocol_fingerprint, dataset_fingerprint, model_fingerprint, seeds, software_versions and upstream_artifacts. Model-independent artifacts use model_fingerprint null. Unknown identifiers may be null in blocked diagnostic reports but not invented as selected real configurations. Payload files use safe relative paths and bytes. `_COMPLETE.json` is written last and records each payload SHA256. Never append multiple producers to one artifact. Run IDs should use new_run_id(), or caller-supplied unique IDs.

Consumers call storage.verify and recheck payload hashes after synchronization. Partial folders, corrupt/missing payloads, wrong fingerprints and blocked reports never count as passed inputs. Filesystem exclusivity is a local collision check, not a distributed Drive locking guarantee: unique producer/run/shard ownership remains mandatory.

The smoke CLI consumes real validated data and adapters, uses FIT-only images with image-group inner train/validation division, checks extraction/repeats and a tiny pooled-view optimizer fit. Its toy fitting is software plumbing, not E01 transfer or G9 calibration. Three image-vector permutation wiring checks do not complete full Stage 3 G0 controls. Exit codes: 0 smoke passed, 2 blocked, 1 failed. Overall Stage 1 readiness requires the integration review as well as passed real data/model/smoke reports.

Detailed Stage 2 cache tensor formats, E01 numerical implementation and later-stage prediction serialization remain future planning. Do not expand them independently within Stage 1.

## Condition-aware extraction and reuse addendum — 15 November 2026

The data wire schema stays stage1.v1; its source hashes, dataset/protocol fingerprints, splits and targets do not change. The additive `ExtractionRequest` schema is extraction.v2. `extract(inputs, sites)` remains a FIT-only image_caption compatibility wrapper. New callers use `adapter.extract_request(request)` and `FeatureBatch.validate_request(request)`.

- `purpose=stage1_smoke` permits FIT only. `purpose=preliminary_features` permits FIT/DEV-S/DEV-C/SCREEN. CONFIRM and unknown purposes are rejected before image IO, rendering, tokenization or cache consumption. This is extraction permission, never permission to fit/select using SCREEN; the existing role-specific fitting rules remain mandatory.
- `condition=image_caption` permits caption decoder and exposed visual sites. `caption_only` permits caption decoder sites with no image loading. `image_only` permits encoder/projector visual_mean sites, no caption text/prefix and no decoder caption pooling. Use one source record per image as provenance, not four language observations. FeatureBatch.row_keys retain the selected source record; view_ids are the semantic feature identity and exclude caption/language for image-only. Inconsistent image hashes/splits and duplicate semantic views are errors.
- Supported caption sites are last_caption and caption_mean at the four deduplicated depths. post_caption and decoder visual-token pooling remain explicitly unvalidated extensions, not required fabricated outputs. At least one exposed encoder/projector reference plus both caption poolings at each depth is required for the scheduled minimum. Physical module paths and output/pooling conventions remain model-specific.
- validate_model.py now checks both caption conditions and the single image-only reference, repeated extraction, finite dimensions and whether exposed visual vectors agree across the original caption contexts. If that invariance fails, audit the site before enabling reuse; never silently duplicate/relabel one vector as four independent language observations. Actual checkpoint validation is still required.

`src.common.cache_keys` separates reuse identities from complete run provenance:

1. **Execution:** pinned weights/tokenizer/processor/template, precision, input policy, actual numerical backend (device type/name/capability, library build/flags, batch=1), inference implementation hashes. Unclassified configuration fields fail closed. Repeat acceptance tolerances and scikit-learn version do not define raw inference values.
2. **Raw feature:** execution fingerprint + input-view fingerprint + physical hook/pooling/output specification + stored float32 convention. Adding a readout C value, threshold, metric or bootstrap never changes this key. Changing weights, input text/image, processor, backend or relevant extraction semantics does. Image-only keys exclude language/caption; other conditions remain distinct. Inputs are serialized without split or category labels.
3. **Readout:** ordered feature references + exact training/selection image IDs, target values/order, known masks/split hashes + scaler/fitting/selection recipe, seeds and fitting software. Any label/mask/partition or fitted-system change changes this identity. Calibration is a separately named downstream fit, with its DEV-C inputs recorded.
4. **Analysis:** immutable prediction artifact references + metric/threshold/calibration-policy/bootstrap specification and software. E19/E25 can reference the same completed E01 predictions.

Full model/protocol/data fingerprints, source commit, split/mask manifests and upstream completion checks remain required provenance. A matching mathematical feature key cannot bypass split restrictions, upstream verification or an incompatible cohort/condition. The helpers define interfaces; bulk feature/cache producers and tensor/shard formats remain Stage 2 work. No existing feature cache is migrated or declared compatible automatically. Publish new complete artifacts without mutating passed inputs.

`adapter.describe_execution()` records actual backend identity and per-site physical specifications in successful model-validation reports. Code fixtures test these contracts; they do not establish actual model or Drive readiness.


## Integration addendum — 15 November 2026

The frozen public data/model APIs remain compatible. storage.publish now additionally accepts a source Path as a payload value and streams it; byte payload behavior is unchanged. Verification also hashes files incrementally. Source downloads use unique completion-verified content-addressed cache directories without hard links or rename requirements. This is an additive helper extension, not a dataset/feature schema change.

The selected L4 candidate/runtime profile and instructions are in stage1_l4.md. Actual GPU inference and Drive-mount durability remain unverified until execution in the user's runtime. Generated runtime configuration uses a passed real dataset artifact for provenance, so its completion report does not fabricate a dataset fingerprint. Runtime-configuration readiness is distinct from model/Stage 1 readiness.


Notebook entry points (user-requested conversion): `scripts/stage1/stage1_cpu.ipynb` runs preparation then validation; `configure_l4.ipynb`, `validate_model.ipynb` and `run_validity_smoke.ipynb` run in that order per MODEL. The five former scripts/stage1 Python entry files are removed. Their shared implementations now live in src/stage1; scientific rules, artifacts and completion gates are unchanged. Notebook syntax and CPU fixture checks do not establish actual Colab/L4/Drive completion.
