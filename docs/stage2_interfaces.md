# Stage 2 executable interface handoff

Interface version: `stage2.plan.v1`, `stage2.features.v1`, `stage2.text.v1`.
Implements `stage2-plan.v1`, review revision 2. The outer completion marker remains
`stage1.v1`. This is the coordinator foundation, not Stage 2 completion.

The immutable worker base is the commit that first adds this document and
`src/stage2/interfaces.py` to phase1. The coordinator supplies its full SHA after
publication. It can also be recovered without relying on a moving branch:

```sh
git log --diff-filter=A --format=%H -- docs/stage2_interfaces.md
```

Use that exact SHA for both task branches. Do not branch from the earlier planning
commit `ec1fb33`, the original Stage 1 foundation or a later moving phase1 head.
After publication, report interface changes to the coordinator before changing them.

## Ownership

| Owner | Files |
|---|---|
| Coordinator | `src/stage2/__init__.py`, `schema.py`, `plans.py`, `readiness.py`, `evidence.py`, `features.py`, `text.py`, `operations.py`, `interfaces.py`; `scripts/stage2/prepare_stage2.ipynb`, `validate_stage2.ipynb`; `tests/test_stage2_foundation.py`; `requirements-stage2-cpu.txt`; this document and integration/evidence reports; shared plan/AGENTS/ownership updates |
| Extraction chat | `src/stage2/extract_features.py`, feature-specific helper files with distinct names; `scripts/stage2/extract_features.ipynb`; `tests/test_stage2_extraction.py`; `docs/stage2_extraction.md` |
| E03 chat | `src/stage2/prepare_e03.py`, analyzer/text-specific helpers with distinct names; `scripts/stage2/prepare_e03.ipynb`; `tests/test_stage2_e03.py`; `docs/stage2_e03.md` |

`features.py` and `text.py` are coordinator serialization/reader modules, not worker
extraction or analyzer implementations. Workers must not replace them. Existing
common/contracts/cache/storage, model adapters, registry, scientific documents and
Stage 1 acceptance behavior are unchanged. No extraction/E03 worker is supplied here.

## Public imports and values

Import callable interfaces from `src.stage2.interfaces`. A verified plan is a
`VerifiedPlan` read-only mapping (nested lists become tuples); use
`src.stage2.schema.plain(value)` when JSON serialization is needed. Its added
`manifest_reference` contains `{path, sha256}` for the original completion marker,
relative to the resolved PHASE1_ROOT. All authoritative paths remain under that
root; no newest-file selection or implicit root is supported.
Retain the verified object during a worker process. Chunk publication rechecks its
immutable marker/payload hash without re-decoding every dataset image on every
flush; a fresh/restarted process must call load_job_plan and fully verify upstream
data again. Actual model inputs retain their adapter image/span checks.

`ContractError` means incompatible, incomplete or malformed input. Writers use
unique run/attempt components, reject an existing destination and return the
`Path` of `_COMPLETE.json`, never an unchecked directory. They validate before
publication and independently verify checksums after publication. No pickle or
shared append log is used. A failed publication attempt remains non-consumable.

Producer metadata contains `run_id`, actual 40-character `code_commit`, nonempty
`software_versions`, and `storage_scope` (`persistent-drive` or `local-development`).
`current_producer(run_id=None, scope=...)` obtains actual HEAD and installed
versions and refuses a dirty checkout. Workers also add actual inference versions
to their producer versions; a fixture string is never production provenance.

## Preparation, receipts and resolved settings

```python
publish_data_plan(root, protocol, dataset_manifest, data_validation, producer)
publish_readiness(root, protocol, model, dataset_manifest, data_validation,
                  configuration_manifest, model_validation, smoke, producer)
publish_model_plan(root, protocol, data_plan_manifest, receipt_manifest, producer)
load_job_plan(plan_manifest, root, protocol)
verify_readiness(receipt_manifest, root, protocol, model_config)
resolve_worker_settings(plan_manifest, root, protocol, model=None)
assert_runtime(adapter, plan)
```

`protocol` is the unchanged parsed `configs/protocol.json`. `model_config` is the
resolved selected entry, not the whole registry. The shared compiler accepts the
completed common data and separate passed deterministic data-validation artifact;
it does not need a model config/receipt. It preserves the frozen dataset's targets,
masks, category order, split and record hashes, all 25 targets and the fixed primary
cohort. It enumerates the ordered 92 independent 32-image chunks and all three
condition row tables, excluding CONFIRM before any preprocessing. The shared plan
contains 13 fixed recipe descriptions, without running an analyzer.

For `publish_data_plan`, `data_validation=None` searches only the declared protocol's
shared validation namespace for the matching typed passed dataset report and uses
manifest-hash order. Every selected report still undergoes full data validation.
Supplied explicit evidence is preferred. Preparation can reuse an existing shared
plan, preventing model binding from changing E03 identity. A failed model receipt
produces a separate blocked diagnostic while the shared plan remains usable.

For `publish_readiness`, `model_validation=None` or `smoke=None` searches only
`smoke/<full model fingerprint>/*/_COMPLETE.json`, checks typed passed payloads,
rejects conflicting execution/site identities, sorts agreeing candidates by
manifest hash and retains their references. It never substitutes final smoke for
the richer model-validation report. The receipt verifies data/config/model/smoke
payload kinds, hashes, dataset/model identities, real L4/backend/execution catalog,
fixed depths, dimensions, repeat/visual checks and relevant local source hashes.
It records all original commits. Source-identical historical evidence need not
share the producer's new HEAD. Original commits must be available locally for
source comparison (`git fetch` the actual missing history if necessary).
Use an LF checkout for byte-level source identity (`git -c core.autocrlf=false clone`
on Windows); CRLF conversion must not be mistaken for the original source bytes.

Existing Stage 1 notebooks keep their exact-commit configuration guard. If they
are rerun, satisfy it with a new lightweight configuration; never rewrite old
metadata. A changed adapter, spans, inference/contract source or execution policy
requires affected validation. `assert_runtime` compares the actual loaded adapter's
configuration and complete execution description before production. A source
receipt alone cannot certify a newly loaded runtime.

`resolve_worker_settings` returns `plan` and absolute `dataset_manifest`; for a
model plan it also returns `model_config` (generated models.json path),
`readiness_manifest` and `model`. Pass MODEL explicitly and require it to match.
The shared plan rejects MODEL. A fresh call fully verifies the plan, so settings
reruns cannot reuse a stale success flag. Consumers must not manually re-enter
configuration/evidence paths. Gemma authentication is still required separately in
each model-loading runtime, using the existing Stage 1 secret-safe flow. No secret
belongs in plan metadata, settings printouts or notebook outputs.

## Same-forward bridge and production sequence

```python
result = extract_with_evidence(adapter, request)
batch, evidence = result.features, result.evidence
compare_public_path(adapter, first_fit_request)
```

The result is `EvidenceBatch(features: FeatureBatch, evidence: tuple[dict,...])`.
Only the current HFAdapter compatibility boundary (including explicit test
subclasses) is supported. Each validated extraction.v2 request is guarded before
IO, uses `attention=False` and forwards its explicit condition/purpose. There is
one existing `_forward` invocation per input for all requested sites; vectors and
token evidence come from that invocation. Pooling dtype/arithmetic and final CPU
float32 cast remain exactly those of the existing adapter. Workers may not call
private adapter methods or reconstruct spans with an inspection forward.

Evidence retains actual row_key, condition, token_count, input_ids_sha256,
caption_alignment, caption_positions, last_caption, image_positions,
role_and_other_positions and processor_attention_mask. Native serialization is
hashed as UTF-8 into native_serialization_sha256; full serialized/decoded text is
discarded. All position lists are ordered, unique, disjoint and exhaustive; masks
must represent unpadded batch one and counts must respect the bound token limit.
image_only requires empty caption positions/null last_caption.

Worker order:

1. Resolve the model plan, authenticate if necessary, load one model once and call
   `assert_runtime` before inputs. Use batch one, purpose=preliminary_features.
2. Call `storage_probe(root, protocol, dataset_fingerprint, producer)` before bulk
   writes. It publishes uniquely, verifies in a fresh Python process and publishes
   a receipt containing the two distinct process IDs and payload marker hash.
   This proves access/readback through the supplied path; it makes no remote-sync
   guarantee. Pass `reference(probe, root)` as `metadata['storage_probe']` below.
3. For the first FIT image, call `compare_public_path` in all nine core views:
   one image_only, four image_caption and four caption_only. This returns the
   bridged batch after exact comparison to public `extract_request`. Capture both
   caption poolings at all fixed depths together; include visual audit hooks in
   the combined forward. Do not run attention inspection across bulk inputs.
4. For each image, run image_only from its English source pointer, then the four
   combined/caption-only views. Accumulate only one image chunk on CPU. Retain the
   once-per-image reference separately from combined-context audit outputs.
5. Repeat the first/last row of every condition chunk once; persist actual repeated
   vectors. A verified resumed chunk includes these audits and needs no new repeat.
6. After all three first FIT condition chunks complete, call
   `publish_resource_report(root, plan, first_chunk_manifests, measurements, producer)`.
   Pass a condition-to-completion-path mapping. This derives actual dimensions,
   token ranges and payload bytes from those verified chunks. Measurements supply
   condition_seconds_per_input, remaining_inputs (condition mappings),
   forward_seconds, audit_seconds, write_seconds, verification_seconds,
   gpu_peak_allocated_bytes, gpu_peak_reserved_bytes and free_space_bytes (null if
   unavailable). The projection includes measured overhead and 25 percent reserve.
   Subsequent chunks require its reference as metadata['resource_report'].
7. Publish complete condition/split release indexes. No worker consumes a directory
   or drops overlong/failed rows. Preserve completed compatible attempts on failure.

## Feature wire format and readers

```python
rows = planned_rows(plan, chunk_id, condition)
publish_feature_chunk(root, plan, chunk_id, condition, attempt_id,
                      payload_paths, metadata)
verify_feature_chunk(manifest, root, protocol, plan)
publish_release_index(root, plan, condition, split, chunk_manifests, run_id)
load_feature_release(index_manifest, root, protocol, role, expected)
release.read_matrix(site_key, language=None)
```

`planned_rows` supplies exact rows.jsonl dictionaries, including zero-based local
row_index, canonical source/semantic identities, caption hashes and per-site
feature keys. Write those dictionaries unchanged with schema.jsonl. image_only
semantic language/caption/draw are null; source_row_key retains its English pointer.
`plan['required_sites'][condition]` fixes sorted storage order; physical specs and
dimensions come from the actual accepted report, not family constants.

`payload_paths` maps exactly these portable filenames to staged Path objects:

- rows.jsonl, token_evidence.jsonl (all conditions, including image_only), audits.json.
- site_000.npy, site_001.npy, etc. in required_sites order: uncompressed little-endian
  float32 C-contiguous finite N-by-D matrices. Object dtype/pickle/NaN are rejected.
- For image_caption only: visual_000.npy, etc. in plan.audit_sites order, containing
  actual same-forward visual audit vectors for all four caption contexts. These are
  audit payloads, never four independent image-only feature matrices.

Each token record adds `visibility_reference=plain(plan['references']['readiness'])`
and the exact `visibility_scope='inherited Stage 1; no bulk attention measurement'`.
This links actual current spans to the accepted visibility check without claiming
new attention measurements. Every row requires evidence.

audits.json fields:

```json
{
  "repeated_vectors": {"0": {"SITE_KEY": [1.0, 2.0]}, "LAST_LOCAL_INDEX": {"SITE_KEY": [1.0, 2.0]}},
  "public_path_vectors": {"FIRST_FIT_IMAGE_LOCAL_INDEX": {"SITE_KEY": [1.0, 2.0]}},
  "visual_references": [{"artifact": {"path": ".../_COMPLETE.json", "sha256": "FULL_SHA256"}, "row_index": 0}]
}
```

Use actual full vectors at every required site, not pass booleans. repeated_vectors
keys are exactly the distinct first/last local indices. public_path_vectors is
required for every view of the first image in the first FIT chunk, not later rows.
For image_caption, visual_references is one entry per row, linking each context to
its explicit image_only completion and matching image/split/local row. For the
other conditions it is empty. The writer compares canonical float32 bytes, stores
vector digests in chunk.json and blocks mismatches. Tolerances remain exactly zero.

chunk.json is coordinator-generated; workers do not hand-author it. It binds full
plan/execution/model/source identities, immutable plan/probe/resource references,
site specs/dimensions, row-manifest hash and vector-content tables. The outer marker
contains producer commit/versions/seeds/storage scope and checksums. A generic
passed marker alone is never sufficient.

`verify_feature_chunk` returns `(chunk_metadata, rows, site_arrays, vector_digests)`
after complete validation. Use it for explicit resume candidates. Release publication
requires every planned chunk of exactly one condition/split. Identical float32
duplicates are retained as evidence and selected by manifest-hash order; conflicting
values block release even if filenames or container checksums differ.

`expected` must include full plan_fingerprint, condition and split; it may include
additional release fields. The reader verifies the plan, receipt, all child markers,
rows, keys, physical sites, dimensions, audits and hashes. It exposes immutable rows
and site_catalog. `read_matrix` returns `(ordered_rows, N-by-D float32 array)` and
rechecks the accepted index/children on each read. Language filtering preserves row
identities, never concatenates splits and rejects image_only language filters.

Reader roles are exact: source_fit=FIT, source_select=DEV-S, calibration=DEV-C,
exploratory_score=SCREEN. CONFIRM is absent. Producer extraction permission is not
permission to fit on SCREEN. Language/label permissions still follow the scientific
task; a FIT reader does not authorize target adaptation.

## Text serialization and partial family handoff

```python
versions = preprocessing_versions('src/stage2/WORKER_ANALYZER.py')
publish_text_preprocessor(root, shared_plan, recipe_id, payload_paths, metadata)
publish_text_release(root, preprocessor_manifest, split, payload_paths, metadata)
publish_text_family(root, recipe, split_payloads, run_id, metadata)
load_text_family(family_manifest, root, protocol, role, expected)
family.read_matrix(split, language=None)
```

Here `recipe_id` is the full recipe_fingerprint from the shared plan; **recipe in
publish_text_family is the completed FIT-preprocessor manifest Path**, not a new
recipe dictionary. `split_payloads` maps explicit eligible split names to completed
text-release manifest paths. This naming preserves the reserved public interface.
Publication of FIT and DEV-S need not wait for SCREEN. A new immutable family index
extends split coverage; `family_complete` becomes true only with all four splits.

Additional text producer metadata: analyzer_file (repository-relative worker
implementation), preprocessing_versions (actual Python/Unicode/NumPy/SciPy/sklearn
plus analyzer SHA256), storage_probe reference, and for split publication zero_rows.
Fitting/transform producers must match actual installed versions and the analyzer
source hash; all transformations match the preprocessor's exact recorded identity.
The CPU requirements profile pins NumPy 2.1.3/sklearn 1.6.1 and the locally checked
SciPy 1.18.1; runtime package/Unicode versions remain recorded, not guessed.

Use `text_rows(shared_plan, recipe, split)` unchanged for rows.jsonl. It preserves
caption hash/source/view IDs and adds image_loss_weight=1 for per-language families
or 0.25 for the pooled four-view baseline. No labels or learned heads are serialized.

Preprocessor payload is preprocessor.json and, for TF-IDF, idf.npy. All use
fit_rows_sha256=fingerprint(text_rows(plan, recipe, 'FIT')) and fit_split='FIT'.
Length JSON has exactly those two keys plus columns (from recipe) and
fitted_parameters=false. TF-IDF JSON additionally contains terms, term_order_sha256
(fingerprint of terms), document_frequency, corpus_term_count, n_fit_documents.
Word terms are `{"n":2,"tokens":["first","second"]}` objects, never delimiter-joined;
character terms are literal strings. Terms are unique and lexicographically ordered.
IDF is finite little-endian float64, with the exact specified smoothed FIT formula.
Empty/over-cap vocabularies and wrong FIT identities are rejected. The worker owns
analyzer/count/cap-tie correctness and must supply its independent recipe tests;
the shared layer validates supplied serialization and identities, not a second E03
fitting implementation.

Split payloads: rows.jsonl plus matrix.npy for length; otherwise rows.jsonl,
data.npy (<f4), indices.npy (<i4), indptr.npy (<i4) for portable CSR. Shape is derived
from the plan and fitted dimension. Indices must be strictly sorted unique within
rows and in range; indptr/nnz bounds, positive finite values and normalized nonzero
rows are checked. Empty sparse rows remain explicit and zero_rows is verified.
No dense conversion or pickle is needed. The family reader returns sparse CSR or
dense values plus exact rows, enforces its fixed role before reading a split, and
requires expected plan_fingerprint and preprocessing_fingerprint. It never merges
partitions or equates different language vocabularies.

## Notebooks, integration and remaining gates

Run prepare_stage2.ipynb from a clean published checkout with PHASE1_ROOT mounted.
The known dataset suffix is prefilled and verified; optional bindings are supplied
once. Default empty bindings compile only shared CPU metadata. Reuse the printed
SHARED_PLAN when adding either model. An explicit generated configuration marker is
required for a model binding; optional report pointers trigger typed discovery.
Setup/execution use separate processes and do not load weights or install GPU code.

validate_stage2.ipynb takes an explicit inventory JSON: data_plan, model_plans,
feature_releases (12 per model), resource_reports (one per model), text_families
(13 complete). validate_suite verifies them and publishes a blocked diagnostic for
missing coverage or a passed stage2_suite_index only when everything is accepted.

Integration correction: the suite payload's feature_releases list contains only
feature release references (24 in a complete two-model suite). Resource-report
references are retained separately in resource_reports, keyed by qwen/gemma; both
types remain explicit upstream artifacts. Do not feed a resource receipt into a
feature reader based on its position in a mixed list. The new integration
regression checks complete inventory orchestration and missing-input blocking.

This foundation has no such completed suite. Full G0 and scientific interpretation
remain later gates. Workers return base/head SHAs, owned files, exact versions,
tests, actual runtime commands/evidence, completion paths/hashes and remaining gaps.

The workers are now integrated on phase1, with coordinator notebook setup fixes.
Use docs/stage2_runbook.md and reports/stage2_integration.md for current notebook
entry instructions and available checks. The original immutable worker base and
public callable signatures remain those documented above.
