# Stage 2 extraction worker

Implemented from exactly `986f812ac6e3811a035391738e295bf55043369a` on
`phase1-stage2-features`, in an isolated clone with LF source bytes. Only this
document, `src/stage2/extract_features.py`, `scripts/stage2/extract_features.ipynb`
and `tests/test_stage2_extraction.py` belong to this change. Coordinator interfaces,
adapters, protocol/model settings and scientific documents are unchanged.

## Entry point and prerequisites

Use the single MODEL-variable notebook, on one L4 with one model at a time. Run
the coordinator preparation notebook against the actual mounted PHASE1_ROOT
first. Supply its model-plan completion path and MODEL=qwen or gemma. The worker
resolves dataset/config/readiness paths and physical sites/dimensions from the
verified plan. No historical run IDs or architecture dimensions are defaults.
Optional RESUME_INDEXES contains explicit immutable progress completion paths.

The implementation accepts the equivalent command:

```sh
python -m src.stage2.extract_features --model qwen \
  --phase1-root /content/drive/MyDrive/phase1 \
  --plan-manifest PATH_PRINTED_BY_PREPARE_STAGE2 --preflight
```

Remove `--preflight` to execute. Repeat `--resume-index EXPLICIT_COMPLETE_PATH` to
join checkpoints from interrupted attempts. Paths may be absolute or portable
root-relative manifest paths as supported by the shared reader. The default
storage scope is persistent-drive; local fixtures use explicit local-development.
The root must already exist. A clean checkout is mandatory for artifact production.
CLI exit codes are 0 passed, 2 contract/storage blocked, 1 runtime failed.

Preflight fully verifies upstream plans/readiness/data/configuration and input
joins before loading weights. It does not request protected model metadata.
Each notebook launch repeats preflight in a new process and rejects changed/dirty
code, so settings reruns cannot launch with stale kernel acceptance flags.
Gemma uses environment HF_TOKEN, then Colab Secrets, then a hidden prompt; the
token is inherited by the child process. An existing environment token wins and
must be explicitly replaced if stale. Tokens are redacted from worker errors and
streamed notebook output. Authentication errors do not print exception details.

## Extraction and evidence

The worker sends only ModelInput records to the existing registry adapter.
Targets/masks/category order remain upstream references. It joins exact selected
caption IDs/text hashes, image IDs/checksums and frozen partitions. CONFIRM is
filtered before request construction, model input IO or resume consumption. Each
eligible input must match the plan, including all en/hi/bn/te contexts and draw 0;
missing inputs or substitutions fail rather than reduce coverage.

For each independent image it processes one English-source-pointer image_only
view, four image_caption views and four caption_only views. Every request has
one input and purpose=preliminary_features. Both caption poolings and all fixed
depths share the same forward. Combined visual hooks run in that same forward.
Workers never call private model methods, perform bulk attention inspection or
reimplement pooling. Stored vectors are unchanged raw little-endian float32, after
the adapter's validated arithmetic. There is no experiment-specific extraction,
normalization, fitting, scoring or output-based site selection.

Before moving beyond the first FIT image, all nine views use the shared exact
bridge/public comparison. These results become the production rows. All other
inputs use extract_with_evidence. Evidence records retain current processed spans,
masks, alignment and hashes and link the original accepted Stage 1 visibility
reference, explicitly labelled as inherited rather than new attention measurement.

Each combined image/site/language is immediately checked against its explicit
image_only vector. audits.json stores equality, maximum absolute difference and
both vector digests, plus completed image-only artifact references. Each condition
chunk repeats its distinct first/last local rows at every requested site. The shared
writer independently checks all rows/sites/dtypes/dimensions/audits and publication
hashes. Repeat and public-path full vectors are persisted as required by the frozen
wire format. Optional post_caption and decoder visual-token poolings stay unsupported.

At most one 32-image chunk of CPU feature/evidence buffers is accumulated. The
final smaller chunk has no padding. Each condition artifact has a unique attempt
namespace. Image-only completes before combined visual references publish. A
separate-process storage probe precedes production; each chunk is verified from
its persistent path after publication. This is filesystem readback evidence, not
a guarantee that Drive has remotely synced under every failure mode.

After the first three FIT condition chunks, the shared resource reporter derives
actual dimensions, token ranges and bytes and adds measured timings, GPU peak
allocated/reserved memory and free space when available. It projects exact
remaining work and storage including measured overhead plus 25 percent reserve.
The report labels the initial public comparison overhead, separate boundary repeats
and the limitations of Drive filesystem capacity information. No runtime or hidden
dimension is inferred from a model-family constant.

## Immutable restart and releases

After each condition publication the worker publishes and prints a unique
stage2_extraction_progress `_COMPLETE.json` under results/stage2-validation. Its
progress.json schema is stage2.extraction-progress.v1 and contains the exact model
plan reference, accepted chunk references, release references, resource reference,
first-chunk measurements bound to their chunk artifacts, and rejected-attempt issues.
This index is a resumable producer inventory, never a consumer release or a claim
that Stage 2 is complete. Preserve the printed manifest path after interruption.

A fresh process fully loads the plan and checks every explicit index, upstream
reference, chunk hash, ordered row/site inventory, token evidence, repeat/public
audits and visual reference. It skips only verified complete compatible chunks.
It does not search directories or choose by modification time. Identical duplicate
vectors are retained and ordered by manifest hash; differing numerical content
blocks before model loading. Corrupt child attempts are recorded and excluded,
then affected units are reextracted under fresh paths. Prior attempts are untouched.
If a corrupt upstream artifact invalidates dependent chunks, those dependents also
fail shared verification and cannot be reused.

First-chunk per-condition timing measurements are saved with every progress index.
This permits recovery after only image-only has published, or after all three first
chunks have published but before the resource report. Accepted repeats are not
performed again. A wholly completed resume loads no model and performs no forwards;
it still verifies storage and publishes new immutable release indexes.

At each partition boundary the shared publisher verifies exact complete coverage
and publishes three condition releases. FIT/DEV-S can be handed off immediately
without waiting for DEV-C/SCREEN. The final worker JSON lists 12 release references
and the resource report for one actual production model. It always reports
stage2_complete=false and scientific_results=false. Coordinator suite validation,
the other model, E03, later full G0, fitting and scoring remain separate work.

Over-limit inputs, malformed spans, nonfinite/wrong-dimension vectors, visual/repeat
mismatches, invalid images, OOM and model failures stop the job. Once launch has
resolved a plan, a separate nonpassed stage2_extraction_diagnostic records safe
error text and exact operation/image/source-row/condition/sites as available, plus
the latest resume index. Prior complete chunks/releases remain intact. There is
no automatic retry with changed precision, resizing, attention or captions.

## Local verification and remaining runtime gates

The local CPU runtime is Python 3.12.14, NumPy 2.1.3, SciPy 1.18.1,
scikit-learn 1.6.1 and Pillow 11.3.0. The 12 extraction tests use synthetic
HFAdapter subclasses and isolated upstream acceptance, with real shared storage,
checksums, arrays, audit validation, resource publication and release/reload code.
They exercise all conditions, fixed shared-site routing, batch one, a 32-image
boundary/one-image tail, 12 partition releases, full resume without model loading,
partial publication/resource crashes, corrupt-child replacement, duplicate-value
conflicts, input/CONFIRM guards, visual/repeat/runtime failures, secret redaction,
and notebook syntax/empty outputs. They do not certify either real checkpoint.

Run all checks from repository root using the selected CPU interpreter:

```powershell
$env:TEMP='C:\stage2-tmp'
$env:TMP=$env:TEMP
python -m unittest discover -s tests -v
```

The short explicit temp root avoids existing Stage 1 Windows path-length failures.
It is test scratch, not an authoritative artifact-root fallback. No packages or
selected inference settings were changed to obtain a test pass.

No actual L4 inference, mounted dataset/model plan, Gemma authentication or
persistent Drive Stage 2 artifact was available in this execution environment.
PyTorch and NVIDIA tooling are absent locally. Fixture artifacts are temporary and
not production handoff references. The historical Stage 1 evidence remains at its
original provenance; no expensive rerun is requested merely because this worker
was added. Actual per-model preflight, runtime equality, first-FIT public/repeat/
visual audits, resource report, mounted readback and complete releases still need
to execute on the authorized L4 runtime. No shared interface change is proposed.
