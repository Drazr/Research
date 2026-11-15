# E03 CPU preprocessing handoff

Worker base: `986f812ac6e3811a035391738e295bf55043369a`.
Branch: `phase1-stage2-e03`. This work implements E03 preprocessing only.
The coordinator's shared interfaces and all scientific documents are unchanged.

## Entry point and execution

Use `scripts/stage2/prepare_e03.ipynb` from a clean published implementation
checkout. Mount the intended Drive account, set PHASE1_ROOT and supply the
completed shared data plan from `prepare_stage2.ipynb`. The plan resolves the
accepted dataset and all recipes/rows. No MODEL, GPU, weights, model readiness
receipt or Hugging Face access is needed. Install the existing
`requirements-stage2-cpu.txt` in a CPU environment when needed.

The notebook defaults to stopping after DEV-S, providing all 13 families' FIT
and DEV-S releases promptly. To complete DEV-C/SCREEN, supply the printed
checkpoint as RESUME_INDEX and set THROUGH_SPLIT to SCREEN. Each invocation
verifies current settings in a fresh process. Do not pull during a run.

Equivalent commands, with explicit real mounted paths:

```sh
python -m src.stage2.prepare_e03 preflight --phase1-root ROOT --manifest SHARED_PLAN
python -m src.stage2.prepare_e03 run --phase1-root ROOT --manifest SHARED_PLAN --through-split DEV-S
python -m src.stage2.prepare_e03 run --phase1-root ROOT --manifest SHARED_PLAN --resume-index CHECKPOINT --through-split SCREEN
python -m src.stage2.prepare_e03 verify-release --phase1-root ROOT --manifest FAMILY --split FIT
```

Use `--storage-scope local-development` for explicit local fixture roots. The
default is persistent-drive; that label does not prove remote synchronization.
The shared storage probe performs an actual independent-process mounted-path
readback. Missing/malformed data, incompatible identities, empty vocabularies,
version drift or readback failures stop the affected run with exit code 2.

## Fixed preprocessing and identities

`e03_text.py` contains all analyzer, fitting, arithmetic and serialization logic;
its source hash is part of the shared preprocessing identity. Python/Unicode,
NumPy/SciPy/sklearn versions are recorded before fitting and must match on resume
and transformation. NumPy 2.1.3 and sklearn 1.6.1 are enforced by the shared
profile. There is no library-default vectorizer or semantic tokenizer.

- Four word families use exact case-sensitive str.split tokens, unigram/bigram
  tuples, min_df 2 and a 20,000 cap. Attached punctuation/combining marks survive.
- Four character families and one pooled character family use literal contiguous
  code-point substrings of lengths 3–5, min_df 2 and a 50,000 cap. Original
  whitespace is included without collapsing, padding or normalization.
- Vocabulary inclusion uses FIT caption-document df. Cap selection ranks raw FIT
  corpus count descending with Unicode lexical ties; final columns are lexical
  tuples/strings. Word JSON stores n and tokens without a delimiter encoding.
- TF is 1+ln(count), IDF is ln((1+N)/(1+df))+1; retained entries are row L2
  normalized in float64 and cast to float32 only afterward. OOV/zero rows survive.
- Four length families store log1p(code-point count), log1p(str.split token count)
  in that order, calculated in float64 and stored in float32, without fitted
  parameters, scaling, PCA or heads.

FIT record-ID and text hashes are recorded separately in checkpoints (and in
TF-IDF preprocessor JSON), in addition to the shared ordered FIT-row hash.
Transformation row manifests retain original caption hashes, selected source
keys, partitions and grouping. Pooled rows carry image_loss_weight 0.25; each
per-language row carries 1.0 for future supervised fitting. No labels enter a
feature matrix. Only FIT captions fit vocabulary/IDF; other eligible partitions
use the frozen preprocessor. CONFIRM is filtered using metadata before any text
is read, and requests for it are rejected.

The shared publishers store little-endian float64 IDF and float32 values, dense
length NPY or CSR data/indices/indptr NPY with int32 indices, strict sorted unique
columns and no explicit sparse zeros. Exact rows.jsonl accompanies each split.
Before publication, bounds, dimensions and numerical invariants are checked by
the shared layer. After each family/split publication, a separate Python process
reconstructs every value from serialized FIT state and compares exact arrays,
including sparse indices/indptr. It does not refit.

## Checkpoints, resume and integration

The worker emits JSON events as work completes. `release_passed` supplies the
family reference, split, dimensions and zero-row count. `checkpoint` supplies an
immutable `_COMPLETE.json` plus SHA256. Its `inventory.json` has schema
`stage2.e03-checkpoint.v1`, shared-plan reference, exact preprocessing versions,
and `families` keyed by full recipe fingerprint. Each entry contains the fitted
preprocessor reference, separate FIT hashes, split-release references and latest
family-index reference when available. Nothing is discovered by directory age.

Checkpoints live under `cache/shared/stage2-e03-checkpoints/<unique-run>/`;
preprocessors/releases/family indexes use the coordinator's
`cache/shared/stage2-text/` namespaces. Every checkpoint is newly published.
Resume validates marker hashes, plan/recipe/runtime identity and completed child
arrays through the public role-restricted reader, reuses all accepted FIT state
and splits, and writes new indexes for extensions. A failed child reload leaves
its completed publication unaccepted by the checkpoint. Preserve the last
successful checkpoint event; interrupted orphan attempts are never auto-selected.

The independent storage probe always runs before new production writes. Checkpoint
`complete` means all 13 E03 families have all four eligible splits, not whole-stage
acceptance. The coordinator can take each entry's `family` reference into its
suite inventory. Stage 3 consumers use the shared load_text_family API with exact
plan_fingerprint/preprocessing_fingerprint and source_fit/source_select roles;
DEV-C/SCREEN remain separately permissioned. Both VLM comparisons reuse the same
E03 references. Full G0, heads, model selection, calibration and scores are later
work.

## Validation and runtime evidence

Local CPU runtime: Python 3.12.14, Unicode 15.0.0, NumPy 2.1.3, SciPy 1.18.1,
scikit-learn 1.6.1 and Pillow 11.3.0. The worker records actual installed values
instead of copying these observations into future identities. Producer provenance
also records Pillow, joblib (locally 1.6.0) and threadpoolctl (locally 3.8.0).

Run `python -m unittest discover -s tests -v`. On this Windows machine the existing
Stage 1 path-length fixtures need TEMP and TMP set to the shorter
`C:\stage2-tmp` root. E03 tests cover exact Unicode/literal
whitespace behavior, hand-computable TF-IDF, document/corpus counts, cap ties and
column order, FIT/DEV separation, CONFIRM text-access sentinels, language guards,
zero/OOV rows, length arithmetic, unambiguous term tuples, actual fresh-process
sparse/dense numeric reload, and all 13 shared publication/partial-handoff/resume
paths with corruption and reader-role rejection.

Final local full regression run: 65 tests passed (57 existing/shared and eight
E03 tests). Notebook cells compile and contain no saved outputs. Importing the
E03 worker does not import Torch. These are CPU software checks.

The full publication fixture uses real shared storage/writers/readers but mocks
upstream dataset acceptance and runs the child family verifier in the fixture
process. The separate numeric reload test really launches another Python process;
the production worker launches the full family verifier in another process.
These fixtures do not certify real mounted-data acceptance or Drive durability.
Empty-string length/transform edge cases are numerical tests; the real dataset
contract itself requires nonblank captions.

No mounted accepted shared plan or complete dataset is available in this local
checkout. No real E03 preprocessing artifacts, real vocabulary dimensions or
real zero-row counts are claimed. Existing upstream evidence pointers remain in
the coordinator documents; they are not newly produced E03 results. Run the
notebook against the actual mounted plan for production artifacts, and return
the resulting checkpoint/family references for coordinator verification.

No shared interface changes are required by this implementation. Shared plan and
text readers revalidate upstream data on repeated publication/reload, so actual
mounted runtime may spend substantial time verifying images. This behavior is
preserved; coordinate any performance change with the interface owner rather
than bypassing verification. Whole-stage completion, model extraction and
scientific interpretation remain outside this worker's acceptance.
