# Coordinator foundation implementation — 15 November 2026

Status: shared software foundation implemented; extraction and E03 workers and
Stage 2 runtime completion remain pending. Scientific configurations, dataset/
protocol fingerprints, Stage 1 code/adapters, masks and partitions are unchanged.

The callable interfaces, precise payload fields and owned files are documented in
docs/stage2_interfaces.md. Both worker branches must begin at the exact coordinator
implementation SHA supplied after publication, not the finalized planning commit.

## Actual checks and historical evidence

The local Windows CPU checks use Python 3.12.14, NumPy 2.1.3, SciPy 1.18.1,
scikit-learn 1.6.1 and Pillow 11.3.0. Run from repository root:

```powershell
$env:TEMP='C:\Users\APOORV\Documents\Codex\s2tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The shorter explicit development temp root is needed for existing Stage 1 cache
fixtures on Windows. An initial run at the long generated checkout path hit three
WinError 206 path errors; all 40 existing tests passed after selecting the shorter
root. No Stage 1 behavior was changed to suppress those errors.

Shared tests exercise the actual frozen population inventory, CONFIRM exclusion,
same-forward routing/evidence, mask errors, immutable receipt/plan serialization,
historical source reuse, wrong-kind and changed-source rejection, worker settings,
canonical float32 conflicts, row/site/dtype/finite checks, visual-reference audits,
role-restricted reload/corruption, independent-process storage probes, text FIT
identity, sparse bounds/zero rows, partial family indexes and transform versions.
Synthetic model/data acceptance fixtures are explicitly isolated from actual
dataset/model validation. They do not certify checkpoint behavior or E03 analyzers.

Final checks: 57 tests (40 preserved Stage 1 regressions and 17 new shared
interface tests), plus notebook cell compilation/no-output checks and Python module
compilation. All passed. Git diff whitespace checks also passed. Production
L4 extraction and full mounted readiness receipt verification remain separate gates.

The Drazr GitHub connection resolved to Drazr. The intended Google Drive connection
was independently checked and returned Tapas Sahu / tps039@gmail.com. Its phase1
folder and both model namespaces were accessible. Folder fetch, rather than the
connector's search results, exposed the JSON payloads. Existing Qwen validation was
located at:

`smoke/280535919b08cad218a47706a293d398b3ee691200becc37d68eb9b6573ba16c/20261115T043216Z-65cc67f45eea4c98b731fc0db1d7cc43`

Original raw JSON bytes for both separate model validations and both final smoke
reports were inspected. Every payload SHA256 matched its original completion
manifest. All four payloads passed with no blockers at original commit
`d73133ce51247f10dd6bc7af35ee9fe9b59ed517`. The model-validation reports include
all three conditions, dimensions, actual execution identity and repeat/visual
checks; final smoke reports include optimizer/permutation checks. Details, full
hashes and original Drive links are in reports/stage2_existing_evidence.json.
Relevant adapter/spans/registry/contracts/cache-key and validation/smoke source
files were byte-compared with their original commit and remain identical.

Observed dimensions from those reports: Qwen decoder 2048, encoder 1280,
projector 2048; Gemma decoder 2560, encoder 1152, projector 2560. These are report
observations, not new inference or family constants. New plans resolve them from
the accepted evidence, not this document. No expensive model rerun is requested
merely to recover missing pointers or because coordinator-only code was added.

The common CPU preparation/validation and both runtime configurations retain the
user-reported passes recorded in AGENTS.md. This task did not remount Drive, read
every dataset image/source through Colab, verify remote sync/durability, launch an
L4, or create a mounted full readiness receipt. Model/smoke payload verification
does not manufacture those remaining checks. Preparation will automatically verify
the dataset, deterministic validation and generated configuration when run against
the actual mounted root. Stage 1 reruns still honor their exact-commit guard.

## Remaining work and handoff

- Run the preparation notebook against the intended mounted root; verify common
  data and bind existing compatible generated configs and both historical reports.
  A shared plan can be produced before either model binding succeeds.
- Extraction chat implements its worker only: plan-resolved settings, actual
  runtime comparison, first-FIT public/bridge equality in nine views, batch-one
  processing, all visual/repeat audits, measured first-chunk resources, immutable
  restart and 12 complete releases per model. On one L4 run models sequentially.
- E03 chat implements its worker only: unchanged native analyzers, exact fixed
  FIT-only vocabulary/cap/IDF semantics, 13 families, sparse/dense transforms and
  deterministic reload. Use the shared plan once, independent of model readiness.
- Coordinator integrates workers and verifies complete indexes plus mounted-path
  readback and all evidence before Stage 2 acceptance. Stage 3 heads/full G0 and
  Stage 4 scores remain outside this foundation.

No extraction/E03 worker module, scientific redesign, Stage 2 completion claim or
main-branch change is included. Shared interfaces stay fixed after publication;
missing capabilities must be reported for coordination before divergent changes.
