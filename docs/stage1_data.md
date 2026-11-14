# S1-A / S1-B data implementation

Base: coordinator frozen-interface commit ea06d93efe8edcb849985cd9f2718f8ffd271e0d.
Only the data reader, preparation/validation CLIs, data tests, dependency pin and these notes are owned here.

Install `python -m pip install -r requirements-data.txt`. Set PHASE1_ROOT explicitly to the mounted Drive phase1 folder, or an explicitly labeled local development directory. Preparation fetches the original Google caption ZIP at revision ad44c380c86cb1240666361caabaf5684507d0ef, the released image archive, attribution CSV and the existing pinned annotation blob; SHA256 checks are mandatory. Source IDs retain zero-based released caption indices. Duplicate texts are preserved and reported, with no semantic review.

Run from the checkout, with the actual code commit:

```powershell
python scripts/stage1/prepare_dataset.py --code-commit <commit> --storage-scope local-development
python scripts/stage1/validate_data.py --code-commit <commit> --dataset-manifest <printed-_COMPLETE.json> --storage-scope local-development
python -m unittest discover -s tests -p "test_stage1_*.py"
```

Shared sources and decoded-image files are completion-verified under datasets/xm3600/<source-fingerprint>. The selected records, original four-language views, image-level target vectors (null means unknown), split manifest, source provenance and validation report are completion-verified under shared/<protocol-fingerprint>/manifests/<dataset-fingerprint>. Validation runs have separate unique namespaces. Paths are relative to PHASE1_ROOT, and image filenames contain only image IDs. Preparation resumes by verifying an already complete artifact; partial directories are rejected instead of overwritten. Source downloads use exclusive publication and checksum verification. Metadata includes CONFIRM; no model is loaded or called by these scripts.

The loader independently reconstructs source text, stable caption IDs, draw-0 choices, labels/masks, all frozen partitions and support counts. It checks all image hashes and decodes all JPEGs with Pillow. Incomplete release intersections are rejected with a request for a coordinator-approved replacement protocol; no split is rerolled here. Every read verifies completion and hashes. This deliberately prioritizes validation over reader speed in Stage 1.

Actual execution evidence is recorded below after validation. Local storage checks do not establish Drive synchronization or durability. Model adapters, inference, GPU checks and full Stage 3 G0 shuffled-label controls are outside S1-A/S1-B.


## Execution on 15 November 2026

The real original release was prepared under the user-approved local-development PHASE1_ROOT `C:\Users\APOORV\Documents\Codex\2026-11-12\ex\work\Research-stage1\.local\phase1`. No Drive mount or sync durability is claimed.

All 3,600 images decoded, with zero exclusions. Original caption counts: en 7,200; hi 8,503; bn 3,600; te 7,200. Selected records: 14,400. One Hindi duplicate-text group (`05e7b12801b04f21`) retains both original indices. All 125 split/category support records and all image assignments matched the planning audit, retaining the 14-category primary cohort and the one unknown plant cell in SCREEN. Splits remain 2,160 / 180 / 180 / 360 / 720. No model processing occurred.

Dataset fingerprint: `54825d236efcd88cc5252918ebce6d1dad6109ef10a244281f262204bc6cc5bd`.
Protocol fingerprint: `8cf019d18a28a8bb18540f4ce2e26b17fc0af15a12c8d016138dc99c397c855a`.
Dataset manifest relative to PHASE1_ROOT: `shared/8cf019d18a28a8bb18540f4ce2e26b17fc0af15a12c8d016138dc99c397c855a/manifests/54825d236efcd88cc5252918ebce6d1dad6109ef10a244281f262204bc6cc5bd/_COMPLETE.json`.

15 unit tests passed (4 data tests plus 11 unchanged coordinator foundation tests). The Windows CSV fixture newline issue and the initial source-artifact fingerprint omission were fixed before the successful full preparation run.

Git CLI push was unavailable because Windows credential storage was unavailable. The connected Drazr GitHub account published implementation commit `a5362c654e31d139d2a02bd12cb0af17f3bc275f`, directly based on the frozen commit. Its tree `02c639a107e5218e3bbdd3ae3c71e5c10ed0dffc` exactly matches local preparation commit `74ced8fe736a34d22ad6be903c0aab76c20ee629`, which remains recorded in the immutable preparation/source completion manifests. The separate validation run records the published GitHub commit. Local branch history was aligned to that identical published tree after verification.

Untested: actual Drive synchronization/durability, Colab execution, network interruption recovery, model adapter/inference/GPU checks, and full Stage 3 G0 controls. Incomplete-population handling rejects the run; publishing a changed population requires the coordinator's replacement protocol. No scientific documents or shared interface code were modified, and no integration into phase1 was performed.

The separate full-data validator passed, including two deterministic reloads, at published code commit a5362c654e31d139d2a02bd12cb0af17f3bc275f. Report relative to PHASE1_ROOT: `shared/8cf019d18a28a8bb18540f4ce2e26b17fc0af15a12c8d016138dc99c397c855a/validation/20261114T200847Z-acdacaf9aa334240a3564604a1a718a3/report.json`.
