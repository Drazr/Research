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
