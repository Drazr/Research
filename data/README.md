# Data and reproducibility metadata

Native image/caption preparation remains Stage 1 work. This repository does not claim to contain the full XM3600 dataset or a completed four-language loader.

The versioned manifests record the released-label image partitions, category order, unknown mask, support cohort and reviewed experiment phases. The annotation Git blob and content SHA256 are in manifests/source_hashes.json. Native captions must come from the original XM3600 release (derived records use orig for original text); independent captions are not translations. Pin their release/hash during Stage 1.

Store the actual images, four native-language caption views and shared annotations under PHASE1_ROOT/datasets/xm3600/<dataset_fingerprint>/. Images and annotations are shared across languages. Persistent preparation caches/shards also stay below PHASE1_ROOT; only machine-readable manifests/provenance belong here.

The existing label/split audit can be reproduced with the standard-library-only tools/reproduce_split_audit.py. This is a design-provenance utility, not the complete Stage 1 loader or a model experiment. It verifies the pinned source before parsing it and writes its cache/output under the supplied phase1 root. It accesses no model outputs.
