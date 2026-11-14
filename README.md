# Research: G9-Q6 multilingual VLM probing

Research context and the execution plan for frozen Qwen/Gemma VLM category probes using XM3600 images, original English/Hindi/Bengali/Telugu captions, and shared image-category labels. The aim is to investigate accessibility, readout portability and reliability. Language-only causal explanations and global novelty are not established by the data/design alone.

## Start here

1. [AGENTS.md](AGENTS.md): rules for every coding chat.
2. [plan.md](plan.md): five execution stages; only Stage 1 is detailed.
3. [research.md](research.md) and [prelim.md](prelim.md): current questions, designs and scientific protocol.
4. [M1.md](M1.md), [M2.md](M2.md), [paper.md](paper.md): methods and bibliography.
5. [Historical literature analysis](references/analysis.md): context, not permission to revive superseded proposals.

## Current status

The scheduled starting block is G0 plus E01/E02/E03/E04/E19/E25. Experiment scripts and model runs are not implemented/completed by uploading these files. The existing label-only split/support audit is included. Exact checkpoints, native-caption acquisition/selection and the common image population still require Stage 1 resolution. configs/models.json is explicitly a skeleton.

Use one model selector for model-specific work, and model/configuration fingerprints for output compatibility. Keep shared preparation/extraction reusable; run E19/E25 from saved E01 predictions. All instructions apply to both model families and all four languages where technically supported.

## Storage and parallel work

Set PHASE1_ROOT to Drive's phase1 folder. Persistent dataset caches, token/processor caches, feature shards, checkpoints, predictions and logs all live beneath it in namespaces described in plan.md. Temporary scratch can be local; consumers require completed, compatible artifacts. No remote Drive folder, dataset transfer or model download is implied by this repository setup.

Main versions reviewed context, configurations, scripts and small reproducibility manifests. Full images/captions/weights/runtime arrays stay outside Git. Stage folders organize code; phase1 is the proposed integration branch and separate task branches/worktrees isolate concurrent chats' edits.

## Reproduce the existing annotation/split audit

Python 3.9+ standard library suffices for this provenance utility. Point the argument to your phase1 storage root:

```shell
python tools/reproduce_split_audit.py --phase1-root /path/to/Drive/phase1
```

For an already downloaded pinned CSV, also pass --annotation-csv /path/to/xm3600_annotation.csv. No Qwen/Gemma inference is run. The full four-language loader, pinned model environments and experimental entry points remain Stage 1/subsequent-stage work as described in plan.md.

## Stage 1 coordinator foundation

The shared implementation APIs are frozen as stage1.v1 in [docs/stage1_contracts.md](docs/stage1_contracts.md). Read [file ownership](docs/stage1_ownership.md) before starting parallel changes. The [integration report](reports/stage1_integration.md) distinguishes the 11 passed software tests from blocked actual-model smoke checks. Data/model worker implementation and Stage 1 completion remain pending.

Run the standard-library foundation checks from the repository root:

```shell
python -m unittest discover -s tests -p test_stage1_foundation.py -v
```

For explicit local-development preflight, run the smoke entry point with --model qwen (or gemma), --phase1-root .local/phase1 and --storage-scope local-development. Expected exit code is 2 while configurations are unresolved. Actual smoke execution additionally needs --dataset-manifest pointing to a passed dataset artifact's _COMPLETE.json and --code-commit identifying the code used. Use the actual persistent Drive root and persistent-drive scope only when that storage is available and verified. No default root is guessed.
