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
