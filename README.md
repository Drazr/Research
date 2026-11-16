# Research: G9-Q6 multilingual VLM probing

Stage 3 is now specified for implementation: [fitting/control contracts](docs/stage3_contracts.md), [shared/worker ownership](docs/stage3_ownership.md) and [planning review](reports/stage3_planning.md). It reuses Stage 2 numeric artifacts on CPU with bounded checkpoints and one common fit engine. No Stage 3 code/fits were added by planning; actual launch waits for accepted Stage 2 suite validation. Both extraction jobs and E03 have user-reported component completion; a final suite pass has not yet been supplied.

Stage 2 software is now integrated on phase1. Start with the [Stage 2 runbook](docs/stage2_runbook.md); [integration verification](reports/stage2_integration.md) records 78 passed combined tests and real-data inventory checks, with actual production/L4 completion still pending. The four entry notebooks are in scripts/stage2. Older foundation/planning notes describe history rather than the latest implementation status.

Research context and the execution plan for frozen Qwen/Gemma VLM category probes using XM3600 images, original English/Hindi/Bengali/Telugu captions, and shared image-category labels. The aim is to investigate accessibility, readout portability and reliability. Language-only causal explanations and global novelty are not established by the data/design alone.

## Start here

1. [AGENTS.md](AGENTS.md): rules for every coding chat.
2. [plan.md](plan.md): five execution stages; Stages 1 and 2 are detailed. Stage 2 coordinator interfaces are implemented; workers/runtime acceptance remain pending.
3. [research.md](research.md) and [prelim.md](prelim.md): current questions, designs and scientific protocol.
4. [M1.md](M1.md), [M2.md](M2.md), [paper.md](paper.md): methods and bibliography.
5. [Historical literature analysis](references/analysis.md): context, not permission to revive superseded proposals.

## Current status

The scheduled starting block is G0 plus E01/E02/E03/E04/E19/E25. Stage 1 notebooks and shared implementations exist. Common data validation, both configurations, a Gemma model-validation pass and user-confirmed final smoke passes for both models were supplied in Colab logs. Typed artifact/readiness verification, including locating the separate post-fix Qwen model-validation report, remains pending before bulk launch. These are evidence-qualified status statements, not scientific results. Selected checkpoints/runtime pins live in configs/models.json and requirements-l4.txt; current mounted storage and evidence pointers are in AGENTS.md.

Use one model selector for model-specific work, and model/configuration fingerprints for output compatibility. Keep shared preparation/extraction reusable; run E19/E25 from saved E01 predictions. All instructions apply to both model families and all four languages where technically supported.

## Storage and parallel work

Set PHASE1_ROOT to Drive's phase1 folder, currently `/content/drive/MyDrive/phase1` in the user's Colab runtime. Persistent dataset caches, token/processor caches, feature shards, checkpoints, predictions and logs live beneath it in namespaces described in plan.md. Temporary scratch can be local; consumers require completed, compatible artifacts. The reported mounted-path CPU pass does not by itself guarantee every remote-sync failure mode.

Main versions reviewed context, configurations, scripts and small reproducibility manifests. Full images/captions/weights/runtime arrays stay outside Git. Stage folders organize code; phase1 is the proposed integration branch and separate task branches/worktrees isolate concurrent chats' edits.

## Reproduce the existing annotation/split audit

Python 3.9+ standard library suffices for this provenance utility. Point the argument to your phase1 storage root:

```shell
python tools/reproduce_split_audit.py --phase1-root /path/to/Drive/phase1
```

For an already downloaded pinned CSV, also pass --annotation-csv /path/to/xm3600_annotation.csv. No Qwen/Gemma inference is run. The implemented four-language loader and Stage 1 notebook entry points are described in plan.md; later-stage implementation remains pending.

## Stage 1 coordinator foundation

The shared implementation APIs are described in [docs/stage1_contracts.md](docs/stage1_contracts.md). Read [file ownership](docs/stage1_ownership.md) before starting parallel changes. The [integration report](reports/stage1_integration.md) preserves the historical checks and subsequent fixes; current user-supplied runtime evidence and remaining gates are summarized in AGENTS.md. Stage 1 overall completion remains pending.

Run the standard-library foundation checks from the repository root:

```shell
python -m unittest discover -s tests -p test_stage1_foundation.py -v
```

For explicit local-development preflight, run the smoke entry point with --model qwen (or gemma), --phase1-root .local/phase1 and --storage-scope local-development. Expected exit code is 2 while configurations are unresolved. Actual smoke execution additionally needs --dataset-manifest pointing to a passed dataset artifact's _COMPLETE.json and --code-commit identifying the code used. Use the actual persistent Drive root and persistent-drive scope only when that storage is available and verified. No default root is guessed.


## Integrated L4 preparation

The two Stage 1 worker implementations are integrated on phase1. The selected Qwen 3B/Gemma 4B profile, source-cache portability fix, actual remaining blockers and Colab/L4 commands are in [docs/stage1_l4.md](docs/stage1_l4.md). Selection/configuration is not actual model validation. Use the intended Drive account and run one model process at a time on a single L4.


Notebook entry points (user-requested conversion): `scripts/stage1/stage1_cpu.ipynb` runs preparation then validation; `configure_l4.ipynb`, `validate_model.ipynb` and `run_validity_smoke.ipynb` run in that order per MODEL. The five former scripts/stage1 Python entry files are removed. Their shared implementations now live in src/stage1; scientific rules, artifacts and completion gates are unchanged. Notebook syntax and CPU fixture checks do not establish actual Colab/L4/Drive completion.

## Stage 2 planning handoff

Current implementation: [executable interfaces and worker ownership](docs/stage2_interfaces.md),
[implementation/evidence report](reports/stage2_implementation.md). Both workers
must branch from the exact coordinator foundation SHA supplied after publication.
The planning history below remains context, not the current implementation status.

[Stage 2 contracts](docs/stage2_contracts.md) fix the shared feature inventory, split/condition guards, shard/storage/resume rules, E03 preprocessing and release checks. [Stage 2 ownership](docs/stage2_ownership.md) defines the future shared-interface, extraction and E03 workstreams. [Planning report](reports/stage2_planning.md) distinguishes finalized design from pending implementation and actual readiness evidence. No Stage 2 code or scientific results were produced by this planning task. Stage 2 execution remains gated on the applicable Stage 1 passes; Stages 3-5 remain overview only.

Final review revision 2 also fixes independent data/model plan bindings, plan-resolved notebook settings, same-forward token evidence, streamed E03 split releases and actual first-chunk resource projections. Both final smoke passes are recorded; they do not by themselves create Stage 2 arrays or replace model-validation payload verification. The next step is the coordinator's shared-interface implementation, not further experimental design or immediate unvalidated bulk extraction.
