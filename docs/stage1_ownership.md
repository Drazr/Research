# Stage 1 ownership and handoff

All workers branch from the coordinator foundation commit on phase1. The coordinator supplies its immutable commit SHA in the handoff. Use separate task branches/worktrees; do not independently merge into phase1. Existing AGENTS.md and scientific constraints apply to every role.

| Owner | Branch | Owned files |
|---|---|---|
| Coordinator (this chat) | phase1 | src/common/contracts.py, src/common/storage.py, package initializers, src/stage1/run_validity_smoke.py, configs/protocol.json, docs/stage1_contracts.md, this ownership file, reports/stage1_integration.md, tests/test_stage1_foundation.py, plan.md, AGENTS.md, README.md |
| Data chat | phase1-stage1-data | src/common/data.py, src/stage1/prepare_dataset.py, src/stage1/validate_data.py, tests/test_stage1_data.py, requirements-data.txt if needed; data-specific implementation notes in docs/stage1_data.md |
| Model chat | phase1-stage1-models | src/models/registry.py, architecture adapters under src/models/, configs/models.json, src/stage1/validate_model.py, tests/test_stage1_models.py, requirements-models.txt; implementation notes in docs/stage1_models.md |

configs/models.json contains the shared skeleton now; the model chat owns populating entries after this handoff. Registry/schema semantics in contracts.py stay coordinator-owned. The data chat must use the common helper API rather than modifying contracts/storage to fit a different reader. Extra files outside owned areas require coordination with their owner, not an automatic extra user-permission flow.

Each branch reports its base commit, changed files, tests run, exact commands/configuration, actual dataset/model artifacts generated, and missing prerequisites. Use --model selectors and separate model/run artifact namespaces. Do not claim actual model tests passed merely because a fake adapter or software fixture passed. Do not alter scientific documents or mark the whole stage complete.

Integration sequence: review both task diffs and compatibility tests; integrate into phase1 preserving unrelated work; run CPU data validation and real FIT-only adapter/smoke checks for both configured models when data/access/hardware permit; record passed/blocked/failed checks and only then update the Stage 1 checklist. Full G0 shuffled-label confirmation remains Stage 3 work. Main is not changed by this foundation commit.


Integration ownership: after the worker handoff, the coordinator also owns the merge fixes for source-cache publication, streaming storage, validator provenance, the selected L4 registry/profile, configure_l4.py, requirements-l4.txt and the cache/L4 regression tests. Data/model task branches retain their original history. Future changes should be based on the integrated phase1 head, not the old frozen foundation SHA.

The coordinator also owns the condition-aware extraction/cache-key integration addendum, src/common/cache_keys.py, tests/test_stage1_extraction.py and its required adapter/validator changes. Stage 1 entry points remain under scripts/stage1; shared libraries stay under src/common and src/models. Future chats must use extraction.v2 for non-default conditions or non-FIT extraction instead of weakening the legacy smoke guard.


Notebook entry points (user-requested conversion): `scripts/stage1/stage1_cpu.ipynb` runs preparation then validation; `configure_l4.ipynb`, `validate_model.ipynb` and `run_validity_smoke.ipynb` run in that order per MODEL. The five former scripts/stage1 Python entry files are removed. Their shared implementations now live in src/stage1; scientific rules, artifacts and completion gates are unchanged. Notebook syntax and CPU fixture checks do not establish actual Colab/L4/Drive completion.
