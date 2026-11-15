# Instructions for all Research chats

Read plan.md, research.md and the relevant prelim.md cards before coding. Also read docs/stage1_contracts.md and docs/stage1_ownership.md before Stage 1 implementation; their frozen APIs and file ownership apply to both worker branches. Use M1.md/M2.md for methodological inspiration. The historical references/analysis.md does not override the current questions or data constraints.

The current implementation scope is Stage 1 of G9-Q6. Stage 1 implementations and notebook entry points exist; Stages 2-5 have overview descriptions only. Do not claim later stages, experiments, Drive durability or model runs are complete because files/directories exist. Use the evidence-qualified status below, rather than older planning statements, for the latest handoff.

- Every model-dependent notebook uses one MODEL selector, qwen or gemma; its shared implementation accepts --model. Resolve exact pinned configuration through a shared registry. Use adapters for architecture-specific code. Do not invent a default checkpoint/revision or silently change model precision/template/processor policy.
- Model-independent preparation runs once. Preserve the common image IDs, caption selections, 25-category order, known-cell masks and image-group partitions across models/languages. All caption views of an image stay in one partition.
- Use original English/Hindi/Bengali/Telugu captions and shared image-presence labels. Captions are independently written, not verified translations. Do not use Bengali/Telugu mention proxies or interpret presence as caption mention, object count, relation, OCR or cultural gold.
- No new human semantic annotation/verification, dataset enrichment, base-model updates, activation patching, steering or free-caption judging. Automatic preparation, model execution and diagnostic readout fitting are allowed.
- Keep CONFIRM out of preliminary model preprocessing, features, scores and outcome inspection. Released-ID completeness/support metadata may be checked. Never reroll a split for an appealing model result.
- Respect FIT / DEV-S / DEV-C / SCREEN roles in prelim.md. Never fit source scalers, select layers or tune models using SCREEN; never describe target-supervised fitting as unchanged-source transfer.
- Preserve the invalid plant.n.02 cell for image 6a20677eee04fd08 as unknown. Keep all 25 targets; use the fixed 14-category primary cohort only when the actual audited population/support matches. Unsupported metrics stay undefined, not zero.
- Store persistent caches, shards, features, checkpoints, predictions and logs under configurable PHASE1_ROOT, pointing to Drive's phase1 folder. Separate shared data artifacts from model/configuration/run namespaces. Never overwrite another job's shard or append concurrently to a shared file. Verify upstream completion/hashes before consumption.
- Share extraction where inputs/masks/sites really permit it and reuse E01 predictions for E19/E25. A shared filename or model-family alias alone does not prove cache compatibility.
- Use the user-requested Stage 1 notebooks as entry points, with reusable Python implementations in src/stage1. Preserve resolved configs/seeds, independent execution processes and resumable outputs. Do not duplicate shared logic per experiment/model without a substantive reason.
- Keep datasets, weights and runtime arrays out of ordinary Git history. Commit reproducibility metadata, source hashes, small manifests, code and documentation. Never commit credentials, access tokens or local secret configuration.
- Each chat owns specified files. Use phase1 as integration branch and distinct task branches/worktrees for parallel changes. Coordinate changes to shared contracts, registry, helpers and plan.md with their owner; review integration before marking a whole stage done. Routine authorized work does not require an extra approval checkpoint.
- G0 is staged: initial data/adapter smoke checks precede bulk work; full shuffled-label controls finish after real feature caches exist, before scientific interpretation. Smoke outputs are not scientific evidence or final checkpoints.

If current instructions conflict with a historical note, follow the user's current instructions and the frozen current protocol. Make necessary changes explicit, preserve unrelated work, and report actual validation and remaining gaps honestly.

User-requested Stage 1 entry points are now notebooks in scripts/stage1. Keep reusable implementations in src/stage1 and preserve reproducibility, model selection, provenance and validity rules. Do not recreate removed Stage 1 CLI entry files without a new need.

## Session decisions and operational handoff — 15 November 2026

### Repository and accounts

- Repository: https://github.com/Drazr/Research. Use the Drazr GitHub connection for writes, not ShauryaVArdhanSinghKaintura-1. phase1 is the integration branch; main is not the target for routine Stage 1 updates.
- Intended Drive owner is Tapas/Drazr. Google Drive and GitHub connections are independent. A screenshot showed both Drive accounts connected, while the last connector profile check still returned Shaurya. Do not infer current connector identity from that historical check or from the active GitHub account. Colab's authenticated Drive mount is also separate from the Codex Drive connector.
- The user runs notebooks in Colab. Actual model runs happen there; this local Windows checkout does not have the selected L4/model runtime. Report user-supplied execution evidence separately from checks performed locally.

### Persistent storage and the prepared data

- Colab PHASE1_ROOT: `/content/drive/MyDrive/phase1`. Mount the intended account at `/content/drive` before using it. Keep paths configurable for other machines; no implicit local fallback.
- The user successfully prepared and validated data through this mounted root. This establishes accessible mounted artifacts and the reported validation pass; it does not establish every Drive sync/remount or concurrent-writer guarantee.
- Dataset manifest (reuse for both models and every model notebook):

  ```text
  /content/drive/MyDrive/phase1/shared/8cf019d18a28a8bb18540f4ce2e26b17fc0af15a12c8d016138dc99c397c855a/manifests/54825d236efcd88cc5252918ebce6d1dad6109ef10a244281f262204bc6cc5bd/_COMPLETE.json
  ```

- Protocol fingerprint: `8cf019d18a28a8bb18540f4ce2e26b17fc0af15a12c8d016138dc99c397c855a`. Dataset fingerprint: `54825d236efcd88cc5252918ebce6d1dad6109ef10a244281f262204bc6cc5bd`.
- Frozen population: 3,600 images, 14,400 selected original-caption records across en/hi/bn/te, all 25 targets with a known mask and a 14-category primary cohort. Image partitions: FIT 2,160, DEV-S 180, DEV-C 180, SCREEN 360, CONFIRM 720. Do not rerun preparation merely because a model adapter or notebook changed; recheck the completed artifact and compatibility instead.
- Keep datasets, source caches, model_cache, runtime-configuration caches, feature shards, readout checkpoints, predictions and logs under PHASE1_ROOT. Raw/frozen base weights and diagnostic readout checkpoints are different artifacts. Preserve immutable completed artifacts, unique run IDs and checksum verification. No hard-link dependency; the source cache uses streamed copies and completion manifests.

### Selected models and numerical policy

- Qwen: `Qwen/Qwen2.5-VL-3B-Instruct`, revision `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Gemma: `google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767`.
- Runtime: full NVIDIA L4, roughly 24 GB nominal VRAM; batch one, dense bfloat16, no quantization, frozen eval mode, eager attention, no generation, maximum 1,024 processed tokens with rejection rather than truncation. On one L4 run models sequentially; independent L4s may run separate model jobs. These settings are model-specific execution choices, not model-independent checkpoints or guarantees for later bulk workloads.
- Qwen processor pixels: min 3,136 / max 200,704. Gemma pan-and-scan disabled. Slow/PIL image processor and fast text tokenizer are explicit. Do not equate these architectures' image preprocessing or visual stages.
- Same-device repeat tolerances are initially atol=rtol=0; investigate a mismatch before documenting any policy change. Do not silently loosen tolerances to obtain a pass.
- Exact selected dependencies live in requirements-l4.txt and resolved configs, including torch 2.9.1, torchvision 0.24.1 and transformers 4.57.6. These are direct pins, not a complete transitive lock. Record installed versions; do not casually upgrade to suppress a deprecation warning.
- configs/models.json is a pinned registry skeleton, not a substitute for each model's generated runtime configuration. Gemma's template/token metadata is resolved through authorized model access. Use the exact matching models.json printed by configure_l4; never use another model's artifact or assume the alias alone identifies it.

### Notebook execution and configuration handoff

1. Run `scripts/stage1/stage1_cpu.ipynb` on CPU: preparation then validation, automatically passing the manifest. This is shared work, run once for both models.
2. For each MODEL separately, on L4 run `configure_l4.ipynb` then `validate_model.ipynb` then `run_validity_smoke.ipynb`. Continue only after each reports passed. configure_l4 requires the real L4/version checks but performs metadata resolution, not weight inference. The latter two load the actual VLM.
3. Notebooks call shared implementations under src/stage1 in separate processes. The five former scripts/stage1 Python entry files were removed at the user's request. Notebook cells do not define __file__; do not tell the user to paste an unmodified CLI body into a cell.
4. Every notebook/runtime has its own variables. Set MODEL before selecting MODEL_CONFIG. Replace DATASET_MANIFEST and MODEL_CONFIG placeholders in the settings cell, or override them after settings and before execution. Check `.is_file()` after mounting Drive. Rerunning settings can reset overrides. Fresh notebooks do not inherit variables from earlier notebooks.
5. CPU outputs do not select a model config. configure_l4 prints a new models_config path on every successful run. Use that exact per-model path in validation and smoke; avoid stale mappings from earlier chat messages. Preserve prior artifacts rather than overwriting them.
6. Use a clean phase1 checkout and record its actual commit. Do not pull changes midway through an active run. After updating code, rerun notebook setup to refresh CODE_COMMIT and common arguments. Current notebooks explicitly require the runtime-configuration artifact's code_commit to match the checkout; rerun lightweight configure_l4 if that gate reports a mismatch. Never forge an artifact's code_commit or bypass the guard. Documentation-only commits do not retroactively invalidate historical reports at their original commit, but this implementation's exact-commit gate still applies to newly launched runs.

### Hugging Face authentication

- Gemma is gated. The user obtained access and successfully configured it after authenticating. That does not authenticate a different Colab runtime automatically.
- Enable notebook access to the `HF_TOKEN` Colab secret in each Gemma notebook. The authentication cell uses an existing HF_TOKEN environment variable, then Colab Secrets, then a hidden prompt; it logs in and exports HF_TOKEN to the child execution process. Use a read-capable token from the same account that has model access. If a stale environment token is present, replace it with the intended secret rather than assuming the secret overrides it.
- Never embed, print or commit tokens. Gemma terms/access approval must be completed by the user. A 401/gated-repository error can mean missing runtime authentication even when configure_l4 previously passed elsewhere. Authenticate in configure, validation and smoke runtimes. Qwen is public and does not require a token.

### Extraction and error lessons

- extraction.v2 makes condition and purpose explicit: image_caption, caption_only, image_only; stage1_smoke permits FIT only, preliminary_features permits FIT/DEV-S/DEV-C/SCREEN. CONFIRM stays closed before IO/tokenization/cache consumption. Image-only is one observation per image, not four language observations. Preserve site/condition compatibility and complete row/view identities.
- Cache identities are layered: execution + input view + physical site for raw features; training targets/splits/masks and fitting recipe for readouts; prediction references and scoring/threshold/bootstrap choices for analyses. Changing a probe C value or analysis metric must not unnecessarily invalidate raw GPU features. Completion checks and full provenance remain required even when mathematical feature keys match.
- Actual Qwen validation exposed a bug: unchanged native captions need not occur verbatim in decoded token text. The fix in commit `d73133ce51247f10dd6bc7af35ee9fe9b59ed517` aligns fast-tokenizer offsets against original serialization, then verifies every non-image processed token, permitting only image-placeholder expansion. Never normalize/change the dataset caption or disable alignment checks to get a pass. Gemma keeps its conservative existing alignment unless evidence supports a tested change.
- Forty CPU regressions passed after that fix. With the actual pinned Qwen tokenizer/template and the deterministic two-FIT-image/four-language selection, 16 caption-condition alignment checks passed and reproduced eight failures of the old decoded-text check. Image expansion in that local check was simulated; it was not L4 inference or a full processor/image execution test.
- TensorFlow oneDNN/CPU-instruction messages and the torch_dtype deprecation warning were not the reported blockers. Diagnose the final JSON status/blockers. A smoke/ artifact directory is also used by model validation; the directory name alone does not prove the final smoke notebook ran.

### Latest observed progress (user-supplied logs; do not overstate)

- CPU preparation and deterministic validation passed. Report relative to PHASE1_ROOT: `shared/8cf019d18a28a8bb18540f4ce2e26b17fc0af15a12c8d016138dc99c397c855a/validation/20261114T223646Z-7444326f00344a1d9d5ba6806d2a0bb4/report.json`.
- Latest reported successful Gemma config: `cache/runtime-configurations/gemma/20261115T042435Z-c81816c4f24e49df82fc30b376c70417/models.json`.
- Latest reported successful Qwen config: `cache/runtime-configurations/qwen/20261115T042448Z-50f4a5c6e98a4a0bb4d8696b3ea0a657/models.json`.
- Gemma model validation subsequently reported passed with no blockers. Artifact: `smoke/ba079306905ee4bbcfbaa8de626437ee3ce2832e2de005abb72a78ee4e6f40c7/20261115T043222Z-b0de75ce0169409c9bddd90ecc6bdbdb`. This was supplied in the validation discussion; its payload has not been independently inspected here.
- No post-fix Qwen validation pass or final run_validity_smoke pass for either model has been supplied yet. Do not mark these complete by inference. Both-model integration acceptance, full Stage 3 G0 controls and scientific experiments remain pending. All run IDs above are historical evidence pointers, not hard-coded defaults for future jobs.
