# Phase 3 execution plan — G9–Q6 Mechanistic & Behavioral Closure (plus isolated G7 probe)

This plan organizes Phase 3 into **two independent, maximally parallelizable tracks** — **Part A: CPU Track (`E40`, `E41`, `E44`, `E12`)** and **Part B: L4 GPU Track (`E39+E31`, `E23`, `E35`)** — building on the certified [Phase 1 review](phase1.md#phase-1-results-review--19-november-2026) and [Phase 2 review](phase2.md#phase-2-results-review--decision-register--20-november-2026). See [prelim.md §13](prelim.md#13-post-phase-1-and-post-phase-2-candidate-preliminary-experiments-e31e45) and [§6](prelim.md#6-detailed-later-preliminary-reserve) (`E12`, `E23`) for full scientific cards.

## Scientific scope & paper-boundary guardrail

Phase 3 targets the two ARR-worthy hooks uncovered in Phase 2 plus Phase 1's visual-routing mystery, while keeping **G7** strictly isolated so the research narrative never splits into two half-finished papers:

- **Primary G9–Q6 story (7 experiments across Tracks A & B):**
  1. *Asymmetric pooling & multimodal routing:* `E40` (CPU) + `E39+E31` (L4 GPU) + `E23` (L4 GPU).
  2. *Ranking vs. label-free probability calibration & selective risk:* `E41` (CPU) + `E44` (CPU) + `E12` (CPU).
- **Isolated G7 behavioral branch (`E35` only, L4 GPU):**
  - Executed and reported in its own namespace (`phase3/g7_e35/`) with its own decision entry.
  - **Paper-scope rule:** Phase 3 synthesis decides whether `E35` becomes the behavioral capstone of the `G9–Q6` paper or splits into a standalone `G7` paper. `E35` never alters `G9–Q6` claims or thresholds.
- **Frozen invariants across both tracks:**
  - Phase 1 (`PHASE1_ROOT/{datasets,shared,cache,features,checkpoints,predictions,results}/`) and Phase 2 (`PHASE1_ROOT/phase2/`) remain **strictly read-only**. All Phase 3 outputs live under `PHASE1_ROOT/phase3/`.
  - **`CONFIRM` (720 images) stays CLOSED** throughout Phase 3. (`CONFIRM` opens only in a single preregistered post-Phase-3 confirmation after the final paper story and any 3rd/larger model are locked.)
  - Reuse the exact frozen `FIT` (2,160) / `DEV-S` (180) / `DEV-C` (180) / `SCREEN` (360) partitions, 25 ordered categories, 14 fixed primary macro, preserved unknown cell `6a20677eee04fd08`, and frozen Stage 4 `(2000, 360)` bootstrap matrix (`PCG64(2026111804)`).
  - Detailed per-stage execution contracts will be finalized one stage at a time before coding each stage.

---

## Part A: CPU Track (`E40`, `E41`, `E44`, `E12`)

**Goal:** Zero-GPU, high-speed (~10–15 min wall-clock) mechanistic and calibration closure reusing Phase 1 & Phase 2 verified features, heads, statistics, and logits. **Track A can start and finish completely independently of Track B.**

| Stage | Work and handoff | Parallel execution | Est. Colab CPU time |
|---|---|---|---|
| **A1. Verify inputs & freeze CPU plan** | Bind Phase 1 Stage 4 release (`eb5cad38...`) and certified Phase 2 plan/outputs (`phase2.md` §2.9 & §4.7). Pre-initialize all `PHASE1_ROOT/phase3/` parent directories, run two-process `storage_probe`, verify `E12` FIT image-pair & source-PCA prerequisites, `E41` FIT/DEV-C/SCREEN bindings, `E44` DEV-C cutoff inputs, and publish immutable `phase3_cpu_plan`. | Runs once on 1 CPU runtime. Flush/unmount Drive afterward before launching Stage A2 parallel runtimes. | ~1–2 min |
| **A2. Parallel CPU fan-out (all 4 experiments × 2 models)** | Launch all 4 CPU experiments independently per `MODEL` (`qwen`, `gemma`):<br>• **`E40`** (cross-depth `L/4, L/2, 3L/4, L` + `caption_only` E30 head × pooling swap)<br>• **`E41`** (label-free `E11` + regularized full-covariance CORAL whitening + `E11+source_dev_c` vs `target_dev_c` ladder across both poolings)<br>• **`E44`** (`E19` 50%/75%/90% selective coverage & retained risk under `caption_mean`, `E11`, and `E17`)<br>• **`E12`** (PCA-64 image-anchored Procrustes vs ridge linear map on `caption_mean` residual `G_res`) | **Up to 8 CPU runtimes in parallel** (4 experiments × 2 models), zero cross-experiment dependencies. | ~2–5 min / run (~5–8 min total in parallel) |
| **A3. CPU Track synthesis & Track B gating** | Authenticate all 8 Stage A2 analysis artifacts, evaluate predeclared Track A decision rules (whether `E40` confirms caption-only vs visual swap, whether `E41`+`E44` solve probability/selective-risk without target labels, whether `E12` closes `G_res`), publish `phase3_cpu_summary`, and confirm/refine Track B GPU priorities (e.g., whether `E39+E31` should run `SCREEN`-only or add `FIT` matched heads). | Runs once on 1 CPU runtime (does not block launching Track B's `SCREEN`-only GPU jobs if already desired). | ~1 min |

### CPU Track parallel worker map (8 independent runs)

| Experiment | Runtime | Reused Phase 1 / Phase 2 inputs (read-only) | New Phase 3 outputs (`PHASE1_ROOT/phase3/...`) |
|---|---|---|---|
| **E40** | CPU × 2 (`qwen`, `gemma`) | Stage 4 matched blocks across all 4 depths × 2 conditions; Stage 2 SCREEN features; Stage 3 heads/scalers; Phase 2 E30 midpoint ref. | `predictions/e40_crossed/<model>/`, `results/e40/<model>/` |
| **E41** | CPU × 2 (`qwen`, `gemma`) | Stage 2 `FIT`/`DEV-C`/`SCREEN` midpoint features (`last_caption` + `caption_mean`); Stage 3 heads & FIT stats; Phase 2 E11/E17 refs. | `predictions/e41_cov/<model>/`, `results/e41/<model>/` |
| **E44** | CPU × 2 (`qwen`, `gemma`) | Stage 4 `DEV-C` & `SCREEN` logits; Phase 2 E17 DEV-C blocks & calibrators; Phase 2 E11 adapted SCREEN blocks + newly scored `caption_mean` DEV-C cutoffs. | `results/e44/<model>/` |
| **E12** | CPU × 2 (`qwen`, `gemma`) | Stage 2 `FIT`/`DEV-S`/`SCREEN` `caption_mean` midpoint features (paired FIT image anchors); Stage 4 `caption_mean` SCREEN blocks. | `maps/e12/<model>/`, `predictions/e12/<model>/`, `results/e12/<model>/` |

---

## Part B: L4 GPU Track (`E39+E31`, `E23`, `E35`)

**Goal:** Minimal, high-signal fresh L4 GPU forwards on **`SCREEN`-only first** (360 images) to resolve token-position pooling noise (`E39`), decoder visual-token routing (`E31`), direct visual-input sensitivity (`E23`), and isolated G7 structured reporting (`E35`).

| Stage | Work and handoff | Parallel execution | Est. Colab time |
|---|---|---|---|
| **B1. Pre-GPU CPU prep & pair/prompt freeze** | Run once on CPU (can overlap with Stage A1):<br>• Build deterministic **E23** disjoint `SCREEN` opposite-label image pairs (label/metadata-only rule, shared across all 4 languages, zero model scores accessed).<br>• Freeze **E39+E31** exact token-span extraction spec (`first`, `middle`, `penultimate`, `last_non_punct`, `first_half_mean`, `second_half_mean`, `last_caption`, `caption_mean` + decoder visual-token mean + caption-to-visual attention mass; `SCREEN` first).<br>• Freeze **E35** fixed category-ID prompt templates & deterministic JSON/Yes-No parser (14 primary categories; isolated G7 schema).<br>• Publish immutable `phase3_gpu_plan` under `phase3/plans/`. | Runs once on 1 CPU runtime; can run **in parallel with Stage A1/A2**. | ~1–2 min (CPU) |
| **B2. Parallel L4 GPU execution (3 independent GPU jobs × 2 models)** | Launch all 3 GPU jobs independently per `MODEL` (`qwen`, `gemma` with `HF_TOKEN` secret), using `bfloat16`, `eager`, batch 1, 32-image chunks, first/last repeat checks, and explicit-ref resume:<br>• **`E39+E31` combined extractor** *(1 shared forward per `SCREEN` view captures all 8 sub-caption token slices + decoder visual-token states + caption-to-visual attention mass across fixed depths)*<br>• **`E23` swapped-image extractor** *(midpoint `image_caption` `last_caption` + `caption_mean` on authentic vs opposite-label crossed `SCREEN` pairs)*<br>• **`E35` G7 structured reporter** *(isolated G7 branch: individual binary queries + joint multilabel category-ID report on 360 `SCREEN` images × 4 caption contexts + image-only)* | **Up to 6 L4 GPU runtimes in parallel** (3 jobs × 2 models), or queued per model on 1–2 L4s. Zero cross-job dependencies. | • `E39+E31`: ~20–35 min / model<br>• `E23`: ~15–25 min / model<br>• `E35`: ~20–35 min / model |
| **B3. Parallel CPU scoring & analysis (as each GPU job finishes)** | As soon as any model's Stage B2 GPU release publishes, immediately score & analyze on CPU:<br>• **`score_e39_e31`** (score frozen `last_caption` & `caption_mean` heads across all 8 sub-caption positions; analyze decoder visual-token accessibility & caption-to-visual attention vs `E01` gaps)<br>• **`score_e23`** (score frozen midpoint heads on authentic vs crossed pairs; compute paired signed shift toward positive image per category & language)<br>• **`score_e35`** *(isolated G7)* (format compliance, per-category F1/AUROC, exact-set accuracy by multiplicity, paired probe-minus-reporter gap vs Phase 1 `E01`/`E02`) | **Up to 6 CPU runtimes in parallel**; each starts immediately when its own Stage B2 GPU release completes (no waiting for other GPU jobs). | ~2–4 min / run (CPU) |
| **B4. Conditional `E39+E31` FIT extension *(optional gate)*** | **Only if** Stage B3 `E39+E31` shows a large position/visual-token effect that requires newly fitted matched heads (beyond frozen Phase 1 heads + attention) to publish cleanly: extract `FIT`+`DEV-S` for the winning sub-caption site / decoder visual-token site only. | Skipped automatically if Stage B3 frozen-head + attention results already settle the mechanism cleanly. | Conditional (~45–75 min / model if opened) |

### L4 GPU Track parallel worker map (6 independent GPU runs + 6 CPU scoring runs)

| Experiment | Scope | GPU workload (Stage B2) | CPU analysis (Stage B3) | Output namespace (`PHASE1_ROOT/phase3/...`) |
|---|---|---|---|---|
| **E39 + E31** | G9–Q6 | 1 shared forward per `SCREEN` draw-0 view (`image_caption` + `caption_only`) capturing 8 caption slices + decoder visual tokens + attention mass. | Frozen-head token-position curves + attention/visual-routing correlations. | `features/e39_e31/<model>/`, `results/e39_e31/<model>/` |
| **E23** | G9–Q6 | Midpoint `image_caption` (`last_caption` + `caption_mean`) on deterministic opposite-label `SCREEN` pairs. | Frozen-head paired signed image-sensitivity shifts & category-wise bootstrap. | `features/e23/<model>/`, `results/e23/<model>/` |
| **E35** | **G7 (Isolated)** | 360 `SCREEN` images × (`en, hi, bn, te` + `image_only`): individual binary queries + joint multilabel fixed-ID generation. | Format compliance, per-category & exact-set metrics, probe-vs-reporter gap. | `g7_e35/outputs/<model>/`, `g7_e35/results/<model>/` |

---

## Joint Phase 3 Final Stage: Unified Synthesis & Paper-Lock Decision

| Stage | Work and handoff | Parallel execution | Est. time |
|---|---|---|---|
| **Stage 5 (Joint Synthesis)** | Authenticate all Track A (`E40`, `E41`, `E44`, `E12`) and Track B (`E39+E31`, `E23`, plus isolated `E35`) completion refs on mounted `PHASE1_ROOT`, run fresh-subprocess independent readback, publish `phase3_release` + `phase3_review.md`, and lock the **post-preliminary confirmation protocol**:<br>1. Final G9–Q6 paper thesis (pooling + calibration/selective risk + routing).<br>2. Explicit disposition of **G7 (`E35`)**: capstone section vs standalone paper.<br>3. Choice of 3rd/larger VLM checkpoint(s) and exact single-shot **`CONFIRM` (720 images)** preregistration. | Runs once on 1 CPU runtime after Track A3 and Track B3 complete. | ~1 min (CPU) |

---

## Recommended execution timeline (maximizing parallelism)

1. **Wave 1 (CPU prep, ~2 min):** Run **Stage A1** (`prepare_phase3_cpu`) and **Stage B1** (`prepare_phase3_gpu`) in parallel (2 CPU runtimes). Flush/unmount Drive.
2. **Wave 2 (Full parallel fan-out):**
   - **CPU runtimes (up to 8 parallel, ~5–8 min):** Run `E40`, `E41`, `E44`, and `E12` for both `qwen` and `gemma`.
   - **L4 GPU runtimes (in parallel with Wave 2 CPU!):** Run `E39+E31`, `E23`, and `E35` for `qwen` and `gemma` (up to 6 L4s, or 2 L4s queueing `E39+E31 → E23 → E35` per model).
3. **Wave 3 (Rolling CPU scoring, ~3 min each):** Run **Stage A3** as soon as CPU Track finishes; launch `score_e39_e31`, `score_e23`, and `score_e35` on CPU immediately as each L4 job completes.
4. **Wave 4 (Final Synthesis, ~1 min):** Run **Joint Stage 5 Synthesis** on CPU, review `phase3_review.md`, and lock the `CONFIRM` + 3rd-model confirmation plan.

---

## Stage A1: finalized CPU foundation specification — 20 November 2026

Planning freeze `phase3-cpu-stageA1.v1`. This section finalizes **CPU Track Stage A1 only**: shared `src/phase3/` storage/namespace/resume contracts, read-only binding of certified Phase 1 + Phase 2 artifacts, inventory resolution and preflight checks for all four CPU Track workers (`E40`, `E41`, `E44`, `E12`), frozen Track A decision thresholds, `PHASE1_ROOT/phase3/` parent pre-initialization, and `scripts/phase3/prepare_phase3_cpu.ipynb`. Stages A2–A3 and Track B remain overview-only until finalized in turn.

### A1.1 Read-only Phase 1 & Phase 2 upstream bindings

Stage A1 binds immutable Phase 1 and Phase 2 pointers once on Colab CPU (pinned `requirements-stage2-cpu.txt`, Python 3.11/3.12, single BLAS/OpenMP thread) with **zero writes outside `PHASE1_ROOT/phase3/`**, **zero raw image IO**, and **`CONFIRM` strictly closed**:

| Upstream artifact | Authoritative relative path under `PHASE1_ROOT` | Pinned SHA-256 | Role in Track A |
|---|---|---|---|
| **Phase 1 Stage 4 release** | `results/stage4/releases/20261119T064951Z-07e6f916a67f46fdb88c1812763534bc/_COMPLETE.json` | `eb5cad38deecdd34e57f88f54a8a4fd8c47d7686fc3d159c240cde657d27c94c` | Bound via `src.phase2.inputs.bind_phase1`: supplies Stage 4 matched blocks across all 4 depths, `(2000, 360)` bootstrap, Stage 3 frozen heads/scalers/FIT stats, Stage 2 feature releases, unflagged G0 (`p = 0.362`). |
| **Phase 2 `phase2_plan`** | `phase2/plans/20261119T163039Z-52eb4650cec3436bb4bf989afe3cc303/_COMPLETE.json` | `c2d8e153802ebfefd1caddcb7b0cfc9e681bd1b978a422e4475c3187effbd62d` | Verifies shared `plan_identity`, `lineage_identity`, `dataset_fingerprint`, midpoints (`qwen` 18, `gemma` 17). |
| **Phase 2 `phase2_e17_auth`** | `phase2/plans/20261119T162232Z-710a83aae9064c1aa26086b34b17bf45/_COMPLETE.json` | `129cba3ef774dc4c69ecfeabc8547e2aa2186567ec1ee77fb047fc39ba96ddff` | Certifies baseline DEV-C transfer lineage for `E41` and `E44`. |
| **Phase 2 E29 analysis & gate** | `qwen`: `.../e29/qwen/20261119T173548Z-.../_COMPLETE.json` & `.../e29_gates/20261119T173548Z-...-gate/_COMPLETE.json`<br>`gemma`: `.../e29/gemma/20261119T173446Z-.../_COMPLETE.json` & `.../e29_gates/20261119T173446Z-...-gate/_COMPLETE.json` | `qwen`: `77a8dc8f...` / `3975fedb...`<br>`gemma`: `ae3b5183...` / `7434bee0...` | Reference midpoint `image_caption` & `caption_only` pooling gaps for `E40` and residual `G_res` for `E12`/`E41`. |
| **Phase 2 E30 analysis** | `qwen`: `phase2/results/e30/qwen/20261119T230416Z-8c601fa863604e3395624e37382fdedb/_COMPLETE.json`<br>`gemma`: `phase2/results/e30/gemma/20261119T225928Z-682c54d91984488aacffba40ab0f6b2c/_COMPLETE.json` | `qwen`: `0dbe25f3...`<br>`gemma`: `0de28dc4...` | Supplies already-scored midpoint `image_caption` crossed blocks so `E40` reuses them without recomputing. |
| **Phase 2 E11 analysis** | `qwen`: `phase2/results/e11/qwen/20261119T230112Z-af0550ca6a0c494e8a4f9d1728af213e/_COMPLETE.json`<br>`gemma`: `phase2/results/e11/gemma/20261119T230113Z-af983c1752eb4861a0c9390d53c096e5/_COMPLETE.json` | `qwen`: `552d0f4f...`<br>`gemma`: `09fa67ee...` | Supplies midpoint `caption_mean` `mean_shift`/`mean_scale` adapted blocks for `E41` and `E44`. |
| **Phase 2 E17 analysis** | `qwen`: `phase2/results/e17/qwen/20261119T174402Z-317626df13c944b188686ba32d9b92cb/_COMPLETE.json`<br>`gemma`: `phase2/results/e17/gemma/20261119T220802Z-167de08008b74942813cdf81ca061e43/_COMPLETE.json` | `qwen`: `fcb7c5b1...`<br>`gemma`: `ab3d8114...` | Supplies midpoint `last_caption` DEV-C off-diagonal blocks & calibrators for `E41` and `E44`. |

*(Full 64-character hashes are pinned in [phase2.md §2.9 & §4.7](phase2.md#29-stage-1--stage-2-certified-completion-ledger--20-november-2026) and embedded in `src/phase3/cpu_plan.py`.)*

### A1.2 Per-worker inventories & scoped authorizations frozen in `phase3_cpu_plan`

Stage A1 resolves and verifies all upstream pointers once so Stage A2 workers never search directories or re-verify raw images:

1. **`E40` inventory (Cross-depth & `caption_only` E30 swap):**
   - All **4 fixed depths** per model (`qwen`: `[9, 18, 27, 36]`; `gemma`: `[9, 17, 26, 34]`) × 2 conditions (`image_caption`, `caption_only`) × 2 poolings (`last_caption`, `caption_mean`) × 4 target languages (`en, hi, bn, te`).
   - Resolves all **64 reused Stage 4 matched SCREEN blocks/model**, all **64 Stage 2 SCREEN feature release refs/model**, and all **64 Stage 3 selected source systems/model**.
   - Reuses Phase 2 E30's 8 midpoint `image_caption` crossed blocks per model; schedules the remaining **7 `(depth, condition)` pairs = 56 new crossed blocks (224 routes) per model**.
   - Preflight assertion: verifies identical dimension, residual-stream hook (`decoder:<depth>`), and `model_plan` reference across `last_caption` and `caption_mean` at every `(depth, condition)`.
2. **`E41` inventory & scoped authorization (`phase3.e41_cov_and_calibration`):**
   - Scope: fixed **midpoint** (`qwen` 18, `gemma` 17) AND **final depth** (`qwen` 36, `gemma` 34) under `image_caption`, for **both** `last_caption` and `caption_mean`.
   - Reuses Phase 2 E17's midpoint `last_caption` DEV-C blocks & calibrators and Phase 2 E11's midpoint `caption_mean` diagonal adapted blocks.
   - Authorizes reading verified Stage 2 `FIT` (2,160 images, **unlabeled input-only** for `E11_cov` CORAL/whitening in source-FIT PCA subspace `r ∈ {64, 256}` — zero target labels!), `DEV-C` (180 images, for `source_dev_c` / `target_dev_c` calibrators on the newly covered pooling/depth cells), and `SCREEN` (360 images) features.
   - Freezes the 7-arm supervision ladder: `raw` (`source_only_transfer`), `source_dev_c` (`source_dev_c_calibrated`), `E11_diag` (`target_fit_input_adapted`), `E11_cov` (`target_fit_input_adapted`), `E11_diag+source_dev_c` (`target_fit_input_adapted_plus_source_dev_c`), `target_dev_c` (`target_dev_c_supervised_calibration`), and `E11_diag+target_dev_c`.
   - Preflight assertion: verifies all 4 language `FIT` releases exist with identical paired 2,160 image IDs and runs an in-memory `en→en` `caption_mean` midpoint SCREEN diagonal reproduction check (`atol=0, rtol=0`) against Stage 4.
3. **`E44` inventory & scoped authorization (`phase3.e44_selective_cutoffs`):**
   - Scope: fixed midpoint `image_caption` (`qwen` 18, `gemma` 17), nominal coverages `(0.50, 0.75, 0.90)` using `c = max(p, 1-p)` with exact tie handling matching Phase 1 `E19`.
   - Reuses Stage 4 `DEV-C` & `SCREEN` logits, Phase 2 E17 `DEV-C` blocks & calibrators, and Phase 2 E11 adapted `SCREEN` blocks.
   - Authorizes scoring Stage 2 midpoint `caption_mean` `DEV-C` features (4 source diagonals + 12 off-diagonal routes under `E11` `mean_scale`) strictly to compute frozen `DEV-C` confidence cutoffs for the `caption_mean` raw and `caption_mean + E11` arms. Never tunes cutoffs on `SCREEN`; never alters Phase 1 E19 artifacts.
4. **`E12` inventory & contract (`phase3.e12_pca_maps`):**
   - Scope: fixed midpoint `image_caption` / `caption_mean` (`qwen` 18, `gemma` 17).
   - Resolves the 4 Stage 2 `FIT` (2,160 paired images), `DEV-S` (180 paired images), and `SCREEN` (360 images) `caption_mean` feature releases + 4 reused Stage 4 `caption_mean` SCREEN blocks per model.
   - Freezes exact `E12` recipe: source-FIT PCA basis `r = min(64, source_rank)`, source logistic head refitted in the `r=64` basis (`C ∈ {0.01, 0.1, 1.0, 10.0}` selected on source `DEV-S` only), and three map arms inside that shared `r=64` basis: (1) `identity_centered`, (2) `orthogonal_procrustes` (SVD orthogonal target-to-source map on paired `FIT` image anchors, zero target labels), (3) `ridge_linear` (regularized linear map, `alpha ∈ {0.01, 0.1, 1.0, 10.0, 100.0}` selected strictly by paired `DEV-S` feature reconstruction MSE, zero target semantic labels). Supervision label: `"paired_image_adapted"`.
   - Preflight assertion: verifies exact 2,160 paired `FIT` and 180 paired `DEV-S` image IDs across all 4 languages without loading feature `.npy` arrays during A1.

### A1.3 Predeclared Track A decision ranges (frozen before Stage A2)

Frozen in `src/phase3/cpu_plan.py` (`PHASE3_CPU_DECISION_RANGES`) and embedded in `phase3_cpu_plan`:

| Experiment | Consequential threshold (14-primary macro, 12-route mean, `valid_draws >= 1900`) | Practically negligible / park bound | Mandatory automatic sanity checks |
|---|---|---|---|
| **E40** | `caption_only` replicates pooling asymmetry if `caption_only` `feature_gain\|last >= 0.50 * image_caption` `feature_gain\|last` (CI excluding 0); depth-localized if final depth `L` `\|interaction\| < 0.015` AUROC. | Swap asymmetry absent at a `(depth, condition)` if both fixed-head gains and `\|interaction\|` CIs lie wholly within `[-0.015, +0.015]`. | Identical depth/hook/width/row keys; `en→en` matched self-check `atol=0, rtol=0` per depth/condition; midpoint `image_caption` matches Phase 2 E30 to `<= 1e-12`. |
| **E41** | Label-free probability parity if `E11_diag + source_dev_c` beats FIT-prior (CI > 0) and trails `target_dev_c` by `< 0.015` NLL; `E11_cov` ranking recovery consequential if AUROC gap reduction `>= 0.010` (or `>= 25%` of `G_res`, CI > 0). | `E11_cov` ranking negligible if AUROC gap-reduction CI upper bound `< 0.005`. | `mean_shift` ranking invariance `<= 1e-12`; `E11_cov` uses zero target labels; positive `T > 0`; `E17` `<= 1e-12` AUROC/AP invariance. |
| **E44** | Selective-confidence recovery consequential if at nominal 75% coverage, mean `\|coverage - 0.75\|` drops to `<= 0.030` AND mean retained risk drops by `>= 0.020` vs Phase 1 raw E19 while positive retention remains `>= 0.50`. | Negligible if retained-risk reduction CI upper bound `< 0.005` or coverage drift worsens. | Cutoffs fitted strictly on `DEV-C` (zero `SCREEN` tuning); Phase 1 E19 raw baseline reproduced to `<= 1e-12`; explicit support status (`>= 20` per outcome class). |
| **E12** | Rotational/linear recovery consequential if `orthogonal_procrustes` or `ridge_linear` reduces `r=64` `G_res` by `>= 0.010` AUROC (or `>= 33%` of `G_res`, CI > 0); constraint matters if `\|ridge - procrustes\| >= 0.010` (CI excluding 0). | Negligible if both maps' AUROC gap-reduction CI upper bounds `< 0.005`. | Zero target semantic labels in map fitting or `DEV-S` MSE ridge selection; `r=64` full-space reference reported separately from reduced-space arms. |

### A1.4 Storage, pre-initialization, and wire schema (`phase3_cpu_plan`)

- **Strict namespace guard (`src/phase3/schema.py`):** `require_phase3_namespace(namespace)` rejects any write outside `PHASE1_ROOT/phase3/`. `require_open_split(split)` rejects `"CONFIRM"` before any IO.
- **Pre-initialized parent directories (`PHASE3_PARENT_DIRS`, created once in Stage A1 before concurrent writers):**
  - `phase3/probes`, `phase3/plans`, `phase3/releases`
  - `phase3/predictions/e40_crossed/{qwen,gemma}`, `phase3/results/e40/{qwen,gemma}`
  - `phase3/predictions/e41_cov/{qwen,gemma}`, `phase3/calibrators/e41/{qwen,gemma}`, `phase3/results/e41/{qwen,gemma}`
  - `phase3/results/e44/{qwen,gemma}`
  - `phase3/maps/e12/{qwen,gemma}`, `phase3/predictions/e12/{qwen,gemma}`, `phase3/results/e12/{qwen,gemma}`
  - Track B parent dirs pre-created simultaneously (`phase3/features/{e39_e31,e23}/{qwen,gemma}`, `phase3/results/{e39_e31,e23}/{qwen,gemma}`, `phase3/g7_e35/{outputs,results}/{qwen,gemma}`) so Drive never races when Track B starts.
- **Published artifact:** `phase3/plans/<run_id>/_COMPLETE.json` (`KIND_CPU_PLAN = "phase3_cpu_plan"`, `SCHEMA_CPU_PLAN = "phase3.cpu_plan.v1"`), containing `schema`, `phase3_version`, `plan_identity`, `lineage_identity`, `dataset_fingerprint`, `protocol_fingerprint`, `phase1_stage4_release`, `phase2_refs`, `bootstrap`, `depths`, `midpoints`, `storage_probe`, `initialized_dirs`, `e40_inventory`, `e41_inventory`, `e44_inventory`, `e12_inventory`, `preflight_checks`, `decision_ranges`, `confirm_excluded: True`.

### A1.5 Stage A1 acceptance criteria & implementation ownership

Stage A1 is a single foundation unit (no parallel split needed for A1; it freezes `src/phase3/interfaces.py` and `docs/phase3_cpu_interfaces.md` so all **4 Stage A2 CPU workers** can then branch and run in parallel):

- **Owned files (Stage A1 Foundation):** `src/phase3/{__init__,schema,cpu_plan,interfaces}.py`, `docs/phase3_cpu_interfaces.md`, `scripts/phase3/prepare_phase3_cpu.ipynb`, `tests/test_phase3_cpu_foundation.py`.
- **Acceptance checklist:**
  - [ ] All local unit/fixture tests in `tests/test_phase3_cpu_foundation.py` pass (namespace guard, `CONFIRM` rejection, two-process `storage_probe`, Phase 2 ref tamper rejection, `phase3_cpu_plan` roundtrip, notebook contract).
  - [ ] `scripts/phase3/prepare_phase3_cpu.ipynb` passes on mounted Colab CPU (`PHASE1_ROOT`), passes `storage_probe` under `phase3/probes/`, pre-initializes all `PHASE3_PARENT_DIRS`, passes `E40`/`E41`/`E12` metadata & `en→en` preflight checks, and publishes immutable `phase3_cpu_plan` `{path, sha256}`.
  - [ ] Zero Phase 1 or Phase 2 mutation; zero `CONFIRM` access.

---

## Stage A2: finalized CPU parallel workers (`E40`, `E41`, `E44`, `E12`) — 20 November 2026

Planning freeze `phase3-cpu-stageA2.v1`. All four Stage A2 CPU workers import shared contracts exclusively from `src.phase3.interfaces` (plus `src.phase2.interfaces`, `src.stage3.{engine,readouts}`, `src.stage4.{plan,predictions,metrics}`) and branch in parallel from `PHASE3_A2_BASE_SHA` on `phase3` with strictly disjoint file ownership.

### A2.1 Shared Stage A2 runtime & contract rules (all 4 workers)

- **Entry binding:** Load `plan = load_phase3_cpu_plan(root, protocol, plan_ref, bind=True)` so `plan.binding` (`Phase1Binding`) is already bound and authenticated.
- **Protocol keys:** Always read categories from `protocol["category_order"]` (25 ordered categories), primary categories from `protocol["primary_categories"]` (14 categories), and SCREEN minimum support from `int(protocol["minimum_positive_and_negative"]["SCREEN"])` (`20`). Preserve unknown cell `6a20677eee04fd08` (`plant.n.02 = null`).
- **Bootstrap & metrics:** Reuse frozen Stage 4 `(2000, 360)` bootstrap (`load_bootstrap(plan.binding.stage4_plan)`) and `src.stage4.metrics` (`probabilities`, `binary_metrics`, `bootstrap_counts`, `bootstrap_binary_metrics`, `fixed_macro`, `bootstrap_interval`, `paired_image_alignment`, `support_status`). Compute all differences, macros, and 12-route means **within every bootstrap draw** before taking linear 2.5th/97.5th percentiles.
- **Colab notebooks:** 4-cell structure (`CODE_REF = 'phase3'`, `PHASE1_ROOT = Path('/content/drive/MyDrive/phase1')`, single `MODEL` selector `'qwen'` or `'gemma'` in Cell 3, `PLAN_REF` in Cell 3, single BLAS/OpenMP thread, pinned `requirements-stage2-cpu.txt`, streaming JSON progress).
- **Guardrails:** Strictly one producer per `<experiment>/<model>`; writes only under `PHASE1_ROOT/phase3/` via `publish_phase3`; Phase 1 & Phase 2 strictly read-only; `CONFIRM` strictly closed (`require_open_split`); explicit-ref resume via `verify_resume_ref`.

### A2.2 Frozen wire schemas & worker specifications

#### Worker 1 — `E40` (`src/phase3/run_e40.py`, `scripts/phase3/run_e40.ipynb`, `tests/test_phase3_e40.py`)
- **Scope:** Cross-depth (`qwen`: `[9, 18, 27, 36]`, `gemma`: `[9, 17, 26, 34]`) and `caption_only` E30 2×2 head × pooling swap.
- **Reuse:** Reuses Phase 2 E30's 8 midpoint `image_caption` crossed blocks (`plan["e40_inventory"][model]["reused_midpoint_crossed_refs"]`) and all 64 Stage 4 matched SCREEN blocks.
- **New scoring:** Scores the remaining **7 `(depth, condition)` cells** (56 new crossed blocks / 224 routes per model: `head_last_on_mean` and `head_mean_on_last`, including crossed `t→t` diagonals) via `src.stage3.engine.transform` after running in-memory `en→en` matched self-checks (`atol=0, rtol=0`) per `(depth, condition, pooling)` and asserting midpoint `image_caption` parity (`<= 1e-12`) against Phase 2 E30.
- **Outputs:**
  - Crossed blocks under `phase3/predictions/e40_crossed/<model>/<run_id>/` (`KIND_PREDICTION`, `SCHEMA_PREDICTION`).
  - Analysis under `phase3/results/e40/<model>/<run_id>/` (`KIND_ANALYSIS`, `SCHEMA_ANALYSIS`) containing per-`(depth, condition)` 4-arm summaries, paired `feature_gain_fixed_head_last`, `feature_gain_fixed_head_mean`, `interaction`, `off_distribution_collapse`, AND the paired `image_caption − caption_only` swap contrast on identical `(2000, 360)` draws.

#### Worker 2 — `E41` (`src/phase3/run_e41.py`, `scripts/phase3/run_e41.ipynb`, `tests/test_phase3_e41.py`)
- **Scope:** Unified 7-arm supervision ladder (`raw`, `source_dev_c`, `E11_diag`, `E11_cov`, `E11_diag+source_dev_c`, `target_dev_c`, `E11_diag+target_dev_c`) across **midpoint** and **final depth** (`image_caption`, both `last_caption` and `caption_mean`).
- **Reuse & new computations:**
  - Reuses Phase 2 E17's midpoint `last_caption` DEV-C blocks/calibrators and Phase 2 E11's midpoint `caption_mean` diagonal adapted blocks.
  - Fits `temp` and `temp_intercept` BFGS calibrators (exact analytic gradient, `gtol=1e-10`, `maxiter=500`, L-BFGS-B polish on `status == 2` precision loss matching Phase 2 `run_e17.py`) on `source_dev_c` and `target_dev_c` for the newly covered `(depth, pooling)` cells (explicitly verifying `T > 0` and `<= 1e-12` AUROC/AP invariance).
  - Computes **`E11_diag`** (`mean_shift` + `mean_scale` from Stage 3 FIT `statistics`, verifying `mean_shift` `<= 1e-12` AUROC/AP invariance) and **`E11_cov`** (regularized full-covariance CORAL whitening in source-FIT PCA subspace `r ∈ {64, 256}` using unlabeled `FIT` features only — **zero target labels** — projecting `H_t` into source-FIT PCA, whitening by shrunk `Sigma_t^{-1/2}`, re-coloring by `Sigma_s^{1/2}`, reconstructing back to raw space, and passing through unchanged source scaler + head).
  - Evaluates all 7 arms on reused Stage 4 `SCREEN` logits/features vs `raw`, vs `FIT-prior`, and vs `target_dev_c` on `(2000, 360)` bootstrap draws.
- **Outputs:** `phase3/predictions/e41_cov/<model>/<run_id>/`, `phase3/calibrators/e41/<model>/<run_id>/`, `phase3/results/e41/<model>/<run_id>/`.

#### Worker 3 — `E44` (`src/phase3/run_e44.py`, `scripts/phase3/run_e44.ipynb`, `tests/test_phase3_e44.py`)
- **Scope:** Selective-confidence coverage and retained risk (`E19` protocol at nominal `50%`, `75%`, `90%` coverage using `c = max(p, 1-p)` with exact tie handling) at midpoint `image_caption` across 5 arms:
  1. `last_caption:raw` (reproducing Phase 1 E19 to `<= 1e-12`),
  2. `last_caption:source_dev_c` (Phase 2 E17 `source_dev_c:temp_intercept`),
  3. `last_caption:target_dev_c` (Phase 2 E17 `target_dev_c:temp_intercept`),
  4. `caption_mean:raw` (cutoffs fitted on Stage 2 `caption_mean` `DEV-C` source diagonals),
  5. `caption_mean:e11_mean_scale` (cutoffs fitted on `caption_mean` `DEV-C` diagonals/adapted routes, evaluated on Phase 2 E11 `mean_scale` `SCREEN` logits).
- **Mandatory reporting:** Actual coverage & `|coverage - nominal|` drift, retained risk vs base error vs score-blind risk at matched coverage, positive vs negative retention rates, explicit support status (`>= 20` per outcome class), and paired differences vs `last_caption:raw` on `(2000, 360)` bootstrap draws (cutoffs held fixed during resampling).
- **Output:** `phase3/results/e44/<model>/<run_id>/` (`KIND_ANALYSIS`, `SCHEMA_ANALYSIS`).

#### Worker 4 — `E12` (`src/phase3/run_e12.py`, `scripts/phase3/run_e12.ipynb`, `tests/test_phase3_e12.py`)
- **Scope:** Image-anchored PCA-64 constrained (`orthogonal_procrustes`) vs flexible (`ridge_linear`) feature maps on midpoint `image_caption` / `caption_mean` (`G_res` residual).
- **Exact recipe:**
  - Verify exact 2,160 paired `FIT` and 180 paired `DEV-S` image IDs across all 4 languages.
  - For each source language `s`, fit a source-`FIT` PCA basis `r = min(64, source_rank)` (mean-centered on source `FIT`) and fit the `r=64` source linear category head (`C ∈ {0.01, 0.1, 1.0, 10.0}` selected on source `DEV-S` primary macro AUROC, deterministic tie-break favoring stronger regularization). Report full-space Stage 4 `caption_mean` performance separately as reference.
  - For each off-diagonal route `s ≠ t` (12 directed routes), project target `t` `FIT`/`DEV-S`/`SCREEN` features into source `s`'s `r=64` basis and compare 3 arms:
    1. `identity_centered` (target centered by target `FIT` mean, shifted to source `FIT` mean in `r=64` space),
    2. `orthogonal_procrustes` (SVD orthogonal `Q ∈ O(r)` minimizing `||Z_t Q - Z_s||_F^2` on paired 2,160 `FIT` image anchors — zero target labels),
    3. `ridge_linear` (`W = (Z_t^T Z_t + alpha I)^{-1} Z_t^T Z_s`, with `alpha ∈ {0.01, 0.1, 1.0, 10.0, 100.0}` selected strictly by paired 180-image `DEV-S` feature reconstruction MSE — **zero target semantic labels!**).
  - Report paired `SCREEN` `AUROC`/`AP`/`Brier`/`NLL` gap reductions, `ridge − procrustes` constraint contrast, recovered fraction of `r=64` `G_res`, and paired `SCREEN` feature reconstruction MSE alongside functional transfer (`supervision_label = "paired_image_adapted"`).
- **Outputs:** `phase3/maps/e12/<model>/<run_id>/` (`KIND_MAPS`, `SCHEMA_MAPS`), `phase3/predictions/e12/<model>/<run_id>/` (`KIND_PREDICTION`, `SCHEMA_PREDICTION`), `phase3/results/e12/<model>/<run_id>/` (`KIND_ANALYSIS`, `SCHEMA_ANALYSIS`).

### A2.3 Stage A2 parallel coding ownership

| Chat | Branch / base | Owned files (exclusive) | Est. Colab CPU runtime |
|---|---|---|---|
| **Chat 1: E40** | Branch from `PHASE3_A2_BASE_SHA` | `src/phase3/run_e40.py`, `scripts/phase3/run_e40.ipynb`, `tests/test_phase3_e40.py` | ~3–5 min / model |
| **Chat 2: E41** | Branch from `PHASE3_A2_BASE_SHA` | `src/phase3/run_e41.py`, `scripts/phase3/run_e41.ipynb`, `tests/test_phase3_e41.py` | ~3–5 min / model |
| **Chat 3: E44** | Branch from `PHASE3_A2_BASE_SHA` | `src/phase3/run_e44.py`, `scripts/phase3/run_e44.ipynb`, `tests/test_phase3_e44.py` | ~2–3 min / model |
| **Chat 4: E12** | Branch from `PHASE3_A2_BASE_SHA` | `src/phase3/run_e12.py`, `scripts/phase3/run_e12.ipynb`, `tests/test_phase3_e12.py` | ~3–5 min / model |


---

## Stage B1: finalized L4 GPU foundation specification (`E39+E31`, `E23`; `E35` dropped) — 20 November 2026

Planning freeze `phase3-gpu-stageB1.v1`. Based on Stage A3 priorities, **`E35` (`G7`) is dropped** (`e35_excluded = True`) so Track B focuses strictly on closing **`G9–Q6`** (`E39+E31` and `E23`).

- **Owned files (Stage B1 Foundation):** `src/phase3/gpu_plan.py`, `scripts/phase3/prepare_phase3_gpu.ipynb`, `tests/test_phase3_gpu_foundation.py` (plus Stage B1 exports in `src/phase3/{schema,interfaces}.py`).
- **Scope & artifacts (`phase3_gpu_plan`, `KIND_GPU_PLAN = "phase3_gpu_plan"`, `SCHEMA_GPU_PLAN = "phase3.gpu_plan.v1"`):**
  - Authenticates Phase 1 (`bind_phase1`), optional Stage A1 `phase3_cpu_plan` (`93cbf480...`), and Stage 1 metadata with zero raw image IO and `CONFIRM` strictly closed.
  - Freezes `E39_SITES` (8 sub-caption token slices + Unicode `P*/Z*/C*` non-punct heuristic in `select_e39_token_indices`), `E31_SITES` (`decoder_visual_mean`, `attention_from_last_caption`, `attention_from_caption_mean`), 12 `SCREEN`-only 32-image chunks (`1,440` views/condition).
  - Freezes `build_e23_pair_plan`: deterministic label/metadata-only (`G9Q6-E23-pair-v1`) disjoint opposite-label `SCREEN` pairs per category (`6a20677eee04fd08 / plant.n.02 = None` preserved; zero model scores accessed), deduplicated directed swapped-image pairs chunked at 32 pairs, plus 8-row `authentic_parity_sample`.
  - Freezes `PHASE3_GPU_DECISION_RANGES` for `E39`, `E31`, and `E23`.
