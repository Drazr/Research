# Phase 2 execution plan — G9–Q6

This plan coordinates E29, E07, E17 and the conditional E30 and E11 runs. See [the Phase 2 experiment cards](prelim.md#12-phase-2) for scientific comparisons and [the Phase 1 review](phase1.md#phase-1-results-review--19-november-2026) for the evidence motivating them. The stages describe dependencies and can overlap. Detailed execution settings will be finalized one stage at a time.

| Stage | Work and handoff | Parallel execution |
|---|---|---|
| **1. Verify inputs and freeze the Phase 2 plan** | Bind the accepted Phase 1 features, fitted heads, logits and bootstrap draws. Freeze the midpoint comparisons and decision thresholds. Reuse the passed G0 gate and check added inputs automatically. Audit E07 caption availability and support, and authorize E17's additional DEV-C scoring. | The E07 audit and E17 authorization are branch-specific. Once common inputs and its own settings are verified, a branch can start while the others prepare. |
| **2. Start E29, E17 and E07** | **E29 CPU** computes the paired pooling contrast from cached SCREEN logits. **E17 CPU** reuses source DEV-C diagonals and scores the missing frozen-source-head routes on target DEV-C. **E07 GPU** extracts only changed caption views after its audit and collects both poolings in the same forward. | Run E29 immediately and launch E17 and E07 as their inputs become ready. Separate model jobs can overlap on separate runtimes. |
| **3. Open conditional branches from E29** | Run **E30 CPU** if E29 finds a consequential reduction. Reuse matched head/pooling scores and compute only the crossed arms. Run **E11 CPU** if a supported, practically meaningful residual remains. Both reuse cached midpoint features and frozen heads. | If both qualify, run them in parallel as soon as E29 finishes, even while E07 and E17 are still running. E11 can inform a draw-0 conclusion before E07. Claims about caption robustness wait for E07. |
| **4. Score and compare as outputs arrive** | Score E07's verified new views with frozen heads and reuse E29's draw-0 reference and paired comparison code. E17 applies fitted calibrators to existing SCREEN logits. Save outputs that another step reuses or that avoid costly recalculation. | E07 scoring can start on verified completed feature units. E17 can calibrate each ready route independently. Neither requires the full Stage 3 conditional workload to finish. |
| **5. Decide which directions survive** | Combine the applicable results. Use E07 for caption sensitivity, E29 and any conditional runs for pooling and residual portability, and E17 for probability quality. Record supported, narrowed, parked and unresolved directions with their evidence. | Move to post-prelim once the surviving questions are focused enough. A third preliminary phase is optional and needs a specific unresolved decision. |

The planned first pass reuses the Phase 1 probe fits. Keep one producer for each shared output, preserve the exact image splits and masks, and keep CONFIRM closed.

---

## Stage 1: finalized specification — 19 November 2026

Planning freeze `phase2-stage1.v1`. This section finalizes **Phase 2 Stage 1 only** (input verification, shared contracts, predeclared gating/decision ranges, E07 caption coverage audit, E17 DEV-C authorization, storage/resume conventions, acceptance criteria, and Stage 1 coding handoff). Stages 2–5 remain overview-only until finalized in turn. No code, notebooks, Drive writes, model runs, or new SCREEN analyses were executed to create this plan.

### 1.1 Evidence boundary: locally verified vs pending runtime checks

| Status | Item | Exact evidence or remaining check |
|---|---|---|
| **Verified locally** | Repository checkout and baseline | Local checkout at `phase1` commit `4fa54340a1f0132a159d3bf40de9038b4907535e`, with local documentation edits on `phase1.md`, `prelim.md`, `phase2.md`, and `phase2_handoff.md`. Preserve and commit these edits cleanly before cutting coding branches/worktrees. |
| **Verified locally** | Shared protocol and frozen splits | [configs/protocol.json](configs/protocol.json) and [data/manifests/g9q6_split_planning_audit.json](data/manifests/g9q6_split_planning_audit.json): languages `en, hi, bn, te`; seed `G9Q6-first-set-20261114-v1`; splits FIT 2,160 / DEV-S 180 / DEV-C 180 / SCREEN 360 / CONFIRM 720; 25 ordered categories; 14 fixed primary categories; minimum support FIT 20 / DEV-S 10 / DEV-C 10 / SCREEN 20; preserved unknown cell `6a20677eee04fd08` (`SCREEN`, `plant.n.02 = 4` → `null`); `caption-draw-v1` selector in [src/common/contracts.py](src/common/contracts.py); 2,000 image bootstrap draws; threshold 0.5; `CONFIRM` excluded. |
| **Verified locally** | Existing Phase 1 code contracts | Verified public loaders and numeric engines in [src/common/data.py](src/common/data.py), [src/common/cache_keys.py](src/common/cache_keys.py), [src/stage2/evidence.py](src/stage2/evidence.py), [src/stage3/data.py](src/stage3/data.py), [src/stage3/engine.py](src/stage3/engine.py), [src/stage3/readouts.py](src/stage3/readouts.py), [src/stage4/plan.py](src/stage4/plan.py), [src/stage4/predictions.py](src/stage4/predictions.py), [src/stage4/metrics.py](src/stage4/metrics.py), [src/stage4/validation.py](src/stage4/validation.py), and [src/stage5/interfaces.py](src/stage5/interfaces.py). Confirmed that `stage1_dataset` stores all original captions per language in `captions/{en,hi,bn,te}.jsonl` with stable IDs `xm3600:ad44c380c86cb1240666361caabaf5684507d0ef:{image_id}:{lang}:{index}`, and Stage 3 persists reusable float64 FIT `statistics` (`mean`, `variance`, `scale`, floor `1e-12`) per group. |
| **Verified locally** | Local Windows environment constraint | On this Windows host, `git` is not on the default shell `PATH` and `python` resolves to Python 3.9 (`C:\Program Files\Python39\python.exe`), not the pinned Colab Python 3.12 CPU ([requirements-stage2-cpu.txt](requirements-stage2-cpu.txt)) or L4 GPU ([requirements-l4.txt](requirements-l4.txt)) runtimes. Coding chats must resolve exact local Git/Python paths explicitly; production runs execute in Colab. |
| **Pending runtime** | Mounted Drive access and Phase 1 payload readback | Actual Colab Drive mount (`PHASE1_ROOT`, default `/content/drive/MyDrive/phase1`, checking any shared-shortcut physical path resolution), read/write probe, and cryptographic hash/readback verification of the accepted Stage 4 release and its transitive Stage 3/2/1 manifests on mounted Drive. |
| **Pending runtime** | E07 original-caption coverage audit | Actual caption multiplicity, duplicate-text rate, genuinely changed SCREEN view counts for draws 1 and 2, and changed-cohort category support are unknown until Stage 1 CPU execution reads mounted `captions/{en,hi,bn,te}.jsonl`. |
| **Pending runtime** | E17 DEV-C diagonal self-check and GPU readiness | Actual mounted verification of Stage 2 DEV-C midpoint feature releases, `atol=0, rtol=0` diagonal logit reproduction against accepted Stage 4 DEV-C blocks, and (only if E07 passes audit) L4 `assert_runtime` and Gemma `HF_TOKEN` authentication. |

### 1.2 Accepted Phase 1 inputs and transitive lineage binding

All Phase 2 work binds one immutable root pointer—the accepted Phase 1 Stage 4 release—and traverses its verified provenance graph under `PHASE1_ROOT` (never mutating any Phase 1 directory or file):

| Artifact | Authoritative relative path under `PHASE1_ROOT` | Pinned SHA-256 / identity | Role in Phase 2 |
|---|---|---|---|
| **Accepted Stage 4 release** | `results/stage4/releases/20261119T064951Z-07e6f916a67f46fdb88c1812763534bc/_COMPLETE.json` | `eb5cad38deecdd34e57f88f54a8a4fd8c47d7686fc3d159c240cde657d27c94c` | Primary entry pointer; supplies verified `stage4_plan`, 153 prediction block refs, 47 analysis refs, and readback receipt (`stage4_complete=true`). |
| **Stage 4 plan & bootstrap** | Resolved from `stage4_release["plan"]` | Verified by `load_stage4_plan` + `load_bootstrap` | Supplies canonical block/route inventory, `screen_ids`, and the frozen `(2000, 360)` `int64` draw matrix (`PCG64(2026111804)`). |
| **Accepted Stage 3 registry & G0 gate** | Resolved from `stage4_release["stage3_registry"]` (`results/stage3-validation/20261118T195046Z-6d86dec9b61d485db66543c70f7aa1b0/_COMPLETE.json`) and `stage4_release["g0_gate"]` | Registry SHA-256 `0d60a6368037d969cac0880d3f8136d6bd5f716cc91ffd46b1a76266b95128c2`; G0 unflagged (`p = 0.362`, 145 groups) | Supplies the 145 frozen ordinary selections, selected float64 `coefficients`/`intercepts`/`estimable`, source-FIT `statistics` (`mean`, `variance`, `scale`), and amended solver-budget plan lineage (`[1000, 5000, 20000]`). Reuse G0 without rerunning it. |
| **Accepted Stage 2 suite** | Resolved from `fit_plan["suite"]` (`results/stage2-validation/20261116T010841Z-17d614fe5da340b4b6c036a431fc6be8/_COMPLETE.json`) | Verified by `suite_inventory` inside `load_fit_plan` | Supplies verified Stage 2 `model_plans`, `model_readiness` receipts (actual model/execution fingerprints, Gemma resolved template/token metadata, dimensions, physical site specs), and the 24 immutable feature releases. |
| **Accepted Stage 1 dataset** | Resolved from `data_plan["references"]["dataset"]` (`shared/8cf019d18a28a8bb18540f4ce2e26b17fc0af15a12c8d016138dc99c397c855a/manifests/54825d236efcd88cc5252918ebce6d1dad6109ef10a244281f262204bc6cc5bd/_COMPLETE.json`) | Protocol FP `8cf019d1...855a`; Dataset FP `54825d23...5bd` | Supplies `captions/{en,hi,bn,te}.jsonl`, draw-0 `records.jsonl`, `targets.jsonl`, `splits.json`, and (only if E07 passes audit) pinned SCREEN image paths under `datasets/xm3600/<source_fp>/images/`. |
| **Stage 5 synthesis (context only)** | `results/stage5/synthesis/20261119T094836Z-a93c8ec225f843f1994337b9b107e3d1/_COMPLETE.json` | `a9a59d0c48ff8d5e6c01d89f40b9a8455bbf610f2ee89b246311628fcb2d3540` | Reviewed motivating evidence appended to [phase1.md](phase1.md#phase-1-results-review--19-november-2026). Formal Stage 5 finalization remains pending and does not block Phase 2. |

**Binding verification rule:** Stage 1 `prepare_phase2` calls the existing `src.stage5.interfaces.open_release(root, protocol, stage4_release_ref)` (which authenticates `stage4_release`, its `independent_readback`, `stage4_plan`, `stage3_system_registry`, `registry_readback`, unflagged `g0_gate`, amended `stage3_plan`, and `stage2_suite_index` metadata without opening raw images or rerunning Stage 2/3/4 validators). It then verifies `load_bootstrap(plan)` and records exact resolved pointers in an immutable `phase2_plan` artifact. Downstream Phase 2 workers verify `phase2_plan` and only the specific payloads they consume once per bounded session.

### 1.3 Shared four-language, split/mask, midpoint, and resampling contracts

- **Models and fixed midpoints:** `qwen` (`Qwen/Qwen2.5-VL-3B-Instruct`, rev `66285546d2b821cf421d4f5eb2576359d3770cd3`, fixed midpoint decoder depth **18**) and `gemma` (`google/gemma-3-4b-it`, rev `093f9f388b31de276ce2de164bdc2081324b9767`, fixed midpoint decoder depth **17**). Resolve actual `model_fingerprint`, `execution_fingerprint`, `site_specs`, `dimensions`, and processor metadata strictly from the bound Stage 2 `model_plan` / `model_readiness` artifacts, not from the [configs/models.json](configs/models.json) skeleton. Every model-dependent entry point takes one explicit `MODEL` (`qwen` or `gemma`); raw coordinates never mix across models.
- **Languages and routes:** All four original caption contexts `('en', 'hi', 'bn', 'te')`, all 12 directed off-diagonal transfers (`s → t`, `s ≠ t`) and 4 receiving-context diagonals (`t → t`). Captions are independently authored; never treat caption indices across languages as aligned translations.
- **Targets, masks, and cohort:** Keep all 25 ordered categories in every score tensor and result table. Preserve the fixed 14-category primary macro (`primary_categories` in `protocol.json`) wherever supported. Preserve the unknown-cell mask (`plant.n.02 = null` for SCREEN image `6a20677eee04fd08`). Never impute unknown cells as zero or drop low-support secondary categories from tables. Any restricted cohort (e.g., E07 genuinely-changed-only subset) must carry a separate cohort label and a matched draw-0 companion computed on the exact same image-category cells.
- **Split permissions and CONFIRM closure:**
  - `FIT` (2,160 images): read-only lineage, fixed FIT-prevalence references, and (for conditional E11) verified Stage 3 FIT-only `statistics` artifacts. No new probe fitting or refitting in this first pass.
  - `DEV-S` (180 images): read-only lineage for the frozen Stage 3 `C` selections. No re-selection.
  - `DEV-C` (180 images): authorized strictly for E17 (reusing the 4 Stage 4 source diagonals per model + newly authorized scoring of the 12 missing frozen-source-head transfer routes per model; see §1.6). Never used for base-head fitting, feature scaling, or E29/E30/E11/E07 tuning.
  - `SCREEN` (360 images): reusable exploratory evaluation only. Never used to fit heads, scalers, maps, or calibrators.
  - **`CONFIRM` (720 images): CLOSED.** Rejected before image IO, tokenization, extraction, scoring, or audit draw materialization across every Phase 2 module.
- **Shared metrics and paired bootstrap:**
  - Reuse [src/stage4/metrics.py](src/stage4/metrics.py) helpers (`probabilities`, `binary_metrics`, `bootstrap_counts`, `bootstrap_binary_metrics`, `fixed_macro`, `bootstrap_interval`, `paired_image_alignment`, `support_status`).
  - Reuse the exact Stage 4 `(2000, 360)` `int64` image draw matrix (`PCG64(2026111804)` on sorted `plan["screen_ids"]`). One image is the resampling unit across languages, poolings, and caption draws.
  - Compute paired differences, 14-category macros, and 12-route means **within every bootstrap draw** before taking linear 2.5th/97.5th percentiles; report `valid_draws / 2000` and preserve undefined draws as `NaN` (serialized `null` with reason if `valid_draws == 0`).
  - Metric conventions match Stage 4: AUROC (tied ranks), AP (step-based, prevalence-sensitive companion), Brier, binary NLL (probabilities clipped to `[1e-15, 1 - 1e-15]` in float64), and 0.5-threshold sensitivity/specificity/F1/balanced accuracy. Gap directions: `G_rank = M(t→t) - M(s→t)` for higher-is-better (`auroc`, `ap`, etc.) and `G_loss = M(s→t) - M(t→t)` for losses (`brier`, `nll`), so positive always denotes a transfer penalty.
  - Claim and supervision labels remain explicit on every result row: `within_context_source_fit`, `source_only_transfer`, `source_dev_c_calibrated`, `target_dev_c_supervised_calibration`, `target_fit_input_adapted`.

### 1.4 Exact artifact inventory per Phase 2 worker

| Worker | Runtime | Reused Phase 1 artifacts (read-only) | New Phase 2 inputs / computations |
|---|---|---|---|
| **E29** (Paired pooling gap contrast) | CPU, per `MODEL` | **16 Stage 4 SCREEN prediction blocks/model** (midpoint `image_caption` `last_caption` [4] + `caption_mean` [4] = 32 routes primary; midpoint `caption_only` `last_caption` [4] + `caption_mean` [4] = 32 routes diagnostic) + Stage 4 `bootstrap.npy` + targets/masks. | Zero new feature reads or scoring. Computes paired `G(last) - G(mean)` and residual `G(mean)` per route and 12-route mean, plus reusable draw-0 midpoint reference and machine-readable `e29_gate_decision`. |
| **E17** (Source vs target calibration) | CPU, per `MODEL` | **4 Stage 4 SCREEN blocks/model** (midpoint `image_caption` `last_caption`, 16 routes) + **4 Stage 4 DEV-C diagonal blocks/model** (4 `s→s` routes) + **4 Stage 3 selected heads & source-FIT scalers/model** (loaded without FIT features) + **4 Stage 2 DEV-C feature releases/model** (midpoint `image_caption` `last_caption`) + `bootstrap.npy`. | Scores the **12 missing off-diagonal `s→t` DEV-C routes/model** once (4 blocks × 3 routes/model), fits global positive-`T` and positive-`T+b` calibrators on source DEV-C and target DEV-C separately, and evaluates all arms on reused SCREEN logits. |
| **E07** (Alternate captions, fixed-head first) | Stage 1 CPU audit; Stage 2 L4 GPU per `MODEL`; Stage 4 CPU scoring | Stage 1 audit: `stage1_dataset` (`captions/{en,hi,bn,te}.jsonl`, `records.jsonl`, `targets.jsonl`, `splits.json`). Stage 2 GPU: Stage 2 `model_plan` + `model_readiness` + `models.json` + pinned SCREEN images (changed views only). Stage 4 CPU: **8 Stage 3 midpoint `image_caption` heads & scalers/model** (`last_caption` + `caption_mean`) + Stage 4 draw-0 SCREEN logits + E29 draw-0 reference + `bootstrap.npy`. | Stage 1 audits draws 1 & 2 vs draw 0. If audit passes, Stage 2 extracts **only unique genuinely-changed SCREEN `image_caption` views** at midpoint, capturing both `last_caption` and `caption_mean` on the same forward via `extract_with_evidence`. Unchanged and byte-identical-text views reuse draw-0 scores without GPU forwards. |
| **E30** *(conditional on E29)* | CPU, per `MODEL` | **8 Stage 4 SCREEN blocks/model** (matched `last→last` and `mean→mean` midpoint `image_caption` arms, 32 routes) + **8 Stage 2 midpoint `image_caption` SCREEN feature groups/model** (4 langs × 2 poolings) + **8 Stage 3 selected heads & scalers/model** + `bootstrap.npy`. | Scores ONLY the **2 crossed arms** (8 blocks / 32 routes per model: `last_caption` head+scaler on `caption_mean` features, and `caption_mean` head+scaler on `last_caption` features) after asserting identical depth, hook, width, and row alignment. |
| **E11** *(conditional on E29)* | CPU, per `MODEL` | **4 Stage 4 SCREEN blocks/model** (midpoint `image_caption` `caption_mean`, 16 routes: supplies identity transfer and `t→t` diagonals) + **4 Stage 3 FIT `statistics` artifacts/model** (`mean`, `variance`, `scale`, floor `1e-12`, paired 2,160 FIT images—no FIT matrix reload needed!) + **4 Stage 3 selected heads/model** + **4 Stage 2 midpoint `image_caption` `caption_mean` SCREEN feature groups/model** + `bootstrap.npy`. | Applies FIT-only `mean_shift` (`h_t - mu_t + mu_s`) and `mean_scale` (`mu_s + sigma_s * (h_t - mu_t) / sigma_t`) to verified SCREEN `caption_mean` features, passes through the unchanged source scaler and head, asserts AUROC/AP invariance on `mean_shift`, and compares all 12 directed transfers + `t→t` diagonals. |

### 1.5 Predeclared decision ranges and E29 gating of E30 and E11

All thresholds below are **frozen before running E29 or inspecting any Phase 2 SCREEN output**. Evaluate gates **separately for Qwen and Gemma**; each model may open E30, E11, both, or neither independently.

#### E29 primary estimands (per model, `image_caption`, fixed midpoint, 14-category primary macro)
- Route gap in draw `b`: `G(s→t, p, b) = (1/14) * sum_{k ∈ primary} [AUROC_k(t→t, p, b) - AUROC_k(s→t, p, b)]`.
- Route pooling reduction in draw `b`: `ΔG(s→t, b) = G(s→t, last_caption, b) - G(s→t, caption_mean, b)`.
- 12-route equal-weight means computed within every draw `b`:
  - `ΔG_mean(b) = (1/12) * sum_{s≠t} ΔG(s→t, b)` with point estimate `ΔG_mean` and 95% CI `[L_Δ, U_Δ]`.
  - `G_res_mean(b) = (1/12) * sum_{s≠t} G(s→t, caption_mean, b)` with point estimate `G_res_mean` and 95% CI `[L_res, U_res]`.
- Accessibility floor: all 4 receiving-context diagonals at `caption_mean` must satisfy `AUROC(t→t, caption_mean) >= 0.750` (primary macro) so a gate never opens on a collapsed baseline.
- Bootstrap validity floor: `valid_draws >= 1900 / 2000` (95%) on the primary 12-route macro contrast.

#### Machine-readable E29 → E30 and E29 → E11 gate rules (per model)

| Downstream branch | `OPEN` condition | `PARK` condition (do not run) | `UNRESOLVED` (hold closed; log reason) |
|---|---|---|---|
| **E30** (head × pooling swap) | Accessibility floor met (`>= 0.750` on all 4 diagonals), `valid_draws >= 1900`, `ΔG_mean >= 0.015` AUROC, AND `L_Δ > 0.005` AUROC. | `valid_draws >= 1900` AND (`U_Δ < 0.015` AUROC OR accessibility floor fails). | All other cases (`valid_draws < 1900`, OR accessibility floor met with `U_Δ >= 0.015` while `ΔG_mean < 0.015` or `L_Δ <= 0.005`). |
| **E11** (FIT mean/scale on `caption_mean`) | Accessibility floor met, `valid_draws >= 1900`, AND either: **(a) Broad residual:** `G_res_mean >= 0.015` AND `L_res > 0.005`; OR **(b) Directional residual:** `L_res > 0.005` AND at least 3 of 12 directed routes have point `G(r, caption_mean) >= 0.025` with route lower 95% CI `> 0.010` (recorded as `directional_residual_trigger`; E11 still scores all 12 routes). | `valid_draws >= 1900` AND (accessibility floor fails OR (`U_res < 0.015` AND fewer than 3 routes meet the directional trigger)). | All other cases (`valid_draws < 1900`, OR accessibility floor met while neither `OPEN` nor `PARK` holds). |

- **E29 `caption_only` diagnostic:** Run alongside `image_caption` on the exact same draws. A comparable `caption_only` reduction does not block E30's swap test, but narrows any vision-specific interpretation of pooling gains.
- **E11 vs E07 timing:** If E29 opens E11, E11 may run immediately in parallel with E30/E07/E17 to settle draw-0 covariate statistics; any claim that E11's result is robust across alternate captions waits for E07.

#### Predeclared Stage 5 interpretation ranges for the other four experiments (frozen here)

| Experiment | Consequential effect range (primary macro, 12-route mean unless stated) | Practically negligible bound (for `park within tested scope`) | Mandatory automatic invariance / sanity checks |
|---|---|---|---|
| **E17** | Proper-loss gain `>= 0.015` NLL OR `>= 0.005` Brier vs raw (and vs FIT-prior reference), with 95% CI excluding `0` favorably. Separate source-DEV-C vs target-DEV-C verdicts. | 95% CI upper bound `< 0.005` NLL AND `< 0.002` Brier improvement. | Per-category SCREEN AUROC/AP unchanged vs raw within `1e-12` (up to ties); positive `T > 0`; `T`-only leaves 0.5 decisions unchanged. |
| **E07** | View-sensitive if `|G(draw_k) - G(draw_0)| >= 0.015` AUROC with 95% CI excluding `0`, or pooling-reduction sign reverses; motivates narrowing claim or a separate E07 refit decision. | View-stable if `|G(draw_k) - G(draw_0)| < 0.015` AUROC and 95% CI lies wholly within `[-0.025, +0.025]` with preserved pooling-reduction sign. | Draw-0 reproduction exact; unchanged/byte-identical views match draw-0 logits to `atol=0, rtol=0`; restricted changed-only cohort paired with exact matched draw-0 subset. |
| **E30** | Fixed-head feature-pooling gain `>= 0.015` AUROC (CI excluding `0`) supports feature-view effect; `|interaction| >= 0.015` AUROC (CI excluding `0`) supports head–pooling dependency. | Both fixed-head feature gains and `|interaction|` have 95% CIs wholly within `[-0.015, +0.015]`. | Identical depth, hook, width, and row keys across crossed arms; flag off-distribution collapse if both crossed arms fall `>= 0.015` below matched `last_caption`. |
| **E11** | `mean_scale` AUROC gap reduction `>= 0.010` (or `>= 25%` of `G_res_mean`) with 95% CI excluding `0` supports narrow covariate-scale account. Loss-only gain (`>= 0.015` NLL or `>= 0.005` Brier without AUROC gain) routes to E17's probability story. | `mean_scale` AUROC gap-reduction CI upper bound `< 0.005` (with precise interval). | `mean_shift` per-category AUROC/AP must match identity within `1e-12` (hard failure if violated); FIT `mu`/`sigma` verified on identical 2,160 paired FIT IDs with `1e-12` variance floor. |

### 1.6 E17 DEV-C scoring authorization and calibration contract

- **Explicit permission upgrade:** Phase 1 Stage 4 restricted DEV-C to the 8 E19 source diagonals (`s = t`). Phase 2 Stage 1 authorizes a scoped `phase2.e17_dev_c_transfer` permission over the accepted Stage 2 midpoint `image_caption` / `last_caption` DEV-C feature releases (`180` images × 4 target languages × 2 models).
- **Exact DEV-C score inventory:**
  - **Reused unchanged from Stage 4:** 4 source diagonals per model (`s = t`, 8 routes total).
  - **Newly authorized & scored once on CPU:** 12 off-diagonal frozen-source-head transfer routes per model (`s ≠ t`, 24 routes total), stored as **4 target-context DEV-C score blocks per model** (each block loads one verified target DEV-C feature matrix once and scores the 3 non-diagonal source heads `s ≠ t` with their frozen Stage 3 `coefficients`, `intercept`, and source-FIT `statistics` in float64).
  - **Mandatory Stage 1/2 reproduction self-check:** Before publishing E17 DEV-C transfer blocks for a model, score the `en → en` DEV-C diagonal through the Phase 2 scorer in memory and assert exact match (`atol=0, rtol=0` or documented exact float64 identity) against Stage 4's accepted `en → en` DEV-C logits.
- **Hard boundaries:**
  - The target calibrator for route `s → t` MUST fit on the **frozen source `s` head scored on target `t`'s DEV-C features**, never on target `t`'s own trained head (which remains a separate reference).
  - Never fit or refit base heads, scalers, or C selection on DEV-C; never tune calibrators on SCREEN; never alter Stage 4's E19 cutoffs/claims; never open CONFIRM.
- **Frozen calibrator fitting recipe (CPU, float64):**
  - Forms: (1) `raw` (`z`), (2) `temp` (`z / exp(log_T)`), (3) `temp_intercept` (`z / exp(log_T) + b`), each fitted on `source_dev_c` (using reused Stage 4 `s→s` DEV-C logits) and separately on `target_dev_c` (using newly scored `s→t` DEV-C logits for `s ≠ t`; for `s = t`, source and target DEV-C coincide and are labeled once).
  - Objective: equal-weight macro binary log-loss across the **14 fixed primary categories** on DEV-C known cells, using natural within-category prevalence: `L = (1/14) * sum_{k ∈ primary} mean_{i ∈ known(DEV-C, k)} [logaddexp(0, z'_{ik}) - y_{ik} z'_{ik}]`.
  - Optimizer: deterministic `scipy.optimize.minimize(..., method="BFGS", jac=True)` (exact analytic gradient, `gtol=1e-10`, `maxiter=500`, zero initialization `log_T = 0.0`, `b = 0.0`, float64). Log convergence, iterations, gradient norm, `T = exp(log_T)`, `b`, and DEV-C loss; non-convergence or non-finite parameters block publication.
  - Persist fitted calibrator parameters as tiny immutable artifacts (`phase2/calibrators/e17/<model>/<run_id>/`) reusable across SCREEN evaluation and downstream reporting.

### 1.7 E07 automatic original-caption coverage audit and stop conditions

Stage 1 executes an automatic, CPU-only audit (`audit_e07_captions`) against the accepted Stage 1 dataset artifact (`captions/{en,hi,bn,te}.jsonl`, `records.jsonl`, `targets.jsonl`, `splits.json`, `dataset.json`) with **zero image IO, zero GPU, zero model weights, and zero CONFIRM draw generation**.

1. **Deterministic selection and draw-0 verification:**
   - For every `(image_id, language)` in `SCREEN` (360 × 4 = 1,440 pairs), read all original captions from `captions/{lang}.jsonl` (preserving exact `caption_id` and unchanged UTF-8 `caption_text`).
   - Run `src.common.contracts.select_caption(candidates, draw_key)` for `draw_key ∈ {0, 1, 2}` (hash of `["caption-draw-v1", image_id, language, draw_key, caption_id]`, lexical `caption_id` tie-break).
   - **Hard assertion:** `draw_key = 0` must reproduce `records.jsonl` identically (`caption_id` and `caption_text` SHA-256) on all 1,440 SCREEN rows. Any mismatch blocks Stage 1 immediately.
   - Record a separate read-only multiplicity summary for `FIT` and `DEV-S` (metadata only, no extraction plan) to inform any later E07 refit decision. Exclude `CONFIRM` completely.
2. **Exact reuse and deduplication accounting (per language and draw `k ∈ {1, 2}`):**
   - Report distribution of `available_caption_count` (`m = 1, 2, >= 3`, min/median/max) and `distinct_text_count` (unique UTF-8 SHA-256 of `caption_text`).
   - Classify each of the 360 SCREEN rows into exactly one mutually exclusive bucket:
     - `same_id_as_draw0`: `caption_id` equals draw 0 → reuse draw-0 features & logits (0 GPU forwards).
     - `distinct_id_same_text_as_draw0`: different `caption_id`, byte-identical `caption_text` SHA-256 to draw 0 → reuse draw-0 float32 features & logits with explicit `caption_id` provenance alias (0 GPU forwards).
     - `changed_text_vs_draw0`: distinct `caption_text` SHA-256 from draw 0 → requires GPU extraction.
   - Deduplicate across draws 1 and 2 by `(image_id, language, caption_text_sha256)`: if draw 2 selects the exact same `caption_text` as draw 1, extract it **once** and alias it cleanly. Publish the exact unique changed-view count per language and total unique GPU forwards per model (`image_caption` only, capturing `last_caption` and `caption_mean` on the same midpoint forward).
3. **Support accounting (all 25 categories, preserving unknown `plant.n.02` on `6a20677eee04fd08`):**
   - Compute known positive/negative image counts on: (a) full 360-image SCREEN, (b) each per-language changed-only subset `S_changed(lang, draw_k)`, and (c) the simultaneous four-language changed intersection `⋂_lang S_changed(lang, draw_k)`.
4. **Automatic stop / downgrade verdict (mutually exclusive & exhaustive, evaluated before any GPU runtime opens):**
   - **`blocked` (HARD STOP — do not launch E07 GPU):**
     - Any draw-0 / target / split / CONFIRM integrity failure, OR full 360-image SCREEN fails to preserve all 14 primary categories at `min(pos, neg) >= 20`; **OR**
     - Fewer than **30** genuinely changed SCREEN images in ALL `(language, draw_k)` pairs (`max n_changed < 30`), OR no `(language, draw_k)` pair has at least **6 of the 14 primary categories** with `min(pos, neg) >= 10` on `S_changed(lang, draw_k)`; **OR**
     - Fewer than **3 of 4** languages achieve `max(n_changed_draw1, n_changed_draw2) >= 60` (two or more languages lack sufficient changed-view coverage → stop for coordinator decision rather than silently running a partial-language study).
   - **`passed_full` (FULL PROCEED):**
     - Not `blocked`, AND all four languages have `max(n_changed_draw1, n_changed_draw2) >= 100` (with both draws 1 and 2 audited and all unique changed views scheduled in 32-image SCREEN-only chunks), AND all 14 primary categories meet `min(pos, neg) >= 10` on each language's best-coverage `S_changed(lang, draw_k)`.
   - **`passed_restricted` (PROCEED WITH MANDATORY DUAL REPORTING):**
     - All remaining non-`blocked`, non-`passed_full` cases (at least 3 of 4 languages reach `>= 60` changed views, while either one language has `< 60` changed views — explicitly flagged `unavailable_single_caption` if `< 18` or `low_change_rate` if `18..59` — OR any language has `60..99` changed views OR fewer than 14 primary categories reach `min(pos, neg) >= 10` on `S_changed(lang, draw_k)`).
     - GPU runs only on the enumerated unique changed views; Stage 4 E07 must report BOTH full 360-image SCREEN (14-category primary macro with explicit unchanged-view fraction) AND the `restricted_changed_cohort` with its exact matched draw-0 companion.

### 1.8 Minimal reusable storage, checkpoint, and resume contracts

- **Distinct persistent namespace:** Keep `PHASE1_ROOT` (Colab `/content/drive/MyDrive/phase1`, configurable) strictly **read-only** for all Phase 1 paths (`datasets/`, `shared/`, `cache/`, `features/`, `checkpoints/`, `predictions/`, `results/`). Write all Phase 2 artifacts under `PHASE1_ROOT/phase2/`. Keeping `phase2/` under the same verified root allows `src.stage3.artifacts.Store` to authenticate relative `{path, sha256}` pointers to both Phase 1 inputs and Phase 2 outputs without duplicating Phase 1 files.
- **Outer marker & publication:** Reuse `src.common.storage.publish` (`stage1.v1` outer `_COMPLETE.json`, payloads written first, marker written last, non-pickle `.npy` / `.json` / `.jsonl` only, full 64-character hashes in manifests).

| Namespace under `PHASE1_ROOT/phase2/` | Immutable artifact kind & schema | Minimal reusable contents |
|---|---|---|
| `plans/<run_id>/` | `phase2_plan` (`phase2.plan.v1`) | Phase 1 bindings, midpoint route tables, frozen E29 gate thresholds, E17 DEV-C permission & block list, E07 audit reference, runtime/versions. |
| `audits/e07/<run_id>/` | `phase2_e07_audit` (`phase2.e07_audit.v1`) | Multiplicity/duplicate/changed-view tables, support tables, verdict (`passed_full` / `passed_restricted` / `blocked`), and unique changed-view manifest (if passed). |
| `features/e07/<model>/<run_id>/chunk_NNNN/` & `features/e07/<model>/releases/<run_id>/` | `phase2_e07_chunk` / `phase2_e07_release` (`phase2.e07_features.v1`) | Only genuinely changed SCREEN `image_caption` views at midpoint (`last_caption` + `caption_mean` float32 `.npy`, `rows.jsonl`, `token_evidence.jsonl`, `audits.json`, `chunk.json`). Never copy draw-0 features. |
| `predictions/e17_dev_c/<model>/<run_id>/` | `phase2_prediction_block` (`phase2.prediction.v1`) | 4 target-context DEV-C blocks/model (12 off-diagonal `s→t` float64 logits + `estimable` + `prevalence` + lineage). Never copy Stage 4 diagonals or SCREEN logits. |
| `calibrators/e17/<model>/<run_id>/` | `phase2_e17_calibrators` (`phase2.calibrators.v1`) | Fitted `T`, `b`, convergence, DEV-C objective, supervision label, and exact DEV-C input refs per route/form. |
| `predictions/e30_crossed/<model>/<run_id>/` | `phase2_prediction_block` (`phase2.prediction.v1`) | *(Conditional on E29)* 8 crossed SCREEN blocks/model (32 routes). Never copy matched arms. |
| `predictions/e11_adapted/<model>/<run_id>/` | `phase2_prediction_block` (`phase2.prediction.v1`) | *(Conditional on E29)* Adapted SCREEN logits (`mean_shift`, `mean_scale`, 12 directed `caption_mean` routes/model) + `transform_params.json` referencing reused Stage 3 FIT `statistics` (never copy FIT matrices). |
| `predictions/e07_screen/<model>/<run_id>/` | `phase2_prediction_block` (`phase2.prediction.v1`) | Newly scored logits for changed SCREEN views + explicit route index referencing Stage 4 draw-0 logits for unchanged views. |
| `results/<exp>/<model>/<run_id>/` & `results/e29_gates/<run_id>/` | `phase2_analysis_block` / `phase2_e29_gate` (`phase2.analysis.v1`) | Completed tidy metric/contrast tables, paired CIs, support/status, exact upstream refs, and machine-readable E29 gate receipts. |
| `releases/<run_id>/` | `phase2_release` (`phase2.release.v1`) | Stage 5 combined Phase 2 synthesis, decision register, and fresh-subprocess readback. |

- **Do NOT persist:** Per-draw bootstrap arrays (recompute/hold in RAM from Stage 4 `bootstrap.npy`), plotting tables, intermediate metric arrays, copied Phase 1 arrays/weights, mutable append logs, `latest` pointers, or pickled objects.
- **Concurrency, pre-initialization, and resume rules:**
  - Stage 1 `prepare_phase2` initializes all shared parent directories under `PHASE1_ROOT/phase2/` ONCE before any parallel Colab worker starts (flush/unmount/remount Drive after initialization if running across separate runtimes, matching the Stage 3 Drive lesson).
  - Strictly one producer per `<experiment>/<model>` namespace; unique `run_id` per attempt; never overwrite or delete prior failed/partial attempts.
  - Resume accepts ONLY explicit `{path, sha256}` completion references (or an explicit JSON index of them). Verify `phase2_plan_identity`, schema, upstream input refs, owner/model, and payload hashes before reuse; reject incompatible, foreign, partial, or corrupt checkpoints automatically.
  - Bound verified Phase 1 reads to once per worker session via hash-keyed `Store` caching; never rerun full Phase 1 Stage 2/3/4 validators as a prerequisite.
  - Stream JSON-line progress events (`unit`, `completed/total`, `read_hash_s`, `compute_s`, `write_s`, `peak_rss_bytes`) with `flush=True` and periodic heartbeats; surface actual child blockers directly rather than swallowing subprocess errors.

### 1.9 Parallel execution order (Stage 1 and immediate Stage 2/3 handoff)

1. **Stage 1 — `prepare_phase2.ipynb` (CPU, once, pinned Python 3.12 CPU env):**
   - **Step 1A (Common bind & pre-init):** Verify mounted `PHASE1_ROOT` and run a two-process `storage_probe` under `PHASE1_ROOT/phase2/probes/`; open and verify the accepted Stage 4 release (`open_release` + `load_bootstrap`), transitive Stage 3/G0/Stage 2/Stage 1 lineage, and midpoint route inventory; pre-create all `PHASE1_ROOT/phase2/` parent directories.
   - **Step 1B (E17 DEV-C preflight — overlaps or runs immediately after 1A):** Verify Stage 2 DEV-C midpoint feature releases and Stage 3 selected heads/scalers for both models, run the in-memory `en→en` DEV-C diagonal reproduction check against Stage 4 (`atol=0, rtol=0`), and freeze the 24-route E17 DEV-C scoring authorization.
   - **Step 1C (E07 caption audit — runs on CPU from the verified Stage 1 dataset manifest):** Execute `audit_e07_captions`, verify draw-0 equality, compute multiplicity/changed-view/support tables, emit `phase2_e07_audit` verdict (`passed_full` / `passed_restricted` / `blocked`) and (if passed) `e07_extraction_plan`.
   - Publish `phase2_plan` and flush/unmount Drive if launching separate Colab runtimes.
2. **Immediate parallel Stage 2 fan-out (once `phase2_plan` passes):**
   - **E29 CPU** (`MODEL='qwen'` and `MODEL='gemma'` on separate CPU runtimes or sequentially in minutes) starts immediately — does NOT wait for E07 GPU or E17.
   - **E17 CPU** (`MODEL='qwen'` and `MODEL='gemma'` on separate CPU runtimes) starts immediately after Step 1B passes — does NOT wait for E29 or E07.
   - **E07 L4 GPU** launches ONLY if Step 1C passes (`passed_full` or `passed_restricted`). Run `MODEL='qwen'` and `MODEL='gemma'` in parallel on two separate L4 runtimes (with `HF_TOKEN` secret enabled for Gemma), or queue them sequentially on one L4 while CPU jobs overlap.
3. **Conditional Stage 3 fan-out (immediately per model as E29 finishes):**
   - As soon as a model's `phase2_e29_gate` publishes, launch any `OPEN` branch (`E30` CPU and/or `E11` CPU) in parallel on separate CPU runtimes without waiting for E07 or E17 to complete.

### 1.10 Stage 1 acceptance criteria

Stage 1 is complete only when **all** items below pass and are recorded with explicit completion paths and SHA-256 hashes:

- [ ] **Clean baseline & frozen foundation:** Local doc edits committed cleanly; Coordinator Stage 1 foundation modules (`src/phase2/`), `scripts/phase2/prepare_phase2.ipynb`, and `tests/test_phase2_foundation.py` pass all local CPU unit/fixture tests (including negative tests for CONFIRM leakage, tampered hash, wrong plan identity, draw-0 mismatch, single-caption stop condition, E17 unauthorized split/target-head misuse, and E29 gate boundary cases).
- [ ] **Mounted storage & Phase 1 lineage verification (Colab CPU):** `prepare_phase2.ipynb` passes on mounted `PHASE1_ROOT` with pinned CPU versions (`numpy==2.1.3`, `scipy==1.18.1`, `scikit-learn==1.6.1`, `Pillow==11.3.0`, single BLAS/OpenMP thread), passes the two-process `storage_probe` under `PHASE1_ROOT/phase2/`, authenticates the exact Stage 4 release (`eb5cad38...`) and transitive Stage 3/G0 (`p=0.362`)/Stage 2/Stage 1 lineage without reopening raw images, and pre-initializes all `PHASE1_ROOT/phase2/` parent directories.
- [ ] **Frozen contracts in `phase2_plan`:** Published `phase2_plan` `_COMPLETE.json` + SHA-256 binds all 16 midpoint SCREEN blocks/model for E29, the exact frozen Stage 4 `(2000, 360)` bootstrap identity, the predeclared E29 → E30/E11 thresholds and Stage 5 decision ranges, and strict CONFIRM exclusion.
- [ ] **E17 DEV-C preflight passed:** All 8 Stage 2 DEV-C midpoint feature releases and 8 Stage 3 midpoint heads/scalers authenticate cleanly; in-memory `en→en` DEV-C diagonal scoring matches Stage 4 accepted DEV-C logits exactly (`atol=0, rtol=0`) for both Qwen and Gemma; the 24-route E17 DEV-C authorization is published.
- [ ] **E07 caption coverage audit completed:** `phase2_e07_audit` `_COMPLETE.json` + SHA-256 passes draw-0 identity assertion on all 1,440 SCREEN pairs, records exact multiplicity/duplicate/changed-view and full vs changed-only support tables for draws 1 and 2 across all four languages, emits an unambiguous verdict (`passed_full`, `passed_restricted`, or `blocked`), and (if passed) publishes the deduplicated unique changed-view extraction manifest (0 CONFIRM rows).
- [ ] **No Phase 1 mutation and no premature outcomes:** Zero writes under Phase 1 namespaces, zero CONFIRM access, and zero Phase 2 SCREEN metrics or GPU extractions executed during Stage 1.

### 1.11 Next coding chats: file ownership and ready-to-paste prompts

Preserve and commit the current local edits (`phase1.md`, `prelim.md`, `phase2.md`, `phase2_handoff.md`) first. Then execute **Chat 1 (Coordinator)** to publish and commit the frozen Phase 2 Stage 1 foundation SHA (`BASE_SHA`). **Chats 2, 3, and 4** branch from that exact immutable `BASE_SHA` in parallel worktrees/branches with strictly disjoint file ownership. Conditional E30 and E11 worker chats will be scoped only after E29's Stage 2 gate receipts decide which models qualify.

| Chat | Branch / base | Owned files (exclusive) | Deliverable |
|---|---|---|---|
| **Chat 1: Coordinator (Stage 1 foundation)** | New branch from clean docs commit on `phase1` | `src/phase2/{__init__,schema,plan,inputs,e07_audit,gates,operations,interfaces}.py`, `scripts/phase2/prepare_phase2.ipynb`, `tests/test_phase2_foundation.py`, `docs/phase2_stage1_interfaces.md`, updates to `phase2.md` | Shared Phase 2 Stage 1 schemas, Phase 1 read-only binder, E17 DEV-C preflight, E07 caption audit, E29 gate evaluator, `prepare_phase2.ipynb`, fixture tests, and frozen worker `BASE_SHA`. |
| **Chat 2: E29 worker** | Branch from `BASE_SHA` | `src/phase2/run_e29.py`, `scripts/phase2/run_e29.ipynb`, `tests/test_phase2_e29.py` | CPU paired pooling gap contrast (`image_caption` primary + `caption_only` diagnostic), draw-0 midpoint reference, and E29 → E30/E11 gate publisher (one `MODEL` selector). |
| **Chat 3: E17 worker** | Branch from `BASE_SHA` | `src/phase2/run_e17.py`, `scripts/phase2/run_e17.ipynb`, `tests/test_phase2_e17.py` | CPU E17 DEV-C 12-route off-diagonal scorer (with diagonal self-check), source/target DEV-C `T` and `T+b` calibrator fitter, and SCREEN calibration evaluator (one `MODEL` selector). |
| **Chat 4: E07 worker** | Branch from `BASE_SHA` | `src/phase2/extract_e07.py`, `src/phase2/score_e07.py`, `scripts/phase2/extract_e07.ipynb`, `scripts/phase2/score_e07.ipynb`, `tests/test_phase2_e07.py` | L4 GPU extractor for audited unique changed SCREEN views (reusing `extract_with_evidence`, both poolings same forward, one `MODEL` selector, `HF_TOKEN` safe for Gemma) + CPU frozen-head scorer/comparator reusing draw-0 logits. Production GPU run gated on Stage 1 audit verdict. |

#### Ready-to-paste prompts for the next coding chats

> **Prompt — Chat 1 (Coordinator: Phase 2 Stage 1 foundation)**
> Work in `C:/Users/APOORV/Documents/Codex/2026-11-12/ex/work/Research-stage1-integrated`. Read `phase2.md` (§1 Stage 1 finalized specification), `phase2_handoff.md`, `prelim.md` §12, `docs/stage4_interfaces.md`, and `src/stage4/` + `src/stage5/interfaces.py`. First verify actual Git and Python paths on Windows and check `git status` so local doc edits are preserved cleanly. Implement ONLY the Coordinator-owned Stage 1 foundation: `src/phase2/{__init__,schema,plan,inputs,e07_audit,gates,operations,interfaces}.py`, `scripts/phase2/prepare_phase2.ipynb`, `tests/test_phase2_foundation.py`, and `docs/phase2_stage1_interfaces.md`. Reuse `src.stage5.interfaces.open_release`, `src.stage4.interfaces`, `src.stage3.data._read_verified_group`, `src.stage4.predictions._selected_parameters`, `src.common.contracts.select_caption`, and `src.common.storage.publish` without modifying any Phase 1 code or artifacts. Implement the `phase2_plan` binder, `storage_probe`, `PHASE1_ROOT/phase2/` parent pre-initialization, E17 DEV-C preflight (including `en→en` diagonal `atol=0, rtol=0` self-check), `audit_e07_captions` with exact stop/downgrade verdicts and deduplicated changed-view manifest, and the frozen E29 gate evaluator. Keep CONFIRM strictly rejected before IO. Run the full Stage 1 fixture test suite and report branch, commit (`BASE_SHA`), owned files, tests passed, and mounted Colab prerequisites separately from local software checks.

> **Prompt — Chat 2 (E29 worker — start after Coordinator publishes `BASE_SHA`)**
> Branch a clean worktree from `<BASE_SHA>` in `Drazr/Research`. Read `phase2.md` §1.3–1.5, §1.8–1.9 and `docs/phase2_stage1_interfaces.md`. You own ONLY `src/phase2/run_e29.py`, `scripts/phase2/run_e29.ipynb`, and `tests/test_phase2_e29.py`. Do not edit Coordinator-owned `src/phase2/` shared modules or Phase 1 files. Implement the CPU E29 paired pooling gap contrast (`MODEL` selector: `qwen` or `gemma`) consuming the 16 bound Stage 4 SCREEN midpoint blocks per model and the frozen Stage 4 `(2000, 360)` bootstrap matrix. Compute `G(last_caption) - G(caption_mean)` and residual `G(caption_mean)` within every draw across the 14 fixed primary categories and equally weighted 12 off-diagonal routes (plus per-route rows, receiving diagonal gains, `caption_only` diagnostic, and AP/Brier/NLL companions). Publish the reusable draw-0 midpoint reference and `phase2_e29_gate` receipt using the Coordinator's `gates` interface, with explicit-reference resume and streamed JSON progress. Add unit/fixture tests covering paired draw arithmetic, NaN/mask handling, gate OPEN/PARK/UNRESOLVED cases, and resume rejection. Report branch, commit, files, tests, and Colab run prerequisites.

> **Prompt — Chat 3 (E17 worker — start after Coordinator publishes `BASE_SHA`)**
> Branch a clean worktree from `<BASE_SHA>` in `Drazr/Research`. Read `phase2.md` §1.3–1.6, §1.8–1.9 and `docs/phase2_stage1_interfaces.md`. You own ONLY `src/phase2/run_e17.py`, `scripts/phase2/run_e17.ipynb`, and `tests/test_phase2_e17.py`. Do not edit Coordinator-owned `src/phase2/` shared modules or Phase 1 files. Implement the CPU E17 worker (`MODEL` selector: `qwen` or `gemma`) that: (1) verifies the Stage 1 `phase2.e17_dev_c_transfer` permission and asserts exact `en→en` DEV-C diagonal reproduction against Stage 4; (2) scores and persists the 12 missing off-diagonal frozen-source-head DEV-C routes once per model (4 target-context blocks × 3 source routes) from verified Stage 2 DEV-C features and Stage 3 selected heads/scalers; (3) fits global float64 `temp` and `temp_intercept` calibrators (equal-weight 14-primary-category macro BCE, deterministic BFGS with analytic gradient) on source DEV-C (reused Stage 4 diagonals) and separately on target DEV-C (frozen source head on target DEV-C, never target-trained heads), persisting tiny immutable calibrator artifacts; and (4) evaluates all arms on reused Stage 4 SCREEN logits against raw and FIT-prior references with paired bootstrap intervals, enforcing per-category AUROC/AP invariance (`<= 1e-12`) and separate 0.5-decision reporting. Keep CONFIRM closed and E19 cutoffs untouched. Add fixture tests and report branch, commit, files, tests, and Colab prerequisites.

> **Prompt — Chat 4 (E07 worker — start after Coordinator publishes `BASE_SHA`; GPU production gated on Stage 1 audit verdict)**
> Branch a clean worktree from `<BASE_SHA>` in `Drazr/Research`. Read `phase2.md` §1.3–1.4, §1.7–1.9, `docs/stage2_interfaces.md`, and `docs/phase2_stage1_interfaces.md`. You own ONLY `src/phase2/extract_e07.py`, `src/phase2/score_e07.py`, `scripts/phase2/extract_e07.ipynb`, `scripts/phase2/score_e07.ipynb`, and `tests/test_phase2_e07.py`. Do not edit Coordinator-owned `src/phase2/` shared modules or Phase 1 files. Implement: (1) the L4 GPU extractor (`MODEL` selector: `qwen` or `gemma`, secret-safe `HF_TOKEN` for Gemma, `assert_runtime` execution check) that refuses to run unless `phase2_e07_audit` verdict is `passed_full` or `passed_restricted`, extracts ONLY the audited deduplicated genuinely-changed SCREEN `image_caption` views at fixed midpoint collecting `last_caption` and `caption_mean` on the same forward via `src.stage2.evidence.extract_with_evidence` (batch 1, `bfloat16`, `eager`, no truncation, 32-image chunks, first/last repeat checks, explicit-ref resume); and (2) the CPU frozen-head scorer/comparator that scores completed changed-view chunks with the 8 accepted Stage 3 midpoint heads/scalers, reuses Stage 4 draw-0 logits for unchanged/byte-identical views (asserting exact draw-0 parity), and reports full 360-image SCREEN plus matched draw-0 `restricted_changed_cohort` paired contrasts using the frozen Stage 4 bootstrap. Keep CONFIRM closed and never refit heads. Add fake-adapter GPU plumbing, byte-identical reuse, and restricted-cohort fixture tests, and report branch, commit, files, tests, and Colab prerequisites separately from production GPU checks.

### 1.12 Stage 1 software integration status — 19 November 2026

All Stage 1 modules (`src/phase2/{__init__,schema,inputs,interfaces,e07_audit,e17_auth,gates,plan,operations}.py`), Colab CPU notebooks (`scripts/phase2/{prepare_phase2,audit_e07,preflight_e17}.ipynb`), interface doc (`docs/phase2_stage1_interfaces.md`), and all five test suites (`tests/test_phase2_{foundation,e07_audit,e17_auth,gates_plan,stage1_integration}.py`) are integrated locally. Local stdlib/mocked fixture checks pass; the three `numpy`/`scipy`/`scikit-learn` end-to-end Phase 1 fixture tests and actual mounted Colab execution remain pending until committed/pushed and run in the pinned Python 3.12 CPU runtime.

**Colab CPU Stage 1 execution order (after pushing the integrated commit and pinning `CODE_REF`):**
1. Run `scripts/phase2/prepare_phase2.ipynb` once (`E07_AUDIT_REF = None`, `E17_AUTH_REF = None`) to verify mounted `PHASE1_ROOT`, run `storage_probe`, pre-initialize `PHASE1_ROOT/phase2/`, and publish `phase2_plan`. Call `drive.flush_and_unmount()` and remount if launching separate runtimes.
2. Run `scripts/phase2/audit_e07.ipynb` and `scripts/phase2/preflight_e17.ipynb` (`MODEL = 'both'`) in parallel on two CPU runtimes (or sequentially). Record the printed `phase2_e07_audit` and `phase2_e17_auth` `{path, sha256}` references.
3. Run `prepare_phase2.ipynb` with `E07_AUDIT_REF`, `E17_AUTH_REF`, and `RUN_VERIFY_STAGE1_BUNDLE = True` (or run `python -u -m src.phase2.operations verify-stage1-bundle`) to certify Stage 1 completion before launching Stage 2.

---

## Stage 2: finalized specification — 19 November 2026

Planning freeze `phase2-stage2.v1`. This section finalizes **Phase 2 Stage 2 only**: executing **E29 CPU** (paired pooling gap contrast + reusable draw-0 reference + E30/E11 gate publication), **E17 CPU** (off-diagonal DEV-C scoring, source vs target DEV-C calibrator fitting, and SCREEN calibration evaluation), and **E07 L4 GPU** (audited unique changed-view SCREEN midpoint extraction, gated on Stage 1's `phase2_e07_audit` verdict). Conditional Stage 3 (E30, E11), Stage 4 E07 CPU scoring (`score_e07`), and Stage 5 synthesis remain overview-only until finalized in turn.

### 2.1 Entry prerequisites and immutable Stage 1 bindings

Stage 2 workers may start only against a certified Stage 1 bundle (`verify-stage1-bundle` passed on mounted Colab CPU):

- **Required Stage 1 completion references (`{path, sha256}`):**
  - `phase2_plan` (`KIND_PLAN`, `SCHEMA_PLAN`): binds Phase 1 Stage 4 release (`eb5cad38...`), Stage 4 `(2000, 360)` bootstrap, midpoints (`qwen` depth 18, `gemma` depth 17), `e29_blocks`, `gate_thresholds`, `stage5_decision_ranges`, and `plan_identity`.
  - `phase2_e17_auth` (`KIND_E17_AUTH`, `SCHEMA_E17_AUTH`): required for **E17**; certifies `permission == "phase2.e17_dev_c_transfer"` and `en→en` DEV-C diagonal reproduction (`atol=0, rtol=0`) for both models.
  - `phase2_e07_audit` (`KIND_E07_AUDIT`, `SCHEMA_E07_AUDIT`): required for **E07 GPU**; must have `draw0_verified == True`, `confirm_excluded == True`, and `verdict ∈ {"passed_full", "passed_restricted"}` (if `verdict == "blocked"`, E07 GPU is skipped and Stage 2 completes with E29 + E17 only).
- **Storage & concurrency invariants:**
  - All Phase 1 paths remain strictly read-only. Every Stage 2 write stays under `PHASE1_ROOT/phase2/` via `src.phase2.interfaces.publish_phase2` (`stage1.v1` `_COMPLETE.json` written last).
  - Flush and unmount/remount Google Drive after Stage 1 before launching parallel Stage 2 Colab runtimes so pre-initialized `PHASE2_PARENT_DIRS` and Stage 1 manifests are visible across runtimes.
  - Strictly one producer per `<experiment>/<model>` namespace; every worker CLI and notebook accepts one explicit `MODEL` selector (`"qwen"` or `"gemma"`).
  - `CONFIRM` remains closed and rejected before any IO via `require_open_split`.
  - All workers emit streamed JSON-line progress events (`event`, `unit`, `completed`, `total`, `read_hash_s`, `compute_s`, `write_s`, `peak_rss_bytes`) with `flush=True` and surface child stderr/blockers directly.

### 2.2 Frozen Stage 2 wire schemas (cross-worker and Stage 3/4 handoff)

All Stage 2 workers import Stage 1 shared contracts exclusively from `src.phase2.interfaces` (plus `src.phase2.gates`, `src.phase2.e17_auth`, `src.phase2.e07_audit`, and `src.stage4.metrics`). No Stage 2 worker edits Coordinator-owned Stage 1 files or Phase 1 files. Downstream Stage 3/4/5 consumers validate Stage 2 outputs purely through these frozen schemas:

| Artifact & namespace | `kind` / `body["schema"]` | Required top-level `body` keys & arrays |
|---|---|---|
| **E29 analysis + draw-0 ref**<br>`phase2/results/e29/<model>/<run_id>/` | `phase2_analysis_block`<br>`phase2.analysis.v1` | `schema`, `experiment` (`"E29"`), `plan`, `plan_identity`, `model`, `midpoint_depth`, `bootstrap` (`shape`, `sha256`), `confirm_excluded` (`True`), `input_blocks` (16 Stage 4 SCREEN refs), `image_caption` (12-route means, 12 per-route rows, 4 receiving diagonals, 25-category & 14-primary macro tables for AUROC, AP, Brier, NLL, 0.5-threshold metrics, paired `ΔG` and `G_res` CIs + `valid_draws`), `caption_only` (identical diagnostic structure), `draw0_reference` (compact reusable draw-0 midpoint point estimates + block refs + target/mask fingerprint for Stage 4 `score_e07`), `gate_ref` (`{path, sha256}` of published `phase2_e29_gate`), `supervision_labels`. |
| **E29 gate receipt**<br>`phase2/results/e29_gates/<run_id>/` | `phase2_e29_gate`<br>`phase2.e29_gate.v1` | Published via `src.phase2.gates.publish_e29_gate` (frozen Stage 1 schema: `decisions["E30"]`, `decisions["E11"]`, `observed`, `thresholds`). |
| **E17 DEV-C off-diagonal block**<br>`phase2/predictions/e17_dev_c/<model>/<run_id>/` | `phase2_prediction_block`<br>`phase2.prediction.v1` | `schema`, `experiment` (`"E17"`), `plan`, `plan_identity`, `e17_auth_ref`, `model`, `midpoint_depth`, `condition` (`"image_caption"`), `pooling` (`"last_caption"`), `split` (`"DEV-C"`), `target_language`, `source_languages` (3 sorted `s ≠ t`), `image_ids_sha256`, `n_images` (`180`), `n_categories` (`25`), `category_order`, `diagonal_self_check` (`en→en`, `passed: True`, `max_abs_diff: 0.0`), `routes` (per-source `route_id`, `source_language`, `row_index`, `source_system_ref`, `supervision_label: "source_only_transfer"`), `upstream_feature_ref`, `confirm_excluded` (`True`). Arrays: `logits.npy` (`float64`, shape `(3, 180, 25)`), `estimable.npy` (`bool`, shape `(3, 25)`), `prevalence.npy` (`float64`, shape `(3, 25)`). |
| **E17 fitted calibrators**<br>`phase2/calibrators/e17/<model>/<run_id>/` | `phase2_e17_calibrators`<br>`phase2.calibrators.v1` | `schema`, `experiment` (`"E17"`), `plan`, `plan_identity`, `e17_auth_ref`, `model`, `midpoint_depth`, `objective` (`"equal_weight_14_primary_macro_bce"`), `optimizer` (`"BFGS"`, `jac: True`, `gtol: 1e-10`, `maxiter: 500`, `init: {"log_T": 0.0, "b": 0.0}`), `source_dev_c` (4 source languages × `{temp, temp_intercept}`: `log_T`, `T`, `b`, `converged`, `nit`, `grad_norm`, `dev_c_loss_raw`, `dev_c_loss_cal`, `dev_c_input_ref`, `supervision_label`), `target_dev_c` (12 off-diagonal `s→t` routes × `{temp, temp_intercept}` with identical fields + `dev_c_block_ref`, `supervision_label: "target_dev_c_supervised_calibration"`), `confirm_excluded` (`True`). |
| **E17 SCREEN analysis**<br>`phase2/results/e17/<model>/<run_id>/` | `phase2_analysis_block`<br>`phase2.analysis.v1` | `schema`, `experiment` (`"E17"`), `plan`, `plan_identity`, `e17_auth_ref`, `calibrators_ref`, `dev_c_block_refs` (4 target-context refs), `model`, `midpoint_depth`, `bootstrap` (`shape`, `sha256`), `arms` (`raw`, `source_dev_c:temp`, `source_dev_c:temp_intercept`, `target_dev_c:temp`, `target_dev_c:temp_intercept`), `fit_prior_reference`, `summaries` (12-route mean + 4-diagonal mean + 16 per-route rows + 25-category & 14-primary macro tables for Brier, NLL, AUROC, AP, paired gains vs `raw` and vs `fit_prior` with 95% CIs and `valid_draws`), `decision_0p5` (separate 0.5-threshold sensitivity/specificity/F1/balanced accuracy + flip rates), `invariance_checks` (`max_abs_auroc_diff <= 1e-12`, `max_abs_ap_diff <= 1e-12`, `all_T_positive: True`, `temp_only_0p5_flips: 0`), `confirm_excluded` (`True`). |
| **E07 changed-view chunk**<br>`phase2/features/e07/<model>/<run_id>/chunk_NNNN/` | `phase2_e07_chunk`<br>`phase2.e07_features.v1` | `schema`, `experiment` (`"E07"`), `plan`, `plan_identity`, `e07_audit_ref`, `model`, `midpoint_depth`, `condition` (`"image_caption"`), `poolings` (`["last_caption", "caption_mean"]`), `split` (`"SCREEN"`), `chunk_index`, `total_chunks`, `n_views`, `dimension`, `model_fingerprint`, `execution_fingerprint`, `site_specs`, `evidence_bridge_sha256`, `repeat_check` (`first_index`, `last_index`, `max_abs_diff_last_caption`, `max_abs_diff_caption_mean`, `passed`), `confirm_excluded` (`True`). Arrays: `last_caption.npy` (`float32`, `(n_views, d)`), `caption_mean.npy` (`float32`, `(n_views, d)`). Files: `rows.jsonl`, `token_evidence.jsonl`, `audits.json`. |
| **E07 feature release**<br>`phase2/features/e07/<model>/releases/<run_id>/` | `phase2_e07_release`<br>`phase2.e07_features.v1` | `schema`, `experiment` (`"E07"`), `plan`, `plan_identity`, `e07_audit_ref`, `e07_verdict`, `model`, `midpoint_depth`, `condition` (`"image_caption"`), `poolings` (`["last_caption", "caption_mean"]`), `split` (`"SCREEN"`), `total_unique_views`, `per_language_counts`, `dimension`, `model_fingerprint`, `execution_fingerprint`, `chunks` (ordered list of `{chunk_index, n_views, ref, rows_sha256}`), `repeat_check_summary`, `confirm_excluded` (`True`). |

### 2.3 Branch A — E29 CPU: paired pooling gap contrast and gate publication

- **Owned files:** `src/phase2/run_e29.py`, `scripts/phase2/run_e29.ipynb`, `tests/test_phase2_e29.py`.
- **Runtime:** Colab CPU (pinned `requirements-stage2-cpu.txt`, single BLAS/OpenMP thread), one runtime per `MODEL` (`qwen`, `gemma`), ~1–2 minutes each.
- **Inputs (zero new feature reads or scoring):**
  - `phase2_plan` ref (validated via `load_phase2_plan`) + `MODEL`.
  - 16 bound Stage 4 SCREEN midpoint prediction blocks per model (`e29_screen_blocks`: `image_caption` and `caption_only` × `last_caption` and `caption_mean` × 4 target languages).
  - Frozen Stage 4 `(2000, 360)` `int64` bootstrap draw matrix (`load_bootstrap`) and verified SCREEN targets/unknown-cell mask (`6a20677eee04fd08` `plant.n.02 = null` preserved).
- **Computation rules:**
  - Verify identical sorted 360-image `screen_ids` and exact `paired_image_alignment` across all 16 blocks before computing any metric.
  - Reuse `src.stage4.metrics` (`probabilities`, `binary_metrics`, `bootstrap_counts`, `bootstrap_binary_metrics`, `fixed_macro`, `bootstrap_interval`, `support_status`).
  - For each condition (`image_caption` primary, `caption_only` diagnostic), compute within **every** draw `b ∈ [0, 1999]`:
    - Per-category and 14-primary-macro metrics (AUROC, AP, Brier, NLL, 0.5-threshold sensitivity/specificity/F1/balanced accuracy) for all 16 routes at both `last_caption` and `caption_mean`.
    - Route transfer gap `G(s→t, p, b)` (`M(t→t) - M(s→t)` for rank/0.5 metrics; `M(s→t) - M(t→t)` for Brier/NLL).
    - Route pooling gap reduction `ΔG(s→t, b) = G(s→t, last_caption, b) - G(s→t, caption_mean, b)`.
    - Receiving diagonal gain `ΔD(t→t, b) = M(t→t, caption_mean, b) - M(t→t, last_caption, b)` (reversed sign for losses so positive is improvement).
    - Equal-weight 12-route means `ΔG_mean(b)` and `G_res_mean(b)`.
  - Take linear 2.5th/97.5th percentiles only after within-draw aggregation; record `valid_draws / 2000` everywhere and preserve undefined draws as `NaN` (`null` in JSON when `valid_draws == 0`). Do NOT persist per-draw bootstrap arrays.
  - Evaluate frozen gates via `src.phase2.gates.evaluate_e29_gates` on `image_caption` primary AUROC estimands and publish `phase2_e29_gate` via `publish_e29_gate`, then publish `phase2_analysis_block` referencing `gate_ref` and embedding `draw0_reference` for Stage 4 `score_e07`.
- **Resume:** Accepts explicit `resume_analysis_ref` and/or `resume_gate_ref`, validated via `verify_resume_ref` against `plan_identity`, `model`, and all 16 input block hashes.

### 2.4 Branch B — E17 CPU: off-diagonal DEV-C scoring, calibration fitting, and SCREEN evaluation

- **Owned files:** `src/phase2/run_e17.py`, `scripts/phase2/run_e17.ipynb`, `tests/test_phase2_e17.py`.
- **Runtime:** Colab CPU (pinned `requirements-stage2-cpu.txt`, single BLAS/OpenMP thread), one runtime per `MODEL` (`qwen`, `gemma`), ~2–4 minutes each. Runs end-to-end on CPU in Stage 2 (DEV-C scoring + calibrator fitting + SCREEN evaluation) so no separate Stage 4 E17 runtime is needed.
- **Step 1 — Authorization & diagonal reproduction self-check:**
  - Verify `phase2_plan` and `phase2_e17_auth` (`permission == "phase2.e17_dev_c_transfer"`, matching `lineage_identity`, `confirm_excluded == True`).
  - Before writing any output, run the in-memory `en→en` DEV-C diagonal self-check for `MODEL` against Stage 4's accepted `en→en` DEV-C logits and assert `max_abs_diff == 0.0` (`atol=0, rtol=0`).
- **Step 2 — Score and persist the 12 missing off-diagonal DEV-C routes (once per model):**
  - For each target language `t ∈ ('en', 'hi', 'bn', 'te')`, load verified Stage 2 midpoint `image_caption` / `last_caption` DEV-C features once via `read_verified_midpoint_features` (180 images, `require_open_split("DEV-C")`).
  - Load the 3 non-diagonal frozen Stage 3 source systems `s ≠ t` via `load_frozen_source_head` (no FIT matrix reload) and compute exact float64 logits `((X.astype(float64) - mean) / scale) @ coef.T + intercept` (matching Stage 4's scorer identically).
  - Assert finite logits on all estimable columns and exact 180-image DEV-C row alignment. Publish 4 target-context blocks under `phase2/predictions/e17_dev_c/<model>/<run_id>/`. Never duplicate Stage 4 diagonals or SCREEN logits.
- **Step 3 — Fit global float64 calibrators on DEV-C:**
  - Forms: `temp` (`z' = z / exp(log_T)`) and `temp_intercept` (`z' = z / exp(log_T) + b`).
  - Objective on DEV-C known cells across the **14 fixed primary categories**:
    `L = (1/14) * sum_{k ∈ primary} mean_{i ∈ known(DEV-C, k)} [logaddexp(0, z'_{ik}) - y_{ik} z'_{ik}]` with exact analytic gradient (`jac=True`).
  - Optimizer: `scipy.optimize.minimize(..., method="BFGS", jac=True, options={"gtol": 1e-10, "maxiter": 500})` initialized at `log_T = 0.0`, `b = 0.0`. Hard-fail publication if `not res.success`, non-finite parameters/loss, or `T = exp(log_T) <= 0`.
  - Fit **two strictly separated scopes**:
    - `source_dev_c` (4 fits/form): fit on reused Stage 4 `s→s` DEV-C logits; applied to all 4 targets `s→t` (`supervision_label = "source_dev_c_calibrated"` for `s ≠ t`; `"within_context_source_fit"` noted for `s == t`).
    - `target_dev_c` (12 fits/form): fit on the newly scored frozen source `s` head on target `t` DEV-C features (`s ≠ t`, `supervision_label = "target_dev_c_supervised_calibration"`). Never use target `t`'s trained head to fit a route calibrator; for `s == t`, source and target DEV-C coincide and are labeled once.
  - Persist under `phase2/calibrators/e17/<model>/<run_id>/`.
- **Step 4 — Evaluate all arms on reused Stage 4 SCREEN logits:**
  - Load the 4 reused Stage 4 midpoint `image_caption` / `last_caption` SCREEN blocks (16 routes) and frozen `(2000, 360)` bootstrap.
  - Evaluate `raw`, `source_dev_c:temp`, `source_dev_c:temp_intercept`, `target_dev_c:temp`, and `target_dev_c:temp_intercept` on identical SCREEN known cells (preserving unknown `plant.n.02` on `6a20677eee04fd08`).
  - Compute paired bootstrap Brier and NLL improvements vs `raw` and vs fixed FIT-prior prevalence reference (14-primary macro, 12-route mean, 4-diagonal mean, per-route, and 25-category tables).
  - **Mandatory automatic checks (hard-fail if violated):**
    - Per-category SCREEN AUROC and AP for every calibrated arm match `raw` within `<= 1e-12` (up to exact ties under strictly monotonic positive scaling/shift).
    - Every fitted `T > 0`.
    - `temp`-only arms produce `0` decision flips at threshold 0.5 relative to `raw` (for non-zero logits; `z == 0` stays at `p == 0.5`).
  - Report 0.5-threshold sensitivity/specificity/F1/balanced accuracy and `temp_intercept` flip rates in a separate `decision_0p5` section. Keep Stage 4 E19 cutoffs untouched.
- **Resume:** Supports explicit resume refs at DEV-C block, calibrator, and final analysis granularity via `verify_resume_ref`.

### 2.5 Branch C — E07 L4 GPU: audited unique changed-view midpoint extraction

- **Owned files:** `src/phase2/extract_e07.py`, `scripts/phase2/extract_e07.ipynb`, `tests/test_phase2_e07_extract.py`. *(Stage 4 CPU scoring `score_e07.py` / `score_e07.ipynb` will be finalized in Stage 4 against the frozen `phase2_e07_release` and E29 `draw0_reference` schemas.)*
- **Runtime:** Colab L4 GPU (pinned `requirements-l4.txt`), one runtime per `MODEL` (`qwen`, `gemma`). `HF_TOKEN` read strictly from Colab Secrets (`google.colab.userdata`) or environment for Gemma and never logged or written to disk.
- **Mandatory pre-GPU gate:**
  - Verify `phase2_plan` and `phase2_e07_audit` **before** loading model weights or opening any image.
  - Hard-refuse if `verdict == "blocked"`, `extraction_plan is None`, `extraction_plan["split"] != "SCREEN"`, or `extraction_plan["confirm_rows"] != 0`.
  - Run `assert_runtime` matching the bound Stage 2 `model_readiness` execution contract (`bfloat16`, `eager`, CUDA L4).
- **Extraction contract:**
  - Instantiate `HFAdapter` strictly from the bound Stage 2 `model_plan` + `models.json` pinned revision (`qwen`: `66285546d2b821cf421d4f5eb2576359d3770cd3`, depth **18**; `gemma`: `093f9f388b31de276ce2de164bdc2081324b9767`, depth **17**). Assert actual `model_fingerprint`, `execution_fingerprint`, `dimension`, and `site_specs` match Stage 2 `model_readiness`.
  - Extract ONLY the deduplicated unique genuinely-changed SCREEN `image_caption` views in `extraction_plan["chunks"]` (32 views/chunk). Never forward unchanged draw-0 views or `distinct_id_same_text_as_draw0` views.
  - Construct `ExtractionRequest` (`condition="image_caption"`, batch 1, no truncation, `max_input_tokens=1024`) requesting BOTH `last_caption` and `caption_mean` midpoint sites on the **same single forward** via `src.stage2.evidence.extract_with_evidence`.
  - Verify pinned SCREEN image SHA-256 against Stage 1 dataset manifest before opening each image; verify caption UTF-8 SHA-256 matches `extraction_plan`.
  - **Per-chunk repeat determinism check:** Re-extract the first and last view of every chunk (`index 0` and `index n-1`) through `extract_with_evidence` and record `max_abs_diff` for both poolings in `audits.json` (hard-fail if non-finite or exceeds Stage 2's repeat tolerance; `0.0` expected on deterministic L4 `eager`).
- **Chunk publication, explicit-ref resume, and release:**
  - Publish each completed 32-view chunk immediately to `phase2/features/e07/<model>/<run_id>/chunk_NNNN/` (`KIND_E07_CHUNK`, `SCHEMA_E07_FEATURES`) with `last_caption.npy`, `caption_mean.npy` (float32), `rows.jsonl`, `token_evidence.jsonl`, and `audits.json`.
  - Resume accepts an explicit mapping/index of completed `{chunk_index: {path, sha256}}` refs; verifies `plan_identity`, `e07_audit_ref`, `model`, `chunk_index`, expected view hashes, and `.npy` hashes before skipping a chunk.
  - Once all chunks complete, publish `phase2/features/e07/<model>/releases/<run_id>/` (`KIND_E07_RELEASE`, `SCHEMA_E07_FEATURES`).
- **Local tests (`tests/test_phase2_e07_extract.py`):** Use an explicit `HFAdapter` fixture subclass (matching Stage 2's `evidence.py` bridge check) to verify blocked-audit refusal, CONFIRM rejection, dual-pooling same-forward extraction, byte-identical exclusion, repeat-check failure detection, chunk resume verification, and release manifest integrity without requiring a GPU.

### 2.6 Stage 2 parallel execution order and expected Colab timing

Once Stage 1's `verify-stage1-bundle` passes and Drive is flushed/remounted, **all 6 Stage 2 runs are mutually independent** and can overlap on separate Colab runtimes:

| Notebook | Runtime | `MODEL` | Dependency | Parallelism | Expected time |
|---|---|---|---|---|---|
| `scripts/phase2/run_e29.ipynb` | Colab CPU | `qwen` / `gemma` | Stage 1 `phase2_plan` | Run both models in parallel (or back-to-back) | ~1–2 min / model |
| `scripts/phase2/run_e17.ipynb` | Colab CPU | `qwen` / `gemma` | Stage 1 `phase2_plan` + `phase2_e17_auth` | Run both models in parallel with E29 & E07 | ~2–4 min / model |
| `scripts/phase2/extract_e07.ipynb` | Colab L4 GPU | `qwen` / `gemma` (`HF_TOKEN` for Gemma) | Stage 1 `phase2_plan` + `phase2_e07_audit` (`verdict != "blocked"`) | Run both models in parallel on 2 L4s, or sequentially on 1 L4 | Scales with unique changed views (~15–35 min / model for ~1,000–2,000 views at batch 1) |

As soon as either model's `run_e29.ipynb` publishes its `phase2_e29_gate`, Stage 3 planning/execution for any `OPEN` branch (`E30` and/or `E11`) can begin immediately without waiting for E07 or E17.

### 2.7 Stage 2 acceptance criteria

Stage 2 is complete only when **all** items below pass and are recorded with explicit `{path, sha256}` completion references:

- [ ] **Local unit/fixture tests pass:** `tests/test_phase2_e29.py`, `tests/test_phase2_e17.py`, `tests/test_phase2_e07_extract.py`, and `tests/test_phase2_stage2_integration.py` pass (covering paired within-draw arithmetic, unknown-cell mask preservation, E29 gate boundary cases, E17 `en→en` diagonal self-check, BFGS convergence/failure, `<= 1e-12` AUROC/AP invariance, `temp`-only 0.5-decision preservation, target-head misuse rejection, E07 blocked-audit refusal, fake-adapter dual-pooling extraction, repeat-check validation, and explicit-ref resume rejection).
- [ ] **E29 completed on Colab CPU (Qwen & Gemma):** Both models publish verified `phase2_analysis_block` (with `draw0_reference`) and `phase2_e29_gate` (`OPEN`/`PARK`/`UNRESOLVED` for E30 and E11), with `valid_draws / 2000` recorded.
- [ ] **E17 completed on Colab CPU (Qwen & Gemma):** Both models pass the `en→en` DEV-C diagonal `atol=0, rtol=0` self-check, publish 4 off-diagonal DEV-C prediction blocks (12 routes/model), publish converged `phase2_e17_calibrators` (`T > 0`, both `source_dev_c` and `target_dev_c` scopes), and publish `phase2_analysis_block` passing per-category AUROC/AP invariance (`<= 1e-12`) and `temp`-only 0.5-decision preservation.
- [ ] **E07 GPU extraction completed or explicitly gated off (Qwen & Gemma):** If `phase2_e07_audit` verdict is `passed_full` or `passed_restricted`, both models publish verified `phase2_e07_release` artifacts covering 100% of `extraction_plan["chunks"]` with passing repeat checks and zero CONFIRM rows. If `verdict == "blocked"`, record the audit-driven skip explicitly with zero GPU runs.
- [ ] **Zero Phase 1 mutation & CONFIRM closed:** No writes outside `PHASE1_ROOT/phase2/`, no CONFIRM access, no Stage 4 E19 cutoff edits, and no refitted base heads/scalers.

### 2.8 Stage 2 coding chats: file ownership and ready-to-paste prompts

Because Stage 1's shared interfaces (`src/phase2/interfaces.py`, `gates.py`, `e17_auth.py`, `e07_audit.py`) are already frozen and committed on `phase2`, **Stage 2 Chats A, B, and C** can branch in parallel directly from the commit that adds this Stage 2 specification (`STAGE2_BASE_SHA`), with strictly disjoint file ownership, followed by **Stage 2 Chat D (Integration)**.

| Chat | Branch / base | Owned files (exclusive) | Deliverable |
|---|---|---|---|
| **Stage 2 Chat A: E29 worker** | Branch from `STAGE2_BASE_SHA` | `src/phase2/run_e29.py`, `scripts/phase2/run_e29.ipynb`, `tests/test_phase2_e29.py` | CPU paired pooling gap contrast (`image_caption` primary + `caption_only` diagnostic), reusable `draw0_reference`, `phase2_e29_gate` publication, CLI + Colab notebook + tests. |
| **Stage 2 Chat B: E17 worker** | Branch from `STAGE2_BASE_SHA` | `src/phase2/run_e17.py`, `scripts/phase2/run_e17.ipynb`, `tests/test_phase2_e17.py` | CPU E17 off-diagonal DEV-C scorer (with `en→en` diagonal self-check), float64 BFGS `temp`/`temp_intercept` calibrators (`source_dev_c` & `target_dev_c`), SCREEN evaluator + invariance checks, CLI + Colab notebook + tests. |
| **Stage 2 Chat C: E07 GPU extractor** | Branch from `STAGE2_BASE_SHA` | `src/phase2/extract_e07.py`, `scripts/phase2/extract_e07.ipynb`, `tests/test_phase2_e07_extract.py` | L4 GPU extractor for audited unique changed SCREEN views (midpoint `image_caption`, both poolings same forward via `extract_with_evidence`, `HF_TOKEN`-safe Gemma, chunk resume, repeat checks), CLI + Colab notebook + fake-adapter tests. |
| **Stage 2 Chat D: Integration** | Merge A, B, C into `phase2` | Merge commits, `tests/test_phase2_stage2_integration.py`, `phase2.md §2.9` | Cross-worker schema compatibility checks (E29 `draw0_reference`, E17 calibrator/analysis, E07 release), notebook `CODE_REF` check, full Stage 1 + Stage 2 test pass, and Colab run guide. |

#### Ready-to-paste prompts for Stage 2 coding chats

> **Prompt — Stage 2 Chat A (E29 CPU worker)**
> Branch a clean worktree from `<STAGE2_BASE_SHA>` on `phase2` in `Drazr/Research`. Read `phase2.md` §1.3–1.5, §2.1–2.3 and `docs/phase2_stage1_interfaces.md`. You own ONLY `src/phase2/run_e29.py`, `scripts/phase2/run_e29.ipynb`, and `tests/test_phase2_e29.py`. Do not edit Stage 1 shared modules or Phase 1 files. Implement the CPU E29 worker (`MODEL` selector: `qwen` or `gemma`) consuming the 16 bound Stage 4 SCREEN midpoint blocks per model (`e29_screen_blocks`) and the frozen Stage 4 `(2000, 360)` bootstrap matrix via `src.stage4.metrics`. Compute within-draw paired `G(last_caption) - G(caption_mean)` and residual `G(caption_mean)` across the 14 fixed primary categories and equally weighted 12 off-diagonal routes (plus per-route rows, 4 receiving diagonals, 25-category tables, AP/Brier/NLL/0.5-threshold companions, and the `caption_only` diagnostic on identical draws). Publish `phase2_e29_gate` via `src.phase2.gates.evaluate_e29_gates` + `publish_e29_gate`, and publish `phase2_analysis_block` (including compact reusable `draw0_reference` for Stage 4 `score_e07`) with explicit-ref resume and streamed JSON progress. Add unit/fixture tests covering paired draw arithmetic, unknown-cell mask handling, gate `OPEN`/`PARK`/`UNRESOLVED` cases, and resume rejection. Report branch, commit, owned files, tests passed, and Colab run prerequisites.

> **Prompt — Stage 2 Chat B (E17 CPU worker)**
> Branch a clean worktree from `<STAGE2_BASE_SHA>` on `phase2` in `Drazr/Research`. Read `phase2.md` §1.3–1.6, §2.1–2.2, §2.4 and `docs/phase2_stage1_interfaces.md`. You own ONLY `src/phase2/run_e17.py`, `scripts/phase2/run_e17.ipynb`, and `tests/test_phase2_e17.py`. Do not edit Stage 1 shared modules or Phase 1 files. Implement the end-to-end CPU E17 worker (`MODEL` selector: `qwen` or `gemma`) that: (1) verifies `phase2_plan` and `phase2_e17_auth` (`phase2.e17_dev_c_transfer`) and asserts exact `en→en` DEV-C diagonal reproduction against Stage 4 (`atol=0, rtol=0`); (2) scores and publishes the 4 target-context off-diagonal DEV-C blocks (12 frozen-source-head `s≠t` routes/model) under `phase2/predictions/e17_dev_c/<model>/<run_id>/` from verified Stage 2 DEV-C features and Stage 3 heads/scalers; (3) fits global float64 `temp` and `temp_intercept` calibrators (equal-weight 14-primary-category macro BCE, analytic-gradient BFGS `gtol=1e-10`, `maxiter=500`, zero init) separately on `source_dev_c` (reused Stage 4 diagonals) and `target_dev_c` (frozen source head on target DEV-C, never target-trained heads), publishing `phase2_e17_calibrators`; and (4) evaluates all 5 arms on reused Stage 4 SCREEN logits vs `raw` and fixed FIT-prior references with paired `(2000, 360)` bootstrap intervals, enforcing per-category AUROC/AP invariance (`<= 1e-12`), `T > 0`, `temp`-only 0.5-decision invariance, and separate `decision_0p5` reporting. Keep CONFIRM closed and E19 cutoffs untouched. Support explicit-ref resume and streamed JSON progress. Add fixture tests and report branch, commit, owned files, tests passed, and Colab prerequisites.

> **Prompt — Stage 2 Chat C (E07 L4 GPU extractor worker)**
> Branch a clean worktree from `<STAGE2_BASE_SHA>` on `phase2` in `Drazr/Research`. Read `phase2.md` §1.3–1.4, §1.7, §2.1–2.2, §2.5, `src/stage2/evidence.py`, and `docs/phase2_stage1_interfaces.md`. You own ONLY `src/phase2/extract_e07.py`, `scripts/phase2/extract_e07.ipynb`, and `tests/test_phase2_e07_extract.py`. Do not edit Stage 1 shared modules or Phase 1 files. Implement the L4 GPU extractor (`MODEL` selector: `qwen` or `gemma`, secret-safe `HF_TOKEN` for Gemma, `assert_runtime` check) that refuses to run unless `phase2_e07_audit` has `verdict ∈ {"passed_full", "passed_restricted"}` and `confirm_rows == 0`. Extract ONLY the deduplicated unique genuinely-changed SCREEN `image_caption` views in `extraction_plan["chunks"]` at fixed midpoint (`qwen` depth 18, `gemma` depth 17), capturing BOTH `last_caption` and `caption_mean` on the same forward via `src.stage2.evidence.extract_with_evidence` (batch 1, `bfloat16`, `eager`, no truncation, 32-view chunks, pinned image/caption SHA-256 verification, first/last repeat determinism checks). Publish immutable `phase2_e07_chunk` artifacts with explicit-ref chunk resume, then publish `phase2_e07_release`. Stream JSON progress events and keep CONFIRM strictly closed. Add fake-`HFAdapter` unit/fixture tests covering blocked-audit refusal, CONFIRM rejection, dual-pooling same-forward extraction, byte-identical exclusion, repeat-check failure detection, chunk resume, and release manifest integrity. Report branch, commit, owned files, tests passed, and Colab L4 prerequisites separately from local checks.

### 2.9 Stage 1 & Stage 2 certified completion ledger — 20 November 2026

All Stage 1 and Stage 2 Colab runs passed on mounted `PHASE1_ROOT` (`/content/drive/MyDrive/phase1`) with zero Phase 1 mutation and `CONFIRM` closed. Authoritative completion references pinned for Stage 3–5 consumption:

| Stage / Artifact | Model / Scope | Authoritative relative path under `PHASE1_ROOT` | SHA-256 | Verified outcome |
|---|---|---|---|---|
| **Stage 1 `phase2_plan`** | Both | `phase2/plans/20261119T163039Z-52eb4650cec3436bb4bf989afe3cc303/_COMPLETE.json` | `c2d8e153802ebfefd1caddcb7b0cfc9e681bd1b978a422e4475c3187effbd62d` | Certified Stage 1 bundle (embeds `e07_audit` & `e17_auth`). |
| **Stage 1 `phase2_e07_audit`** | Both | `phase2/audits/e07/20261119T161858Z-a6b9ac3da1324696adf82048b0661bd4/_COMPLETE.json` | `cdee16135af47f595a278316769b5bf26193fde172851df0a4f40275c23cbc38` | Verdict `passed_restricted` (858 unique changed SCREEN views, 29 chunks; 0 CONFIRM). |
| **Stage 1 `phase2_e17_auth`** | Both | `phase2/plans/20261119T162232Z-710a83aae9064c1aa26086b34b17bf45/_COMPLETE.json` | `129cba3ef774dc4c69ecfeabc8547e2aa2186567ec1ee77fb047fc39ba96ddff` | `phase2.e17_dev_c_transfer` authorized; `en→en` diagonal `atol=0, rtol=0`. |
| **Stage 2 E29 analysis** | `qwen` | `phase2/results/e29/qwen/20261119T173548Z-73ad41f4722e42fa931817b6f2b28681/_COMPLETE.json` | `77a8dc8ff50f8886cb706e09e7540952a66c6d87fa7f1278a830f196016c9926` | `passed`, `valid_draws = 2000/2000`. |
| **Stage 2 E29 gate** | `qwen` | `phase2/results/e29_gates/20261119T173548Z-73ad41f4722e42fa931817b6f2b28681-gate/_COMPLETE.json` | `3975fedbd35e63b2e21f5f94b196782f5642ec9e35ec99bb84d7b547c40341ef` | **`E30 = OPEN`** (`consequential_pooling_reduction`); **`E11 = OPEN`** (`broad_and_directional_residual_trigger`). |
| **Stage 2 E29 analysis** | `gemma` | `phase2/results/e29/gemma/20261119T173446Z-35a50762bbf445439faff5951c00eb80/_COMPLETE.json` | `ae3b5183ce126c7e2f6545384949798a4831756da460294585abf980fa12ec16` | `passed`, `valid_draws = 2000/2000`. |
| **Stage 2 E29 gate** | `gemma` | `phase2/results/e29_gates/20261119T173446Z-35a50762bbf445439faff5951c00eb80-gate/_COMPLETE.json` | `7434bee02f3875af67287a520e0731fd1fa8d066c189159a85fc13b0e65fe512` | **`E30 = OPEN`** (`consequential_pooling_reduction`); **`E11 = OPEN`** (`broad_residual_trigger`). |
| **Stage 2 E17 analysis** | `qwen` *(authoritative)* | `phase2/results/e17/qwen/20261119T174402Z-317626df13c944b188686ba32d9b92cb/_COMPLETE.json` | `fcb7c5b1e31f1dfdb43572e077879fdacba18cc8384f77b647d195c04227d771` | `passed`; `en→en` DEV-C `max_abs_diff = 0.0`; calibrators `7d3f3def...`. *(Canonical single Qwen E17 run.)* |
| **Stage 2 E17 analysis** | `gemma` | `phase2/results/e17/gemma/20261119T220802Z-167de08008b74942813cdf81ca061e43/_COMPLETE.json` | `ab3d81143f917c2133d9beb888f64b38330995bcb4f325b7c33ed629e5b05c50` | `passed`; `en→en` DEV-C `max_abs_diff = 0.0`; calibrators `e8a1a9b3...`. |
| **Stage 2 E07 release** | `qwen` | `phase2/features/e07/qwen/releases/20261119T185419Z-7dd0387228af4589b14b976c6b569043/_COMPLETE.json` | `b05a325cae38c89633d237e57c8cd0f7b40ad30a48bd60b40d0d895bad4764d3` | `passed`, depth 18, 858 unique views / 29 chunks (`passed_restricted`). |
| **Stage 2 E07 release** | `gemma` | `phase2/features/e07/gemma/releases/20261119T185440Z-bb5649a5ecdb418eab80a50a8bf748c8/_COMPLETE.json` | `ff5c8d2d2bb04374c735136c5e059390001ff4acdaff7a68ef5c4cc803b0b55e` | `passed`, depth 17, 858 unique views / 29 chunks (`passed_restricted`). |

---

## Stage 3: finalized specification — 20 November 2026

Planning freeze `phase2-stage3.v1`. This section finalizes **Phase 2 Stage 3 only**: executing the two conditional CPU branches opened by Stage 2 E29 for **both** Qwen and Gemma — **E30 CPU** (head × pooling swap at fixed midpoint `image_caption`) and **E11 CPU** (FIT-only `mean_shift` and `mean_scale` covariate adjustment on midpoint `image_caption` / `caption_mean`). Stage 4 (`score_e07` CPU scoring/comparison) and Stage 5 synthesis remain overview-only until finalized in turn.

### 3.1 Gate activation, entry prerequisites, and shared runtime guardrails

- **Gate enforcement:** Both `E30` and `E11` are `OPEN` for both `qwen` and `gemma` per the verified `phase2_e29_gate` receipts in §2.9 (`valid_draws = 2000/2000`). Every Stage 3 worker MUST authenticate both `phase2_plan` and the model's `phase2_e29_gate` via `src.phase2.gates.load_e29_gate`, assert matching `plan_identity` and `model`, and **hard-refuse** to execute if `decisions[branch]["decision"] != "OPEN"`.
- **No Phase 1 mutation, no refitting, CONFIRM closed:** All Phase 1 artifacts remain strictly read-only. Never refit base heads, scalers, or `C` selections; never touch `DEV-C` or `CONFIRM` (`require_open_split("SCREEN")` enforced before feature reads).
- **Phase 1 binding & protocol key lessons (mandatory for all Stage 3 CPU workers):**
  - `load_phase2_plan(...)` returns `plan.binding = None`; always resolve `binding = bind_phase1(plan.root, plain(plan.protocol), plain(plan["stage4_release"]))` explicitly before calling `inputs.py` resolvers.
  - Read ordered categories from `protocol["category_order"]` (not `"categories"`) and SCREEN minimum support from `int(protocol["minimum_positive_and_negative"]["SCREEN"])` (not `"minimum_support"` or `SCREEN_MINIMUM`).
  - Score features through `src.stage3.engine.transform(X, stats) @ coef.T + intercept` in float64 and compute metrics/bootstrap via `src.stage4.metrics` (`probabilities`, `binary_metrics`, `bootstrap_counts`, `bootstrap_binary_metrics`, `fixed_macro`, `bootstrap_interval`, `paired_image_alignment`, `support_status`).
- **Concurrency & storage:** Pre-initialized parent directories already exist under `PHASE1_ROOT/phase2/` (`predictions/e30_crossed/<model>`, `results/e30/<model>`, `predictions/e11_adapted/<model>`, `results/e11/<model>`). One `MODEL` (`qwen` or `gemma`) per Colab CPU runtime; immutable `publish_phase2` (`stage1.v1` `_COMPLETE.json` written last); explicit-ref resume via `verify_resume_ref`; streamed JSON-line progress events (`flush=True`).

### 3.2 Frozen Stage 3 wire schemas (cross-worker and Stage 5 handoff)

Stage 3 workers import shared contracts exclusively from `src.phase2.interfaces`, `src.phase2.gates`, `src.phase2.plan`, `src.stage3.engine`, `src.stage3.readouts`, `src.stage4.plan`, `src.stage4.predictions`, and `src.stage4.metrics`. Do not modify Stage 1/2 shared modules or Phase 1 files.

| Artifact & namespace | `kind` / `body["schema"]` | Required top-level `body` keys & arrays |
|---|---|---|
| **E30 crossed prediction block**<br>`phase2/predictions/e30_crossed/<model>/<run_id>/` | `phase2_prediction_block`<br>`phase2.prediction.v1` | `schema`, `experiment` (`"E30"`), `plan`, `plan_identity`, `gate_ref`, `model`, `midpoint_depth`, `condition` (`"image_caption"`), `split` (`"SCREEN"`), `arm` (`"head_last_on_mean" \| "head_mean_on_last"`), `head_pooling`, `feature_pooling`, `target_language`, `source_languages` (`["en", "hi", "bn", "te"]`), `n_images` (`360`), `n_categories` (`25`), `category_order`, `image_ids_sha256`, `provenance_checks` (`depth_match`, `hook_match`, `dimension`, `row_keys_match`, `model_plan_match`), `matched_self_check` (`route: "en->en"`, `pooling`, `passed: True`, `max_abs_diff: 0.0`, `atol: 0.0`, `rtol: 0.0`), `routes` (4 entries: `route_id`, `source_language`, `row_index`, `source_system_ref`, `supervision_label`), `upstream_feature_ref`, `confirm_excluded` (`True`). Arrays: `logits.npy` (`float64`, `(4, 360, 25)`), `estimable.npy` (`bool`, `(4, 25)`), `prevalence.npy` (`float64`, `(4, 25)`). |
| **E30 analysis block**<br>`phase2/results/e30/<model>/<run_id>/` | `phase2_analysis_block`<br>`phase2.analysis.v1` | `schema`, `experiment` (`"E30"`), `plan`, `plan_identity`, `gate_ref`, `model`, `midpoint_depth`, `condition` (`"image_caption"`), `bootstrap` (`shape`, `sha256`), `matched_block_refs` (8 reused Stage 4 SCREEN refs: 4 `last_on_last` + 4 `mean_on_mean`), `crossed_block_refs` (8 newly published refs: 4 `head_last_on_mean` + 4 `head_mean_on_last`), `provenance_checks`, `arms` (`last_on_last`, `mean_on_mean`, `head_last_on_mean`, `head_mean_on_last` — each with 12-route mean gap `G`, 12-route mean transfer, 4-diagonal mean, 16 per-route rows, 25-category & 14-primary macro tables for AUROC, AP, Brier, NLL, 0.5-threshold metrics + 95% CIs & `valid_draws`), `contrasts` (`feature_gain_fixed_head_last`, `feature_gain_fixed_head_mean`, `head_gain_fixed_feature_last`, `head_gain_fixed_feature_mean`, `interaction`, `total_matched_reduction` — all paired within every bootstrap draw), `off_distribution_collapse` (`drop_threshold: 0.015`, `transfer_collapse_flagged`, `diagonal_collapse_flagged`, `head_last_on_mean_vs_last_on_last`, `head_mean_on_last_vs_last_on_last`), `supervision_labels`, `confirm_excluded` (`True`). |
| **E11 adapted prediction block**<br>`phase2/predictions/e11_adapted/<model>/<run_id>/` | `phase2_prediction_block`<br>`phase2.prediction.v1` | `schema`, `experiment` (`"E11"`), `plan`, `plan_identity`, `gate_ref`, `model`, `midpoint_depth`, `condition` (`"image_caption"`), `pooling` (`"caption_mean"`), `split` (`"SCREEN"`), `variant` (`"mean_shift" \| "mean_scale"`), `target_language`, `source_languages` (3 sorted `s ≠ t`), `n_images` (`360`), `n_categories` (`25`), `category_order`, `image_ids_sha256`, `fit_pairing_check` (`paired_fit_images: 2160`, `fit_image_ids_sha256`, `fit_rows_match_across_languages: True`, `variance_floor: 1e-12`), `identity_self_check` (`route: "en->en"`, `passed: True`, `max_abs_diff: 0.0`, `atol: 0.0`, `rtol: 0.0`), `routes` (3 entries: `route_id`, `source_language`, `row_index`, `source_system_ref`, `source_fit_stats_ref`, `target_fit_stats_ref`, `delta_mean_l2`, `scale_ratio_summary`, `supervision_label: "target_fit_input_adapted"`), `upstream_feature_ref`, `confirm_excluded` (`True`). Arrays: `logits.npy` (`float64`, `(3, 360, 25)`), `estimable.npy` (`bool`, `(3, 25)`), `prevalence.npy` (`float64`, `(3, 25)`). |
| **E11 analysis block**<br>`phase2/results/e11/<model>/<run_id>/` | `phase2_analysis_block`<br>`phase2.analysis.v1` | `schema`, `experiment` (`"E11"`), `plan`, `plan_identity`, `gate_ref`, `model`, `midpoint_depth`, `condition` (`"image_caption"`), `pooling` (`"caption_mean"`), `bootstrap` (`shape`, `sha256`), `identity_block_refs` (4 reused Stage 4 `caption_mean` SCREEN refs), `adapted_block_refs` (8 newly published refs: 4 `mean_shift` + 4 `mean_scale`), `fit_statistics_refs` (4 reused Stage 3 FIT `statistics` refs), `fit_pairing_check`, `arms` (`identity`, `mean_shift`, `mean_scale`, plus `target_diagonal_ref`), `summaries` (12-route mean gap `G`, 12-route mean absolute metrics, 4-diagonal accessibility reference, 12 per-route rows, 25-category & 14-primary macro tables for AUROC, AP, Brier, NLL, 0.5-threshold metrics, paired gains vs `identity` for `mean_shift` and `mean_scale`, `mean_scale` vs `mean_shift` contrast, and fraction of `G_res_mean` recovered with 95% CIs & `valid_draws`), `invariance_checks` (`tolerance: 1e-12`, `mean_shift_max_abs_auroc_diff <= 1e-12`, `mean_shift_max_abs_ap_diff <= 1e-12`, `mean_shift_logit_offset_max_std <= 1e-10`, `passed: True`), `supervision_labels`, `caption_robustness_note`, `confirm_excluded` (`True`). |

### 3.3 Branch A — E30 CPU: head × pooling swap at fixed midpoint

- **Owned files:** `src/phase2/run_e30.py`, `scripts/phase2/run_e30.ipynb`, `tests/test_phase2_e30.py`.
- **Runtime:** Colab CPU (`requirements-stage2-cpu.txt`, Python 3.11/3.12, single BLAS/OpenMP thread), one runtime per `MODEL` (`qwen`, `gemma`), ~2–3 minutes each.
- **Inputs (reused read-only from Phase 1):**
  - `phase2_plan` ref + model's `phase2_e29_gate` ref (must have `decisions["E30"]["decision"] == "OPEN"`).
  - **8 reused Stage 4 SCREEN midpoint `image_caption` blocks/model:** 4 `last_on_last` + 4 `mean_on_mean` (32 matched routes, never duplicated).
  - **8 verified Stage 2 SCREEN midpoint `image_caption` feature groups/model:** 4 languages × 2 poolings (`read_verified_midpoint_features(..., "SCREEN")`).
  - **8 frozen Stage 3 midpoint `image_caption` source systems/model:** 4 languages × 2 poolings (`load_frozen_source_head`, no FIT feature reloads).
  - Frozen Stage 4 `(2000, 360)` bootstrap matrix (`load_bootstrap`) and SCREEN targets/mask (`6a20677eee04fd08` `plant.n.02 = null` preserved).
- **Mandatory provenance & reproduction checks (before scoring crossed arms):**
  1. Assert identical model, midpoint depth (`qwen` 18, `gemma` 17), condition (`image_caption`), residual-stream block hook (`decoder:<depth>`), feature dimension `d`, `model_plan` reference, and exact sorted 360-image SCREEN row keys across `last_caption` and `caption_mean` for all 4 languages.
  2. Run in-memory `en→en` matched self-checks for both `last_caption` and `caption_mean` (`transform(X_en, stats_en) @ coef_en.T + intercept_en`) against Stage 4's accepted `en→en` SCREEN logits and assert `max_abs_diff == 0.0` (`atol=0, rtol=0`).
- **Crossed scoring (8 new blocks / 32 routes per model):**
  - `head_last_on_mean`: for each target language `t`, score all 4 frozen `last_caption` heads+scalers (`s ∈ {en, hi, bn, te}`, including the crossed receiving diagonal `t→t`) on target `t`'s `caption_mean` SCREEN features.
  - `head_mean_on_last`: for each target language `t`, score all 4 frozen `caption_mean` heads+scalers (`s ∈ {en, hi, bn, te}`, including the crossed receiving diagonal `t→t`) on target `t`'s `last_caption` SCREEN features.
  - Assert finite float64 logits on all estimable columns and publish under `phase2/predictions/e30_crossed/<model>/<run_id>/`.
- **Paired contrasts & collapse diagnostic (within every bootstrap draw `b ∈ [0, 1999]`):**
  - Within each arm `A ∈ {last_on_last, mean_on_mean, head_last_on_mean, head_mean_on_last}`, define route gap `G(s→t; A, b) = M(t→t; A, b) - M(s→t; A, b)` (reversed sign for losses `brier`/`nll`), 12-route mean gap `G_mean(A, b)`, 12-route mean transfer `T_mean(A, b)`, and 4-diagonal mean `D_mean(A, b)` on the 14 fixed primary categories (plus all 25 categories and per-route rows).
  - Compute paired within-draw contrasts:
    - Fixed `last_caption` head feature gain: `ΔG_feat|last(b) = G_mean(last_on_last, b) - G_mean(head_last_on_mean, b)` (and per-route / absolute transfer & diagonal shifts).
    - Fixed `caption_mean` head feature gain: `ΔG_feat|mean(b) = G_mean(head_mean_on_last, b) - G_mean(mean_on_mean, b)`.
    - Head-pooling swap contrasts at fixed features: `G_mean(last_on_last, b) - G_mean(head_mean_on_last, b)` and `G_mean(head_last_on_mean, b) - G_mean(mean_on_mean, b)`.
    - Interaction: `I(b) = ΔG_feat|last(b) - ΔG_feat|mean(b)`.
    - Total matched reduction: `G_mean(last_on_last, b) - G_mean(mean_on_mean, b)`.
  - **Off-distribution collapse check (per §1.5):** Flag `off_distribution_collapse` if **both** `head_last_on_mean` and `head_mean_on_last` fall `>= 0.015` AUROC below matched `last_on_last` (evaluated separately on 12-route mean absolute transfer AUROC and on 4-diagonal mean AUROC, recording exact point deltas and 95% CIs).
  - Label `s == t` rows `"within_context_source_fit"` (matched) / `"within_context_source_fit"` (crossed diagonal, explicitly carrying `arm` to distinguish crossed vs matched) and `s ≠ t` rows `"source_only_transfer"`.
- **Resume:** Supports explicit `resume_crossed_refs` (mapping `"{arm}:{target_lang}" -> {path, sha256}`) and `resume_analysis_ref` via `verify_resume_ref`.

### 3.4 Branch B — E11 CPU: FIT-only mean/scale covariate adjustment on `caption_mean`

- **Owned files:** `src/phase2/run_e11.py`, `scripts/phase2/run_e11.ipynb`, `tests/test_phase2_e11.py`.
- **Runtime:** Colab CPU (`requirements-stage2-cpu.txt`, Python 3.11/3.12, single BLAS/OpenMP thread), one runtime per `MODEL` (`qwen`, `gemma`), ~2–3 minutes each.
- **Inputs (reused read-only from Phase 1; ZERO FIT feature matrix reloads):**
  - `phase2_plan` ref + model's `phase2_e29_gate` ref (must have `decisions["E11"]["decision"] == "OPEN"`).
  - `e11_fit_statistics_refs(binding, model)`: resolves the 4 reused Stage 4 `caption_mean` SCREEN blocks (supplies `identity` transfer routes and unchanged `t→t` accessibility diagonals), the 4 Stage 3 FIT `statistics` artifacts (`mean`, `variance`, `scale`, floor `1e-12`), the 4 frozen Stage 3 `caption_mean` heads (`load_frozen_source_head`), and the 4 Stage 2 `caption_mean` SCREEN feature releases (`read_verified_midpoint_features(..., "SCREEN")`).
  - Frozen Stage 4 `(2000, 360)` bootstrap matrix (`load_bootstrap`) and SCREEN targets/mask (`6a20677eee04fd08` `plant.n.02 = null` preserved).
- **Mandatory FIT pairing & identity self-checks (before publishing adapted blocks):**
  1. Verify via `src.stage3.readouts.expected_rows(fit, gid, "FIT", store)` and `load_statistics(...)` that all 4 language `caption_mean` FIT `statistics` artifacts are estimated on the **exact same 2,160 paired FIT `image_id`s** (`paired_fit_images == 2160`, identical ordered `image_id` list across `en, hi, bn, te`, variance floor `1e-12` satisfied). Never open FIT `.npy` matrices.
  2. Run in-memory `en→en` `caption_mean` SCREEN diagonal self-check (`transform(X_en, stats_en) @ coef_en.T + intercept_en`) against Stage 4's accepted `en→en` `caption_mean` SCREEN logits and assert `max_abs_diff == 0.0` (`atol=0, rtol=0`).
- **FIT-only feature transforms & scoring (8 new blocks / 24 adapted transfer routes per model):**
  - For each target language `t ∈ ('en', 'hi', 'bn', 'te')`, load verified `caption_mean` SCREEN features `H_t` (`(360, d)`, float64) once, and for each non-diagonal source `s ≠ t` (3 sorted sources):
    - **`mean_shift`:** `H_shift = H_t - mu_t + mu_s`, then score through the **unchanged source scaler and head**: `Z_shift = transform(H_shift, stats_s) @ coef_s.T + intercept_s`.
    - **`mean_scale`:** `H_scale = mu_s + sigma_s * ((H_t - mu_t) / sigma_t)` (using the exact verified `stats_s["scale"]` and `stats_t["scale"]`, already floored at `1.0` where `variance <= 1e-12`), then score through the **unchanged source scaler and head**: `Z_scale = transform(H_scale, stats_s) @ coef_s.T + intercept_s`.
  - Label every adapted route `"target_fit_input_adapted"`; label reused `identity` off-diagonals `"source_only_transfer"` and unchanged `t→t` diagonals `"within_context_source_fit"`.
  - Publish 4 `mean_shift` and 4 `mean_scale` blocks under `phase2/predictions/e11_adapted/<model>/<run_id>/`. Never duplicate Stage 4's `identity` blocks or `t→t` diagonals.
- **Paired comparisons & mandatory hard invariance check (within every bootstrap draw `b ∈ [0, 1999]`):**
  - Evaluate `identity`, `mean_shift`, and `mean_scale` against the unchanged `t→t` receiving-context diagonals on identical SCREEN known cells for AUROC, AP, Brier, NLL, and 0.5-threshold metrics (14-primary macro, 12-route mean, 12 per-route rows, and 25-category tables).
  - Compute within-draw paired improvements vs `identity` for both `mean_shift` and `mean_scale`:
    - AUROC / AP gap reduction `G(identity, b) - G(variant, b)` (equivalently `M(variant, b) - M(identity, b)` since the `t→t` diagonal is identical across arms),
    - Proper-loss reductions `Brier(identity, b) - Brier(variant, b)` and `NLL(identity, b) - NLL(variant, b)`,
    - `mean_scale` minus `mean_shift` paired contrast (isolating diagonal scaling beyond constant logit offset),
    - Fraction of residual gap recovered: `[G_mean(identity, b) - G_mean(mean_scale, b)] / G_mean(identity, b)` (masked `NaN` if `G_mean(identity, b) <= 0`, reported alongside point ratio).
  - **Hard automatic invariance check (HARD FAIL if violated):**
    - For every directed route `s ≠ t` and every estimable category `k`, assert `|AUROC_k(mean_shift) - AUROC_k(identity)| <= 1e-12` and `|AP_k(mean_shift) - AP_k(identity)| <= 1e-12` (and verify the per-category logit difference `Z_shift[:, k] - Z_identity[:, k]` has `std <= 1e-10` across the 360 SCREEN images).
  - Record explicit `caption_robustness_note`: E11 informs draw-0 `caption_mean` covariate statistics; caption-robustness claims wait for Stage 4 E07.
- **Resume:** Supports explicit `resume_adapted_refs` (mapping `"{variant}:{target_lang}" -> {path, sha256}`) and `resume_analysis_ref` via `verify_resume_ref`.

### 3.5 Stage 3 parallel execution order and expected Colab timing

All 4 Stage 3 runs are **mutually independent CPU jobs** and can run in parallel immediately on separate Colab CPU runtimes:

| Notebook | Runtime | `MODEL` | Required Cell 3 refs | Parallelism | Est. Time |
|---|---|---|---|---|---|
| `scripts/phase2/run_e30.ipynb` | **Colab CPU** | `qwen` / `gemma` | `PLAN_REF` + model's `E29_GATE_REF` | Both models in parallel with E11 | **~2–3 min / model** |
| `scripts/phase2/run_e11.ipynb` | **Colab CPU** | `qwen` / `gemma` | `PLAN_REF` + model's `E29_GATE_REF` | Both models in parallel with E30 | **~2–3 min / model** |

### 3.6 Stage 3 acceptance criteria

Stage 3 is complete only when **all** items below pass and are recorded with explicit `{path, sha256}` completion references:

- [ ] **Local unit/fixture tests pass:** `tests/test_phase2_e30.py` and `tests/test_phase2_e11.py` pass (covering closed-gate `PARK`/`UNRESOLVED` refusal, provenance mismatch rejection, `en→en` `atol=0, rtol=0` self-check, E30 2×2 paired contrast & interaction & off-distribution collapse detection, E11 paired 2,160 FIT check, `mean_shift` `<= 1e-12` AUROC/AP invariance & hard-failure on violation, `mean_scale` variance-floor handling, unknown-cell mask preservation, and explicit-ref resume rejection).
- [ ] **E30 completed on Colab CPU (Qwen & Gemma):** Both models pass provenance checks and `en→en` `atol=0, rtol=0` matched self-checks, publish 8 crossed SCREEN blocks (32 routes/model), and publish verified `phase2_analysis_block` with paired feature-gain, interaction, and off-distribution collapse diagnostics (`valid_draws / 2000` recorded).
- [ ] **E11 completed on Colab CPU (Qwen & Gemma):** Both models verify 2,160 paired FIT IDs (0 FIT matrix reloads) and `en→en` `atol=0, rtol=0` identity self-check, publish 8 adapted SCREEN blocks (`mean_shift` + `mean_scale`, 24 routes/model), and publish verified `phase2_analysis_block` passing `mean_shift` `<= 1e-12` AUROC/AP invariance and separating ranking vs proper-loss gains (`valid_draws / 2000` recorded).
- [ ] **Zero Phase 1 mutation & CONFIRM closed:** No writes outside `PHASE1_ROOT/phase2/`, no `DEV-C` or `CONFIRM` access, and no base head/scaler refitting.

### 3.7 Stage 3 coding chats: file ownership and ready-to-paste prompts

No separate Stage 3 foundation chat is needed — Stage 1's shared modules (`src/phase2/interfaces.py`, `gates.py`, `plan.py`, `inputs.py`) already cover Stage 3. **Stage 3 Chats A and B** branch in parallel directly from the commit that adds this Stage 3 specification (`STAGE3_BASE_SHA`), followed by merging/testing/pushing on `phase2`:

| Chat | Branch / base | Owned files (exclusive) | Deliverable |
|---|---|---|---|
| **Stage 3 Chat A: E30 CPU worker** | Branch from `STAGE3_BASE_SHA` | `src/phase2/run_e30.py`, `scripts/phase2/run_e30.ipynb`, `tests/test_phase2_e30.py` | CPU head × pooling swap scorer (8 crossed blocks/model), matched-arm `en→en` self-check, paired 2×2 contrasts + interaction + off-distribution collapse diagnostic, CLI + Colab notebook + tests. |
| **Stage 3 Chat B: E11 CPU worker** | Branch from `STAGE3_BASE_SHA` | `src/phase2/run_e11.py`, `scripts/phase2/run_e11.ipynb`, `tests/test_phase2_e11.py` | CPU FIT-only `mean_shift`/`mean_scale` adapter & scorer (8 adapted blocks/model, zero FIT matrix reloads), `mean_shift` `<= 1e-12` invariance enforcement, paired ranking vs loss contrasts, CLI + Colab notebook + tests. |
| **Stage 3 Chat C: Integration** | Merge A & B into `phase2` | Merge commits, `phase2.md §3.8` | Cross-check schemas, `CODE_REF = 'phase2'`, full Stage 1–3 test pass, and push to `origin/phase2`. |

#### Ready-to-paste prompts for Stage 3 coding chats

> **Prompt — Stage 3 Chat A (E30 CPU worker)**
> Branch a clean worktree from `<STAGE3_BASE_SHA>` on `phase2` in `Drazr/Research`. Read `phase2.md` §1.3–1.5, §3.1–3.3 and `docs/phase2_stage1_interfaces.md`. You own ONLY `src/phase2/run_e30.py`, `scripts/phase2/run_e30.ipynb`, and `tests/test_phase2_e30.py`. Do not edit Stage 1/2 shared modules or Phase 1 files. Implement the CPU E30 worker (`MODEL` selector: `qwen` or `gemma`, notebook `CODE_REF = 'phase2'`) that: (1) loads `phase2_plan` + explicitly binds `Phase1Binding` via `bind_phase1`, loads `phase2_e29_gate` via `load_e29_gate`, and refuses to run unless `decisions["E30"]["decision"] == "OPEN"`; (2) verifies identical midpoint depth, hook, dimension, `model_plan`, and 360-image SCREEN row keys across `last_caption` and `caption_mean`, and asserts `en→en` matched-arm reproduction (`atol=0, rtol=0`) against Stage 4; (3) scores ONLY the 2 crossed arms (8 blocks / 32 routes per model: `head_last_on_mean` and `head_mean_on_last`, including crossed `t→t` diagonals) via `src.stage3.engine.transform` and publishes under `phase2/predictions/e30_crossed/<model>/<run_id>/`; and (4) evaluates all 4 arms (reusing 8 Stage 4 matched blocks) on the frozen `(2000, 360)` bootstrap via `src.stage4.metrics` (using `protocol["category_order"]` and `protocol["minimum_positive_and_negative"]["SCREEN"]`), computing within-draw paired feature gains, head-swap contrasts, interaction, and off-distribution collapse flags (`>= 0.015` AUROC drop vs matched `last_on_last`), publishing under `phase2/results/e30/<model>/<run_id>/`. Keep CONFIRM closed and support explicit-ref resume and streamed JSON progress. Add unit/fixture tests and report branch, commit SHA, owned files, tests passed, and Colab run prerequisites.

> **Prompt — Stage 3 Chat B (E11 CPU worker)**
> Branch a clean worktree from `<STAGE3_BASE_SHA>` on `phase2` in `Drazr/Research`. Read `phase2.md` §1.3–1.5, §3.1–3.2, §3.4 and `docs/phase2_stage1_interfaces.md`. You own ONLY `src/phase2/run_e11.py`, `scripts/phase2/run_e11.ipynb`, and `tests/test_phase2_e11.py`. Do not edit Stage 1/2 shared modules or Phase 1 files. Implement the CPU E11 worker (`MODEL` selector: `qwen` or `gemma`, notebook `CODE_REF = 'phase2'`) that: (1) loads `phase2_plan` + explicitly binds `Phase1Binding` via `bind_phase1`, loads `phase2_e29_gate` via `load_e29_gate`, and refuses to run unless `decisions["E11"]["decision"] == "OPEN"`; (2) resolves `e11_fit_statistics_refs`, verifies all 4 `caption_mean` Stage 3 FIT `statistics` artifacts use the exact same 2,160 paired FIT image IDs with `1e-12` variance floor (zero FIT feature matrix reloads), and asserts `en→en` `caption_mean` SCREEN diagonal reproduction (`atol=0, rtol=0`) against Stage 4; (3) scores `mean_shift` (`h_t - mu_t + mu_s`) and `mean_scale` (`mu_s + sigma_s * (h_t - mu_t) / sigma_t`) on verified `caption_mean` SCREEN features through the unchanged source scaler (`src.stage3.engine.transform`) and source head, publishing 8 adapted blocks (24 `s≠t` routes/model) under `phase2/predictions/e11_adapted/<model>/<run_id>/` with `"target_fit_input_adapted"` supervision labels; and (4) evaluates `identity` (reused 4 Stage 4 `caption_mean` SCREEN blocks), `mean_shift`, and `mean_scale` against unchanged `t→t` diagonals on the frozen `(2000, 360)` bootstrap via `src.stage4.metrics` (using `protocol["category_order"]` and `protocol["minimum_positive_and_negative"]["SCREEN"]`), hard-failing if `mean_shift` per-category AUROC/AP differs from `identity` by `> 1e-12` or logit offset std `> 1e-10`, and reporting ranking vs proper-loss gains and recovered fraction of `G_res_mean` under `phase2/results/e11/<model>/<run_id>/`. Keep CONFIRM closed and support explicit-ref resume and streamed JSON progress. Add unit/fixture tests and report branch, commit SHA, owned files, tests passed, and Colab run prerequisites.

---

## Stage 4: finalized specification — 20 November 2026

Planning freeze `phase2-stage4.v1`. Because Stage 2 `run_e17.py` already completed E17 SCREEN calibration evaluation end-to-end on CPU (§2.9), **Phase 2 Stage 4 consists solely of E07 CPU frozen-head scoring and paired alternate-caption comparison** (`score_e07.py` / `score_e07.ipynb`). Because E07 scoring depends only on completed Stage 1 + Stage 2 artifacts (`phase2_plan`, `phase2_e07_audit`, `phase2_e07_release`, and E29 `draw0_reference`) and has zero dependency on E30 or E11, **Stage 4 can be coded and executed in parallel with Stage 3 (`E30` and `E11`)**. Stage 5 synthesis remains overview-only until finalized in turn.

### 4.1 Entry prerequisites, zero-changed-language handling, and shared guardrails

- **Required Stage 1 & Stage 2 references (`{path, sha256}`, pinned in §2.9):**
  - `phase2_plan` (`KIND_PLAN`), `phase2_e07_audit` (`verdict == "passed_restricted"`, 858 unique views / 29 chunks), model's `phase2_e07_release` (`qwen`: `b05a325c...`; `gemma`: `ff5c8d2d...`), and model's Stage 2 E29 `phase2_analysis_block` (`qwen`: `77a8dc8f...`; `gemma`: `ae3b5183...`, supplying `draw0_reference`).
- **Critical `passed_restricted` dataset property (verified in Stage 2 `extract_e07`):**
  - Unique changed SCREEN views across draws 1 & 2 are `en: 267`, `hi: 327`, **`bn: 0`**, `te: 264` (`total_unique_views = 858`, 29 chunks). Bengali (`bn`) is a single-caption language in SCREEN (`n_changed == 0`, `unchanged_fraction == 1.0`).
  - `score_e07` MUST handle any language with `n_changed == 0` gracefully: reuse 100% of Stage 4 draw-0 logits for that language on `full_screen_360` (`atol=0, rtol=0`), mark that language `"unavailable"` (`reason: "unavailable_single_caption_zero_changed_views"`) in `restricted_changed_cohort` without dividing by zero or crashing, and report BOTH the full 12-route macro AND the `changed_targets_only` 9-route macro (`t ∈ {en, hi, te}`, 3 non-diagonal sources each).
- **Mandatory runtime & contract guardrails:**
  - Explicitly resolve `binding = bind_phase1(plan.root, plain(plan.protocol), plain(plan["stage4_release"]))` after `load_phase2_plan(...)`.
  - Read categories from `protocol["category_order"]` and SCREEN minimum support from `int(protocol["minimum_positive_and_negative"]["SCREEN"])`.
  - Hard-refuse if `e07_audit` verdict is `"blocked"`, if `e07_release` has `confirm_excluded != True` or `plan_identity` mismatch or any failed repeat check, or if any chunk row hash/shape fails verification.
  - Never refit heads/scalers, never mutate Phase 1, and keep `CONFIRM` closed (`require_open_split("SCREEN")`).

### 4.2 Frozen Stage 4 wire schemas (Stage 5 handoff)

| Artifact & namespace | `kind` / `body["schema"]` | Required top-level `body` keys & arrays |
|---|---|---|
| **E07 changed-view prediction block**<br>`phase2/predictions/e07_screen/<model>/<run_id>/` | `phase2_prediction_block`<br>`phase2.prediction.v1` | `schema`, `experiment` (`"E07"`), `plan`, `plan_identity`, `e07_audit_ref`, `e07_release_ref`, `e07_verdict`, `model`, `midpoint_depth`, `condition` (`"image_caption"`), `pooling` (`"last_caption" \| "caption_mean"`), `split` (`"SCREEN"`), `target_language`, `source_languages` (`["en", "hi", "bn", "te"]`), `n_screen_images` (`360`), `n_changed_views`, `n_categories` (`25`), `category_order`, `screen_ids_sha256`, `draw0_block_ref` (reused Stage 4 SCREEN block ref), `draw0_self_check` (`route: "en->en"`, `pooling`, `passed: True`, `max_abs_diff: 0.0`, `atol: 0.0`, `rtol: 0.0`), `draw_indices` (`draw_1`, `draw_2`: `changed_text_vs_draw0`, `unchanged_reused_from_draw0`, `unchanged_fraction`, `changed_row_map` mapping `screen_row_index -> {changed_view_index, image_id, caption_id, caption_text_sha256, chunk_ref}`), `routes` (4 source entries with `source_language`, `row_index`, `source_system_ref`, `supervision_label`), `upstream_chunk_refs`, `confirm_excluded` (`True`). Arrays: `logits.npy` (`float64`, `(4, n_changed_views, 25)`), `estimable.npy` (`bool`, `(4, 25)`), `prevalence.npy` (`float64`, `(4, 25)`). |
| **E07 SCREEN analysis block**<br>`phase2/results/e07/<model>/<run_id>/` | `phase2_analysis_block`<br>`phase2.analysis.v1` | `schema`, `experiment` (`"E07"`), `plan`, `plan_identity`, `e07_audit_ref`, `e07_release_ref`, `e07_verdict`, `e29_analysis_ref`, `model`, `midpoint_depth`, `condition` (`"image_caption"`), `poolings` (`["last_caption", "caption_mean"]`), `bootstrap` (`shape`, `sha256`), `prediction_block_refs` (`{pooling: {lang: ref}}`, 8 refs), `draw0_parity_check` (`e29_draw0_max_abs_diff <= 1e-12`, `unchanged_views_atol: 0.0`, `unchanged_views_rtol: 0.0`, `passed: True`), `coverage_summary` (per-language & draw 1/2 counts, `unchanged_view_fraction`, `eligible_changed_languages`, `unavailable_languages`), `full_screen_360` (draws `draw_0`, `draw_1`, `draw_2` × both poolings: 12-route mean AND `changed_targets_only` 9-route mean, 4 diagonals, 16 per-route rows, 25-category & 14-primary macro tables for AUROC, AP, Brier, NLL, 0.5-threshold metrics, paired draw shifts `G(draw_k) - G(draw_0)`, absolute diagonal/transfer shifts, paired pooling-reduction shift `ΔG(draw_k) - ΔG(draw_0)`, `pooling_reduction_sign_preserved` + 95% CIs & `valid_draws`), `restricted_changed_cohort` (per-target-language `S_changed(t, draw_k)` paired with exact matched `draw_0` companion on identical image-category cells, reporting 14-primary & `supported_primary_ge_10` macros + explicit category lists/denominators, marking `n_changed == 0` languages `"unavailable"`, plus aggregate over eligible changed targets and multi-language changed intersection metadata), `supervision_labels`, `interpretation_scope` (`"frozen_draw0_heads_test_view_sensitivity"`), `confirm_excluded` (`True`). |

### 4.3 E07 CPU scoring & dual-cohort comparison specification

- **Owned files:** `src/phase2/score_e07.py`, `scripts/phase2/score_e07.ipynb`, `tests/test_phase2_e07_score.py`.
- **Runtime:** Colab CPU (`requirements-stage2-cpu.txt`, Python 3.11/3.12, single BLAS/OpenMP thread), one runtime per `MODEL` (`qwen`, `gemma`), ~2–4 minutes each.
- **Step 1 — Authenticate upstream artifacts & draw-0 self-check:**
  - Verify `phase2_plan`, `phase2_e07_audit`, `phase2_e07_release` (including every referenced `phase2_e07_chunk` `_COMPLETE.json`, `rows.jsonl`, `last_caption.npy`, `caption_mean.npy` hash/shape/finiteness, and `repeat_check_summary.all_passed == True`), and the model's Stage 2 E29 `phase2_analysis_block` (`draw0_reference`).
  - Load the 8 frozen Stage 3 midpoint `image_caption` source systems (`last_caption` and `caption_mean` × 4 languages via `load_frozen_source_head`) and the 8 reused Stage 4 draw-0 SCREEN midpoint `image_caption` prediction blocks.
  - Run in-memory `en→en` draw-0 SCREEN reproduction checks for BOTH `last_caption` and `caption_mean` against Stage 4 (`transform(X_en, stats_en) @ coef_en.T + intercept_en`, asserting `max_abs_diff == 0.0`, `atol=0, rtol=0`) before publishing any E07 score blocks.
- **Step 2 — Score unique changed views & assemble draw-1 / draw-2 tensors:**
  - For each `pooling ∈ ("last_caption", "caption_mean")` and each `target_language ∈ ("en", "hi", "bn", "te")`:
    - Collect all verified unique changed views for `target_language` from the release chunks in deterministic order.
    - If `n_changed_views > 0`, score all 4 frozen source heads (`s ∈ {en, hi, bn, te}`) at `pooling` via `transform(X_changed.astype(float64), stats_s) @ coef_s.T + intercept_s` (asserting finite float64 logits on all estimable columns). If `n_changed_views == 0` (e.g., `bn`), construct an empty `(4, 0, 25)` float64 array.
    - For each `draw_k ∈ (1, 2)`, map each changed view whose `draw_keys` contains `draw_k` onto its exact sorted `screen_row_index` in `screen_ids`. Cross-check that the mapped count equals `audit_body["draw_buckets"][target_language][f"draw_{draw_k}"]["changed_text_vs_draw0"]` identically and that no `(image_id, draw_k)` collides.
    - Publish the compact changed-view prediction block under `phase2/predictions/e07_screen/<model>/<run_id>/`.
    - In memory, form the full 360-image draw-`k` logit tensor `(4, 360, 25)` by copying the verified Stage 4 draw-0 logit tensor and replacing ONLY the mapped changed rows; assert bit-for-bit (`atol=0, rtol=0`) equality with Stage 4 draw-0 logits on every unchanged / byte-identical row.
- **Step 3 — Dual-cohort paired bootstrap evaluation (within every draw `b ∈ [0, 1999]`):**
  - Load the frozen Stage 4 `(2000, 360)` bootstrap matrix (`load_bootstrap`) and SCREEN targets/unknown-cell mask (`6a20677eee04fd08` `plant.n.02 = null` preserved).
  - **Draw-0 parity assertion:** Recompute draw-0 `image_caption` metrics on the reused Stage 4 blocks and assert exact agreement (`max_abs_diff <= 1e-12`) against `e29_body["draw0_reference"]["point_estimates"]`.
  - **Cohort A (`full_screen_360`):** For `draw_k ∈ {1, 2}` and `pooling ∈ {last_caption, caption_mean}`, compute within every bootstrap draw `b`:
    - All 16 routes, 4 receiving diagonals, 12-route mean gap `G(draw_k, p, b)` AND `changed_targets_only` 9-route mean gap (excluding `n_changed == 0` targets),
    - Paired draw shifts `G(draw_k, p, b) - G(draw_0, p, b)`, absolute diagonal/transfer shifts `M(draw_k, p, b) - M(draw_0, p, b)`,
    - Paired pooling reduction `ΔG(draw_k, b) = G(draw_k, last_caption, b) - G(draw_k, caption_mean, b)` and its paired draw shift `ΔG(draw_k, b) - ΔG(draw_0, b)`, plus explicit `pooling_reduction_sign_preserved` check (`sign(ΔG(draw_k)) == sign(ΔG(draw_0))`),
    - All metrics (AUROC primary, AP, Brier, NLL, 0.5-threshold metrics) on 14 fixed primary categories and all 25 categories.
  - **Cohort B (`restricted_changed_cohort`, mandatory under `passed_restricted`):**
    - For each `(target_language t, draw_k)`:
      - If `|S_changed(t, draw_k)| == 0`, emit `status: "unavailable"`, `reason: "unavailable_single_caption_zero_changed_views"`, `n_images: 0`.
      - Otherwise, slice the frozen `(2000, 360)` bootstrap image counts to the exact `S_changed(t, draw_k)` indices (preserving the exact image resampling unit across `draw_k` and `draw_0`!). Compute BOTH `draw_k` and the **exact matched `draw_0` companion** on identical `S_changed(t, draw_k)` image-category cells (preserving unknown `plant.n.02` if present in the subset). Report:
        - `supported_primary_ge_10` macro (all primary categories achieving `min(pos, neg) >= 10` on `S_changed(t, draw_k)`, with explicit category list and support denominators),
        - Full 14-primary macro (with explicit `status` / `descriptive_low_support` flag if any primary category falls below support),
        - Per-route and target-level paired shifts `G(draw_k) - G(draw_0)` and `ΔG(draw_k) - ΔG(draw_0)` with 95% CIs and `valid_draws`.
    - Compute the eligible-targets aggregate (`restricted_changed_cohort` mean across the 9 directed routes `s ≠ t` for `t` with `|S_changed(t, draw_k)| > 0`, paired with its exact matched draw-0 companion on the exact same route-image-category cells) and record 3-language / 4-language simultaneous changed intersection counts.
  - Publish `phase2_analysis_block` under `phase2/results/e07/<model>/<run_id>/`.
- **Resume:** Supports explicit `resume_prediction_refs` (`"{pooling}:{lang}" -> {path, sha256}`) and `resume_analysis_ref` via `verify_resume_ref`.

### 4.4 Combined Stage 3 + Stage 4 parallel execution order (Colab CPU)

All **6 remaining Phase 2 CPU runs** (`E30`, `E11`, and `score_e07` × `qwen`, `gemma`) are mutually independent and can execute in parallel on separate Colab CPU runtimes:

| Stage | Notebook | Runtime | `MODEL` | Required Cell 3 refs | Est. Time |
|---|---|---|---|---|---|
| **Stage 3** | `scripts/phase2/run_e30.ipynb` | **Colab CPU** | `qwen` / `gemma` | `PLAN_REF` + model's `E29_GATE_REF` | **~2–3 min / model** |
| **Stage 3** | `scripts/phase2/run_e11.ipynb` | **Colab CPU** | `qwen` / `gemma` | `PLAN_REF` + model's `E29_GATE_REF` | **~2–3 min / model** |
| **Stage 4** | `scripts/phase2/score_e07.ipynb` | **Colab CPU** | `qwen` / `gemma` | `PLAN_REF` + `E07_AUDIT_REF` + model's `E07_RELEASE_REF` + model's `E29_ANALYSIS_REF` | **~2–4 min / model** |

### 4.5 Stage 4 acceptance criteria

- [ ] **Local unit/fixture tests pass:** `tests/test_phase2_e07_score.py` passes (covering blocked-audit refusal, zero-changed-language `bn` handling without division by zero, `en→en` draw-0 `atol=0, rtol=0` self-check, bit-for-bit draw-0 reuse on unchanged/byte-identical rows, `draw0_reference` parity assertion, `full_screen_360` + `restricted_changed_cohort` matched draw-0 pairing, unknown-cell mask preservation, and explicit-ref resume rejection).
- [ ] **E07 CPU scoring completed on Colab CPU (Qwen & Gemma):** Both models pass draw-0 self-checks and `E29` `draw0_reference` parity, publish 8 compact `phase2_prediction_block` artifacts (`phase2/predictions/e07_screen/<model>/`), and publish verified `phase2_analysis_block` (`phase2/results/e07/<model>/`) with both `full_screen_360` (12-route + 9-route changed-targets) and `restricted_changed_cohort` dual reports (`valid_draws / 2000` recorded).
- [ ] **Zero Phase 1 mutation, no head refitting, CONFIRM closed.**

### 4.6 Stage 4 coding chat: file ownership and ready-to-paste prompt

Can run **in parallel with Stage 3 Chats A (`E30`) and B (`E11`)** from the same committed base SHA (`STAGE34_BASE_SHA`):

| Chat | Branch / base | Owned files (exclusive) | Deliverable |
|---|---|---|---|
| **Stage 4 Chat C: E07 CPU scorer** | Branch from `STAGE34_BASE_SHA` | `src/phase2/score_e07.py`, `scripts/phase2/score_e07.ipynb`, `tests/test_phase2_e07_score.py` | CPU E07 frozen-head scorer for audited changed SCREEN views + exact draw-0 reuse + dual-cohort (`full_screen_360` & `restricted_changed_cohort`) paired bootstrap comparator, CLI + Colab notebook + tests. |

> **Prompt — Stage 4 Chat C (E07 CPU scorer & comparator)**
> Branch a clean worktree from `<STAGE34_BASE_SHA>` on `phase2` in `Drazr/Research`. Read `phase2.md` §1.3–1.5, §1.7, §2.9, §4.1–4.3 and `docs/phase2_stage1_interfaces.md`. You own ONLY `src/phase2/score_e07.py`, `scripts/phase2/score_e07.ipynb`, and `tests/test_phase2_e07_score.py`. Do not edit Stage 1/2 shared modules or Phase 1 files. Implement the CPU E07 scorer/comparator (`MODEL` selector: `qwen` or `gemma`, notebook `CODE_REF = 'phase2'`) that: (1) loads `phase2_plan` + explicitly binds `Phase1Binding` via `bind_phase1`, verifies `phase2_e07_audit` (refusing `blocked`), `phase2_e07_release` (verifying all chunk manifests, `.npy` hashes/shapes, and passing repeat checks), and the model's Stage 2 E29 `phase2_analysis_block` (`draw0_reference`); (2) runs in-memory `en→en` draw-0 SCREEN reproduction self-checks (`atol=0, rtol=0`) for both `last_caption` and `caption_mean` against Stage 4; (3) scores ONLY the verified unique changed views per `(pooling, target_language)` with the 4 frozen Stage 3 midpoint `image_caption` heads/scalers via `src.stage3.engine.transform` (gracefully handling zero-changed languages like `bn` with `(4, 0, 25)` empty logits), verifies mapped draw-1/draw-2 counts match `audit_body["draw_buckets"]`, publishes 8 compact `phase2_prediction_block` artifacts under `phase2/predictions/e07_screen/<model>/<run_id>/`, and reuses Stage 4 draw-0 logits bit-for-bit (`atol=0, rtol=0`) on all unchanged/byte-identical rows; and (4) evaluates BOTH `full_screen_360` (12-route mean + `changed_targets_only` 9-route mean, asserting `<= 1e-12` parity with E29 `draw0_reference`) AND `restricted_changed_cohort` (per-target `S_changed(t, draw_k)` with exact matched draw-0 companion on identical image-category cells and sliced `(2000, 360)` bootstrap counts, marking `n_changed == 0` languages `"unavailable"`, reporting `supported_primary_ge_10` and 14-primary macros, pooling-reduction shifts, and sign preservation) using `protocol["category_order"]` and `protocol["minimum_positive_and_negative"]["SCREEN"]`, publishing under `phase2/results/e07/<model>/<run_id>/`. Keep CONFIRM closed and never refit heads. Support explicit-ref resume and streamed JSON progress. Add unit/fixture tests covering zero-changed language handling, unchanged-row `atol=0` reuse, `draw0_reference` parity, restricted-cohort pairing, and resume rejection. Report branch, commit SHA, owned files, tests passed, and Colab run prerequisites.

### 4.7 Stage 3 & Stage 4 certified completion ledger — 20 November 2026

All 6 Stage 3 (`E30`, `E11`) and Stage 4 (`score_e07`) Colab CPU runs passed on mounted `PHASE1_ROOT` with `0.0` self-check/parity diffs, zero Phase 1 mutation, and `CONFIRM` closed:

| Stage / Artifact | Model | Authoritative relative path under `PHASE1_ROOT` | SHA-256 | Verified outcome |
|---|---|---|---|---|
| **Stage 3 E30 analysis** | `qwen` | `phase2/results/e30/qwen/20261119T230416Z-8c601fa863604e3395624e37382fdedb/_COMPLETE.json` | `0dbe25f3fe6659b6e18419153c354b97ee4b124d8b29bd5096e7652c7a0b1f9f` | `passed`; `feature_gain|last = +0.0919`, `feature_gain|mean = +0.0054`, `interaction = +0.0865`; no off-distribution collapse. |
| **Stage 3 E30 analysis** | `gemma` | `phase2/results/e30/gemma/20261119T225928Z-682c54d91984488aacffba40ab0f6b2c/_COMPLETE.json` | `0de28dc4d6d4592e4312c3b64c86c8810267d1a4d10f9f36bb319a32b5382c6a` | `passed`; `feature_gain|last = +0.0512`, `feature_gain|mean = +0.0059`, `interaction = +0.0453`; no off-distribution collapse. |
| **Stage 3 E11 analysis** | `qwen` | `phase2/results/e11/qwen/20261119T230112Z-af0550ca6a0c494e8a4f9d1728af213e/_COMPLETE.json` | `552d0f4f59bf0760ff7ee8da62b64197550d9087071becd2a260080f203f4c1b` | `passed`; 2,160 paired FIT IDs (0 FIT matrix reloads); `en→en` `0.0`; `mean_shift` AUROC/AP diff `0.0`. |
| **Stage 3 E11 analysis** | `gemma` | `phase2/results/e11/gemma/20261119T230113Z-af983c1752eb4861a0c9390d53c096e5/_COMPLETE.json` | `09fa67ee3d4aed49726e57403077f47aeca47bab6fe4a67559ce2def0173dbbc` | `passed`; 2,160 paired FIT IDs (0 FIT matrix reloads); `en→en` `0.0`; `mean_shift` AUROC/AP diff `0.0`. |
| **Stage 4 E07 analysis** | `qwen` | `phase2/results/e07/qwen/20261119T230606Z-f629f48ed8e7431b930fce6b47d8fd53/_COMPLETE.json` | `f8bda6f082862a07a15b42d497d605894ab4ceed7403dfdb257d9ff013f3ee09` | `passed` (`passed_restricted`); `e29_draw0_max_abs_diff = 0.0`; eligible `[en, hi, te]`, unavailable `[bn]`. |
| **Stage 4 E07 analysis** | `gemma` | `phase2/results/e07/gemma/20261119T230653Z-7ee10a1674f245ea97d5e50a6a6645ae/_COMPLETE.json` | `4603c8d8120d816ce14a7c299e561248a1f63fa670ac6c776e7a6034f09bdd95` | `passed` (`passed_restricted`); `e29_draw0_max_abs_diff = 0.0`; eligible `[en, hi, te]`, unavailable `[bn]`. |

---

## Stage 5: finalized specification — 20 November 2026

Planning freeze `phase2-stage5.v1`. This section finalizes **Phase 2 Stage 5**: authenticating every Phase 2 Stage 1–4 artifact on mounted `PHASE1_ROOT`, evaluating all five experiments (`E29`, `E30`, `E11`, `E17`, `E07`) across both models (`qwen`, `gemma`) against the predeclared `STAGE5_DECISION_RANGES` (`src.phase2.gates`), publishing the immutable **`phase2_release`** bundle (`phase2/releases/<run_id>/`) with a human-readable **`phase2_review.md`** decision register, and certifying completion with a **fresh-subprocess independent readback**.

### 5.1 Entry prerequisites and complete pinned input inventory

Stage 5 runs once on **Colab CPU** (pinned `requirements-stage2-cpu.txt`, ~30–60 seconds) with **zero new model scoring, zero feature reads, zero bootstrap recomputation, zero Phase 1 mutation, and `CONFIRM` closed**. It takes the exact pinned Stage 1–4 completion references from §2.9 and §4.7:

- **Stage 1:** `phase2_plan`, `phase2_e07_audit`, `phase2_e17_auth`.
- **Stage 2:** `E29` analysis & gate (`qwen`, `gemma`), `E17` analysis (`qwen` authoritative run `20261119T174402Z-...`, `gemma`), `E07` feature releases (`qwen`, `gemma`).
- **Stage 3:** `E30` analysis (`qwen`, `gemma`), `E11` analysis (`qwen`, `gemma`).
- **Stage 4:** `E07` analysis (`qwen`, `gemma`).

All 15 references are verified via `load_phase2_artifact` / `load_phase2_plan` / `load_e29_gate` / `load_e07_audit` / `load_e17_auth`, asserting identical `plan_identity`, `dataset_fingerprint`, `confirm_excluded == True`, and matching upstream pointers (`e29_gate_ref`, `e07_audit_ref`, `e07_release_ref`, `e17_auth_ref`, `e29_analysis_ref`).

### 5.2 Predeclared decision register and evaluation rules

Per [prelim.md §12.4](prelim.md#124-decision-boundary-after-the-round) and [phase2.md §1.5](#15-predeclared-decision-ranges-and-e29-gating-of-e30-and-e11), Stage 5 evaluates each research direction per model (`qwen`, `gemma`) and combined using the frozen constants in `src.phase2.gates.STAGE5_DECISION_RANGES`, assigning exactly one of four outcomes:

- **`retain`** — effect meets the predeclared consequential threshold with 95% CI excluding zero favorably, all mandatory invariance/sanity checks pass, and no disqualifying contrary control invalidates the scope.
- **`revise`** — a genuine effect is observed, but narrowed or qualified by a control/diagnostic (e.g., E30 head–pooling interaction or off-distribution asymmetry, E11 loss-only improvement routing to E17, E17 target-only calibration without source portability, E07 restricted-cohort/view sensitivity or `bn` single-caption coverage limit, or E29 `caption_only` mirroring `image_caption`).
- **`park within tested scope`** — 95% CI lies wholly within the predeclared practically negligible bound with adequate precision (`valid_draws >= 1900`).
- **`insufficient evidence`** — interval straddles the negligible and consequential ranges, `valid_draws < 1900`, or cohort support is insufficient.

Every decision entry MUST record:
1. `direction_id` & `experiment` (`E29`, `E30`, `E11`, `E17_source`, `E17_target`, `E07_view_stability`)
2. `decision` (`retain` | `revise` | `park within tested scope` | `insufficient evidence`)
3. `exact_comparison` (metric, arm contrast, supervision label, cohort)
4. `effect` & `ci_95` (`[lower, upper]`, `valid_draws / 2000`)
5. `denominators` (`n_images`, `n_routes`, `n_primary_categories`, and for E07 `n_changed_views` + `unchanged_view_fraction`)
6. `invariance_and_sanity_checks` (exact recorded tolerances and pass/fail booleans)
7. `strongest_contrary_evidence` (e.g., E29 `caption_only` reduction, E30 asymmetric `head_mean_on_last` drop, E11 `mean_shift` vs `mean_scale` split, E17 `0.5`-threshold flips on `temp_intercept`, E07 `bn` single-caption unavailability and `restricted_changed_cohort` vs `full_screen_360` difference)
8. `remaining_alternative_explanation`

### 5.3 Frozen Stage 5 wire schema (`phase2_release`) and independent readback

- **Artifact namespace:** `phase2/releases/<run_id>/` (`KIND_RELEASE = "phase2_release"`, `SCHEMA_RELEASE = "phase2.release.v1"`).
- **Files published inside `phase2/releases/<run_id>/`:**
  - `body.json`: structured JSON containing `schema`, `phase2_version`, `plan`, `plan_identity`, `dataset_fingerprint`, `protocol_fingerprint`, `confirm_excluded` (`True`), `stage1_refs` (`phase2_plan`, `e07_audit`, `e17_auth`), `stage2_refs` (`e29`, `e29_gates`, `e17`, `e07_releases`), `stage3_refs` (`e30`, `e11`), `stage4_refs` (`e07`), `integrity_checks` (all cross-artifact lineage, invariance, self-check, and parity assertions), `experiment_summaries` (compact tidy metrics/CIs for E29, E30, E11, E17, E07 × both models), `decision_register` (per-model and combined verdicts), `post_prelim_recommendation` (`proceed_to_post_prelim` vs `optional_phase3_needed` with explicit surviving directions and open questions), and `independent_readback` (`verified: True`, `subprocess_exit_code: 0`, `verified_artifact_count`).
  - `phase2_review.md`: concise, publication-grade Markdown synthesis report (ready to copy into `phase2.md` / `prelim.md` or inspect directly) with summary tables, decision register, strongest contrary evidence, and next-step recommendation.
- **Fresh-subprocess independent readback:** After publishing the pending `phase2_release` (or via a dedicated `--verify-release` CLI command invoked in a clean `subprocess.run([sys.executable, '-u', '-m', 'src.phase2.synthesize', 'verify-release', ...])`), a fresh Python subprocess re-authenticates the published `phase2_release` `_COMPLETE.json`, its `body.json` & `phase2_review.md` SHA-256s, and every referenced Stage 1–4 completion marker on disk without sharing in-memory state.

### 5.4 Owned files and implementation scope (single chat, no parallel split needed)

Stage 5 is a single cohesive CPU synthesis module (~30–60s runtime on Colab CPU), so it runs as **one coding chat** branching from `STAGE5_BASE_SHA` on `phase2`:

| Owned files (exclusive) | Deliverable |
|---|---|
| `src/phase2/synthesize.py` | Cross-stage verifier, automatic `STAGE5_DECISION_RANGES` evaluator, tidy `phase2_release` + `phase2_review.md` publisher, fresh-subprocess `verify-release` readback CLI. |
| `scripts/phase2/synthesize_phase2.ipynb` | Colab CPU notebook (`CODE_REF = 'phase2'`, all 15 Stage 1–4 refs pre-filled in Cell 3, runs synthesis + fresh-subprocess readback and prints `phase2_review.md` + final `phase2_release` `{path, sha256}`). |
| `tests/test_phase2_synthesize.py` | Unit/fixture tests covering cross-stage lineage mismatch rejection, tampered analysis rejection, `CONFIRM` exclusion check, all 4 decision outcomes (`retain`, `revise`, `park within tested scope`, `insufficient evidence`), fresh-subprocess readback, and notebook contract. |

### 5.5 Stage 5 acceptance criteria

- [ ] **Local tests pass:** `tests/test_phase2_synthesize.py` passes (plus all Stage 1–4 tests remain green).
- [ ] **Mounted Colab CPU synthesis + fresh-subprocess readback pass:** `scripts/phase2/synthesize_phase2.ipynb` authenticates all 15 Stage 1–4 refs on mounted `PHASE1_ROOT`, publishes `phase2/releases/<run_id>/_COMPLETE.json` + `phase2_review.md`, and passes the fresh-subprocess independent readback check with exit code `0`.
- [ ] **Decision register complete & `CONFIRM` still closed:** Every direction carries an explicit decision (`retain` / `revise` / `park within tested scope` / `insufficient evidence`), exact effect/CI/denominators, strongest contrary evidence, and remaining alternative explanation; zero Phase 1 mutation and zero `CONFIRM` access.

### 5.6 Ready-to-paste prompt for Stage 5 coding chat

> **Prompt — Stage 5 (Phase 2 synthesis, decision register, and independent readback)**
> Work from `<STAGE5_BASE_SHA>` on branch `phase2` in `C:/Users/APOORV/Documents/Codex/2026-11-12/ex/work/Research-stage1-integrated`. Read `phase2.md` (§1.5, §2.9, §4.7, §5.1–5.5), `prelim.md` (§12.4), `src/phase2/gates.py` (`STAGE5_DECISION_RANGES`), and the top-level output schemas in `src/phase2/{run_e29,run_e17,run_e30,run_e11,score_e07}.py`. You own ONLY `src/phase2/synthesize.py`, `scripts/phase2/synthesize_phase2.ipynb`, and `tests/test_phase2_synthesize.py`. Do not edit Stage 1–4 modules or Phase 1 files. Implement the CPU Stage 5 synthesizer that: (1) authenticates all 15 Stage 1–4 refs (`phase2_plan`, `e07_audit`, `e17_auth`, E29 analysis & gates, E17 analysis, E07 releases, E30 analysis, E11 analysis, E07 analysis for both `qwen` and `gemma`) checking matching `plan_identity`, `dataset_fingerprint`, `confirm_excluded=True`, and all recorded invariance/self-check/parity flags; (2) extracts tidy comparison tables across all 5 experiments and evaluates every direction against `src.phase2.gates.STAGE5_DECISION_RANGES`, assigning `retain` / `revise` / `park within tested scope` / `insufficient evidence` with exact comparison, effect, 95% CI, denominators, strongest contrary evidence, and remaining alternative explanation; (3) publishes `phase2_release` (`KIND_RELEASE`, `SCHEMA_RELEASE`) under `phase2/releases/<run_id>/` containing `body.json` and human-readable `phase2_review.md`; and (4) provides a `verify-release` CLI subcommand invoked in a fresh subprocess to independently re-authenticate the release and all upstream markers on disk. Pre-fill all 15 certified Stage 1–4 `{path, sha256}` refs from `phase2.md` §2.9 & §4.7 directly in Cell 3 of `scripts/phase2/synthesize_phase2.ipynb` (`CODE_REF = 'phase2'`) so it runs out-of-the-box on Colab CPU (~30–60s) and prints `phase2_review.md` plus the final `phase2_release` ref. Add thorough unit/fixture tests in `tests/test_phase2_synthesize.py` and report commit SHA, tests passed, and Colab run readiness.

---

## Phase 2 Results Review & Decision Register — 20 November 2026

- **Release schema:** `phase2.release.v1` (`phase2-stage5.v1`)
- **Plan identity:** `d54c462c03842268a4c38ee12a82be07842179dd88b393e51b1bc5f86a9e95fa`
- **Dataset fingerprint:** `54825d236efcd88cc5252918ebce6d1dad6109ef10a244281f262204bc6cc5bd`
- **CONFIRM status:** `CLOSED` (`confirm_excluded = True`)
- **E07 coverage verdict:** `passed_restricted`

### 1. Combined Decision Register

| Direction | Experiment | Combined Decision | Qwen | Gemma |
|---|---|---|---|---|
| `E29_pooling_gap_reduction` | E29 | **revise** | `revise` | `revise` |
| `E30_pooling_feature_vs_head` | E30 | **revise** | `revise` | `revise` |
| `E11_fit_covariate_scale` | E11 | **revise** | `revise` | `revise` |
| `E17_source_calibration` | E17 | **revise** | `revise` | `retain` |
| `E17_target_calibration` | E17 | **revise** | `revise` | `revise` |
| `E07_caption_view_stability` | E07 | **revise** | `revise` | `revise` |

### 2. Per-Model Quantitative Summary (Primary 14-Category Macro, 95% Paired Bootstrap CI)

#### QWEN

| Experiment | Key Contrast / Metric | Estimate [95% CI] |
|---|---|---|
| **E29** | `image_caption` ΔG = G(last) − G(mean) (AUROC) | +0.0798 [+0.0733, +0.0866] |
| **E29** | `image_caption` residual G(mean) (AUROC) | +0.0301 [+0.0267, +0.0338] |
| **E29** | `caption_only` diagnostic ΔG (AUROC) | +0.0640 [+0.0555, +0.0725] |
| **E30** | Fixed `last` head feature gain (AUROC) | +0.0919 [+0.0853, +0.0985] |
| **E30** | Fixed `mean` head feature gain (AUROC) | +0.0054 [-0.0028, +0.0135] |
| **E30** | Head × pooling interaction (AUROC) | +0.0865 [+0.0773, +0.0958] |
| **E11** | `mean_scale` AUROC gap reduction | +0.0005 [+0.0002, +0.0008] |
| **E11** | `mean_scale` recovered fraction of G_res | 0.0166 [0.0054, 0.0277] |
| **E11** | `mean_scale` NLL / Brier gain vs identity | +0.2013 [+0.1919, +0.2111] / +0.0429 [+0.0408, +0.0450] |
| **E17** | `source_dev_c:temp` NLL / Brier gain vs raw | +0.0393 [+0.0373, +0.0413] / +0.0034 [+0.0033, +0.0036] |
| **E17** | `source_dev_c:temp_intercept` NLL / Brier gain vs raw | +0.0397 [+0.0378, +0.0417] / +0.0035 [+0.0033, +0.0037] |
| **E17** | `target_dev_c:temp` NLL / Brier gain vs raw | +0.1809 [+0.1681, +0.1938] / +0.0236 [+0.0218, +0.0255] |
| **E17** | `target_dev_c:temp_intercept` NLL / Brier gain vs raw | +0.2103 [+0.1982, +0.2227] / +0.0365 [+0.0339, +0.0391] |
| **E07** | Draw 1 vs 0 full-12 / changed-9 ΔG_last (AUROC) | +0.0044 [-0.0010, +0.0094] / +0.0059 [-0.0014, +0.0125] |
| **E07** | Draw 2 vs 0 full-12 / changed-9 ΔG_last (AUROC) | +0.0064 [+0.0010, +0.0115] / +0.0086 [+0.0014, +0.0153] |
| **E07** | Draw 1 / Draw 2 restricted-cohort ΔG_last (AUROC) | n/a / n/a |

#### GEMMA

| Experiment | Key Contrast / Metric | Estimate [95% CI] |
|---|---|---|
| **E29** | `image_caption` ΔG = G(last) − G(mean) (AUROC) | +0.0460 [+0.0404, +0.0516] |
| **E29** | `image_caption` residual G(mean) (AUROC) | +0.0157 [+0.0125, +0.0193] |
| **E29** | `caption_only` diagnostic ΔG (AUROC) | +0.0304 [+0.0244, +0.0361] |
| **E30** | Fixed `last` head feature gain (AUROC) | +0.0512 [+0.0457, +0.0567] |
| **E30** | Fixed `mean` head feature gain (AUROC) | +0.0059 [-0.0008, +0.0128] |
| **E30** | Head × pooling interaction (AUROC) | +0.0453 [+0.0369, +0.0531] |
| **E11** | `mean_scale` AUROC gap reduction | +0.0003 [+0.0000, +0.0005] |
| **E11** | `mean_scale` recovered fraction of G_res | 0.0167 [0.0004, 0.0349] |
| **E11** | `mean_scale` NLL / Brier gain vs identity | +0.0662 [+0.0613, +0.0710] / +0.0167 [+0.0152, +0.0182] |
| **E17** | `source_dev_c:temp` NLL / Brier gain vs raw | +0.0493 [+0.0454, +0.0536] / +0.0039 [+0.0036, +0.0043] |
| **E17** | `source_dev_c:temp_intercept` NLL / Brier gain vs raw | +0.0517 [+0.0479, +0.0559] / +0.0043 [+0.0039, +0.0047] |
| **E17** | `target_dev_c:temp` NLL / Brier gain vs raw | +0.1140 [+0.1026, +0.1267] / +0.0108 [+0.0093, +0.0124] |
| **E17** | `target_dev_c:temp_intercept` NLL / Brier gain vs raw | +0.1274 [+0.1166, +0.1388] / +0.0159 [+0.0144, +0.0175] |
| **E07** | Draw 1 vs 0 full-12 / changed-9 ΔG_last (AUROC) | +0.0009 [-0.0033, +0.0051] / +0.0012 [-0.0045, +0.0067] |
| **E07** | Draw 2 vs 0 full-12 / changed-9 ΔG_last (AUROC) | +0.0023 [-0.0022, +0.0068] / +0.0031 [-0.0029, +0.0091] |
| **E07** | Draw 1 / Draw 2 restricted-cohort ΔG_last (AUROC) | n/a / n/a |

### 3. Detailed Decision Notes (Strongest Contrary Evidence & Alternative Explanations)

#### QWEN
- **`E29_pooling_gap_reduction` → `revise`**: image_caption midpoint 12-route mean AUROC gap reduction Delta_G = G(last_caption) - G(caption_mean).
  - Effect: `+0.0798 [+0.0733, +0.0866]` (valid draws: `2000`).
  - Strongest contrary evidence: caption_only diagnostic Delta_G = +0.0640 [+0.0555, +0.0725]; residual image_caption G_res = +0.0301 [+0.0267, +0.0338] remains above zero.
  - Remaining alternative explanation: Independent head fits at last_caption vs caption_mean (tested by E30); caption-only lexical pooling contribution if caption_only mirrors image_caption.
- **`E30_pooling_feature_vs_head` → `revise`**: 2x2 head x pooling swap at midpoint image_caption (fixed-head feature gains & paired interaction on 12-route mean AUROC gap).
  - Effect: `+0.0919 [+0.0853, +0.0985]` (valid draws: `2000`).
  - Strongest contrary evidence: Interaction = +0.0865 [+0.0773, +0.0958] and fixed-head-mean feature gain = +0.0054 [-0.0028, +0.0135]; asymmetric head_mean_on_last drop (collapse_flagged=False).
  - Remaining alternative explanation: Asymmetric feature-distribution compatibility: last_caption heads benefit directly from caption_mean features without refitting, whereas caption_mean heads degrade when evaluated on last_caption features.
- **`E11_fit_covariate_scale` → `revise`**: target_fit_input_adapted mean_scale vs identity on caption_mean midpoint (12-route mean AUROC gap reduction; NLL/Brier companions).
  - Effect: `+0.0005 [+0.0002, +0.0008]` (valid draws: `2000`).
  - Strongest contrary evidence: mean_scale AUROC gap reduction = +0.0005 [+0.0002, +0.0008] (recovered fraction = 0.0166 [0.0054, 0.0277]); NLL gain = +0.2013 [+0.1919, +0.2111], Brier gain = +0.0429 [+0.0408, +0.0450].
  - Remaining alternative explanation: Diagonal coordinate scaling cannot correct cross-lingual rotation/subspace misalignment (E12 candidate); loss-only shifts belong with probability calibration (E17); draw-0 only until paired with E07.
- **`E17_source_calibration` → `revise`**: source_dev_c_calibrated (source_dev_c:temp_intercept) vs raw & FIT-prior on midpoint last_caption SCREEN (12-route mean NLL & Brier gain).
  - Effect: `+0.0397 [+0.0378, +0.0417]` (valid draws: `2000`).
  - Strongest contrary evidence: Ranking (AUROC/AP) is strictly invariant (max_diff=0.00e+00); 0.5-decision flip rate on source_dev_c:temp_intercept = 0.0026; NLL gain vs FIT-prior = -0.1003 [-0.1181, -0.0836].
  - Remaining alternative explanation: Global 1- or 2-parameter logit scaling cannot fix per-category or ranking transfer errors; source DEV-C may under-correct target-specific logit shifts.
- **`E17_target_calibration` → `revise`**: target_dev_c_supervised_calibration (target_dev_c:temp_intercept) vs raw & FIT-prior on midpoint last_caption SCREEN (12-route mean NLL & Brier gain).
  - Effect: `+0.2103 [+0.1982, +0.2227]` (valid draws: `2000`).
  - Strongest contrary evidence: Ranking (AUROC/AP) is strictly invariant (max_diff=0.00e+00); 0.5-decision flip rate on target_dev_c:temp_intercept = 0.0783; NLL gain vs FIT-prior = +0.0702 [+0.0603, +0.0800].
  - Remaining alternative explanation: Global 1- or 2-parameter logit scaling cannot fix per-category or ranking transfer errors; target DEV-C requires 180 labeled target examples.
- **`E07_caption_view_stability` → `revise`**: Frozen-head alternate-caption draws 1 & 2 vs draw 0 (draw_2 changed_9_g_last_shift worst-case AUROC gap shift).
  - Effect: `+0.0086 [+0.0014, +0.0153]` (valid draws: `2000`).
  - Strongest contrary evidence: Verdict=passed_restricted: unavailable single-caption languages=['bn'] (bn has 0 alternate captions in XM3600 SCREEN); sign_reversed=False; draw_1 restricted g_last_shift=n/a.
  - Remaining alternative explanation: Frozen-head E07 tests evaluation-view sensitivity with heads trained on draw 0; it does not test training-view refit sensitivity or separate wording from caption content.

#### GEMMA
- **`E29_pooling_gap_reduction` → `revise`**: image_caption midpoint 12-route mean AUROC gap reduction Delta_G = G(last_caption) - G(caption_mean).
  - Effect: `+0.0460 [+0.0404, +0.0516]` (valid draws: `2000`).
  - Strongest contrary evidence: caption_only diagnostic Delta_G = +0.0304 [+0.0244, +0.0361]; residual image_caption G_res = +0.0157 [+0.0125, +0.0193] remains above zero.
  - Remaining alternative explanation: Independent head fits at last_caption vs caption_mean (tested by E30); caption-only lexical pooling contribution if caption_only mirrors image_caption.
- **`E30_pooling_feature_vs_head` → `revise`**: 2x2 head x pooling swap at midpoint image_caption (fixed-head feature gains & paired interaction on 12-route mean AUROC gap).
  - Effect: `+0.0512 [+0.0457, +0.0567]` (valid draws: `2000`).
  - Strongest contrary evidence: Interaction = +0.0453 [+0.0369, +0.0531] and fixed-head-mean feature gain = +0.0059 [-0.0008, +0.0128]; asymmetric head_mean_on_last drop (collapse_flagged=False).
  - Remaining alternative explanation: Asymmetric feature-distribution compatibility: last_caption heads benefit directly from caption_mean features without refitting, whereas caption_mean heads degrade when evaluated on last_caption features.
- **`E11_fit_covariate_scale` → `revise`**: target_fit_input_adapted mean_scale vs identity on caption_mean midpoint (12-route mean AUROC gap reduction; NLL/Brier companions).
  - Effect: `+0.0003 [+0.0000, +0.0005]` (valid draws: `2000`).
  - Strongest contrary evidence: mean_scale AUROC gap reduction = +0.0003 [+0.0000, +0.0005] (recovered fraction = 0.0167 [0.0004, 0.0349]); NLL gain = +0.0662 [+0.0613, +0.0710], Brier gain = +0.0167 [+0.0152, +0.0182].
  - Remaining alternative explanation: Diagonal coordinate scaling cannot correct cross-lingual rotation/subspace misalignment (E12 candidate); loss-only shifts belong with probability calibration (E17); draw-0 only until paired with E07.
- **`E17_source_calibration` → `retain`**: source_dev_c_calibrated (source_dev_c:temp_intercept) vs raw & FIT-prior on midpoint last_caption SCREEN (12-route mean NLL & Brier gain).
  - Effect: `+0.0517 [+0.0479, +0.0559]` (valid draws: `2000`).
  - Strongest contrary evidence: Ranking (AUROC/AP) is strictly invariant (max_diff=0.00e+00); 0.5-decision flip rate on source_dev_c:temp_intercept = 0.0017; NLL gain vs FIT-prior = +0.0556 [+0.0380, +0.0727].
  - Remaining alternative explanation: Global 1- or 2-parameter logit scaling cannot fix per-category or ranking transfer errors; source DEV-C may under-correct target-specific logit shifts.
- **`E17_target_calibration` → `revise`**: target_dev_c_supervised_calibration (target_dev_c:temp_intercept) vs raw & FIT-prior on midpoint last_caption SCREEN (12-route mean NLL & Brier gain).
  - Effect: `+0.1274 [+0.1166, +0.1388]` (valid draws: `2000`).
  - Strongest contrary evidence: Ranking (AUROC/AP) is strictly invariant (max_diff=0.00e+00); 0.5-decision flip rate on target_dev_c:temp_intercept = 0.0377; NLL gain vs FIT-prior = +0.1312 [+0.1189, +0.1434].
  - Remaining alternative explanation: Global 1- or 2-parameter logit scaling cannot fix per-category or ranking transfer errors; target DEV-C requires 180 labeled target examples.
- **`E07_caption_view_stability` → `revise`**: Frozen-head alternate-caption draws 1 & 2 vs draw 0 (draw_2 changed_9_g_last_shift worst-case AUROC gap shift).
  - Effect: `+0.0031 [-0.0029, +0.0091]` (valid draws: `2000`).
  - Strongest contrary evidence: Verdict=passed_restricted: unavailable single-caption languages=['bn'] (bn has 0 alternate captions in XM3600 SCREEN); sign_reversed=False; draw_1 restricted g_last_shift=n/a.
  - Remaining alternative explanation: Frozen-head E07 tests evaluation-view sensitivity with heads trained on draw 0; it does not test training-view refit sensitivity or separate wording from caption content.

### 4. Next-Step Recommendation

- **Recommendation:** `proceed_to_post_prelim`
- **Surviving directions:** `E29_pooling_gap_reduction`, `E30_pooling_feature_vs_head`, `E11_fit_covariate_scale`, `E17_source_calibration`, `E17_target_calibration`, `E07_caption_view_stability`
- **Parked directions:** none
- **Unresolved directions:** none
- **CONFIRM:** CLOSED (untouched throughout Phase 1 and Phase 2)
