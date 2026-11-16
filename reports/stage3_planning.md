# Stage 3 planning review — 16 November 2026

Status: stage3-plan.v1 specified; implementation, numerical fixtures and production fits not performed. The current task is documentation/planning. No Python, notebooks, dependencies, model/experiment JSON configurations or previously completed artifacts are changed by this review.

Read docs/stage3_contracts.md for the numerical/role/cache/gate specification and docs/stage3_ownership.md for shared-interface and worker handoff. These implement the existing G9-Q6 scheduled block, not new experiments or a claimed novel result.

## Sources reviewed and decisions frozen

Reviewed AGENTS.md/plan.md, research.md, the shared protocol/G0 and E01-E04 cards in prelim.md, M1 frozen-probe/control principles, M2 supervision/coordinate warnings and Stage 2 accepted-reader/family semantics. Checked the pinned scikit-learn 1.6.1 logistic/loss implementation for sample-weight/L2/intercept conventions; primary-source links are placed beside the objective in the contract. User requirements for cross-script reuse, limited meaningful checkpoints, all four native contexts, both models, fixed partitions/masks and no new manual annotation remain binding.

The specification freezes source-language/site systems and visual/E03 exceptions; 25 independent binary logits; weighted source-FIT-only statistics with a fixed variance guard; dense centering versus sparse scaling; explicit fractional pooled-image weights; sum-loss/L2 C semantics; one solver/tolerance/deterministic retry policy; source-DEV-S macro selection with smallest-C ties; no refit after selection; and true/control target namespaces. No real SCREEN outcomes, DEV-C selection, raw-image processing, VLM execution, extra vocabulary fitting, hidden capacity map or new annotation work is introduced.

## Workload and reuse check

Current catalog gives 66 VLM systems per model (32 combined, 32 caption-only, 2 visual) plus 13 E03 = 145. Four C values x 25 category heads = 14,500 ordinary nominal fits. Three matched randomized-target pipelines add 43,500, totaling 58,000 before retries. This exposes the potential CPU cost rather than promising Stage 3 is automatically cheap relative to Stage 2. Actual dimensions/nonzeros/solver timings must drive resource projections before bulk work.

Cache source statistics once per exact input/row/weight recipe, then reuse them for all categories, Cs and control seeds. Retain completed fit blocks, source development logits/convergence/selection records, shared image-vector/mask assignments and G0 diagnostic/null blocks. Selected systems reference original FIT weights rather than copying/refitting. Numeric sessions verify immutable inputs at open/resume and retain bounded in-memory arrays; no repeated whole-image scans, per-head refitting of scalers, default persisted standardized feature copies or per-iteration checkpoint flood. Stage 4 later saves canonical scientific predictions once for E01/E19/E25 and related consumers.

Checkpoint boundaries are 25-head C blocks, system selection indexes and 100-draw null blocks, with explicit immutable restart references. Loss on hard interruption is bounded to the unfinished block, not falsely described as zero. Ordinary/G0 workers share the coordinator's one fit/scaler/selection engine, preventing drift across scripts and target assignments.

## G0 and interpretation check

Three fixed seeds use within-partition whole image-label vectors with masks, shared across all languages/models/recipes and preserving image weights. Every control reruns the same C-selection process using its permitted source development assignment; it does not borrow ordinary selected hyperparameters. Structural identity/role/solver/mask violations block acceptance. SCREEN control scores are the declared validity-only exception to ordinary scientific evaluation being Stage 4.

The global anomaly screen fixes its full recipe/category/seed family and max-deviation statistic in advance, uses 1,999 additional shared within-SCREEN vector permutations without refitting, and reports a plus-one conditional global p with p<=0.01 flagged for pipeline investigation. This addresses multiple diagnostic comparisons and chance variation without insisting each finite-sample AUROC equals 0.5. It is not a task-contribution hypothesis test, proof of no leakage or a per-category significance claim. Its draws are separate from later 2,000 paired-image scientific bootstraps. No control seed is rerolled to pass; any repair retains original evidence and receives a versioned identity.

## Current evidence and remaining acceptance

The user reported complete Qwen/Gemma extraction (12 releases each) and a complete E03 checkpoint on 16 November. Earlier exact pointers and source evidence remain in AGENTS.md and Stage 2 reports. The user is assembling/running validate_stage2.ipynb; no passed Stage 2 suite result was supplied when this document was prepared. Thus Stage 3 design may be finalized and implementation fixtures may be prepared, but actual fitting needs the accepted passed suite and verified inputs. No completed G0 gate, model signal or research finding is inferred from raw extraction success.

Before workers start, the coordinator must implement/test the shared numerical/cache/reader interfaces and publish an immutable implementation base. Before bulk production, run the specified first-source true/seed-0 units with measured resource/serialization/equality checks; reuse them afterward. Before Stage 4 ordinary scoring, verify all required candidates/selections, target assignments, masks, convergence, source-only choices, G0 diagnostics and independent persistent readback, then publish the accepted system registry. All these implementation/runtime items remain pending.
