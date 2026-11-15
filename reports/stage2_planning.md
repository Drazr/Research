# Stage 2 planning handoff — 15 November 2026

Status: stage2-plan.v1 specified; implementation and execution not started. This report documents a planning review, not a software test or actual data/model run. The user explicitly requested no code in this planning turn.

## Reviewed context and decisions

Reviewed AGENTS.md, plan.md, research.md, the shared protocol and scheduled cards in prelim.md, relevant M1/M2 methodological/interpretation constraints, Stage 1 contracts and the existing extraction/cache/storage interfaces. Preserved the existing scientific scope, 25-target/unknown-mask data, frozen partitions, all four languages, both pinned models and no-new-human-work boundary. No new literature/novelty claims were added.

The full operational specification is docs/stage2_contracts.md; parallel file ownership and required executable-interface freeze are docs/stage2_ownership.md. plan.md connects both to the five-stage pipeline. AGENTS.md and README.md point future chats to these current instructions. Scientific source files, JSON configurations, notebooks, Python modules, dependency pins and dataset fingerprints are unchanged.

Finalized: per-model readiness gates, three extraction conditions, exact core site/pooling inventory, explicit unsupported E04 arms, raw float32 storage, one image-only reference per image with combined-visual audits, 32-image chunks and deterministic row ordering, immutable append-only publication, checksummed resume/conflict rules, layered cache identities and role-specific release indexes. Finalized 13 E03 input families, their fixed FIT-only analyzers/vocabulary/IDF formulas and sparse/dense artifacts. E03 supervised fitting and scoring are explicitly assigned to Stages 3/4, so preprocessing is not mistaken for a completed experiment.

Arithmetic review: FIT+DEV-S+DEV-C+SCREEN = 2,880 eligible images. Four contexts give 11,520 rows in each caption condition. Two caption conditions plus one image-only condition yield 25,920 unique forward inputs per model before repeats. Independent 32-image partition chunking gives 68+6+6+12 = 92 image chunks, three conditions = 276 logical condition-chunks per model. E03 has 4 word + 4 character + 4 length + 1 pooled character = 13 families. CONFIRM's 720 images are excluded from preprocessing.

## Readiness and acceptance register

| Item | Status and evidence |
|---|---|
| Stage 2 design/contract/ownership specification | Written; planning only |
| Shared Stage 2 executable interfaces and immutable worker-base SHA | Pending implementation; this planning commit is not that base |
| Common CPU dataset and validation | User supplied passed logs; immutable manifest in AGENTS.md |
| Both model runtime configurations | User supplied passed logs; future launches still need matching checkout/config identities |
| Gemma real model validation | User supplied passed validation log; payload not independently inspected here |
| Qwen post-alignment-fix real model validation | No passing result supplied yet |
| Final Stage 1 validity smoke, each model | No passing results supplied yet |
| Coordinator-accepted per-model readiness receipt | Pending actual artifact verification and missing passes |
| Stage 2 implementation and meaningful software tests | Not performed; no code written |
| Actual first FIT extraction/repeats/readback | Pending |
| Both complete model feature inventories and release indexes | Pending |
| E03 vocabulary/IDF/transform artifacts | Pending |
| Stage 2 integration/completion | Pending all required implementation/runtime evidence |
| Full G0 controls, probe fitting, scores and scientific interpretation | Later stages, not completed by planning |

## Review boundaries and next handoff

The specification defines failure behavior rather than promising that every input fits the L4/token limit. Invalid/overlong inputs and model failures stop the affected release; they do not silently alter the common cohort. Actual dimensions/backend/site availability are resolved from accepted model reports, not guessed architecture constants. Existing feature/state inspection and tokenizer fixtures do not certify future production correctness.

No new manual annotation/semantic checking, optional reserve experiments, cross-language maps, tuned/logit lenses, activation editing or base-model training were introduced. Optional unvalidated E04 positions stay explicitly unavailable. The common labels permit caption-predictability comparisons, not verified mention or language-only causal claims. Plan finalization does not establish novelty, publication viability or model signal.

Next implementation handoff: complete/verify outstanding Stage 1 evidence, then have the coordinator commit shared Stage 2 schemas/helpers and publish the exact implementation base for the extraction/E03 workstreams. Those workstreams may implement while independent readiness issues are resolved, but no model's bulk extraction starts before its actual gate passes. E03 preprocessing can use the passed common data independently. Any subsequent scientific or extraction-policy change must be versioned before affected outcomes.
