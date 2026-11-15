# Stage 2 planning handoff — 15 November 2026

Status: stage2-plan.v1 finalized, review revision 2; implementation and execution not started. This report documents a planning review, not a software test or actual data/model run. The user requested specifications without code; the follow-up review preserves that boundary.

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
| Qwen post-alignment-fix separate model-validation report | Pointer not supplied in this chat; find/verify compatible existing report before asking for any rerun |
| Final Stage 1 validity smoke, Qwen | User explicitly identified the latest Qwen output as a passed run_validity_smoke execution with no blockers; pointer below |
| Final Stage 1 validity smoke, Gemma | User explicitly identified the latest Gemma output as a passed run_validity_smoke execution with no blockers; pointer below |
| Coordinator-accepted per-model readiness receipt | Pending typed payload/checksum/identity verification, including the separate Qwen model-validation report |
| Stage 2 implementation and meaningful software tests | Not performed; no code written |
| Actual first FIT extraction/repeats/readback | Pending |
| Both complete model feature inventories and release indexes | Pending |
| E03 vocabulary/IDF/transform artifacts | Pending |
| Stage 2 integration/completion | Pending all required implementation/runtime evidence |
| Full G0 controls, probe fitting, scores and scientific interpretation | Later stages, not completed by planning |

## Review boundaries and next handoff

The specification defines failure behavior rather than promising that every input fits the L4/token limit. Invalid/overlong inputs and model failures stop the affected release; they do not silently alter the common cohort. Actual dimensions/backend/site availability are resolved from accepted model reports, not guessed architecture constants. Existing feature/state inspection and tokenizer fixtures do not certify future production correctness.

No new manual annotation/semantic checking, optional reserve experiments, cross-language maps, tuned/logit lenses, activation editing or base-model training were introduced. Optional unvalidated E04 positions stay explicitly unavailable. The common labels permit caption-predictability comparisons, not verified mention or language-only causal claims. Plan finalization does not establish novelty, publication viability or model signal.

Next implementation handoff: have the coordinator commit shared Stage 2 schemas/helpers/evidence bridge and publish the exact implementation base while verifying outstanding evidence pointers/receipts. Workers may implement while independent readiness verification is resolved, but no model's bulk extraction starts before its actual gate passes. E03 preprocessing can use the passed common data independently. Any subsequent scientific or extraction-policy change must be versioned before affected outcomes.

## Final review revision 2

The user clarified that the two latest passed logs came from run_validity_smoke.ipynb. Record them as actual user-reported final smoke successes, not unspecified smoke-directory outputs. Relative to PHASE1_ROOT:

- Qwen: `smoke/280535919b08cad218a47706a293d398b3ee691200becc37d68eb9b6573ba16c/20261115T051925Z-97628c94b3554efd8feb02e243f47ca4`.
- Gemma: `smoke/ba079306905ee4bbcfbaa8de626437ee3ce2832e2de005abb72a78ee4e6f40c7/20261115T051949Z-0027b8f47b434ef682c03a5fe266b8cf`.

This review did not read their payloads or connect to the actual L4/Drive runtime. It separates execution status supplied by the user from the future coordinator's automatic verification. No passed run is erased by a missing chat pointer, documentation change or absence of local GPU access; no richer model-validation condition/site report is invented from a final smoke summary either.

The review checked the specification against current FeatureBatch/extraction/storage interfaces and closed these operational gaps without code:

| Review item | Final decision |
|---|---|
| Bulk extraction currently returns vectors but discards processing evidence | One coordinator bridge returns vectors and actual evidence from the same existing forward, with first-FIT equality to the public path; no duplicated inference or bulk attention inspection |
| Users repeatedly had to replace paths across fresh notebook runtimes | One verified plan supplies dataset/config/evidence settings to workers, with fail-fast preflight before weight loading; HF secret access remains per-runtime |
| One generic plan could accidentally block CPU E03 on unrelated GPU readiness | Separate shared data/text plan, per-model readiness/job plans and final suite integration index; kinds and payload checks are explicit |
| Missing report paths could cause needless expensive reruns | Prefer supplied paths; otherwise discover only typed compatible artifacts in the declared namespace, never the newest folder; verify identity and retain original provenance |
| E03 family publication could unnecessarily delay fitting until SCREEN transformed | Immutable FIT-preprocessor and split releases permit complete FIT/DEV-S handoff while other transformation/extraction work continues |
| Stage 1 timing/VRAM could be mistaken for a bulk-job guarantee | Actual first-chunk dimensions, token lengths, memory, throughput and byte counts drive projections; preserve policy on failures and do not guess family constants |
| Existing smoke successes could be misclassified or overinterpreted | Both final smoke passes are recorded; coordinator receipt and full later G0 remain separate acceptance gates |

The existing arithmetic, all three conditions, 32-image chunking, E03 recipes, split/role guards, no-new-human-work boundary, optional-site limits and later-stage responsibilities were rechecked and retained. Only Markdown documentation changed. Source code, notebooks, JSON configurations, dependency pins, data and inference arithmetic remain untouched. This is a finalized design for subsequent implementation; it is not a claim that any future implementation is already tested or that research novelty/results are established.
