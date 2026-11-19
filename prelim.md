# G9-Q6 preliminary designs: initial set and later reserve

Prepared 14 November 2026 from [M1.md](M1.md), [M2.md](M2.md) and the ongoing G9-Q6 formulations in [research.md](research.md).

**Purpose:** discover reproducible, interpretable phenomena; determine which directions merit further preliminary work; and learn which questions are answerable with the released data. The first set can leave several strong directions open. It does not select a winning explanation in advance.

**Reviewed 14 November 2026. The strict initial set contains 12 experiment families: E01-E05, E14, E19 and E24-E28.** Each has an a-priori G9-Q6 question and a comparison that can be fixed without assuming an outcome from E01-E04. There are also **15 reserved later-preliminary families**, an uncounted G0 validity gate, and the uncounted E16 selection/reporting guard. The previous catalog mixed these phases; the review below corrects that assignment.

**Five additions are genuinely new: E24-E28.** E05, E14 and E19 are revised existing designs made independent of outcome triggers, not new additions. All original IDs remain traceable. There are 27 counted families across both phases; layers, routes, seeds, category pairs and minor variants remain within families. A distinct analysis can reuse predictions and need no extra VLM inference; it is not another independent replication.

**Phase 1 status.** Six experiments were completed (E01–E04, E19 and E25), and G0 passed as a separate validity gate. The other six initial designs and all 15 reserve designs remain unrun. See [the Phase 1 results review](phase1.md#phase-1-results-review--19-november-2026). No human annotation, semantic validation, new image/caption collection or new gold dataset is required.

## 1. Exact research scope

- **Q6:** can another context's readout be incompatible while category information remains accessible to a readout fitted within the target context?
- **G9:** how do category discrimination, readout portability and probability/correctness reliability relate?
- **Languages:** English, Hindi, Bengali and Telugu throughout. They share image-presence labels; they do not have separate translated visual gold.
- **Models:** one Qwen open VLM checkpoint and one Gemma open VLM checkpoint for the first reference comparison. Record exact checkpoint/revision, processor and chat-template configuration before inference; no specific version is presumed here. Analyze them separately, then compare patterns. Different outcomes can support a bounded model-family question rather than automatically invalidate a direction.
- **Targets:** the released 25 broad category-presence bits, with invalid cells masked. This phase predicts category presence through diagnostic readouts. It does not require generated captions or a VLM's natural-language judgments. G9 correctness targets, where used, are correctness of those diagnostic category decisions.
- **Excluded:** G7's joint generation/reporting study, the standby questions as independent projects, native answer-quality scoring, translated question construction, mention extraction, cultural/OCR/instance/relation gold, activation patching, steering, model editing and base-VLM training.

The input condition is **a real image plus an independently written original caption**, not a semantically equivalent translated question. Outcomes can identify context-dependent accessibility/compatibility and useful follow-up paths. They cannot by themselves explain a pure language-only benchmark penalty, prove missing visual knowledge or establish causal use of a decoded feature.

## 2. Shared concrete protocol

### 2.1 Released data and immutable image-group partitions

Use the original XM3600 captions and images with the shared [annotation CSV](https://github.com/uriberger/cross_lingual_diff_in_descs/blob/main/data/xm3600_annotation.csv). The [annotation repository](https://github.com/uriberger/cross_lingual_diff_in_descs#object-mentions-dataset) distinguishes original captions from English translations; do not accidentally use its non-English `caption` field as native text when that field contains the English translation. Prefer the original XM3600 release; the original-text field in derived records is `orig`. Bengali/Telugu derived mention synsets are unnecessary.

The planning audit used the previously reviewed Git blob `6b99def06a621c570edf074c26741868a4df7353`, checked its Git content hash, and found the expected 3,600 image rows/25 roots. This is a reproducibility/data-integrity check, not fresh human verification of annotation correctness.

Default partitions are already deterministically defined at **image level**:

| Partition | Images | Allowed role |
|---|---:|---|
| FIT | 2,160 | Readout fitting, feature statistics, source PCA, training anchors and inner cross-fitting |
| DEV-S | 180 | Declared source/target model, feature, layer and mapping selection |
| DEV-C | 180 | Probability calibration and source confidence-policy fitting |
| SCREEN | 360 | Reusable exploratory evaluation for this and later preliminary rounds |
| CONFIRM | 720 | Untouched later confirmation after a focused question/protocol is fixed |

Assignment rule: canonicalize each hexadecimal image ID to 16 lowercase characters; sort IDs by the bytes of `SHA256("G9Q6-first-set-20261114-v1" + "|" + image_id)`; allocate the contiguous blocks in the table order. The seed is fixed before model results and is not re-rolled for a favorable effect. All captions, languages, input variants and image partners inherit the image's partition. Repeated captions never become independent train/test images. Existing label counts can define support before outcomes are observed; no model features or predictions from CONFIRM are accessed during preliminary work.

If the released images/original four-language records are incomplete in an implementation, G0 first defines the common valid-image intersection and logs exclusions, then recomputes a versioned split **before any model scores**. Do not silently apply the audited 3,600-row counts to a smaller collection. The partition proportions are a concrete initial default, not a statistical-power guarantee.

### 2.2 Support, masks and what counts as estimable

Keep all 25 targets in the data manifest. A classifier requiring two classes is not fitted when FIT lacks one. Low-support estimates remain explicitly descriptive or undefined; they are not set to zero. The primary macro cohort is fixed from released labels before model outputs: at least 20 known positive and 20 known negative images in FIT and SCREEN, and at least 10 of each in DEV-S and DEV-C. This is a minimum reporting rule, not proof of adequate precision. Per-stratum/correctness analyses have additional support requirements.

The automatic audit gives **14 primary categories** under this split. Positive counts below are shared by all four language arms; full negative/unknown counts are in the planning audit. No category is removed from the release or declared scientifically unimportant.

| Category | FIT + | DEV-S + | DEV-C + | SCREEN + | Primary cohort |
|---|---:|---:|---:|---:|---|
| vehicle.n.01 | 421 | 38 | 30 | 72 | Yes |
| timepiece.n.01 | 34 | 4 | 2 | 1 | Low-support secondary |
| electronic_equipment.n.01 | 114 | 12 | 6 | 19 | Low-support secondary |
| plaything.n.01 | 72 | 6 | 6 | 5 | Low-support secondary |
| building.n.01 | 884 | 83 | 77 | 147 | Yes |
| clothing.n.01 | 765 | 58 | 54 | 112 | Yes |
| animal.n.01 | 289 | 19 | 35 | 45 | Yes |
| writing_implement.n.01 | 11 | 2 | 0 | 3 | Low-support secondary |
| kitchen_utensil.n.01 | 54 | 11 | 3 | 16 | Low-support secondary |
| furniture.n.01 | 698 | 55 | 43 | 135 | Yes |
| food.n.01 | 397 | 33 | 28 | 74 | Yes |
| musical_instrument.n.01 | 27 | 5 | 2 | 8 | Low-support secondary |
| weapon.n.01 | 30 | 3 | 5 | 4 | Low-support secondary |
| sun.n.01 | 51 | 5 | 5 | 10 | Low-support secondary |
| jewelry.n.01 | 61 | 3 | 9 | 10 | Low-support secondary |
| body_of_water.n.01 | 342 | 32 | 34 | 57 | Yes |
| plant.n.02 | 1109 | 97 | 109 | 182 | Yes |
| hand_tool.n.01 | 13 | 0 | 0 | 3 | Low-support secondary |
| sky.n.01 | 822 | 79 | 79 | 138 | Yes |
| tableware.n.01 | 248 | 17 | 14 | 48 | Yes |
| bag.n.01 | 166 | 12 | 12 | 33 | Yes |
| home_appliance.n.01 | 45 | 3 | 1 | 4 | Low-support secondary |
| mountain.n.01 | 294 | 25 | 25 | 59 | Yes |
| body_part.n.01 | 988 | 71 | 79 | 147 | Yes |
| person.n.01 | 911 | 77 | 66 | 135 | Yes |

Image `6a20677eee04fd08` has `plant.n.02 = 4` and falls in SCREEN in this split. That one category cell is unknown, not positive, negative or a count. Mask it consistently across languages and methods. Multiplicity/co-occurrence calculations that require a complete vector must exclude an incomplete vector or explicitly use a known-cell definition; do not silently infer the missing bit. Negative labels follow the released blank-cell convention and remain fallible.

All method comparisons use the **same known image-category cells**. An aggregate whose membership changes must show the changed support and a matched-cohort companion. In subgroup or correctness analyses, require at least 20 of each relevant outcome class for an inferential summary; otherwise show counts and mark the result insufficiently resolved. These thresholds are conservative reporting gates, not claims of power.

### 2.3 Fixed context, feature and readout definitions

For the default context, put the image before the unchanged original caption in the checkpoint's native chat template. A shared neutral field label such as `Caption:\n` is identical across languages; no category question, native verbalizer, answer, root label or image-category target is included. At the default site, extract the state at the **last actual caption-content token**, before any appended assistant-generation boundary. Store token spans and masks from the processor so punctuation, image and role tokens are not silently mixed. E04 tests other declared sites. E24 is the declared input-order exception and always uses a common terminal position after both modalities, not the default last-caption site.

Use four fixed decoder depths `ceil(L/4), ceil(L/2), ceil(3L/4), L`, deduplicated for the checkpoint; `L` is its decoder block count. All four E01 profiles are reported. Additional initial tests with a single primary depth use the fixed midpoint `ceil(L/2)`, avoiding an outcome-selected depth. Record a consistent block-output residual-stream hook: do not silently treat a final-normalized state as equivalent to earlier unnormalized states. E02 adds exposed visual-encoder/projector features; these are distinct sites, not assumed to have the same dimension or to be language-conditioned. Model runs use evaluation mode and a fixed processor policy. No gold answer is teacher-forced into any measured state.

Choose one released caption per image/language by a fixed hash of image ID, language and caption index (draw key 0). E07 uses draw keys 1 and 2 as additional selections; a repeated choice is recorded, not invented into a new caption. Keep the same image population and image weighting. Caption indices across languages have no translation meaning.

The default readout has 25 independent sigmoid outputs, **not a 25-way softmax**. Fit unweighted binary logistic loss with L2 regularization and an intercept on known FIT cells. Standardize features from source FIT only, with a fixed numerical variance floor. A shared inverse-regularization grid `C ∈ {0.01, 0.1, 1, 10}` is selected per source/depth on DEV-S primary-cohort macro AUROC; use a deterministic tie rule favoring stronger regularization. Log convergence failures. Do not add class weighting silently: it changes the probability target's effective training prior.

A source-trained system includes its scaler, depth/site choice and head. Applying it to a target context keeps those fixed unless a card explicitly defines adaptation. Target-trained heads are ordinary supervised comparators, not an oracle upper bound. The primary threshold is 0.5; threshold-tuned outcomes, if added, are separately labeled.

When a card adds caption or language views, each independent image keeps its declared total loss weight, and sample weights must have the same total across matched arms. Preserve the loss/L2 normalization convention as well as C: multiplying view rows while silently increasing effective training weight changes regularization. Intercepts are unpenalized unless explicitly declared otherwise. Record actual solver objectives and normalization during implementation.

### 2.4 Different measurements and their events

For known category bit `y`, let `p` be the presence score and `d = 1[p >= 0.5]` the diagnostic decision. Presence calibration concerns `p` versus `y`. Correctness confidence concerns the separate event `z = 1[d == y]`; `c = p` for a positive decision and `c = 1-p` for a negative decision, hence `max(p,1-p)` under that 0.5 rule. A learned correctness head predicts `z`, not presence. None of these is a verified free-caption hallucination score or the base VLM's self-reported confidence.

Record per-category AUROC and average precision for discrimination, Brier score and binary log-loss for probabilities, and sensitivity/specificity plus F1 or balanced accuracy for thresholded decisions. Include FIT-prevalence probability and always-absent decision references. Log-loss uses a fixed numerical clipping convention. ECE/reliability diagrams are descriptive complements, not the sole calibration evidence. Macro summaries over the fixed primary cohort are primary; micro/low-support summaries cannot replace them. AUPR/AP and correctness detection depend on their respective event prevalences.

For transfer discrimination use `gap(s,t,k,l) = D(t,t,k,l) - D(s,t,k,l)` at a matched site/depth and identical target cells. For loss scores state the direction explicitly, e.g. `NLL(s,t) - NLL(t,t)`. Show absolute values as well as differences; a tiny penalty preserving a poor baseline is not strong performance. Never compare Qwen and Gemma raw feature coordinates directly.

### 2.5 Uncertainty, exploratory reuse and result integrity

Use paired image-group resampling for method/language differences, retaining all relevant predictions/views of a sampled image together. A concrete default is 2,000 bootstrap draws with 95% intervals; these are exploratory uncertainty summaries, not a substitute for effect size or automatic confirmation after a large search. Undefined one-class draws stay undefined with their valid-draw count reported; do not fabricate an AUROC of zero. E23 uses disjoint image-pair clusters for its primary within-category paired contrast.

The initial-set report contains all declared route/depth/category results, unsuccessful variants and support failures. A technical support failure means unavailable, not scientific evidence against a direction. If formal multiple hypothesis tests are added, define the comparison family and adjustment before interpreting its selected extrema. Cross-category correlations and shared images mean the 16 grid cells and repeated captions are not independent replicates. Convex-head optimizer seeds are not new independent evidence; nonlinear/mixture fitting variation and existing-caption variation should be named separately.

SCREEN is explicitly exploratory and reusable for choosing later preliminary questions. Findings identified from it remain exploratory even with narrow resampling intervals. DEV-S chooses fitted systems; DEV-C calibrates them; SCREEN does not quietly become another tuning set without a logged new preliminary version. **CONFIRM remains closed** until the later focused study's protocol is fixed. No acceptance criterion guarantees a publication or global novelty.

## 3. G0: automatic validity gate — not counted as an experiment

**Phase 1 status:** passed and unflagged (p = 0.362).

- Verify unique canonical image IDs, exact image/caption/label joins, raw original-language field provenance, duplicate captions, missing files and the actual common four-language population. Use lossless integer/hex conversion; large IDs must not pass through lossy floating-point parsing.
- Check label alphabet, unknown masks, all split/image overlaps and category supports. Verify that labels, category indices and target language metadata do not enter the representation input through filenames, prompt construction or preprocessing.
- Check processor/token-span alignment, deterministic repeated features, feature shapes and attention visibility. A visual prefix with no access to later text should not acquire unexplained language differences; a model with a different visibility pattern must be documented rather than forced into that expectation.
- Fit prior/constant references and three fixed within-partition whole-vector label permutations, shared across languages and carrying masks with their vectors. This is an image-level random-target control adapted from M1, not the original word-type selectivity test. Under imbalance, high accuracy can be trivial; inspect discrimination against its appropriate null and fitting/evaluation behavior.
- If a control suggests identity leakage, label leakage, wrong token extraction, one-class fitting or an optimizer failure, stop substantive interpretation and repair the automatic pipeline. No image review or new human semantic judgments are a repair step.

## 4. Design review: what qualifies for the initial set

The review used each G9-Q6 question and the methods in M1/M2, not assumed model results. Initial membership requires: a substantive pre-result question; a fixed, interpretable comparison; valid released-data targets; no new human work; and a distinct decision beyond a renamed metric, seed or parameter variant. A dependency on an already specified feature extraction or head fit is an **execution dependency**, not an assumed favorable result. An unavailable input serialization or insufficient released-label support is a technical eligibility gate, not a positive scientific result.

Among preplannable studies, the initial set emphasizes broad discovery contrasts. Tailored repairs, localization of an observed pattern and higher-complexity explanatory readouts remain in the later reserve. Their later status is a design choice about what uncertainty to address next, not a claim that science forbids preplanning them. Basic per-category/length summaries and caption-repeat checks can accompany initial reporting without being inflated into more independent experiments. No strong outcome or cross-language deficit is assumed.

### 4.1 Review decisions and corrections

| Review finding | Decision |
|---|---|
| The original 23-card catalog included result-triggered branches under a first-set title | Separate 12 strict initial families from 15 later-preliminary families; keep all original IDs |
| Capacity, common-rule fitting and usable confidence are core G9-Q6 references even before a gap exists | Reframe E05, E14 and E19 as initial tests; count them as existing designs, not additions |
| E16 audits how existing depth profiles are selected | Make it an uncounted shared reporting guard; preserve its card for traceability |
| Input-order changes are confounded if one arm is read before the image | E24 uses the same terminal boundary AFTER both inputs; unsupported/rewritten serializations are unavailable |
| E01 refits at each depth and never applies a head across depth | E26 tests this distinct same-input portability question without fitting alignment maps |
| Similar aggregate results need not refer to the same failed images | Add E25 with fixed-head paired errors, label-stratified references and explicit low-error limits; disclose reused predictions |
| Changing both train and test caption selections differs from adding training views at fixed evaluation | E27 varies training-view diversity; retain E07 as a later caption-dependence study, with ordinary repeat checks available earlier |
| More rows can silently alter regularization or independent-sample claims | Keep equal total image weight and the same loss/L2 convention in E14/E27; duplicate-row weighting is an uncounted equivalence control |
| Co-occurrence stratification alone does not test readout extrapolation to an unsupported fitting combination | E28 changes head-training support under matched image counts/priors and familiar-only selection; do not call the VLM unexposed |
| Nonlinear heads can accidentally add cross-category sharing and site selection | E05 uses independent category heads, a common reduced input, a fixed midpoint, explicit tuning/seeds and a full-linear reference |
| An intercept change and joint slope/intercept refitting are different | E14 includes a fixed-slope offset control; only that control mechanically preserves per-category ranking |
| Calibration and confidence ranking were conflated in the initial E19 | Make raw source confidence the initial policy; defer E17 correction comparisons; positive temperature alone cannot improve within-category ranking |
| A decoder's reported last hidden state can include a final normalization absent at earlier depths | Log consistent residual-stream hooks and actual modality visibility before interpreting profiles |
| Derived error labels, optional description lengths and reused image pairs can introduce less visible estimation leakage | E08 error targets are cross-fitted; E06 coding uses prefix-only hyperparameter selection; E23 does not pool cross-category pairs as independent |
| Some proposed interpretations would require mention, relation, native assertion or causal-use gold | Retain only image-category claims and explicit context/measurement limits; no new manual validation |

All initial families cover English, Hindi, Bengali and Telugu and both model families where technically supported. E28 has a preselected category-pair cohort rather than all 25 targets. All cards use the shared split/masks; the same 720 CONFIRM images remain closed. This is a design review, not measured evidence that a direction is promising or novel.

### 4.2 Remaining initial catalog

These six initial designs remain unrun. The other six are listed under [Phase 1](#51-phase-1). The result review found no unrun family whose defining contrast was already answered. Keep related families available for later prioritization, especially where they change the supervision, input, readout, or evaluation question.

| ID | Initial experiment family | Provenance | A-priori reason |
|---|---|---|---|
| E05 | Accessibility under different probe capacities | revised existing card | Readout-family dependence is inherent in a claim about accessible information, before observing failure. |
| E14 | Shared, context-offset and context-specific category heads | revised existing card | Shared visual gold makes a common decision rule versus context-specific rules an a-priori question. |
| E24 | Modality order at a readout that sees both inputs | new initial test | Modality placement is a distinct input-layout factor; E04 changes location while holding the input fixed. |
| E26 | Same-input portability across decoder depths | new initial test | Same-input cross-depth application provides a distinct operational portability reference. |
| E27 | Existing-caption diversity in readout training | new initial test | Available training-caption diversity can be varied at fixed images and fixed test views. |
| E28 | Category co-occurrence support shift at readout fitting | new initial test | Shared category vectors support a priori readout generalization to held-out co-occurrence cells. |

### 4.3 Later-preliminary reserve

These 15 designs remain available **after initial results identify a consequential uncertainty**. Their recorded triggers are not facts already found. They are not a post-preliminary competing-explanation protocol, and their eventual use is not automatic. The next preliminary round may retain, refine or park them in response to the actual evidence.

| ID | Reserved family | Outcome-dependent reason to consider it |
|---|---|---|
| E06 | Learning efficiency and independent-image support | For stable above-prior categories; particularly before interpreting weak target-trained comparators. |
| E07 | Dependence on which existing caption is selected | For any promising language or route pattern before promoting it as a research direction. |
| E08 | Length and surface-form sensitivity on unchanged captions | When a gap correlates with caption/token lengths or differs by pooling site. |
| E09 | Category co-occurrence and scene-load heterogeneity | When an aggregate effect could conceal a small category/scene subset. |
| E10 | Geometric agreement versus actual transfer | After useful category signal is established; weak/unstable probes cannot support a strong geometry claim. |
| E11 | Mean and scale adjustment before unchanged-head transfer | When a useful source/target signal coexists with a transfer deficit or score shift. |
| E12 | Image-anchored constrained versus flexible feature maps | Only if there is useful absolute signal and a stable unresolved transfer deficit. |
| E13 | Category-score imitation versus full-feature matching | When feature-map results raise the question of what information alignment should preserve. |
| E15 | Source-only PCA bottlenecks and transfer retention | When high-dimensional estimation or source-specific features remain plausible concerns. |
| E17 | Source-only versus target-context probability recalibration | When discrimination and proper-score/decision outcomes do not move together. |
| E18 | Conditional reliability beyond aggregate calibration | When aggregate calibration appears good or masks category-specific behavior. |
| E20 | Learned correctness readouts versus margin-based confidence | Only for categories with enough correct and incorrect out-of-fold decisions to estimate a second target. |
| E21 | Source-trained layer mixtures versus a selected single layer | When source accessibility, transfer or reliability peak at different fixed depths. |
| E22 | Category-anchor alignment versus image-context alignment | When independent-caption feature matching is ambiguous and sufficiently supported categories exist. |
| E23 | Image sensitivity with the caption held fixed | When a promising result still leaves the contribution of the image unclear. |

### 4.4 Ideas not added as distinct initial families

- Another length bin, selected depth, seed, source-target route, positive/negative metric or calibration diagram: reporting dimensions or sensitivity checks, not new experiments by themselves.
- A raw versus positive-temperature ranking contest on the same category decisions: the per-category order is unchanged; probability calibration can still matter later.
- Projection onto an unchanged linear head's own weight span: it preserves that head's scores, so it cannot test an improvement claim.
- A new translation, native mention/semantic judge, synthetic truth claim or targeted object manipulation: the required independent gold is unavailable under the no-new-human-work boundary.
- Immediate mean correction, unconstrained alignment, prototype mapping or learned correctness heads merely because they are implementable: reserve them for a consequential observed uncertainty, with their explicit supervision budgets.

This review does not assert that every possible initial experiment has been exhausted. It retains the distinct additions justified here and rejects count inflation or unsupported claims.

## 5. Detailed strict-initial cards

All prerequisites below mean reusable data, features or fits. They do not require E01-E04 to show a gap. Every initial card admits strong, weak, null or insufficiently resolved outcomes.

### 5.1 Phase 1

The six families below were executed for both checkpoints, with E03's shallow reference shared across them. See [the Phase 1 results review](phase1.md#phase-1-results-review--19-november-2026) for evidence and interpretation limits.

#### E01. Within-context accessibility and the complete transfer grid

**Phase:** initial. **Status:** completed in Phase 1. **Provenance:** original initial test. **Role:** Anchor.

**Execution dependencies:** G0. Shared validity and selection guards always apply.

**When to use:** Entry experiment after G0 passes.

**Question:** Does a source-language head fail on another caption context even when that target context supports a useful category readout?

**Concrete comparison:** For each model and fixed decoder depth, fit one multilabel linear readout per language on the same FIT images. Apply every fitted readout unchanged to all four languages' SCREEN features: four matched-context cells and twelve transfer cells. Use the identical category IDs, cohort and default caption draw.

**Readouts:** Per-category and supported-category macro AUROC/AP; Brier/NLL; fixed-threshold outcomes. Plot complete grids and depth profiles. For a route s→t, compare D(t,t) and D(s,t) on exactly the same target images and report both absolute values.

**Fitting / supervision boundary:** Source-only regularization/layer selection uses that source's DEV-S. Target-trained heads are explicitly supervised comparators, not knowledge oracles. No target refitting of the transferred head.

**What the result can decide:** A stable transfer deficit with useful target-context decoding motivates compatibility-related branches. Weak diagonal and transferred scores motivate estimation/capacity/context checks. Strong transfer can still motivate confidence or category-specific questions.

**Interpretation limit:** A diagonal-minus-transfer gap does not uniquely identify coordinates or absent information. A target-trained head can be worse than a source head. Native caption content is not controlled; this reference measurement is not itself a novel contribution.

**Method lineage:** S01, S31, S39, T27. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E02. Image, caption and combined information sources

**Phase:** initial. **Status:** completed in Phase 1. **Provenance:** original initial test. **Role:** Diagnostic contrast.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** Initial reference block; do not wait for a large language gap.

**Question:** How much category predictability is already available without the image, and does adding the image change accessibility or transfer?

**Concrete comparison:** Compare frozen VLM features from image-plus-original-caption and caption-only inputs at the same caption-defined readout position. Add a single image-only reference using encoder/projector features; any decoder image-only carrier text is fixed and has no four-language interpretation. Fit separate, matched-capacity heads for each feature source.

**Readouts:** Absolute discrimination and proper scores, combined-minus-caption-only paired differences, and source-target grid changes. Keep architecture stages separate.

**Fitting / supervision boundary:** All heads and scaling are fitted on FIT; source-only selection uses DEV-S. Image identity determines all partitions. A stage that never attends to trailing caption tokens is not four language-specific observations.

**What the result can decide:** Strong caption-only predictability makes text/context explanations worth retaining. Reproducible combined-input gains motivate image-related analysis. Similar visual-only scores with different combined-context scores locate an observational contrast to investigate.

**Interpretation limit:** Removing a modality changes the input distribution; combined-minus-text performance is not a causal decomposition. Label predictability from text is not a caption-mention annotation. No new captions, blank synthetic images or semantic judgments are needed.

**Method lineage:** S24, S47, S16, S25, T21. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E03. Shallow caption predictability reference

**Phase:** initial. **Status:** completed in Phase 1. **Provenance:** original initial test. **Role:** Diagnostic contrast.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** Initial reference block; complements E02 rather than replacing it.

**Question:** Are category decisions or apparent language differences predictable from simple surface features of the released captions?

**Concrete comparison:** Fit a word/character TF-IDF linear classifier and a length-only classifier against image-presence labels, using untouched native captions. Fit a pooled character-feature classifier as a separately named multilingual baseline; do not equate separate vocabularies' dimensions for direct coefficient comparison.

**Readouts:** Held-out within-context discrimination/proper scores and differences from caption-only neural readouts. Report whether strong category patterns recur in both shallow and neural features.

**Fitting / supervision boundary:** Vocabularies, IDF weights, length normalization and classifiers use FIT only. The paired image population and caption draw match E01. No translated vocabulary or category-name mapping is constructed.

**What the result can decide:** A shallow baseline matching a proposed deep effect motivates a stronger lexical/context account. Deep improvement beyond these baselines can justify retaining representation-level analysis, subject to image-source checks.

**Interpretation limit:** Correlated words can predict unmentioned categories through scene associations. This experiment does not identify true native mentions, linguistic understanding or pretraining memorization. Language-specific baseline accuracy is not an intrinsic ranking of languages.

**Method lineage:** S02, S15, S16, T11. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E04. Readout position and pooling sensitivity

**Phase:** initial. **Status:** completed in Phase 1. **Provenance:** original initial test. **Role:** Diagnostic contrast.

Phase 1 compared last-caption and caption-mean pooling. The optional post-caption and decoder visual-token pooling arms were unavailable.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** Initial extraction block; use only positions exposed reliably by the checkpoint.

**Question:** Is an apparent accessibility/transfer gap stable across positions, or specific to a narrow readout location?

**Concrete comparison:** At the same fixed depths compare the last caption token, mean caption-token state and a fixed common post-caption end-of-turn/assistant-boundary position. Separately inspect decoder visual-token pooling where available. Document which tokens can attend to which modalities and positions under the actual attention mask.

**Readouts:** Same-context and transferred discrimination/proper-score profiles at each site; compare the complete prespecified route grid, whether or not E01 contains a gap.

**Fitting / supervision boundary:** Independent heads have matched capacity and FIT/DEV-S support. Do not select a site from SCREEN peaks. Pool caption tokens separately from image and role tokens; otherwise sequence length changes mixture weights.

**What the result can decide:** A site-specific gap motivates information-distribution or measurement-route follow-ups. A persistent gap across sites motivates broader compatibility analysis. A language difference in an ostensibly identical prefix state triggers an extraction audit.

**Interpretation limit:** Many causal decoders' image-prefix states cannot see later captions; inspect the architecture rather than assuming language conditioning. Better pooled decoding does not prove that the model routes or uses that information during answering.

**Method lineage:** S05, S47, S24, S29, T40. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E19. Unadapted source confidence as a selective decision policy

**Phase:** initial. **Status:** completed in Phase 1. **Provenance:** revised existing card. **Role:** Decision-use analysis.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** Initial G9 reference after source heads exist; no calibration result, transfer deficit or E17 run is required.

**Question:** Does unadapted source confidence identify reliable target-context category decisions, beyond aggregate category ranking?

**Concrete comparison:** Freeze each E01 source head at the fixed midpoint depth and its 0.5 decisions. For each category, obtain source-DEV-C quantile cutoffs on raw correctness confidence c=max(p,1-p) targeting 50%, 75% and 90% coverage. Apply those identical cutoffs to each target context, with deterministic include-ties handling and actual source/target coverage reported. All policies retain the same underlying binary classifier. Target risk/coverage curves are descriptive evaluations, never a source of policy cutoffs. Recalibrated policies belong only to a later E17 extension.

**Readouts:** Actual coverage and retained error rate; P(retained | y=1), retained-positive sensitivity and overall P(retained and d=1 | y=1); corresponding negative-case results; macro category summaries and prior/always-absent references. Include a score-blind retention reference at matched actual coverage, whose expected risk is the unselected base error rate. Show category AUROC/AP, base error and error prevalence alongside selective results.

**Fitting / supervision boundary:** Cutoffs use only source DEV-C confidence values; no target labels, target score quantiles or target calibrator select them. Decisions are never re-fitted for an easier retained set. All four diagonal-context and twelve cross-context policies are reported; unsupported error subgroups remain unresolved.

**What the result can decide:** Useful discrimination with weak error triage can open a G9 reliability direction immediately. Stable confidence triage can support a portability question even when aggregate scores look similar. Low risk obtained mainly by retaining absent-category cases limits the practical claim.

**Interpretation limit:** This studies diagnostic-head decisions, not natural-language abstention or VLM self-awareness. Coverage changes the population. Positive-temperature-only scaling preserves each category's confidence ordering and 0.5 decisions, so it is not a new ranking method; later calibration comparisons must not imply otherwise.

**Method lineage:** S33, S47, T27. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E25. Image-level agreement of cross-context errors

**Phase:** initial. **Status:** completed in Phase 1. **Provenance:** new initial test. **Role:** Paired outcome analysis; reuses predictions.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** Initial paired analysis of E01 scores at the fixed midpoint; no aggregate gap, calibration failure or selected hard-example list is needed.

**Question:** Do caption contexts fail on the same images, or can similar aggregate scores conceal different error sets?

**Concrete comparison:** For each fixed source head, compare its four context-conditioned 0.5 decisions on identical SCREEN image-category cells. For all six unordered language pairs report decision disagreement and common-wrong events. Separately compare the four diagonal heads, clearly labeling that this also changes the fitted head. Define e_t=1[d_t!=y]; within y=0 and y=1 compare observed common-error frequency with the product of marginal error frequencies. A fixed label-stratified permutation of one context's error indicators provides an image-association null while preserving both error prevalences.

**Readouts:** Per-category base errors, disagreement, P(e_s=1,e_t=1 | y), excess over the conditional-independence reference, and P(e_t=1 | e_s=1,y) when its denominator is supported. Use 1,000 fixed-seed within-label permutations and paired image bootstrap intervals. Show constant/prior references and positive/negative strata; high agreement from both predicting absence is not recognition success.

**Fitting / supervision boundary:** No new head, selected direction or semantic correctness judgments are fitted. The fixed source head ensures that the main contrast changes context rather than classifier. Diagonal comparisons are secondary. Masks and common cells match E01. Sparse error-conditionals are reported as unresolved with counts, not zero or evidence of independence.

**What the result can decide:** Shared difficult images can open a common-difficulty question; excess disagreement can open a context-specific accessibility/reliability question even with similar averages. The analysis does not need one language to perform worse and does not select a preferred language or ensemble.

**Interpretation limit:** Error agreement cannot establish the underlying cause. Image difficulty, independent caption content and annotation noise all remain possible. This is a distinct scientific analysis of reused predictions, not an extra independent VLM run or an oracle context-selection experiment.

**Method lineage:** S33, S39, S16, T27. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

### 5.2 Remaining initial designs

#### E05. Accessibility under different probe capacities

**Phase:** initial. **Provenance:** revised existing card. **Role:** Diagnostic contrast.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** Initial capacity reference after G0; no weak linear result or language gap is required.

**Question:** How dependent are within-context accessibility and portability on the diagnostic readout family?

**Concrete comparison:** At the fixed midpoint depth ceil(L/2), compare a linear head with independent per-category one-hidden-layer MLPs. For the primary capacity comparison both receive the same 64-dimensional source-FIT PCA features (clipped to valid rank), the same source scaler, and the same known cells. Also retain E01's full-feature linear reference so PCA-induced loss is visible. Transfer each whole source system unchanged; target-context comparators are fitted on target FIT at this identical depth.

**Readouts:** Within-context and transferred AUROC/AP and proper scores; train/evaluation gaps, parameter counts and fitting variation. Apply the same automatic shuffled-label controls to both families. Show reduced-linear versus full-linear results before attributing any change to nonlinearity.

**Fitting / supervision boundary:** Freeze the fitting recipe before scores: linear C grid from the shared protocol; MLP widths {16,64}, L2 coefficients {0.001,0.1} on mean loss, ReLU activation, Adam learning rate 0.001, maximum 200 epochs, and earliest minimum-DEV-S-log-loss checkpoint, with fixed seeds {0,1,2}. Select width/L2 by source DEV-S macro AUROC averaged across seeds; report all seeds, not a winning seed. The MLP grid has four configurations, but more parameters and checkpoint selection than the linear family; these are explicit differences. Each category has its own nonlinear head, avoiding an unannounced category-sharing change. No VLM weights are updated.

**What the result can decide:** A nonlinear gain can open a readout-family-dependent accessibility direction; similar estimates can support using the simpler family. A large PCA penalty or unstable nonlinear fitting leaves the comparison unresolved. No outcome is assumed in order to run this initial contrast.

**Interpretation limit:** Capacity families have unequal parameterization and optimization; a finite comparison does not establish the best possible decoder. Nonlinear success does not show causal use by the VLM, and failure does not prove information absence. Later E06 can address finite-sample estimation if it becomes consequential.

**Method lineage:** S01, S44, S02, S10, S16. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E14. Shared, context-offset and context-specific category heads

**Phase:** initial. **Provenance:** revised existing card. **Role:** Discovery contrast.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** Initial common-rule reference; the shared image targets justify the comparison before any transfer result.

**Question:** How much category information supports a shared decision rule across the four observed contexts?

**Concrete comparison:** Compare a pooled shared linear head, jointly fitted shared slopes with language-specific intercepts, and separate language heads at the fixed midpoint depth. Every FIT image contributes total weight one across its four views in pooled fits. Include a frozen-shared-slope intercept-only control to distinguish changing an offset from refitting slopes. Add the four fixed leave-one-language-out shared heads as source-only pooled-context comparators; do not search source subsets.

**Readouts:** Per-context discrimination and Brier/NLL, parameter counts and FIT-to-SCREEN differences. An intercept change at fixed slopes cannot change within-category AUROC/AP; joint refitting can. Keep those two comparisons separate. Separate target-seen pooled fits from leave-one-language-out transfer.

**Fitting / supervision boundary:** Seen-context pooled models choose C on the equal-language DEV-S macro objective; a leave-one-language-out model uses only its three included languages for all fitting/selection. Intercept-only controls hold pooled slopes fixed and fit offsets on FIT. Separate heads follow their source-only DEV-S protocol. These target-seen models are supervised diagnostics, not zero-shot systems; image weighting and loss/regularization normalization remain explicit.

**What the result can decide:** Shared success can open a common-rule direction; useful context-specific slopes can open a context-dependent readout question. Fixed-slope offsets changing only probability quality can open a score-scale question. Weak or unstable fits remain informative about this reference family, without assuming a pre-existing gap.

**Interpretation limit:** Extra parameters and heterogeneous caption content can account for differences. A shared fitted head does not prove a shared causal representation. This is not an independent source-portfolio study or reactivation of Q11.

**Method lineage:** S39, S34, S44, T23, T47. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E24. Modality order at a readout that sees both inputs

**Phase:** initial. **Provenance:** new initial test. **Role:** Diagnostic input contrast.

**Execution dependencies:** E01, E04. Shared validity and selection guards always apply.

**When to use:** Initial extraction comparison if the checkpoint genuinely supports both native serializations. Technical support, not an E01-E04 outcome, determines availability.

**Question:** Are accessibility and portability sensitive to where the image occurs relative to the unchanged native caption?

**Concrete comparison:** For exactly the same image-caption pair, compare image-then-caption with caption-then-image inside the same user turn. In both arms read the identical native terminal boundary AFTER BOTH modalities, at all four fixed depths. Fit own-order heads and transfer them unchanged across order within each language; show within-order four-language transfer grids separately. Keep image pixels/crops, caption bytes and all other template tokens identical except the necessary modality placement. The last caption token is NOT a valid common site for the caption-first arm of a causal model.

**Readouts:** Own-order accessibility, same-language cross-order portability, cross-context grids at each order, and proper scores. Report actual processed token order, spans, attention visibility and any unavoidable sequence-position differences. Compare with E04's fixed-input site contrast rather than combining the two into a single winner.

**Fitting / supervision boundary:** Each order uses the same FIT image IDs and matched linear-head recipe; source-order preprocessing/head stay frozen on cross-order application. Select only on the appropriate source-order DEV-S. G0 must verify that the processor preserved the requested order and the terminal state can see both inputs. If a model rewrites both orders identically or cannot serialize an order, report unavailable/no distinct manipulation; do not force it using activation edits.

**What the result can decide:** Order-sensitive accessibility or portability opens a concrete input-layout/measurement direction. Stability narrows that particular concern. Both outcomes are interpretable before knowing whether language transfer succeeds.

**Interpretation limit:** Order changes positions and attention opportunities together; this is not a pure causal test of language or one attention mechanism. A syntactically valid reordered input can be atypical of training. The contrast establishes operational robustness, not information loss or task use.

**Method lineage:** S05, S06, S24, S47, T40. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E26. Same-input portability across decoder depths

**Phase:** initial. **Provenance:** new initial test. **Role:** Readout portability reference.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** Initial structural reference wherever the four decoder-depth features share a coordinate dimension and consistent residual-stream hook. No language gap is required.

**Question:** How unusual is cross-context readout incompatibility compared with moving a readout across depth while keeping the entire input unchanged?

**Concrete comparison:** Within each language and model, take a head fitted at one fixed depth and apply its unchanged scaler/head to each adjacent depth's features for the exact same image-caption input and site. Use both directions for all adjacent pairs (six directed routes when four depths exist). Compare with an independently fitted head at the receiving depth and with E01's same-depth cross-language routes. Do not fit a cross-depth map or expand into every layer-language combination.

**Readouts:** Receiving-depth own-fit versus transferred AUROC/AP, Brier/NLL, thresholded outcomes and raw feature/logit scale shifts; full route profiles. Report useful absolute own-depth signal before interpreting a transfer penalty.

**Fitting / supervision boundary:** Only source-depth FIT fits and source-depth DEV-S selection define the transferred system; receiving-depth labels are not used to repair it. Use the same block-output residual-stream interface at all depths, without silently mixing final-normalized and unnormalized states. If dimensions/hooks differ, mark unsupported rather than invent a bridge. The own-receiving-depth head remains a supervised comparator.

**What the result can decide:** Same-input cross-depth failure can show that operational incompatibility is not unique to changing a caption language. Stability across depth with context-sensitive transfer can motivate a more specific contrast. Either result refines which portability phenomena deserve later explanation.

**Interpretation limit:** Depth also changes processing, feature scale and coordinates; this is an operational reference, not a nuisance-matched causal control for language. It cannot assign a language gap to rotation, information absence or an abstract processing stage. E01 already varies the training depth; this card uniquely tests a head at a DIFFERENT receiving depth.

**Method lineage:** S01, S29, S31, S33, T40. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E27. Existing-caption diversity in readout training

**Phase:** initial. **Provenance:** new initial test. **Role:** Training-view comparison.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** Initial comparison once released native-caption multiplicity is checked automatically. No caption-sensitivity result from E07 is required.

**Question:** Does exposure to more of each training image's existing caption contexts improve within-context accessibility or portability without adding independent images?

**Concrete comparison:** At the fixed midpoint depth within each source language, compare training on canonical draw-0 caption features against training on every distinct released native caption of the same FIT images. Give image i total loss weight one: each of its m_i captions receives weight 1/m_i. Hold the canonical source-FIT scaler fixed across arms and preserve the total image-weight/L2 convention. Evaluate each fitted system separately on fixed held-out caption draws 0, 1 and 2; do not average test predictions into an ensemble. Repeat existing caption rows with split weights as an uncounted objective-equivalence check, not a third treatment.

**Readouts:** Per-draw within-context and cross-context AUROC/AP and proper scores, image-paired gains, caption counts/duplicate rates and fitting stability. Display absolute canonical-arm performance. Report variation across evaluation draws separately from image bootstrap uncertainty.

**Fitting / supervision boundary:** Image partitions and independent training-image count are identical. Both arms use the same four-value C grid and the same canonical source DEV-S selection objective; richer-view fitting does not get a larger tuning budget. Source-only preprocessing and trained heads remain fixed at transfer. All four languages are sources and targets. If only one distinct caption exists for an image, that image contributes the same single view in both arms.

**What the result can decide:** Consistent gains can open a training-context coverage direction; source-only gains can open a context-specialization question. No gain can justify retaining the simpler readout protocol. None assumes that canonical-caption transfer first failed.

**Interpretation limit:** Additional captions add observations of the same images, not independent visual support or verified sentence equivalence. Content diversity and wording cannot be separated using these records alone. Unlike E07, which changes which single captions train and test the reference analysis together, this card changes TRAINING-view diversity while holding each evaluation draw fixed.

**Method lineage:** S02, S11, S15, S44, T45, T47, T49. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E28. Category co-occurrence support shift at readout fitting

**Phase:** initial. **Provenance:** new initial test. **Role:** Matched training-support stress test.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** Initial label-defined generalization question. Pairs and support rules are selected before model outputs; no E01 gap or E09 heterogeneity result is required.

**Question:** Can a diagnostic category head generalize to a released category combination absent from its own fitting positives, and does that behavior differ by caption context?

**Concrete comparison:** For a target root A and context root B, train a binary A head in two matched arms at the fixed midpoint. The withheld-combination arm uses m FIT positives from A=1,B=0; the mixed-support control uses m/2 positives from that cell and m/2 from A=1,B=1. Its familiar m/2 positives are the same core images used in the withheld arm. Both use the exact same m FIT negatives with A=0, so independent image count and A prevalence match. Let m=2*floor(min(n_FIT(10),n_FIT(11))/2). Fit both arms for each source language and apply each whole system unchanged to all four SCREEN caption contexts. Use common SCREEN A=0 negatives to evaluate A=1,B=0 and A=1,B=1 positives separately. Report both arm differences within each positive stratum and their difference, rather than only a pooled winning score. Use three fixed hash-ranked FIT allocations {0,1,2}, sharing each allocation's image lists across languages and arms where applicable.

**Readouts:** Primary within-stratum AUROC and paired arm differences, each absolute baseline, AP with actual stratum prevalence, positive sensitivity/negative specificity, and training-allocation variation. This deliberately balanced training task does not supply naturally calibrated presence probabilities; its raw Brier/NLL is not G9 calibration evidence.

**Fitting / supervision boundary:** Freeze a common scaler using withheld-arm FIT inputs only and apply it to both arms, avoiding hidden exposure to A=1,B=1 positives through preprocessing. Use the shared C grid selected only on source DEV-S familiar positives A=1,B=0 and A=0 negatives; unfamiliar positive combinations never select a head. The same deterministic image masks are used for En/Hi/Bn/Te. All support masks come from the released category CSV. A pair is eligible only if m>=60, enough FIT negatives exist, DEV-S has >=10 familiar positives and >=10 negatives, and SCREEN has >=20 positives in EACH stratum plus >=40 common negatives. The label-only rule ranks eligible pairs by descending minimum SCREEN positive-stratum count, lexical ties, then takes the first three distinct target roots. It accesses no model scores and no CONFIRM features or outputs.

**What the result can decide:** A co-occurrence-sensitive loss of generalization can open a precise readout-support/context question. Stable generalization can narrow concerns about these particular category associations. Differences across native caption contexts motivate follow-up without assuming a pure language effect or requiring object-mention gold.

**Interpretation limit:** Only the diagnostic head lacks the held-out positive combination; the frozen VLM may have seen it in pretraining. The fixed positive-image budget trades familiar-combination examples for new-combination examples, so it cannot isolate exposure from per-cell sample allocation. Both scene distribution and caption content can change with co-occurrence. This does not certify a spurious shortcut, compositional reasoning, object counts or relations. E09 only stratifies an unchanged training distribution; this card changes training support with an explicit matched control. All three category pairs remain one family.

**Method lineage:** S02, S16, S30, S44, T27, T42. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

### 5.3 E28 automatic label-only eligibility check

The pinned-CSV audit found 72 ordered eligible pairs under E28's fixed support rule. It selected the following three distinct target roots using only category counts, with no model outcomes. `10` means A present/B absent and `11` means both present. Each table row is a case inside E28, not another experiment. Unknown A/B cells are excluded from that pair's cohorts; the selected image lists must be identical across languages.

| Target A | Context B | m positives per training arm | FIT 10 / 11 | DEV-S 10 | SCREEN 10 / 11 | SCREEN common A=0 negatives |
|---|---|---:|---|---:|---|---:|
| plant.n.02 | building.n.01 | 548 | 548 / 561 | 41 | 91 / 91 | 177 |
| building.n.01 | person.n.01 | 420 | 420 / 464 | 43 | 76 / 71 | 213 |
| body_part.n.01 | plant.n.02 | 464 | 524 / 464 | 37 | 78 / 68 | 213 |

These counts establish only design eligibility, not power, effect size or generalization. If G0 finds missing released images/captions and changes the common image population, recompute/version the label-only eligibility audit before any model results; do not silently reuse this table. No CONFIRM model input, feature or prediction has been used.

## 6. Detailed later-preliminary reserve

These cards retain their result-dependent activation conditions and must not be described as already executed initial tests. Their presence preserves methodological options, not a promise to pursue them all.

#### E06. Learning efficiency and independent-image support

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Discovery contrast.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** For stable above-prior categories; particularly before interpreting weak target-trained comparators.

**Question:** Is the apparent gap mainly associated with finite-sample readout estimation, or does it persist as training support increases?

**Concrete comparison:** Fit the same linear head on nested 25%, 50% and 100% subsets of FIT image IDs, shared across languages. Preserve the head family and the source's full-FIT-selected regularization for the primary curve; a separately labeled re-tuned curve can assess sensitivity if necessary.

**Readouts:** Within-context and transfer learning curves with actual positive/negative image counts, train/evaluation differences and uncertainty. Prequential codelength is an optional summary within this family, not an additional experiment.

**Fitting / supervision boundary:** Subsets are generated before scores are observed and every caption view stays with its image. Any optional online code respects image blocks, prefix-only fitting and an explicit prior code. The optional prequential code must not inherit a C chosen using the full FIT sequence: choose hyperparameters using only the current prefix and its internal image-group development split, or fix them externally before coding. A fixed-C learning curve and a valid prequential code are different protocols.

**What the result can decide:** Gaps shrinking with support motivate an estimation-limited direction or parking premature missing-information claims. Persistent differences at the available support justify more focused compatibility work, not extrapolation beyond the data.

**Interpretation limit:** A short curve cannot establish asymptotic performance or sufficient power. Rare categories may be unestimable at smaller subsets. Repeated captions cannot be counted as new independent visual examples.

**Method lineage:** S11, S12, S44, T11. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E07. Dependence on which existing caption is selected

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Discovery contrast.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** For any promising language or route pattern before promoting it as a research direction.

**Question:** Is the observed pattern robust to the released collection's existing caption variation?

**Concrete comparison:** Repeat the reference analysis under three prespecified deterministic draws of one original caption per image/language. Select independently within each language's released captions; caption index is not a translation alignment. Report duplicate selections for images with too few unique captions.

**Readouts:** Image-paired changes in language gaps, route rankings, category effects and calibration. Distinguish caption-draw variation from fitting-seed variation.

**Fitting / supervision boundary:** Keep image partitions, primary category cohort and model/preprocessing settings fixed. Refit source heads on the drawn FIT captions; selection remains within that source's development data.

**What the result can decide:** Stable effects remain candidate directions. Effects dominated by a particular caption draw motivate a context-selection/measurement question, or a narrower claim rather than blanket rejection.

**Interpretation limit:** The small released writer pool is not a population sample of a language community. Caption resampling does not create independent images or prove that meanings are matched. No caption is translated, repaired or manually checked.

**Method lineage:** S15, S29, S46, T45. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E08. Length and surface-form sensitivity on unchanged captions

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Paired analysis.

**Execution dependencies:** E01, E03. Shared validity and selection guards always apply.

**When to use:** When a gap correlates with caption/token lengths or differs by pooling site.

**Question:** Does the pattern persist on automatically defined image subsets with comparable caption lengths, and after accounting for simple surface features?

**Concrete comparison:** Record character length, model-token length and Unicode-script composition from original text. Define pairwise common-image length-ratio bands from FIT, then report the existing frozen heads on the corresponding SCREEN strata. A nuisance-only error predictor is a separately fitted reference, not a semantic matching device.

**Readouts:** Gap stability across length bands, support/prevalence changes, and predictability of errors from the surface features alone. Use the same image subset for both members of a language comparison.

**Fitting / supervision boundary:** Band boundaries/nuisance models use FIT and DEV-S, not SCREEN performance. No truncation, transliteration, translation or newly written equivalent caption is introduced. If a nuisance-only model predicts classifier errors, its FIT error targets must come from image-group out-of-fold category decisions with fold-local preprocessing, not in-sample mistakes; evaluate it on the final frozen head's held-out decisions.

**What the result can decide:** Length-dependent patterns motivate a more specific input-processing or measurement direction. Persistence limits a simple length-only account but does not remove content confounds.

**Interpretation limit:** Matching length is not matching meaning, token difficulty or information content. Stratum selection changes the evaluated population. Observational script statistics cannot establish a causal script/tokenizer failure.

**Method lineage:** S15, S29, S46, T10, T16. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E09. Category co-occurrence and scene-load heterogeneity

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Paired analysis.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** When an aggregate effect could conceal a small category/scene subset.

**Question:** Are gaps widespread, or concentrated in particular category associations or images containing more annotated categories?

**Concrete comparison:** Use only the released binary vector to define category multiplicity and training-estimated co-occurrence. For supported categories, choose the strongest FIT co-occurring root by a fixed association rule, then stratify existing SCREEN outcomes by that root and by multiplicity bands.

**Readouts:** Per-category route gaps and proper scores within supported strata; compare the full result with leave-one-category-out macro summaries. Report all designated strata, not only the largest gap.

**Fitting / supervision boundary:** Association choices and multiplicity boundaries use FIT labels only. Test labels define evaluation strata but are never model input or cue guidance. Unknown cells do not count as absent categories.

**What the result can decide:** Concentrated effects motivate a specific category/context direction; broad effects support a more general question. Aggregate findings disappearing after removing one root require a narrower claim.

**Interpretation limit:** Category multiplicity is not object count, crowding, salience or spatial complexity. Co-occurrence can legitimately help recognition; correlation does not prove a shortcut. Unsupported strata stay unestimated.

**Method lineage:** S24, S16, S33, T27. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E10. Geometric agreement versus actual transfer

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Discovery contrast.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** After useful category signal is established; weak/unstable probes cannot support a strong geometry claim.

**Question:** Do similar category directions or representation geometries imply useful cross-context readout transfer in this setting?

**Concrete comparison:** Compute signed present-positive cosine between independently fitted category-head directions in the common raw basis, and linear CKA on image-aligned SCREEN context rows. Add FIT split-half direction stability as a reference. Compare these quantities with E01 transfer scores and gaps.

**Readouts:** Full category/route scatterplots and paired uncertainty for geometry-versus-function relationships; show absolute diagonal discrimination. Do not select examples from a best-looking plot.

**Fitting / supervision boundary:** Undo feature standardization in coefficients before comparing directions. Directions are fitted without SCREEN labels; CKA uses paired image IDs, not assumed sentence equivalence. Do not compare raw Qwen and Gemma coordinate vectors.

**What the result can decide:** High geometric agreement with poor useful transfer motivates measurement/decision-specific explanations. Apparent disagreement below within-context estimator stability motivates better estimation first.

**Interpretation limit:** This can characterize dissociation, not prove shared semantic computation or causal use. Geometry/transfer dissociation already exists in the literature; a dataset-specific observation alone needs further contribution checking.

**Method lineage:** S39, S17, S46, T40. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E11. Mean and scale adjustment before unchanged-head transfer

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Discovery contrast.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** When a useful source/target signal coexists with a transfer deficit or score shift.

**Question:** How much of the operational transfer deficit is sensitive to simple input-distribution statistics?

**Concrete comparison:** For each route compare unchanged source preprocessing, mean-shift correction h_t−mu_t+mu_s, and diagonal mean/scale correction mu_s+sigma_s*(h_t−mu_t)/sigma_t. Apply the same source-fitted head/scaler after each raw-space map; protect zero-variance coordinates with a fixed numerical floor.

**Readouts:** Changes in category AUROC/AP, Brier/NLL, thresholded outcomes and logit distributions. Report all three variants together.

**Fitting / supervision boundary:** mu/sigma use corresponding source/target FIT inputs only, with equal weight per image. No target labels are used to fit the corrections; target-input access makes this input-informed adaptation, not unchanged-source transfer.

**What the result can decide:** Large consistent gains motivate an offset/scale-sensitive compatibility direction. Calibration-only gains motivate the confidence branch. No gain parks these particular corrections, not all coordinate accounts.

**Interpretation limit:** Context means may reflect different caption content or pooling, not a pure language component. Scaling can change useful information and margins; correction success is not a unique causal explanation.

**Method lineage:** T14, T04, T06, S39, S33. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E12. Image-anchored constrained versus flexible feature maps

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Conditional extension.

**Execution dependencies:** E01, E11. Shared validity and selection guards always apply.

**When to use:** Only if there is useful absolute signal and a stable unresolved transfer deficit.

**Question:** Can a map estimated from existing same-image contexts improve portability, and does its constraint matter?

**Concrete comparison:** At a source-DEV-S-selected depth, fit a source-FIT PCA basis with r=min(64, source rank). In that shared reduced basis compare identity/centering-only transfer, a target-to-source orthogonal Procrustes map and a regularized unconstrained linear map, using FIT image pairs. Fit the source category head in that same basis for every map.

**Readouts:** Held-out category transfer and proper scores alongside paired feature-reconstruction error. Report map rank/support and whether gains extend beyond reconstruction.

**Fitting / supervision boundary:** Select ridge strength using paired DEV-S reconstruction without target semantic labels; fixed r avoids target test selection. Map fitting is paired-image adaptation. Full-space reference scores are separate from reduced-space comparisons.

**What the result can decide:** Constraint-sensitive functional improvement motivates a precise mapping question. Better reconstruction without useful transfer limits feature-matching accounts. Unstable maps motivate rank/sample analysis rather than a missing-information conclusion.

**Interpretation limit:** Image pairing is weak semantic correspondence; captions may differ legitimately. A flexible fitted map can learn context structure rather than merely rotate coordinates. No patching or base-model updates occur.

**Method lineage:** T02, T04, T14, S34, S44. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E13. Category-score imitation versus full-feature matching

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Conditional extension.

**Execution dependencies:** E01, E12. Shared validity and selection guards always apply.

**When to use:** When feature-map results raise the question of what information alignment should preserve.

**Question:** Does transferring a category-defined decision target behave differently from matching a source context's entire feature representation?

**Concrete comparison:** Keep each source head fixed. On paired FIT images train a target-context linear student to imitate its 25 source category probabilities, then compare with the hard-label target head from E01 and the image-anchored feature-map route from E12. Use the same reduced input dimension for matched comparisons where applicable.

**Readouts:** True-label AUROC/AP and proper scores, teacher agreement, and teacher/student error agreement. Agreement is never substituted for category correctness.

**Fitting / supervision boundary:** Use out-of-fold source teacher scores on FIT image groups so soft targets do not contain the teacher's in-sample memorization. No new captions or translations are generated. Select student regularization on declared development data and label that supervision.

**What the result can decide:** Divergent true-label outcomes motivate a task-sufficient-versus-whole-feature alignment direction. Teacher agreement without correctness gain motivates a reference-bias concern or parking that imitation method.

**Interpretation limit:** This is a new readout-level adaptation of distillation, not a reproduction of sentence distillation. Source scores can be wrong or miscalibrated; no preserved native meaning or universal English reference is established.

**Method lineage:** T18, T55, T47, S33, S34. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E15. Source-only PCA bottlenecks and transfer retention

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Discovery contrast.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** When high-dimensional estimation or source-specific features remain plausible concerns.

**Question:** Do dimensions useful for source-context prediction help or hurt transfer as representation bandwidth increases?

**Concrete comparison:** Fit source-FIT PCA once; compare linear heads on the first 16, 64 and 256 components, clipped to valid rank, plus the full feature reference. Regularization uses source DEV-S only. Apply each fitted system unchanged to target SCREEN features.

**Readouts:** Source and transfer discrimination/proper-score retention relative to full features, explained variance and fitting stability. Keep all dimensionality results rather than a target-selected winner.

**Fitting / supervision boundary:** No target-guided component subset or semantic interpretation is used. A language-specific PCA basis remains part of that source system, including when applied to another context.

**What the result can decide:** Source gains accompanied by transfer deterioration motivate a specificity/estimation direction. Parallel improvements motivate insufficient bandwidth. Unstable low-rank results do not establish an intrinsically low-dimensional shared concept.

**Interpretation limit:** High variance is not semantic relevance. Projection onto the existing head's own weight span cannot change that unchanged head's scores and is not proposed as a separate improvement experiment. No LLM-judged component selection is required.

**Method lineage:** S30, S17, S44, T04. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E17. Source-only versus target-context probability recalibration

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Discovery contrast.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** When discrimination and proper-score/decision outcomes do not move together.

**Question:** Can a probability-scale adjustment account for poor reliability while leaving category ranking unchanged?

**Concrete comparison:** For a frozen head compare raw logits, positive-temperature scaling, and positive-temperature-plus-intercept scaling. Fit each global calibration form on source DEV-C, then separately fit the target-DEV-C comparator. Use supported categories with equal category contribution and natural within-category prevalence.

**Readouts:** Brier/NLL, reliability curves, ECE as descriptive support, and fixed-threshold outcomes. Keep per-category AUROC/AP as invariant checks for the monotone transforms.

**Fitting / supervision boundary:** The base head/depth were selected independently using FIT/DEV-S. Target calibration uses existing labels but is target-supervised adaptation. Neither calibration form is selected using SCREEN performance.

**What the result can decide:** Substantial proper-score improvement with preserved ranking motivates a calibration-sensitive direction. Residual ranking deficits require another account. Source calibration failing where target calibration helps motivates a portability question about probability scales.

**Interpretation limit:** Positive temperature/intercept transforms cannot fix per-category ranking; temperature alone cannot change a 0.5 decision. Calibrated scores do not prove accessible information is used by the VLM. Global calibration may hide category-specific residuals.

**Method lineage:** S33, S39, T14, T19. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E18. Conditional reliability beyond aggregate calibration

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Paired analysis.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** When aggregate calibration appears good or masks category-specific behavior.

**Question:** Can an apparently reliable aggregate score coexist with concentrated errors or miscalibration in specific released-data groups?

**Concrete comparison:** Reuse frozen E01 scores and optionally E17 calibrators. Evaluate per-category reliability and automatically defined caption-length/category-multiplicity strata, using boundaries fixed from FIT. Include constant-prior and always-absent decision references.

**Readouts:** Per-category Brier/NLL, positive/negative error rates and reliability summaries; show both macro and micro aggregates and actual group support. Do not equate low error on negatives with successful positive-category recognition.

**Fitting / supervision boundary:** No new model or semantic labels are needed. The primary group list is fixed before SCREEN inspection; later interesting groups are labeled exploratory and cannot replace the full report.

**What the result can decide:** Hidden group failures motivate a conditional-reliability direction. Uniform stability makes a group-specific narrative less promising at this resolution. Unsupported groups remain unresolved rather than counted as passing.

**Interpretation limit:** Sparse bins and correlated category bits limit precision. These groups do not label object salience, culture or semantic caption content. Aggregate reliability cannot be converted into an individual-image guarantee.

**Method lineage:** S33, S39, S15, T27. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E20. Learned correctness readouts versus margin-based confidence

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Conditional extension.

**Execution dependencies:** E01, E17. Shared validity and selection guards always apply.

**When to use:** Only for categories with enough correct and incorrect out-of-fold decisions to estimate a second target.

**Question:** Do frozen representations contain transferable decision-correctness information beyond the category head's own confidence?

**Concrete comparison:** Generate source-FIT out-of-fold category decisions using image-group cross-fitting; label each decision correct/incorrect from the existing CSV. Fit a regularized correctness head from the same pre-answer features and compare with a margin-only head and c=max(p,1−p). Apply the systems to held-out target decisions from the final source classifier.

**Readouts:** Correctness AUROC/AP, proper scores and selective error/coverage, with error prevalence and absolute base classification performance. Show cross-fitted source behavior alongside target transfer.

**Fitting / supervision boundary:** Correctness labels never come from in-sample predictions of their underlying classifier. Keep all tuning within FIT/DEV-S and calibration within DEV-C. Cross-fit versus full-fit decision-distribution differences must be reported.

**What the result can decide:** A stable extra correctness signal motivates a confidence-readout branch. No gain favors simpler confidence measurement or leaves the issue unresolved if errors are scarce.

**Interpretation limit:** The second head can learn category associations or error patterns; it is not evidence of VLM metacognition. This adapts correctness probing to diagnostic classifier decisions, not the source papers' free-text answer correctness. If error counts are insufficient, mark the correctness target unestimable; never manufacture errors or obtain manual labels to enable the study.

**Method lineage:** S33, S47, S16, T18. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E21. Source-trained layer mixtures versus a selected single layer

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Conditional extension.

**Execution dependencies:** E01, E16. Shared validity and selection guards always apply.

**When to use:** When source accessibility, transfer or reliability peak at different fixed depths.

**Question:** Is useful information distributed across depths in a way a simple source-trained mixture can retain?

**Concrete comparison:** Fit a softmax-weighted mixture over the fixed same-dimensional decoder depths plus a linear category head, using source FIT only. Compare with the source-DEV-S-selected single-layer head and an equal-weight mixture. Use predeclared per-layer scaling from source FIT; do not add encoder/projector vectors of incompatible dimension.

**Readouts:** Within-context/transfer discrimination and proper scores, mixture weights and stability. If extending to correctness mixtures, use E20's derived out-of-fold labels and report it within this family.

**Fitting / supervision boundary:** Regularization and stopping use source DEV-S. Target-specific mixture weights are a different supervised comparator, not the main transfer system. Extra mixture parameters and training flexibility must be explicit.

**What the result can decide:** Consistent source-trained gains motivate a distributed-accessibility direction. Source-only gains without transfer benefit motivate depth-specific source specialization. Unstable learned weights favor the simpler reference or further preliminary checks.

**Interpretation limit:** Mixture success does not identify a causal processing stage; layer coordinates/scales can be incompatible. Learned weights from one context are not universal language-neutral components.

**Method lineage:** S33, S31, S44, T40. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E22. Category-anchor alignment versus image-context alignment

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Conditional extension.

**Execution dependencies:** E01, E12. Shared validity and selection guards always apply.

**When to use:** When independent-caption feature matching is ambiguous and sufficiently supported categories exist.

**Question:** Does constraining alignment to the shared category target behave differently from matching complete same-image caption contexts?

**Concrete comparison:** Using FIT labels, compute source/target present-minus-absent feature means for the primary categories. Fit an orthogonal target-to-source map from those category-direction anchors in a common source PCA basis with r=min(16, primary-category count−1, source rank), which is at most 13 under the audited cohort. Refit the identity, image-anchored and direct target-supervised comparators at this same r; do not compare only against E12’s larger basis.

**Readouts:** Held-out category discrimination/proper scores, direction-anchor stability and reconstruction versus transfer. Include a category-identity-permuted training-anchor null as a fitting control, never as semantic truth.

**Fitting / supervision boundary:** This uses target FIT labels and is explicitly target-supervised diagnostic adaptation. Prototype support, dimension and regularization are fixed from training/development data; all SCREEN labels remain scoring-only.

**What the result can decide:** A stable contrast with full-context alignment motivates target-granularity as a research direction. Similarity to a direct supervised head may show that the map simply learned the category task. Unstable prototypes leave the direction unresolved.

**Interpretation limit:** Co-occurring categories affect prototypes, which are not isolated concept vectors. Sparse anchors underdetermine maps; supervised alignment cannot prove pre-existing language-independent coordinates or solve missing sentence equivalence.

**Method lineage:** S20, S34, S17, T18, T47, T49. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

#### E23. Image sensitivity with the caption held fixed

**Phase:** later. **Provenance:** retained later-preliminary candidate. **Role:** Conditional input contrast.

**Execution dependencies:** E01, E02. Shared validity and selection guards always apply.

**When to use:** When a promising result still leaves the contribution of the image unclear.

**Question:** Do category scores respond to changing the image when the actual caption text is unchanged?

**Concrete comparison:** Freeze a source-trained head. For each supported category, automatically pair distinct SCREEN images with opposite known category labels, using a rule fixed from label/metadata before reading model scores. For each original caption language of the anchor image, compare h(I_anchor,c_anchor) with h(I_partner,c_anchor), retaining the real labels of the image actually shown. Reuse the same image pairs across languages.

**Readouts:** Signed score change toward the positive image, decision changes and relation to the original E01 gap. Report unpaired leftover images, category/multiplicity imbalance and image-pair uncertainty. Pairings from different categories can share images; do not pool their pairs as independent replicates. Keep primary uncertainty category-wise, or use a valid joint dependency-cluster procedure if an aggregate is later needed.

**Fitting / supervision boundary:** No new image is generated or edited. Caption text is copied unchanged; original category annotations supply all outcome labels. All members stay in SCREEN and never enter fitting; pair allocation is disjoint for the primary category analysis.

**What the result can decide:** Consistent image-sensitive scores support investigating input-image dependence. Weak sensitivity despite useful caption-only prediction motivates a context-dominance account or measurement refinement.

**Interpretation limit:** The moved caption can be off-distribution. It is not a verified false or misleading caption: its original image label does not tell us what it mentions. Many image properties change, so a response cannot be attributed uniquely to the chosen category or a pure language effect.

**Method lineage:** S16, S25, S24, S06, T49. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

## 7. E16: uncounted selection and reporting guard

#### E16. Dependence on the layer-selection rule

**Phase:** guard. **Provenance:** uncounted reporting guard. **Role:** Uncounted selection/reporting guard.

**Execution dependencies:** E01. Shared validity and selection guards always apply.

**When to use:** Always accompany any selected-depth summary; this is a reporting guard, not an independent initial or later experiment.

**Question:** Does the conclusion depend on choosing a layer with information from the target context?

**Concrete comparison:** Using E01's fixed depth grid, compare a prespecified midpoint, source-DEV-S-selected depth, and target-DEV-S-selected depth for each unchanged source head. Also show the full unselected profiles. The last choice is explicitly target-supervised selection.

**Readouts:** Differences in transfer gap, category discrimination and confidence reliability under the three declared policies. Report selection variability across the existing caption draws when available.

**Fitting / supervision boundary:** Depth choice never uses SCREEN or CONFIRM labels. The same per-source heads and target SCREEN cohort are reused, so this is an analysis of selection, not another independent model family.

**What the result can decide:** A selection-dependent result must carry the corresponding supervision claim. This audit can identify a pipeline/reporting issue but is not counted as a separate research direction or extra experiment.

**Interpretation limit:** A target-development-selected peak is not an upper bound on all knowledge and is not zero-shot selection. This audit prevents an interpretive artifact; it is not a novel contribution merely because performance changes.

**Method lineage:** S31, S46, T27, S29. See the primary-source index and the corresponding M1/M2 cards. These are bounded methodological adaptations, not necessarily the source paper's original task or evidence that our proposed comparison is novel.

## 8. What initial outcomes can open or close

These are possible paths, not paths already observed. Several may survive together. A preliminary null can park a particular method or tightly bounded claim, not disprove every version of a research direction.

| Candidate path | Evidence that could justify another preliminary round | What still needs resolution |
|---|---|---|
| Accessibility versus compatibility | Useful target-context signal with a stable transfer deficit in E01, surviving estimation/site checks | Whether content, coordinates, fitting or context features explain the deficit |
| Probability/decision reliability | E01 proper scores and E19 raw selective decisions reveal a consequential discrepancy | Whether later E17/E18 or E20 is justified; conditional failures, error support and closest confidence-probing literature |
| Context/visual dependence | E02/E03/E24/E27 identify differences related to evidence source, layout or training-caption coverage | Which later E07/E08/E23 uncertainty is relevant; OOD effects, content confounds and the maximum defensible claim |
| Common versus context-specific readout | E14 distinguishes shared rules, frozen-slope offsets and context-specific fits | Extra supervision/capacity; whether later geometry or mapping work in E10-E15/E22 addresses the actual uncertainty |
| Estimation and diagnostic measurement | E04/E05/E26 or the E16 guard change what a representation claim can support | Whether later sample/rank checks E06/E15 are useful, or the issue is a pipeline correction rather than a research contribution |
| Image-level error structure | E25 finds shared or differing error sets after accounting for marginal rates | Whether content, common image difficulty, label noise or readout differences warrant focused follow-up |
| Category-association support | E28 reveals co-occurrence-sensitive generalization at matched head-training support | Whether later E09 localization or another tightly controlled follow-up is justified; head-only exposure and content limitations |
| Category- or model-specific behavior | A repeatable effect is concentrated in supported roots or differs between Qwen/Gemma | Whether a bounded explanatory question is worthwhile; no assumption that universal effects are the only valuable ones |

For each observed candidate, retain: the phenomenon in a precise sentence, paired effect size and uncertainty, absolute baseline quality, relevant control outcomes, fitting/selection history, compatible explanations, unresolved confounds and the next preliminary uncertainty. Label it **retain**, **revise**, **park within tested scope**, or **insufficient evidence**. No automatic rule discards a question merely for failing a significance cutoff; no isolated peak is promoted merely because it looks large.

Useful evidence must survive the checks relevant to its claim, such as caption-draw dependence, estimator stability or site choice. Not every family needs every other family as a gate. If a direction survives these checks, the next preliminary round can examine its most consequential unresolved assumption. Full systematic competing-explanation testing belongs to the later focused study; essential alternative-account controls already belong here.

## 9. Novelty and relevance checkpoints

The reference grid, readout controls, normalization, PCA, calibration and similarity measures are established methods. Their inclusion supplies sound discovery tools; it does not make each card a new research contribution. M1/M2 already contain close prior work on cross-language transfer, confidence features, representation/output discrepancies and alignment. A reliable pattern needs a precise comparison with the closest literature before promotion as a novel direction.

In particular, [Where Do Multilingual Vision-Language Encoders Fail on Low-Resource Languages?](https://aclanthology.org/2026.emnlp-main.840/) already studies multilingual encoder failures and layer trajectories on XM3600-related retrieval. That is a novelty comparator, not an activation-intervention recipe imported into this plan. Our independent category-presence targets, caption-conditioned VLM readouts and confidence relationships need a substantive new finding, not merely a new four-language score table. [Shared Doubt](https://aclanthology.org/2026.emnlp-main.2654/) and M1 S39 also make generic confidence-transfer or geometry/function dissociation claims insufficient on their own.

Before a path enters the post-preliminary phase, ask: is the observation reproducible and nontrivial; does it concern the intended accessibility/reliability problem rather than a format artifact; is its data/supervision interpretation honest; and what exact claim remains open after reading the closest papers? Replication or a bounded counterexample can be valuable, but its contribution must be articulated rather than assumed. Final novelty remains unverified here.

## 10. Deliberately deferred ideas

- G7 joint reporting, retrieval-augmented generation and concept-guided prompting: they alter the output/task information budget and belong to the separate ongoing branch, not this G9-Q6 first set.
- Free-caption semantic uncertainty, mention-truth extraction, native category-name scoring, translation quality, cultural facts, OCR and relation/instance targets: their required independent semantic gold is unavailable under the user constraint.
- Activation patching, language erasure, steering, base-model unfreezing, new corpus training or an English-pivot translation dataset: none is required or proposed here.
- Unseeded adversarial dictionaries, unrestricted iterative pseudo-pair selection and LLM interpretation of PCA components: assumptions/selection complexity are not yet justified by a simpler observed phenomenon.
- A separate class-weighting study, source-language portfolio search, nonlinear alignment zoo or many near-identical prompt variants: these are not counted as additional quality experiments merely to increase the catalog. Any later addition needs a distinct unresolved decision and the same data boundary.

The initial set starts from common targets and interpretable contrasts with frozen comparisons. Later complexity is justified by an observed uncertainty rather than a quota. Negative results, estimator failures and unresolved ambiguity remain part of the discovery record.

## 11. Primary-source index and provenance

S references identify the exact M1 paper; T references identify the exact M2 paper. This index attributes methodological inspiration. Experiment comparisons, defaults and decision rules in this document are our proposed designs, not published XM3600 results from these sources.

- **S01:** [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644).
- **S02:** [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/).
- **S05:** [Encoded but Not Routed: Explaining the Table-Chart Gap in Scientific Claim Verification](https://arxiv.org/abs/2606.01679).
- **S06:** [How Multimodal LLMs Solve Image Tasks: A Lens on Visual Grounding, Task Reasoning, and Answer Decoding](https://arxiv.org/abs/2508.20279).
- **S10:** [Information-Theoretic Probing for Linguistic Structure](https://aclanthology.org/2020.acl-main.420/).
- **S11:** [Information-Theoretic Probing with Minimum Description Length](https://aclanthology.org/2020.emnlp-main.14/).
- **S12:** [A Bayesian Framework for Information-Theoretic Probing](https://aclanthology.org/2021.emnlp-main.229/).
- **S15:** [What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties](https://aclanthology.org/P18-1198/).
- **S16:** [Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?](https://aclanthology.org/2021.eacl-main.295/).
- **S17:** [Subspace Chronicles: How Linguistic Information Emerges, Shifts and Interacts during Language Model Training](https://aclanthology.org/2023.findings-emnlp.879/).
- **S20:** [The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets](https://arxiv.org/abs/2310.06824).
- **S24:** [Response Wide Shut? Surprising Observations in Basic Vision Language Model Capabilities](https://aclanthology.org/2025.acl-long.1241/).
- **S25:** [Mirage Probes: How Vision Models Fake Visual Understanding](https://arxiv.org/abs/2606.13870).
- **S29:** [Lingua Franca or Probing Artifact? Rethinking Latent Language in Multilingual LLMs](https://aclanthology.org/2026.findings-emnlp.504/).
- **S30:** [Probe Generalization as Subspace Selection for OOD Deception Detection](https://arxiv.org/abs/2609.02893).
- **S31:** [A Shared Geometry of Difficulty in Multilingual Language Models](https://aclanthology.org/2026.acl-short.66/).
- **S33:** [Shared Doubt: Zero-Shot Cross-Lingual Confidence Estimation for Language Models](https://arxiv.org/abs/2605.31220).
- **S34:** [Examining Cross-lingual Contextual Embeddings with Orthogonal Structural Probes](https://aclanthology.org/2021.emnlp-main.376/).
- **S39:** [Cross-Lingual Transfer and Shared Geometry of Deception Probes in LLMs](https://aclanthology.org/2026.mrl-1.33/).
- **S44:** [Pareto Probing: Trading Off Accuracy for Complexity](https://aclanthology.org/2020.emnlp-main.254/).
- **S46:** [Leaving the English Basin: A Logit-Lens View of Cross-Lingual Emotion Transfer in Multilingual LLMs](https://aclanthology.org/2026.mrl-1.7/).
- **S47:** [HALP: Detecting Hallucinations in Vision-Language Models without Generating a Single Token](https://aclanthology.org/2026.eacl-long.287/).
- **T02:** [Offline bilingual word vectors, orthogonal transformations and the inverted softmax](https://arxiv.org/abs/1702.03859).
- **T04:** [Learning principled bilingual mappings of word embeddings while preserving monolingual invariance](https://aclanthology.org/D16-1250/).
- **T06:** [A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings](https://aclanthology.org/P18-1073/).
- **T10:** [Cross-Lingual Ability of Multilingual BERT: An Empirical Study](https://arxiv.org/abs/1912.07840).
- **T11:** [What makes multilingual BERT multilingual?](https://arxiv.org/abs/2010.10938).
- **T14:** [Cross-lingual Alignment Methods for Multilingual BERT: A Comparative Study](https://aclanthology.org/2020.findings-emnlp.83/).
- **T16:** [When Being Unseen from mBERT is just the Beginning: Handling New Languages With Multilingual Language Models](https://arxiv.org/abs/2010.12858).
- **T18:** [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://aclanthology.org/2020.emnlp-main.365/).
- **T19:** [Language-agnostic BERT Sentence Embedding](https://aclanthology.org/2022.acl-long.62/).
- **T21:** [VECO: Variable and Flexible Cross-lingual Pre-training for Language Understanding and Generation](https://aclanthology.org/2021.acl-long.308/).
- **T23:** [Choosing Transfer Languages for Cross-Lingual Learning](https://aclanthology.org/P19-1301/).
- **T27:** [IGLUE: A Benchmark for Transfer Learning across Modalities, Tasks, and Languages](https://proceedings.mlr.press/v162/bugliarello22a.html).
- **T40:** [First Align, then Predict: Understanding the Cross-Lingual Ability of Multilingual BERT](https://arxiv.org/abs/2101.11109).
- **T42:** [Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/abs/2112.09118).
- **T45:** [Crossmodal-3600: A Massively Multilingual Multimodal Evaluation Dataset](https://aclanthology.org/2022.emnlp-main.45/).
- **T47:** [MURAL: Multimodal, Multitask Representations Across Languages](https://aclanthology.org/2021.findings-emnlp.293/).
- **T49:** [Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training](https://aclanthology.org/2023.acl-long.315/).
- **T55:** [Cross-lingual and Multilingual CLIP](https://aclanthology.org/2022.lrec-1.739/).

The numerical support tables and partition rule derive from the pinned released annotation CSV. The separate M1/M2 methodology files, research questions, bibliography and historical analysis are unchanged. The workspace retains the prior prelim.md, original design JSON/generator, reviewed card JSON, phase-review manifest, pinned-CSV audits and exact image-ID partitions. Phase 1 outcome evidence is in [phase1.md](phase1.md#phase-1-results-review--19-november-2026).

## 12. Phase 2

The [Phase 1 review](phase1.md#phase-1-results-review--19-november-2026) supports three open questions. How much of the transfer penalty survives a change in pooling? Does it survive another released caption draw? Can the readout's ranking and probability quality be improved separately? The next round uses [M1](M1.md) for probing controls and measurement, and [M2](M2.md) for transfer comparisons and supervision boundaries. It selects three existing cards (E07, E11 and E17) and adds two distinct pooling contrasts (E29 and E30). These are proposals for preliminary screening, with the timing gates below.

| Order | Experiment | Decision it can change | Readiness and cost |
|---|---|---|---|
| Start | **E29. Paired pooling effect on the transfer gap** | Whether to prioritize a pooling-dependent effect, a residual portability effect, or both | CPU. Reuses canonical SCREEN logits and paired image draws. |
| Start | **E07. Alternate released captions, fixed-head first** | Whether route and pooling patterns survive changes in the original caption view | Automatic availability/support audit first. New VLM forwards only for changed SCREEN views. |
| Start | **E17. Source versus target calibration** | Whether to keep score calibration as a distinct G9 direction | CPU. Reuse SCREEN scores and score the missing DEV-C routes from cached features under a new authorization. |
| Conditional | **E30. Swap readout and feature pooling independently** | Whether better-pooled inputs help an unchanged head or the gain depends on matching the head to the pooling | CPU. Open if E29 shows a consequential reduction. Use cached midpoint activations and fitted heads. |
| Conditional | **E11. FIT-only mean/scale adjustment** | Whether simple feature statistics account for part of the residual transfer penalty | CPU. Open if E29 leaves a supported, practically meaningful residual. Caption-robust claims wait for E07. |

See [phase2.md](phase2.md) for the five-stage execution order, shared work and parallel branches.

### 12.1 Shared analysis and validity rules

- Use both frozen checkpoints, all four caption contexts, fixed midpoints (Qwen 18, Gemma 17), all 12 directed transfers and four receiving-context diagonals. Keep matched image/category cells per contrast and the fixed 14-category primary macro where supported. Apply the known-cell mask, including the unknown plant cell. Preserve all scheduled rows and label low support or undefined outcomes. For a renamed or restricted cohort, show a draw-0 companion on exactly that cohort and keep its summary separate.
- Use the established split and category support rules. Use source FIT for fitted readouts and any allowed preprocessing, DEV-S for source selection, and DEV-C only for its authorized calibration role. SCREEN is reused exploratory evaluation. CONFIRM stays closed. One image is the unit of resampling regardless of languages, repeated captions or category count.
- Register exact primary contrasts, decision-relevant effect ranges, support rules, and claim scopes before the next SCREEN analysis. Use AUROC for ranking and Brier/log loss for probability quality. Report AP as a prevalence-sensitive companion. Use the same paired image bootstrap across matched arms. Compute differences and means within each draw, and report actual valid-draw counts. A wide interval leaves an effect unresolved. Parking a specific alternative requires precision relative to the predeclared meaningful effect range or evidence from an explicit discriminating control. Direction counts or the largest observed route do not establish a general effect.
- Freeze weights/checkpoints, preprocessing, caption bytes and join identities. Preserve the accepted G0 and apply automatic input, split, score-reproduction and support checks to added arms. Use only released labels, images and original captions. No new annotation or caption-equivalence judgments are required. Keep source-only transfer, target-input adaptation, and target-label adaptation separate in the reports. [M1 S02 — probing control tasks](https://aclanthology.org/D19-1275/), [M2 T27 — transfer settings in IGLUE](https://proceedings.mlr.press/v162/bugliarello22a.html).

### 12.2 Experiments to start

#### E29. Does mean pooling reduce the paired transfer gap?

**Why now.** E04 showed general score gains. Phase 1's midpoint AUROC gap estimates changed from 0.110 to 0.030 in Qwen and 0.062 to 0.016 in Gemma. Those were differences of reported point estimates, with no uncertainty for the change in the gap. This new contrast asks whether transferred scores improve relative to their own receiving-context diagonals. It is more specific than another E04 score table.

**Comparison.** For each source s, target t and pooling p, define G(p) = AUROC(t→t, p) − AUROC(s→t, p) on identical target cells. Estimate G(last-caption) − G(caption-mean) within every bootstrap draw. Use the equally weighted 12-route mean per model at the midpoint with image-plus-caption as the primary contrast. Show each route, the residual mean-pooling gap, both absolute arms and the receiving diagonal gain. Repeat the contrast for caption-only as a diagnostic, and present AP plus paired proper-loss results separately. Use the same E01/E04 prediction identities and fit histories in every comparison.

**Decision.** A supported reduction with a useful target diagonal justifies treating pooling as a contributor and testing E30. A remaining consequential gap leaves readout incompatibility open and admits E11. A similar reduction under caption-only weakens an account that requires image input. Imprecise reduction or residual estimates leave the corresponding question open. Independent fits at each pooling are part of this estimand. [M1 S05 — pooled and final-position probes](https://arxiv.org/abs/2606.01679), [M2 T40 — separating transfer and prediction](https://arxiv.org/abs/2101.11109).

#### E07. Does the pattern survive another released caption?

**Why now.** E01 showed directional gaps, and E03 established strong caption-only lexical signal. One original caption draw cannot establish stability to which description of the image was selected.

**Comparison.** Begin with the existing E07 question, using a cheaper frozen-head screen before its full refit. Audit the pinned original-caption inventory and preselect draws 1 and 2 through the established deterministic selector. Record exact repeated text and caption IDs. On supported same-image cohorts compare each draw to draw 0 with the same frozen source and receiving-context heads, separately at each pooling. Count how many views actually changed for every context. Report the paired draw change in the transfer contrast, both absolute diagonal/transfer changes, and the pooling interaction using only matched views. Include a matched draw-0 result for the subset with genuinely changed captions and any restricted macro. If distinct draws or category support are sparse, report the coverage limitation before launching model runs. Reuse draw-0 activations and any byte-identical inputs under the same verified execution policy.

**Decision.** Stable gaps across eligible views support pursuing a portability account. If the gap or ordering depends materially on the selected views, narrow the claim to the view or run the existing E07 refit extension with each draw's FIT and DEV-S counterparts. The frozen screen measures test-view sensitivity given heads trained on draw 0. The refit extension additionally changes the head's training view. An observed draw effect cannot distinguish wording from semantic content. [M2 T27 — explicitly separated settings](https://proceedings.mlr.press/v162/bugliarello22a.html).

#### E17. Does probability calibration transfer?

**Why now.** E01 showed a mismatch between ranking and proper losses. E19's selective-risk support limits leave the reliability question unresolved. A no-GPU probability diagnostic has higher decision value than immediately fitting a correctness head.

**Comparison.** Use the combined last-caption midpoint logits for both checkpoints and the raw-versus-calibrated arms in E17. Keep one selected source head per source context. For each model, reuse the four source DEV-C diagonal score routes and compute the 12 missing source-on-target DEV-C routes once from verified cached target features. The target calibrator must see the **same frozen source head scored on that target's DEV-C inputs**. A target-trained head's diagonal is a separate reference. Fit positive temperature and positive temperature plus intercept under the same fixed category weights and objective on source DEV-C and separately on target DEV-C. Evaluate all arms on common SCREEN cells. Compare paired Brier and log-loss improvements against raw scores and the fixed FIT-prior reference. Verify per-category AUROC/AP invariance up to ties and numerical tolerance, and report 0.5 decision changes separately. Keep the original E19 cutoffs and claims scoped to their raw policy.

**Decision.** If source calibration improves proper losses on supported routes, retain a portable probability correction as a direction. If only target calibration improves them, retain probability-scale transfer under target supervision as the narrower direction. If improvements are small with informative intervals, park these calibrator forms in this setting. Ranking gaps require separate evidence either way. [M1 S33 — decision confidence](https://arxiv.org/abs/2605.31220), [Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html).

### 12.3 Experiments gated by the first results

#### E30. Does a fixed head benefit from different pooling?

When E29 establishes consequential gap reduction, cross the two cached feature poolings with the two corresponding source heads at the combined midpoint. Apply each whole head, including its original scaler, unchanged to last-caption and caption-mean features in the same residual-stream basis. Reuse E01/E04's matched cells. Score the two crossed arms on the exact target rows and assert feature width, hook, checkpoint and input provenance before using either. Use separate source and receiving-context heads in each arm to report both the matched transfer gap and the absolute outcomes. Compute the effect of switching features for each fixed head, and their paired interaction. Retain the original training and regularization identities. The interaction compares two fitted systems and can include their fit differences. Gains for a fixed head support a feature-view contribution without refitting. Gains requiring a matching pooled head support a head–pooling dependency. Crossed-head failure in both directions may reflect an off-distribution feature swap, which limits the mechanism claim. [M1 S05 — readout location](https://arxiv.org/abs/2606.01679), [M2 T40 — separating representations and predictors](https://arxiv.org/abs/2101.11109).

#### E11. Can simple FIT feature statistics narrow the residual gap?

When E29 leaves a supported residual at caption-mean midpoint, test E11 on that fixed representation and all directed routes. Use paired image populations on FIT to estimate source/target feature means and scales from training inputs. Keep the unchanged head and source scaler in every arm. Compare the identity, mean shift and mean-plus-diagonal-scale variants, with a fixed scale floor and the unchanged target-fit diagonal as an accessibility reference. A mean shift is a constant logit offset for a linear head with a fixed affine scaler, so its within-category ranking should stay invariant. Treat that as a control. Separate proper-loss changes from ranking changes under diagonal rescaling, and label both maps as target-input-informed. Material discriminative recovery supports a narrow covariate-scale account. Loss-only recovery belongs with E17's probability story. A precise lack of benefit weakens these simple transformations in this setting and provides a cleaner prerequisite for deciding whether E12 is warranted. [M2 T04 — normalization and mapping constraints](https://aclanthology.org/D16-1250/).

### 12.4 Decision boundary after the round

Report each direction as **retain**, **revise**, **park within tested scope** or **insufficient evidence**. Record the exact comparison, effect, interval, denominators, strongest contrary evidence, and remaining alternative explanation. Several directions can survive independently. Reserve focused competing-explanation work for the supported questions that remain after this round.

The remaining cards stay in the catalog. E05/E06/E15 wait for a specific capacity, sample or rank limitation. E08 can localize a draw or pooling sensitivity to measured surface features, while E09/E18 need adequate stratum support and a specific heterogeneity claim. E10/E12/E13/E14/E22 wait for a residual that calls for geometry, an alignment objective or shared slopes beyond simple controls. E20 needs enough out-of-fold decision errors and a reason to learn a correctness score beyond the calibration checks. E21/E26 can address deeper layer-specific questions after the pooling dependency is understood. E23/E24 require changed visual inputs or a supported common terminal site, so they are candidates if the results leave a specific visual-input question. E27 separates training-view diversity from test-view sensitivity if E07 warrants it. E28 addresses co-occurrence extrapolation and remains separate until a support-shift question earns priority. G7 remains outside this round.

Persist only reusable alternate-view features and verified completed comparison outputs, plus DEV-C score blocks and fitted calibrator or transform parameters reused downstream. Reuse the accepted features, logits and resampling identities where compatible. Intermediate metric arrays and plotting inputs can stay in memory. No experiment in this section has been run, and the designs carry no novelty claim.
