# Research questions: XM3600 and multilingual VLM probing

Updated 14 November 2026. The active shortlist is restricted to questions requiring **no additional manual annotation, translation checking, native-speaker review, human response scoring or visual adjudication** beyond the released data. It retains **5 first-pass entries (Q2, Q6, Q9, Q11, Q12)** and **4 probing-refinement entries (G2, G7, G9, G14)**. These overlap; they are not nine independent contributions. Seventeen earlier entries were removed from the active shortlist. Surviving compound questions have also lost their human-judgment-dependent branches. Original IDs are preserved for traceability. The nine member questions are organized into six groups: two ongoing groups and four standby groups. Grouping preserves each member question's formulation, scope and limits.

The 42-paper reviews from both passes remain historical literature analysis in [analysis.md](references/analysis.md). Their unresolved topics and data requirements are not all current research recommendations. Candidate novelty remains unverified.

## Current question priorities

**Ongoing: G9, Q6 and G7.** G9 is the integrated question about discrimination, transfer compatibility and confidence; Q6 focuses on its accessibility-versus-readout distinction. G7 concerns joint reporting versus individual accessibility. G9 and Q6 overlap and should be developed as a connected direction.

**Standby: Q2, Q9, Q11, Q12, G2 and G14.** Their question statements and boundaries are preserved for later consideration. Q9 is on standby as a separate question; confidence remains part of the ongoing G9 formulation. This status selection does not establish novelty or specify experiments.

## No-new-human-work boundary

- Existing images, original captions, language/image metadata and the shared category CSV are available inputs/targets. Feature extraction, model inference, diagnostic readout fitting and deterministic scoring are computational work; no new human semantic labels are assumed.
- Language conditioning can use the original captions as supplied. No newly translated instructions, native category-name inventory or meaning-equivalent caption set is required. The captions are independent descriptions, so results concern the observed caption contexts and do not isolate a language-only causal effect.
- Where a retained question concerns model answers, its output space is the fixed category identifiers/values defined by the release. Formal cue truth and category-set correctness come from the existing labels. This supplies an operational machine task; it does not supply verified native-language assertion meaning. Unparseable output is a format outcome, not a human-judged semantic error.
- The proxy-target branch uses the repository's released English/Hindi mention synsets and supplied hierarchy. They remain pipeline-dependent proxies, not newly verified human assertions. Bengali/Telugu derived mention outputs are excluded from that branch under the [authors' warning](https://github.com/uriberger/cross_lingual_diff_in_descs#object-mentions-dataset); original Bengali/Telugu captions and shared presence targets remain usable for the other retained questions.
- No new human audit of the image annotations is assumed. Nonbinary/invalid cells can be treated as unknown by a deterministic validity rule rather than visually resolved. Blank cells retain the release's unmarked/absent convention; resulting conclusions are conditional on these fallible annotations.
- Automatically generated translations, semantic judges or extracted free-text claims do not replace missing independent gold. Questions requiring their verified meaning or correctness have been removed. Routine implementation and automatic processing remain necessary.

## Dataset facts and question scope

Sources: [XM3600 paper](https://aclanthology.org/2022.emnlp-main.45/), [annotation paper](https://aclanthology.org/2025.naacl-long.478/), [CSV](https://github.com/uriberger/cross_lingual_diff_in_descs/blob/main/data/xm3600_annotation.csv), [repository documentation](https://github.com/uriberger/cross_lingual_diff_in_descs#object-mentions-dataset), and [loader/filter implementation](https://github.com/uriberger/cross_lingual_diff_in_descs/blob/main/utils.py).

- There are 3,600 unique image IDs and 25 category columns. These are shared human annotations of category presence **per image**, not separate annotations for English, Hindi, Bengali, and Telugu.
- XM3600 supplies independently written captions across 36 languages, including those four languages. It does not include Tamil. Its multilingual captions describe the same images but are not sentence translations of one another.
- Three different targets must remain separate: **image presence**, **human caption mention**, and **model caption mention**. Not mentioning an object is not evidence that it is absent, or that a model failed to recognize it.
- The 25 labels are broad categories. A positive `animal.n.01` label does not certify that a dog, rather than a cat, is visible. A probe or hallucination metric must not silently promote these labels into fine-grained object, attribute, count, relationship, or localization ground truth.
- The original mention pipeline removes mentions when their root category is absent. That filtering is appropriate for its saliency analysis. That filtered mention inventory is unsuitable as independent hallucination ground truth because unsupported category mentions have already been removed.
- The repository warns against its translation-dependent mention outputs for Bengali and Telugu, among five languages. That warning does not invalidate the shared image labels or the original human-written Bengali and Telugu XM3600 captions. Reliable native-language mention extraction needs its own validation.
- Human caption mention frequencies are noisy estimates from a small annotator pool. They are dataset-specific descriptive choices, not population-level facts about Hindi, Bengali, Telugu, or their cultures.
- Location buckets are not equally reliable cultural labels. The XM3600 paper explicitly reports global fallback selection for its Hindi image bucket. A bucket named Hindi cannot automatically be treated as a set of Hindi-region photographs.

### Direct label audit

The CSV was inspected directly during this review. Snapshot identifier: GitHub **blob** SHA `6b99def06a621c570edf074c26741868a4df7353` (a file-content identifier, not a repository commit).

| Category | Images marked `1` | Category | Images marked `1` |
|---|---:|---|---:|
| plant | 1,867 | body part | 1,608 |
| person | 1,483 | building | 1,478 |
| sky | 1,398 | clothing | 1,242 |
| furniture | 1,176 | vehicle | 700 |
| food | 660 | body of water | 575 |
| mountain | 513 | animal | 482 |
| tableware | 415 | bag | 287 |
| electronic equipment | 205 | plaything | 118 |
| kitchen utensil | 100 | jewelry | 95 |
| sun | 91 | home appliance | 64 |
| musical instrument | 53 | weapon | 51 |
| timepiece | 48 | writing implement | 25 |
| hand tool | 20 | | |

**One unresolved cell:** image `6a20677eee04fd08`, `plant.n.02`, CSV line 444 contains `4`. The published loader only treats `1` as present. Its intended meaning is unresolved; it is not established as a count or an ordinary binary label. The 1,867 plant positives above count only literal `1` values.

Blank cells function as unmarked/absent categories in the released annotation convention; the paper describes annotating root-category presence/absence. They are still fallible annotations, not guarantees about tiny, occluded, or ambiguous objects. Their annotation reliability remains a limitation on questions requiring exact visual truth.

Category support is strongly imbalanced. Rare categories have limited evidence for fine-grained conclusions about language differences.

## Ongoing

### G9–Q6: category accessibility and cross-language readout reliability

#### G9. How are category discrimination, cross-language readout compatibility and confidence reliability related under released caption contexts?

**Status:** Ongoing. **Origin:** Second-pass probing refinement.

**Scientific question:** This combines the surviving presence-only parts of Q6 and Q9, with source-language choice from Q11 as a related aspect. It asks whether these properties of a readout vary together.

**Existing data:** Released images, original English/Hindi/Bengali/Telugu captions, existing metadata and shared category-presence targets.

**Automatic scope; no new human labels:** Category targets and language membership need no additional human verification. Representations, category decisions, confidence scores and transfer quantities are computationally obtainable.

**Claim boundary:** This does not establish language-only causality, meaning-equivalent caption content, actual caption assertions, question comprehension or generative language compliance. Within-language decoding failure is not proof of absent information; confidence scores are not automatically reliable.

**Novelty status:** Transfer matrices, confidence estimation and XM3600 representation analysis are close prior work. The narrower shared-category question remains a candidate, not a confirmed gap.

#### Q6. Can cross-language readout incompatibility coexist with accessible image-category information under the released native-caption contexts?

**Status:** Ongoing. **Origin:** First-pass question.

**Scientific question:** The issue is whether portability of a source-language readout and within-language accessibility of a shared category target diverge on the same image collection.

**Existing data:** The same images, the 25 shared category targets, original English/Hindi/Bengali/Telugu captions and their existing language/image metadata.

**Automatic scope; no new human labels:** The released captions supply the language-conditioned text without new translated questions or category-name mappings. Readout targets are the existing category indices, and correctness is defined against the released presence labels.

**Claim boundary:** Native captions differ in meaning, length and collection conditions. This supports conclusions about the observed caption contexts, not a causal effect of language alone or controlled semantic equivalence. Decodability can include information available from caption text; it does not by itself locate the visual source or establish native causal use. Shared image-only states are not language-specific observations.

**Novelty status:** Cross-language readout transfer, alignment and XM3600 representation trajectories are close prior work. A distinct contribution tied to the shared presence targets remains to be established.

### G7: joint category reporting

#### G7. Can joint reporting of the released categories become unreliable while individual category information remains accessible?

**Status:** Ongoing. **Origin:** Second-pass probing refinement.

**Scientific question:** The surviving issue is the relationship between category accessibility and structured reporting of several category decisions under the available native-caption contexts.

**Existing data:** The existing multilabel root-category vector for each image, original captions and language/image metadata.

**Automatic scope; no new human labels:** Reporting means output over the fixed category-ID space, with correctness defined by the released vector. Target sets and parseable answer correctness are computable without native label names, human enumeration judgments or translated task templates.

**Claim boundary:** The shared task interface uses the same category identifiers across caption contexts. This does not test natural Indic enumeration, native instruction understanding, physical object counts or relation binding. Reporting errors and format errors remain distinct. The labels describe category multiplicity, not number of object instances.

**Novelty status:** Multi-object reporting, salience and shortcut effects are already studied. The relationship to category accessibility across these contexts is only a provisional candidate.

## Standby

### G2–Q2: dependence on released mention-proxy targets

#### G2. How sensitive are multilingual category-probing conclusions to the released caption-mention proxy targets?

**Status:** Standby. **Origin:** Second-pass probing refinement.

**Scientific question:** This refines Q2: the scientific issue is dependence of apparent category accessibility on the released supervision/target definition, while image-presence labels are shared.

**Existing data:** The released English/Hindi mention synsets, original captions and shared root-category image labels.

**Automatic scope; no new human labels:** All targets already exist or can be aggregated through the supplied hierarchy. No newly verified mention annotation or translation is required.

**Claim boundary:** The conclusion is pipeline-specific. The released proxy cannot certify actual human assertions or prove that a gap is artificial. Bengali/Telugu derived mention outputs are excluded; independent model-caption factuality is outside scope.

**Novelty status:** Positive-unlabeled supervision and caption incompleteness are established. The exact multilingual probing-validity contribution still needs checking.

#### Q2. How does using the released caption-mention proxies change the apparent English–Hindi category-probing gap?

**Status:** Standby. **Origin:** First-pass question.

**Scientific question:** The unresolved issue is how the estimated language gap depends on whether the target is shared image presence or the repository's released mention proxy. This is a question about measurement and supervision choices in the released data.

**Existing data:** Shared root-category presence, original English/Hindi captions and the existing xm3600_en.json and xm3600_hi.json synset records. The repository's supplied hierarchy permits automatic aggregation to the shared roots.

**Automatic scope; no new human labels:** Both target types come from released files; representations and readouts can be obtained computationally. No fresh caption-assertion annotation, translation checking or QA gold is part of this question.

**Claim boundary:** The mention targets remain translation/extraction-dependent proxies, including any dependence introduced by image-presence filtering. The question cannot establish true human mention error rates, genuine descriptive preferences or model-caption hallucination. Bengali and Telugu are excluded from the mention-proxy branch because the authors advise against those derived outputs.

**Novelty status:** Incomplete caption supervision and multilingual evaluation selection are established topics. The specific sensitivity of category-probing conclusions to these released targets remains a candidate needing literature checks.

### G14–Q12: category accessibility and contradictory structured cues

#### G14. Can category-presence accessibility diverge from acceptance of a contradictory structured cue across native-caption contexts?

**Status:** Standby. **Origin:** Second-pass probing refinement.

**Scientific question:** This refines the surviving formal-cue version of Q12: recoverability of the shared presence target and agreement of a structured answer with a conflicting cue are distinct properties.

**Existing data:** Released category-presence labels, images, original captions and existing metadata. Cue truth is derived from the released label value.

**Automatic scope; no new human labels:** Formal category-ID/value inputs and fixed-space outputs permit automatic truth assignment and answer scoring. The branch excludes newly translated premises and open-ended response interpretation.

**Claim boundary:** The retained claim concerns the structured task and observed caption contexts. It does not certify misleading native-prose semantics, cultural counterfactuals or natural-language hallucination. Category decodability is not proof of native causal use.

**Novelty status:** M²CQA, misleading-context work and diagnostic probes are close prior art. This constrained question still needs a specific novelty check.

#### Q12. Can contradictory category-ID cues change structured category answers while image-category information remains accessible?

**Status:** Standby. **Origin:** First-pass question.

**Scientific question:** The retained issue is disagreement between recoverable category-presence information and a structured category decision when an input cue conflicts with the released image label.

**Existing data:** Released images, original caption contexts and the shared category labels. The truth of a category-ID cue is computable from the existing image-category cell.

**Automatic scope; no new human labels:** Cues are formal category-ID/value statements and answers use the same fixed label space. Their truth and answer agreement can be derived programmatically. No newly translated natural-language propositions or manual response-meaning judgments are required.

**Claim boundary:** This is a structured cue-conflict question under the released caption contexts. It does not establish comprehension of misleading Indic prose, native-language answer quality or full hallucination behavior. Invalid outputs have a format status, not an invented semantic label. Decodability does not establish that visual information caused the answer or disappeared.

**Novelty status:** Counterfactual hallucination, misleading prompts and representation/output discrepancies already have close prior work. A generic contradictory-cue result is insufficient; the narrower multilingual category-readout contribution remains unverified.

### Q9: readout confidence reliability

#### Q9. Can transferred category readouts differ in confidence reliability even when their discrimination transfers across native-caption contexts?

**Status:** Standby. **Origin:** First-pass question.

**Scientific question:** Discrimination of a released presence target and reliability of confidence in that same decision are different properties; the question concerns their relationship across the available language contexts.

**Existing data:** Shared category-presence labels, original captions and language/image metadata for English, Hindi, Bengali and Telugu. Readout scores are computational model outputs.

**Automatic scope; no new human labels:** Correctness and confidence targets refer only to the released presence labels. They do not depend on judging generated text, extracting caption claims or validating multilingual answer meanings.

**Claim boundary:** This is readout confidence about a defined category target, not general VLM confidence or caption factuality. Sparse categories limit conclusions, and language/context differences remain observational. The underlying labels are fallible release annotations.

**Novelty status:** Multilingual calibration and confidence-probe transfer already exist. Their specific relationship to shared category discrimination in this setting needs a dedicated novelty check.

### Q11: source-language portfolios

#### Q11. How does the source-language portfolio relate to transfer of shared category-presence readouts?

**Status:** Standby. **Origin:** First-pass question.

**Scientific question:** The remaining question concerns the portability of diagnostic readouts whose source representations come from one or more available caption languages.

**Existing data:** Released images, original English/Hindi/Bengali/Telugu caption contexts, language/image metadata and the same image-level category labels for every language.

**Automatic scope; no new human labels:** Source-language membership and category correctness are already defined by the release. Source selection, representation extraction and readout learning require computation rather than new human labels.

**Claim boundary:** The original branch about preserving native mention preferences is removed. Transfer here means diagnostic readout transfer, not verified improvements in generation, language fidelity or cultural suitability. Results remain conditional on the observed captions and model.

**Novelty status:** Source-language selection and multi-source transfer are established. A generic best-source comparison is insufficient; the specific category-accessibility relationship remains provisional.

## Relationship between the retained entries

| Refined entry | Related first-pass entries | Scope |
|---|---|---|
| G2 | Q2 | English/Hindi released mention-proxy targets; no verified assertion claims |
| G7 | Related to the reporting/accessibility distinction | Fixed category-ID reporting; not native enumeration or object counts |
| G9 | Q6, Q9, Q11 | Shared presence targets under original caption contexts |
| G14 | Q12 | Formal category-ID cues and outputs; not verified native misleading prose |

## Removed from the active shortlist

Only IDs and removal reasons are retained here. The earlier research-question formulations are no longer active. These removals are provisional: the earlier screen identified missing requirements in their stated forms, but did not establish that every useful automated refinement is impossible.

### First pass — 7 removed

| ID | Missing requirement |
|---|---|
| Q1 | Separating descriptive omission and model assertions requires reliable original-language assertion/mention labels that are not supplied for the full core setting. |
| Q3 | Native-content preservation and English-pivot fidelity require bilingual meaning/assertion judgments absent from the release. |
| Q4 | Quantifying actual translation/extraction errors requires independent semantic gold. The pipeline's own output cannot verify itself. |
| Q5 | Evaluator correctness and meaning-equivalent assertions require new factuality/criterion and semantic-equivalence judgments. |
| Q7 | Separating comprehension from answer expression needs verified multilingual questions/answers and semantic/language-compliance gold. |
| Q8 | Native category naming, ambiguity and ontology equivalence require native terminology judgments beyond the shared roots. |
| Q10 | Actual coverage versus unsupported assertions needs reliable human/model caption-assertion labels. Decodability alone cannot supply them. |

### Second pass — 10 removed

| ID | Missing requirement |
|---|---|
| G1 | The full presence/selection/asserted-output distinction depends on missing validated assertion labels. Its presence-only part is already covered by G9. |
| G3 | Attributing misalignment to true caption-content differences needs meaning-equivalence/asserted-content annotations. Shared image identity is insufficient. |
| G4 | Source fidelity versus image truth needs bilingual source/translation assertion labels. |
| G5 | Naming and ontology-boundary claims need native lexical and possibly subtype judgments. |
| G6 | Distinguishing semantic correctness from output-language compliance needs reliable semantic-answer/language labels absent from the release. |
| G8 | Semantically equivalent scripts/input forms are not supplied; the intended claim needs equivalence or OCR verification. |
| G10 | Judge-error recognition requires human criterion/factuality gold beyond broad image presence. |
| G11 | Relation, instance, count, field-value and operation labels are absent. |
| G12 | Cultural identities, native names and associated facts are not annotated. |
| G13 | OCR transcripts, text regions and cross-language answer-evidence bindings are absent. |

## Limits on finalizing the retained questions

- The main targets are released broad-category presence, pipeline-defined mention proxies in the restricted branch, or formal category-ID decisions. Full caption factuality, natural-language answer semantics, translation fidelity, cultural facts, OCR, relations and instance counts are outside the shortlist.
- A caption can itself reveal category information. Accessibility in an image-plus-caption state does not establish that the information came from vision or that it is used causally by the model. A failed readout does not prove absence of information.
- Repeated captions share image-level labels and do not create new independent images. Rare categories and the unresolved nonbinary cell limit the release's evidential strength. No fresh human verification is implied by retaining a question.
- G2/Q2 concern sensitivity to the released proxy pipeline. They cannot certify true native mention rates or prove that a measured language gap is artificial.
- Model access, computation, automatic preprocessing and meaningful method implementation are still required. None of the retained entries requires new manual semantic annotations under its stated scope.
- The earlier question tags in [paper.md](paper.md) predate this constraint and document the broader screening. They are not an updated paper-retention decision for this smaller shortlist.
- Novelty still needs question-specific literature checks. No model results, established causal explanations or publication outcome are claimed. The shortlist specifies questions and boundaries, not experiments or competing-explanation protocols.
