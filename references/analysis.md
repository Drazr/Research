# Paper analysis supporting the candidate research questions

**Active-scope update (13 November 2026).** The current shortlist in [research.md](../research.md) excludes questions requiring new manual annotations or human verification. It retains Q2, Q6, Q9, Q11, Q12 and G2, G7, G9, G14 only in their stated restricted forms. The two paper-review passes below preserve historical literature findings and broader unresolved topics; removed question IDs and human-data-dependent branches are not active recommendations.

Updated 13 November 2026. This file retains both sets of reviews of the 42 category-4 papers and the supporting novelty checks. The second set follows the requested order for each paper: unresolved questions, dataset suitability/extra data, then novelty assessment. Statements about published methods describe prior work.

The candidate questions, their motivation and data requirements are in [research.md](../research.md). Q1–Q12 and G1–G14 refer to that file. Unresolved questions inferred from a paper's scope are our research inferences unless explicitly attributed to the authors; they are not proof of global novelty.

## First-pass paper reviews

### 1. Cross-Lingual and Cross-Cultural Variation in Image Descriptions

[Berger and Ponti, NAACL 2025](https://aclanthology.org/2025.naacl-long.478/). Read: §§3–4, Limitations, annotation/extraction appendices.

**Already covered:** shared image annotations, translation-to-English entity extraction, category saliency, specificity, and cross-language human mention variation. **Unresolved question / scope:** whether a VLM's lower Indic caption score reflects category recognition failure or a mismatch with the dataset's mention choices. The explicit translation and annotator limitations make unvalidated Bengali/Telugu mention targets unsuitable.

Related question IDs: Q1, Q3, Q4.

### 2. Semantic and Expressive Variations in Image Captions Across Languages

[Ye et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_Semantic_and_Expressive_Variations_in_Image_Captions_Across_Languages_CVPR_2025_paper.pdf). Read: §§2–4, limitations, translation/user-study details.

**Already covered:** human and model captions across seven languages; multilingual captions cover more concepts than monolingual sets; scene-graph analysis and language-specific fine-tuning effects. **Unresolved question / scope:** distinguishing grounded complementary coverage from extra unsupported content, and connecting it to category readouts under Indic prompts.

Related question IDs: Q1, Q10.

### 3. Multi30K: Multilingual English-German Image Descriptions

[Elliott et al., 2016](https://aclanthology.org/W16-3210/). Read: §§2.1–2.4 and Discussion.

**Already covered:** translated and independently collected descriptions are different data conditions, even for identical images. **Unresolved question / scope:** whether claims about Indic grounding depend on imposing English-caption content rather than allowing native descriptions.

Related question IDs: Q1, Q3.

### 4. ImageEval 2025: The First Arabic Image Captioning Shared Task

[Bashiti et al., 2025](https://aclanthology.org/2025.arabicnlp-sharedtasks.52/). Read: evaluation, challenges, future directions, limitations.

**Already covered:** native Arabic captioning with automatic, judge, and human assessments; culturally specific descriptions and limitations of lexical metrics. **Unresolved question / scope:** whether the same Indic output is scored differently because of reference choice or evaluator language rather than factual content.

Related question IDs: Q4, Q5.

### 5. CaMMT: Benchmarking Culturally Aware Multimodal Machine Translation

[Villa-Cueva et al., EMNLP Findings 2025](https://aclanthology.org/2025.findings-emnlp.1220/). Read: evaluation, §6 Discussion, limitations.

**Already covered:** image-supported versus text-only translation, native preferences, cultural-item retention; visual context sometimes hurts. **Unresolved question / scope:** whether translation used inside our evaluation changes category mentions, and whether images help preserve them or encourage additions.

Related question IDs: Q4.

### 6. MaXM: Towards Multilingual Visual Question Answering

[Changpinyo et al., EMNLP Findings 2023](https://aclanthology.org/2023.findings-emnlp.176/). Read: TransVQ2A construction, validation, evaluation protocol and discussion.

**Already covered:** generating multilingual VQA from XM3600 captions, translation-based validation, and target-language answers. Its language subsets are not a fully parallel question bank. **Unresolved question / scope:** how caption-derived question selection changes which visually present categories are tested across languages.

Related question IDs: Q2, Q7.

### 7. CVQA: Culturally-diverse Multilingual Visual Question Answering Benchmark

[Romero et al., 2024](https://arxiv.org/abs/2406.05967). Read: dataset collection, results, §5 Limitations.

**Already covered:** locally authored cultural questions with parallel English versions. It explicitly warns that question difficulty differs across country-language subsets. **Unresolved question / scope:** whether apparent language effects persist when image, category, task, and answer target are all held constant.

Related question IDs: Q1, Q2.

### 8. M5: Diverse Multilingual and Multicultural Vision-Language Evaluation

[Schneider and Sitaram, EMNLP Findings 2024](https://aclanthology.org/2024.findings-emnlp.250/). Read: datasets, evaluation, §§8.1–8.3.

**Already covered:** broad multilingual VLM evaluation including XM3600; limitations in caption metrics, multilingual answer matching, and single-prompt evaluation. **Unresolved question / scope:** how much those measurement choices alter the claimed Indic gap or its representation-level interpretation.

Related question IDs: Q4, Q5.

### 9. All Languages Matter: Evaluating LMMs on Culturally Diverse 100 Languages

[Vayani et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Vayani_All_Languages_Matter_Evaluating_LMMs_on_Culturally_Diverse_100_Languages_CVPR_2025_paper.html). Read: benchmark construction, question-type scoring, analyses and conclusion.

**Already covered:** broad cultural/language/script coverage with several answer formats and judge-based scoring. **Unresolved question / scope:** whether a small, tightly controlled task isolates a failure that broad aggregate comparisons cannot explain.

Related question IDs: Q5, Q7.

### 10. MEGAVERSE

[Ahuja et al., NAACL 2024](https://aclanthology.org/2024.naacl-long.143/). Read: multimodal experiments, tokenizer discussion, contamination methods and limitations.

**Already covered:** multilingual multimodal benchmarking including image captioning; tokenizer-fertility associations and contamination concerns. **Unresolved question / scope:** whether tokenization correlates with a specific failure stage after controlling prompt content and output budget.

Related question IDs: Q8.

### 11. MTVQA

[Tang et al., ACL Findings 2025](https://aclanthology.org/2025.findings-acl.404/). Read: collection, answer protocol, metrics and multilingual text-centric analyses.

**Already covered:** native text-rich image questions, multilingual visual/text alignment, exact-answer evaluation. **Unresolved question / scope:** little that can be tested directly about OCR with the current CSV.

### 12. M3Exam

[Zhang et al., 2023](https://arxiv.org/abs/2306.05179). Read: multilingual/multimodal experiments and discussion.

**Already covered:** multilingual official-exam evaluation; invalid options and prompt-dependent behavior complicate raw accuracy. **Unresolved question / scope:** whether a recognition-versus-generation difference remains after answer-format errors are audited.

Related question IDs: Q7.

### 13. EXAMS-V

[Das et al., ACL 2024](https://aclanthology.org/2024.acl-long.420/). Read: dataset structure, evaluation setup, language analysis and limitations.

**Already covered:** multilingual exam images intertwining OCR, diagrams, domain difficulty and language; some apparent differences come from option-label conventions. **Unresolved question / scope:** a narrower study can remove those intertwined demands before attributing error to visual-language integration.

Related question IDs: Q1, Q7.

### 14. M4U

[Wang et al., WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Wang_M4U_Evaluating_Multilingual_Understanding_and_Reasoning_for_Large_Multimodal_Models_WACV_2026_paper.html). Read: benchmark design, experimental setup and visual-necessity analysis.

**Already covered:** multilingual reasoning evaluation motivated partly by strong text-only shortcuts. **Unresolved question / scope:** whether an apparently successful object probe or generated answer actually needs the matching image, rather than category priors or context.

### 15. Parrot: Multilingual Visual Instruction Tuning

[Sun et al., ICML 2025](https://proceedings.mlr.press/v267/sun25ad.html). Read: MMMB construction/evaluation, multilingual erosion analysis; intervention architecture is background only.

**Already covered:** multilingual visual-token alignment and English-centric instruction-tuning erosion. MMMB already uses paired yes/no questions with joint correctness. **Unresolved question / scope:** whether content remains decodable when native answers or target-language compliance degrade.

Related question IDs: Q1, Q7.

### 16. Cross-Lingual Gaps in Visual STEM Reasoning: Auditing VLMs Across Six Indian Languages

[MRL 2026](https://aclanthology.org/2026.mrl-1.13/). Read: paired-language experiment, extraction/no-image/CoT analyses and limitations.

**Already covered:** matched Indic translations, extraction failures, language consistency and English text in images. It explicitly cautions that some CoT/no-image effects reflect response-format changes. **Unresolved question / scope:** whether similar behavioral gaps remain after independent image labels, audited category meanings and representation probes remove several measurement ambiguities.

Related question IDs: Q1, Q4, Q7.

### 17. Systematic Performance Degradation in Indic VLMs: Evidence from Hindi and Telugu

[HinTel-AlignBench, ALVR 2026](https://aclanthology.org/2026.alvr-main.26/). Read: data construction, task-specific findings and limitations.

**Already covered:** translated and native Hindi/Telugu visual tasks with verification; caption-derived distractors can allow statistical shortcuts. **Unresolved question / scope:** whether caption-driven evaluation samples hide categories and whether the same failure survives a fixed label-based question set.

Related question IDs: Q2.

### 18. Not Truly Multilingual: Script Consistency as a Missing Dimension in VLM Evaluation

[PuMVR, EMNLP Findings 2026](https://aclanthology.org/2026.findings-emnlp.232/). Read: parallel-script construction, SCR, results and limitations.

**Already covered:** controlled same-language multi-script VLM evaluation, script consistency and visual-grounding comparisons. **Unresolved question / scope:** whether script changes affect category readouts and answer expression differently on real photographs.

Related question IDs: Q8.

### 19. Many Dialects, Many Languages, One Cultural Lens

[BanglaVerse, EMNLP Findings 2026](https://aclanthology.org/2026.findings-emnlp.2129/). Read: construction, caption/VQA evaluation, RQs and limitations.

**Already covered:** fixed-image language/dialect comparisons in Bengali cultural content; captioning can deteriorate more than answer-constrained VQA. **Unresolved question / scope:** whether the recognition/generation disparity survives independently labelled visual-category evaluation and evaluator validation.

Related question IDs: Q1, Q5, Q7.

### 20. Bangla-Bayanno

[Hasan et al., 2025 preprint](https://arxiv.org/abs/2508.19887). Read: refinement workflow, translation assessment and future work.

**Already covered:** LLM-assisted Bengali VQA translation refinement; reported translation assessment is small relative to the released corpus. **Unresolved question / scope:** whether refinement preserves category meaning, polarity and answer truth rather than only fluency.

Related question IDs: Q4.

### 21. JEEM: Vision-Language Understanding in Four Arabic Dialects

[Kadaoui et al., EACL Findings 2026](https://aclanthology.org/2026.findings-eacl.18/). Read: human annotation, caption metric meta-evaluation, DCScore and error taxonomy.

**Already covered:** dialect/cultural captioning and VQA; content, fluency and authenticity are separated; DCScore already matches primitive information units. **Unresolved question / scope:** using language-independent image truth to check which Indic evaluation errors originate from language processing versus real category errors.

Related question IDs: Q5.

### 22. Multimodal Evaluation of Russian-language Architectures

[MERA Multi, EACL 2026](https://aclanthology.org/2026.eacl-long.94/). Read: image tasks, scoring, data protection and reproducibility limitations.

**Already covered:** symbolic and semantic scoring, coverage-aware reporting, reproducible multimodal evaluation and leakage protections. **Unresolved question / scope:** no compelling new XM3600-specific research question follows independently.

### 23. JMMMU

[Onohara et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.43/). Read: culture-agnostic/culture-specific design, image-translation ablation and item-level errors.

**Already covered:** near-equal language averages can conceal many item-level answer reversals; image-text translation and instruction following affect scores. **Unresolved question / scope:** whether paired category-level reversals correspond to probe changes or only output/evaluator changes.

Related question IDs: Q1, Q7.

### 24. KOFFVQA

[Kim and Jung, CVPR Workshops 2025](https://openaccess.thecvf.com/content/CVPR2025W/BEAM/html/Kim_KOFFVQA_An_Objectively_Evaluated_Free-form_VQA_Benchmark_for_Large_Vision-Language_CVPRW_2025_paper.html). Read: criteria-based scoring and evaluator comparisons.

**Already covered:** explicit partial-credit grading criteria improve free-form VQA judgment; giving images to a judge can sometimes hurt. **Unresolved question / scope:** whether an audited concept-level rubric changes Indic model rankings relative to unrestricted judging.

Related question IDs: Q5.

### 25. KRETA

[Hwang et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1696/). Read: stepwise image decomposition, QA quality control and reading/reasoning breakdown.

**Already covered:** Korean text-rich VQA and structured generation/verification. **Unresolved question / scope:** no direct reading-versus-reasoning experiment follows from the XM3600 CSV.

### 26. VMMU

[Vietnamese multimodal reasoning, AACL Findings 2026](https://aclanthology.org/2026.findings-aacl.44/). Read: genuine-multimodal filtering, separated input variants, §5.3 translation controls.

**Already covered:** native-language exams, text/visual separation and audited English translation; English translation does not uniformly help. **Unresolved question / scope:** when an English pivot helps or hurts independently verified object truth versus natural caption content.

Related question IDs: Q3, Q7.

### 27. Afri-MCQA

[ACL 2026](https://aclanthology.org/2026.acl-long.1869/). Read: native/English text VQA, open versus MCQ formats, language-understanding controls and limitations.

**Already covered:** linguistic and cultural gaps, answer-format differences, text/speech controls and native human evaluation. **Unresolved question / scope:** whether a shared visual-category task with internal probes narrows the gap to a particular measurable stage.

Related question IDs: Q1, Q7.

### 28. Lost in Translation: Do LVLM Judges Generalize Across Languages?

[MM-JudgeBench, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1746/). Read: translation construction, order/length bias evaluation and limitations.

**Already covered:** multilingual multimodal judge reliability, biases and training-based improvements. **Unresolved question / scope:** how evaluator errors distort the specific relationship between image-presence probes and Indic caption quality.

Related question IDs: Q5.

### 29. VLURes

[Atuhurra et al., ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1367/). Read: image-only versus image-plus-text tasks, difficulty signals and limitations.

**Already covered:** multilingual long-text grounding, irrelevant text and distractors. **Unresolved question / scope:** whether false contextual mentions disrupt already-decodable root-category evidence under Indic prompts.

Related question IDs: Q12.

### 30. CLAIM

[Ye et al., 2025 preprint](https://arxiv.org/abs/2506.11073). Read: multilingual POPE/MME evaluation, language-ID probes, attention analysis and limitations.

**Already covered:** multilingual object hallucination, language-specific attention-head probes and cross-language attention intervention. It explicitly acknowledges possible reinforcement of English-centric bias. **Unresolved question / scope:** whether English-oriented evaluation or pivoting improves category truth while worsening alignment with native human content choices.

Related question IDs: Q3.

### 31. Mind the (Language) Gap

[MMCRICBench, MRL 2025](https://aclanthology.org/2025.mrl-main.38/). Read: scorecard construction, English/Hindi image variants, OCR and CoT comparisons.

**Already covered:** Hindi versus English text **inside images**, with English questions, and numerical reasoning. **Unresolved question / scope:** no direct numerical/OCR gap can be tested using root-category labels.

### 32. Crossmodal-3600

[Thapliyal et al., EMNLP 2022](https://aclanthology.org/2022.emnlp-main.45/). Read: image sampling, caption protocol, human/metric comparison and limitations.

**Already covered:** native multilingual captions, common style guidance and validation of model comparison using the released references. **Unresolved question / scope:** whether the benchmark's original evaluation behavior holds for current instruction VLMs and whether native references sufficiently constrain category-level correctness.

Related question IDs: Q1, Q5.

### 33. IGLUE

[Bugliarello et al., ICML 2022](https://proceedings.mlr.press/v162/bugliarello22a.html). Read: standardized transfer setups, zero/few-shot results and outlook.

**Already covered:** multilingual multimodal transfer across tasks, strong translate-test baselines and nontrivial few-shot behavior. **Unresolved question / scope:** which source-language readouts transfer category information without importing source-language caption preferences.

Related question IDs: Q6, Q11.

### 34. Visually Grounded Reasoning across Languages and Cultures

[MaRVL, Liu et al., EMNLP 2021](https://aclanthology.org/2021.emnlp-main.818/). Read: concept selection, annotation, translate-test comparisons and outlook; readable arXiv full text used where PDF extraction failed.

**Already covered:** English WordNet/ImageNet concepts are not culturally universal or equally salient; native-driven concept and image collection addresses this. **Unresolved question / scope:** whether our category names introduce lexical ambiguity even though the image labels are shared.

Related question IDs: Q8.

### 35. xGQA

[Pfeiffer et al., ACL Findings 2022](https://aclanthology.org/2022.findings-acl.196/). Read: question-type analysis, zero/few-shot results and contemporary work.

**Already covered:** large cross-lingual VQA gaps, question-type differences and partial few-shot recovery. It explicitly discusses alignment between translated choices and English prediction labels; deeper heads were already examined in related work. **Unresolved question / scope:** distinguishing coordinate/readout mismatch from lower category decodability under controlled Indic prompts.

Related question IDs: Q6, Q7.

### 36. Babel-ImageNet

[Geigle et al., ACL 2024](https://aclanthology.org/2024.acl-long.277/). Read: lexical mapping, evaluation coverage and limitations.

**Already covered:** WordNet-linked multilingual label evaluation, low-resource gaps and uneven lexical coverage; inherited Anglocentric concepts remain a limitation. **Unresolved question / scope:** whether alias/definition sensitivity masquerades as visual-category failure on our shared images.

Related question IDs: Q8.

### 37. IndicVisionBench

[Faraz et al., 2025 preprint](https://arxiv.org/abs/2511.04727). Read: parallel/native task creation, answer formats, no-image comparison and discussion.

**Already covered:** Indic cultural VQA/OCR/MMT, multiple answer formats, adversarial questions, paired-language and no-image evaluation. **Unresolved question / scope:** representation-grounded diagnosis using independently labelled generic category truth, distinct from cultural knowledge.

Related question IDs: Q1, Q2, Q12.

### 38. Cross-Lingual Text-Rich Visual Comprehension: An Information Theory Perspective

[XT-VQA, Yu et al.](https://arxiv.org/abs/2412.17787). Read: source/target-language formulation, translation validation, visual-information analysis and distillation.

**Already covered:** language mismatch between image text and questions, visual-information dependence and cross-lingual distillation. **Unresolved question / scope:** question-language versus required-response-language effects in a simpler task without an OCR requirement.

Related question IDs: Q7.

### 39. Mitigating Multilingual Hallucination in Large VLMs

[Qu et al., MHR](https://arxiv.org/abs/2408.00550). Read: invalid-response analysis, multilingual benchmark extension and two-stage training.

**Already covered:** instruction-following and factual hallucination are distinct problems; multilingual POPE/MME/AMBER extensions and cross-lingual preference construction. **Unresolved question / scope:** how those distinctions relate to category decodability and native mention choices.

Related question IDs: Q1, Q7.

### 40. CCHall

[Zhang et al., ACL 2025](https://aclanthology.org/2025.acl-long.1485/). Read: task definitions, generated/translated data construction, XM3600 component and analyses.

**Already covered:** joint cross-language/cross-modal hallucination detection, including XM3600-derived examples; language compliance, image consistency and length-related effects. **Unresolved question / scope:** independently quantifying category truth versus language compliance without collapsing them into one label.

Related question IDs: Q1, Q5, Q7.

### 41. Once Correct, Still Wrong

[M²CQA, Mousi et al., 2026](https://arxiv.org/abs/2602.05437). Read: true/counterfactual pair design, conditional hallucination rate and prompting analyses.

**Already covered:** conditional false-premise acceptance despite success on the true statement, across English/Arabic/dialects; reasoning-first can worsen it. **Unresolved question / scope:** whether that behavior coexists with a stable, cross-language object-presence readout on a matched Indic category task.

Related question IDs: Q12.

### 42. Centurio: On Drivers of Multilingual Ability of Large VLMs

[Geigle et al., 2025](https://arxiv.org/abs/2501.05122). Read: controlled training-mixture experiments, task/language evaluation, output-language reporting and limitations.

**Already covered:** controlled multilingual training composition, multilingual OCR, and target-language output behavior. **Unresolved question / scope:** whether the same behavioral score arises from different category-readout versus language-expression failure profiles across models.

Related question IDs: Q1, Q7.

## Additional close work that changes the novelty assessment

These checks go beyond the original fourth category. They are included here because they constrain what we can honestly claim; the bibliography file itself is not being expanded or pruned in this task.

| Close work | What is already covered | Consequence for our project |
|---|---|---|
| [Object Hallucination in Image Captioning / CHAIR](https://aclanthology.org/D18-1437/) | Ground-truth object labels for caption hallucination; standard metrics can miss image inconsistency; image errors versus language priors. | Root-category precision is an adaptation of established evaluation, not a new hallucination metric. |
| [Does Object Grounding Really Reduce Hallucination of Large VLMs?](https://aclanthology.org/2024.emnlp-main.159/) | Grounding/hallucination evaluation must distinguish QA from open caption generation. | Good object-QA performance cannot establish reliable caption generation. This limits the scope of Q1. |
| [Responses Fall Short of Understanding](https://arxiv.org/abs/2604.04411) | Representation-versus-response discrepancies in visual document understanding. | “The probe knows more than the answer” is not a sufficient new claim. |
| [Encoded but Not Routed](https://arxiv.org/abs/2606.01679) | Layer/token-position probes of encoded evidence versus predictions under controlled table/chart formats. | Comparing intermediate and prediction positions is established. The potentially distinctive question concerns multilingual category truth and description choices. |
| [Decodable but Misrouted](https://arxiv.org/abs/2609.18860) | Harmful-meme readout gaps, calibrated native baselines, Spanish/Hindi-English robustness, image controls and interventions. | Cross-lingual probe superiority and calibration-controlled readout gaps are already studied. Its diagnostic scope is relevant to the question assessment. |
| [Where Do Multilingual Vision-Language Encoders Fail on Low-Resource Languages?](https://arxiv.org/abs/2608.30725) | XM3600 retrieval, low-resource representation trajectories, language-ID erasure controls, cross-language substitution and calibration. | Neither language-ID separability nor representation drift alone is a new explanation. This is particularly close to any proposed XM3600 layer-wise transfer study. |
| [MultiInpaintCOCO](https://arxiv.org/abs/2611.03309) | Fine-grained multilingual minimal pairs, including Hindi; incorrect English captions can beat correct non-English captions. | English preference and multilingual visual-detail failure are already established. Q3 needs the additional truth-versus-native-content distinction. |
| [Evaluation of Multilingual Image Captioning: How far can we get with CLIP models?](https://arxiv.org/abs/2502.06600) | Multilingual CLIPScore variants, translation-quality-aware evaluation and native multilingual/multicultural tests. | A multilingual caption metric or translated evaluator baseline alone is not new. |
| [A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates](https://aclanthology.org/2025.findings-acl.638/) | Calibrated caption-quality uncertainty and word-level error assessment. | Do not sell uncertainty-aware caption evaluation or conformalization as an untouched direction. |
| [SpanCalib-VLM](https://arxiv.org/abs/2608.29974) | Multilingual hallucination-span detection and calibrated fusion. | A cross-language confidence detector is a crowded extension; Q9 needs a specific unresolved category-readout reliability question. |
| [SPECI](https://aclanthology.org/2026.aacl-long.7/) | Caption specificity via atomic facts and WordNet-based granularity; multilingual lexical limitations. | Specificity scoring is established, and our CSV cannot verify fine-grained specificity by itself. |
| [Transparent Human Evaluation for Image Captioning / THuMB](https://aclanthology.org/2022.naacl-main.254/) | Separate precision, salient-information recall, fluency and conciseness. | Our factuality/content-choice distinction needs attribution and a narrower multilingual diagnostic result. |
| [Scalable Performance Analysis for Vision-Language Models](https://aclanthology.org/2023.starsem-1.26/) | Behavioral performance correlations with extracted linguistic features. | Category/length correlations are useful analysis but do not independently establish a mechanism or novelty. |

Targeted searches covered multilingual category probes, XM3600 representation analysis, caption saliency and omissions, caption-derived VQA sampling, English preference, metric reliability, and cross-language calibration. Primary papers and author releases were used for claims; search snippets and third-party paper summaries were discovery aids only. Absence of a search hit is not evidence that a question has never been studied. Additional close-work entries above are targeted novelty checks, not claimed full systematic reviews of every cited paper.

## Second-pass paper reviews: questions → dataset fit → novelty

### P01. Cross-Lingual and Cross-Cultural Variation in Image Descriptions

[Paper](https://aclanthology.org/2025.naacl-long.478/). Basis: entity extraction/filtering, saliency analysis and stated translation/annotator limitations.

1. **Unresolved questions.** When a category is visible but rarely described in a language, is its information nevertheless decodable from a VLM's image-conditioned states? Can separate readouts predict presence and the tendency to mention it? A broader question is whether specificity differences reflect lexical inventories, communicative selection or different visual distinctions; human mention statistics alone cannot distinguish these.

2. **Dataset fit and extra data.** Shared labels support broad category-presence questions. Human/model mention questions need validated original-language assertions, especially for Bengali and Telugu. Fine specificity and population-level cultural questions require subtype labels or more independent annotators.

3. **Novelty assessment.** Human saliency variation is already established. Visual Constraints on Linguistic Selection already predicts linguistic choices from visual input. The joint VLM presence/mention question needs further multilingual communicative-selection and omission literature checks; relates to G1/G3/G5.

### P02. Semantic and Expressive Variations in Image Captions Across Languages

[Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_Semantic_and_Expressive_Variations_in_Image_Captions_Across_Languages_CVPR_2025_paper.pdf). Basis: human/model concept coverage, grounded scene graphs, multilingual training comparisons.

1. **Unresolved questions.** Does multilingual descriptive diversity correspond to different available category information, or different selection from similarly available information? When multilingual captions describe different content, how much of low representation similarity is genuine semantic difference rather than failed cross-language alignment? Which complementary information is useful and reliable?

2. **Dataset fit and extra data.** Presence labels and independent captions support coarse category questions. Caption-asserted content and semantic equivalence need annotation. Attributes, relations and object instances require richer visual labels.

3. **Novelty assessment.** Multilingual coverage gains, including grounded scene-graph coverage, are already studied. The narrower content-difference versus representational-misalignment question needs checks against multilingual alignment and Common Ground with Visual Representations; relates to G1/G3.

### P03. Multi30K: Multilingual English-German Image Descriptions

[Paper](https://aclanthology.org/W16-3210/). Basis: parallel translations versus independent native descriptions and their corpus differences.

1. **Unresolved questions.** Does a probe trained on translated descriptions generalize to independently authored descriptions of the same visual content? If it fails, is the failure attributable to missing content, translation-associated wording, or incompatible representation coordinates? The question concerns annotation provenance, not just language identity.

2. **Dataset fit and extra data.** XM3600 supplies independent native captions, but not a gold collection-mode or parallel-meaning annotation. Verified translations, paraphrases and content labels are extra text data.

3. **Novelty assessment.** Translated versus independent descriptions are the source contribution. Translationese and cross-domain probe generalization are close. Whether provenance/content shift accounts for an apparent multilingual probing gap remains provisional; relates to G3.

### P04. ImageEval 2025: The First Arabic Image Captioning Shared Task

[Paper](https://aclanthology.org/2025.arabicnlp-sharedtasks.52/). Basis: different caption styles, evaluation dimensions and culturally specific descriptions.

1. **Unresolved questions.** Does a model retain general visual information but fail to retrieve or express a culturally specific identity? Are apparent representation-to-caption failures influenced by whether the reference favors brevity, completeness or named cultural detail? These are different knowledge and evaluation demands.

2. **Dataset fit and extra data.** Broad category accessibility is within scope. Cultural identity, associated facts, naming and style quality require identity/fact labels, native terminology and human judgments.

3. **Novelty assessment.** Separate content/style criteria already exist. Visual-reference factual-access probes are close. The language-conditioned cultural identity/association question needs additional knowledge-access literature checks; relates to G1/G12.

### P05. CaMMT: Benchmarking Culturally Aware Multimodal Machine Translation

[Paper](https://aclanthology.org/2025.findings-emnlp.1220/). Basis: visual context sometimes improves and sometimes worsens translation; cultural-item retention versus surface metrics.

1. **Unresolved questions.** When visual context changes a translation, is the system preserving the source proposition or replacing it with plausible image content? Can pre-answer readouts distinguish information supplied by the source text from information supported by the image, especially when those sources disagree?

2. **Dataset fit and extra data.** Root presence and source captions are available. Source-asserted versus translated content needs bilingual annotation; cultural items, relations and fine-grained fidelity require richer labels.

3. **Novelty assessment.** Visual grounding and source reliance in multimodal translation are established. Separate latent source-fidelity and image-truth questions, especially in Indic settings, need further multimodal MT hallucination and quality-estimation checks; relates to G4.

### P06. MaXM: Towards Multilingual Visual Question Answering

[Paper](https://aclanthology.org/2023.findings-emnlp.176/). Basis: caption-derived question generation, validation and limitations of caption-derived negative answers.

1. **Unresolved questions.** Do caption-derived evaluation examples systematically exclude visually present but unmentioned categories? More directly for probing, does training or selecting probe targets through language-specific captions create a false picture of which visual categories the model represents? Is a probe predicting presence or merely the annotators' mention distribution?

2. **Dataset fit and extra data.** Independent presence labels make this question relevant to the dataset. Reliable caption mentions are needed to address language-dependent selection or incomplete supervision. Property, count and relation questions require extra labels.

3. **Novelty assessment.** XM3600-to-VQA generation is already done. Incomplete captions, positive-unlabeled supervision and selection bias are established topics. The specific language-conditioned probing-validity question remains a candidate; relates to G2.

### P07. CVQA: Culturally-diverse Multilingual Visual Question Answering Benchmark

[Paper](https://arxiv.org/abs/2406.05967). Basis: native cultural questions, parallel English and stated cross-region difficulty differences.

1. **Unresolved questions.** Are culture-specific failures due to visual recognition, linking an entity to stored facts, linguistic access to those facts, or expression of an answer? Which of these remains difficult after item difficulty is held fixed? Comparing unrelated regional subsets cannot isolate those stages.

2. **Dataset fit and extra data.** The current data supports generic-category accessibility. Cultural identity, entity-to-fact access and native factual knowledge require additional identity, name, fact and answer annotations.

3. **Novelty assessment.** The source identifies comparability concerns. Visual factual-association probes and cultural concept adaptation are close. Multilingual entity-to-fact accessibility needs further review; relates to G12.

### P08. M5

[Paper](https://aclanthology.org/2024.findings-emnlp.250/). Basis: multiple tasks, XM3600 caption metrics, answer matching and prompting limitations.

1. **Unresolved questions.** Do different tasks reveal different parts of the same visual information, or is the apparent task difference mostly scoring and response format? Can a presence readout generalize between recognition and caption-generation instructions on the same image? Does poor caption scoring coexist with stable category accessibility?

2. **Dataset fit and extra data.** Shared presence supports questions about category accessibility across captioning and recognition tasks. Task correctness and asserted caption content need validation. Entailment and reasoning require task-specific labels.

3. **Novelty assessment.** Cross-task probes and representation/response discrepancies already exist. The unresolved candidate concerns shared category accessibility and native mention behavior, rather than another generic task-transfer question; relates to G1/G6/G9.

### P09. All Languages Matter

[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Vayani_All_Languages_Matter_Evaluating_LMMs_on_Culturally_Diverse_100_Languages_CVPR_2025_paper.html). Basis: broad linguistic/cultural coverage, question formats and script/resource associations.

1. **Unresolved questions.** Does weak performance in a script or resource group reflect poorer semantic representations, unfamiliar category terms, or failure to follow the requested output language/format? Which patterns survive controlling item content, rather than comparing different cultural subsets?

2. **Dataset fit and extra data.** The same visual labels apply across the core languages. Script equivalence and response-language/correctness require extra text annotations. Resource-family generalization needs broader language and exposure data.

3. **Novelty assessment.** PuMVR already addresses script consistency, and Hinck et al. analyze image-induced English replies internally. The category-content versus language-control question needs a narrower novelty assessment; relates to G6/G8.

### P10. MEGAVERSE

[Paper](https://aclanthology.org/2024.naacl-long.143/). Basis: tokenizer discussion, multimodal scores and contamination/prompting limitations.

1. **Unresolved questions.** Are language-associated probe differences driven partly by token budget, truncation and sampled token positions rather than semantic accessibility? Are representation conclusions stable when semantically equivalent input variants have different segmentation costs? The upstream role of pretraining exposure is a separate, harder question.

2. **Dataset fit and extra data.** The image labels are relevant to category accessibility. Semantic-equivalent prompt forms need validation; causal tokenizer or pretraining-exposure claims need controlled model resources. Training-set membership is not provided by XM3600.

3. **Novelty assessment.** Tokenizer-performance associations and XM3600 low-resource encoder trajectories are established. Semantic accessibility associated with input form remains a candidate question; token-count correlation alone does not establish novelty; relates to G8.

### P11. MTVQA

[Paper](https://aclanthology.org/2025.findings-acl.404/). Basis: multilingual text-rich visual understanding and answer requirements.

1. **Unresolved questions.** In multilingual text-rich images, can a model encode the visible characters yet fail to map them into the question language, or does it fail to read the characters at all? Can a downstream reasoning failure coexist with accurate text recovery? These are scientifically distinct bottlenecks.

2. **Dataset fit and extra data.** The CSV is insufficient for OCR or text-to-question binding. Transcripts, text regions, scripts, semantic evidence and answers are needed; generic scene labels do not supply them.

3. **Novelty assessment.** MTVQA/XT-VQA and document-understanding readout-gap work are close. Script decoding versus cross-language semantic binding needs a specific literature check; relates to extra-data G13.

### P12. M3Exam

[Paper](https://arxiv.org/abs/2306.05179). Basis: multilingual exam outcomes, invalid choices and multimodal task demands.

1. **Unresolved questions.** Does an exam failure arise before visual evidence is represented, during relation/quantity reasoning, or at mapping a decision to an allowed answer? Can a decoded entity signal coexist with failure to represent the relation needed by the question? Successful entity recognition is weaker than compositional understanding.

2. **Dataset fit and extra data.** The roots support category recognition only. Relational, arithmetic and multistep reasoning questions need instance, relation, count, operation and solution labels.

3. **Novelty assessment.** Encoded-but-not-routed and visual-document probing already address response gaps. Compositional grounding literature constrains the relation/operation question; relates to G11, with the coarse category aspect covered by G1/G6.

### P13. EXAMS-V

[Paper](https://aclanthology.org/2024.acl-long.420/). Basis: text rendered in exam images, subject/language differences and region-dependent difficulty.

1. **Unresolved questions.** How much of a representation gap changes when the same question moves between rendered image text and ordinary text, holding the visual evidence constant? Does the model decode question content but fail to bind it to the diagram? This separates input access from reasoning difficulty.

2. **Dataset fit and extra data.** The roots are relevant only to simple category content. Presentation equivalence needs verified text/rendering annotations; natural exam/OCR questions require transcripts, diagrams and domain evidence.

3. **Novelty assessment.** EXAMS-V, VMMU and pixel-language probing already address textual presentation in multimodal tasks. The remaining semantic-accessibility question needs comparison with those studies; relates to G8/G13.

### P14. M4U

[Paper](https://openaccess.thecvf.com/content/WACV2026/html/Wang_M4U_Evaluating_Multilingual_Understanding_and_Reasoning_for_Large_Multimodal_Models_WACV_2026_paper.html). Basis: visual necessity, expert reasoning and cross-language question/image-text interactions.

1. **Unresolved questions.** When a model succeeds without the image, does a probe of its normal image-present states still recover image-specific truth, or mostly task priors? Can stronger instruction demand change which category information is accessible at the answer position? The distinction is between measured accessibility and actual evidential dependence.

2. **Dataset fit and extra data.** Shared labels support broad category-truth questions. Image-region dependence, exact object identity and expert reasoning need localization or domain annotations beyond the roots.

3. **Novelty assessment.** Mirage Probes already studies internal patterns associated with visually ungrounded success. Text shortcuts and modality collapse are established. No independent gap is retained solely from image necessity; relevant to the scope of G1/G2/G7.

### P15. Parrot: Multilingual Visual Instruction Tuning

[Paper](https://proceedings.mlr.press/v267/sun25ad.html). Basis: multilingual erosion, alignment diagnostics and MMMB's paired answers.

1. **Unresolved questions.** Does language-specific alignment preserve category information, language identity, or both? Can language compliance degrade while category accessibility survives? A broader training question asks whether alignment changes these two targets at different points in instruction tuning.

2. **Dataset fit and extra data.** Presence labels are available. Requested/actual response language, correctness and caption assertions need validation. Training-time erosion questions require suitable checkpoints or controlled training resources.

3. **Novelty assessment.** Parrot already proposes language-conditioned visual alignment, and Hinck et al. study English replies. Joint semantic/language-control accessibility remains the narrower candidate; training-time erosion also requires a checkpoint-literature check; relates to G6.

### P16. Cross-Lingual Gaps in Visual STEM Reasoning

[Paper](https://aclanthology.org/2026.mrl-1.13/). Basis: matched Indic questions, answer extraction, CoT/no-image ablations and translation limitations.

1. **Unresolved questions.** When longer reasoning instructions worsen an Indic score, does category evidence become less accessible before answering, change during generated text, or remain accessible while answer formatting fails? Are apparent language penalties also penalties in the observation/extraction process?

2. **Dataset fit and extra data.** Shared category truth is relevant to accessibility and response correctness. Assertion, language and format validity require extra labels. Mathematical reasoning requires structured domain targets.

3. **Novelty assessment.** The source already identifies extraction-related effects. Representation/response gaps and reasoning-induced hallucinations are established. The precise language-conditioned evidence-accessibility question needs further overthinking and evidence-persistence literature checks; relates to G6/G14.

### P17. HinTel-AlignBench

[Paper](https://aclanthology.org/2026.alvr-main.26/). Basis: translated/native tasks and the stated weakness of caption-generated distractors.

1. **Unresolved questions.** Could an answer-success probe classify easy synthetic questions using option wording and task regularities rather than image-grounded evidence? Do supposedly good representations remain predictive when those regularities are balanced? This concerns whether the probe's target is itself a valid measure of grounding.

2. **Dataset fit and extra data.** Independent root labels are relevant to benchmark-construction and probing-validity questions. Caption mentions and question correctness need annotation; fine-grained negative objects require new visual labels.

3. **Novelty assessment.** Distractor shortcuts and Mirage Probes are established, and the source flags weak distractors. Language-conditioned construction effects on probing conclusions remain the narrower candidate; relates to G2.

### P18. Not Truly Multilingual: Script Consistency as a Missing Dimension

[Paper](https://aclanthology.org/2026.findings-emnlp.232/). Basis: same-language parallel scripts, consistency, visual grounding and scope limitations.

1. **Unresolved questions.** Does a script-dependent answer change reflect different category information, a different accessible readout of that information, or different output-language/format control? Does strong script identification coexist with stable category discrimination? Script identity alone does not answer either question.

2. **Dataset fit and extra data.** Existing images and labels are suitable for a bounded category question. Verified parallel-script forms are extra text data. Natural multi-script-community claims require appropriate language coverage.

3. **Novelty assessment.** Script-dependent accuracy and consistency are the source contribution. Cross-script hidden-state probing and pixel-language work need review before the content-versus-control question can claim novelty; relates to G8.

### P19. BanglaVerse

[Paper](https://aclanthology.org/2026.findings-emnlp.2129/). Basis: fixed cultural content across languages/dialects, caption/VQA differences and human evaluation coverage.

1. **Unresolved questions.** Can general category recognition survive while culturally specific identity or associated facts become inaccessible under another language/dialect? Does the larger caption-generation penalty concern unavailable information, lexical expression, or evaluator sensitivity? Behavioral averages do not uniquely separate those possibilities.

2. **Dataset fit and extra data.** Generic category accessibility and mention questions fit the roots. Cultural facts, entity identity, dialect meaning and evaluator correctness require additional expert annotations.

3. **Novelty assessment.** The source compares recognition-style and generative tasks and addresses cultural/linguistic deficits. Cultural-knowledge probing and visual-reference factual access are close; relates to G1/G12.

### P20. Bangla-Bayanno

[Paper](https://arxiv.org/abs/2508.19887). Basis: LLM-assisted refinement, translation evaluation and acknowledged residual terminology issues.

1. **Unresolved questions.** Does refinement preserve the visual proposition, or mainly produce more familiar/fluent phrasing? If probe transfer improves after refinement, is that semantic recovery or a shift toward the source model's preferred wording? Can a probe separate those effects?

2. **Dataset fit and extra data.** Root truth anchors category meaning. Source/translated/refined text meaning, polarity and fidelity need bilingual annotation. Fine cultural terminology or object repairs require richer ground truth.

3. **Novelty assessment.** Translation quality, refinement and translationese probes are established. The narrower question concerns visual-truth fidelity versus linguistic familiarity and needs multimodal MT literature checks; relates to G3/G4.

### P21. JEEM

[Paper](https://aclanthology.org/2026.findings-eacl.18/). Basis: consistency/relevance/fluency/authenticity evaluation, DCScore and judge limitations.

1. **Unresolved questions.** Can a model represent category truth correctly while using the wrong dialect/register? In an open VLM judge, is the relevant image-content violation decodable even when the judge's final score favors a fluent but inaccurate caption? This probes the evaluator rather than only the caption generator.

2. **Dataset fit and extra data.** Shared presence is available. Dialect/register quality, caption criterion violations and judge correctness need human labels; internal judge questions require open-model access.

3. **Novelty assessment.** JEEM/DCScore already separate evaluation dimensions. MM-JudgeBench and multilingual detector/probe work constrain the content-versus-judgment question. Latent judge/reward-model literature remains to be checked; relates to G6/G10.

### P22. Multimodal Evaluation of Russian-language Architectures

[Paper](https://aclanthology.org/2026.eacl-long.94/). Basis: image tasks, symbolic/semantic scoring and stack-dependent reproducibility.

1. **Unresolved questions.** Are inferred representation differences stable across task formats and computational settings, or are they artifacts of the measured score and extraction stack? The scientifically broader question is whether a transferable capability readout remains meaningful when tasks demand different evidence.

2. **Dataset fit and extra data.** The roots support a limited category-accessibility question. They do not label general Russian multimodal capability, audio/video competence or all task-specific knowledge.

3. **Novelty assessment.** No additional independent question is retained from this paper. Cross-task accessibility is covered by G1/G9; reproducibility concerns alone do not establish a new substantive gap.

### P23. JMMMU

[Paper](https://aclanthology.org/2025.naacl-long.43/). Basis: culture-agnostic/culture-specific split, translation ablations and item-level answer reversals.

1. **Unresolved questions.** When aggregate language accuracy is similar but the successful items differ, do the languages expose different category information or different decision boundaries over similarly decodable information? Are reversals concentrated near ambiguous category boundaries, or do confidently decodable cases also flip?

2. **Dataset fit and extra data.** Shared category truth is relevant to item-level language disagreement and confidence. Cultural expertise and image-text translation require additional domain labels and transcripts.

3. **Novelty assessment.** Item-level language reversals are already reported, and confidence transfer is established. The relationship between directional disagreement, accessibility and decision reliability needs further multilingual readout checks; relates to G9.

### P24. KOFFVQA

[Paper](https://openaccess.thecvf.com/content/CVPR2025W/BEAM/html/Kim_KOFFVQA_An_Objectively_Evaluated_Free-form_VQA_Benchmark_for_Large_Vision-Language_CVPRW_2025_paper.html). Basis: explicit grading criteria and evaluator reliability comparisons.

1. **Unresolved questions.** When a judge gives the wrong overall score, did it fail to recognize the violated criterion, or recognize it but combine criteria poorly? Does the internal distinction between content correctness and surface quality survive across languages? Criterion recognition and final score aggregation are different targets.

2. **Dataset fit and extra data.** Covered category truth can ground a limited factuality question. Rubric violations, partial credit and full caption quality require human criterion labels and suitable judge access.

3. **Novelty assessment.** Criteria-based judging is already the source method. Reward-model probing, MM-JudgeBench and latent score aggregation are close. Whether factual-criterion accessibility diverges from multilingual judgment remains provisional; relates to G10.

### P25. KRETA

[Paper](https://aclanthology.org/2025.emnlp-main.1696/). Basis: reading/reasoning task separation and model-dependent Korean/English CoT effects.

1. **Unresolved questions.** Is apparent reasoning failure preceded by weak text recognition, failure to bind recognized symbols to the visual structure, or a later response problem? Do different languages change those stages differently? “System 1/System 2” task labels do not themselves identify corresponding internal mechanisms.

2. **Dataset fit and extra data.** Root labels provide only a coarse recognition target. Reading/reasoning questions require transcripts, locations, relations and task answers; captions are not exhaustive transcriptions.

3. **Novelty assessment.** Visual-document, MTVQA/XT-VQA and compositional grounding work are close. Stage-specific OCR/semantic accessibility needs further review; relates to extra-data G11/G13.

### P26. VMMU

[Paper](https://aclanthology.org/2026.findings-aacl.44/). Basis: rendered/separated question variants, English translation controls and native-domain evidence.

1. **Unresolved questions.** Why can English translation or input separation change answers when the intended task is unchanged? Does the change concern textual proposition accessibility, diagram interpretation, cross-modal binding, or answer selection? Does the native condition preserve distinctions that a translation obscures?

2. **Dataset fit and extra data.** The roots support only a limited category question. Meaning-equivalence and presentation need text annotations. Vietnamese exam reasoning requires visual evidence, transcripts and domain labels.

3. **Novelty assessment.** The source already addresses translation and presentation effects. The question concerns semantic accessibility under equivalent inputs, not the first native-versus-English or rendered-text comparison; relates to G3/G4/G8.

### P27. Afri-MCQA

[Paper](https://aclanthology.org/2026.acl-long.1869/). Basis: native/English cultural QA, format comparisons and language-understanding controls.

1. **Unresolved questions.** Can cultural facts be accessible from a native textual name but inaccessible from the image, or accessible through English but not the native question? Which linking step fails? Separately, does speech input fail before a usable linguistic representation is formed?

2. **Dataset fit and extra data.** Generic category questions fit the existing labels. Cultural identity/fact access requires names and fact relations; speech questions additionally require recordings, transcripts and alignment.

3. **Novelty assessment.** Visual factual-association probes and the source language controls already cover important components. Language-conditioned cultural identity/fact access needs further review; relates to extra-data G12. Speech is outside the present image-caption scope.

### P28. Lost in Translation: Do LVLM Judges Generalize Across Languages?

[Paper](https://aclanthology.org/2026.findings-acl.1746/). Basis: multilingual judge decisions, position/length bias and translated evaluation data.

1. **Unresolved questions.** Is a judge's cross-language preference failure due to weaker encoding of factual differences between candidates, or to a decision rule favoring a language, position or style despite encoded factual distinctions? Can a factuality readout transfer when the final preference does not?

2. **Dataset fit and extra data.** Category presence supports some caption-factuality questions. Verified candidate meanings, human judgments and open-judge states are needed; full chart/preference quality is outside the roots.

3. **Novelty assessment.** The source evaluates multilingual preferences and biases. Reward-model and multilingual hallucination probes are close. The factual-encoding versus judgment question requires a dedicated latent-judge literature check; relates to G10.

### P29. VLURes

[Paper](https://aclanthology.org/2026.findings-acl.1367/). Basis: long text, unrelated material and image-only versus joint tasks.

1. **Unresolved questions.** Does irrelevant or false text reduce accessibility of visual-category information, or leave it decodable while changing the selected answer? How does language affect that distinction as context grows? Persistent image evidence and successful resistance to misleading context need not coincide.

2. **Dataset fit and extra data.** The roots are relevant to broad visual-category truth. Textual assertions need validation. Natural long-document grounding requires documents, grounded evidence spans and discourse labels.

3. **Novelty assessment.** Distracting context, prompt-induced hallucination and evidence propagation already have prior work. The specific multilingual visual-truth accessibility question remains provisional; relates to G14.

### P30. CLAIM

[Paper](https://arxiv.org/abs/2506.11073). Basis: language-ID head probes, multilingual object evaluation, attention differences and English-centricity limitation.

1. **Unresolved questions.** How much do language-identity separability and category-truth accessibility actually covary? Does English-like geometry predict better presence discrimination, better answer expression, or merely more English-like output? Could native description choices remain appropriate despite representational differences?

2. **Dataset fit and extra data.** The roots support category-truth accessibility. Output language and mentions need annotation. Image-region claims require boxes, and cultural judgments need richer targets.

3. **Novelty assessment.** CLAIM already uses language-ID probes. The low-resource encoder-failure paper limits claims that decodable language identity identifies a cause. Target-specific content/compliance/transfer questions remain candidates; relates to G1/G6/G9.

### P31. Mind the (Language) Gap / MMCRICBench

[Paper](https://aclanthology.org/2025.mrl-main.38/). Basis: English/Hindi scorecards, fixed English QA and numerical tasks.

1. **Unresolved questions.** When the image script changes but the numerical facts do not, is a value decoded incorrectly, associated with the wrong table field, or correctly represented but used incorrectly in calculation? Those targets are more informative than a single multilingual numerical accuracy score.

2. **Dataset fit and extra data.** The main table/numerical question is outside the CSV. Cell transcripts, field-value bindings, quantities, operations and answers are required.

3. **Novelty assessment.** The source and KRETA/XT-VQA already cover related tasks. Multilingual table-binding probes need further review. No new dataset-only question is retained; relates to extra-data G11/G13.

### P32. Crossmodal-3600

[Paper](https://aclanthology.org/2022.emnlp-main.45/). Basis: image sampling, caption style guidance, annotator protocol and benchmark validation.

1. **Unresolved questions.** Do inferred language-specific representations reflect language-conditioned content selection, a small pool of annotator preferences, or collection guidance? How much can a probe trained on this benchmark generalize to new descriptive styles and scenes rather than memorizing image/annotator regularities?

2. **Dataset fit and extra data.** Existing captions and shared labels support limited category/mention questions. Separating language, annotator and instruction provenance needs more independent native writers and collection metadata.

3. **Novelty assessment.** The source and Berger/Ponti already discuss collection limits; descriptive-choice classifiers also exist. Language versus annotator/provenance remains an unresolved scope issue, not a population-level cultural finding; relates to G1/G3.

### P33. IGLUE

[Paper](https://proceedings.mlr.press/v162/bugliarello22a.html). Basis: zero/few-shot transfer, task differences and proposed multi-source extensions.

1. **Unresolved questions.** Does a source-trained readout fail because the target information is less accessible, represented in different coordinates, or poorly calibrated? Can multiple source languages improve access without relying on additional target labels or source-specific mention distributions?

2. **Dataset fit and extra data.** The common presence target is suitable for transfer-accessibility questions. Caption-content questions need validated assertions. Task-general capability and adaptation claims require additional labels/resources.

3. **Novelty assessment.** Transfer, alignment, calibration and source selection are established in IGLUE and multilingual probing. The content-controlled visual-category accessibility question needs a distinct contribution; relates to G3/G9.

### P34. MaRVL / Visually Grounded Reasoning across Languages and Cultures

[Paper](https://aclanthology.org/2021.emnlp-main.818/). Basis: native concept selection, non-universal WordNet concepts and reasoning over image pairs.

1. **Unresolved questions.** Does a model encode a visual distinction but lack a familiar native name for the chosen ontology category? Are coarse, transferred categories aligned across languages while culturally important fine distinctions are not? Does successful entity decoding coexist with failed relation binding?

2. **Dataset fit and extra data.** Broad-category naming questions are relevant. Native concept inventories, subtype boundaries, cultural identities and image-pair relations need new judgments and visual labels.

3. **Novelty assessment.** MaRVL establishes concept-selection problems; Babel-ImageNet and cultural adaptation are close. Multilingual lexical/hierarchy and relational accessibility questions require further checks; relates to G5/G11/G12.

### P35. xGQA

[Paper](https://aclanthology.org/2022.findings-acl.196/). Basis: structural question types, English answer space and zero/few-shot alignment analysis.

1. **Unresolved questions.** When a translated choice cannot be matched to the answer space, is the underlying visual category available independently of that lexical mapping? Do selected-choice and open-answer conditions access the same content? The broad output-alignment issue is already explicit in this paper.

2. **Dataset fit and extra data.** Root truth supports coarse category-versus-naming questions. Native meanings and label mappings need verification. Fine subtype choices or relations require extra labels.

3. **Novelty assessment.** The source already discusses English prediction-label alignment. Lexical readout and category-content accessibility need a narrower novelty assessment; relates to G5/G6/G9.

### P36. Babel-ImageNet

[Paper](https://aclanthology.org/2024.acl-long.277/). Basis: WordNet/BabelNet mapping, uneven language coverage and Anglocentric ontology limitations.

1. **Unresolved questions.** Does apparent cross-language visual failure reflect a missing/ambiguous label mapping rather than poor category evidence? Can native definitions recover a category readout when a conventional translation does not? At a deeper level, are the category boundaries themselves equivalent across languages?

2. **Dataset fit and extra data.** The shared 25-category ontology supports broad lexical-access questions. Polysemy and boundary equivalence need native judgments; fine hierarchy questions require new subtype labels.

3. **Novelty assessment.** Multilingual label mapping is established. Lexical grounding, verbalizers and hierarchy-sensitive evaluation constrain the ontology-aware accessibility question; relates to G5.

### P37. IndicVisionBench

[Paper](https://arxiv.org/abs/2511.04727). Basis: native/parallel questions, multiple answer formats, adversarial premises and no-image comparisons.

1. **Unresolved questions.** Are recognition, cultural identity and resistance to false premises different representation-level capabilities, or are benchmark scores conflating them with answer format? Does an adversarial statement change category accessibility or mainly bias the final decision?

2. **Dataset fit and extra data.** Generic category truth is available. Textual premise meanings need verification. Cultural identity, facts and OCR need separate annotations beyond root presence.

3. **Novelty assessment.** The source already includes adversarial questions and image-necessity analysis. M²CQA and prompt-induced hallucination are close. Generic adversarial Indic difficulty is not new; relates to G1/G14/G12.

### P38. XT-VQA

[Paper](https://arxiv.org/abs/2412.17787). Basis: image-text/question language mismatch and visual-information analysis.

1. **Unresolved questions.** Does cross-language text-rich failure occur at character recognition, semantic interpretation, or binding image text to the question? Does reduced output dependence on the image imply reduced category accessibility, or can those measures disagree? These require distinct targets rather than a single similarity or attention statistic.

2. **Dataset fit and extra data.** The roots do not control image-text language or label OCR semantics. Transcripts, source meanings and answer mappings are required for the main question; generic category accessibility is a narrower fit.

3. **Novelty assessment.** The source already studies cross-language visual/text mismatch. Document and OCR probing are close. Symbol-to-semantic accessibility needs further checking; relates to extra-data G13, with G6 covering the limited category analog.

### P39. Mitigating Multilingual Hallucination in Large VLMs / MHR

[Paper](https://arxiv.org/abs/2408.00550). Basis: multilingual invalid responses versus factual failures and English-oriented preference construction.

1. **Unresolved questions.** Can requested-language control, correct category information and factual assertions diverge independently? Would a hidden-state detector trained against a single hallucination label learn language/format artifacts rather than image inconsistency? Does English-oriented supervision transfer truth or merely response conventions?

2. **Dataset fit and extra data.** Category truth is available. Semantic correctness, response-language compliance, format validity and assertion status need separate validated labels. Full hallucination spans and training effects require additional resources.

3. **Novelty assessment.** MHR already separates instruction following and factuality. Team champ and other multilingual hidden-state hallucination detectors constrain novelty. The joint content/compliance/assertion question remains provisional; relates to G1/G6.

### P40. CCHall

[Paper](https://aclanthology.org/2025.acl-long.1485/). Basis: cross-language/cross-modal hallucination definitions and XM3600-derived caption examples.

1. **Unresolved questions.** Can a probe distinguish valid cross-language complementarity from genuine image inconsistency? Does disagreement mean two captions describe different true categories, or that one asserts an absent category? Can a detector separate object mention, asserted presence and linguistic compliance instead of learning a conflated label?

2. **Dataset fit and extra data.** Shared image truth and captions are relevant to broad complementarity-versus-inconsistency questions. Assertions need bilingual annotation; relationships, identity and depicted/uncertain claims require richer labels.

3. **Novelty assessment.** CCHall already includes XM3600. Not Every Mention Is a Claim directly addresses non-assertion and hidden-state detector limitations. Cross-language complementarity versus assertion/truth needs further entailment literature checks; relates to G1/G3.

### P41. Once Correct, Still Wrong / M²CQA

[Paper](https://arxiv.org/abs/2602.05437). Basis: true/counterfactual pairs, conditional hallucination rate and language/prompting differences.

1. **Unresolved questions.** When a model correctly answers a true statement but accepts a false alternative, is independently labelled visual truth still decodable? Does it also represent the misleading proposition, with language affecting the final selection between them? The conditional behavior is already known; the candidate concerns separable latent targets.

2. **Dataset fit and extra data.** The roots support broad visual truth. Verified proposition meanings are additional text labels. Counts and culturally specific counterfactual identities require new ground truth.

3. **Novelty assessment.** Conditional hallucination is the source contribution. Mirage, prompt-induced hallucination and layer-wise confound work are close. The multilingual visual-accessibility versus premise-acceptance question remains provisional; relates to G14.

### P42. Centurio

[Paper](https://arxiv.org/abs/2501.05122). Basis: controlled language mixtures, output-language fidelity and limits of multicultural training coverage.

1. **Unresolved questions.** Do different training mixtures change visual-category accessibility, cross-language readout compatibility, native descriptive choice and output-language control independently? Can similar benchmark scores conceal different failure profiles? Which changes occur during training rather than only at the final checkpoint?

2. **Dataset fit and extra data.** Shared labels support descriptive questions about category accessibility and caption content. Training-mixture causality needs comparable checkpoints/training resources; full cultural knowledge needs extra labels.

3. **Novelty assessment.** Centurio controls training composition; Parrot and Hinck et al. address erosion/fidelity. Joint target-specific accessibility remains a descriptive candidate, while causal training questions require additional resources; relates to G1/G6/G9.

### Closest-work ledger from this second pass

The distinction below is deliberate: **targeted primary-text inspection** is sufficient to reject an overly broad novelty claim, but is not a completed systematic novelty review. Entries screened at abstract level remain leads, not evidence that the precise candidate question has already been done. Methods from intervention papers are not adopted merely because those papers are listed.

| Primary source | Evidence inspected in this pass | Claim it rules out or narrows |
|---|---|---|
| [A Large-Scale Multilingual Study of Visual Constraints on Linguistic Selection of Descriptions](https://aclanthology.org/2023.findings-eacl.172/) | Primary paper, published research setup and classifier formulation | First prediction of linguistic choices from visual representations; narrows G1/G3. |
| [Why do LLaVA Vision-Language Models Reply to Images in English?](https://aclanthology.org/2024.findings-emnlp.783/) | Primary paper's stated findings and analysis scope | First internal analysis of image-induced English replies; narrows G6. |
| [Not Every Mention Is a Claim](https://arxiv.org/abs/2610.23610) | Primary text, non-assertion analysis, hidden-state absence probes and position-matched evaluations | First distinction between mentions and factual assertions; first probe of that detector failure; narrows G1/G2/G14. |
| [Can VLMs Recall Factual Associations From Visual References?](https://aclanthology.org/2025.findings-emnlp.850/) | Primary paper, factual-reference setup and §6.2 linking-failure probe | First visual-reference factual linking or its hidden-state detector; narrows G12. |
| [Multi-Object Hallucination in Vision-Language Models / ROPE](https://arxiv.org/abs/2407.06192) | Primary text, task/analysis structure and §5.3 class/order effects | First multi-object demand effect, salience/order study or co-occurrence shortcut finding; narrows G7. |
| [Mirage Probes](https://arxiv.org/abs/2606.13870) | Primary text's framework, contrasted mechanisms and experimental scope | First internal probe of visually ungrounded success or generic text-shortcut diagnosis. |
| [Team champ at SHROOM-Visions](https://aclanthology.org/2026.uncertainlp-main.62/) | Primary paper's multilingual hidden-state and text-to-vision transfer setup | First multilingual span-level VLM hallucination probe or text-to-vision probe transfer. |
| [Mechanisms of Prompt-Induced Hallucination](https://aclanthology.org/2026.acl-long.1941/) | Primary paper's controlled task and mechanistic scope | First explanation of prompt-over-vision behavior; narrows G14. |
| [Overthinking Causes Hallucination](https://arxiv.org/abs/2603.07619) | Primary text's layer-wise hypothesis framework and analysis scope | First layer-wise competing-object-hypothesis or confound-propagation study; narrows G14. |
| [Probing Contextual Language Models for Common Ground with Visual Representations](https://aclanthology.org/2021.naacl-main.422/) | Primary abstract/formulation screening | Text representations can already encode visually useful category information; further method review needed for G3/G5. |
| [Attribution-Guided Multi-Object Hallucination and Bias Detection](https://aclanthology.org/2026.eacl-long.210/) | Primary abstract screening | Multi-layer attribution for multi-object detection is established; methods-level comparison remains for G7. Its attribution method is not part of our design. |
| [Each Judge Its Own Yardstick](https://arxiv.org/abs/2606.22918) | Primary abstract screening; video-domain lead | Internal representational competence of VLM judges is already a research topic; full comparison needed before G10 novelty. |

Also carry forward the first pass's close comparisons: the XM3600 low-resource encoder-failure paper, Responses Fall Short, Encoded but Not Routed, Decodable but Misrouted, multilingual CLIPScore evaluation, calibrated caption assessment, SpanCalib-VLM, CHAIR and THuMB. They constrain any broad claim about representation/response gaps, geometry, confidence or factuality. This second pass does not reset those findings.

### What still needs checking before a novelty claim

| Candidate | Exact further-check target | Claim that would be too broad |
|---|---|---|
| G1 | Joint multilingual presence/selection/assertion probes with independent image truth and sparse human references | “VLMs internally know what they fail to say.” |
| G2 | Multilingual caption-label missingness/selection effects on **probe training and interpretation**, including positive-unlabeled and incomplete-caption supervision | “Captions omit objects” or “sampling affects scores.” |
| G3 | Semantic-content-controlled multilingual representation comparisons using independent versus translated captions | “Natural captions are not translations” or “alignment improves transfer.” |
| G4 | Separate latent targets for source fidelity and image truth in multimodal translation, including source-reliance probes | “Images sometimes hurt translation.” |
| G5 | Hierarchy-aware cross-language lexical grounding and label-verbalizer probes | “Synonyms change model answers.” |
| G6 | The relationship between category accessibility and language fidelity | “Images trigger English replies.” |
| G7 | Hidden-state category readouts under joint reporting demand with a language interaction | “Multiple objects increase hallucination.” |
| G8 | Semantic accessibility across script and input forms | “Non-Latin scripts are harder.” |
| G9 | The relationship between transfer, confidence reliability and independent visual truth | “Shared geometry supports transfer.” |
| G10 | Factual-violation accessibility versus final multilingual judge preference | “Judges are biased” or “a probe predicts judge error.” |
| G11 | Multilingual entity-to-relation/operation binding probes, not generic entity classifiers | “Relations are harder than nouns.” |
| G12 | Language-conditioned visual identity versus named-entity factual-association accessibility | “Models fail at cultural knowledge.” |
| G13 | Stage-specific multilingual OCR-symbol-to-question semantics probes | “OCR is weak in low-resource languages.” |
| G14 | The relationship between image truth and textual-proposition accessibility across languages | “Text priors override vision” or “intermediate states predict hallucination.” |

Perform backward/forward citation checks around the nearest paper for the **specific surviving empirical result**, including new preprints before submission. “Not found in our targeted search” remains a provisional assessment, never proof of absence. An established method does not make a research question novel; the unresolved scientific distinction must be clear.

## Status

Both sets of 42 paper reviews are retained. The candidate questions remain provisional, with dataset limits and close prior work documented. Question selection and relevant-paper pruning precede experimental design.
