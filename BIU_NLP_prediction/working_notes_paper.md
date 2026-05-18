# BIU NLP at HIPE 2026: Zero/Few-Shot LLM Prompting for Person–Place Relation Qualification in Historical Newspapers

**[Author Name]**  
Department of Computer Science, Bar-Ilan University, Ramat Gan, Israel  
[email@biu.ac.il]

---

## Abstract

We describe the BIU NLP system submitted to the HIPE 2026 shared task on person–place relation qualification in multilingual historical newspapers. Our approach uses zero/few-shot prompting of open-weight large language models (LLMs) without any task-specific fine-tuning. We evaluate four models — Gemma-4-26B (MoE), Gemma-3-27B, Mistral-Small-24B, and Aya-Expanse-32B — under varying prompt languages (English, native) and few-shot counts (0, 3, 5, 10). For each test file, we select the best configuration per model based on development set macro-recall, and submit the top three models as separate runs. Our best submission (Gemma-3-27B, English prompt, zero-shot) ranks **9th out of 46** for French and **19th out of 47** for English in the main evaluation. In the generalization setting (surprise French test), Gemma-4-26B ranks **6th out of 46** (main) and **3rd out of 44** (binary evaluation).

---

## 1. Introduction

HIPE 2026 is a shared task at CLEF 2026 focused on **person–place relation extraction and qualification** in digitized historical European newspapers. Building on the entity recognition and linking tasks of HIPE-2020 and HIPE-2022 [1, 2], HIPE 2026 targets the question *Who was where, when?* — aiming to support the reconstruction of life trajectories and mobility patterns from historical sources.

The task presents several challenges that make standard NLP approaches difficult to apply directly: (1) text is in three languages (German, English, French) spanning the 18th to early 20th century; (2) documents are digitized from microfilm and contain significant OCR noise; (3) the relations are semantically nuanced and context-dependent, not reducible to simple entity co-occurrence; and (4) training data is limited relative to the number and diversity of newspapers covered.

Given these characteristics, we hypothesized that large language models with strong multilingual and reasoning capabilities, prompted with clear task definitions and label constraints, could compete effectively without the overhead of fine-tuning. This paper describes our system, experimental results, and analysis.

---

## 2. Task Description

The HIPE 2026 relation qualification task takes as input a historical newspaper article with pre-annotated named entities (persons and locations, linked to Wikidata where possible), and a set of candidate **person–place pairs** extracted from the article. For each pair, the system must assign labels for two relations:

- **`at`**: Does the article provide textual evidence that the person was at the location at any point before publication? Labels: `TRUE`, `PROBABLE`, `FALSE`.
- **`isAt`**: Does the article support that the person's location in the article's immediate temporal horizon (current, ongoing, or very recent) is the given place? Labels: `TRUE`, `FALSE`.

The two relations are constrained: if `at` is `FALSE`, then `isAt` must also be `FALSE`; if `isAt` is `TRUE`, then `at` must also be `TRUE`. The label `PROBABLE` is only valid for `at`.

The evaluation metric is **global macro-recall**, averaged equally over both relations. This metric rewards systems that correctly recall positive instances of both relations. A system that predicts all pairs as `FALSE` achieves approximately **0.4167** macro-recall (the class imbalance baseline), since `FALSE` is the dominant label for both relations.

A secondary **binary evaluation** setting is also provided, in which `PROBABLE` predictions for `at` are mapped to `TRUE`, simplifying the problem to binary classification for both relations.

---

## 3. Data

The HIPE 2026 dataset is derived from the [Impresso project](https://impresso-project.ch), which digitized and annotated historical European newspapers. The data builds on the HIPE-2022 v2.1 named entity-annotated corpus [2], which covers multiple newspaper collections in German, English, and French spanning roughly 1780–1950. Named entity annotations (persons and locations) include Wikidata entity identifiers. Person–place pairs were extracted from each article, pre-annotated by an LLM ensemble, and manually reviewed.

### 3.1 Dataset Statistics

Table 1 shows the number of articles and person–place pairs per language and split used in our experiments.

**Table 1: Dataset statistics (articles / annotated pairs)**

| Split | German | English | French | Total (impresso) |
|-------|--------|---------|--------|------------------|
| Train | 88 / 1,224 | 56 / 496 | 317 / 4,450 | 461 / 6,170 |
| Dev   | 32 / 432   | 17 / 151  | 107 / 1,498 | 156 / 2,081 |
| Test (impresso) | 19 / 238 | 19 / 162 | 19 / 238 | 57 / 638 |
| Test (surprise-FR) | — | — | 30 / 480 | 30 / 480 |

The **surprise test set** is drawn from a different corpus (literary works in French) not revealed before submission, designed to test out-of-domain generalization. The label distribution is heavily skewed toward `FALSE` for both relations, reflecting the fact that most person–place pairs in a given article have no meaningful spatial relationship.

### 3.2 Data Characteristics

Key challenges of the data include: (1) OCR noise from historical digitization, including hyphenation artifacts, spelling variants, and line-break insertions within tokens; (2) archaic and historical spelling, particularly in German and older French; (3) long documents containing many persons and places, requiring the model to correctly scope its attention to the pair in question; and (4) ambiguous entity mentions (a person may appear in different surface forms across the article).

---

## 4. Approach

### 4.1 Overview

We frame the task as a **per-pair text classification** problem and address it with zero/few-shot prompting of large language models, with no task-specific fine-tuning. Each person–place pair is classified independently: the model receives the full article text and the two entity mentions (person and location surface forms), and outputs a structured prediction.

### 4.2 Prompt Design

We designed prompts that instruct the model to act as a historian reading a historical European newspaper article. The prompt includes:

1. **System role**: "You are a historian working on person-location relations in multilingual historical European newspapers."
2. **Task definition**: explicit descriptions of the `at` and `isAt` relations, including their temporal scope and label constraints.
3. **Decision guidance**: rules for when to use each label (e.g., `PROBABLE` only for indirect evidence; `isAt` must be `TRUE` or `FALSE` only; logical constraints between the two relations).
4. **Robustness rules**: explicit instructions to handle OCR noise, line-break artifacts, historical spelling variants, and hyphenation without using external knowledge.
5. **Chain-of-thought format**: the model is instructed to first output a brief explanation (`at_explanation`, `isAt_explanation`, at most 100 words each, citing only article evidence) before giving the label.
6. **Structured JSON output**: the final answer must be a JSON object with keys `at_explanation`, `at`, `isAt_explanation`, `isAt`.

The structured output with chain-of-thought reasoning serves two purposes: it encourages the model to ground its decision in the article text, and it provides interpretable traces for error analysis.

Prompts were prepared in **English** and in the **native document language** (German, French). The English prompt includes a note that the document is in the respective language; the native prompt is fully written in that language. We hypothesized that native-language prompts might improve performance for multilingual models, particularly for German.

### 4.3 Few-Shot Examples

We experimented with 0, 3, 5, and 10 in-context examples per prompt. Examples were sampled from the training set, selecting instances that cover a mix of `TRUE`, `PROBABLE`, and `FALSE` labels to avoid inducing a label bias. Each example includes the full JSON output (with explanations). At 10-shot, prompts become long (several thousand tokens), so we verified that all models supported the required context lengths.

### 4.4 Models

We evaluated four open-weight instruction-tuned models, all loaded via HuggingFace Transformers in bfloat16 precision on NVIDIA A100-SXM4-80GB GPUs:

**Table 2: Models evaluated**

| Model | HuggingFace ID | Parameters | Architecture |
|-------|----------------|------------|--------------|
| gemma4 | google/gemma-4-26B-A4B-it | 26B total (4B active) | MoE |
| gemma3 | google/gemma-3-27b-it | 27B | Dense |
| mistral | mistralai/Mistral-Small-24B-Instruct-2501 | 24B | Dense |
| aya | CohereForAI/aya-expanse-32b | 32B | Dense, multilingual |

Gemma-4 is a Mixture-of-Experts model with only 4B active parameters per forward pass despite its 26B total parameter count, making it computationally efficient. Aya-Expanse is purpose-built for multilingual tasks and covers all three task languages natively.

### 4.5 Configuration Selection and Submission Strategy

For each combination of model × prompt language × shot count, we ran inference on the development set and computed global macro-recall. We selected the best configuration per model per language, then submitted the top-3 models as run1/run2/run3 for each test file. This yielded 12 submission files (3 runs × 4 test files). Each run uses a single model across all test files.

The final submitted configurations are:

**Table 3: Submitted configurations per test file**

| Test file | run1 | run2 | run3 |
|-----------|------|------|------|
| impresso-test-de | gemma4 · en/3-shot | mistral · native/0-shot | gemma3 · native/0-shot |
| impresso-test-en | gemma4 · en/5-shot | gemma3 · en/0-shot | mistral · native/0-shot |
| impresso-test-fr | gemma4 · en/3-shot | gemma3 · en/0-shot | mistral · native/0-shot |
| surprise-test-fr | gemma4 · en/3-shot | gemma3 · en/0-shot | mistral · native/0-shot |

---

## 5. Results

### 5.1 Development Set Results

Table 4 reports the best development set macro-recall per model across all prompt language and shot count configurations. Gemma-4 consistently achieved the highest scores, with 10-shot English prompts performing best across all three languages.

**Table 4: Best development set results (global macro-recall)**

| Model | DE | EN | FR | Best config |
|-------|----|----|-----|-------------|
| gemma-4-26B (MoE) | **0.7518** | **0.7803** | **0.7481** | en / 10-shot |
| gemma-3-27b | 0.5622 | 0.7209 | 0.6611 | native / 0-shot (DE); en / 0-shot (EN, FR) |
| Mistral-Small-24B | 0.5917 | 0.5935 | 0.5914 | native / 0-shot |
| aya-expanse-32b | 0.5950 | 0.5967 | 0.6183 | en / 0-shot |

*All-FALSE baseline: ~0.4167*

All models substantially exceed the all-FALSE baseline. Gemma-4 shows a particularly large margin, especially for English (+0.36) and French (+0.33). Native-language prompting generally benefited Mistral (which is not a dedicated multilingual model) and Gemma-3 for German, while English prompts performed best for French and English across all models.

### 5.2 Effect of Shot Count

Increasing the number of in-context examples generally improved Gemma-4's performance up to 10 shots (e.g., +0.12 macro-recall for DE from 0-shot to 10-shot). For Gemma-3 and Mistral, few-shot examples provided inconsistent benefits and sometimes hurt performance, possibly due to prompt length effects or the model distributing attention away from the test article. We therefore submitted zero-shot configurations for these two models.

### 5.3 Official Test Results

Table 5 and 6 report official test set results from the HIPE 2026 evaluation. Rankings are shown as rank / total submissions in each category (46–47 impresso submissions, 44 surprise submissions).

**Table 5: Main evaluation — impresso profile score (macro-recall)**

| Profile | run1 — gemma4 | run2 — gemma3 | run3 — mistral |
|---------|--------------|--------------|----------------|
| Overall (mean DE/EN/FR) | 41/46 — 0.4429 | **20/46 — 0.5781** | 24/46 — 0.5390 |
| German | 37/46 — 0.4495 | 31/46 — 0.5050 | **20/46 — 0.5451** |
| English | 36/47 — 0.4583 | **19/47 — 0.5762** | 28/47 — 0.5101 |
| French | 42/46 — 0.4208 | **9/46 — 0.6529** | 23/46 — 0.5617 |
| Surprise-FR (generalization) | **6/46 — 0.6837** | 22/46 — 0.5265 | 17/46 — 0.6000 |

**Table 6: Binary evaluation — PROBABLE "at" mapped to TRUE**

| Profile | run1 — gemma4 | run2 — gemma3 | run3 — mistral |
|---------|--------------|--------------|----------------|
| Overall (mean DE/EN/FR) | 41/46 — 0.5252 | **20/46 — 0.6721** | 24/46 — 0.6274 |
| German | 40/46 — 0.5329 | 24/46 — 0.6231 | **23/46 — 0.6381** |
| English | 36/47 — 0.5382 | **21/47 — 0.6524** | 28/47 — 0.5904 |
| French | 42/46 — 0.5044 | **14/46 — 0.7409** | 23/46 — 0.6538 |
| Surprise-FR | **3/44 — 0.8804** | 26/44 — 0.6473 | 20/44 — 0.7262 |

Per-relation breakdown for the best run (run2, French):
- `at` macro-recall: 0.5703 (main), 0.7462 (binary)
- `isAt` macro-recall: 0.7356 (main), 0.7356 (binary)
- `at` accuracy: 0.6387 (main), 0.7227 (binary)
- `isAt` accuracy: 0.7773 (main), 0.7773 (binary)

---

## 6. Analysis and Discussion

### 6.1 Gemma-3 as the Most Reliable System

Gemma-3 (run2) is our strongest submission overall, ranking 20/46 in the main evaluation and achieving the best individual result among our runs (French, rank 9/46). It performs consistently across all three languages using a simple zero-shot English prompt, making it a robust baseline for this type of historical relation extraction task. The strong French performance may reflect the larger French training set (317 articles, 4,450 pairs) combined with the model's stronger French language capabilities.

### 6.2 The Gemma-4 Anomaly

The most striking finding is the divergence between Gemma-4's development and test performance. On the dev set, Gemma-4 was the clear winner (0.75–0.78 macro-recall across languages). Yet on the impresso test sets it ranked last among our three runs (41/46 overall). Conversely, on the surprise French test set it ranked first among our runs and achieved rank 6/46 overall (rank 3/44 in binary evaluation).

Several hypotheses may explain this:

1. **Dev/test domain shift within impresso**: the dev and test splits may cover different newspaper titles or time periods, and Gemma-4 may have captured idiosyncrasies of the dev set rather than generalizable patterns.
2. **PROBABLE label calibration**: Gemma-4's dev advantage may partly come from more frequent `PROBABLE` predictions for `at`. If the test set has fewer ambiguous cases (or a different distribution), this strategy could backfire. The binary evaluation, which maps `PROBABLE` → `TRUE`, shows a milder drop for Gemma-4 (0.52 vs 0.44 in main), consistent with this hypothesis.
3. **Few-shot example bias**: the 3/5-shot examples we used for the impresso test submissions may have introduced a distribution shift not present in the dev set.
4. **Surprise set characteristics**: the surprise set (French literary works) may be stylistically closer to Gemma-4's training distribution, explaining its strong generalization there.

### 6.3 Prompt Language Effects

Native-language prompting helped Mistral most substantially on German (0.5917 native vs 0.5389 en, 0-shot), which aligns with Mistral Small being primarily trained on English and European languages but less robustly on historical German. For French, native prompts did not help any model meaningfully, possibly because English and French are both well-represented in all model training data. For Gemma-3, native prompts helped on German (0.5622 native vs 0.5299 en) but the difference was modest.

### 6.4 Few-Shot Scaling

Few-shot examples improved Gemma-4 substantially but hurt Gemma-3 and Mistral in most settings. This may reflect differences in instruction-following ability and context utilization: Gemma-4's larger capacity allows it to integrate more examples effectively, while smaller models may struggle to attend to both the examples and the target article when the prompt grows long. Future work could explore retrieval-augmented few-shot selection (choosing examples similar to the target pair) rather than fixed examples.

### 6.5 Progress Beyond Baseline

All three submitted runs substantially exceed the all-FALSE baseline (~0.4167). Our best run achieves 0.5781 in the main evaluation (overall) and 0.6721 in the binary evaluation — representing gains of +0.16 and +0.26 respectively over baseline. The French result of 0.6529 (rank 9/46) represents a +0.24 gain over baseline, and the surprise-FR binary result of 0.8804 (rank 3/44) represents a +0.46 gain. These results demonstrate that zero/few-shot LLM prompting is a strong approach for this task even without fine-tuning.

---

## 7. Perspectives for Future Work

Several directions emerge from this work:

**Fine-tuned smaller models**: The strong zero-shot performance of large models motivates exploring whether a smaller model (7B–13B) fine-tuned on the HIPE 2026 training data could match or exceed the zero-shot performance of the 24–27B models, with lower inference cost.

**Retrieval-augmented few-shot**: Rather than fixed in-context examples, selecting examples based on similarity to the target pair (e.g., by article topic, entity type, or language period) could improve few-shot performance for all models.

**Relation-aware training**: The logical constraints between `at` and `isAt` (e.g., isAt=TRUE implies at=TRUE) could be enforced more rigorously via constrained decoding or post-processing, rather than relying solely on the model's prompt comprehension.

**Cross-lingual transfer**: The French training set is roughly 6× larger than German and 9× larger than English. Exploring cross-lingual transfer (training on French, evaluating on German/English) could improve low-resource language performance.

**Error analysis on the Gemma-4 dev/test gap**: A detailed comparison of Gemma-4 predictions on dev vs. test would help understand whether the discrepancy is driven by label calibration, document type, or entity type differences.

---

## References

[1] Ehrmann, M., et al. (2020). *CLEF HIPE 2020: Named Entity Recognition and Linking on Historical Newspapers*. CLEF 2020 Working Notes.

[2] Ehrmann, M., et al. (2022). *HIPE 2022: Participation Guidelines*. Zenodo. https://doi.org/10.5281/zenodo.6045662

[3] HIPE 2026 Participation Guidelines. Zenodo. https://doi.org/10.5281/zenodo.17800136

[4] Team, G., et al. (2024). *Gemma: Open Models Based on Gemini Research and Technology*. arXiv:2403.08295.

[5] Mistral AI. (2024). *Mistral Small*. https://mistral.ai

[6] Üstün, A., et al. (2024). *Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model*. arXiv:2402.07827.