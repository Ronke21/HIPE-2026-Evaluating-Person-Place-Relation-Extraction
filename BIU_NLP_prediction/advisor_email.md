# Email to PhD Advisor — HIPE 2026 Shared Task

---

**To:** [Advisor]  
**Subject:** CLEF HIPE 2026 — Shared Task Participation & Results

---

Dear [Advisor],

I wanted to share the results of our participation in the **HIPE 2026 shared task** (CLEF 2026), which I completed recently. Below is a summary of the task, data, approach, and results.

---

## The Task

HIPE 2026 is a multilingual **relation extraction** shared task focused on historical newspaper text. The specific sub-task we addressed is **person–place relation qualification**: given a newspaper article with named entity annotations, classify the relationship between each annotated *person* and *place* entity pair.

Two relations are classified per pair:

- **`at`** — did the article provide evidence that the person was at the location at any point (before publication)? Labels: `TRUE`, `PROBABLE`, `FALSE`
- **`isAt`** — does the article suggest the location is the person's current/habitual location at publication time? Labels: `TRUE`, `FALSE`

The key challenge is that the text is **historical** (19th–early 20th century), contains **OCR noise** (digitized from microfilm), and must be handled in **three languages** (German, English, French). The relations are inherently ambiguous — the task requires careful reading of context, not just entity co-occurrence. The evaluation metric is **global macro-recall** over both relations.

17 teams participated in HIPE 2026.

---

## Data

The dataset was provided by the [Impresso project](https://impresso-project.ch), which digitized historical European newspapers. Each data instance is a newspaper article with pre-annotated named entities (persons and places), from which *person–place pairs* are sampled for classification.

### Dataset Splits

| Language | Train (articles / pairs) | Dev (articles / pairs) | Test (articles / pairs) |
|----------|--------------------------|------------------------|--------------------------|
| German   | 88 / 1,224 | 32 / 432 | 19 / 238 |
| English  | 56 / 496   | 17 / 151 | 19 / 162 |
| French   | 317 / 4,450 | 107 / 1,498 | 19 / 238 |
| **Total (impresso)** | **461 / 6,170** | **156 / 2,081** | **57 / 638** |
| Surprise-FR (hidden corpus) | — | — | 30 / 480 |

The **surprise test set** was an additional French test set from a corpus not revealed before submission, designed to test generalization beyond the Impresso domain.

Label distribution in training data is strongly skewed toward `FALSE` (most person-place pairs in an article have no meaningful relation), making the task a class-imbalance problem. The all-FALSE baseline achieves ~0.4167 macro-recall.

---

## Approach

We treated the task as a **zero/few-shot prompting** problem — no fine-tuning was performed. Each person–place pair is classified independently by prompting a large language model with the full article text and the two entity mentions.

**Prompt design:** The prompt instructs the model to act as a historian reading historical European text, provides the relation definitions and label constraints, asks the model to first write a brief chain-of-thought explanation for each relation, and then output a structured JSON answer:

```json
{"at_explanation": "...", "at": "TRUE/PROBABLE/FALSE",
 "isAt_explanation": "...", "isAt": "TRUE/FALSE"}
```

**Prompt language:** We tested both English-language prompts and native-language prompts (German, French).

**Few-shot variants:** 0, 3, 5, and 10 in-context examples.

**Models evaluated** (via HuggingFace Transformers, bfloat16, NVIDIA A100-SXM4-80GB GPUs):

| Model | Parameters | Notes |
|-------|-----------|-------|
| google/gemma-4-26B-A4B-it | 26B total, 4B active (MoE) | Best dev scores overall |
| google/gemma-3-27b-it | 27B dense | Most consistent on test |
| mistralai/Mistral-Small-24B-Instruct-2501 | 24B dense | Strongest native-language prompting |
| CohereForAI/aya-expanse-32b | 32B dense | Multilingual, evaluated on dev only |

**Submission strategy:** For each test file, we selected the best configuration (prompt language × shot count) per model using dev set macro-recall, then submitted the top-3 models as separate runs (run1, run2, run3).

---

## Results

### Development Set (Global Macro-Recall, Best Config per Model)

| Model | DE | EN | FR |
|-------|----|----|-----|
| gemma-4-26B (en, 10-shot) | **0.7518** | **0.7803** | **0.7481** |
| gemma-3-27b (en, 0-shot) | 0.5622 | 0.7209 | 0.6611 |
| Mistral-Small-24B (native, 0-shot) | 0.5917 | 0.5935 | 0.5914 |

*All-FALSE baseline: ~0.4167*

### Official Test Results (17 teams, 46–47 submissions per language)

Ranks shown as **rank / total submissions** in that category.

#### Main Evaluation

| Profile | run1 — gemma4 | run2 — gemma3 | run3 — mistral |
|---------|--------------|--------------|----------------|
| Overall (mean de/en/fr) | 41/46 — 0.4429 | **20/46 — 0.5781** | 24/46 — 0.5390 |
| German  | 37/46 — 0.4495 | 31/46 — 0.5050 | **20/46 — 0.5451** |
| English | 36/47 — 0.4583 | **19/47 — 0.5762** | 28/47 — 0.5101 |
| French  | 42/46 — 0.4208 | **9/46 — 0.6529** | 23/46 — 0.5617 |
| Surprise-FR (generalization) | **6/46 — 0.6837** | 22/46 — 0.5265 | 17/46 — 0.6000 |

#### Binary Evaluation (PROBABLE "at" mapped to TRUE)

| Profile | run1 — gemma4 | run2 — gemma3 | run3 — mistral |
|---------|--------------|--------------|----------------|
| Overall | 41/46 — 0.5252 | **20/46 — 0.6721** | 24/46 — 0.6274 |
| German  | 40/46 — 0.5329 | 24/46 — 0.6231 | **23/46 — 0.6381** |
| English | 36/47 — 0.5382 | **21/47 — 0.6524** | 28/47 — 0.5904 |
| French  | 42/46 — 0.5044 | **14/46 — 0.7409** | 23/46 — 0.6538 |
| Surprise-FR | **3/44 — 0.8804** | 26/44 — 0.6473 | 20/44 — 0.7262 |

### Highlights

- **Best overall run: Gemma-3 (run2), rank 20/46** with consistent performance across all three languages
- **Best single result: Gemma-3 French, rank 9/46** (main) and rank 14/46 (binary)
- **Best generalization: Gemma-4 (run1), Surprise-FR rank 6/46** (main) and **rank 3/44** (binary) — notably strong on the unseen test corpus
- An interesting anomaly: Gemma-4 had the best development scores (0.71–0.78) but underperformed on the impresso test sets while excelling on the surprise set — possibly reflecting better generalization at the cost of overfitting to impresso dev patterns

### Next Steps

- Submit a working notes paper to CLEF 2026 (due May 28) describing the system
- Investigate the Gemma-4 dev/test discrepancy
- The dataset and task are well-suited for exploring fine-tuned smaller models or cross-lingual transfer

The full code, results, and submission files are available at:  
https://github.com/Ronke21/HIPE-2026-Evaluating-Person-Place-Relation-Extraction

Please let me know if you have any questions or feedback.

Best regards,  
Ronke

---
*Results source: https://github.com/hipe-eval/hipe-2026-eval*
