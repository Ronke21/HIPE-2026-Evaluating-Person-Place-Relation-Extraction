# HIPE CLEF 2026 — Multilingual Relation Extraction

Bar-Ilan University NLP group workspace for the [HIPE 2026 shared task](https://hipe-eval.github.io/HIPE-2026/) at CLEF 2026.

**Task:** extraction and qualification of person–place relations (*at* / *isAt*) in multilingual historical newspapers.  
**Languages:** German (DE), English (EN), French (FR) + surprise French test set.  
**Team:** BIU NLP, Bar-Ilan University.

---

## What is HIPE 2026?

HIPE 2026 asks: *Who was where, when?*

Given a historical newspaper article with named entity annotations, the task is to classify each **person–place pair** with two binary-ish relations:

- **`at`** — the person is physically at (or was at) the place
- **`isAt`** — the person's permanent/habitual location is the place

Each relation is classified as one of: `TRUE`, `FALSE`, `PROBABLE`, `POSSIBLE`.

Evaluation metric: **global macro-recall** over both relations.

---

## Repository Layout

```
HIPE CLEF 2026 - Multilingual Relation Extraction/
│
├── HIPE-2026-data/          ← Official task data (submodule)
│   ├── data/                   train/dev splits per language and corpus
│   ├── official_test_unlabeled/  unlabeled test files for submission
│   ├── schemas/                JSON schema for the data format
│   └── scripts/                official evaluation scripts
│
├── hipe-2026-llm-baseline/  ← Official LLM baseline (submodule)
│
├── hipe-2026-eval/          ← Official evaluation results repo (submodule)
│
├── BIU_NLP_prediction/      ← Our prediction system (see README inside)
│   ├── predict_hf.py           Main inference script (HuggingFace Transformers)
│   ├── predict_openrouter.py   Alternative via OpenRouter API
│   ├── prompts/                Prompt templates (en / de / fr)
│   ├── logs/                   Run logs (gitignored)
│   ├── results/                All eval and test outputs + results summaries
│   └── submission/             Final submitted files + email
│
└── ARCHIVE/                 ← Earlier experiments and scratch work (gitignored)
```

---

## Our Approach

Zero/few-shot prompting of open-weight LLMs — no fine-tuning.  
Models are loaded via HuggingFace Transformers and prompted to output a structured JSON classification for each person–place pair, with a chain-of-thought explanation before each label.

Models evaluated:
- **google/gemma-4-26B-A4B-it** — MoE, 4B active / 26B total
- **google/gemma-3-27b-it** — 27B dense
- **mistralai/Mistral-Small-24B-Instruct-2501** — 24B dense
- **CohereForAI/aya-expanse-32b** — 32B dense, multilingual
- **Qwen/Qwen3-30B-A3B-Instruct-2507** — MoE, 3B active / 30B total

Prompt variants: English-language prompt vs. native-language prompt (DE/FR).  
Few-shot counts: 0, 3, 5, 10.

---

## Best Results (Dev Set, Global Macro-Recall)

| Model | DE | EN | FR |
|---|---|---|---|
| gemma-4-26B (en, 10-shot) | **0.7518** | **0.7803** | **0.7481** |
| gemma-3-27b (en, 0-shot) | 0.5622 | 0.7209 | 0.6611 |
| Mistral-Small-24B (native, 0-shot) | 0.5917 | 0.5935 | 0.5914 |
| aya-expanse-32b (en, 0-shot) | 0.5950 | 0.5967 | 0.6183 |

*All-FALSE baseline: ~0.4167*

---

## Submission

Final submission sent 2026-05-08 (updated). 12 files — 3 runs × 4 test files, one model per run, selected by best dev score per language:

| Test file | run1 | run2 | run3 |
|---|---|---|---|
| impresso-test-de | gemma4 · en/3-shot (dev 0.7119) | mistral · native/0-shot (dev 0.5917) | gemma3 · native/0-shot (dev 0.5622) |
| impresso-test-en | gemma4 · en/5-shot (dev 0.7550) | gemma3 · en/0-shot (dev 0.7209) | mistral · native/0-shot (dev 0.5935) |
| impresso-test-fr | gemma4 · en/3-shot (dev 0.7218) | gemma3 · en/0-shot (dev 0.6611) | mistral · native/0-shot (dev 0.5914) |
| surprise-test-fr | gemma4 · en/3-shot (dev 0.7218) | gemma3 · en/0-shot (dev 0.6611) | mistral · native/0-shot (dev 0.5914) |

Submission files and email: [`BIU_NLP_prediction/submission/`](BIU_NLP_prediction/submission/)

---

## Official Results (HIPE 2026, Preliminary)

17 teams, 46–47 submissions per language. Ranks shown as **rank / total**.

### Main Evaluation

| Profile | run1 — gemma4 | run2 — gemma3 | run3 — mistral |
|---------|--------------|--------------|----------------|
| Overall (mean de/en/fr) | 41/46 — 0.4429 | **20/46 — 0.5781** | 24/46 — 0.5390 |
| German  | 37/46 — 0.4495 | 31/46 — 0.5050 | **20/46 — 0.5451** |
| English | 36/47 — 0.4583 | **19/47 — 0.5762** | 28/47 — 0.5101 |
| French  | 42/46 — 0.4208 | **9/46 — 0.6529** | 23/46 — 0.5617 |
| Surprise-FR (generalization) | **6/46 — 0.6837** | 22/46 — 0.5265 | 17/46 — 0.6000 |

### Binary Evaluation (PROBABLE "at" → TRUE)

| Profile | run1 — gemma4 | run2 — gemma3 | run3 — mistral |
|---------|--------------|--------------|----------------|
| Overall (mean de/en/fr) | 41/46 — 0.5252 | **20/46 — 0.6721** | 24/46 — 0.6274 |
| German  | 40/46 — 0.5329 | 24/46 — 0.6231 | **23/46 — 0.6381** |
| English | 36/47 — 0.5382 | **21/47 — 0.6524** | 28/47 — 0.5904 |
| French  | 42/46 — 0.5044 | **14/46 — 0.7409** | 23/46 — 0.6538 |
| Surprise-FR | **3/44 — 0.8804** | 26/44 — 0.6473 | 20/44 — 0.7262 |

**Highlights:**
- **run2 (Gemma-3) French: rank 9/46** — strongest impresso result
- **run1 (Gemma-4) Surprise-FR: rank 6/46 main · rank 3/44 binary** — best generalization
- **run2 overall: rank 20/46** — consistent across all three languages

Full results with sub-scores and efficiency profiles: [`BIU_NLP_prediction/results/OFFICIAL_RESULTS_team10.md`](BIU_NLP_prediction/results/OFFICIAL_RESULTS_team10.md)

---

## Quick Start

See [`BIU_NLP_prediction/README.md`](BIU_NLP_prediction/README.md) for setup instructions, usage examples, and full documentation.

```bash
conda activate hipe2026
cd "BIU_NLP_prediction"

# Evaluate gemma3 on all dev languages, 0-shot, English prompt
CUDA_VISIBLE_DEVICES=0,1 python predict_hf.py \
    --models gemma3_27b --mode eval --shots 0 --prompt-langs en
```

---

## References

- HIPE 2026 website: https://hipe-eval.github.io/HIPE-2026/
- Data repository: https://github.com/hipe-eval/HIPE-2026-data
- Evaluation results: https://github.com/hipe-eval/hipe-2026-eval
- Participation guidelines: https://doi.org/10.5281/zenodo.17800136
