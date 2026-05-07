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
├── HIPE-2026-data/          ← Official task data (cloned from hipe-eval/HIPE-2026-data)
│   ├── data/                   train/dev splits per language and corpus
│   ├── official_test_unlabeled/  unlabeled test files for submission
│   ├── schemas/                JSON schema for the data format
│   └── scripts/                official evaluation scripts
│
├── hipe-2026-llm-baseline/  ← Official LLM baseline (from organizers)
│
├── BIU_NLP_prediction/      ← Our prediction system (see README inside)
│   ├── predict_hf.py           Main inference script (HuggingFace Transformers)
│   ├── predict_openrouter.py   Alternative via OpenRouter API
│   ├── prompts/                Prompt templates (en / de / fr)
│   ├── results_hf/             All eval and test outputs + RESULTS_SUMMARY.md
│   └── submission/             Final submitted files + email + zip
│
└── ARCHIVE/                 ← Earlier experiments and scratch work
```

---

## Our Approach

Zero/few-shot prompting of open-weight LLMs — no fine-tuning.  
Models are loaded via HuggingFace Transformers and prompted to output a structured JSON classification for each person–place pair.

Models evaluated:
- **google/gemma-4-26B-A4B-it** — MoE, 4B active / 26B total
- **google/gemma-3-27b-it** — 27B dense
- **mistralai/Mistral-Small-24B-Instruct-2501** — 24B dense
- **CohereForAI/aya-expanse-32b** — 32B dense, multilingual
- **Qwen/Qwen3-30B-A3B-Instruct** — MoE, 3B active / 30B total

Prompt variants: English-language prompt vs. native-language prompt (DE/FR).  
Few-shot counts: 0, 3, 5, 10.

---

## Best Results (Dev Set, Global Macro-Recall)

| Model | DE | EN | FR |
|---|---|---|---|
| gemma-4-26B (en, 3-shot) | **0.7119** | **0.7349** | 0.5252 |
| gemma-3-27b (en, 0-shot) | 0.5622 | 0.7209 | **0.6611** |
| Mistral-Small-24B (native, 0-shot) | 0.5917 | 0.5935 | 0.5914 |
| aya-expanse-32b (en, 0-shot) | 0.5950 | 0.5967 | 0.6183 |

*All-FALSE baseline: ~0.4167*

---

## Submission

Submitted to HIPE 2026 on 2026-05-07.  
12 files (3 runs × 4 test files), one model per run:

| Test file | run1 | run2 | run3 |
|---|---|---|---|
| impresso-test-de | gemma4 (0.7119) | mistral (0.5917) | gemma3 (0.5622) |
| impresso-test-en | gemma4 (0.7349) | gemma3 (0.7209) | mistral (0.5935) |
| impresso-test-fr | gemma3 (0.6611) | mistral (0.5914) | gemma4 (0.5252) |
| surprise-test-fr | gemma3 (0.6611) | mistral (0.5914) | gemma4 (0.5252) |

Submission files and email: `BIU_NLP_prediction/submission/`  
Full results and analysis: `BIU_NLP_prediction/results_hf/RESULTS_SUMMARY.md`

---

## Quick Start

See [`BIU_NLP_prediction/README.md`](BIU_NLP_prediction/README.md) for setup instructions, usage examples, and full documentation of the prediction system.

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
- Participation guidelines: https://doi.org/10.5281/zenodo.17800136
