# BIU NLP — HIPE 2026 System

Bar-Ilan University NLP group submission for the [HIPE 2026 shared task](https://hipe-eval.github.io/HIPE-2026/) (CLEF 2026).  
**Task:** Person–place relation qualification (*at* / *isAt*) in multilingual historical newspapers (German, English, French).

---

## Approach

Zero/few-shot prompting of open-weight large language models — **no fine-tuning**.

Each person–place entity pair is classified via a structured JSON prompt that first elicits a short chain-of-thought explanation, then a label for each relation:

```json
{"at_explanation": "...", "at": "TRUE/PROBABLE/FALSE",
 "isAt_explanation": "...", "isAt": "TRUE/FALSE"}
```

For each test file we evaluated all combinations of prompt language × shot count on the corresponding development set (metric: global macro-recall), then submitted the top-3 model configurations as separate runs.

---

## Models

| Key | HuggingFace ID | Parameters | VRAM (bfloat16) |
|-----|----------------|------------|-----------------|
| `gemma4_26b` | google/gemma-4-26B-A4B-it | 26B total (4B active, MoE) | ~52 GB |
| `gemma3_27b` | google/gemma-3-27b-it | 27B dense | ~54 GB |
| `mistral_small_24b` | mistralai/Mistral-Small-24B-Instruct-2501 | 24B dense | ~48 GB |
| `aya_32b` | CohereForAI/aya-expanse-32b | 32B dense | ~64 GB |

> Aya was evaluated on dev only and not included in the final submission.

---

## Submitted Runs

12 files total — 3 runs × 4 test files. Each run uses a distinct model.

| Test file | run1 | run2 | run3 |
|-----------|------|------|------|
| impresso-test-de | gemma4 · en/3-shot | mistral · native/0-shot | gemma3 · native/0-shot |
| impresso-test-en | gemma4 · en/5-shot | gemma3 · en/0-shot | mistral · native/0-shot |
| impresso-test-fr | gemma4 · en/3-shot | gemma3 · en/0-shot | mistral · native/0-shot |
| surprise-test-fr | gemma4 · en/3-shot | gemma3 · en/0-shot | mistral · native/0-shot |

Submitted files are in [`submission/`](submission/).

---

## Official Results (HIPE 2026, Final — team10 = BIU_NLP, Bar-Ilan University)

17 teams participated. Results from the [official evaluation repository](https://github.com/hipe-eval/hipe-2026-eval) (final de-anonymized release, 2026-05-20).  
Ranks shown as **rank / total submissions** in that category.

### Main Evaluation

| Profile | run1 (gemma4) | run2 (gemma3) | run3 (mistral) |
|---------|--------------|--------------|----------------|
| Overall (mean de/en/fr) | 41/46 — 0.4429 | **20/46 — 0.5781** | 24/46 — 0.5390 |
| German  | 37/46 — 0.4495 | 31/46 — 0.5050 | **20/46 — 0.5451** |
| English | 36/47 — 0.4583 | **19/47 — 0.5762** | 28/47 — 0.5101 |
| French  | 42/46 — 0.4208 | **9/46 — 0.6529**  | 23/46 — 0.5617 |
| Surprise-FR (generalization) | **6/46 — 0.6837** | 22/46 — 0.5265 | 17/46 — 0.6000 |
| Efficiency overall | 32/32 — 0.4429 | **30/32 — 0.5781** | 30/32 — 0.5390 |

### Binary Evaluation (PROBABLE "at" → TRUE)

| Profile | run1 (gemma4) | run2 (gemma3) | run3 (mistral) |
|---------|--------------|--------------|----------------|
| Overall (mean de/en/fr) | 41/46 — 0.5252 | **20/46 — 0.6721** | 24/46 — 0.6274 |
| German  | 40/46 — 0.5329 | 24/46 — 0.6231 | **23/46 — 0.6381** |
| English | 36/47 — 0.5382 | **21/47 — 0.6524** | 28/47 — 0.5904 |
| French  | 42/46 — 0.5044 | **14/46 — 0.7409** | 23/46 — 0.6538 |
| Surprise-FR | **3/44 — 0.8804** | 26/44 — 0.6473 | 20/44 — 0.7262 |

Full results with per-language sub-scores and efficiency profiles: [`results/OFFICIAL_RESULTS_team10.md`](results/OFFICIAL_RESULTS_team10.md).

---

## Dev Results Summary

Best macro-recall per model × language on the development set:

| Model | DE | EN | FR |
|-------|----|----|-----|
| gemma-4-26B-A4B-it (MoE) | **0.7518** (native/10-shot) | **0.7803** (en/10-shot) | **0.7481** (en/10-shot) |
| gemma-3-27b-it | 0.5622 (native/0-shot) | 0.7209 (en/0-shot) | 0.6611 (en/0-shot) |
| Mistral-Small-24B | 0.5917 (native/0-shot) | 0.5935 (native/0-shot) | 0.5914 (native/0-shot) |
| aya-expanse-32b | 0.5950 (en/0-shot) | 0.5967 (en/0-shot) | 0.6183 (en/0-shot) |

*All-FALSE baseline: ~0.4167*

Full grids: [`results/RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md).

---

## Repository Structure

```
BIU_NLP_prediction/
├── predict_hf.py                   # Main inference script (HuggingFace Transformers)
├── predict_openrouter.py           # Alternative script using OpenRouter API
├── prompts/
│   ├── classify_pair_de.txt        # German prompt template
│   ├── classify_pair_en.txt        # English prompt template
│   └── classify_pair_fr.txt        # French prompt template
├── requirements.txt
├── logs/                           # Run logs (gitignored)
├── submission/                     # Final submitted files (12 JSONL + email)
├── results/
│   ├── RESULTS_SUMMARY.md          # Dev evaluation results across all models
│   ├── OFFICIAL_RESULTS_team10.md  # Official competition results (team10)
│   ├── run_gemma3_aya/             # Eval + test outputs: gemma-3-27b, aya-expanse-32b
│   ├── run_gemma4/                 # Eval + test outputs: gemma-4-26B-A4B-it
│   └── run_mistral/                # Eval + test outputs: Mistral-Small-24B
└── ARCHIVE/                        # Superseded submissions and old result drafts
```

---

## Setup

```bash
conda create -n hipe2026 python=3.10 -y
conda activate hipe2026

pip install torch==2.5.1+cu121 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

pip install transformers>=5.8.0 accelerate>=1.11.0 bitsandbytes \
    openai scikit-learn jsonschema protobuf
```

> **Note:** `gemma-4-26B-A4B-it` requires `transformers>=5.8.0`. Use the conda env python explicitly for gemma4 (see examples below).

---

## Usage

```bash
conda activate hipe2026
cd BIU_NLP_prediction
```

### Dev evaluation (quick smoke test)
```bash
CUDA_VISIBLE_DEVICES=0 python predict_hf.py \
    --models gemma3_27b --mode eval --shots 0 --max-pairs 5 --prompt-langs en
```

### Full dev evaluation
```bash
CUDA_VISIBLE_DEVICES=0,1 python predict_hf.py \
    --models gemma3_27b --mode eval --shots 0 3 5 10 --prompt-langs en native \
    2>&1 | tee run_gemma3_eval.log
```

### Generate test submission files
```bash
CUDA_VISIBLE_DEVICES=0,1 python predict_hf.py \
    --models gemma3_27b --mode test --shots 0 --prompt-langs en native \
    2>&1 | tee run_gemma3_test.log
```

### Gemma-4 (use conda env python directly)
```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/nlp/ronke21/miniconda3/envs/hipe2026/bin/python predict_hf.py \
    --models gemma4_26b --mode all --shots 0 3 5 10 --prompt-langs en native \
    2>&1 | tee run_gemma4.log
```

### Two models in parallel across GPU groups
```bash
python predict_hf.py \
    --models gemma3_27b mistral_small_24b \
    --mode all --shots 0 3 5 10 --prompt-langs en native \
    --gpu-groups 0,1 2,3 \
    2>&1 | tee full_run.log
```

---

## Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--models` | all | Model keys (e.g. `gemma3_27b mistral_small_24b`) |
| `--mode` | all | `eval` (dev scoring), `test` (submission files), `all` |
| `--prompt-langs` | all | `en`, `native`, or `all` |
| `--shots` | 0 5 10 | Few-shot counts |
| `--gpu-groups` | — | Comma-separated GPU lists for parallel runs |
| `--output-dir` | `results/` | Output base directory |
| `--load-in-4bit` | off | 4-bit quantization (reduces VRAM ~4×) |
| `--batch-size` | 32 | Pairs per `model.generate` call |

---

## Output Layout

```
results/run_{model}/
  eval/{model}/{prompt_lang}/{n}shot/
    dev-{lang}.jsonl            ← predictions
    dev-{lang}-scores.txt       ← macro-recall breakdown (at / isAt / global)
    dev-{lang}-traces.jsonl     ← per-pair debug traces
  eval/{model}/
    comparison-{lang}.txt       ← cross-config comparison table
  test/{model}/{prompt_lang}/{n}shot/
    BIU-{model}-{pl}-{n}shot_HIPE-2026-v1.0-{testfile}_run1.jsonl
  run.log
  summary.txt
```