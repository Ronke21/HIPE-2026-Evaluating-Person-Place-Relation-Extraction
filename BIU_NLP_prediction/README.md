# BIU NLP — HIPE 2026 Prediction System

Bar-Ilan University NLP group submission for the [HIPE 2026 shared task](https://hipe-eval.github.io/HIPE-2026/) (CLEF 2026).  
**Task:** person–place relation qualification (*at* / *isAt*) in multilingual historical newspapers.

---

## Approach

Zero/few-shot prompting of open-weight large language models via HuggingFace Transformers — **no fine-tuning**.  
Each person–place entity pair is classified with a structured JSON response:

```json
{"at": "TRUE/FALSE/PROBABLE/POSSIBLE", "isAt": "TRUE/FALSE/PROBABLE/POSSIBLE"}
```

For each test file we selected the best configuration (prompt language × few-shot count) per model, ranked by global macro-recall on the dev set of the corresponding language, then submitted the top-3 models as run1/run2/run3.

---

## Models

| Key | HuggingFace ID | Parameters | VRAM (bfloat16) |
|---|---|---|---|
| `gemma4_26b` | google/gemma-4-26B-A4B-it | 26B total (4B active, MoE) | ~52 GB |
| `gemma3_27b` | google/gemma-3-27b-it | 27B dense | ~54 GB |
| `mistral_small_24b` | mistralai/Mistral-Small-24B-Instruct-2501 | 24B dense | ~48 GB |
| `aya_32b` | CohereForAI/aya-expanse-32b | 32B dense | ~64 GB |
| `qwen3_6_27b` | Qwen/Qwen3-30B-A3B-Instruct-2507 | 30B total (3B active, MoE) | ~60 GB |

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

> **Important:** `gemma-4-26B-A4B-it` requires `transformers>=5.8.0` (the `hipe2026` conda env).  
> The system Python (3.9, transformers 4.x) is not compatible with gemma4's tokenizer.

---

## Usage

Always activate the conda env and run from this directory:

```bash
conda activate hipe2026
cd "HIPE CLEF 2026 - Multilingual Relation Extraction/BIU_NLP_prediction"
```

### Quick smoke test (5 pairs per cell)
```bash
CUDA_VISIBLE_DEVICES=0 python predict_hf.py \
    --models gemma3_27b --mode eval --shots 0 --max-pairs 5 --prompt-langs en
```

### Full eval run (all shots, both prompt languages, eval only)
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

### Run two models in parallel across two GPU groups
```bash
python predict_hf.py \
    --models gemma3_27b mistral_small_24b \
    --mode all --shots 0 3 5 10 --prompt-langs en native \
    --gpu-groups 0,1 2,3 \
    2>&1 | tee full_run.log
```

### gemma4 — must use conda env python directly (not system python)
```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/nlp/ronke21/miniconda3/envs/hipe2026/bin/python predict_hf.py \
    --models gemma4_26b --mode all --shots 0 3 5 --prompt-langs en native \
    2>&1 | tee run_gemma4.log
```

### Fill missing cells into an existing run directory
Use `--output-dir` pointing to an existing run dir and `--_worker` to skip creating a new timestamped subdir. Resume logic skips cells where output files already exist.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
/home/nlp/ronke21/miniconda3/envs/hipe2026/bin/python predict_hf.py \
    --models gemma4_26b --mode test --shots 3 --prompt-langs en \
    --output-dir results_hf/run_20260506_200805 --_worker \
    2>&1 | tee run_gemma4_fill.log
```

---

## Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--models` | all | Model keys to run (e.g. `gemma3_27b mistral_small_24b`) |
| `--mode` | all | `eval` (dev scoring), `test` (submission files), `all` (both) |
| `--prompt-langs` | all | `en`, `native`, or `all` |
| `--shots` | 0 5 10 | Few-shot counts |
| `--gpu-groups` | — | Comma-separated GPU lists for parallel model runs |
| `--output-dir` | `results_hf/` | Output base dir |
| `--load-in-4bit` | off | 4-bit quantization (reduces VRAM ~4×) |
| `--batch-size` | 32 | Pairs per `model.generate` call |

---

## Output Layout

```
results_hf/
  run_{timestamp}/
    eval/{model}/{prompt_lang}/{n}shot/
      dev-{lang}.jsonl          ← predictions
      dev-{lang}-scores.txt     ← macro-recall breakdown (at / isAt / global)
      dev-{lang}-traces.jsonl   ← per-pair debug traces
    eval/{model}/
      comparison-{lang}.txt     ← cross-config comparison table
    test/{model}/{prompt_lang}/{n}shot/
      BIU-{model}-{pl}-{n}shot_HIPE-2026-v1.0-{testfile}_run1.jsonl
    run.log
    summary.txt

  RESULTS_SUMMARY.md            ← aggregated results across all runs
```

Score files use the format:
```
'global': macro_recall=0.XXXX
'at':     macro_recall=0.XXXX
'isAt':   macro_recall=0.XXXX
```

---

## Best Dev Results (2026-05-07)

| Model | DE | EN | FR |
|---|---|---|---|
| gemma-4-26B (en/3shot) | **0.7119** | **0.7349** | 0.5252 |
| gemma-3-27b (en/0shot) | 0.5622 | 0.7209 | **0.6611** |
| Mistral-Small-24B (native/0shot) | 0.5917 | 0.5935 | 0.5914 |
| aya-expanse-32b (en/0shot) | 0.5950 | 0.5967 | 0.6183 |

*All-FALSE baseline: ~0.4167*

---

## Submission

The final submitted files are in `submission/`:
- 12 JSONL files (3 runs × 4 test files), one per model per test file
- `email.txt` — submission email body
- `BIU_NLP_HIPE2026_submission.zip` — zip to attach to the submission email

See `results_hf/RESULTS_SUMMARY.md` for the full analysis.

---

## Files

| File | Description |
|---|---|
| `predict_hf.py` | Main inference script (HuggingFace Transformers) |
| `predict_openrouter.py` | Alternative script using OpenRouter API |
| `prompts/classify_pair_{lang}.txt` | Prompt templates per language |
| `requirements.txt` | Python dependencies |
| `results_hf/` | All eval and test prediction outputs |
| `submission/` | Final submission files |
