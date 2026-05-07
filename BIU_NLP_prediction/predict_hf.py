#!/usr/bin/env python3

# conda activate hipe2026 
# cd HIPE\ CLEF\ 2026\ -\ Multilingual\ Relation\ Extraction/BIU_NLP_prediction/
# CUDA_VISIBLE_DEVICES=6 python predict_hf.py --models gemma4_26b --model-ids gemma4_26b=google/gemma-4-26B-A4B-it
# CUDA_VISIBLE_DEVICES=7 python predict_hf.py --models qwen3_6_27b --model-ids qwen3_6_27b=Qwen/Qwen3.6-27B-FP8

"""HIPE-2026 HuggingFace prediction script — BIU NLP.

Identical evaluation logic to predict_openrouter.py, but loads models
directly from HuggingFace using the `transformers` library — no server needed.

Models:
  gemma3_27b  → google/gemma-3-27b-it
  gemma4_31b  → google/gemma-3-27b-it   (Gemma 4 not yet on HF; override if available)
  qwen3_5     → Qwen/Qwen2.5-32B-Instruct
  aya_32b     → CohereForAI/aya-expanse-32b

NOTE: gpt_oss_20b is an OpenAI-internal model and is not available on HuggingFace.
      Use --model-ids gpt_oss_20b=<your-hf-id> to substitute an equivalent.

Memory requirements (bfloat16 / 4-bit):
  27B → ~54 GB / ~14 GB    32B → ~64 GB / ~18 GB
  Use --load-in-4bit if VRAM is limited.

Conda environment setup:
  conda create -n hipe2026 python=3.10 -y
  conda activate hipe2026
  pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \\
      --index-url https://download.pytorch.org/whl/cu121
  pip install transformers accelerate==1.11.0 bitsandbytes openai scikit-learn jsonschema protobuf

Quick start:
  conda activate hipe2026

  # smoke test, one model, 5 pairs per cell:
  python predict_hf.py --models gemma4_31b --mode eval --shots 0 --max-pairs 5 --prompt-langs en

  # full run — both models in parallel across two GPU groups, all shots, eval + test submissions:
  python predict_hf.py --mode all --shots 0 3 5 --prompt-langs all --batch-size 32 --gpu-groups 1,2,3 5,6,7 2>&1 | tee full_run.log

Output layout (same as predict_openrouter.py):
  results_hf/run_{timestamp}/
    eval/{model}/{prompt_lang}/{n}shot/dev-{lang}.jsonl
    eval/{model}/{prompt_lang}/{n}shot/dev-{lang}-scores.txt
    eval/{model}/{prompt_lang}/{n}shot/dev-{lang}-traces.jsonl
    eval/{model}/comparison-{lang}.txt
    test/{model}/{prompt_lang}/{n}shot/BIU-{model}-{pl}-{n}shot_{stem}_run1.jsonl
    run.log
    summary.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os

# Prevent CUDA memory fragmentation: after OOM + empty_cache() cycles,
# expandable segments let CUDA reuse existing allocations instead of
# requiring new contiguous blocks — fixes batch-size=1 OOM on long prompts.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository paths  (identical to predict_openrouter.py)
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
BASELINE_SRC = REPO_ROOT / "hipe-2026-llm-baseline" / "src"
HIPE_DATA_DIR = REPO_ROOT / "HIPE-2026-data"

LANG_PROMPT_FILES: dict[str, Path] = {
    "de": THIS_DIR / "prompts" / "classify_pair_de.txt",
    "en": THIS_DIR / "prompts" / "classify_pair_en.txt",
    "fr": THIS_DIR / "prompts" / "classify_pair_fr.txt",
}

SANDBOX_DEV: dict[str, Path] = {
    "de": HIPE_DATA_DIR / "data" / "sandbox" / "de-dev.jsonl",
    "en": HIPE_DATA_DIR / "data" / "sandbox" / "en-dev.jsonl",
    "fr": HIPE_DATA_DIR / "data" / "sandbox" / "fr-dev.jsonl",
}

SANDBOX_TRAIN: dict[str, Path] = {
    "de": HIPE_DATA_DIR / "data" / "sandbox" / "de-train.jsonl",
    "en": HIPE_DATA_DIR / "data" / "sandbox" / "en-train.jsonl",
    "fr": HIPE_DATA_DIR / "data" / "sandbox" / "fr-train.jsonl",
}

TEST_FILES: list[Path] = [
    HIPE_DATA_DIR / "official_test_unlabeled" / "HIPE-2026-v1.0-impresso-test-de.jsonl",
    HIPE_DATA_DIR / "official_test_unlabeled" / "HIPE-2026-v1.0-impresso-test-en.jsonl",
    HIPE_DATA_DIR / "official_test_unlabeled" / "HIPE-2026-v1.0-impresso-test-fr.jsonl",
    HIPE_DATA_DIR / "official_test_unlabeled" / "HIPE-2026-v1.0-surprise-test-fr.jsonl",
]

_TEST_FILE_LANG: dict[str, str] = {f.stem: f.stem[-2:] for f in TEST_FILES}

# ---------------------------------------------------------------------------
# Baseline imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(BASELINE_SRC))

from hipe2026_mistral_baseline.io_hipe import HipeDocument, load_jsonl, write_jsonl  # noqa: E402
from hipe2026_mistral_baseline.pair_generation import PairTask, build_pair_tasks  # noqa: E402
from hipe2026_mistral_baseline.parsing import ParseError, parse_model_response  # noqa: E402
from hipe2026_mistral_baseline.prompting import (  # noqa: E402
    _build_pair_context,
    _format_mentions,
    build_prompt,
)
from hipe2026_mistral_baseline.validation import (  # noqa: E402
    apply_prediction_to_pair,
    conservative_default_prediction,
    validate_prediction,
)

# ---------------------------------------------------------------------------
# Model registry  (HuggingFace model IDs)
# ---------------------------------------------------------------------------
DEFAULT_MODELS: dict[str, str] = {
    "gemma3_27b":  "google/gemma-3-27b-it",
    "aya_32b":     "CohereLabs/aya-expanse-32b",
    "mistral_small_24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    "qwen3_30b":   "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",  # ~60 GiB — needs 1×A100-80GB or 4-bit quant
    "gemma4_26b":  "google/gemma-4-26B-A4B-it",      # ~60 GiB — needs 1×A100-80GB or 4-bit quant
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HuggingFace runner
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _GenerationResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    elapsed_seconds: float


class HuggingFaceRunner:
    """Loads a model from HuggingFace and runs local inference."""

    def __init__(
        self,
        model_id: str,
        *,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        flash_attn: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Install required packages: pip install transformers accelerate bitsandbytes torch"
            ) from exc

        self.model_id = model_id
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._lock = threading.Lock()   # serialize GPU calls from multiple threads

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        elif load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        # Flash Attention 2 — optional, falls back silently if not installed
        extra_kwargs: dict = {}
        if flash_attn:
            try:
                import flash_attn  # noqa: F401
                extra_kwargs["attn_implementation"] = "flash_attention_2"
                LOGGER.info("Flash Attention 2 enabled for %s", model_id)
            except ImportError:
                LOGGER.warning(
                    "flash-attn not installed — falling back to default attention "
                    "(pip install flash-attn to enable)"
                )

        LOGGER.info("Loading tokenizer: %s", model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        # Left-padding is required for correct batch generation with decoder-only models
        self._tokenizer.padding_side = "left"

        # Cap per-GPU model weight usage so each GPU keeps headroom for KV cache.
        # Without this, device_map="auto" may pack all weights onto GPU 0 (≥79 GiB
        # on an A100-80GB), leaving <200 MiB free — not enough for even batch=1.
        num_gpus = torch.cuda.device_count()
        max_memory: dict | None = None
        if num_gpus > 1 and not load_in_4bit and not load_in_8bit:
            gpu_total_gib = int(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
            per_gpu_gib = max(10, gpu_total_gib - 15)  # leave ≥15 GiB per GPU for KV cache
            max_memory = {i: f"{per_gpu_gib}GiB" for i in range(num_gpus)}
            LOGGER.info("max_memory: %d GiB × %d GPUs (headroom=15 GiB)", per_gpu_gib, num_gpus)

        LOGGER.info(
            "Loading model: %s  dtype=%s  4bit=%s  8bit=%s  flash_attn=%s",
            model_id, torch_dtype, load_in_4bit, load_in_8bit,
            "flash_attention_2" in extra_kwargs,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=dtype,
            quantization_config=quant_config,
            max_memory=max_memory,
            **extra_kwargs,
        )
        self._model.eval()
        LOGGER.info("Model loaded: %s", model_id)

    def generate(self, prompt: str) -> _GenerationResult:
        import torch

        messages = [{"role": "user", "content": prompt}]

        if getattr(self._tokenizer, "chat_template", None):
            input_ids = self._tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
        else:
            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids

        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)
        n_prompt_tokens = input_ids.shape[-1]

        gen_kwargs: dict = dict(
            max_new_tokens=self._max_new_tokens,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        if self._temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self._temperature)
        else:
            gen_kwargs["do_sample"] = False

        started = time.perf_counter()
        with self._lock:
            with torch.no_grad():
                output_ids = self._model.generate(input_ids, **gen_kwargs)
        elapsed = time.perf_counter() - started

        new_tokens = output_ids[0][n_prompt_tokens:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return _GenerationResult(
            text=text,
            prompt_tokens=n_prompt_tokens,
            completion_tokens=len(new_tokens),
            elapsed_seconds=elapsed,
        )

    def generate_batch(self, prompts: list[str]) -> list[_GenerationResult]:
        """Run inference on a batch of prompts in one model.generate call."""
        import torch

        has_template = bool(getattr(self._tokenizer, "chat_template", None))

        # Apply chat template (or use raw prompt) for each item, producing strings
        if has_template:
            text_inputs = [
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]
        else:
            text_inputs = prompts

        # Tokenize together with left-padding so all sequences align on the right
        encoded = self._tokenizer(
            text_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        n_padded_prompt = input_ids.shape[1]

        gen_kwargs: dict = dict(
            max_new_tokens=self._max_new_tokens,
            pad_token_id=self._tokenizer.pad_token_id,
            attention_mask=attention_mask,
        )
        if self._temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self._temperature)
        else:
            gen_kwargs["do_sample"] = False

        try:
            started = time.perf_counter()
            with self._lock:
                with torch.no_grad():
                    output_ids = self._model.generate(input_ids, **gen_kwargs)
            elapsed = time.perf_counter() - started
        except torch.cuda.OutOfMemoryError:
            # Free the cache and retry with half the batch size
            torch.cuda.empty_cache()
            if len(prompts) == 1:
                raise
            half = len(prompts) // 2
            LOGGER.warning(
                "OOM on batch_size=%d for model %s — splitting to %d + %d and retrying",
                len(prompts), self.model_id, half, len(prompts) - half,
            )
            return self.generate_batch(prompts[:half]) + self.generate_batch(prompts[half:])

        results = []
        per_item_elapsed = elapsed / len(prompts)
        for i in range(len(prompts)):
            new_tokens = output_ids[i][n_padded_prompt:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            prompt_tokens = int(attention_mask[i].sum().item())
            results.append(_GenerationResult(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(new_tokens),
                elapsed_seconds=per_item_elapsed,
            ))
        return results


# ---------------------------------------------------------------------------
# Few-shot sampler  (identical to predict_openrouter.py)
# ---------------------------------------------------------------------------
_PREFERRED_LABEL_ORDER = [
    ("TRUE", "TRUE"),
    ("TRUE", "FALSE"),
    ("PROBABLE", "FALSE"),
    ("FALSE", "FALSE"),
]


class FewShotSampler:
    """Samples balanced few-shot examples from the sandbox training data."""

    def __init__(self, n_shots: int, seed: int = 42) -> None:
        self._n_shots = n_shots
        self._rng = random.Random(seed)
        self._cache: dict[str, list[tuple[PairTask, dict]]] = {}

    def _load(self, lang: str) -> list[tuple[PairTask, dict]]:
        if lang in self._cache:
            return self._cache[lang]
        path = SANDBOX_TRAIN.get(lang)
        if not path or not path.exists():
            LOGGER.warning("No training file for lang='%s', few-shot unavailable", lang)
            self._cache[lang] = []
            return []
        pool: list[tuple[PairTask, dict]] = []
        for doc in load_jsonl(path):
            for pair in doc.sampled_pairs:
                if pair.at is None or pair.is_at is None:
                    continue
                task = PairTask(
                    document_id=doc.document_id,
                    language=doc.language,
                    publication_date=doc.date,
                    text=doc.text,
                    pair=pair,
                )
                pool.append((task, {
                    "at": pair.at,
                    "isAt": pair.is_at,
                    "at_explanation": pair.at_explanation or "",
                    "isAt_explanation": pair.is_at_explanation or "",
                }))
        LOGGER.info("Loaded %d training examples for lang=%s", len(pool), lang)
        self._cache[lang] = pool
        return pool

    def sample(self, lang: str) -> list[tuple[PairTask, dict]]:
        if self._n_shots == 0:
            return []
        pool = self._load(lang)
        if not pool:
            return []
        by_label: dict[tuple, list] = {}
        for item in pool:
            key = (item[1]["at"], item[1]["isAt"])
            by_label.setdefault(key, []).append(item)
        selected: list[tuple[PairTask, dict]] = []
        for label_key in _PREFERRED_LABEL_ORDER:
            if len(selected) >= self._n_shots:
                break
            bucket = by_label.get(label_key, [])
            if bucket:
                selected.append(self._rng.choice(bucket))
        used_ids = {id(ex) for ex in selected}
        remainder = [ex for ex in pool if id(ex) not in used_ids]
        while len(selected) < self._n_shots and remainder:
            pick = self._rng.choice(remainder)
            selected.append(pick)
            remainder = [ex for ex in remainder if id(ex) != id(pick)]
        return selected[: self._n_shots]


# ---------------------------------------------------------------------------
# Prompt construction  (identical to predict_openrouter.py)
# ---------------------------------------------------------------------------
_ARTICLE_SNIPPET_CHARS = 400


def _format_few_shot_block(examples: list[tuple[PairTask, dict]]) -> str:
    if not examples:
        return ""
    lines = [
        "Here are some labeled examples from the training data to guide your classification:",
        "",
    ]
    for i, (ex_task, gold) in enumerate(examples, start=1):
        snippet = ex_task.text[:_ARTICLE_SNIPPET_CHARS].rstrip()
        if len(ex_task.text) > _ARTICLE_SNIPPET_CHARS:
            snippet += "..."
        output_json = json.dumps(
            {
                "person": ex_task.pair.person_value,
                "place": ex_task.pair.place_value,
                "at_explanation": gold["at_explanation"],
                "at": gold["at"],
                "isAt_explanation": gold["isAt_explanation"],
                "isAt": gold["isAt"],
            },
            ensure_ascii=False,
        )
        lines += [
            f"--- Example {i} ---",
            f"Language: {ex_task.language}  |  Date: {ex_task.publication_date or 'unknown'}",
            f"Article excerpt: {snippet}",
            f"Person: {ex_task.pair.person_value}  "
            f"(mentions: {_format_mentions(ex_task.pair.pers_mentions_list)})",
            f"Place: {ex_task.pair.place_value}  "
            f"(mentions: {_format_mentions(ex_task.pair.loc_mentions_list)})",
            f"Output: {output_json}",
            "",
        ]
    lines.append("Now classify the following new pair:")
    return "\n".join(lines)


def _build_prompt(
    task: PairTask,
    template: str,
    examples: list[tuple[PairTask, dict]] | None,
) -> str:
    if not examples:
        return build_prompt(task, template)
    few_shot_block = _format_few_shot_block(examples)
    pair_ctx = _build_pair_context(task)
    return template.format(
        language=task.language,
        publication_date=task.publication_date or "unknown",
        person_id=task.pair.pers_entity_id,
        person_qid=task.pair.pers_wikidata_qid or "null",
        person_mentions=_format_mentions(task.pair.pers_mentions_list),
        person_value=task.pair.person_value,
        place_id=task.pair.loc_entity_id,
        place_qid=task.pair.loc_wikidata_qid or "null",
        place_mentions=_format_mentions(task.pair.loc_mentions_list),
        place_value=task.pair.place_value,
        article_text=task.text,
        pair_context=f"{few_shot_block}\n{pair_ctx}",
    )


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def _predict_pair(
    task: PairTask,
    *,
    runner: HuggingFaceRunner,
    prompt_template: str,
    examples: list[tuple[PairTask, dict]] | None = None,
) -> tuple[object, dict]:
    prompt = _build_prompt(task, prompt_template, examples)
    result = runner.generate(prompt)
    used_default = False
    error: str | None = None
    try:
        parsed = parse_model_response(result.text)
        validated = validate_prediction(task, parsed)
    except Exception as exc:
        used_default = True
        error = str(exc)
        validated = conservative_default_prediction(str(exc))

    updated_pair = apply_prediction_to_pair(task.pair, validated)
    return updated_pair, {
        "document_id": task.document_id,
        "pers_entity_id": task.pair.pers_entity_id,
        "loc_entity_id": task.pair.loc_entity_id,
        "raw_output": result.text,
        "at": validated.at,
        "isAt": validated.is_at,
        "used_default": used_default,
        "error": error,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "elapsed_seconds": result.elapsed_seconds,
        "n_shots": len(examples) if examples else 0,
    }


def run_documents(
    documents: list[HipeDocument],
    *,
    runner: HuggingFaceRunner,
    prompt_template: str,
    sampler: FewShotSampler | None = None,
    max_pairs: int | None = None,
    max_workers: int = 1,
    batch_size: int = 1,
) -> tuple[list[HipeDocument], list[dict]]:
    """Predict all sampled pairs.

    batch_size > 1: groups pairs into batches for a single model.generate call,
    which is the primary way to improve GPU utilisation.
    max_workers is kept for prompt-preparation parallelism but GPU calls are
    always serialized by the runner lock.
    """
    example_cache: dict[str, list] = {}
    doc_task_examples: list[tuple[HipeDocument, list, list | None]] = []
    for doc in documents:
        tasks = build_pair_tasks(doc)
        if sampler is not None:
            if doc.language not in example_cache:
                example_cache[doc.language] = sampler.sample(doc.language) or None
            examples = example_cache[doc.language]
        else:
            examples = None
        doc_task_examples.append((doc, tasks, examples))

    flat: list[tuple[int, int, HipeDocument, object, list | None]] = []
    for doc_idx, (doc, tasks, examples) in enumerate(doc_task_examples):
        for task_idx, task in enumerate(tasks):
            flat.append((doc_idx, task_idx, doc, task, examples))

    total_api = min(len(flat), max_pairs) if max_pairs is not None else len(flat)
    api_flat      = flat[:total_api]
    fallback_flat = flat[total_api:]

    total_docs = len(documents)
    pair_results: dict[tuple[int, int], object] = {}
    all_traces: list[dict] = []

    # Process in batches; each batch is one model.generate call
    for batch_start in range(0, len(api_flat), batch_size):
        batch = api_flat[batch_start: batch_start + batch_size]

        prompts = [
            _build_prompt(task, prompt_template, examples)
            for _, _, _, task, examples in batch
        ]

        try:
            gen_results = runner.generate_batch(prompts)
        except Exception as exc:
            LOGGER.error("generate_batch failed for batch starting at %d: %s", batch_start, exc)
            gen_results = [
                _GenerationResult(text="", prompt_tokens=None, completion_tokens=None, elapsed_seconds=0.0)
                for _ in batch
            ]

        for (doc_idx, task_idx, doc, task, examples), result in zip(batch, gen_results):
            used_default = False
            error: str | None = None
            try:
                parsed = parse_model_response(result.text)
                validated = validate_prediction(task, parsed)
            except Exception as exc:
                used_default = True
                error = str(exc)
                validated = conservative_default_prediction(str(exc))

            updated_pair = apply_prediction_to_pair(task.pair, validated)
            pair_results[(doc_idx, task_idx)] = updated_pair
            trace = {
                "document_id": task.document_id,
                "pers_entity_id": task.pair.pers_entity_id,
                "loc_entity_id": task.pair.loc_entity_id,
                "raw_output": result.text,
                "at": validated.at,
                "isAt": validated.is_at,
                "used_default": used_default,
                "error": error,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "elapsed_seconds": result.elapsed_seconds,
                "n_shots": len(examples) if examples else 0,
            }
            all_traces.append(trace)
            LOGGER.info(
                "[%d/%d] %s  pair %d  at=%-8s isAt=%-5s  fallback=%s  shots=%d  %.1fs",
                doc_idx + 1, total_docs, doc.document_id, task_idx + 1,
                trace["at"], trace["isAt"], trace["used_default"], trace["n_shots"],
                trace["elapsed_seconds"],
            )

    for doc_idx, task_idx, doc, task, _ in fallback_flat:
        pair_results[(doc_idx, task_idx)] = apply_prediction_to_pair(
            task.pair, conservative_default_prediction("max_pairs limit reached"),
        )

    output_docs: list[HipeDocument] = []
    for doc_idx, (doc, tasks, _) in enumerate(doc_task_examples):
        predicted_pairs = [pair_results[(doc_idx, i)] for i in range(len(tasks))]
        output_docs.append(HipeDocument(
            document_id=doc.document_id,
            media=doc.media,
            source=doc.source,
            date=doc.date,
            language=doc.language,
            text=doc.text,
            sampled_pairs=predicted_pairs,
        ))

    return output_docs, all_traces


# ---------------------------------------------------------------------------
# Scorer  (identical to predict_openrouter.py)
# ---------------------------------------------------------------------------
def score_predictions(gold_file: Path, pred_file: Path) -> str:
    env = {**os.environ, "PYTHONPATH": str(HIPE_DATA_DIR / "scripts")}
    proc = subprocess.run(
        [
            sys.executable,
            str(HIPE_DATA_DIR / "scripts" / "file_scorer_evaluation.py"),
            "--schema_file", str(HIPE_DATA_DIR / "schemas" / "hipe-2026-data.schema.json"),
            "--gold_data_file", str(gold_file),
            "--predictions_file", str(pred_file),
        ],
        capture_output=True, text=True, env=env,
    )
    out = proc.stdout
    if proc.returncode != 0:
        out += f"\n[scorer exited {proc.returncode}]\n{proc.stderr}"
    return out


def _parse_metric(score_text: str, target: str, metric: str) -> str:
    match = re.search(
        rf"'{re.escape(target)}'.*?{re.escape(metric)}=([0-9.]+)", score_text
    )
    return match.group(1) if match else "n/a"


def _write_traces(traces: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")


def _output(text: str, log_fh=None) -> None:
    print(text)
    if log_fh is not None:
        log_fh.write(text + "\n")
        log_fh.flush()


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison_table(
    model_key: str,
    lang: str,
    scores: dict[tuple[str, int], str],
    log_fh=None,
) -> None:
    keys = sorted(scores.keys())
    col_w = 13
    col_labels = [f"{pl}/{n}shot" for pl, n in keys]

    _output(f"\n{'='*76}", log_fh)
    _output(f"  COMPARISON  model={model_key}  dev-lang={lang}", log_fh)
    _output(f"{'='*76}", log_fh)
    header = f"  {'Metric':<26}" + "".join(f"  {lbl:>{col_w}}" for lbl in col_labels)
    _output(header, log_fh)
    _output("  " + "-" * (26 + (col_w + 2) * len(keys)), log_fh)

    for target in ("at", "isAt", "global"):
        for metric in ("macro_recall", "accuracy"):
            if target == "global" and metric == "accuracy":
                continue
            row = f"  {target}.{metric:<22}"
            for key in keys:
                row += f"  {_parse_metric(scores[key], target, metric):>{col_w}}"
            _output(row, log_fh)
    _output("", log_fh)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
_SUMMARY_METRICS = [
    ("at",     "macro_recall"),
    ("at",     "accuracy"),
    ("isAt",   "macro_recall"),
    ("isAt",   "accuracy"),
    ("global", "macro_recall"),
]
_SUMMARY_COL_HEADERS = ["at.rec", "at.acc", "isAt.rec", "isAt.acc", "glb.rec"]


def print_summary_table(
    summary_scores: dict[tuple, str],
    log_fh=None,
    timestamp: str = "",
) -> None:
    if not summary_scores:
        return
    title = "SUMMARY — HIPE-2026 BIU NLP (HuggingFace)" + (f"  ({timestamp})" if timestamp else "")
    col_w = 9
    header = (
        f"  {'model':<13} {'lang':<5} {'prompt':<8} {'shots':<6}"
        + "".join(f"  {h:>{col_w}}" for h in _SUMMARY_COL_HEADERS)
    )
    sep = "  " + "-" * (len(header) - 2)

    _output(f"\n{'='*len(header)}", log_fh)
    _output(title, log_fh)
    _output("=" * len(header), log_fh)
    _output(header, log_fh)
    _output(sep, log_fh)

    prev_model = None
    for key in sorted(summary_scores.keys()):
        model_key, lang, prompt_lang, n_shots = key
        if prev_model and prev_model != model_key:
            _output(sep, log_fh)
        prev_model = model_key
        score_text = summary_scores[key]
        vals = [_parse_metric(score_text, t, m) for t, m in _SUMMARY_METRICS]
        row = (
            f"  {model_key:<13} {lang:<5} {prompt_lang:<8} {n_shots:<6}"
            + "".join(f"  {v:>{col_w}}" for v in vals)
        )
        _output(row, log_fh)
    _output("=" * len(header) + "\n", log_fh)


# ---------------------------------------------------------------------------
# Eval mode
# ---------------------------------------------------------------------------
def _run_one_lang_cell(
    lang: str,
    gold_file: Path,
    prompt_lang: str,
    n_shots: int,
    model_key: str,
    runner: HuggingFaceRunner,
    loaded_templates: dict[str, str],
    output_dir: Path,
    max_pairs: int | None,
    max_workers: int,
    batch_size: int,
) -> tuple[str, str]:
    template = loaded_templates["en"] if prompt_lang == "en" else loaded_templates[lang]

    # For English data, "native" prompt resolves to the same template as "en" — skip re-running
    if prompt_lang == "native" and lang == "en":
        en_score_file = output_dir / model_key / "en" / f"{n_shots}shot" / "dev-en-scores.txt"
        if en_score_file.exists():
            LOGGER.info("SKIP native/dev-en (template identical to en/dev-en) — reusing en result")
            return lang, en_score_file.read_text(encoding="utf-8")

    cell_dir = output_dir / model_key / prompt_lang / f"{n_shots}shot"
    cell_dir.mkdir(parents=True, exist_ok=True)

    pred_file  = cell_dir / f"dev-{lang}.jsonl"
    score_file = cell_dir / f"dev-{lang}-scores.txt"

    if pred_file.exists() and score_file.exists():
        LOGGER.info("SKIP (cached): %s / %s / %dshot / dev-%s", model_key, prompt_lang, n_shots, lang)
        return lang, score_file.read_text(encoding="utf-8")

    LOGGER.info("RUN: dev-%s  prompt=%s  shots=%d  model=%s  batch=%d", lang, prompt_lang, n_shots, runner.model_id, batch_size)
    sampler = FewShotSampler(n_shots=n_shots) if n_shots > 0 else None
    docs = load_jsonl(gold_file)
    predicted_docs, traces = run_documents(
        docs,
        runner=runner,
        prompt_template=template,
        sampler=sampler,
        max_pairs=max_pairs,
        max_workers=max_workers,
        batch_size=batch_size,
    )

    write_jsonl(predicted_docs, pred_file)
    _write_traces(traces, cell_dir / f"dev-{lang}-traces.jsonl")

    score_output = score_predictions(gold_file, pred_file)
    score_file.write_text(score_output, encoding="utf-8")
    return lang, score_output


def run_eval(
    model_key: str,
    runner: HuggingFaceRunner,
    loaded_templates: dict[str, str],
    prompt_lang_keys: list[str],
    output_dir: Path,
    shots_list: list[int],
    max_pairs: int | None,
    summary_scores: dict,
    log_fh=None,
    max_workers: int = 1,
    batch_size: int = 1,
) -> None:
    LOGGER.info("=== EVAL  model=%s  prompt_langs=%s  shots=%s ===", model_key, prompt_lang_keys, shots_list)
    lang_cell_scores: dict[str, dict[tuple, str]] = {lang: {} for lang in SANDBOX_DEV}

    for prompt_lang in prompt_lang_keys:
        for n_shots in shots_list:
            # Dev languages run sequentially to avoid loading pressure on GPU
            for lang, gold_file in SANDBOX_DEV.items():
                lang, score_output = _run_one_lang_cell(
                    lang, gold_file, prompt_lang, n_shots,
                    model_key, runner, loaded_templates, output_dir,
                    max_pairs, max_workers, batch_size,
                )
                lang_cell_scores[lang][(prompt_lang, n_shots)] = score_output
                summary_scores[(model_key, lang, prompt_lang, n_shots)] = score_output

                _output(f"\n{'='*64}", log_fh)
                _output(f"  MODEL       : {runner.model_id}", log_fh)
                _output(f"  DEV LANG    : {lang}", log_fh)
                _output(f"  PROMPT LANG : {prompt_lang}", log_fh)
                _output(f"  SHOTS       : {n_shots}", log_fh)
                _output(f"{'='*64}", log_fh)
                _output(score_output, log_fh)

    for lang in SANDBOX_DEV:
        cell_scores = lang_cell_scores[lang]
        if len(cell_scores) > 1:
            print_comparison_table(model_key, lang, cell_scores, log_fh)
            cmp_path = output_dir / model_key / f"comparison-{lang}.txt"
            with cmp_path.open("w", encoding="utf-8") as fh:
                for (pl, n), text in sorted(cell_scores.items()):
                    fh.write(f"\n{'='*64}\nprompt={pl}  shots={n}\n{'='*64}\n{text}\n")
            LOGGER.info("Comparison saved: %s", cmp_path)


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------
def run_test(
    model_key: str,
    runner: HuggingFaceRunner,
    loaded_templates: dict[str, str],
    prompt_lang_keys: list[str],
    output_dir: Path,
    shots_list: list[int],
    batch_size: int = 1,
) -> None:
    LOGGER.info("=== TEST  model=%s  prompt_langs=%s  shots=%s ===", model_key, prompt_lang_keys, shots_list)

    for prompt_lang in prompt_lang_keys:
        for n_shots in shots_list:
            cell_dir = output_dir / model_key / prompt_lang / f"{n_shots}shot"
            cell_dir.mkdir(parents=True, exist_ok=True)
            sampler = FewShotSampler(n_shots=n_shots) if n_shots > 0 else None

            for test_file in TEST_FILES:
                stem = test_file.stem
                file_lang = stem[-2:]
                template = (
                    loaded_templates["en"]
                    if prompt_lang == "en"
                    else loaded_templates.get(file_lang, loaded_templates["en"])
                )
                submission_name = f"BIU-{model_key}-{prompt_lang}-{n_shots}shot_{stem}_run1.jsonl"
                pred_file = cell_dir / submission_name
                LOGGER.info("%s  prompt=%s  shots=%d  →  %s", test_file.name, prompt_lang, n_shots, submission_name)
                docs = load_jsonl(test_file)
                predicted_docs, traces = run_documents(
                    docs, runner=runner, prompt_template=template, sampler=sampler,
                    batch_size=batch_size,
                )
                write_jsonl(predicted_docs, pred_file)
                _write_traces(traces, cell_dir / f"test-{stem}-traces.jsonl")

            print(f"\nSubmission files — {model_key} / {prompt_lang} / {n_shots}-shot:")
            for p in sorted(cell_dir.glob("BIU-*.jsonl")):
                print(f"  {p}")


# ---------------------------------------------------------------------------
# Parallel GPU-group orchestration
# ---------------------------------------------------------------------------
def _collect_summary_from_disk(eval_dir: Path) -> dict[tuple, str]:
    """Walk eval_dir and reconstruct summary_scores from written score files.

    Path pattern: eval_dir/{model_key}/{prompt_lang}/{n}shot/dev-{lang}-scores.txt
    """
    summary: dict[tuple, str] = {}
    for score_file in sorted(eval_dir.rglob("*-scores.txt")):
        parts = score_file.relative_to(eval_dir).parts
        if len(parts) != 4:
            continue
        model_key, prompt_lang, shot_dir, filename = parts
        try:
            n_shots = int(shot_dir.replace("shot", ""))
        except ValueError:
            continue
        lang = filename.replace("dev-", "").replace("-scores.txt", "")
        summary[(model_key, lang, prompt_lang, n_shots)] = score_file.read_text(encoding="utf-8")
    return summary


def _build_worker_argv(args: argparse.Namespace, model_key: str, output_dir: Path) -> list[str]:
    """Reconstruct CLI args for a single-model worker subprocess."""
    argv = [
        "--models", model_key,
        "--output-dir", str(output_dir),
        "--_worker",
        "--mode", args.mode,
        "--dtype", args.dtype,
        "--device-map", args.device_map,
        "--temperature", str(args.temperature),
        "--max-tokens", str(args.max_tokens),
        "--workers", str(args.workers),
        "--batch-size", str(args.batch_size),
        "--shots", *[str(s) for s in args.shots],
        "--prompt-langs", *args.prompt_langs,
    ]
    if args.max_pairs is not None:
        argv += ["--max-pairs", str(args.max_pairs)]
    if args.load_in_4bit:
        argv.append("--load-in-4bit")
    if args.load_in_8bit:
        argv.append("--load-in-8bit")
    if args.flash_attn:
        argv.append("--flash-attn")
    if args.model_ids:
        argv += ["--model-ids", *args.model_ids]
    return argv


def _orchestrate_parallel(
    args: argparse.Namespace,
    selected: list[str],
    gpu_groups: list[str],
    output_dir: Path,
    log_fh,
    timestamp: str,
) -> None:
    """Spawn one subprocess per model, distributed round-robin across GPU groups."""
    import subprocess as sp

    _output(f"\nParallel mode: {len(selected)} model(s) across {len(gpu_groups)} GPU group(s)", log_fh)
    for i, (model_key, gpus) in enumerate(
        zip(selected, [gpu_groups[i % len(gpu_groups)] for i in range(len(selected))])
    ):
        _output(f"  {model_key:15} → GPUs {gpus}", log_fh)

    procs: list[tuple[str, str, sp.Popen]] = []
    for i, model_key in enumerate(selected):
        gpus = gpu_groups[i % len(gpu_groups)]
        worker_log = output_dir / f"worker_{model_key}.log"
        argv = _build_worker_argv(args, model_key, output_dir)
        cmd = [sys.executable, str(Path(__file__).resolve())] + argv
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus}
        LOGGER.info("Spawning worker: model=%s  GPUs=%s  log=%s", model_key, gpus, worker_log)
        fh = worker_log.open("w", encoding="utf-8")
        p = sp.Popen(cmd, env=env, stdout=fh, stderr=fh)
        procs.append((model_key, gpus, p, fh))

    # Wait for all workers; report exit status
    all_ok = True
    for model_key, gpus, p, fh in procs:
        p.wait()
        fh.close()
        if p.returncode == 0:
            LOGGER.info("Worker finished OK: model=%s  GPUs=%s", model_key, gpus)
            _output(f"  [OK]   {model_key} (GPUs {gpus})", log_fh)
        else:
            LOGGER.error("Worker FAILED: model=%s  GPUs=%s  exit=%d", model_key, gpus, p.returncode)
            _output(f"  [FAIL] {model_key} (GPUs {gpus}) — exit {p.returncode}  see worker_{model_key}.log", log_fh)
            all_ok = False

    # Collect results written by workers and print combined summary
    eval_dir = output_dir / "eval"
    if eval_dir.exists():
        summary_scores = _collect_summary_from_disk(eval_dir)
        if summary_scores:
            print_summary_table(summary_scores, log_fh=log_fh, timestamp=timestamp)
            summary_path = output_dir / "summary.txt"
            with summary_path.open("w", encoding="utf-8") as sf:
                print_summary_table(summary_scores, log_fh=sf, timestamp=timestamp)
            LOGGER.info("Summary written: %s", summary_path)

    if not all_ok:
        LOGGER.error("One or more workers failed — check worker_*.log files in %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HIPE-2026 HuggingFace prediction — BIU NLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Default models (HuggingFace IDs):\n"
            + "\n".join(f"  {k:<14} → {v}" for k, v in DEFAULT_MODELS.items())
        ),
    )
    p.add_argument(
        "--models", nargs="+", default=["all"], metavar="MODEL",
        help=f"Model keys: {list(DEFAULT_MODELS.keys())} or 'all'. Default: all.",
    )
    p.add_argument(
        "--model-ids", nargs="+", default=[], metavar="KEY=HF_ID",
        help="Override or add model IDs, e.g. --model-ids gemma4_31b=google/gemma-4-31b-it",
    )
    p.add_argument(
        "--mode", choices=["eval", "test", "all"], default="all",
        help="eval=dev scoring, test=submission files, all=both. Default: all.",
    )
    p.add_argument(
        "--prompt-langs", nargs="+", choices=["en", "native", "all"],
        default=["all"], metavar="LANG",
        help="Prompt language(s): 'en', 'native', or 'all' (both). Default: all.",
    )
    p.add_argument(
        "--shots", nargs="+", type=int, default=[0, 5, 10], metavar="N",
        help="Few-shot counts to run. Default: 0 5 10.",
    )
    p.add_argument(
        "--max-pairs", type=int, default=None,
        help="Cap inference calls per cell. Remaining pairs get FALSE/FALSE fallback.",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers per cell. Default: 1 (GPU inference is serialized).",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of pairs per model.generate call. Default: 32.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument(
        "--device-map", default="auto",
        help="HuggingFace device_map. Default: 'auto' (spreads across available GPUs/CPU).",
    )
    p.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
        help="Model weight dtype. Default: bfloat16.",
    )
    p.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load model in 4-bit quantization via bitsandbytes (reduces VRAM ~4x).",
    )
    p.add_argument(
        "--load-in-8bit", action="store_true",
        help="Load model in 8-bit quantization via bitsandbytes (reduces VRAM ~2x).",
    )
    p.add_argument(
        "--output-dir", default=str(THIS_DIR / "results_hf"),
        help="Base results directory. Each run creates a timestamped sub-dir. Default: ./results_hf/",
    )
    p.add_argument(
        "--flash-attn", action="store_true",
        help="Enable Flash Attention 2 (requires: pip install flash-attn). Falls back silently if not installed.",
    )
    p.add_argument(
        "--gpu-groups", nargs="+", default=None, metavar="GPUS",
        help=(
            "Run models in parallel, one subprocess per GPU group. "
            "Each group is a comma-separated list of GPU indices. "
            "Models are assigned round-robin. "
            "Example: --gpu-groups 5,6,7 1,2,4  (2 models in parallel)"
        ),
    )
    # Internal flag used by worker subprocesses — not shown in help
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return p


def main() -> None:
    from datetime import datetime

    parser = _build_parser()
    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        parser.error("--load-in-4bit and --load-in-8bit are mutually exclusive.")

    models: dict[str, str] = dict(DEFAULT_MODELS)
    for pair in args.model_ids:
        if "=" not in pair:
            parser.error(f"--model-ids entries must be KEY=HF_ID, got: {pair!r}")
        k, v = pair.split("=", 1)
        models[k] = v

    selected: list[str] = list(models.keys()) if "all" in args.models else args.models
    for key in selected:
        if key not in models:
            parser.error(f"Unknown model '{key}'. Available: {list(models.keys())}")

    prompt_lang_keys: list[str] = (
        ["en", "native"] if "all" in args.prompt_langs else args.prompt_langs
    )
    shots_list: list[int] = sorted(set(args.shots))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Workers receive the already-timestamped dir from the orchestrator — don't nest further
    if args._worker:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── logging ────────────────────────────────────────────────────────────
    log_path = output_dir / "run.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)
    log_fh = log_path.open("a", encoding="utf-8")
    LOGGER.info("Log file: %s", log_path)

    # ── config ─────────────────────────────────────────────────────────────
    loaded_templates: dict[str, str] = {}
    for lang, path in LANG_PROMPT_FILES.items():
        loaded_templates[lang] = path.read_text(encoding="utf-8")

    LOGGER.info("Models       : %s", {k: models[k] for k in selected})
    LOGGER.info("Mode         : %s", args.mode)
    LOGGER.info("Prompt langs : %s", prompt_lang_keys)
    LOGGER.info("Shots        : %s", shots_list)
    LOGGER.info("Max pairs    : %s", args.max_pairs or "unlimited")
    LOGGER.info("Device map   : %s", args.device_map)
    LOGGER.info("Dtype        : %s  4bit=%s  8bit=%s", args.dtype, args.load_in_4bit, args.load_in_8bit)
    LOGGER.info("Output       : %s", output_dir)

    # ── parallel mode: orchestrate subprocesses, one per GPU group ────────
    gpu_groups = args.gpu_groups
    if gpu_groups and len(gpu_groups) > 1 and len(selected) > 1 and not args._worker:
        _orchestrate_parallel(args, selected, gpu_groups, output_dir, log_fh, timestamp)
        log_fh.close()
        return

    # ── sequential mode (default, or inside a worker subprocess) ──────────
    summary_scores: dict[tuple, str] = {}
    failed_models: list[str] = []

    for model_key in selected:
        LOGGER.info("=== LOADING %s (%s) ===", model_key, models[model_key])
        try:
            runner = HuggingFaceRunner(
                model_id=models[model_key],
                device_map=args.device_map,
                torch_dtype=args.dtype,
                temperature=args.temperature,
                max_new_tokens=args.max_tokens,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                flash_attn=args.flash_attn,
            )
        except Exception as exc:
            LOGGER.error("FAILED to load model '%s' (%s): %s", model_key, models[model_key], exc, exc_info=True)
            failed_models.append(model_key)
            continue

        try:
            if args.mode in ("eval", "all"):
                run_eval(
                    model_key=model_key,
                    runner=runner,
                    loaded_templates=loaded_templates,
                    prompt_lang_keys=prompt_lang_keys,
                    output_dir=output_dir / "eval",
                    shots_list=shots_list,
                    max_pairs=args.max_pairs,
                    summary_scores=summary_scores,
                    log_fh=log_fh,
                    max_workers=args.workers,
                    batch_size=args.batch_size,
                )

            if args.mode in ("test", "all"):
                run_test(
                    model_key=model_key,
                    runner=runner,
                    loaded_templates=loaded_templates,
                    prompt_lang_keys=prompt_lang_keys,
                    output_dir=output_dir / "test",
                    shots_list=shots_list,
                    batch_size=args.batch_size,
                )
        except Exception as exc:
            LOGGER.error("FAILED during eval/test for '%s': %s", model_key, exc, exc_info=True)
            failed_models.append(model_key)
        finally:
            # Always unload model to free VRAM before the next one
            del runner

    # ── final summary ──────────────────────────────────────────────────────
    if summary_scores:
        print_summary_table(summary_scores, log_fh=log_fh, timestamp=timestamp)
        summary_path = output_dir / "summary.txt"
        with summary_path.open("w", encoding="utf-8") as sf:
            print_summary_table(summary_scores, log_fh=sf, timestamp=timestamp)
        LOGGER.info("Summary written: %s", summary_path)

    if failed_models:
        LOGGER.error("MODELS THAT FAILED: %s", failed_models)

    log_fh.close()
    LOGGER.info("All done. Log: %s", log_path)


if __name__ == "__main__":
    main()