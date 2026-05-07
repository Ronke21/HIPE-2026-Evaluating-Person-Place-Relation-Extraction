#!/usr/bin/env python3
"""HIPE-2026 OpenRouter prediction script — BIU NLP.

Classifies person-place relations (at / isAt) using strong LLMs via OpenRouter.
Compares zero-shot vs few-shot prompting AND English vs native-language prompts.

Full evaluation matrix (per model, per dev language):
  prompt_lang  ×  n_shots  →  predictions + official scorer output
  ─────────────────────────────────────────────────────────────────
  en           ×  0-shot
  en           ×  N-shot   (N drawn from --shots, e.g. 3)
  native       ×  0-shot
  native       ×  N-shot

Models:
  nemotron  → nvidia/nemotron-3-super-120b-a12b:free
  gptoss    → openai/gpt-oss-120b:free
  minimax   → minimax/minimax-m2.5:free
  gemma     → google/gemma-4-31b-it:free
  glm       → z-ai/glm-4.5-air:free

Quick start:
  export OPENROUTER_API_KEY=sk-or-...
  pip install openai

  # full matrix, smoke test (20 pairs per cell):
  python predict_openrouter.py --mode eval --shots 0 3 --max-pairs 20

  # full matrix, all pairs:
  python predict_openrouter.py --mode eval --shots 0 3

  # test submissions with a specific config:
  python predict_openrouter.py --mode test --shots 3 --prompt-langs native --models nemotron

Output layout:
  results/
    eval/{model}/{prompt_lang}/{n}shot/dev-{lang}.jsonl        predictions
    eval/{model}/{prompt_lang}/{n}shot/dev-{lang}-scores.txt   scorer output
    eval/{model}/{prompt_lang}/{n}shot/dev-{lang}-traces.jsonl raw traces
    eval/{model}/comparison-{lang}.txt                         full matrix side-by-side
    test/{model}/{prompt_lang}/{n}shot/BIU-{model}-{pl}-{n}shot_{stem}_run1.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
# Repository paths
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

# Infer language from test file stem (last two chars before .jsonl)
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
# Model registry
# ---------------------------------------------------------------------------
MODELS: dict[str, str] = {
    # "nemotron": "nvidia/nemotron-3-super-120b-a12b:free",  # disabled: poor JSON compliance
    "gptoss":   "openai/gpt-oss-120b:free",
    "minimax":  "minimax/minimax-m2.5:free",
    "gemma":    "google/gemma-4-31b-it:free",
    "glm":      "z-ai/glm-4.5-air:free",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenRouter runner
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _GenerationResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    elapsed_seconds: float


class OpenRouterRunner:
    """Sends prompts to an OpenRouter-hosted model via the OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        request_delay: float = 1.0,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package: pip install openai") from exc

        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._request_delay = request_delay
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            max_retries=max_retries,
        )

    def generate(self, prompt: str) -> _GenerationResult:
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        elapsed = time.perf_counter() - started
        if not response.choices:
            # Some models return an empty choices list on error even with HTTP 200.
            finish = getattr(response, "error", None) or "no choices returned"
            LOGGER.warning("Empty choices from %s: %s", self.model, finish)
            text = ""
        else:
            text = response.choices[0].message.content or ""
            finish = response.choices[0].finish_reason
            if finish == "length":
                LOGGER.warning("Response truncated (finish_reason=length) — raise --max-tokens")
        usage = response.usage
        if self._request_delay > 0:
            time.sleep(self._request_delay)
        return _GenerationResult(
            text=text,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Few-shot sampler
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
        """Return up to n_shots examples balanced across label combinations."""
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
# Prompt construction
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
    """Build the final prompt, injecting few-shot examples when provided."""
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
    runner: OpenRouterRunner,
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
    runner: OpenRouterRunner,
    prompt_template: str,
    sampler: FewShotSampler | None = None,
    max_pairs: int | None = None,
    max_workers: int = 8,
) -> tuple[list[HipeDocument], list[dict]]:
    """Predict all sampled pairs in parallel.  prompt_template is fixed for this run."""
    # Build per-doc task lists and resolve few-shot examples up front
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

    # Flatten to indexed work items
    flat: list[tuple[int, int, HipeDocument, object, list | None]] = []
    for doc_idx, (doc, tasks, examples) in enumerate(doc_task_examples):
        for task_idx, task in enumerate(tasks):
            flat.append((doc_idx, task_idx, doc, task, examples))

    total_api = min(len(flat), max_pairs) if max_pairs is not None else len(flat)
    api_flat    = flat[:total_api]
    fallback_flat = flat[total_api:]

    total_docs = len(documents)
    pair_results: dict[tuple[int, int], object] = {}
    all_traces: list[dict] = []
    traces_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _predict_pair, task,
                runner=runner, prompt_template=prompt_template, examples=examples,
            ): (doc_idx, task_idx, doc)
            for doc_idx, task_idx, doc, task, examples in api_flat
        }
        for future in as_completed(future_map):
            doc_idx, task_idx, doc = future_map[future]
            updated_pair, trace = future.result()
            pair_results[(doc_idx, task_idx)] = updated_pair
            with traces_lock:
                all_traces.append(trace)
            LOGGER.info(
                "[%d/%d] %s  pair %d  at=%-8s isAt=%-5s  fallback=%s  shots=%d",
                doc_idx + 1, total_docs, doc.document_id, task_idx + 1,
                trace["at"], trace["isAt"], trace["used_default"], trace["n_shots"],
            )

    for doc_idx, task_idx, doc, task, _ in fallback_flat:
        pair_results[(doc_idx, task_idx)] = apply_prediction_to_pair(
            task.pair, conservative_default_prediction("max_pairs limit reached"),
        )

    # Reconstruct documents in original order
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
# Scorer
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
    """Print to stdout and mirror to the log file when provided."""
    print(text)
    if log_fh is not None:
        log_fh.write(text + "\n")
        log_fh.flush()


# ---------------------------------------------------------------------------
# Comparison table  (key = (prompt_lang, n_shots))
# ---------------------------------------------------------------------------
def print_comparison_table(
    model_key: str,
    lang: str,
    scores: dict[tuple[str, int], str],
    log_fh=None,
) -> None:
    """Print all (prompt_lang × n_shots) metric combinations side by side."""
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
# Summary table  (all models × langs × prompt_langs × shots in one view)
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
    """Print a single table covering every (model, lang, prompt_lang, n_shots) cell."""
    if not summary_scores:
        return

    title = f"SUMMARY — HIPE-2026 BIU NLP" + (f"  ({timestamp})" if timestamp else "")
    col_w = 9
    header = (
        f"  {'model':<11} {'lang':<5} {'prompt':<8} {'shots':<6}"
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
        vals = [
            _parse_metric(score_text, target, metric)
            for target, metric in _SUMMARY_METRICS
        ]
        row = (
            f"  {model_key:<11} {lang:<5} {prompt_lang:<8} {n_shots:<6}"
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
    runner: OpenRouterRunner,
    loaded_templates: dict[str, str],
    output_dir: Path,
    max_pairs: int | None,
    max_workers: int,
) -> tuple[str, str]:
    """Run (or resume) one (lang, prompt_lang, n_shots) cell. Returns (lang, score_text)."""
    template = loaded_templates["en"] if prompt_lang == "en" else loaded_templates[lang]
    cell_dir = output_dir / model_key / prompt_lang / f"{n_shots}shot"
    cell_dir.mkdir(parents=True, exist_ok=True)

    pred_file  = cell_dir / f"dev-{lang}.jsonl"
    score_file = cell_dir / f"dev-{lang}-scores.txt"

    # Resume: skip cell entirely if both output files already exist
    if pred_file.exists() and score_file.exists():
        LOGGER.info(
            "SKIP (cached): %s / %s / %dshot / dev-%s",
            model_key, prompt_lang, n_shots, lang,
        )
        return lang, score_file.read_text(encoding="utf-8")

    LOGGER.info(
        "RUN: dev-%s  prompt=%s  shots=%d  model=%s",
        lang, prompt_lang, n_shots, runner.model,
    )
    sampler = FewShotSampler(n_shots=n_shots) if n_shots > 0 else None
    docs = load_jsonl(gold_file)
    predicted_docs, traces = run_documents(
        docs,
        runner=runner,
        prompt_template=template,
        sampler=sampler,
        max_pairs=max_pairs,
        max_workers=max_workers,
    )

    write_jsonl(predicted_docs, pred_file)
    _write_traces(traces, cell_dir / f"dev-{lang}-traces.jsonl")

    score_output = score_predictions(gold_file, pred_file)
    score_file.write_text(score_output, encoding="utf-8")

    return lang, score_output


def run_eval(
    model_key: str,
    runner: OpenRouterRunner,
    loaded_templates: dict[str, str],
    prompt_lang_keys: list[str],
    output_dir: Path,
    shots_list: list[int],
    max_pairs: int | None,
    summary_scores: dict,
    log_fh=None,
    max_workers: int = 8,
) -> None:
    LOGGER.info(
        "=== EVAL  model=%s  prompt_langs=%s  shots=%s ===",
        model_key, prompt_lang_keys, shots_list,
    )

    # Per-lang comparison scores: lang -> {(prompt_lang, n_shots): score_text}
    lang_cell_scores: dict[str, dict[tuple, str]] = {lang: {} for lang in SANDBOX_DEV}

    for prompt_lang in prompt_lang_keys:
        for n_shots in shots_list:
            # Languages run sequentially — free-tier rate limits can't handle concurrent langs
            for lang, gold_file in SANDBOX_DEV.items():
                _, score_output = _run_one_lang_cell(
                    lang, gold_file, prompt_lang, n_shots,
                    model_key, runner, loaded_templates, output_dir,
                    max_pairs, max_workers,
                )
                lang_cell_scores[lang][(prompt_lang, n_shots)] = score_output
                summary_scores[(model_key, lang, prompt_lang, n_shots)] = score_output

                _output(f"\n{'='*64}", log_fh)
                _output(f"  MODEL       : {runner.model}", log_fh)
                _output(f"  DEV LANG    : {lang}", log_fh)
                _output(f"  PROMPT LANG : {prompt_lang}", log_fh)
                _output(f"  SHOTS       : {n_shots}", log_fh)
                _output(f"{'='*64}", log_fh)
                _output(score_output, log_fh)

    # Per-lang comparison tables
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
    runner: OpenRouterRunner,
    loaded_templates: dict[str, str],
    prompt_lang_keys: list[str],
    output_dir: Path,
    shots_list: list[int],
) -> None:
    LOGGER.info(
        "=== TEST  model=%s  prompt_langs=%s  shots=%s ===",
        model_key, prompt_lang_keys, shots_list,
    )

    for prompt_lang in prompt_lang_keys:
        for n_shots in shots_list:
            cell_dir = output_dir / model_key / prompt_lang / f"{n_shots}shot"
            cell_dir.mkdir(parents=True, exist_ok=True)

            sampler = FewShotSampler(n_shots=n_shots) if n_shots > 0 else None

            for test_file in TEST_FILES:
                stem = test_file.stem   # e.g. HIPE-2026-v1.0-impresso-test-de
                file_lang = stem[-2:]   # "de", "en", "fr"
                template = (
                    loaded_templates["en"]
                    if prompt_lang == "en"
                    else loaded_templates.get(file_lang, loaded_templates["en"])
                )

                submission_name = f"BIU-{model_key}-{prompt_lang}-{n_shots}shot_{stem}_run1.jsonl"
                pred_file = cell_dir / submission_name

                LOGGER.info(
                    "%s  prompt=%s  shots=%d  →  %s",
                    test_file.name, prompt_lang, n_shots, submission_name,
                )
                docs = load_jsonl(test_file)
                predicted_docs, traces = run_documents(
                    docs,
                    runner=runner,
                    prompt_template=template,
                    sampler=sampler,
                )
                write_jsonl(predicted_docs, pred_file)
                _write_traces(traces, cell_dir / f"test-{stem}-traces.jsonl")

            print(f"\nSubmission files — {model_key} / {prompt_lang} / {n_shots}-shot:")
            for p in sorted(cell_dir.glob("BIU-*.jsonl")):
                print(f"  {p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HIPE-2026 OpenRouter prediction — BIU NLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Models: " + ", ".join(f"{k}={v}" for k, v in MODELS.items()),
    )
    p.add_argument(
        "--api-key", default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var).",
    )
    p.add_argument(
        "--models", nargs="+", default=["all"], metavar="MODEL",
        help=f"Model keys: {list(MODELS.keys())} or 'all'. Default: all.",
    )
    p.add_argument(
        "--mode", choices=["eval", "test", "all"], default="all",
        help="eval=dev scoring, test=submission files, all=both. Default: all.",
    )
    p.add_argument(
        "--prompt-langs", nargs="+", choices=["en", "native", "all"],
        default=["all"], metavar="LANG",
        help=(
            "Prompt language(s) to use: "
            "'en' (English prompt for all docs), "
            "'native' (language-matched prompt), "
            "'all' (both, default)."
        ),
    )
    p.add_argument(
        "--shots", nargs="+", type=int, default=[0, 5, 10], metavar="N",
        help="Few-shot counts to run. Pass multiple to compare, e.g. --shots 0 5 10. Default: 0 5 10.",
    )
    p.add_argument(
        "--max-pairs", type=int, default=None,
        help="Cap API calls per cell in eval mode. Remaining pairs get FALSE/FALSE fallback.",
    )
    p.add_argument("--workers", type=int, default=8,
        help="Parallel API workers per dev-language cell. Default: 8.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument(
        "--request-delay", type=float, default=1.0,
        help="Seconds between API calls (free-tier rate limiting). Default: 1.0.",
    )
    p.add_argument(
        "--output-dir", default=str(THIS_DIR / "results"),
        help="Root output directory. Default: ./results/",
    )
    return p


def main() -> None:
    from datetime import datetime

    parser = _build_parser()
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        parser.error(
            "Provide --api-key or set OPENROUTER_API_KEY.\n"
            "Get a key at https://openrouter.ai/keys"
        )

    selected: list[str] = list(MODELS.keys()) if "all" in args.models else args.models
    for key in selected:
        if key not in MODELS:
            parser.error(f"Unknown model '{key}'. Available: {list(MODELS.keys())}")

    prompt_lang_keys: list[str] = (
        ["en", "native"] if "all" in args.prompt_langs else args.prompt_langs
    )
    shots_list: list[int] = sorted(set(args.shots))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── logging setup ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"run_{timestamp}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)
    log_fh = log_path.open("a", encoding="utf-8")   # plain text mirror for print() output
    LOGGER.info("Log file: %s", log_path)

    # ── config ─────────────────────────────────────────────────────────────
    loaded_templates: dict[str, str] = {}
    for lang, path in LANG_PROMPT_FILES.items():
        loaded_templates[lang] = path.read_text(encoding="utf-8")
        LOGGER.info("Prompt [%s]: %s", lang, path)

    LOGGER.info("Models       : %s", selected)
    LOGGER.info("Mode         : %s", args.mode)
    LOGGER.info("Prompt langs : %s", prompt_lang_keys)
    LOGGER.info("Shots        : %s", shots_list)
    LOGGER.info("Max pairs    : %s", args.max_pairs or "unlimited")
    LOGGER.info("Output       : %s", output_dir)

    n_cells = len(selected) * len(prompt_lang_keys) * len(shots_list) * len(SANDBOX_DEV)
    LOGGER.info("Total eval cells: %d", n_cells)

    # ── runs ───────────────────────────────────────────────────────────────
    summary_scores: dict[tuple, str] = {}   # (model, lang, prompt_lang, n_shots) → score text

    for model_key in selected:
        runner = OpenRouterRunner(
            model=MODELS[model_key],
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            request_delay=args.request_delay,
        )

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
            )

        if args.mode in ("test", "all"):
            run_test(
                model_key=model_key,
                runner=runner,
                loaded_templates=loaded_templates,
                prompt_lang_keys=prompt_lang_keys,
                output_dir=output_dir / "test",
                shots_list=shots_list,
            )

    # ── final summary ──────────────────────────────────────────────────────
    if summary_scores:
        print_summary_table(summary_scores, log_fh=log_fh, timestamp=timestamp)
        summary_path = output_dir / f"summary_{timestamp}.txt"
        with summary_path.open("w", encoding="utf-8") as sf:
            print_summary_table(summary_scores, log_fh=sf, timestamp=timestamp)
        LOGGER.info("Summary written: %s", summary_path)

    log_fh.close()
    LOGGER.info("All done. Log: %s", log_path)


if __name__ == "__main__":
    main()
