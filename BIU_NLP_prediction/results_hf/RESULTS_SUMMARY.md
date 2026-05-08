# HIPE 2026 — Results Summary

**Team:** BIU NLP, Bar-Ilan University  |  **Task:** person–place relation qualification (at / isAt)  |  **Updated:** 2026-05-08

---

## 1. Leaderboard — Best Dev Score per Model × Language  (global macro-recall)

| Model | DE | EN | FR |
|---|---|---|---|
| gemma-3-27b-it | **0.5622** (native/0sh) | **0.7209** (en/0sh) | **0.6611** (en/0sh) |
| gemma-4-26B-A4B-it (MoE) | **0.7518** (native/10sh) | **0.7803** (en/10sh) | **0.7481** (en/10sh) |
| Mistral-Small-24B | **0.5917** (native/0sh) | **0.5935** (native/0sh) | **0.5914** (native/0sh) |
| aya-expanse-32b | **0.5950** (en/0sh) | **0.5967** (en/0sh) | **0.6183** (en/0sh) |
| Qwen3-30B-A3B-2507 (MoE) | — | — | — |

*All-FALSE baseline: glb.rec ≈ 0.4167*

---

## 2. Full Eval Grids per Model

### gemma-3-27b-it

**Prompt: en**

| Shots | DE | EN | FR |
|---|---|---|---|
| 0-shot | 0.5299 (at 0.3927 / isAt 0.6671) | 0.7209 (at 0.5890 / isAt 0.8528) | 0.6611 (at 0.5147 / isAt 0.8074) |
| 3-shot | 0.5256 (at 0.3668 / isAt 0.6843) | 0.6662 (at 0.4428 / isAt 0.8895) | 0.5413 (at 0.3952 / isAt 0.6875) |
| 5-shot | 0.4975 (at 0.3919 / isAt 0.6032) | 0.6881 (at 0.4956 / isAt 0.8805) | 0.4747 (at 0.3551 / isAt 0.5944) |

**Prompt: native**

| Shots | DE | EN | FR |
|---|---|---|---|
| 0-shot | 0.5622 (at 0.4919 / isAt 0.6326) | 0.7209 (at 0.5890 / isAt 0.8528) | 0.6528 (at 0.5280 / isAt 0.7777) |
| 3-shot | 0.5393 (at 0.4314 / isAt 0.6472) | 0.6662 (at 0.4428 / isAt 0.8895) | 0.4972 (at 0.3557 / isAt 0.6386) |
| 5-shot | 0.5163 (at 0.4281 / isAt 0.6046) | 0.6881 (at 0.4956 / isAt 0.8805) | 0.4520 (at 0.3511 / isAt 0.5528) |

### gemma-4-26B-A4B-it (MoE)

**Prompt: en**

| Shots | DE | EN | FR |
|---|---|---|---|
| 0-shot | 0.5541 (at 0.4322 / isAt 0.6760) | 0.7347 (at 0.6558 / isAt 0.8137) | 0.5252 (at 0.4327 / isAt 0.6178) |
| 3-shot | 0.7119 (at 0.6238 / isAt 0.8000) | 0.7349 (at 0.6411 / isAt 0.8287) | 0.7218 (at 0.6567 / isAt 0.7869) |
| 5-shot | 0.4789 (at 0.3925 / isAt 0.5652) | 0.7550 (at 0.6534 / isAt 0.8565) | 0.7379 (at 0.6612 / isAt 0.8145) |
| 10-shot | 0.7316 (at 0.6495 / isAt 0.8136) | 0.7803 (at 0.6854 / isAt 0.8753) | 0.7481 (at 0.6725 / isAt 0.8236) |

**Prompt: native**

| Shots | DE | EN | FR |
|---|---|---|---|
| 0-shot | 0.5096 (at 0.4060 / isAt 0.6132) | — | 0.4957 (at 0.4044 / isAt 0.5870) |
| 3-shot | 0.7229 (at 0.6578 / isAt 0.7879) | — | 0.7386 (at 0.6821 / isAt 0.7950) |
| 5-shot | 0.4748 (at 0.3844 / isAt 0.5652) | — | 0.7432 (at 0.6757 / isAt 0.8108) |
| 10-shot | 0.7518 (at 0.6529 / isAt 0.8508) | — | 0.7359 (at 0.6714 / isAt 0.8004) |

### Mistral-Small-24B

**Prompt: en**

| Shots | DE | EN | FR |
|---|---|---|---|
| 0-shot | 0.5389 (at 0.5002 / isAt 0.5775) | 0.5741 (at 0.5168 / isAt 0.6314) | 0.5749 (at 0.5205 / isAt 0.6293) |
| 5-shot | 0.5122 (at 0.4505 / isAt 0.5738) | 0.5653 (at 0.4715 / isAt 0.6591) | 0.5720 (at 0.4969 / isAt 0.6472) |
| 10-shot | 0.5214 (at 0.4396 / isAt 0.6033) | 0.5496 (at 0.4604 / isAt 0.6389) | 0.5360 (at 0.4528 / isAt 0.6193) |

**Prompt: native**

| Shots | DE | EN | FR |
|---|---|---|---|
| 0-shot | 0.5917 (at 0.5875 / isAt 0.5959) | 0.5935 (at 0.5279 / isAt 0.6591) | 0.5914 (at 0.5677 / isAt 0.6151) |
| 5-shot | 0.5595 (at 0.5280 / isAt 0.5909) | 0.5773 (at 0.4715 / isAt 0.6832) | 0.5776 (at 0.5456 / isAt 0.6097) |
| 10-shot | 0.5720 (at 0.5038 / isAt 0.6402) | 0.5586 (at 0.4506 / isAt 0.6667) | 0.5356 (at 0.4861 / isAt 0.5851) |

### aya-expanse-32b

**Prompt: en**

| Shots | DE | EN | FR |
|---|---|---|---|
| 0-shot | 0.5950 (at 0.4862 / isAt 0.7039) | 0.5967 (at 0.4045 / isAt 0.7888) | 0.6183 (at 0.4591 / isAt 0.7775) |
| 3-shot | 0.5680 (at 0.3867 / isAt 0.7493) | 0.5823 (at 0.3922 / isAt 0.7723) | 0.4260 (at 0.3407 / isAt 0.5113) |

### Qwen3-30B-A3B-2507 (MoE)

*No valid eval results.*

---

## 3. Test Predictions — Best per Model

### impresso-test-de

| Rank | Model | Best Dev | Config | File |
|---|---|---|---|---|
| 1 | gemma-4-26B-A4B-it (MoE) | 0.7316 | en/10shot | `BIU-gemma4_26b-en-10shot_HIPE-2026-v1.0-impresso-test-de_run1.jsonl` |
| 2 | Mistral-Small-24B | 0.5917 | native/0shot | `BIU-mistral_small_24b-native-0shot_HIPE-2026-v1.0-impresso-test-de_run1.jsonl` |
| 3 | gemma-3-27b-it | 0.5622 | native/0shot | `BIU-gemma3_27b-native-0shot_HIPE-2026-v1.0-impresso-test-de_run1.jsonl` |
| — | Qwen3-30B-A3B-2507 (MoE) | — | — | *no test file* |
| — | aya-expanse-32b | — | — | *no test file* |

### impresso-test-en

| Rank | Model | Best Dev | Config | File |
|---|---|---|---|---|
| 1 | gemma-4-26B-A4B-it (MoE) | 0.7550 | en/5shot | `BIU-gemma4_26b-en-5shot_HIPE-2026-v1.0-impresso-test-en_run1.jsonl` |
| 2 | gemma-3-27b-it | 0.7209 | en/0shot | `BIU-gemma3_27b-en-0shot_HIPE-2026-v1.0-impresso-test-en_run1.jsonl` |
| 3 | Mistral-Small-24B | 0.5935 | native/0shot | `BIU-mistral_small_24b-native-0shot_HIPE-2026-v1.0-impresso-test-en_run1.jsonl` |
| — | Qwen3-30B-A3B-2507 (MoE) | — | — | *no test file* |
| — | aya-expanse-32b | — | — | *no test file* |

### impresso-test-fr

| Rank | Model | Best Dev | Config | File |
|---|---|---|---|---|
| 1 | gemma-4-26B-A4B-it (MoE) | 0.7481 | en/10shot | `BIU-gemma4_26b-en-10shot_HIPE-2026-v1.0-impresso-test-fr_run1.jsonl` |
| 2 | gemma-3-27b-it | 0.6611 | en/0shot | `BIU-gemma3_27b-en-0shot_HIPE-2026-v1.0-impresso-test-fr_run1.jsonl` |
| 3 | Mistral-Small-24B | 0.5914 | native/0shot | `BIU-mistral_small_24b-native-0shot_HIPE-2026-v1.0-impresso-test-fr_run1.jsonl` |
| — | Qwen3-30B-A3B-2507 (MoE) | — | — | *no test file* |
| — | aya-expanse-32b | — | — | *no test file* |

### surprise-test-fr

| Rank | Model | Best Dev | Config | File |
|---|---|---|---|---|
| 1 | gemma-4-26B-A4B-it (MoE) | 0.7379 | en/5shot | `BIU-gemma4_26b-en-5shot_HIPE-2026-v1.0-surprise-test-fr_run1.jsonl` |
| 2 | gemma-3-27b-it | 0.6611 | en/0shot | `BIU-gemma3_27b-en-0shot_HIPE-2026-v1.0-surprise-test-fr_run1.jsonl` |
| 3 | Mistral-Small-24B | 0.5914 | native/0shot | `BIU-mistral_small_24b-native-0shot_HIPE-2026-v1.0-surprise-test-fr_run1.jsonl` |
| — | Qwen3-30B-A3B-2507 (MoE) | — | — | *no test file* |
| — | aya-expanse-32b | — | — | *no test file* |

---

## 4. Optimal Submission — Best 3 Models per Test File

*(gemma3 / gemma4 / mistral — best config per model)*

**impresso-test-de:**

  - run1: google/gemma-4-26B-A4B-it  dev=0.7316  en/10shot
  - run2: mistralai/Mistral-Small-24B-Instruct-2501  dev=0.5917  native/0shot
  - run3: google/gemma-3-27b-it  dev=0.5622  native/0shot

**impresso-test-en:**

  - run1: google/gemma-4-26B-A4B-it  dev=0.7550  en/5shot
  - run2: google/gemma-3-27b-it  dev=0.7209  en/0shot
  - run3: mistralai/Mistral-Small-24B-Instruct-2501  dev=0.5935  native/0shot

**impresso-test-fr:**

  - run1: google/gemma-4-26B-A4B-it  dev=0.7481  en/10shot
  - run2: google/gemma-3-27b-it  dev=0.6611  en/0shot
  - run3: mistralai/Mistral-Small-24B-Instruct-2501  dev=0.5914  native/0shot

**surprise-test-fr:**

  - run1: google/gemma-4-26B-A4B-it  dev=0.7379  en/5shot
  - run2: google/gemma-3-27b-it  dev=0.6611  en/0shot
  - run3: mistralai/Mistral-Small-24B-Instruct-2501  dev=0.5914  native/0shot

---

## 5. Delta vs Submission 2 (sent 2026-05-08)

| Test file | Model | Sub2 Dev | Current Best | Delta |
|---|---|---|---|---|
| impresso-test-de | gemma-4-26B-A4B-it (MoE) | 0.7119 (en/3sh) | 0.7316 (en/10sh) | +0.0197 **← IMPROVED** |
| impresso-test-de | Mistral-Small-24B | 0.5917 (native/0sh) | 0.5917 (native/0sh) | +0.0000 |
| impresso-test-de | gemma-3-27b-it | 0.5622 (native/0sh) | 0.5622 (native/0sh) | +0.0000 |
| impresso-test-en | gemma-4-26B-A4B-it (MoE) | 0.7550 (en/5sh) | 0.7550 (en/5sh) | +0.0000 |
| impresso-test-en | gemma-3-27b-it | 0.7209 (en/0sh) | 0.7209 (en/0sh) | +0.0000 |
| impresso-test-en | Mistral-Small-24B | 0.5935 (native/0sh) | 0.5935 (native/0sh) | +0.0000 |
| impresso-test-fr | gemma-4-26B-A4B-it (MoE) | 0.7218 (en/3sh) | 0.7481 (en/10sh) | +0.0263 **← IMPROVED** |
| impresso-test-fr | gemma-3-27b-it | 0.6611 (en/0sh) | 0.6611 (en/0sh) | +0.0000 |
| impresso-test-fr | Mistral-Small-24B | 0.5914 (native/0sh) | 0.5914 (native/0sh) | +0.0000 |
| surprise-test-fr | gemma-4-26B-A4B-it (MoE) | 0.7218 (en/3sh) | 0.7379 (en/5sh) | +0.0161 **← IMPROVED** |
| surprise-test-fr | gemma-3-27b-it | 0.6611 (en/0sh) | 0.6611 (en/0sh) | +0.0000 |
| surprise-test-fr | Mistral-Small-24B | 0.5914 (native/0sh) | 0.5914 (native/0sh) | +0.0000 |

---

## 6. Coverage Summary

| Model | DE eval | EN eval | FR eval | Test files |
|---|---|---|---|---|
| gemma-3-27b-it | 6/8 | 6/8 | 6/8 | 4/4 |
| gemma-4-26B-A4B-it (MoE) | 8/8 | 4/8 | 8/8 | 4/4 |
| Mistral-Small-24B | 6/8 | 6/8 | 6/8 | 4/4 |
| aya-expanse-32b | 2/8 | 2/8 | 2/8 | 0/4 |
| Qwen3-30B-A3B-2507 (MoE) | 0/8 | 0/8 | 0/8 | 0/4 |
