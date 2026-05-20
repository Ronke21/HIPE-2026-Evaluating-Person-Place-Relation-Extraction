# HIPE-2026 Official Evaluation Results — team10 (BIU_NLP)

Source: https://github.com/hipe-eval/hipe-2026-eval (final de-anonymized release, updated 2026-05-20)
Files: `HIPE_2026_evaluation_results.md` and `HIPE_2026_evaluation_results-binary.md`

## Team identity

| team id | name | affiliation | country |
|---------|------|-------------|---------|
| team10 | **BIU_NLP** | Bar-Ilan University | IL |

## Run mapping (final updated submission)

| Run | Model | Config |
|-----|-------|--------|
| run1 | google/gemma-4-26B-A4B-it (MoE, 26B total / 4B active, ~52 GB) | en/3-shot (de, fr, surprise-fr) · en/5-shot (en) |
| run2 | google/gemma-3-27b-it (27B dense, ~54 GB) | en/0-shot (en, fr, surprise-fr) · native/0-shot (de) |
| run3 | mistralai/Mistral-Small-24B-Instruct-2501 (24B dense, ~48 GB) | native/0-shot (all) |

---

## MAIN EVALUATION

### Accuracy Profile — Overall (mean across de/en/fr)

| rank | out of | run  | mean impresso profile score |
|------|--------|------|-----------------------------|
| 20   | 46     | run2 | 0.5781 |
| 24   | 46     | run3 | 0.5390 |
| 41   | 46     | run1 | 0.4429 |

### Accuracy Profile — German (impresso-test-de)

| rank | out of | run  | impresso score | at macro recall | at accuracy | isAt macro recall | isAt accuracy |
|------|--------|------|---------------|-----------------|-------------|-------------------|---------------|
| 20   | 46     | run3 | 0.5451 | 0.4452 | 0.6261 | 0.6450 | 0.7899 |
| 31   | 46     | run2 | 0.5050 | 0.4740 | 0.5882 | 0.5360 | 0.7311 |
| 37   | 46     | run1 | 0.4495 | 0.3765 | 0.6429 | 0.5224 | 0.7311 |

### Accuracy Profile — English (impresso-test-en)

| rank | out of | run  | impresso score | at macro recall | at accuracy | isAt macro recall | isAt accuracy |
|------|--------|------|---------------|-----------------|-------------|-------------------|---------------|
| 19   | 47     | run2 | 0.5762 | 0.4807 | 0.5309 | 0.6718 | 0.7346 |
| 28   | 47     | run3 | 0.5101 | 0.4586 | 0.5062 | 0.5615 | 0.6481 |
| 36   | 47     | run1 | 0.4583 | 0.3962 | 0.4444 | 0.5205 | 0.6111 |

### Accuracy Profile — French (impresso-test-fr)

| rank | out of | run  | impresso score | at macro recall | at accuracy | isAt macro recall | isAt accuracy |
|------|--------|------|---------------|-----------------|-------------|-------------------|---------------|
| 9    | 46     | run2 | 0.6529 | 0.5703 | 0.6387 | 0.7356 | 0.7773 |
| 23   | 46     | run3 | 0.5617 | 0.5743 | 0.6891 | 0.5492 | 0.6891 |
| 42   | 46     | run1 | 0.4208 | 0.3387 | 0.5672 | 0.5030 | 0.6597 |

### Generalization Profile — Surprise-FR (summary)

| rank | out of | run  | surprise profile score |
|------|--------|------|------------------------|
| 6    | 46     | run1 | 0.6837 |
| 17   | 46     | run3 | 0.6000 |
| 22   | 46     | run2 | 0.5265 |

### Generalization Profile — Surprise-FR (detailed)

| rank | out of | run  | surprise score | at macro recall | at accuracy |
|------|--------|------|---------------|-----------------|-------------|
| 6    | 46     | run1 | 0.6837 | 0.6837 | 0.8354 |
| 17   | 46     | run3 | 0.6000 | 0.6000 | 0.6042 |
| 22   | 46     | run2 | 0.5265 | 0.5265 | 0.4729 |

### Efficiency Profile — Overall

| rank | out of | run  | mean eff rank | acc rank | param rank | size rank | mean score | params | size (MB) |
|------|--------|------|---------------|----------|------------|-----------|------------|--------|-----------|
| 30   | 32     | run2 | 24.6667 | 17 | 28 | 29 | 0.5781 | 27B | 54,000 |
| 30   | 32     | run3 | 24.6667 | 21 | 26 | 27 | 0.5390 | 24B | 48,000 |
| 32   | 32     | run1 | 31.0000 | 38 | 27 | 28 | 0.4429 | 26B | 52,000 |

### Balanced Efficiency Profile — Overall

| rank | out of | run  | balanced rank | acc rank | param rank | size rank | mean score | params | size (MB) |
|------|--------|------|---------------|----------|------------|-----------|------------|--------|-----------|
| 25   | 31     | run2 | 22.75 | 17 | 28 | 29 | 0.5781 | 27B | 54,000 |
| 27   | 31     | run3 | 23.75 | 21 | 26 | 27 | 0.5390 | 24B | 48,000 |
| 31   | 31     | run1 | 32.75 | 38 | 27 | 28 | 0.4429 | 26B | 52,000 |

### Efficiency Profile — Per Language

**German:**

| rank | out of | run  | mean eff rank | acc rank | param rank | size rank | score  | params | size (MB) |
|------|--------|------|---------------|----------|------------|-----------|--------|--------|-----------|
| 25   | 32     | run3 | 23.3333 | 17 | 26 | 27 | 0.5451 | 24B | 48,000 |
| 31   | 32     | run2 | 28.3333 | 28 | 28 | 29 | 0.5050 | 27B | 54,000 |
| 32   | 32     | run1 | 29.6667 | 34 | 27 | 28 | 0.4495 | 26B | 52,000 |

**English:**

| rank | out of | run  | mean eff rank | acc rank | param rank | size rank | score  | params | size (MB) |
|------|--------|------|---------------|----------|------------|-----------|--------|--------|-----------|
| 27   | 31     | run2 | 25.0000 | 16 | 29 | 30 | 0.5762 | 27B | 54,000 |
| 28   | 31     | run3 | 26.6667 | 25 | 27 | 28 | 0.5101 | 24B | 48,000 |
| 31   | 31     | run1 | 30.0000 | 33 | 28 | 29 | 0.4583 | 26B | 52,000 |

**French:**

| rank | out of | run  | mean eff rank | acc rank | param rank | size rank | score  | params | size (MB) |
|------|--------|------|---------------|----------|------------|-----------|--------|--------|-----------|
| 21   | 27     | run2 | 21.6667 | 8  | 28 | 29 | 0.6529 | 27B | 54,000 |
| 25   | 27     | run3 | 24.3333 | 20 | 26 | 27 | 0.5617 | 24B | 48,000 |
| 27   | 27     | run1 | 31.3333 | 39 | 27 | 28 | 0.4208 | 26B | 52,000 |

---

## BINARY EVALUATION (PROBABLE "at" → TRUE)

### Accuracy Profile — Overall

| rank | out of | run  | mean impresso profile score |
|------|--------|------|-----------------------------|
| 20   | 46     | run2 | 0.6721 |
| 24   | 46     | run3 | 0.6274 |
| 41   | 46     | run1 | 0.5252 |

### Accuracy Profile — German

| rank | out of | run  | impresso score | at macro recall | at accuracy | isAt macro recall | isAt accuracy |
|------|--------|------|---------------|-----------------|-------------|-------------------|---------------|
| 23   | 46     | run3 | 0.6381 | 0.6312 | 0.6807 | 0.6450 | 0.7899 |
| 24   | 46     | run2 | 0.6231 | 0.7102 | 0.7185 | 0.5360 | 0.7311 |
| 40   | 46     | run1 | 0.5329 | 0.5435 | 0.6471 | 0.5224 | 0.7311 |

### Accuracy Profile — English

| rank | out of | run  | impresso score | at macro recall | at accuracy | isAt macro recall | isAt accuracy |
|------|--------|------|---------------|-----------------|-------------|-------------------|---------------|
| 21   | 47     | run2 | 0.6524 | 0.6330 | 0.6852 | 0.6718 | 0.7346 |
| 28   | 47     | run3 | 0.5904 | 0.6193 | 0.5988 | 0.5615 | 0.6481 |
| 36   | 47     | run1 | 0.5382 | 0.5559 | 0.4444 | 0.5205 | 0.6111 |

### Accuracy Profile — French

| rank | out of | run  | impresso score | at macro recall | at accuracy | isAt macro recall | isAt accuracy |
|------|--------|------|---------------|-----------------|-------------|-------------------|---------------|
| 14   | 46     | run2 | 0.7409 | 0.7462 | 0.7227 | 0.7356 | 0.7773 |
| 23   | 46     | run3 | 0.6538 | 0.7584 | 0.7437 | 0.5492 | 0.6891 |
| 42   | 46     | run1 | 0.5044 | 0.5059 | 0.5672 | 0.5030 | 0.6597 |

### Generalization Profile — Surprise-FR (summary)

| rank | out of | run  | surprise profile score |
|------|--------|------|------------------------|
| 3    | 44     | run1 | 0.8804 |
| 20   | 44     | run3 | 0.7262 |
| 26   | 44     | run2 | 0.6473 |

### Generalization Profile — Surprise-FR (detailed)

| rank | out of | run  | surprise score | at macro recall | at accuracy |
|------|--------|------|---------------|-----------------|-------------|
| 3    | 44     | run1 | 0.8804 | 0.8804 | 0.8917 |
| 20   | 44     | run3 | 0.7262 | 0.7262 | 0.6813 |
| 26   | 44     | run2 | 0.6473 | 0.6473 | 0.5771 |

### Efficiency Profile — Overall

| rank | out of | run  | mean eff rank | acc rank | param rank | size rank | mean score | params | size (MB) |
|------|--------|------|---------------|----------|------------|-----------|------------|--------|-----------|
| 31   | 33     | run2 | 24.6667 | 17 | 28 | 29 | 0.6721 | 27B | 54,000 |
| 31   | 33     | run3 | 24.6667 | 21 | 26 | 27 | 0.6274 | 24B | 48,000 |
| 33   | 33     | run1 | 31.0000 | 38 | 27 | 28 | 0.5252 | 26B | 52,000 |

### Balanced Efficiency Profile — Overall

| rank | out of | run  | balanced rank | acc rank | param rank | size rank | mean score | params | size (MB) |
|------|--------|------|---------------|----------|------------|-----------|------------|--------|-----------|
| 28   | 34     | run2 | 22.75 | 17 | 28 | 29 | 0.6721 | 27B | 54,000 |
| 29   | 34     | run3 | 23.75 | 21 | 26 | 27 | 0.6274 | 24B | 48,000 |
| 34   | 34     | run1 | 32.75 | 38 | 27 | 28 | 0.5252 | 26B | 52,000 |

### Efficiency Profile — Per Language

**German:**

| rank | out of | run  | mean eff rank | acc rank | param rank | size rank | score  | params | size (MB) |
|------|--------|------|---------------|----------|------------|-----------|--------|--------|-----------|
| 22   | 26     | run3 | 24.3333 | 20 | 26 | 27 | 0.6381 | 24B | 48,000 |
| 25   | 26     | run2 | 26.0000 | 21 | 28 | 29 | 0.6231 | 27B | 54,000 |
| 26   | 26     | run1 | 30.6667 | 37 | 27 | 28 | 0.5329 | 26B | 52,000 |

**English:**

| rank | out of | run  | mean eff rank | acc rank | param rank | size rank | score  | params | size (MB) |
|------|--------|------|---------------|----------|------------|-----------|--------|--------|-----------|
| 27   | 31     | run2 | 25.6667 | 18 | 29 | 30 | 0.6524 | 27B | 54,000 |
| 29   | 31     | run3 | 26.6667 | 25 | 27 | 28 | 0.5904 | 24B | 48,000 |
| 31   | 31     | run1 | 30.0000 | 33 | 28 | 29 | 0.5382 | 26B | 52,000 |

**French:**

| rank | out of | run  | mean eff rank | acc rank | param rank | size rank | score  | params | size (MB) |
|------|--------|------|---------------|----------|------------|-----------|--------|--------|-----------|
| 23   | 28     | run2 | 22.6667 | 11 | 28 | 29 | 0.7409 | 27B | 54,000 |
| 26   | 28     | run3 | 24.3333 | 20 | 26 | 27 | 0.6538 | 24B | 48,000 |
| 28   | 28     | run1 | 31.3333 | 39 | 27 | 28 | 0.5044 | 26B | 52,000 |

---

## Highlights

- **Best result:** run2 (Gemma-3) French — rank 9/46 main · rank 14/46 binary
- **Best generalization:** run1 (Gemma-4) surprise-FR — rank 6/46 main · rank 3/44 binary (0.8804)
- **Notable anomaly:** run1 had the best dev scores (0.71–0.76) but ranked lowest on
  all three impresso test sets (ranks 36–42), while ranking best on surprise-FR.
  Possible causes: PROBABLE over-prediction, dev/test domain shift, few-shot example bias.
