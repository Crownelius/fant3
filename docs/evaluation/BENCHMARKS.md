# Benchmark Evaluation

Source: `scripts/eval_benchmarks.py`

## Overview

Three benchmarks are supported. All are **eval-only** — they must never appear as training sources (the `EVAL_DATASETS` dict in `fant2/data/registry.py` enforces this at the registry level; the 13-gram decontamination filter enforces it at the document level).

| Benchmark | Task type | Total problems | Subset used | Answer format |
|-----------|-----------|---------------|-------------|---------------|
| GSM8K (Grade-School Math 8K) | Math word problems | 1 319 (test set) | Up to `--n` | Final numeric answer extracted from `#### N` in gold |
| MMLU (Massive Multitask Language Understanding) | 4-way multiple choice, 57 subjects | 14 042 (test set) | Up to `--n` | Letter logit comparison (A/B/C/D) |
| MATH-500 | Competition math (Hendrycks MATH) | 500 (test set) | Up to `--n` | LaTeX `\boxed{...}` |

## Running an evaluation

```bash
python scripts/eval_benchmarks.py \
    --ckpt /path/to/checkpoint.pt \
    --tokenizer output/tokenizer/tokenizer_v2.json \
    --benchmark gsm8k \
    --n 100 \
    --device cuda \
    --dtype bf16
```

The script auto-detects CUDA if `--device` is omitted. Checkpoints must contain a `cfg_dict` key (saved by the Colab notebook cell 20) so the model can be rebuilt without knowing the scale preset.

## Model loading

`_load_model_and_tok()` handles three checkpoint formats:
1. **Bare state dict** — treated as scale `"1b"` (legacy, not recommended)
2. **`{"model": state, "cfg_scale": str}`** — uses the named preset builder (older format)
3. **`{"model": state, "cfg_dict": dict}`** — reconstructs `FANT3Config` from the saved dict (current, preferred)

Format 3 is the only format guaranteed to reconstruct the exact architecture for 20m/50m/150m/350m/742m scales defined inline in the Colab notebook, since those configs are not registered as named presets.

---

## GSM8K evaluation

### Gold answer extraction

GSM8K answer fields end with `#### N` where N is the final numeric answer. `_gsm8k_gold()` extracts N using the regex `####\s*(-?[\d,]+(?:\.\d+)?)` and strips commas.

### Prediction extraction

Predictions are extracted from greedy completions using a three-stage priority cascade:

1. **`<|answer|>...<|/answer|>` tag** — if present, extract the last number inside the tag.
2. **`\boxed{...}`** — if present, extract the number inside.
3. **Last plain number** — fallback to the last occurrence of the pattern `-?\d+(?:,\d{3})*(?:\.\d+)?` in the completion.

Comparison is exact string match after stripping commas. For example, `"72"` matches `"72"` but not `"72.0"`.

### Prompt format

Each question is wrapped in the ChatML format matching training:

```
<|bos|><|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
```

Greedy generation runs for up to 256 new tokens, stopping at `<|im_end|>` or `<|eos|>`.

---

## MMLU evaluation

MMLU uses **letter-logit comparison** rather than greedy generation. This is more efficient (one forward pass per question) and more robust at early training stages when the model may not yet generate coherent text.

### Prompt format

```
Question: {q}
A. {choice_0}
B. {choice_1}
C. {choice_2}
D. {choice_3}
Answer: 
```

### Letter-logit extraction

`_mmlu_letter_logits()` runs one forward pass and reads the logits at the final token position. For each letter in `["A", "B", "C", "D"]`, the logit is retrieved from the vocabulary:

1. Try `tok._tok.token_to_id("A")` (exact)
2. Try `tok._tok.token_to_id(" A")` (with leading space — BPE (Byte-Pair Encoding) may merge the space)
3. Fall back to `tok.encode("A")[0]`

The predicted answer is `argmax(logits_A, logits_B, logits_C, logits_D)`. No softmax is applied — raw logit comparison is equivalent.

This method avoids the greedy-generation overhead and is immune to early-training format failures (the model does not need to emit `<|answer|>` tags to get a score).

---

## MATH-500 evaluation

MATH-500 is the 500-problem subset of the Hendrycks MATH benchmark from `HuggingFaceH4/MATH-500`.

### Gold answer extraction

Gold answers are enclosed in LaTeX `\boxed{...}`. `_math500_gold()` extracts the content and strips commas.

### Prediction extraction

For MATH-500, the extraction priority is:

1. **`\boxed{...}`** in the completion — directly extracted.
2. **Last plain number** — fallback.

Comparison is exact string match (after comma stripping), so `"\frac{1}{2}"` only matches `"\frac{1}{2}"` not `"0.5"`. This is strict but consistent with standard MATH evaluation.

Greedy generation runs for up to 512 new tokens (longer than GSM8K because competition math solutions can be verbose).

---

## Wilson 95% Confidence Interval

All three benchmarks report a 95% Wilson CI (Confidence Interval) alongside the point accuracy. The Wilson score interval is used instead of the normal approximation because it is well-behaved for small `n` and for probabilities near 0 or 1, where the normal approximation gives impossible intervals (below 0 or above 1).

### Formula

Given `k` correct answers out of `n` total, with `z = 1.96` (the 1.96 standard-deviation z-score for 95% confidence):

```
p̂ = k / n

centre = (p̂ + z² / (2n)) / (1 + z² / n)

spread  = z * sqrt(p̂(1-p̂)/n + z²/(4n²)) / (1 + z² / n)

lower = max(0, centre - spread)
upper = min(1, centre + spread)
```

In Python (`scripts/eval_benchmarks.py`, `_wilson_ci()`):

```python
def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))
```

### Practical interpretation

| n (problems evaluated) | CI half-width (at 50% accuracy) |
|---|---|
| 50 | ±13.8% |
| 100 | ±9.8% |
| 200 | ±6.9% |
| 500 | ±4.4% |
| 1 000 | ±3.1% |

For a 3–5 percentage-point improvement to be statistically distinguishable from noise, at least 500 problems are needed. The standard FANT evaluation uses 1 000 problems (`eval_1k.py`) for Campaign N comparisons and 50–200 for quick directional checks during training.

### Example output

```
=== RESULT: gsm8k ===
  n         100
  correct   6
  accuracy  6.00%  [95% CI 2.78% – 12.52%]
```

A result of 6/100 on GSM8K at early training (CE ≈ 6–7) is expected — the model has not yet learned arithmetic. The CI confirms this is consistent with 1/13 random chance for a numeric answer space.
