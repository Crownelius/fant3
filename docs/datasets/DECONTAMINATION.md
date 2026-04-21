# Decontamination — 13-Gram SHA-1 Filter

Source: `scripts/decontaminate.py`

## Why decontamination matters

If a training document contains text from a benchmark test question, the model may "memorise" the answer rather than learn to reason. This inflates eval scores without reflecting true generalisation. The standard practice (used by DeepMind Gopher, Meta Llama, and others) is to remove training documents that share long n-gram subsequences with test questions.

## What benchmarks are covered

The filter covers the three benchmarks used in `scripts/eval_benchmarks.py`:

| Benchmark | HF (Hugging Face) ID | Config | Split | Field | Questions |
|-----------|---------------------|--------|-------|-------|-----------|
| GSM8K (Grade-School Math 8K) | `gsm8k` | main | test | `question` | 1 319 |
| MATH-500 | `HuggingFaceH4/MATH-500` | — | test | `problem` | 500 |
| MMLU (Massive Multitask Language Understanding) | `cais/mmlu` | all | test | `question` | 14 042 |

The combined hash set across all three benchmarks contains **457 910 unique 13-gram hashes** (as measured on 2026-04-19, saved to `output/decontamination/ngram_hashes.json`).

## How it works

### Step 1 — Tokenise to words

Each text (training document or benchmark question) is lowercased and split into alphanumeric word tokens using the regex `[A-Za-z0-9]+`. Punctuation and whitespace are discarded. This normalisation catches surface-level variation such as extra punctuation, differing quote styles, and minor capitalisation changes.

### Step 2 — Extract sliding 13-grams

A 13-gram is a contiguous sequence of 13 word tokens. For a text with N word tokens, there are N − 12 overlapping 13-grams. The gram size of 13 is the DeepMind Gopher standard — long enough that random collisions between unrelated texts are astronomically unlikely (the probability of a false positive on a 13-word sequence drawn from a 50 000-word vocabulary is roughly 10⁻⁶²), short enough to catch near-verbatim reproductions.

### Step 3 — SHA-1 hash (truncated to 16 hex chars)

Each 13-gram string (space-joined lowercased tokens) is hashed with SHA-1 and truncated to the first 16 hex characters (64 bits). The full set of benchmark hashes is stored as a JSON list in `output/decontamination/ngram_hashes.json`. This file is built once on first run and reused on subsequent runs — no re-download needed.

### Step 4 — Document-level flag

A training document is flagged as **contaminated** if any of its 13-grams matches any benchmark hash. This is a conservative (document-level) filter: the entire document is dropped even if only one 13-gram matches.

```python
# Public API
from scripts.decontaminate import is_contaminated

if is_contaminated(text):
    continue  # skip this document
```

`is_contaminated()` loads the global hash set lazily on first call and caches it in memory for the rest of the process lifetime.

### Step 5 — Diagnostic mode

Running the script directly produces a per-source contamination report:

```bash
python scripts/decontaminate.py --n-docs 2000
```

Sample output (2026-04-19 results on the 6-source distillation mix):

```
source                                                    seen  contam   rate
-------------------------------------------------------------------------------------
HuggingFaceFW/fineweb-edu                                 2000       0   0.00%
crownelius/Opus-4.6-Reasoning-3300x                       2000       0   0.00%
ianncity/KIMI-K2.5-1000000x (General-Distillation)        2000       2   0.10%
AI-MO/NuminaMath-CoT                                      2000       8   0.40%
mlabonne/FineTome-100k                                    2000       2   0.10%
Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b (s1)  2000       2   0.10%
```

NuminaMath-CoT at 0.40% is the highest rate in the 6-source mix — expected because NuminaMath was constructed from competition math problems, some of which overlap with the MATH-500 test set.

## Limitations — verbatim only, not paraphrase-proof

The 13-gram filter catches **verbatim or near-verbatim overlap only**. A paraphrased version of a benchmark question — one that changes enough words to break all 13-gram windows — will pass the filter undetected. A more robust approach (embedding-based locality-sensitive hashing, for example) would catch semantic duplicates, but this is not yet implemented. The 13-gram approach is the accepted standard in the field at this time and is sufficient to prevent the most common form of benchmark contamination.

## Cache management

```bash
# Build the cache from scratch (re-downloads benchmark test sets, ~50 MB)
python scripts/decontaminate.py --rebuild-cache

# Quick self-test with a known GSM8K question
python scripts/decontaminate.py --test
```

The cache lives at `output/decontamination/ngram_hashes.json`. It is safe to commit to version control (it contains only hashes, no benchmark content).
