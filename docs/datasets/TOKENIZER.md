# Tokenizer — `tokenizer_v2.json`

Source: `scripts/retrain_tokenizer.py`
Output: `output/tokenizer/tokenizer_v2.json`
Loader: `fant2.tokenizer.bpe.FANT2Tokenizer`

## Background

The original FANT 2 tokenizer was trained on 5 000 FineWeb-Edu documents only. That corpus is biased toward plain English prose. The resulting vocabulary:

- Over-subscribed to common English words
- Assigned multiple tokens to math numerals and LaTeX sequences that the training mix uses heavily
- Had no first-class merges for the ChatML control tokens (`<|im_start|>`, `<|think|>`, etc.)

The retrained `tokenizer_v2.json` fixes all three issues by training on a representative sample of the actual training mix.

## Training procedure

`scripts/retrain_tokenizer.py` uses the HuggingFace `tokenizers` library's BPE (Byte-Pair Encoding) trainer:

```bash
python scripts/retrain_tokenizer.py \
    --n-docs 100000 \
    --vocab-size 32768 \
    --out output/tokenizer/tokenizer_v2.json
```

### Training corpus (6-source mix, weighted)

| Source | HF ID | Weight | Role |
|--------|-------|--------|------|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 40% | Base prose distribution — keeps general English fluent |
| Crownelius Opus 4.6 | `crownelius/Opus-4.6-Reasoning-3300x` | 20% | Distillation target — trains first-class merges for `<|think|>`, `<|answer|>` |
| Kimi K2.5 | `ianncity/KIMI-K2.5-1000000x` | 15% | Long reasoning traces — trains efficient CoT (Chain of Thought) token merges |
| NuminaMath CoT | `AI-MO/NuminaMath-CoT` | 10% | Math LaTeX — trains numerals and symbols |
| FineTome 100K | `mlabonne/FineTome-100k` | 10% | High-quality SFT (Supervised Fine-Tuning) pairs |
| Superior Reasoning | `Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b` | 5% | Input/output reasoning pairs |

The script streams 100 K total documents proportioned by these weights, filters to minimum length 32 characters, and passes them through `extract_text()` so the BPE trainer sees the exact same text format that the model will train on (including ChatML wrappers).

BPE trainer settings:
- `vocab_size = 32 768` — matches FANT3Config `vocab_size` field
- `min_frequency = 2` — a merge must appear at least twice to be included
- Uses the GPT-4 Unicode regex pre-tokeniser pattern for consistent word-boundary splitting

## Vocabulary size

32 768 token IDs (2¹⁵). The first 8 IDs (0–7) are reserved for special tokens; the remainder are BPE merges sorted by frequency.

## Special tokens

These tokens are added after BPE training with `add_special_tokens()` to guarantee single-token IDs regardless of frequency:

| Token | Meaning | Role |
|-------|---------|------|
| `<\|bos\|>` | Begin of sequence | Prepended to every input at inference |
| `<\|eos\|>` | End of sequence | Signals end of generation |
| `<\|pad\|>` | Padding | Used to pad short documents; targets at pad positions are set to `-100` (ignored by the loss) |
| `<\|im_start\|>` | Message start | ChatML (Chat Markup Language) role header |
| `<\|im_end\|>` | Message end | ChatML role footer |
| `<\|think\|>` | Thinking start | Opens the internal reasoning block |
| `<\|/think\|>` | Thinking end | Closes the internal reasoning block |
| `<\|answer\|>` | Answer start | Opens the graded answer block |
| `<\|/answer\|>` | Answer end | Closes the graded answer block; eval extracts content between these tags |

The sanity check at the end of `retrain_tokenizer.py` verifies that all 8 special tokens encode to a single ID each. A failure here would mean the token was fragmented during BPE training and the format alignment would break at training time.

## Compression improvement

On the distillation distribution (Opus 4.6 reasoning traces + math LaTeX + ChatML-formatted SFT pairs), `tokenizer_v2.json` achieves **10–18% better compression per character** compared to the original FineWeb-only tokenizer. This means:

- Fewer tokens per training document → more documents fit in a fixed-length context window
- The model sees more semantic content per forward pass
- Training is more sample-efficient at a fixed step count

The improvement is most pronounced on math expressions (longer merged numerals) and the `<|answer|>` / `<|think|>` control sequences (single-token IDs vs multi-token splits in the old vocabulary).

## Usage

```python
from fant2.tokenizer.bpe import FANT2Tokenizer

tok = FANT2Tokenizer.load("output/tokenizer/tokenizer_v2.json")

ids = tok.encode("What is 4 + 3?")
text = tok.decode(ids)

# With BOS / EOS
ids = tok.encode("Hello", add_bos=True, add_eos=True)
```

## Smoke test

```bash
# Quick 1 000-document smoke run (outputs to tokenizer_v2_smoke.json)
python scripts/retrain_tokenizer.py --smoke
```
