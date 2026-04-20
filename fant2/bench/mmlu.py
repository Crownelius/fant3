"""
MMLU (Massive Multitask Language Understanding) evaluation.

MMLU (Hendrycks et al. 2020) is a 4-choice exam over 57 academic subjects
ranging from elementary mathematics to law, medicine, and abstract algebra.
The standard protocol is the same pseudo-likelihood scoring used by
lm-eval-harness: tokenize "Question + Answer" for each candidate and pick
the candidate with the highest summed log-probability over the answer's
own tokens.

The HF dataset (`cais/mmlu`) gives each example as:
    "question": str
    "subject":  str
    "choices":  List[str]   # 4 candidates, in order A B C D
    "answer":   int         # gold index 0-3

Usage
-----

    from datasets import load_dataset
    from fant2.bench import evaluate_mmlu

    mmlu = load_dataset("cais/mmlu", "all", split="test")
    result = evaluate_mmlu(model, tokenizer, mmlu, max_problems=500)
    print(f"MMLU accuracy: {result['accuracy']:.1%}")
"""

from typing import Dict, Iterable, Optional, List

import torch

from .arc import _score_continuation_logprob


_LETTERS = ["A", "B", "C", "D"]


def _format_mmlu_prompt(question: str, choices: List[str]) -> str:
    """
    Standard MMLU prompt: question + lettered choices + 'Answer:'.

    The continuation we score is the choice TEXT (not just the letter),
    matching the lm-eval-harness 'mmlu' (not 'mmlu_letters') protocol —
    this is more discriminative for small models because the letters
    have near-uniform priors.
    """
    lines = [f"Question: {question}"]
    for letter, text in zip(_LETTERS, choices):
        lines.append(f"{letter}. {text}")
    lines.append("Answer:")
    return "\n".join(lines) + " "


def evaluate_mmlu(
    model,
    tokenizer,
    dataset: Iterable,
    max_problems: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a FANT2Model on MMLU via length-normalized continuation scoring.

    Each example must have:
        "question": str
        "choices":  List[str]  (length 4)
        "answer":   int        (gold index in [0, 4))

    Returns:
        dict with "correct", "total", "accuracy", "per_subject" (if subject
        keys are present in the dataset items).
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    correct = 0
    total = 0
    per_subject: Dict[str, Dict[str, int]] = {}

    for i, ex in enumerate(dataset):
        if max_problems is not None and i >= max_problems:
            break
        question: str = ex["question"]
        choices: List[str] = ex["choices"]
        gold: int = int(ex["answer"])
        subject: str = ex.get("subject", "all")

        if not (0 <= gold < len(choices)):
            continue  # malformed

        prompt = _format_mmlu_prompt(question, choices)
        scores: List[float] = []
        for choice_text in choices:
            lp = _score_continuation_logprob(
                model, tokenizer, prompt, choice_text, device,
            )
            # Length-normalize so longer answers don't get penalized.
            n_tok = max(1, len(tokenizer.encode(choice_text, add_bos=False, add_eos=False)))
            scores.append(lp / n_tok)

        pred = int(torch.tensor(scores).argmax().item())

        total += 1
        bucket = per_subject.setdefault(subject, {"correct": 0, "total": 0})
        bucket["total"] += 1
        if pred == gold:
            correct += 1
            bucket["correct"] += 1

        if verbose and (i + 1) % 50 == 0:
            acc = correct / max(total, 1)
            print(f"    [mmlu {i+1}] acc={acc:.3f} ({correct}/{total})")

    if was_training:
        model.train()

    return {
        "correct":     correct,
        "total":       total,
        "accuracy":    correct / max(total, 1),
        "per_subject": per_subject,
    }
