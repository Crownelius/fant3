"""
HellaSwag commonsense-completion evaluation.

HellaSwag (Zellers et al. 2019) is a hard commonsense reasoning benchmark.
Each example has a context and 4 possible completions, only one of which
is the natural continuation. We score by per-token log-likelihood
(length-normalized, which is the standard LM-harness protocol).

Usage
-----

    from datasets import load_dataset
    from fant2.bench import evaluate_hellaswag

    hs = load_dataset("hellaswag", split="validation")
    result = evaluate_hellaswag(model, tokenizer, hs, max_problems=500)
    print(f"HellaSwag accuracy: {result['accuracy']:.1%}")
"""

from typing import Dict, Iterable, Optional, List

import torch
import torch.nn.functional as F

from .arc import _score_continuation_logprob


def evaluate_hellaswag(
    model,
    tokenizer,
    dataset: Iterable,
    max_problems: Optional[int] = None,
    device: Optional[str] = None,
    length_normalize: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a FANT2Model on HellaSwag.

    Expects each example to have:
        "ctx":      str     # the context to complete
        "endings":  List[str]  # 4 candidate completions
        "label":    str or int # the correct ending index

    Args:
        length_normalize: if True, divide log-likelihood by the number of
            continuation tokens (the default protocol; undoes the bias
            toward shorter completions).

    Returns:
        dict with "correct", "total", "accuracy"
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    correct = 0
    total = 0

    for i, ex in enumerate(dataset):
        if max_problems is not None and i >= max_problems:
            break
        context = ex["ctx"]
        endings: List[str] = ex["endings"]
        label = int(ex["label"])

        scores = []
        for ending in endings:
            lp = _score_continuation_logprob(
                model, tokenizer, context + " ", ending, device
            )
            if length_normalize:
                n_tok = max(1, len(tokenizer.encode(ending)))
                lp = lp / n_tok
            scores.append(lp)

        pred_idx = int(torch.tensor(scores).argmax().item())

        total += 1
        if pred_idx == label:
            correct += 1

        if verbose and (i + 1) % 50 == 0:
            acc = correct / max(total, 1)
            print(f"    [hellaswag {i+1}] acc={acc:.3f} ({correct}/{total})")

    if was_training:
        model.train()

    return {
        "correct":  correct,
        "total":    total,
        "accuracy": correct / max(total, 1),
    }
