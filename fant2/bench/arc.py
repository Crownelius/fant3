"""
AI2 Reasoning Challenge (ARC) multi-choice evaluation.

ARC-Easy and ARC-Challenge (Clark et al. 2018) are multiple-choice science
questions. Each example has a question stem and 4 candidate answers; the
task is to pick the right one.

We score by computing the log-likelihood of each candidate answer's tokens
conditional on the question, then picking the argmax. This is the standard
"pseudo-likelihood" protocol used by lm-eval-harness.

Usage
-----

    from datasets import load_dataset
    from fant2.bench import evaluate_arc_multichoice

    arc = load_dataset("ai2_arc", "ARC-Easy", split="test")
    result = evaluate_arc_multichoice(model, tokenizer, arc, max_problems=500)
    print(f"ARC-Easy accuracy: {result['accuracy']:.1%}")
"""

from typing import Dict, Iterable, Optional, List

import torch
import torch.nn.functional as F


@torch.no_grad()
def _score_continuation_logprob(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    device: str,
) -> float:
    """
    Compute log p(continuation | prompt), summed over the continuation tokens.

    This is standard LM-harness style: tokenize (prompt + continuation), run
    the model, and sum the log-probabilities of the continuation's tokens.
    """
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    full_ids   = tokenizer.encode(prompt + continuation, add_bos=True, add_eos=False)
    cont_start = len(prompt_ids)

    if len(full_ids) > model.config.max_seq_len:
        # Truncate from the left (keep the continuation)
        overflow = len(full_ids) - model.config.max_seq_len
        full_ids = full_ids[overflow:]
        cont_start = max(0, cont_start - overflow)

    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    out = model(input_tensor)
    logits = out["logits"][0]  # (T, V)

    # logits[t] predicts full_ids[t+1]. We want the log-prob of each
    # continuation token at the position where it IS the target.
    # For token at full_ids[cont_start .. end], its logits are at [cont_start-1 .. end-1].
    total_logp = 0.0
    n_cont = len(full_ids) - cont_start
    for i in range(n_cont):
        logit_idx = cont_start - 1 + i
        target_idx = cont_start + i
        if logit_idx < 0:
            continue
        logp = F.log_softmax(logits[logit_idx], dim=-1)
        total_logp += float(logp[full_ids[target_idx]].item())
    return total_logp


def evaluate_arc_multichoice(
    model,
    tokenizer,
    dataset: Iterable,
    max_problems: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a FANT2Model on an ARC-style multi-choice dataset.

    Expects each example to have:
        "question":     str
        "choices":      dict with "text": List[str] and "label": List[str]
        "answerKey":    str (one of the labels)

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
        question = ex["question"]
        choices = ex["choices"]
        labels: List[str] = choices["label"]
        texts:  List[str] = choices["text"]
        answer_key: str = ex["answerKey"]

        if answer_key not in labels:
            continue  # malformed

        prompt = f"Question: {question}\nAnswer: "
        scores = []
        for text in texts:
            lp = _score_continuation_logprob(model, tokenizer, prompt, text, device)
            # Length-normalize so longer answers don't get penalized
            # (matches MMLU's protocol — fixes the un-normed-scorer artifact
            # that left ARC results stuck near random in Option H/H2).
            n_tok = max(1, len(tokenizer.encode(text, add_bos=False, add_eos=False)))
            scores.append(lp / n_tok)

        pred_idx = int(torch.tensor(scores).argmax().item())
        pred_label = labels[pred_idx]

        total += 1
        if pred_label == answer_key:
            correct += 1

        if verbose and (i + 1) % 50 == 0:
            acc = correct / max(total, 1)
            print(f"    [arc {i+1}] acc={acc:.3f} ({correct}/{total})")

    if was_training:
        model.train()

    return {
        "correct":  correct,
        "total":    total,
        "accuracy": correct / max(total, 1),
    }
