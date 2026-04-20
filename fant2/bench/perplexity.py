"""
Perplexity evaluation.

Computes the average negative-log-likelihood per token on a held-out text
corpus, then exponentiates to get the standard perplexity metric:

    PPL = exp( - (1/N) * sum_i log p(x_i | x_<i) )

Usage
-----

    tok   = FANT2Tokenizer.load("data/tokenizer.json")
    model = FANT2Model(cfg)
    model.load_state_dict(ckpt["model"])

    from fant2.data import HuggingFaceStream, TokenizedBatchStream
    stream = TokenizedBatchStream(
        HuggingFaceStream(dataset_name="wikitext", dataset_config="wikitext-2-v1"),
        tokenizer=tok, batch_size=1, seq_len=1024,
    )
    result = evaluate_perplexity(model, stream, max_batches=200)
    print(f"perplexity: {result['perplexity']:.3f}")
"""

from typing import Dict, Iterable, Optional

import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_perplexity(
    model,
    batch_stream: Iterable,
    max_batches: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute perplexity over a batch stream.

    Args:
        model:       FANT2Model (or any model that returns {"logits": (B, T, V)})
        batch_stream: iterator of (input_ids, target_ids) tensor pairs
        max_batches: optional cap on number of batches evaluated
        device:      override; defaults to model's device
        verbose:     print per-batch progress

    Returns:
        dict with "loss", "perplexity", "n_tokens"
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    for batch_idx, (input_ids, target_ids) in enumerate(batch_stream):
        if max_batches is not None and batch_idx >= max_batches:
            break
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        out = model(input_ids)
        logits = out["logits"]  # (B, T, V)

        # Token-level NLL, summed (not averaged), so we can compute true
        # per-token perplexity over the entire corpus.
        B, T, V = logits.shape
        nll = F.cross_entropy(
            logits.reshape(-1, V),
            target_ids.reshape(-1),
            reduction="sum",
            ignore_index=-100,
        )
        total_nll += float(nll.item())
        total_tokens += int((target_ids != -100).sum().item())

        if verbose and (batch_idx + 1) % 20 == 0:
            interim_ppl = math.exp(total_nll / max(total_tokens, 1))
            print(f"    [batch {batch_idx+1}] ppl={interim_ppl:.3f}")

    if was_training:
        model.train()

    if total_tokens == 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "n_tokens": 0}

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return {
        "loss":       avg_nll,
        "perplexity": ppl,
        "n_tokens":   total_tokens,
    }
