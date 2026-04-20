"""
Phase 6 — SimPO + KTO preference data + step.

Implements the synthetic preference loop that the trainer's
`_phase6_simpo_kto_forward` calls per outer step:

  1. Sample a procedurally-generated math problem.
  2. Build a (prompt, chosen, rejected) triple where:
       * chosen   = well-formatted CoT with the correct numeric answer
       * rejected = wrong number, unformatted text, or "I don't know"
  3. Tokenize prompt+chosen and prompt+rejected.
  4. Score the response tokens (mask the prompt) against the live model
     with grad and against a frozen reference without grad.
  5. Compose `simpo_loss + 0.5 * kto_loss`.

**Training data policy: NO public benchmarks.** The locked spec names
UltraFeedback / Tulu / Magpie-Pro as Phase 6 alignment sources, but the
user has explicitly forbidden training on any public benchmark. This
module therefore derives preference pairs from `ProceduralMathStream` —
the chosen/rejected differential gives a clean preference signal without
touching external data. Acceptance gates (eval) may still use real
preference benchmarks; that's eval, not training.

Designed to be importable from both:
  * `fant2.training.phase6_simpo_kto` (real training entry point)
  * `scripts.option_f_phase6_simpo` (smoke gate)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch

from .phase5_rollout import (
    MathExample,
    ProceduralMathStream,
    format_prompt,
    response_logp_sum,
)


# -----------------------------------------------------------------------------
# 1. Synthetic preference example + stream
# -----------------------------------------------------------------------------

@dataclass
class PrefExample:
    prompt: str          # the formatted question
    chosen: str          # well-formatted, correct response
    rejected: str        # malformed, wrong, or unhelpful response
    template: str        # which math template the prompt came from
    rejected_kind: str   # "wrong" | "unformatted" | "unhelpful"


_REJECTED_KINDS = ("wrong", "unformatted", "unhelpful")
_UNHELPFUL_VARIANTS = (
    "I don't know.",
    "no idea",
    "ask someone else",
    "skip",
)


def _make_chosen(gold: str) -> str:
    return (
        f" Let me work it out step by step. The answer is {gold}.</think>"
        f"<answer>{gold}</answer>"
    )


def _make_rejected(gold: str, kind: str, rng: random.Random) -> str:
    if kind == "wrong":
        # Pick a wrong number that is plausibly close to the gold value.
        try:
            g = int(gold)
        except ValueError:
            g = 0
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        wrong = g + delta
        return (
            f" Let me work it out step by step. The answer is {wrong}.</think>"
            f"<answer>{wrong}</answer>"
        )
    if kind == "unformatted":
        return f" the answer is just {gold} I think"
    # unhelpful
    return " " + rng.choice(_UNHELPFUL_VARIANTS)


class SyntheticPreferenceStream:
    """
    Procedurally-generated (prompt, chosen, rejected) triples.

    The prompt is a math word problem (reused from `ProceduralMathStream`).
    The chosen response is a well-formatted reasoning + correct answer.
    The rejected response is one of three failure modes:
      * wrong: well-formatted reasoning but the wrong number
      * unformatted: ignores the schema, gives the right number anyway
      * unhelpful: gives no answer at all

    This gives the model a real signal to learn from: format AND
    correctness AND non-evasion. No external data is used.
    """

    def __init__(
        self,
        seed: int = 0,
        max_value: int = 20,
        problems: Optional[Iterator[MathExample]] = None,
    ):
        if problems is None:
            problems = ProceduralMathStream(seed=seed, max_value=max_value)
        self._problems = iter(problems)
        self._rng = random.Random(seed + 1)

    def __iter__(self) -> Iterator[PrefExample]:
        return self

    def __next__(self) -> PrefExample:
        ex = next(self._problems)
        kind = self._rng.choice(_REJECTED_KINDS)
        prompt = format_prompt(ex.question)
        chosen = _make_chosen(ex.gold_answer)
        rejected = _make_rejected(ex.gold_answer, kind, self._rng)
        return PrefExample(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            template=ex.template,
            rejected_kind=kind,
        )


# -----------------------------------------------------------------------------
# 2. One SimPO + KTO outer step on a single preference triple
# -----------------------------------------------------------------------------

@dataclass
class PrefStepResult:
    loss: torch.Tensor          # composite SimPO + 0.5 * KTO, with grad
    simpo: torch.Tensor         # SimPO term (with grad)
    kto: torch.Tensor           # KTO term (with grad)
    chosen_lp: float            # detached scalars for logging
    rejected_lp: float
    ref_chosen_lp: float
    ref_rejected_lp: float
    margin: float               # chosen_lp/|c| - rejected_lp/|r|
    chosen_len: int
    rejected_len: int


def _encode_pair(tokenizer, prompt: str, response: str) -> tuple[List[int], int]:
    """Tokenize prompt+response and return (full_ids, prompt_len)."""
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    full_ids = tokenizer.encode(prompt + response, add_bos=True, add_eos=True)
    # Defensive: tokenizer may merge across the prompt/response boundary, so
    # the encoded `prompt + response` might not literally start with prompt_ids.
    # Find the longest common prefix in token-ids.
    n = 0
    while n < len(prompt_ids) and n < len(full_ids) and prompt_ids[n] == full_ids[n]:
        n += 1
    prompt_len = n if n > 0 else len(prompt_ids)
    return full_ids, prompt_len


def simpo_kto_step(
    *,
    model,
    ref_model,
    tokenizer,
    example: PrefExample,
    simpo_beta: float = 2.0,
    simpo_gamma: float = 1.6,
    kto_beta: float = 0.1,
    kto_weight: float = 0.5,
    device: str = "cpu",
) -> PrefStepResult:
    """
    Run one SimPO + KTO step on a single preference triple.

    Both losses are scalar; the caller (trainer hook or smoke gate) sums
    across the batch and calls `.backward()`. The reference model is
    expected to be a frozen `copy.deepcopy(model)` snapshot.
    """
    from .losses import simpo_loss, kto_loss

    chosen_ids, chosen_prompt_len = _encode_pair(
        tokenizer, example.prompt, example.chosen,
    )
    rejected_ids, rejected_prompt_len = _encode_pair(
        tokenizer, example.prompt, example.rejected,
    )

    chosen_resp_len = max(1, len(chosen_ids) - chosen_prompt_len)
    rejected_resp_len = max(1, len(rejected_ids) - rejected_prompt_len)

    # Live policy log-probs (with grad).
    chosen_lp = response_logp_sum(
        model, chosen_ids, chosen_prompt_len, device=device, no_grad=False,
    )
    rejected_lp = response_logp_sum(
        model, rejected_ids, rejected_prompt_len, device=device, no_grad=False,
    )
    # Frozen reference log-probs (no grad).
    ref_chosen_lp = response_logp_sum(
        ref_model, chosen_ids, chosen_prompt_len, device=device, no_grad=True,
    ).detach()
    ref_rejected_lp = response_logp_sum(
        ref_model, rejected_ids, rejected_prompt_len, device=device, no_grad=True,
    ).detach()

    # SimPO operates on (B,) tensors of summed log-probs and lengths.
    chosen_lp_b = chosen_lp.unsqueeze(0)
    rejected_lp_b = rejected_lp.unsqueeze(0)
    chosen_len_b = torch.tensor([chosen_resp_len], dtype=torch.float32, device=device)
    rejected_len_b = torch.tensor([rejected_resp_len], dtype=torch.float32, device=device)
    simpo = simpo_loss(
        chosen_lp_b, rejected_lp_b, chosen_len_b, rejected_len_b,
        beta=simpo_beta, gamma=simpo_gamma,
    )

    # KTO needs the reference log-probs.
    kto = kto_loss(
        chosen_lp_b,
        rejected_lp_b,
        ref_chosen_lp.unsqueeze(0),
        ref_rejected_lp.unsqueeze(0),
        beta=kto_beta,
    )

    loss = simpo + kto_weight * kto

    margin = float(
        (chosen_lp.detach() / chosen_resp_len)
        - (rejected_lp.detach() / rejected_resp_len)
    )
    return PrefStepResult(
        loss=loss,
        simpo=simpo,
        kto=kto,
        chosen_lp=float(chosen_lp.detach().item()),
        rejected_lp=float(rejected_lp.detach().item()),
        ref_chosen_lp=float(ref_chosen_lp.item()),
        ref_rejected_lp=float(ref_rejected_lp.item()),
        margin=margin,
        chosen_len=chosen_resp_len,
        rejected_len=rejected_resp_len,
    )


# -----------------------------------------------------------------------------
# 3. Trainer-compatible Phase 6 batch stream
# -----------------------------------------------------------------------------

class Phase6BatchStream:
    """
    Yields trivial `(input_ids, target_ids)` dummy tensors that satisfy the
    trainer's per-step contract while exposing the raw `PrefExample` list
    for the SimPO+KTO hook to consume via `self.last_examples`.

    Mirrors `Phase5BatchStream`'s pattern exactly.
    """

    def __init__(
        self,
        tokenizer,
        pairs: Iterator[PrefExample],
        batch_size: int = 2,
        device: str = "cpu",
    ):
        self.tokenizer = tokenizer
        self.pairs = iter(pairs)
        self.batch_size = batch_size
        self.device = device
        self.last_examples: List[PrefExample] = []

    def __iter__(self):
        return self

    def __next__(self):
        self.last_examples = [next(self.pairs) for _ in range(self.batch_size)]
        dummy = torch.zeros((self.batch_size, 1), dtype=torch.long, device=self.device)
        return dummy, dummy
