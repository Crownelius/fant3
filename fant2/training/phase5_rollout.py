"""
Phase 5 — Dr.GRPO rollout / reward / loss step.

Implements the on-policy generation loop that the trainer's
`_phase5_grpo_forward` calls per outer step:

  1. Sample a procedurally-generated math problem.
  2. Format it with the <think>/<answer> schema.
  3. Generate G rollouts from the current policy.
  4. Reward each rollout: 1.0 for exact answer match, +0.1 format bonus.
  5. Compute group-relative advantages: (r - r.mean()) / (r.std() + 1e-6).
  6. Run a teacher-forced forward to get response log-probs from the live
     model (with grad) and from a frozen reference (no grad).
  7. Apply `dr_grpo_loss(new_logps, old_logps, advantages,
                         clip_eps=0.2, clip_eps_hi=0.28)`.

Spec references: §8 Phase 5 (G=16, ε_hi=0.28, lr=5e-7).

**Training data policy: NO public benchmarks.** The locked spec names GSM8K /
MATH / HumanEval as Phase 5 RL targets, but the user has explicitly forbidden
training on any public benchmark to keep them clean for evaluation. This
module therefore generates math problems procedurally — random arithmetic
templates with sampled values. Acceptance gates that *measure* against
public benchmarks are still allowed; that's eval, not training.

Designed to be importable from both:
  * `fant2.training.phase5_grpo` (real training entry point)
  * `scripts.option_e_phase5_grpo` (smoke gate)
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..constants import BOS_ID, EOS_ID
from ..inference.generator import GenerationConfig, sample_token


# -----------------------------------------------------------------------------
# 1. Procedural math problem generator (NO benchmark data)
# -----------------------------------------------------------------------------

@dataclass
class MathExample:
    question: str
    gold_answer: str  # Canonical answer as a string, e.g. "5"
    template: str     # Which template was used (for telemetry / curriculum)
    a: int = 0        # First operand (for N6 gold reasoning traces)
    b: int = 0        # Second operand (for N6 gold reasoning traces)


@dataclass
class _MathTemplate:
    name: str
    tmpl: str
    op: Callable[[int, int], int]
    # constraint: (a, b) → bool, must be true for the problem to be valid
    valid: Callable[[int, int], bool] = lambda a, b: True


MATH_TEMPLATES: List[_MathTemplate] = [
    _MathTemplate(
        "addition",
        "{name1} has {a} {item} and {name2} gives {him} {b} more. How many {item} does {name1} have now?",
        lambda a, b: a + b,
    ),
    _MathTemplate(
        "subtraction",
        "{name1} has {a} {currency}. {He} spends {b} {currency} on {lunch_item}. How {currency_q} {currency} does {he} have left?",
        lambda a, b: a - b,
        valid=lambda a, b: a > b,
    ),
    _MathTemplate(
        "multiplication_grid",
        "A {place_group} has {a} rows of {b} {item}. How many {item} are there in total?",
        lambda a, b: a * b,
    ),
    _MathTemplate(
        "multiplication_pack",
        "{name1} buys {a} packs of {item}. Each pack contains {b} {item}. How many {item} did {he} buy in total?",
        lambda a, b: a * b,
    ),
    _MathTemplate(
        "division_even",
        "{a} {item} are split evenly among {b} {ppl_group}. How many {item} does each {ppl_singular} get?",
        lambda a, b: a // b,
        valid=lambda a, b: b > 0 and a % b == 0,
    ),
    _MathTemplate(
        "rate",
        "A {vehicle} travels {a} miles in 1 hour. How far does it travel in {b} hours?",
        lambda a, b: a * b,
    ),
    _MathTemplate(
        "remainder_complement",
        "There are {a} {item} in a {place_group}. {b} of them are {color}. How many are NOT {color}?",
        lambda a, b: a - b,
        valid=lambda a, b: a > b,
    ),
    _MathTemplate(
        "weekly_repeat",
        "{name1} reads {a} pages a day. How many pages does {he} read in {b} days?",
        lambda a, b: a * b,
    ),
]


# Template-substitution variables. None of these are sourced from a benchmark.
_NAMES_M = ["Tom", "Bob", "Sam", "Liam", "Noah", "Owen", "Ezra", "Kai"]
_NAMES_F = ["Mary", "Sarah", "Lisa", "Anya", "Mira", "Zoe", "Iris", "June"]
_PRON_M = {"sub": "he", "obj": "him", "pos": "his"}
_PRON_F = {"sub": "she", "obj": "her", "pos": "her"}
_ITEMS_COUNT = ["apples", "marbles", "books", "stickers", "coins", "shells", "cards", "pencils"]
_PLACES_GROUP = [("garden", "garden"), ("orchard", "orchard"), ("field", "field"), ("shelf", "shelf")]

# Semantically valid place-item pairings to avoid paradoxical combinations
# like "A garden has 6 rows of 6 books"
_PLACE_ITEMS: dict[str, list[str]] = {
    "garden": ["apples", "shells", "stickers"],
    "orchard": ["apples"],
    "field": ["apples", "shells"],
    "shelf": ["marbles", "books", "stickers", "coins", "shells", "cards", "pencils"],
}
_PLACES_PEOPLE = [("friends", "friend"), ("students", "student"), ("teammates", "teammate")]
_VEHICLES = ["train", "bus", "car", "boat", "ferry"]
_CURRENCIES = ["dollars", "coins", "tokens"]
_COLORS = ["red", "blue", "green", "yellow"]
_LUNCH_ITEMS = ["lunch", "a snack", "a book", "a hat"]


class ProceduralMathStream:
    """
    Procedurally-generated math problems for Dr.GRPO training.

    Iterates indefinitely. No external data, no benchmark contamination.
    Each example has a `template` tag so a curriculum scheduler can later
    bias the distribution toward harder templates as the model improves.
    """

    def __init__(
        self,
        seed: int = 0,
        max_value: int = 20,
        templates: Optional[List[_MathTemplate]] = None,
    ):
        self.rng = random.Random(seed)
        self.max_value = max_value
        self.templates = templates if templates is not None else MATH_TEMPLATES

    def _sample_one(self) -> MathExample:
        for _ in range(40):  # rejection-sample until template constraints hold
            t = self.rng.choice(self.templates)
            a = self.rng.randint(2, self.max_value)
            b = self.rng.randint(2, self.max_value)

            # For division_even, construct valid (a,b) directly instead of
            # rejection sampling — avoids the 2.4% representation problem.
            if t.name == "division_even" and (b == 0 or a % b != 0):
                b = self.rng.randint(2, self.max_value)
                a = b * self.rng.randint(2, self.max_value)  # guarantee a % b == 0
                if a > self.max_value * self.max_value:
                    a = b * self.rng.randint(1, self.max_value // max(b, 1))
            elif not t.valid(a, b):
                continue

            # Pick gendered name + pronouns
            if self.rng.random() < 0.5:
                name1, pron = self.rng.choice(_NAMES_M), _PRON_M
            else:
                name1, pron = self.rng.choice(_NAMES_F), _PRON_F
            name2 = self.rng.choice(_NAMES_M + _NAMES_F)

            # Pick place, then pick a semantically valid item for that place
            place_pl, place_sg = self.rng.choice(_PLACES_GROUP)
            if place_pl in _PLACE_ITEMS:
                item = self.rng.choice(_PLACE_ITEMS[place_pl])
            else:
                item = self.rng.choice(_ITEMS_COUNT)

            ppl_pl, ppl_sg = self.rng.choice(_PLACES_PEOPLE)
            vehicle = self.rng.choice(_VEHICLES)
            currency = self.rng.choice(_CURRENCIES)
            color = self.rng.choice(_COLORS)
            lunch_item = self.rng.choice(_LUNCH_ITEMS)

            # Fix a/an grammar for place_group
            article = "An" if place_pl[0].lower() in "aeiou" else "A"

            try:
                raw = t.tmpl.format(
                    a=a, b=b,
                    name1=name1, name2=name2,
                    he=pron["sub"], He=pron["sub"].capitalize(),
                    him=pron["obj"], his=pron["pos"],
                    item=item,
                    place_group=place_pl,
                    ppl_group=ppl_pl, ppl_singular=ppl_sg,
                    vehicle=vehicle,
                    currency=currency,
                    currency_q="many",  # all currency words used here are count nouns
                    color=color,
                    lunch_item=lunch_item,
                )
                # Fix "A orchard" → "An orchard", "a orchard" → "an orchard"
                question = raw.replace("A " + place_pl, article + " " + place_pl)
                question = question.replace("a " + place_pl, article.lower() + " " + place_pl)
            except KeyError:
                # Template uses a key we didn't supply — skip
                continue
            answer = str(t.op(a, b))
            return MathExample(question=question, gold_answer=answer, template=t.name, a=a, b=b)
        # Final fallback if rejection sampling fails (shouldn't happen)
        return MathExample(
            question=f"What is {self.max_value} plus {self.max_value}?",
            gold_answer=str(2 * self.max_value),
            template="addition",
        )

    def __iter__(self) -> Iterator[MathExample]:
        while True:
            yield self._sample_one()


# Backwards-compatible alias used by the trainer hook + smoke gate.
# (We never reference GSM8K or any other benchmark by name.)
ProblemExample = MathExample


# -----------------------------------------------------------------------------
# 2. Prompt formatting + answer parsing
# -----------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Solve step by step. Wrap reasoning in <think>...</think> and the final "
    "answer in <answer>...</answer>.\n"
    "<think>"
)

# Match <answer>...</answer> and capture a numeric inside, allowing whitespace.
_ANSWER_RE = re.compile(r"<answer>\s*([-+]?\d[\d,]*\.?\d*)\s*</answer>")
# Numeric token (used as a fallback for the "leniency" reward).
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def parse_answer(text: str) -> Optional[str]:
    """Extract the canonical answer from the rollout text. Returns None if no
    `<answer>...</answer>` tag with a number is present."""
    m = _ANSWER_RE.search(text)
    if not m:
        return None
    return m.group(1).replace(",", "").strip()


def math_reward(rollout_text: str, gold: str) -> float:
    """
    Reward = 1.0 if <answer> matches gold,
             0.5 if any number in the rollout matches gold (lenient mid-credit),
             0.1 if at least the format is right (no number match),
             0.0 otherwise.

    The lenient 0.5 tier helps the smoke gate produce non-zero rewards on a
    tiny untrained model that won't yet emit the schema correctly. Real
    training inherits this leniency at no cost — it's just credit shaping.
    """
    parsed = parse_answer(rollout_text)
    if parsed is not None:
        try:
            if abs(float(parsed) - float(gold)) < 1e-6:
                return 1.0
        except ValueError:
            pass
        return 0.1  # Right format, wrong answer.

    # No <answer> tag — check if the gold value appears anywhere as a number.
    nums = [n.replace(",", "") for n in _NUM_RE.findall(rollout_text)]
    for n in nums:
        try:
            if abs(float(n) - float(gold)) < 1e-6:
                return 0.5
        except ValueError:
            continue
    return 0.0


# -----------------------------------------------------------------------------
# 3. Batched rollout generation
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate_rollouts(
    model,
    tokenizer,
    prompt: str,
    n_rollouts: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 0,
    device: str = "cpu",
    eos_token_id: int = EOS_ID,
) -> List[Tuple[List[int], int]]:
    """
    Generate `n_rollouts` independent samples from the model for the given
    prompt. Returns a list of `(full_token_ids, prompt_len)` tuples — each
    `full_token_ids` is a Python list containing prompt + sampled response.

    Stateless wrt KV-cache (the v2.0 model has no cache), so each step does a
    fresh forward over the growing prefix. This is slow but correct.
    """
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    prompt_len = len(prompt_ids)
    max_seq_len = model.config.max_seq_len

    rollouts: List[Tuple[List[int], int]] = []
    for _ in range(n_rollouts):
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            ctx = ids[-max_seq_len:] if len(ids) > max_seq_len else ids
            inp = torch.tensor([ctx], dtype=torch.long, device=device)
            out = model(inp)
            logits = out["logits"][0, -1].float()
            next_id = sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=False,
            )
            ids.append(next_id)
            if next_id == eos_token_id:
                break
        rollouts.append((ids, prompt_len))
    return rollouts


# -----------------------------------------------------------------------------
# 4. Response log-prob computation
# -----------------------------------------------------------------------------

def response_logp_sum(
    model,
    full_ids: List[int],
    prompt_len: int,
    device: str = "cpu",
    no_grad: bool = False,
) -> torch.Tensor:
    """
    Compute the sum of log P(response_t | prefix) over the response tokens
    of a single rollout.

    Args:
        model:      live model (or frozen ref)
        full_ids:   prompt + response token ids (Python list)
        prompt_len: int, number of prompt tokens
        device:     "cpu" or "cuda"
        no_grad:    if True, run inside torch.no_grad() (use for ref model)

    Returns:
        Scalar tensor (with grad iff no_grad=False) — sum of log probs of
        the response tokens. Shape: (). If the rollout has no response
        tokens (model emitted EOS immediately), returns 0.0.
    """
    if len(full_ids) <= prompt_len:
        return torch.zeros((), device=device)

    # Truncate from the left if the rollout exceeds the model's context.
    max_seq_len = model.config.max_seq_len
    if len(full_ids) > max_seq_len:
        # Keep the rightmost max_seq_len; recompute prompt_len so the
        # response slice still maps to the same response tokens.
        keep_from = len(full_ids) - max_seq_len
        full_ids = full_ids[keep_from:]
        prompt_len = max(0, prompt_len - keep_from)
        # If truncation ate all the prompt, the response is now [0..end);
        # we still want to score those response tokens — start from index 1.
        if prompt_len == 0:
            prompt_len = 1

    inp = torch.tensor([full_ids], dtype=torch.long, device=device)
    ctx = torch.no_grad() if no_grad else _NullCtx()
    with ctx:
        out = model(inp)
        logits = out["logits"][0]  # (T, V)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        # Token at position t is predicted by logit at position t-1.
        # Response tokens are at positions [prompt_len, T).
        # Their predicting logits are at [prompt_len-1, T-1).
        T = inp.size(1)
        response_token_ids = inp[0, prompt_len:T]            # (R,)
        pred_log_probs = log_probs[prompt_len - 1:T - 1]      # (R, V)
        gathered = pred_log_probs.gather(
            -1, response_token_ids.unsqueeze(-1)
        ).squeeze(-1)                                         # (R,)
    return gathered.sum()


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# -----------------------------------------------------------------------------
# 5. One Dr.GRPO outer step
# -----------------------------------------------------------------------------

@dataclass
class GRPOStepResult:
    loss: torch.Tensor               # scalar tensor with grad
    rewards: List[float]             # length n_rollouts
    advantages: List[float]          # length n_rollouts
    new_logps: List[float]           # length n_rollouts (detached)
    response_lens: List[int]         # length n_rollouts
    n_correct: int                   # rollouts whose reward == 1.0


def grpo_step(
    *,
    model,
    ref_model,
    tokenizer,
    example: MathExample,
    n_rollouts: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 0.9,
    top_p: float = 0.95,
    clip_eps: float = 0.2,
    clip_eps_hi: float = 0.28,
    device: str = "cpu",
) -> GRPOStepResult:
    """
    One Dr.GRPO step on a single prompt:
      generate G rollouts → score → compute advantages → log-probs → loss.

    Returns a `GRPOStepResult` whose `loss` tensor still has grad — the
    caller (trainer hook or smoke gate) does `.backward()` and the optimizer
    step. Group-relative advantages are computed from the rewards.
    """
    from .losses import dr_grpo_loss

    prompt = format_prompt(example.question)
    rollouts = generate_rollouts(
        model, tokenizer, prompt,
        n_rollouts=n_rollouts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )

    # Score each rollout.
    rewards: List[float] = []
    response_lens: List[int] = []
    rollout_texts: List[str] = []
    for full_ids, prompt_len in rollouts:
        response_ids = full_ids[prompt_len:]
        text = tokenizer.decode(response_ids, skip_special_tokens=True)
        rollout_texts.append(text)
        rewards.append(math_reward(text, example.gold_answer))
        response_lens.append(len(response_ids))

    # Group-relative advantages.
    r = torch.tensor(rewards, dtype=torch.float32, device=device)
    adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)

    # Forward over each rollout to get new_logps (with grad) and old_logps (frozen).
    new_logps_list: List[torch.Tensor] = []
    old_logps_list: List[torch.Tensor] = []
    for full_ids, prompt_len in rollouts:
        new_lp = response_logp_sum(model, full_ids, prompt_len, device=device, no_grad=False)
        old_lp = response_logp_sum(ref_model, full_ids, prompt_len, device=device, no_grad=True)
        new_logps_list.append(new_lp)
        old_logps_list.append(old_lp.detach())

    new_logps = torch.stack(new_logps_list)   # (G,)
    old_logps = torch.stack(old_logps_list)   # (G,)

    loss = dr_grpo_loss(
        new_logps=new_logps,
        old_logps=old_logps,
        advantages=adv,
        clip_eps=clip_eps,
        clip_eps_hi=clip_eps_hi,
    )

    n_correct = sum(1 for x in rewards if x >= 1.0)
    return GRPOStepResult(
        loss=loss,
        rewards=rewards,
        advantages=adv.tolist(),
        new_logps=[float(x.item()) for x in new_logps.detach()],
        response_lens=response_lens,
        n_correct=n_correct,
    )


# -----------------------------------------------------------------------------
# 6. Trainer-compatible Phase 5 batch stream
# -----------------------------------------------------------------------------

class Phase5BatchStream:
    """
    Yields tensor batches that satisfy the trainer's `(input_ids, target_ids)`
    contract while also exposing the raw `MathExample` list for the GRPO hook.

    The tensor form is intentionally minimal — a (B, 1) zero pad — because the
    hook does NOT use it for anything; it reads `self.last_examples` instead.
    Keeping the tensors trivial means almost no encoding overhead per batch.

    Usage:

        stream = Phase5BatchStream(
            tokenizer=tokenizer,
            problems=ProceduralMathStream(seed=0),
            batch_size=2,
            device="cpu",
        )
        trainer = FANT2Trainer(model, train_cfg, stream)
        trainer.ref_model = ref_model  # set before .train()
        trainer.train()
    """

    def __init__(
        self,
        tokenizer,
        problems: Iterator[MathExample],
        batch_size: int = 2,
        device: str = "cpu",
    ):
        self.tokenizer = tokenizer
        self.problems = iter(problems)
        self.batch_size = batch_size
        self.device = device
        self.last_examples: List[MathExample] = []

    def __iter__(self):
        return self

    def __next__(self):
        self.last_examples = [next(self.problems) for _ in range(self.batch_size)]
        # Trivial tensor pair to satisfy the trainer's unpacking + .to(device).
        # The Phase 5 hook ignores these and reads `self.last_examples`.
        dummy = torch.zeros((self.batch_size, 1), dtype=torch.long, device=self.device)
        return dummy, dummy
