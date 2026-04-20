"""
FANT 2 text generation (greedy / top-k / top-p / temperature sampling).

The generator is stateless on top of FANT2Model — it just wraps `forward()`
in a loop. The model itself is call-by-value on the prefix (no KV cache in
v2.0; to be added in a follow-up with a dedicated `forward_with_kv_cache`).

Usage
-----

    tok   = FANT2Tokenizer.load("data/tokenizer.json")
    model = FANT2Model(cfg)
    model.load_state_dict(ckpt["model"])
    gen = FANT2Generator(model, tok)

    # Simple completion
    out = gen.generate("The Apollonian gasket is ", max_new_tokens=64)
    print(out)

    # With sampling hyperparameters
    out = gen.generate(
        "Write a Python function: ",
        max_new_tokens=128,
        temperature=0.8, top_p=0.9, top_k=50,
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Iterable

import torch
import torch.nn.functional as F

from ..constants import BOS_ID, EOS_ID, PAD_ID
from ..model import FANT2Model
from ..tokenizer import FANT2Tokenizer


# -----------------------------------------------------------------------------
# Sampling utilities
# -----------------------------------------------------------------------------

def top_k_top_p_filter(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("inf"),
) -> torch.Tensor:
    """
    Apply top-k and/or top-p (nucleus) filtering to the final-position logits.

    Args:
        logits:  (vocab_size,) raw logits for the next token
        top_k:   if > 0, keep only the top k tokens
        top_p:   if < 1.0, keep the smallest set whose cumulative prob >= top_p
        filter_value: what to replace filtered logits with (default: -inf)

    Returns:
        The filtered logits (same shape).
    """
    logits = logits.clone()
    # top-k
    if top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))
        kth_value = torch.topk(logits, top_k)[0][..., -1, None]
        logits[logits < kth_value] = filter_value
    # top-p (nucleus)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Mask = True for tokens to REMOVE (cumprob already covers top_p)
        mask = cumprobs > top_p
        # Shift right by one so we always keep at least the single top token
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        indices_to_remove = sorted_indices[mask]
        logits[indices_to_remove] = filter_value
    return logits


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
) -> int:
    """
    Sample one token id from the (vocab_size,) logits.

    If greedy=True, returns argmax and ignores temperature/top_k/top_p.
    """
    if greedy:
        return int(logits.argmax().item())
    if temperature <= 0.0:
        return int(logits.argmax().item())

    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# -----------------------------------------------------------------------------
# Generator
# -----------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    greedy: bool = False
    # Stop tokens (any match → halt)
    eos_token_ids: List[int] = field(default_factory=lambda: [EOS_ID])
    # Repetition penalty (CTRL, Keskar 2019)
    repetition_penalty: float = 1.0
    # Minimum new tokens before EOS is honored
    min_new_tokens: int = 0


class FANT2Generator:
    """
    Stateless text generator on top of a FANT2Model.

    v2.0 does NOT implement a KV cache — every `generate_token` call re-runs
    the full model on the current prefix. This is fine for short prompts and
    small models, but should be replaced with a proper KV-cached forward for
    serving larger models.
    """

    def __init__(
        self,
        model: FANT2Model,
        tokenizer: FANT2Tokenizer,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            device = next(model.parameters()).device
        self.device = device
        self.model.eval()

    # -------------------------------------------------------------------------
    # Low-level: generate raw token ids
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def generate_ids(
        self,
        prompt_ids: List[int],
        cfg: Optional[GenerationConfig] = None,
    ) -> List[int]:
        """
        Generate token ids starting from `prompt_ids`.

        Returns the NEW token ids only (not including the prompt).
        """
        if cfg is None:
            cfg = GenerationConfig()

        ids = list(prompt_ids)
        generated: List[int] = []
        max_len = self.model.config.max_seq_len

        for step in range(cfg.max_new_tokens):
            # Keep the last max_seq_len tokens as the model's context
            if len(ids) > max_len:
                ctx = ids[-max_len:]
            else:
                ctx = ids

            input_tensor = torch.tensor([ctx], dtype=torch.long, device=self.device)
            out = self.model(input_tensor)
            logits = out["logits"][0, -1]  # (vocab_size,)

            # Repetition penalty
            if cfg.repetition_penalty != 1.0:
                for tid in set(ids):
                    v = logits[tid]
                    logits[tid] = v / cfg.repetition_penalty if v > 0 else v * cfg.repetition_penalty

            next_id = sample_token(
                logits,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                greedy=cfg.greedy,
            )

            # EOS handling
            if next_id in cfg.eos_token_ids and step >= cfg.min_new_tokens:
                break

            ids.append(next_id)
            generated.append(next_id)

        return generated

    # -------------------------------------------------------------------------
    # High-level: generate text strings
    # -------------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        greedy: bool = False,
        add_bos: bool = True,
        return_full_text: bool = True,
    ) -> str:
        """
        Generate a completion for a raw text prompt.

        Args:
            return_full_text: if True, returns prompt + completion; else just completion
        """
        prompt_ids = self.tokenizer.encode(prompt, add_bos=add_bos, add_eos=False)
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )
        new_ids = self.generate_ids(prompt_ids, cfg)

        if return_full_text:
            completion = self.tokenizer.decode(prompt_ids + new_ids, skip_special_tokens=True)
        else:
            completion = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return completion

    # -------------------------------------------------------------------------
    # Streaming (yield tokens as they're generated)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        greedy: bool = False,
        add_bos: bool = True,
    ) -> Iterable[str]:
        """
        Yield decoded text chunks as tokens are generated.

        Note: since BPE tokens do not have a 1-to-1 mapping to characters,
        we decode incrementally by keeping a running id list and diffing
        the string output.
        """
        prompt_ids = self.tokenizer.encode(prompt, add_bos=add_bos, add_eos=False)
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )

        ids = list(prompt_ids)
        max_len = self.model.config.max_seq_len
        prev_text = self.tokenizer.decode(ids, skip_special_tokens=True)

        for step in range(cfg.max_new_tokens):
            ctx = ids[-max_len:] if len(ids) > max_len else ids
            input_tensor = torch.tensor([ctx], dtype=torch.long, device=self.device)
            out = self.model(input_tensor)
            logits = out["logits"][0, -1]
            next_id = sample_token(
                logits,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                greedy=cfg.greedy,
            )
            if next_id in cfg.eos_token_ids:
                break
            ids.append(next_id)
            new_text = self.tokenizer.decode(ids, skip_special_tokens=True)
            delta = new_text[len(prev_text):]
            prev_text = new_text
            if delta:
                yield delta
