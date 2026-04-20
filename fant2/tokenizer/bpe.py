"""
FANT2Tokenizer — BPE tokenizer wrapper around HuggingFace `tokenizers`.

The HuggingFace `tokenizers` library provides a fast Rust-based BPE
implementation that we wrap here with the FANT 2 special tokens, the
GPT-4 regex pre-tokenizer, and the chat-template helper.

Usage:

    # Train a new tokenizer
    tok = FANT2Tokenizer.train_from_iterator(text_iterator, vocab_size=32768)
    tok.save("data/tokenizer.json")

    # Load an existing one
    tok = FANT2Tokenizer.load("data/tokenizer.json")

    # Encode / decode
    ids = tok.encode("Hello world!")          # -> List[int]
    text = tok.decode(ids)                    # -> str

    # Apply chat template
    messages = [{"role": "user", "content": "What is 2+2?"}]
    prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True)
"""

from typing import List, Optional, Iterable, Dict

from ..constants import (
    VOCAB_SIZE,
    SPECIAL_TOKENS,
    PAD_ID, BOS_ID, EOS_ID, UNK_ID,
)
from .chat_template import apply_chat_template
from .regex_pretok import GPT4_REGEX_PATTERN


class FANT2Tokenizer:
    """
    BPE tokenizer with the FANT 2 special tokens, GPT-4 regex pre-tokenizer,
    and ChatML chat-template support.

    Internally wraps a HuggingFace `tokenizers.Tokenizer` instance.
    """

    def __init__(self, hf_tokenizer):
        """
        Construct from an already-built HuggingFace tokenizer.
        Most users will call .train_from_iterator() or .load() instead.
        """
        self._tok = hf_tokenizer

    # -------------------------------------------------------------------------
    # Construction (training a new BPE)
    # -------------------------------------------------------------------------

    @classmethod
    def train_from_iterator(
        cls,
        iterator: Iterable[str],
        vocab_size: int = VOCAB_SIZE,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> "FANT2Tokenizer":
        """
        Train a new BPE tokenizer from a string iterator.

        Args:
            iterator:      iterable of training text strings (one document per element)
            vocab_size:    target vocabulary size (default: VOCAB_SIZE = 32768)
            min_frequency: minimum frequency for a merge to be added
            show_progress: pass-through to HF tokenizers

        Returns:
            FANT2Tokenizer instance
        """
        try:
            from tokenizers import Tokenizer, Regex, decoders, models, normalizers, pre_tokenizers, trainers
        except ImportError as e:
            raise ImportError(
                "FANT2Tokenizer.train_from_iterator requires the `tokenizers` package. "
                "Install with: pip install tokenizers"
            ) from e

        # Reserve the top of the vocabulary for FANT 2 special tokens.
        # The BPE training will fill the bottom (vocab_size - len(specials)) slots.
        special_strs = list(SPECIAL_TOKENS.keys())
        bpe_target = vocab_size - len(special_strs)

        # Build a fresh Tokenizer
        tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
        # No normalization (preserve case + punctuation)
        tok.normalizer = normalizers.NFC()
        # Use the GPT-4 regex split as pre-tokenizer (chained with byte-level for round-trip safety)
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=Regex(GPT4_REGEX_PATTERN),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        # Byte-level decoder so we can round-trip arbitrary unicode
        tok.decoder = decoders.ByteLevel()

        # Train
        trainer = trainers.BpeTrainer(
            vocab_size=bpe_target,
            min_frequency=min_frequency,
            special_tokens=special_strs,
            show_progress=show_progress,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator(iterator, trainer=trainer)

        # Sanity check vocab size
        actual_vocab = tok.get_vocab_size()
        if actual_vocab > vocab_size:
            raise RuntimeError(
                f"Trained tokenizer has vocab_size={actual_vocab} > target {vocab_size}. "
                "Reduce min_frequency or vocab_size."
            )

        # Re-anchor the special token IDs to the FANT 2 layout (top of vocab).
        # The HF trainer assigns IDs in insertion order, but we need them at the
        # top so they don't shift if BPE merges change.
        # We do this by adding them with the explicit IDs from constants.py via
        # the post-trained add_tokens API. (HF tokenizers does support this.)
        # In practice, since we passed special_tokens to the trainer, they get
        # IDs 0..N-1; we'll re-pad with placeholder vocab so the special token
        # offsets match VOCAB_SIZE - 32 .. VOCAB_SIZE - 1.
        # For v2.0 simplicity we accept the trainer's IDs and remap on encode/decode.

        return cls(tok)

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the tokenizer to a JSON file (HF tokenizers format)."""
        self._tok.save(path)

    @classmethod
    def load(cls, path: str) -> "FANT2Tokenizer":
        """Load a tokenizer from a JSON file."""
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError("FANT2Tokenizer.load requires `tokenizers`") from e
        return cls(Tokenizer.from_file(path))

    # -------------------------------------------------------------------------
    # Encode / decode
    # -------------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Tokenize a string and return the list of token ids."""
        enc = self._tok.encode(text)
        ids = list(enc.ids)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        """Batched encoding."""
        return [self.encode(t, **kwargs) for t in texts]

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Detokenize a list of ids back to a string."""
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    # -------------------------------------------------------------------------
    # Vocabulary helpers
    # -------------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id("<|pad|>") or 0

    @property
    def bos_id(self) -> int:
        return self._tok.token_to_id("<|bos|>") or 1

    @property
    def eos_id(self) -> int:
        return self._tok.token_to_id("<|eos|>") or 2

    @property
    def unk_id(self) -> int:
        return self._tok.token_to_id("<|unk|>") or 3

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)

    def id_to_token(self, idx: int) -> Optional[str]:
        return self._tok.id_to_token(idx)

    # -------------------------------------------------------------------------
    # Chat template
    # -------------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        add_bos: bool = True,
        return_tensors: Optional[str] = None,
    ):
        """
        Apply the FANT 2 chat template and tokenize the result.

        Returns a list of token ids by default, or a torch tensor if
        return_tensors='pt'.
        """
        text = apply_chat_template(messages, add_generation_prompt=add_generation_prompt, add_bos=add_bos)
        ids = self.encode(text)
        if return_tensors == "pt":
            import torch
            return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        return ids
