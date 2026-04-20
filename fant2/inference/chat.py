"""
Multi-turn chat session wrapper.

Builds up a list of {role, content} messages, applies the ChatML template
via FANT2Tokenizer.apply_chat_template, and generates the assistant's reply
via FANT2Generator.

Usage
-----

    chat = ChatSession(generator, system="You are a helpful assistant.")
    reply = chat.send("What is the Apollonian gasket?")
    print(reply)
    reply = chat.send("And what is its Hausdorff dimension?")
    print(reply)

    for msg in chat.history:
        print(msg["role"], ":", msg["content"])
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ..constants import IM_END_ID, EOS_ID
from .generator import FANT2Generator, GenerationConfig


@dataclass
class ChatSession:
    """
    Stateful multi-turn chat wrapper.

    Holds the message history, applies the chat template on each .send(),
    and reuses a single FANT2Generator for sampling.
    """

    generator: FANT2Generator
    system: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)

    # Generation defaults (can be overridden per-.send call)
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    max_new_tokens: int = 512
    greedy: bool = False

    def __post_init__(self):
        if self.system:
            self.history.append({"role": "system", "content": self.system})

    def reset(self) -> None:
        """Clear the conversation history (keeps the system prompt if any)."""
        sys_msgs = [m for m in self.history if m["role"] == "system"]
        self.history = sys_msgs

    def send(
        self,
        user_message: str,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        greedy: Optional[bool] = None,
    ) -> str:
        """
        Send a user message and return the assistant's reply.
        Both messages are appended to self.history.
        """
        self.history.append({"role": "user", "content": user_message})

        prompt_ids = self.generator.tokenizer.apply_chat_template(
            self.history,
            add_generation_prompt=True,
            add_bos=True,
        )
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids[0].tolist()

        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            top_k=top_k if top_k is not None else self.top_k,
            top_p=top_p if top_p is not None else self.top_p,
            greedy=greedy if greedy is not None else self.greedy,
            eos_token_ids=[IM_END_ID, EOS_ID],
        )

        new_ids = self.generator.generate_ids(prompt_ids, cfg)
        reply = self.generator.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        self.history.append({"role": "assistant", "content": reply})
        return reply
