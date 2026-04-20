"""
Chat template formatter for FANT 2.

The format is ChatML-compatible (the same one used by Qwen, ChatGPT-4o,
DeepSeek-V3, Mistral, and most modern instruct models). Each message is wrapped
with role + content delimiters, and an explicit think/answer split is supported
for the Phase 5 GRPO reasoning fine-tune.

Format:
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_message}<|im_end|>

For reasoning rollouts (Phase 5):
    <|im_start|>assistant
    <|think|>
    {chain of thought}
    <|/think|>
    <|answer|>
    {final answer}
    <|/answer|><|im_end|>
"""

from typing import List, Dict, Optional

from ..constants import SPECIAL_TOKENS


# String form of the special tokens, for use in formatted text
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK = "<|think|>"
THINK_END = "<|/think|>"
ANSWER = "<|answer|>"
ANSWER_END = "<|/answer|>"
BOS = "<|bos|>"
EOS = "<|eos|>"

ROLE_NAMES = {"system", "user", "assistant", "tool"}


def format_message(role: str, content: str) -> str:
    """Format a single message in ChatML."""
    if role not in ROLE_NAMES:
        raise ValueError(f"Unknown role {role!r}; must be one of {ROLE_NAMES}")
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def format_assistant_reasoning(thought: str, answer: str) -> str:
    """Format an assistant turn with explicit think+answer split (Phase 5 GRPO)."""
    return (
        f"{IM_START}assistant\n"
        f"{THINK}\n{thought}\n{THINK_END}\n"
        f"{ANSWER}\n{answer}\n{ANSWER_END}{IM_END}\n"
    )


def apply_chat_template(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
    add_bos: bool = True,
) -> str:
    """
    Apply the chat template to a list of messages.

    Args:
        messages: list of dicts with keys "role" and "content"
        add_generation_prompt: if True, append "<|im_start|>assistant\\n"
                                to prompt the model to generate
        add_bos: if True, prepend the BOS token

    Returns:
        the formatted string ready to be tokenized

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "What is 2+2?"},
        ... ]
        >>> apply_chat_template(messages, add_generation_prompt=True)
        '<|bos|><|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n<|im_start|>user\\nWhat is 2+2?<|im_end|>\\n<|im_start|>assistant\\n'
    """
    parts = []
    if add_bos:
        parts.append(BOS)
    for msg in messages:
        parts.append(format_message(msg["role"], msg["content"]))
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def split_thought_and_answer(text: str) -> Optional[Dict[str, str]]:
    """
    Inverse helper: parse a formatted assistant turn back into thought/answer.

    Returns None if the text does not contain the expected reasoning markers.
    Used by the Phase 5 GRPO trainer to extract the answer for the reward function.
    """
    if THINK not in text or THINK_END not in text:
        return None
    if ANSWER not in text or ANSWER_END not in text:
        return None
    thought_start = text.index(THINK) + len(THINK)
    thought_end = text.index(THINK_END)
    ans_start = text.index(ANSWER) + len(ANSWER)
    ans_end = text.index(ANSWER_END)
    return {
        "thought": text[thought_start:thought_end].strip(),
        "answer":  text[ans_start:ans_end].strip(),
    }
