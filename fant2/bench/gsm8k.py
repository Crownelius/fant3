"""
GSM8K math word problem evaluation.

GSM8K (Cobbe et al. 2021) is a set of 8.5K grade-school math word problems.
Each problem has a chain-of-thought solution ending in a line like:

    #### 42

We prompt the model with the problem, generate a chain-of-thought,
extract the number after the `####` marker (or the last number in the
output if no marker), and compare against the ground truth.

Usage
-----

    from datasets import load_dataset
    from fant2.bench import evaluate_gsm8k

    gsm8k = load_dataset("gsm8k", "main", split="test")
    result = evaluate_gsm8k(generator, gsm8k, max_problems=200)
    print(f"GSM8K accuracy: {result['accuracy']:.1%}")
"""

import re
from typing import Dict, Iterable, Optional


# Regex for finding the final `#### 42` marker
GSM8K_ANSWER_PATTERN = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")
# Fallback: any number in the last line
LAST_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_gsm8k_answer(text: str) -> Optional[float]:
    """
    Extract the final numeric answer from a GSM8K solution string.

    Tries `#### N` first, then the last number in the text, then returns None.
    """
    m = GSM8K_ANSWER_PATTERN.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    nums = LAST_NUMBER_PATTERN.findall(text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def _format_gsm8k_prompt(question: str) -> str:
    """Default GSM8K prompt template: question + 'Let's think step by step.'"""
    return (
        f"Question: {question}\n"
        f"Let's think step by step.\n"
    )


def evaluate_gsm8k(
    generator,
    dataset: Iterable,
    max_problems: Optional[int] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,   # greedy by default
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a FANT2Generator on GSM8K.

    Args:
        generator:    FANT2Generator (or any object with .generate(prompt, ...) returning a str)
        dataset:      an iterable that yields dicts with "question" and "answer" keys
                      (the HuggingFace `gsm8k` datasets format)
        max_problems: optional cap on the number of problems evaluated
        max_new_tokens: how long to let the model think
        temperature:  0.0 = greedy decoding
        verbose:      print progress every 20 problems

    Returns:
        dict with "correct", "total", "accuracy"
    """
    correct = 0
    total = 0

    for i, ex in enumerate(dataset):
        if max_problems is not None and i >= max_problems:
            break
        question = ex["question"]
        gt_text = ex["answer"]
        gt_num = extract_gsm8k_answer(gt_text)
        if gt_num is None:
            continue  # skip malformed

        prompt = _format_gsm8k_prompt(question)
        completion = generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            greedy=(temperature == 0.0),
            return_full_text=False,
        )
        pred_num = extract_gsm8k_answer(completion)

        total += 1
        if pred_num is not None and abs(pred_num - gt_num) < 1e-6:
            correct += 1

        if verbose and (i + 1) % 20 == 0:
            acc = correct / max(total, 1)
            print(f"    [gsm8k {i+1}] acc={acc:.3f} ({correct}/{total})")

    return {
        "correct":  correct,
        "total":    total,
        "accuracy": correct / max(total, 1),
    }
