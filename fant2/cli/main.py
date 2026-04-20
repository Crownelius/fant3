"""
The unified `fant2` CLI entry point.

Usage:
    fant2 <subcommand> [args...]

If the subcommand is one of the 7 `train-phaseN` commands, the remaining
args are passed through to the corresponding phaseN_*.py script. Otherwise
a short dispatcher in this module handles it.
"""

import argparse
import os
import sys
from typing import List, Optional

from ..config import fant2_default, fant2_tiny
from ..model import FANT2Model
from ..tokenizer import FANT2Tokenizer


# -----------------------------------------------------------------------------
# Subcommand table
# -----------------------------------------------------------------------------

# (command name, description, dispatcher function or None for a passthrough to a module.main())
_PHASE_MODULES = {
    "train-phase0": "fant2.training.phase0_bpe",
    "train-phase1": "fant2.training.phase1_jepa",
    "train-phase2": "fant2.training.phase2_moe",
    "train-phase3": "fant2.training.phase3_calibrate",
    "train-phase4": "fant2.training.phase4_refine",
    "train-phase5": "fant2.training.phase5_grpo",
    "train-phase6": "fant2.training.phase6_simpo_kto",
}


# -----------------------------------------------------------------------------
# Helper: load model + tokenizer from args (shared by generate/chat/eval)
# -----------------------------------------------------------------------------

def _load_model_and_tokenizer(
    tokenizer_path: str,
    checkpoint_path: Optional[str],
    preset: str,
    device: str,
):
    import torch
    tok = FANT2Tokenizer.load(tokenizer_path)
    cfg = fant2_tiny() if preset == "tiny" else fant2_default()
    model = FANT2Model(cfg)
    if checkpoint_path:
        # weights_only=False because the checkpoint also contains a TrainConfig
        # dataclass; we trust our own checkpoints.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  loaded model from {checkpoint_path}")
    model = model.to(device)
    model.eval()
    return model, tok, cfg


# -----------------------------------------------------------------------------
# Subcommand: info
# -----------------------------------------------------------------------------

def cmd_info(args: List[str]) -> int:
    p = argparse.ArgumentParser(prog="fant2 info")
    p.add_argument("--preset", choices=["tiny", "default"], default="default")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    ns = p.parse_args(args)

    cfg = fant2_tiny() if ns.preset == "tiny" else fant2_default()
    print(cfg.summary())
    print()
    model = FANT2Model(cfg)
    print(model.parameter_summary())

    if ns.tokenizer:
        if os.path.exists(ns.tokenizer):
            tok = FANT2Tokenizer.load(ns.tokenizer)
            print(f"\nTokenizer: {ns.tokenizer}")
            print(f"  vocab_size: {tok.vocab_size:,}")
        else:
            print(f"\nTokenizer: {ns.tokenizer} (NOT FOUND)")
    return 0


# -----------------------------------------------------------------------------
# Subcommand: generate
# -----------------------------------------------------------------------------

def cmd_generate(args: List[str]) -> int:
    import torch
    from ..inference import FANT2Generator

    p = argparse.ArgumentParser(prog="fant2 generate")
    p.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--preset", choices=["tiny", "default"], default="default")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--greedy", action="store_true")
    ns = p.parse_args(args)

    model, tok, _ = _load_model_and_tokenizer(
        ns.tokenizer, ns.checkpoint, ns.preset, ns.device
    )
    gen = FANT2Generator(model, tok, device=ns.device)
    out = gen.generate(
        ns.prompt,
        max_new_tokens=ns.max_new_tokens,
        temperature=ns.temperature,
        top_k=ns.top_k,
        top_p=ns.top_p,
        greedy=ns.greedy,
    )
    print(out)
    return 0


# -----------------------------------------------------------------------------
# Subcommand: chat
# -----------------------------------------------------------------------------

def cmd_chat(args: List[str]) -> int:
    import torch
    from ..inference import FANT2Generator, ChatSession

    p = argparse.ArgumentParser(prog="fant2 chat")
    p.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--preset", choices=["tiny", "default"], default="default")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--system", type=str,
                   default="You are a helpful, harmless, and honest assistant.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    ns = p.parse_args(args)

    model, tok, _ = _load_model_and_tokenizer(
        ns.tokenizer, ns.checkpoint, ns.preset, ns.device
    )
    gen = FANT2Generator(model, tok, device=ns.device)
    chat = ChatSession(
        generator=gen,
        system=ns.system,
        temperature=ns.temperature,
        max_new_tokens=ns.max_new_tokens,
    )

    print(f"fant2 chat — type 'exit' or Ctrl+C to quit")
    print(f"system: {ns.system}")
    print()
    while True:
        try:
            user = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit", "bye"):
            break
        reply = chat.send(user)
        print(f"assistant> {reply}")
        print()
    return 0


# -----------------------------------------------------------------------------
# Subcommand: eval-ppl
# -----------------------------------------------------------------------------

def cmd_eval_ppl(args: List[str]) -> int:
    import torch
    from ..bench import evaluate_perplexity
    from ..data import HuggingFaceStream, SyntheticStream, TokenizedBatchStream

    p = argparse.ArgumentParser(prog="fant2 eval-ppl")
    p.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--preset", choices=["tiny", "default"], default="default")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use-hf", action="store_true")
    p.add_argument("--hf-dataset", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-batches", type=int, default=100)
    ns = p.parse_args(args)

    model, tok, _ = _load_model_and_tokenizer(
        ns.tokenizer, ns.checkpoint, ns.preset, ns.device
    )

    text_stream = HuggingFaceStream(dataset_name=ns.hf_dataset) if ns.use_hf else SyntheticStream()
    stream = TokenizedBatchStream(
        text_stream=text_stream,
        tokenizer=tok,
        batch_size=ns.batch_size,
        seq_len=ns.seq_len,
        device=ns.device,
    )

    result = evaluate_perplexity(model, stream, max_batches=ns.max_batches)
    print(f"loss:       {result['loss']:.4f}")
    print(f"perplexity: {result['perplexity']:.3f}")
    print(f"n_tokens:   {result['n_tokens']:,}")
    return 0


# -----------------------------------------------------------------------------
# Subcommand: eval-gsm8k
# -----------------------------------------------------------------------------

def cmd_eval_gsm8k(args: List[str]) -> int:
    import torch
    from ..inference import FANT2Generator
    from ..bench import evaluate_gsm8k

    p = argparse.ArgumentParser(prog="fant2 eval-gsm8k")
    p.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--preset", choices=["tiny", "default"], default="default")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-problems", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)  # greedy
    ns = p.parse_args(args)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` package required for GSM8K eval.")
        return 1

    model, tok, _ = _load_model_and_tokenizer(
        ns.tokenizer, ns.checkpoint, ns.preset, ns.device
    )
    gen = FANT2Generator(model, tok, device=ns.device)
    print(f"Loading GSM8K test split...")
    gsm8k = load_dataset("gsm8k", "main", split="test")

    result = evaluate_gsm8k(
        gen, gsm8k,
        max_problems=ns.max_problems,
        max_new_tokens=ns.max_new_tokens,
        temperature=ns.temperature,
    )
    print(f"\nGSM8K: {result['correct']}/{result['total']} = {result['accuracy']:.1%}")
    return 0


# -----------------------------------------------------------------------------
# Subcommand: eval-arc / eval-hellaswag
# -----------------------------------------------------------------------------

def cmd_eval_arc(args: List[str]) -> int:
    import torch
    from ..bench import evaluate_arc_multichoice

    p = argparse.ArgumentParser(prog="fant2 eval-arc")
    p.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--preset", choices=["tiny", "default"], default="default")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--subset", choices=["ARC-Easy", "ARC-Challenge"], default="ARC-Easy")
    p.add_argument("--max-problems", type=int, default=500)
    ns = p.parse_args(args)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` package required for ARC eval.")
        return 1

    model, tok, _ = _load_model_and_tokenizer(
        ns.tokenizer, ns.checkpoint, ns.preset, ns.device
    )
    print(f"Loading ai2_arc/{ns.subset}/test ...")
    arc = load_dataset("ai2_arc", ns.subset, split="test")
    result = evaluate_arc_multichoice(model, tok, arc, max_problems=ns.max_problems)
    print(f"\n{ns.subset}: {result['correct']}/{result['total']} = {result['accuracy']:.1%}")
    return 0


def cmd_eval_hellaswag(args: List[str]) -> int:
    import torch
    from ..bench import evaluate_hellaswag

    p = argparse.ArgumentParser(prog="fant2 eval-hellaswag")
    p.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--preset", choices=["tiny", "default"], default="default")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-problems", type=int, default=500)
    ns = p.parse_args(args)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` package required for HellaSwag eval.")
        return 1

    model, tok, _ = _load_model_and_tokenizer(
        ns.tokenizer, ns.checkpoint, ns.preset, ns.device
    )
    print(f"Loading hellaswag/validation ...")
    hs = load_dataset("hellaswag", split="validation")
    result = evaluate_hellaswag(model, tok, hs, max_problems=ns.max_problems)
    print(f"\nHellaSwag: {result['correct']}/{result['total']} = {result['accuracy']:.1%}")
    return 0


# -----------------------------------------------------------------------------
# Top-level dispatcher
# -----------------------------------------------------------------------------

_COMMANDS = {
    "info":            cmd_info,
    "generate":        cmd_generate,
    "chat":            cmd_chat,
    "eval-ppl":        cmd_eval_ppl,
    "eval-gsm8k":      cmd_eval_gsm8k,
    "eval-arc":        cmd_eval_arc,
    "eval-hellaswag":  cmd_eval_hellaswag,
}


def _print_help() -> None:
    print("fant2 — FANT 2 Fractal Atomic Neural Topology v2")
    print()
    print("Usage: fant2 <subcommand> [args...]")
    print()
    print("Training phases (thin wrappers over FANT2Trainer):")
    for cmd in _PHASE_MODULES:
        print(f"  {cmd}")
    print()
    print("Inference and evaluation:")
    for cmd in _COMMANDS:
        print(f"  {cmd}")
    print()
    print("Run `fant2 <subcommand> --help` for subcommand-specific options.")


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        return 0

    subcommand = argv[0]
    rest = argv[1:]

    # Phase scripts: re-exec the module's main() with the remaining argv
    if subcommand in _PHASE_MODULES:
        module_name = _PHASE_MODULES[subcommand]
        import importlib
        module = importlib.import_module(module_name)
        # Temporarily replace sys.argv so the phase's argparse sees only its args
        orig_argv = sys.argv
        try:
            sys.argv = [subcommand] + rest
            return int(module.main() or 0)
        finally:
            sys.argv = orig_argv

    if subcommand in _COMMANDS:
        return _COMMANDS[subcommand](rest)

    print(f"Unknown subcommand: {subcommand!r}")
    _print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
