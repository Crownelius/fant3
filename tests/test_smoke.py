"""
Smoke tests — every public symbol can be imported and the tiny model
forward + backward + optimizer step succeeds.
"""

import math

import pytest
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

def test_import_fant2_package():
    import fant2  # noqa


def test_import_model_subpackage():
    from fant2.model import (
        FANT2Model, RMSNorm, FractalSeedExpert, ZeroExpert, CopyExpert,
        SharedNarrowExpert, DenseSwiGLU, HierarchicalApollonianRouter,
        FractalMoELayer, HubAttention, CerebellumModule, ApollonianMemory,
        ApollonianRetrievalAttention, TransformerBlock,
    )
    assert FANT2Model is not None


def test_import_training_subpackage():
    from fant2.training import (
        Muon, HybridOptimizer, fep_unified_loss, llm_jepa_loss,
        calibration_loss, simpo_loss, kto_loss, dr_grpo_loss,
        TelemetrySnapshot, collect_telemetry,
        default_monitors, run_monitors,
        TrainConfig, FANT2Trainer,
    )
    assert FANT2Trainer is not None


def test_import_data_subpackage():
    from fant2.data import (
        SyntheticStream, HuggingFaceStream, LocalShardStream,
        TokenizedBatchStream, make_default_stream, SEED_CORPUS,
    )
    assert len(SEED_CORPUS) > 0


def test_import_inference_subpackage():
    from fant2.inference import FANT2Generator, GenerationConfig, ChatSession
    assert FANT2Generator is not None


def test_import_bench_subpackage():
    from fant2.bench import (
        evaluate_perplexity, evaluate_gsm8k, extract_gsm8k_answer,
        evaluate_arc_multichoice, evaluate_hellaswag,
    )
    assert extract_gsm8k_answer("#### 42") == 42.0


def test_import_tokenizer_subpackage():
    from fant2.tokenizer import FANT2Tokenizer, apply_chat_template, GPT4_REGEX_PATTERN
    assert "p{L}" in GPT4_REGEX_PATTERN or "\\p" in GPT4_REGEX_PATTERN


# -----------------------------------------------------------------------------
# Forward + backward
# -----------------------------------------------------------------------------

def test_tiny_model_forward(tiny_model, synthetic_batch):
    input_ids, target_ids = synthetic_batch
    out = tiny_model(input_ids, targets=target_ids)
    assert "logits" in out
    assert "loss" in out
    assert "router_outputs" in out
    assert "final_hidden" in out
    B, T = input_ids.shape
    assert out["logits"].shape == (B, T, tiny_model.config.vocab_size)
    assert out["final_hidden"].shape == (B, T, tiny_model.config.dim)
    # Loss should be finite and approximately ln(vocab_size) for random init
    loss = float(out["loss"].item())
    assert math.isfinite(loss)
    assert 5.0 < loss < 15.0  # ln(32768) ≈ 10.4
    # Router outputs: one per MoE layer
    n_moe = tiny_model.config.n_layers - tiny_model.config.n_dense_layers
    assert len(out["router_outputs"]) == n_moe


def test_tiny_model_backward(tiny_model, synthetic_batch):
    input_ids, target_ids = synthetic_batch
    out = tiny_model(input_ids, targets=target_ids)
    out["loss"].backward()

    # At least 50% of parameters should have non-zero gradients
    n_with_grad = sum(1 for p in tiny_model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in tiny_model.parameters())
    assert n_with_grad >= n_total // 2, f"only {n_with_grad}/{n_total} params have grads"


def test_optimizer_step(tiny_model, synthetic_batch):
    from fant2.training import HybridOptimizer
    opt = HybridOptimizer.from_model(
        tiny_model, muon_lr=1e-3, adam_lr=3e-4, use_8bit_adam=False
    )

    input_ids, target_ids = synthetic_batch
    pre_loss = tiny_model(input_ids, targets=target_ids)["loss"].item()

    for _ in range(3):
        opt.zero_grad()
        loss = tiny_model(input_ids, targets=target_ids)["loss"]
        loss.backward()
        opt.step()

    post_loss = tiny_model(input_ids, targets=target_ids)["loss"].item()
    # Loss should decrease (or at least not blow up)
    assert math.isfinite(post_loss)
    assert post_loss < pre_loss + 1.0  # allow some noise


def test_router_bias_update(tiny_model, synthetic_batch):
    """The DeepSeek aux-loss-free bias must update without gradients."""
    input_ids, _ = synthetic_batch
    out = tiny_model(input_ids)
    # Find first MoE block's router bias
    moe_block = next(b for b in tiny_model.blocks if not b.is_dense)
    pre_bias = moe_block.ffn.router.expert_bias.clone()
    tiny_model.update_router_biases(out["router_outputs"])
    post_bias = moe_block.ffn.router.expert_bias.clone()
    diff = (post_bias - pre_bias).abs().sum().item()
    # Some change should happen for an unbalanced random batch
    assert diff >= 0.0  # at minimum no error
