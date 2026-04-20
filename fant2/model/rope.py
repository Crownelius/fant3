"""
Partial RoPE — Rotary Position Embedding applied to only a fraction of the head dimension.

From Phi-4-Mini (arXiv 2503.01743): rotating only 25% of head_dim gives cleaner gradients
and easier length extrapolation than full RoPE.

The remaining 75% of head_dim is treated as "no positional information" by attention.
This split is what makes YaRN extrapolation cheap to fine-tune later.
"""

from typing import Tuple

import torch


def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    rope_partial: float = 0.25,
) -> torch.Tensor:
    """
    Precompute the complex exponentials e^{iθ_k m} for partial RoPE.

    Returns a tensor of shape (max_seq_len, rope_dim // 2) where
    rope_dim = int(head_dim * rope_partial), as a complex64 tensor.
    """
    rope_dim = int(head_dim * rope_partial)
    # rope_dim must be even (we pair adjacent dims for the rotation)
    rope_dim = rope_dim - (rope_dim % 2)
    if rope_dim <= 0:
        raise ValueError(f"head_dim={head_dim} * rope_partial={rope_partial} → rope_dim={rope_dim} (must be > 0)")

    # frequencies: 1 / theta^(2k/rope_dim) for k = 0, 1, ..., rope_dim/2 - 1
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [T, rope_dim/2]
    # complex exponential e^{i * freqs}
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [T, rope_dim/2] complex64
    return freqs_cis


def apply_rotary_emb_partial(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_partial: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply partial RoPE to query and key tensors.

    Args:
        xq: query tensor [B, T, n_heads, head_dim]
        xk: key tensor   [B, T, n_kv_heads, head_dim]
        freqs_cis: precomputed [T, rope_dim/2] complex tensor
        rope_partial: fraction of head_dim to rotate (default 0.25)

    Returns:
        (xq_rotated, xk_rotated) with same shapes as input.
    """
    B, T, Hq, D = xq.shape
    rope_dim = int(D * rope_partial)
    rope_dim = rope_dim - (rope_dim % 2)

    if rope_dim <= 0:
        return xq, xk

    # Split each head into [rotated_part, untouched_part]
    xq_rope, xq_pass = xq[..., :rope_dim], xq[..., rope_dim:]
    xk_rope, xk_pass = xk[..., :rope_dim], xk[..., rope_dim:]

    # Cast to float32 for the complex math (RoPE requires it)
    xq_rope_f32 = xq_rope.float().reshape(B, T, Hq, rope_dim // 2, 2)
    xk_rope_f32 = xk_rope.float().reshape(B, T, xk_rope.shape[2], rope_dim // 2, 2)

    # View as complex (last dim = real/imag pair)
    xq_complex = torch.view_as_complex(xq_rope_f32)  # [B, T, Hq, rope_dim/2]
    xk_complex = torch.view_as_complex(xk_rope_f32)

    # freqs_cis is [T, rope_dim/2]; broadcast to [1, T, 1, rope_dim/2]
    freqs_cis_b = freqs_cis[:T].view(1, T, 1, -1).to(xq_complex.device)

    xq_rotated = torch.view_as_real(xq_complex * freqs_cis_b).reshape(B, T, Hq, rope_dim)
    xk_rotated = torch.view_as_real(xk_complex * freqs_cis_b).reshape(B, T, xk_rope.shape[2], rope_dim)

    # Cast back to original dtype and re-concat with the un-rotated part
    xq_out = torch.cat([xq_rotated.to(xq.dtype), xq_pass], dim=-1)
    xk_out = torch.cat([xk_rotated.to(xk.dtype), xk_pass], dim=-1)

    return xq_out, xk_out
