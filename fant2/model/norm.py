"""
RMSNorm — root mean square layer normalization.

Used throughout FANT 2 instead of LayerNorm for the standard reasons:
- one fewer parameter (no bias / mean subtraction)
- faster (no mean computation)
- empirically equivalent or better at this scale (Llama, Mistral, DeepSeek)
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square layer normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always compute the norm in float32 for numerical stability,
        # then cast back to the input dtype (bf16 friendly).
        out = self._norm(x.float()).type_as(x)
        return out * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"
