"""
Router-grad preconditioning via empirical Fisher (Brehmer-Cranmer-Kling-Tait-
Plehn, CERN CDS record 2752417, "Better Higgs Measurements through Information
Geometry", JHEP 01 (2019) 181).

The natural-gradient update ``theta_{t+1} = theta_t - eta * F^{-1} * g`` with
the full Fisher ``F = E[grad log p * grad log p^T]`` is infeasible at language-
model scale. The *diagonal empirical* Fisher is the cheap approximation that
still captures the main effect: per-parameter gradient variance.

We apply it **only to the MoE routing parameters** (``megapool_proj`` and
``level_proj`` inside MatryoshkaRouter) because:

1. Routing gradients are the noisiest part of MoE training — the same input
   can hop between experts between steps, producing large gradient magnitudes
   that swamp the trunk's learning signal.
2. Routing parameters are tiny (``dim * n_megapools`` each) so the Fisher
   state is free (<1 MB even at 1B scale).
3. The rest of the model benefits from plain momentum; only the router has
   pathological curvature that a Fisher preconditioner fixes.

Usage inside a training loop:

    fisher_state = {}
    for step, batch in enumerate(loader):
        loss = model(batch).loss
        loss.backward()
        precondition_router_grads_(model, fisher_state)
        optimizer.step()
        optimizer.zero_grad()
"""

from __future__ import annotations
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


def _is_router_param(name: str, patterns: Iterable[str]) -> bool:
    return any(pat in name for pat in patterns)


@torch.no_grad()
def precondition_router_grads_(
    model: nn.Module,
    fisher_state: Dict[str, torch.Tensor],
    beta: float = 0.99,
    eps: float = 1e-5,
    patterns: Optional[Iterable[str]] = None,
) -> int:
    # Mutates grads of all router params in-place:
    #     EMA of g^2 (diagonal empirical Fisher F_diag)
    #     g <- g / (sqrt(F_diag) + eps)
    # Returns the number of params preconditioned (useful for a sanity log).
    if patterns is None:
        patterns = ("megapool_proj", "level_proj")
    touched = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not _is_router_param(name, patterns):
            continue
        g2 = p.grad.detach().pow(2)
        prev = fisher_state.get(name)
        if prev is None or prev.shape != g2.shape:
            prev = torch.zeros_like(g2)
            fisher_state[name] = prev
        prev.mul_(beta).add_(g2, alpha=1.0 - beta)
        p.grad.div_(prev.sqrt().add_(eps))
        touched += 1
    return touched
