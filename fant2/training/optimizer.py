"""
Muon — orthogonalized momentum optimizer for matrix parameters.

Reference:
    Keller Jordan, "Modded NanoGPT: Muon optimizer" (2024)
    https://github.com/KellerJordan/modded-nanogpt
    https://kellerjordan.github.io/posts/muon/

Result from Jordan's NanoGPT speed runs: Muon achieves the same loss as AdamW
in 52% of the wall-clock time. This is the largest known optimization speedup
for transformer pre-training.

How it works:
    1. Standard SGD with momentum is run on each 2D parameter
    2. The momentum-smoothed update is then orthogonalized via 5 steps of a
       quintic Newton-Schulz iteration on the singular values
    3. The orthogonalized update is applied with the configured learning rate

The orthogonalization step pushes the update toward the unit-Frobenius-norm
matrix that has the same row/column space as the gradient. This is equivalent
to running a Stiefel-manifold update in the natural-gradient direction. It
gives Muon both the speed of SGD and the dimension-independence of Adam.

Constraints:
    - Muon only handles 2D parameters (matrices). Use AdamW or similar for
      1D parameters (norms, biases, scalars) and 3D+ parameters (rare).
    - The matrix must be at least 2x2.
    - For matrices that are taller than wide, Muon transposes internally.

Hybrid optimizer pattern:
    matrix_params, scalar_params = partition_params(model)
    muon = Muon(matrix_params, lr=1e-3)
    adam = bnb.optim.AdamW8bit(scalar_params, lr=3e-4)
    # In the train loop:
    muon.step(); muon.zero_grad()
    adam.step(); adam.zero_grad()
"""

import math
from typing import Iterable, List, Optional, Tuple

import torch


# -----------------------------------------------------------------------------
# Newton-Schulz quintic iteration
# -----------------------------------------------------------------------------

@torch.no_grad()
def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Quintic Newton-Schulz iteration for matrix orthogonalization.

    Tuned coefficients from Keller Jordan (chosen to make the orthogonalization
    converge in ~5 steps from any reasonable initialization, while remaining
    stable in bf16).

    Args:
        G:     a 2D tensor (the gradient or update direction)
        steps: number of NS iterations (5 is the standard tuned value)
        eps:   numerical stabilizer for the initial normalization

    Returns:
        Orthogonalized version of G (same shape and dtype).
    """
    assert G.dim() == 2, f"Newton-Schulz expects 2D tensor, got {G.dim()}D"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16) if G.dtype not in (torch.bfloat16, torch.float32) else G

    # Transpose if taller than wide so the inner matmul is on the smaller side
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    # Normalize so the spectral norm is roughly 1
    X = X / (X.norm() + eps)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(G.dtype)


# -----------------------------------------------------------------------------
# Muon optimizer
# -----------------------------------------------------------------------------

class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for 2D parameters only.

    Args:
        params:        an iterable of 2D parameters
        lr:            learning rate
        momentum:      momentum coefficient (Jordan default: 0.95)
        nesterov:      use Nesterov-style lookahead
        ns_steps:      number of Newton-Schulz iterations (5 is default)
        weight_decay:  decoupled weight decay coefficient
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # Validate that all params are 2D
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2:
                    raise ValueError(
                        f"Muon only handles 2D parameters; got {p.dim()}D shape {tuple(p.shape)}. "
                        "Use a hybrid optimizer (Muon for matrices, AdamW for scalars)."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]

                # Momentum update (in place)
                buf.mul_(mom).add_(grad)

                # Nesterov lookahead
                if nesterov:
                    update = grad.add(buf, alpha=mom)
                else:
                    update = buf.clone()

                # Newton-Schulz orthogonalization
                update = newton_schulz5(update, steps=ns_steps)

                # Decoupled weight decay
                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                # Scale-aware step size: spectral norm of an orthogonal matrix
                # is 1, so we rescale by sqrt(out/in) to match the expected
                # update magnitude of an Adam-style optimizer.
                fan_out, fan_in = p.shape
                scale = max(1.0, math.sqrt(max(fan_out, fan_in) / max(min(fan_out, fan_in), 1)))
                p.add_(update, alpha=-lr * scale)

        return loss


# -----------------------------------------------------------------------------
# Parameter partitioning helper
# -----------------------------------------------------------------------------

def partition_params_for_muon(
    model: torch.nn.Module,
    skip_substrings: Tuple[str, ...] = (
        "embed",       # token embedding (use AdamW; embeddings have special init)
        "lm_head",     # tied to embedding, so same as above
        "fuzz",        # learned scalars
        "leak_rate",   # cerebellum scalar
        "alpha",       # output gates
        "output_gate",
    ),
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Partition a model's parameters into:
        muon_params : 2D matrices that should be optimized by Muon
        adam_params : 1D parameters (norms, biases) and special-cased 2D ones

    Args:
        model:           the model to partition
        skip_substrings: parameter name substrings that force AdamW even if 2D

    Returns:
        (muon_params, adam_params) — two disjoint lists
    """
    muon_params: List[torch.nn.Parameter] = []
    adam_params: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Force-skip patterns
        if any(s in name for s in skip_substrings):
            adam_params.append(p)
            continue
        # 2D matrices → Muon, everything else → AdamW
        if p.dim() == 2:
            muon_params.append(p)
        else:
            adam_params.append(p)

    return muon_params, adam_params


# -----------------------------------------------------------------------------
# Hybrid optimizer wrapper (convenience)
# -----------------------------------------------------------------------------

class HybridOptimizer:
    """
    Wraps two underlying optimizers (typically Muon for matrices and AdamW8bit
    for everything else) and exposes a single .step() / .zero_grad() interface.

    Usage:
        opt = HybridOptimizer.from_model(model, muon_lr=1e-3, adam_lr=3e-4)
        opt.step()
        opt.zero_grad()
    """

    def __init__(self, optimizers: List[torch.optim.Optimizer]):
        self.optimizers = optimizers

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        muon_lr: float = 1e-3,
        adam_lr: float = 3e-4,
        muon_momentum: float = 0.95,
        adam_betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        use_8bit_adam: bool = True,
    ) -> "HybridOptimizer":
        muon_params, adam_params = partition_params_for_muon(model)

        optimizers: List[torch.optim.Optimizer] = []

        if muon_params:
            optimizers.append(Muon(
                muon_params,
                lr=muon_lr,
                momentum=muon_momentum,
                nesterov=True,
                ns_steps=5,
                weight_decay=weight_decay,
            ))

        if adam_params:
            adam_cls: type
            if use_8bit_adam:
                try:
                    import bitsandbytes as bnb
                    adam_cls = bnb.optim.AdamW8bit
                except ImportError:
                    adam_cls = torch.optim.AdamW
            else:
                adam_cls = torch.optim.AdamW
            optimizers.append(adam_cls(
                adam_params,
                lr=adam_lr,
                betas=adam_betas,
                weight_decay=weight_decay,
            ))

        return cls(optimizers)

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {f"opt_{i}": o.state_dict() for i, o in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict):
        for i, o in enumerate(self.optimizers):
            o.load_state_dict(state_dict[f"opt_{i}"])
