"""
Spectral-radius diagnostic for the Mythos-style LTI recurrent update in
``fant3.model.recursion.MoRShared``.

The Mythos / RDT (Recurrent-Depth Transformer) literature motivates
constraining the recurrent update operator so that its spectral radius
:math:`\\rho(A) < 1`, guaranteeing stability of the linear component of

    ``current_{k+1} = A * current_k + B * x_original + C * retrieved
                       + block(current_k + loop_emb[k])``

In our implementation we parameterize :math:`A` as a DIAGONAL matrix (one
scalar per hidden dimension) and, when ``cfg.mor_spectral_constraint`` is
True, we return ``-softplus(a_diag)``.  That guarantees each diagonal entry
is in :math:`(-\\infty, 0)`.  Since the linear step of the recurrence is
:math:`I + A` on a residual stream, we want the entries of
:math:`(I + A)` to lie in :math:`(-1, 1)`, which is equivalent to the
diagonal entries of :math:`A` lying in :math:`(-2, 0)`.

This module provides the utilities to check that empirically during or
after training.

Usage::

    from fant3.diagnostics.spectral import (
        spectral_radius_report, assert_lti_stable
    )
    report = spectral_radius_report(model)
    print(report)
    assert_lti_stable(model)          # raises AssertionError if unstable

Attaching ``spectral_radius_report`` to a training loop as a periodic
telemetry print (every 500 steps, say) is cheap — it only reads a length-
``dim`` parameter tensor per MoR layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


# Lower/upper bounds on diag(A) that keep (I + A) contraction-stable
#   |1 + a_ii| < 1   <=>   a_ii in (-2, 0)
_A_MIN = -2.0
_A_MAX = 0.0


@dataclass
class MoRSpectralEntry:
    """Per-MoR-layer spectral diagnostic snapshot."""

    layer_path:            str          # dotted path, e.g. "mor"
    lti_enabled:           bool
    spectral_constrained:  bool
    a_diag_raw_min:        float
    a_diag_raw_max:        float
    a_effective_min:       float        # entries of (A) after softplus negation
    a_effective_max:       float
    i_plus_a_min:          float        # entries of (I + A)
    i_plus_a_max:          float
    abs_i_plus_a_max:      float        # L-infinity norm of (I + A) -- diag => spectral radius for diagonal operator
    stable:                bool

    def summary_line(self) -> str:
        if not self.lti_enabled:
            return f"{self.layer_path}: LTI disabled"
        flag = "OK  " if self.stable else "FAIL"
        constraint = "spectral" if self.spectral_constrained else "raw"
        return (
            f"{self.layer_path}: [{flag}] {constraint:>8}  "
            f"A_eff=[{self.a_effective_min:+.4f}, {self.a_effective_max:+.4f}]  "
            f"(I+A)=[{self.i_plus_a_min:+.4f}, {self.i_plus_a_max:+.4f}]  "
            f"rho={self.abs_i_plus_a_max:.4f}"
        )


@torch.no_grad()
def _layer_report(layer_path: str, mor_module) -> MoRSpectralEntry:
    """Compute spectral stats for one ``MoRShared`` instance."""
    lti_enabled = bool(getattr(mor_module, "lti_enabled", False))
    spec_enabled = bool(getattr(mor_module, "spectral_enabled", False))

    if not lti_enabled or not hasattr(mor_module, "a_diag"):
        return MoRSpectralEntry(
            layer_path=layer_path,
            lti_enabled=False,
            spectral_constrained=False,
            a_diag_raw_min=float("nan"),
            a_diag_raw_max=float("nan"),
            a_effective_min=float("nan"),
            a_effective_max=float("nan"),
            i_plus_a_min=float("nan"),
            i_plus_a_max=float("nan"),
            abs_i_plus_a_max=float("nan"),
            stable=True,
        )

    a_raw = mor_module.a_diag.detach().float()
    a_eff = (-F.softplus(a_raw)) if spec_enabled else a_raw
    i_plus_a = 1.0 + a_eff
    abs_i_plus_a_max = i_plus_a.abs().max().item()

    stable = (_A_MIN < a_eff.min().item()) and (a_eff.max().item() < _A_MAX) \
        if spec_enabled else (abs_i_plus_a_max < 1.0)

    return MoRSpectralEntry(
        layer_path=layer_path,
        lti_enabled=True,
        spectral_constrained=spec_enabled,
        a_diag_raw_min=a_raw.min().item(),
        a_diag_raw_max=a_raw.max().item(),
        a_effective_min=a_eff.min().item(),
        a_effective_max=a_eff.max().item(),
        i_plus_a_min=i_plus_a.min().item(),
        i_plus_a_max=i_plus_a.max().item(),
        abs_i_plus_a_max=abs_i_plus_a_max,
        stable=bool(stable),
    )


def spectral_radius_report(model: torch.nn.Module) -> List[MoRSpectralEntry]:
    """Walk a FANT 3 model and return a per-MoR-layer spectral report.

    Safe to call during training; runs entirely under ``torch.no_grad()``
    and only reads parameter tensors.  Typical cost per call is O(dim)
    per MoR layer (no matrix ops on activations).
    """
    # Lazy import so this module does not require the full fant3 package to
    # be importable.  We detect MoRShared by duck-typing on attribute names.
    out: List[MoRSpectralEntry] = []
    for name, m in model.named_modules():
        # Duck-type: a MoRShared exposes .lti_enabled and .max_depth
        if hasattr(m, "lti_enabled") and hasattr(m, "max_depth") and hasattr(m, "block"):
            out.append(_layer_report(name or "<root>", m))
    return out


def format_report(entries: List[MoRSpectralEntry]) -> str:
    """Pretty-print a report to a single multi-line string."""
    if not entries:
        return "(no MoR layers found; nothing to report)"
    lines = ["MoR spectral report:"]
    for e in entries:
        lines.append("  " + e.summary_line())
    any_unstable = any((e.lti_enabled and not e.stable) for e in entries)
    lines.append("")
    lines.append("OVERALL: " + ("UNSTABLE" if any_unstable else "STABLE"))
    return "\n".join(lines)


def assert_lti_stable(model: torch.nn.Module) -> None:
    """Raise ``AssertionError`` if ANY enabled MoR LTI layer is unstable.

    Useful as an after-opt-step guard in training loops::

        opt.step()
        if step % 500 == 0:
            assert_lti_stable(model)
    """
    entries = spectral_radius_report(model)
    offenders = [e for e in entries if e.lti_enabled and not e.stable]
    if offenders:
        msg = "MoR LTI recurrence is NOT spectrally stable:\n"
        for e in offenders:
            msg += "  " + e.summary_line() + "\n"
        raise AssertionError(msg)


__all__ = [
    "MoRSpectralEntry",
    "spectral_radius_report",
    "format_report",
    "assert_lti_stable",
]
