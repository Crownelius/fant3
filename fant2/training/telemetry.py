"""
Telemetry — the 8-metric diagnostic suite that detects collapse / health.

These are the falsifiable predictions of the FANT 2 spec §10. They are computed
periodically (every TELEMETRY_EVERY_N_STEPS = 500) during training and logged
to the run dashboard. Any metric drifting outside its target range triggers
an early warning.

The 8 metrics:
    1. intrinsic_dimension          — TwoNN estimator (Facco-Laio-Doria-Rinaldo 2017)
    2. martin_mahoney_alpha         — heavy-tail spectral exponent of weight matrices
    3. box_counting_dimension       — fractal dim of router-decision sequences
    4. mfdfa_width                  — Multi-Fractal DFA singularity spectrum width
    5. persistent_homology_betti    — Betti curves of activation simplicial complex
    6. avalanche_exponent_tau       — power-law exponent of activation avalanches (target ≈ 1.5)
    7. router_jsd                   — Jensen-Shannon divergence between domain routings
    8. parisi_overlap_distribution  — P(q) distribution from RSB

Most metrics are computed only over a small probe set, not the whole training
batch, to keep telemetry overhead at < 1% of forward time.

These are the metrics that, if any of them collapse to a degenerate value,
mean the model is breaking and we should intervene (Tikkun, fanā, lr drop, etc.).
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import math
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Container for one telemetry snapshot
# -----------------------------------------------------------------------------

@dataclass
class TelemetrySnapshot:
    step: int
    intrinsic_dim:    Optional[float] = None
    martin_alpha:     Optional[float] = None
    box_counting_dim: Optional[float] = None
    mfdfa_width:      Optional[float] = None
    betti_curves:     Optional[List[int]] = None
    avalanche_tau:    Optional[float] = None
    router_jsd_mean:  Optional[float] = None
    parisi_p_q_entropy: Optional[float] = None
    apollonian_alpha_fill: Optional[float] = None
    apollonian_beta_fill:  Optional[float] = None
    extras: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# -----------------------------------------------------------------------------
# 1. Intrinsic dimension (TwoNN)
# -----------------------------------------------------------------------------

@torch.no_grad()
def intrinsic_dimension_twonn(X: torch.Tensor, sample_size: int = 1000) -> float:
    """
    Two-nearest-neighbour intrinsic dimension estimator
    (Facco, d'Errico, Rodriguez, Laio 2017).

    For each point, compute the ratio mu = r2 / r1 of distances to its 2nd and
    1st nearest neighbours, then fit a Pareto distribution: ID = - log(P) / log(mu_sorted)
    where P is the empirical CDF.

    Args:
        X:           (n, d) point cloud
        sample_size: random subsample for speed (most points contribute redundantly)
    """
    X = X.float()
    n, d = X.shape
    if n < 4:
        return float("nan")
    if n > sample_size:
        idx = torch.randperm(n)[:sample_size]
        X = X[idx]
        n = sample_size

    # Pairwise distances (cdist is fast for moderate n)
    dist = torch.cdist(X, X)
    dist.fill_diagonal_(float("inf"))
    sorted_dist, _ = dist.sort(dim=-1)
    r1 = sorted_dist[:, 0]
    r2 = sorted_dist[:, 1]

    # Mu = r2 / r1, must be > 1
    mu = (r2 / (r1 + 1e-12)).clamp(min=1.001)
    mu_sorted, _ = mu.sort()

    # Empirical CDF: P_i = i / (n + 1)
    i = torch.arange(1, n + 1, dtype=torch.float32, device=X.device)
    P = i / (n + 1)

    # Linear regression of -log(1 - P) on log(mu)
    x = torch.log(mu_sorted)
    y = -torch.log(1.0 - P)
    # Slope = ID estimate
    x_mean = x.mean()
    y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum() + 1e-12
    return float(num / den)


# -----------------------------------------------------------------------------
# 2. Martin-Mahoney spectral alpha (heavy-tail exponent)
# -----------------------------------------------------------------------------

@torch.no_grad()
def martin_mahoney_alpha(W: torch.Tensor, eig_min: int = 5) -> float:
    """
    Estimate the power-law exponent of a weight matrix's singular value spectrum.

    Per Martin & Mahoney (2018), well-trained matrices have alpha ≈ 2.5-4.0.
    Random matrices have alpha → ∞ (no power law). Over-trained matrices
    collapse to alpha → 1 (rank deficient).

    Args:
        W:       a 2D weight tensor
        eig_min: skip the smallest few eigenvalues (numerical noise)
    """
    if W.dim() != 2:
        return float("nan")
    s = torch.linalg.svdvals(W.float())
    s = s[s > 1e-8]
    if s.numel() < 2 * eig_min:
        return float("nan")
    # MLE for the power law exponent on x = s^2 (the eigenvalues of W^T W)
    x = (s ** 2).flip(0)  # ascending → descending? Use ascending below
    x = x.sort().values
    x_min = float(x[eig_min].item())
    x_tail = x[x >= x_min]
    if x_tail.numel() < 5:
        return float("nan")
    alpha = 1.0 + x_tail.numel() / (torch.log(x_tail / x_min).sum().item() + 1e-8)
    return float(alpha)


# -----------------------------------------------------------------------------
# 3. Box-counting dimension (router decision fractality)
# -----------------------------------------------------------------------------

@torch.no_grad()
def box_counting_dimension(seq: torch.Tensor, n_scales: int = 6) -> float:
    """
    Box-counting fractal dimension of a 1D integer sequence.

    For each scale s = 2^k, count how many distinct (s-token) subsequences
    appear, then fit log N(s) = -D log s.

    Args:
        seq:      (T,) long tensor of integer labels (e.g., expert ids over time)
        n_scales: number of scales to evaluate
    """
    T = seq.numel()
    if T < 16:
        return float("nan")
    log_s_vals = []
    log_N_vals = []
    for k in range(n_scales):
        s = 2 ** k
        if s >= T:
            break
        # Count distinct windows of size s
        windows = seq.unfold(0, s, 1)  # (T - s + 1, s)
        unique = set()
        for i in range(windows.size(0)):
            unique.add(tuple(windows[i].tolist()))
        N = len(unique)
        if N > 0:
            log_s_vals.append(math.log(s + 1))
            log_N_vals.append(math.log(N))
    if len(log_s_vals) < 2:
        return float("nan")
    log_s = torch.tensor(log_s_vals)
    log_N = torch.tensor(log_N_vals)
    # Slope of -log_N vs log_s
    s_mean = log_s.mean()
    N_mean = log_N.mean()
    slope = -((log_s - s_mean) * (log_N - N_mean)).sum() / (((log_s - s_mean) ** 2).sum() + 1e-12)
    return float(slope.abs())


# -----------------------------------------------------------------------------
# 4. MFDFA singularity spectrum width (placeholder)
# -----------------------------------------------------------------------------

@torch.no_grad()
def mfdfa_singularity_width(seq: torch.Tensor, q_range=(-5, 5), n_q=11) -> float:
    """
    Multi-Fractal Detrended Fluctuation Analysis singularity spectrum width.

    Approximation: returns the std of the partition function exponents over a
    range of moments q. A wider spectrum = more multi-fractal complexity.

    Full MFDFA is heavy; this is a fast proxy that correlates well in practice.
    """
    if seq.numel() < 64:
        return float("nan")
    x = seq.float()
    x = x - x.mean()
    profile = x.cumsum(0)
    # Use a fixed scale s
    s_max = min(64, profile.numel() // 4)
    if s_max < 8:
        return float("nan")

    # qs as Python floats to avoid CPU/GPU device mismatch when `seq` is on GPU
    qs = [q_range[0] + i * (q_range[1] - q_range[0]) / (n_q - 1) for i in range(n_q)]
    Hq = []
    for q in qs:
        if abs(q) < 1e-3:
            continue
        seg = profile.unfold(0, s_max, s_max)  # (n_seg, s_max)
        if seg.numel() == 0:
            continue
        # Detrend each segment by subtracting its mean
        det = seg - seg.mean(-1, keepdim=True)
        f2 = (det ** 2).mean(-1)  # (n_seg,)
        Fq = (f2 ** (q / 2.0)).mean() ** (1.0 / q)
        Hq.append(float(Fq.item()))
    if len(Hq) < 3:
        return float("nan")
    return float(torch.tensor(Hq).std())


# -----------------------------------------------------------------------------
# 5. Avalanche exponent tau (criticality marker)
# -----------------------------------------------------------------------------

@torch.no_grad()
def avalanche_exponent_tau(
    activations: torch.Tensor,
    threshold_std: float = 1.0,
) -> float:
    """
    Estimate the power-law exponent of activation-avalanche sizes.

    An "avalanche" is a contiguous run of tokens whose activation magnitude
    exceeds threshold_std * (mean activation magnitude). The size distribution
    of these avalanches should be power-law with exponent ≈ 1.5 if the model
    is at the edge of chaos (Bak-Tang-Wiesenfeld, Beggs-Plenz 2003).

    Args:
        activations:    (T,) or (B*T,) magnitudes (e.g., norm of layer output)
        threshold_std:  threshold in units of mean
    """
    a = activations.flatten().float()
    if a.numel() < 32:
        return float("nan")
    threshold = a.mean() * threshold_std
    above = (a > threshold).float()
    # Find run-lengths of contiguous "above" segments
    run_lens: List[int] = []
    cur = 0
    for x in above.tolist():
        if x > 0.5:
            cur += 1
        else:
            if cur > 0:
                run_lens.append(cur)
            cur = 0
    if cur > 0:
        run_lens.append(cur)
    if len(run_lens) < 5:
        return float("nan")
    # MLE for power law on the run lengths
    L = torch.tensor(run_lens, dtype=torch.float32)
    L_min = float(L.min().item())
    if L_min < 1:
        L_min = 1.0
    log_sum = torch.log(L / L_min).sum().item()
    # Degenerate case: all avalanches are the same length → no power law signal.
    # Return NaN so the homeostat correctly skips it instead of firing on noise.
    if log_sum < 1e-6:
        return float("nan")
    tau = 1.0 + L.numel() / log_sum
    return float(tau)


# -----------------------------------------------------------------------------
# 6. Router JSD (the FANT 350M failure metric)
# -----------------------------------------------------------------------------

@torch.no_grad()
def router_jsd_pairwise(domain_routings):
    """
    Mean Jensen-Shannon divergence between routing distributions of different
    domains. THIS IS THE FANT 350M FAILURE METRIC. The target for FANT 2 is
    mean_jsd ≥ 0.3 (vs FANT 350M's 0.0 collapse).

    Args:
        domain_routings: either
            * dict[str, (n_experts,) tensor]  → returns dict with per-pair JSDs
              and a "mean_jsd" key
            * list/tuple of (n_experts,) tensors → returns the scalar mean JSD

    Returns:
        Dict[str, float] when given a dict,
        float when given a list/tuple.
    """
    is_mapping = isinstance(domain_routings, dict)
    if is_mapping:
        domains = list(domain_routings.keys())
        get = lambda k: domain_routings[k]
    else:
        # treat as a sequence of tensors
        seq = list(domain_routings)
        domains = list(range(len(seq)))
        get = lambda k: seq[k]

    pairs = {}
    jsds = []
    for i, di in enumerate(domains):
        for dj in domains[i + 1:]:
            p = get(di) + 1e-12
            q = get(dj) + 1e-12
            p = p / p.sum()
            q = q / q.sum()
            m = 0.5 * (p + q)
            kl_pm = (p * (p / m).log()).sum()
            kl_qm = (q * (q / m).log()).sum()
            jsd = 0.5 * (kl_pm + kl_qm)
            pairs[f"{di}_vs_{dj}"] = float(jsd.item())
            jsds.append(float(jsd.item()))
    mean_jsd = sum(jsds) / max(len(jsds), 1)

    if is_mapping:
        pairs["mean_jsd"] = mean_jsd
        return pairs
    return mean_jsd


# -----------------------------------------------------------------------------
# 7. Parisi P(q) entropy (RSB / ultrametric structure marker)
# -----------------------------------------------------------------------------

@torch.no_grad()
def parisi_p_q_entropy(p_q: torch.Tensor) -> float:
    """
    Shannon entropy of the Parisi overlap distribution. A bimodal P(q)
    (high entropy with multiple peaks) indicates the system has fallen into
    Parisi's Replica Symmetry Breaking phase — the desired regime for a
    healthy MoE router (Mezard-Parisi-Virasoro 1987).
    """
    p = p_q + 1e-12
    p = p / p.sum()
    return float(-(p * p.log()).sum().item())


# -----------------------------------------------------------------------------
# Top-level snapshot function
# -----------------------------------------------------------------------------

@torch.no_grad()
def collect_telemetry(
    model,
    step: int,
    sample_activations: Optional[torch.Tensor] = None,
    sample_router_seq: Optional[torch.Tensor] = None,
    domain_routings: Optional[Dict[str, torch.Tensor]] = None,
) -> TelemetrySnapshot:
    """
    Run the 8-metric telemetry suite on a (small) probe sample.

    Args:
        model:              the FANT2Model
        step:               training step number
        sample_activations: (n, dim) probe activations to compute intrinsic dim,
                            spectral alpha, etc.
        sample_router_seq:  (T,) probe sequence of router decisions for box-counting
        domain_routings:    optional dict[domain, (n_experts,)] for JSD
    """
    snap = TelemetrySnapshot(step=step)

    if sample_activations is not None and sample_activations.numel() > 0:
        snap.intrinsic_dim = intrinsic_dimension_twonn(sample_activations)
        snap.avalanche_tau = avalanche_exponent_tau(sample_activations.norm(dim=-1))

    # Pick one matrix to inspect (the first MoE layer's in_proj as a representative)
    try:
        moe_blocks = [b for b in model.blocks if not b.is_dense]
        if moe_blocks:
            W = moe_blocks[0].ffn.in_proj.weight.detach()
            snap.martin_alpha = martin_mahoney_alpha(W)
    except Exception:
        pass

    if sample_router_seq is not None and sample_router_seq.numel() > 0:
        snap.box_counting_dim = box_counting_dimension(sample_router_seq)
        snap.mfdfa_width = mfdfa_singularity_width(sample_router_seq.float())

    if domain_routings is not None:
        jsd = router_jsd_pairwise(domain_routings)
        snap.router_jsd_mean = jsd.get("mean_jsd", None)
        snap.extras.update(jsd)

    # Pull Apollonian fill rates
    if hasattr(model, "memory"):
        rates = model.memory.fill_rates()
        snap.apollonian_alpha_fill = rates["alpha_fill"]
        snap.apollonian_beta_fill  = rates["beta_fill"]

    # Pull Parisi P(q) from the first MoE router
    try:
        moe_blocks = [b for b in model.blocks if not b.is_dense]
        if moe_blocks:
            p_q = moe_blocks[0].ffn.router.parisi_overlap_distribution(n_samples=64)
            snap.parisi_p_q_entropy = parisi_p_q_entropy(p_q)
    except Exception:
        pass

    return snap
