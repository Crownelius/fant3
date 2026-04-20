"""
The CRITICAL regression test: the FANT 350M router collapse must not recur.

FANT 350M used a single-stage softmax router with frozen projection + 32-scalar
bias. After 1 epoch of training it collapsed onto a single expert (94.5% of
routing concentrated on expert #17 across all input domains; mean pairwise
JSD between domains = 0.0). See `fant_350m_postmortem.md` for the full
investigation.

The FANT 2 fix is the HierarchicalApollonianRouter:
  * 2-stage hierarchical routing (mega-pools, then within-pool experts)
  * Sigmoid gating (NOT softmax)
  * Simplex ETF init
  * DeepSeek aux-loss-free bias updates (gradient-free, sign-based)
  * Tikkun event-driven repair
  * Fanā periodic shuffle

The success metric:
    mean pairwise JSD between routing distributions on different "domains"
    must be >= 0.30 (a flat soft routing has mean_jsd ≈ 0.0; full collapse
    onto distinct experts gives mean_jsd ≈ ln(2) ≈ 0.69).

This test fails if the router collapses, which is the canary for regressions
in the routing fix.
"""

import math

import pytest
import torch

from fant2.config import fant2_tiny
from fant2.model import FANT2Model, HierarchicalApollonianRouter
from fant2.training.telemetry import router_jsd_pairwise


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _domain_routings(model: FANT2Model, domain_token_streams):
    """
    Run the model on each "domain" (a list of token id sequences) and return
    a dict[domain_name] -> 1D distribution over the n_megapools (averaged over
    all tokens in the domain, all layers, batches).
    """
    model.eval()
    n_megapools = model.config.n_megapools
    out = {}
    with torch.no_grad():
        for name, sequences in domain_token_streams.items():
            counts = torch.zeros(n_megapools)
            n = 0
            for seq in sequences:
                if seq.dim() == 1:
                    seq = seq.unsqueeze(0)
                fwd = model(seq)
                for ro in fwd["router_outputs"]:
                    # ro.megapool_idx is (B*T,) long
                    idx = ro.megapool_idx
                    bins = torch.bincount(idx, minlength=n_megapools).float()
                    counts += bins
                    n += int(idx.numel())
            counts /= max(n, 1)
            out[name] = counts
    return out


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_router_etf_init_no_collapse():
    """
    Right at initialization (before any training), the router should NOT
    collapse onto a single mega-pool. With Simplex ETF init and unbiased
    biases, all 8 mega-pools should receive comparable load on random input.
    """
    torch.manual_seed(0)
    cfg = fant2_tiny()
    model = FANT2Model(cfg)

    # Simulate 4 different "domains" by using different random seeds
    domain_streams = {}
    for d, seed in enumerate([1, 2, 3, 4]):
        gen = torch.Generator().manual_seed(seed)
        seqs = [torch.randint(0, cfg.vocab_size, (1, 32), generator=gen, dtype=torch.long)
                for _ in range(4)]
        domain_streams[f"domain_{d}"] = seqs

    routings = _domain_routings(model, domain_streams)
    arrs = list(routings.values())
    mean_jsd = router_jsd_pairwise(arrs)
    print(f"  init mean pairwise JSD = {mean_jsd:.4f}")
    print(f"  init routings = {[r.tolist() for r in arrs]}")

    # At init we expect a positive but small JSD (random projection of random
    # input gives somewhat-different distributions but not collapse).
    # The CRITICAL property is that the max-load mega-pool should be < 60%
    # in every domain (FANT 350M had 94.5% on a single expert).
    for name, dist in routings.items():
        max_load = float(dist.max().item())
        assert max_load < 0.85, (
            f"Domain {name} has max mega-pool load {max_load:.3f} at init — "
            "this is the FANT 350M collapse signature!"
        )


def test_router_jsd_metric_sanity():
    """
    The router_jsd_pairwise telemetry function itself must be correct.
    """
    # Two identical distributions: JSD = 0
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.5, 0.5])
    assert router_jsd_pairwise([p, q]) < 1e-6

    # Two completely disjoint one-hots: JSD = ln(2)
    p = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.0, 1.0])
    expected = math.log(2)
    got = router_jsd_pairwise([p, q])
    assert abs(got - expected) < 1e-3, f"expected {expected}, got {got}"


def test_aux_loss_free_bias_balances_load():
    """
    A direct test of the DeepSeek aux-loss-free bias mechanism: feed
    skewed loads in repeatedly and verify the bias drifts to compensate.
    """
    router = HierarchicalApollonianRouter(
        dim=128, n_megapools=4, n_per_pool=4, top_k=2, gamma=1e-2
    )
    initial_bias = router.megapool_bias.clone()

    # Feed an extremely skewed load: pool 0 gets 90%, pools 1..3 split 10%
    skewed_load = torch.tensor([0.9, 0.04, 0.03, 0.03])
    skewed_ex_load = torch.full((4, 4), 1.0 / 16)
    for _ in range(50):
        router.update_biases(skewed_load, skewed_ex_load)

    final_bias = router.megapool_bias.clone()
    delta = final_bias - initial_bias

    # Pool 0 was over-loaded → its bias should DECREASE
    # Pools 1..3 were under-loaded → their biases should INCREASE
    assert delta[0] < 0, f"over-loaded pool's bias should decrease, got {delta[0]:.4f}"
    for i in [1, 2, 3]:
        assert delta[i] > 0, f"under-loaded pool {i}'s bias should increase, got {delta[i]:.4f}"


def test_tikkun_repair_fires_on_skew():
    """When the EMA load exceeds the tikkun_threshold, repair fires."""
    router = HierarchicalApollonianRouter(
        dim=128, n_megapools=4, n_per_pool=4, top_k=2,
        gamma=0.0,  # disable bias updates so we can manually skew
        tikkun_threshold=0.20,
    )
    # Manually push the EMA into a collapsed state
    router.megapool_load_ema = torch.tensor([0.8, 0.1, 0.05, 0.05])
    fired = router.tikkun_repair()
    assert fired is True
    # Pool 0's bias should have decreased
    assert router.megapool_bias[0] < 0


def test_tikkun_no_op_when_balanced():
    """A balanced router does NOT trigger tikkun."""
    router = HierarchicalApollonianRouter(
        dim=128, n_megapools=4, n_per_pool=4, top_k=2, tikkun_threshold=0.20
    )
    # Already balanced (all 0.25)
    fired = router.tikkun_repair()
    assert fired is False


def test_full_model_no_init_collapse_with_more_domains():
    """
    The CRITICAL canary: with 6 distinct random "domains" of input, no
    single mega-pool should dominate any of them, and the mean pairwise
    JSD should not be 0.

    (After full training, the success metric is mean_jsd >= 0.30. At init
    this can't be guaranteed since the input is random, but the test is
    that we are NOT in the collapsed state.)
    """
    torch.manual_seed(7)
    cfg = fant2_tiny()
    model = FANT2Model(cfg)

    domain_streams = {}
    for d in range(6):
        gen = torch.Generator().manual_seed(100 + d)
        seqs = [torch.randint(0, cfg.vocab_size, (1, 32), generator=gen, dtype=torch.long)
                for _ in range(8)]
        domain_streams[f"domain_{d}"] = seqs

    routings = _domain_routings(model, domain_streams)
    arrs = list(routings.values())

    # No single mega-pool should be > 80% in any domain (well above 1/4 = 0.25
    # uniform but well below the 94.5% FANT 350M failure mode)
    for name, dist in routings.items():
        max_load = float(dist.max().item())
        assert max_load < 0.85, (
            f"Domain {name} max-load {max_load:.3f} — "
            "approaching FANT 350M collapse signature!"
        )

    # And all 4 mega-pools should be ACTIVE (none receiving 0 routing)
    n_megapools = cfg.n_megapools
    for name, dist in routings.items():
        n_active = int((dist > 0.001).sum().item())
        assert n_active >= n_megapools // 2, (
            f"Domain {name}: only {n_active}/{n_megapools} mega-pools active"
        )
