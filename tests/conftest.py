"""Shared pytest fixtures for the FANT 2 test suite."""

import pytest
import torch

from fant2.config import fant2_tiny
from fant2.model import FANT2Model


@pytest.fixture(scope="session")
def tiny_cfg():
    return fant2_tiny()


@pytest.fixture(scope="function")
def tiny_model(tiny_cfg):
    """Fresh tiny model for each test (so weight changes don't leak)."""
    torch.manual_seed(0)
    return FANT2Model(tiny_cfg)


@pytest.fixture(scope="session")
def device():
    return "cpu"


@pytest.fixture(scope="function")
def synthetic_batch(tiny_cfg):
    """A reproducible (input_ids, target_ids) batch."""
    torch.manual_seed(42)
    B, T = 2, 32
    ids = torch.randint(0, tiny_cfg.vocab_size, (B, T + 1), dtype=torch.long)
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
