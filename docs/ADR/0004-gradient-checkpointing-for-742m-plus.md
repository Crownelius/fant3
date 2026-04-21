# ADR 0004: Gradient Checkpointing Mandatory at 742m and Above

## Status

Accepted (implemented 2026-04-19, auto-enabled in notebook cell 8)

---

## Context

The first 742m training attempt on Colab A100 (94.97 GiB GPU) OOM'd (Out of Memory) at the very first forward pass:

```
OutOfMemoryError: CUDA out of memory.
Tried to allocate 2.73 GiB.
GPU 0 has a total capacity of 94.97 GiB of which 479.88 MiB is free.
Of the allocated memory 93.58 GiB is allocated by PyTorch,
and 279.08 MiB is reserved by PyTorch but unallocated.
```

Failure location: `matryoshka_moe.py:248` — `W_up_sel = self.W_up[idx]` during forward.

The pre-run VRAM estimate was 40–50 GB, wrong by 2x. The estimate was extrapolated from the 150m run's 38 GB peak without accounting for sequence length scaling:

**The key insight missed:** At MoE + MoR (Mixture of Recursions) architectures, activation memory does not scale as `O(B × T × dim)` like a plain transformer. It scales as:

```
activation_peak ≈ B × T × dim × n_recursion_depths × moe_fanout
```

where `moe_fanout` captures the per-expert intermediate tensors `(active_experts, batch, seq, hidden)` that must be held during the backward pass. At T=512, the activations alone project to ~145 GB for the 742m config — well past A100 95 GB.

An emergency patch (drop T=512 → T=256) reduced the activation memory to ~40 GB, but doubled the number of steps needed to preserve training token count.

The proper fix is **gradient checkpointing**: re-compute activations during the backward pass instead of storing them during the forward pass. This trades ~25–40% compute for 3–5x less activation memory.

---

## Decision

Implement gradient checkpointing using `torch.utils.checkpoint.checkpoint(use_reentrant=False)` at three levels:

**1. DenseBlock and MoEBlock** (`fant3/model/fant3_model.py`):

```python
def forward(self, x, mask=None):
    if self.use_gc and self.training:
        from torch.utils.checkpoint import checkpoint
        return checkpoint(self._forward_inner, x, mask, use_reentrant=False)
    return self._forward_inner(x, mask)
```

**2. MoR recursion passes** (`fant3/model/recursion.py`):

Each inner pass of the MoR recursion is wrapped independently:

```python
current = checkpoint(self.block, current, mask, use_reentrant=False)
```

This is the biggest saver: without it, each of the 2–3 recursion passes stores its full activation tensor, multiplying the activation budget by n_recursion_depths.

**Config flag** (`fant3/config.py`):

```python
use_gradient_checkpointing: bool = False
```

**Auto-enable in notebook** (cell 8):

```python
cfg.use_gradient_checkpointing = TARGET_SCALE in ('742m', '1b')
```

**Local verification (150m smoke on RTX 3060 12 GB):**

| Setting | CE loss | Peak VRAM |
|---|---|---|
| `use_gradient_checkpointing=False` | 10.5625 | 4.24 GB |
| `use_gradient_checkpointing=True` | 10.5625 | 1.60 GB |

2.65x VRAM reduction, bit-exact identical loss. Proves correctness.

**Production 742m result:** VRAM stable at 45.66 GB throughout 10,000-step Tier C run on A100 96 GB. No OOM. Restored from the emergency T=256 patch back to T=1024.

---

## Consequences

**Benefits:**
- 742m training fits comfortably on A100 80 GB (45.66 GB actual vs 94.97 GB available)
- 1b training projected to fit at ~50–60 GB (same architecture, larger dim and more layers)
- No accuracy impact — loss is mathematically identical with or without checkpointing
- `use_reentrant=False` allows side effects in the checkpointed function (Python objects can be modified — e.g. `self.last_router_info = router_info` is written twice, idempotently)

**Drawbacks:**
- ~25–40% compute overhead per step (recomputation of activations during backward)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` must be set before any `import torch` statement; setting it later is a no-op. This was Bug 3 in the 742m chapter — the env var was in cell 20 while CUDA was initialized in cells 3/5/17.
- MoE `W_up` gather tensors are still allocated during gradient-checkpoint recompute. At B=2, T=1024, this creates 72 GB peak despite checkpointing. Lesson: effective batch of 8 must be achieved through GRAD_ACCUM=8, not BATCH_SIZE=2, at 742m.
- Gradient checkpointing only helps during training. Inference VRAM is unaffected.

**Rule for this architecture:**

> Every 2x increase in sequence length requires either 4x more VRAM or gradient checkpointing.

At MoE+MoR, use the formula `activation_peak ≈ 2 × B × T × dim × n_recursion_depths` as a lower bound; actual peak will be 1.5–2x higher due to MoE expert intermediate tensors.

---

## Alternatives Considered

**Drop sequence length permanently (T=256 emergency patch)**  
Rejected as final solution: halves training signal quality (context window) and doubles steps needed for same token count. Kept as emergency fallback if A100 80 GB is unavailable.

**Flash Attention**  
Relevant for attention VRAM but does not help with MoE expert tensors, which are the dominant term at 742m scale. Flash Attention is not yet integrated into MASA (Multi-head Attention with Shared Atoms). Deferred.

**Reduce batch size only (no gradient checkpointing)**  
The 150m chapter used B=2 with GRAD_ACCUM=1 → effective batch 2. At 742m with the same approach, VRAM remains ~70+ GB because MoE expert tensors scale with T, not B. Batch size reduction alone is insufficient.

**Mixed precision for intermediate tensors**  
The model already uses bf16 (bfloat16) throughout. Further precision reduction (fp8 or int8 activations) would require quantization-aware training changes. Deferred to post-launch.

**Offload optimizer to CPU**  
bitsandbytes 8-bit AdamW (already in use) reduces optimizer state from 3x model params (float32 Adam) to ~0.5x (int8 quantized). This is already applied. CPU offload via DeepSpeed ZeRO-3 would help further but adds infrastructure complexity not warranted at this scale.
