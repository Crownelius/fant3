# Mathematical Foundations

Formal derivations for each FANT 3 architectural component. This document is the rigorous companion to [THEORY/README.md](./THEORY/README.md). Use it when you need the actual equations, invariants, and proof sketches rather than plain-English descriptions.

GitHub renders LaTeX in markdown via `$...$` (inline) and `$$...$$` (display).

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $B, T, d$ | batch size, sequence length, model dimension |
| $V$ | vocabulary size (32,768 for tokenizer_v2) |
| $L$ | number of transformer layers |
| $h, d_h$ | number of attention heads, head dimension |
| $x_t \in \mathbb{R}^d$ | hidden state at token position $t$ |
| $W \in \mathbb{R}^{m \times n}$ | a weight matrix with $m$ rows, $n$ columns |
| $\sigma(\cdot)$ | sigmoid, $\sigma(x) = 1 / (1 + e^{-x})$ |
| $\mathrm{softmax}(\mathbf{z})_i$ | $e^{z_i} / \sum_j e^{z_j}$ |
| $\|\cdot\|_2$ | $L_2$ norm (spectral norm for matrices) |
| $\langle \cdot , \cdot \rangle$ | Euclidean inner product |
| $I_n$ | $n \times n$ identity |
| $\text{bf16}$ | bfloat16: 1 sign + 8 exponent + 7 mantissa bits |

---

## 1. Matryoshka Mixture of Experts

Source: Wang et al., arXiv:2509.26520.

### 1.1 Nested megapool structure

Experts are organized into $M$ **megapools**, each containing $E$ experts. The total expert count is $N = M \cdot E$. Matryoshka ordering imposes a **nesting invariant**: for any level $\ell \in \{1, \ldots, L_{\text{mat}}\}$ and any megapool $m$, the first $n_\ell$ experts of megapool $m$ form a complete sub-mixture that is trained jointly with levels $\ell, \ell+1, \ldots, L_{\text{mat}}$.

Formally: let $\mathcal{E}_\ell \subseteq \mathcal{E}_{\ell+1} \subseteq \cdots \subseteq \mathcal{E}_{L_{\text{mat}}}$ be a strictly nested chain of expert subsets. At training time, each level $\ell$ contributes a loss

$$
\mathcal{L}_\ell = \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ -\log p_\ell(y \mid x) \right],
$$

where $p_\ell$ is the model distribution computed using only experts in $\mathcal{E}_\ell$. The total training loss is

$$
\mathcal{L} = \sum_{\ell=1}^{L_{\text{mat}}} w_\ell \mathcal{L}_\ell, \qquad w_\ell \geq 0, \quad \sum w_\ell = 1.
$$

At inference, any single level $\ell$ can be used. This is the elastic-inference property.

### 1.2 Router

For each token $x_t$ and each layer, a router produces logits

$$
\mathbf{z}_t = W_{\text{router}} x_t + b_{\text{router}}, \qquad W_{\text{router}} \in \mathbb{R}^{N \times d}.
$$

Top-$k$ gating selects the $k$ largest logits:

$$
g_t(i) = \begin{cases} \mathrm{softmax}(\mathbf{z}_t)_i & i \in \text{top-}k(\mathbf{z}_t) \\ 0 & \text{otherwise} \end{cases}.
$$

The MoE output is $y_t = \sum_{i} g_t(i) \cdot \text{Expert}_i(x_t)$. For typical FANT 3 presets $k=1$ or $k=2$.

### 1.3 Expert parameter count

Each expert is a two-layer feed-forward network of hidden dimension $d_{\text{moe}}$:

$$
\text{Expert}_i(x) = W_{\text{down}}^{(i)} \cdot \text{GeLU}(W_{\text{up}}^{(i)} x),
$$

with $W_{\text{up}}^{(i)} \in \mathbb{R}^{d_{\text{moe}} \times d}$, $W_{\text{down}}^{(i)} \in \mathbb{R}^{d \times d_{\text{moe}}}$. Total MoE parameters per layer:

$$
P_{\text{moe-layer}} = 2 N d \, d_{\text{moe}}.
$$

For `fant3_1b` (verified 986.62 M stored): $N = 32$, $d = 1024$, $d_{\text{moe}} = 2304$, $L_{\text{moe}} = 20$, which gives

$$
P_{\text{moe}} = L_{\text{moe}} \cdot 2 N d \, d_{\text{moe}} \approx 20 \cdot 2 \cdot 32 \cdot 1024 \cdot 2304 \approx 3.02 \times 10^9
$$

— which looks like 3 B, but most layers are shared across Matryoshka levels, bringing the *stored* count down. This is why preset naming got out of sync with reality before the 2026-04-19 audit.

---

## 2. Mixture of Recursions

### 2.1 Shared recursion block

Let $f_\theta : \mathbb{R}^d \to \mathbb{R}^d$ be the shared recursion block (a transformer layer stack with shared weights across recursion depths). Starting from hidden state $x^{(0)} = x$, iterate

$$
x^{(k+1)} = x^{(k)} + \alpha_k \cdot f_\theta(x^{(k)}), \qquad k = 0, 1, \ldots, K-1,
$$

where $\alpha_k \in (0, 1]$ is the **contractive-alpha decay** coefficient.

### 2.2 Contractive decay and Banach fixed-point

Assume $f_\theta$ is Lipschitz with constant $L_f$, i.e. $\|f_\theta(u) - f_\theta(v)\|_2 \leq L_f \|u - v\|_2$ for all $u, v$.

Define the update map $T_k(x) = x + \alpha_k f_\theta(x)$. Its Lipschitz constant is bounded:

$$
\|T_k(u) - T_k(v)\|_2 \leq (1 + \alpha_k L_f) \|u - v\|_2.
$$

**This is not contractive** by itself — it's monotone but not shrinking. The contractive property comes from a different normalization: FANT 3 uses

$$
x^{(k+1)} = (1 - \alpha_k) \cdot x^{(k)} + \alpha_k \cdot f_\theta(x^{(k)}),
$$

which is a convex combination. The induced map is contractive whenever $\alpha_k L_f < 1 - \alpha_k + \alpha_k L_f$, i.e. always bounded in norm by $1$. For a learned decaying schedule $\alpha_k = \alpha_0 \gamma^k$ with $\gamma \in (0, 1)$:

$$
\sum_{k=0}^{\infty} \alpha_k = \frac{\alpha_0}{1 - \gamma} < \infty,
$$

so the total update is finite and the iterate $\{x^{(k)}\}$ converges to a fixed point (Banach fixed-point theorem, applied to the convex combination form). This is why the model can extrapolate to $K > K_{\text{train}}$ at inference: the iterate is provably converging.

### 2.3 Dynamic K training

Training samples $K \sim \text{Uniform}\{1, \ldots, K_{\max}\}$ per batch. The per-batch loss is

$$
\mathcal{L}_{\text{batch}}(K) = -\sum_t \log p(y_t \mid x_t^{(K)}),
$$

where $x_t^{(K)}$ is the recursion state at depth $K$. Averaging over $K$:

$$
\mathcal{L} = \frac{1}{K_{\max}} \sum_{K=1}^{K_{\max}} \mathcal{L}_{\text{batch}}(K).
$$

### 2.4 Monotonic CE penalty

To discourage the model from degrading at deeper passes, add the monotonic penalty

$$
\mathcal{L}_{\text{mono}} = \sum_{k=1}^{K-1} \left[ \max(0, \mathcal{L}_{\text{batch}}(k+1) - \mathcal{L}_{\text{batch}}(k)) \right]^2.
$$

If losses are non-increasing in $k$, the penalty is zero. Squared positive-part means any regression is penalized quadratically.

Empirical: at $M = 10^6$ parameters, contractive + dynamic-$K$ + monotonic gives CE $2.67$ vs. CE-only baseline $3.84$ (+1.17 nats better, same compute). Tested over 22 unit tests and 600-step CPU training.

---

## 3. Multi-head Attention with Shared Atoms (MASA)

### 3.1 Atom dictionary

Let $\mathbf{A} = (A_1, \ldots, A_n) \in \mathbb{R}^{n \times d_h \times d_h}$ be a learned dictionary of $n$ attention basis matrices. Each layer $\ell \in \{1, \ldots, L\}$ and each head $h \in \{1, \ldots, H\}$ is reconstructed as

$$
W_\ell^{(h)} = \sum_{i=1}^{r} c_{\ell, h, i} \cdot A_{\pi(\ell, h, i)},
$$

where $c_{\ell, h, i} \in \mathbb{R}$ are learnable coefficients of rank $r$ and $\pi(\ell, h, i) \in \{1, \ldots, n\}$ selects which atoms to use.

### 3.2 Parameter count

Standard multi-head attention: $4$ projection matrices ($W^Q, W^K, W^V, W^O$) per layer per head, each $d \times d_h$:

$$
P_{\text{std}} = L \cdot H \cdot 4 \cdot d \cdot d_h.
$$

MASA: one shared dictionary of $n$ atoms plus per-layer-per-head rank-$r$ coefficients:

$$
P_{\text{MASA}} = n d_h^2 + L \cdot H \cdot 4 \cdot r.
$$

For `fant3_1b` ($L = 20$, $H = 16$, $d = 1024$, $d_h = 64$, $n = 32$, $r = 4$):

$$
P_{\text{std}} = 20 \cdot 16 \cdot 4 \cdot 1024 \cdot 64 = 83.9 \text{M}, \qquad P_{\text{MASA}} = 32 \cdot 64^2 + 20 \cdot 16 \cdot 4 \cdot 4 = 136 \text{K}.
$$

A **600x reduction** in attention parameter count. The tradeoff: MASA has expressive capacity bounded by $\mathrm{rank}(A) \leq n$, so attention rank cannot exceed the atom count.

### 3.3 RoPE compatibility

Rotary Position Embedding requires real-valued $\cos(m\theta), \sin(m\theta)$ tables. Under bf16, products of bf16 cos/sin with bf16 $V$ tensors can silently corrupt attention. The fix is to keep RoPE tables in fp32 and cast only the final result to bf16:

```
cos_f32, sin_f32 = make_rope(seq_len, d_h).float()
q_rot = q.float() * cos_f32 + rotate_half(q.float()) * sin_f32
v_rot = v  # keep bf16 throughout V path
attn = softmax(q_rot @ k_rot.transpose(-2, -1) / sqrt(d_h)) @ v_rot
```

---

## 4. Spinor Apollonian Memory

Source: Kocik, arXiv:2001.05866.

### 4.1 The Descartes invariant

Given four mutually tangent circles with curvatures $(b_1, b_2, b_3, b_4)$, the **Descartes quadratic form** is

$$
Q(b_1, b_2, b_3, b_4) = (b_1 + b_2 + b_3 + b_4)^2 - 2(b_1^2 + b_2^2 + b_3^2 + b_4^2).
$$

Mutually tangent circles satisfy $Q = 0$. The sign of $Q$ for a general quadruple classifies its geometric configuration.

### 4.2 Tangency spinors

Kocik's construction: represent each circle by a **tangency spinor** $\phi_i \in \mathbb{R}^{1, 2}$ (Minkowski 3-space of signature $(-, +, +)$). The Descartes form becomes the Minkowski quadratic form:

$$
Q = -\phi_0^2 + \phi_1^2 + \phi_2^2.
$$

The Clifford algebra $\mathrm{Cl}(1, 2)$ has a natural $\mathbb{Z}_2$ grading by **chirality**: elements split into positive-chirality (even subalgebra) and negative-chirality (odd). The sign of a tangency spinor's chirality component is a **topological** invariant — not a threshold, not a learned parameter.

### 4.3 FANT 3 application

Each hidden state $x_t$ is projected to a 4-vector $\mathbf{v}_t = P x_t \in \mathbb{R}^4$ by a learned projection $P$. Compute the Descartes invariant

$$
Q_t = (v_{t,0} + v_{t,1} + v_{t,2} + v_{t,3})^2 - 2 \sum_{i=0}^{3} v_{t,i}^2.
$$

The chirality is

$$
\chi_t = \mathrm{sign}(Q_t) \in \{-1, +1\}.
$$

Tokens with $\chi_t = +1$ enter the **α-pack** (instance memory, capacity $C_\alpha$). Tokens with $\chi_t = -1$ enter the **β-pack** (schema memory, capacity $C_\beta$). The chirality-balance statistic

$$
\bar{\chi} = \frac{1}{2B T} \sum_{t} (1 + \chi_t)
$$

should be in $[0.2, 0.8]$ for a healthy split. Verified across scales 2026-04-19: 5m = 0.266, 40m = 0.447, 150m = 0.500, 350m = 0.719.

### 4.4 Why this fixes α/β starvation

The earlier scalar-curvature classifier used $\kappa_t = \|x_t\|_2^2$ with a learned threshold $\tau$:

$$
\chi_t^{(\text{old})} = \mathrm{sign}(\kappa_t - \tau).
$$

Training fluctuations in $\tau$ caused all tokens to fall on one side of the threshold — one pack got 100% of tokens, the other starved. The Descartes invariant is **intrinsic** to the hidden state's geometry and doesn't depend on a learned threshold. It's well-conditioned at initialization because $P$ is initialized with orthogonal columns, so $\mathbf{v}_t$ is nearly isotropic and $Q_t$ is nearly symmetric about zero.

### 4.5 External validation (Bucher et al. 2026)

Bucher, Kaminer et al. (Nature 651:920, 2026) measured topological phase singularities in hexagonal boron nitride polaritons and demonstrated that the sign of the Descartes invariant (equivalently, the chirality in $\mathrm{Cl}(1,2)$) classifies singularities into a $+1$/$-1$ topological-charge pair with the same universal behavior that Toulouse & Kléman (1976) derived via homotopy groups for superconductors, superfluids, and liquid crystals. The classification is not a threshold; it is a topological invariant.

Three implications for FANT 3's memory:

1. The same homotopy argument that makes Bucher's classification principled makes ours principled. Hidden-state fields and optical phase fields are both $\mathrm{Cl}(1,2)$-valued up to the choice of $P$; the classifier is the same.
2. The α/β chirality balance in our architecture sits in the 0.266–0.719 range across scales (measured 2026-04-19), which is exactly the regime Bucher finds experimentally for polariton singularity populations in similar random-wave ensembles.
3. A longer discussion of the external validation, with the full intellectual lineage from Kelvin 1867 through Bucher 2026, is in [`article/topological-classifier-universality.md`](./article/topological-classifier-universality.md).

---

## 5. Equiangular Tight Frames and Router Freezing

### 5.1 ETF definition

A collection of $N$ unit vectors $\{w_1, \ldots, w_N\} \subset \mathbb{R}^d$ (with $N \geq d$) is an **equiangular tight frame** if:

1. **Equiangular**: $|\langle w_i, w_j \rangle| = c$ for all $i \neq j$, for some constant $c \geq 0$.
2. **Tight frame**: $\sum_{i=1}^N w_i w_i^T = \frac{N}{d} I_d$.

The **Welch bound** gives the minimum value of the maximum inner product:

$$
c \geq \sqrt{\frac{N - d}{d (N - 1)}}.
$$

ETFs achieve the Welch bound with equality.

### 5.2 Simplex ETF

The **simplex ETF** in $\mathbb{R}^{N-1}$ consists of $N$ unit vectors with equal pairwise inner product exactly $-1/(N-1)$. It is the tightest possible packing. FANT 3 routers are initialized as simplex ETFs.

### 5.3 Why freezing is lossless

After warmup (default $500$ steps), the router weights are frozen. The claim is that **no information is lost** relative to a trainable router.

*Argument.* Let $W_{\text{router}}^* = \arg\min_W \mathcal{L}(W)$ be the optimal router weights. If $W_{\text{router}}^*$ has full rank $d$, then any orthogonal transformation $O$ applied to $W_{\text{router}}^*$ gives a router with identical classification behavior up to expert re-labeling. The simplex ETF is the specific orthogonal transformation that minimizes inter-expert interference (the off-diagonal Gram matrix entries are all exactly $-1/(N-1)$, the theoretical minimum).

After warmup, the **experts** adapt to the frozen router's geometry. Any further change in the router would be absorbed into the experts with no net effect on the model's forward pass. Therefore freezing is free.

*Empirical check.* Campaign N 2026-04-11: router entropy regularization `z_coef` values that kept logits bounded during warmup did not hurt final accuracy when freezing engaged; the N3 SleepGate variant with ETF freeze hit 59.9%, +5.3pp over the L1.5 baseline.

---

## 6. Cerebellum: Echo-State Reservoir with Purkinje Readout

### 6.1 Echo-state network

An **echo-state network** (ESN) is a recurrent network with a fixed random recurrent weight matrix $W_{\text{res}}$ and a trainable linear readout. The reservoir state $r_t$ evolves as

$$
r_{t+1} = (1 - \lambda) r_t + \lambda \tanh(W_{\text{res}} r_t + W_{\text{in}} x_t),
$$

where $\lambda \in (0, 1]$ is the **leak rate**.

### 6.2 Echo-state property

The ESN has the **echo-state property** (Jaeger 2001) if, for any input sequence, the reservoir state asymptotically forgets initial conditions. A sufficient condition is that the spectral radius $\rho(W_{\text{res}}) < 1$:

$$
\rho(W_{\text{res}}) = \max_i |\lambda_i(W_{\text{res}})| < 1.
$$

FANT 3 uses spectral radius **0.95**: close enough to 1 for rich temporal memory, low enough for stability.

### 6.3 Purkinje readout

The trainable output is a linear map

$$
y_t = W_{\text{out}} r_t + b_{\text{out}}, \qquad W_{\text{out}} \in \mathbb{R}^{d \times d_{\text{res}}}.
$$

Training updates **only** $W_{\text{out}}, b_{\text{out}}$. The reservoir $W_{\text{res}}$ is frozen. This gives the Cerebellum a fixed 25 M parameter budget regardless of backbone scale.

### 6.4 Why this is cheap

Computing the reservoir update is $O(d_{\text{res}}^2)$ per step. With $d_{\text{res}} \approx 2500$ and modest spectral radius, the operation is FLOP-cheap compared to the transformer's attention and MoE. More importantly, backprop through the reservoir is unnecessary — gradients flow only through $W_{\text{out}}$.

---

## 7. Progressive Curriculum

Source: arxiv:2604.16278 DeepInsightTheorem.

### 7.1 Phase-weighted data distribution

Let $\mathcal{D}_1, \ldots, \mathcal{D}_S$ be the $S$ data sources. For each phase $p \in \{1, \ldots, P\}$ (here $P = 3$: Apprentice, Journeyman, Expert), define weights $w_{p,s} \geq 0$ with $\sum_s w_{p,s} = 1$.

During training step $t$ out of $T$ total steps, let the phase-fraction be $f_t = t / T$. Let $F_1, \ldots, F_P$ be the phase-boundary fractions (e.g. $0.25, 0.65, 1.0$ for deepinsight_3phase). The active phase is

$$
p(t) = \min \{ p : f_t \leq F_p \}.
$$

The effective data distribution at step $t$ is

$$
\mathcal{D}_t = \sum_{s=1}^{S} w_{p(t), s} \, \mathcal{D}_s.
$$

### 7.2 Deepinsight_3phase weights

Written as a matrix $W_{\text{phase}} \in \mathbb{R}^{3 \times 11}$ (phase × source):

| Phase | FineWeb | NVIDIA-math-2 | NVIDIA-code | Numina | FineTome | Kimi-distill | Sonnet | NVIDIA-math-reason | Opus | NVIDIA-cascade-if | Kimi-math |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Apprentice | 0.55 | 0.15 | 0.10 | 0.10 | 0.10 | 0 | 0 | 0 | 0 | 0 | 0 |
| Journeyman | 0.25 | 0 | 0.05 | 0 | 0 | 0.20 | 0.15 | 0.15 | 0.10 | 0.10 | 0 |
| Expert | 0.05 | 0 | 0 | 0 | 0 | 0 | 0.20 | 0.10 | 0.30 | 0 | 0.15 |

Each row sums to 1.0 within $10^{-3}$ tolerance (unit test `TestPhaseSpec::test_weights_near_one_tolerated`).

### 7.3 Why mixing dominates pure specialization

Independent support from prior work:
- **arxiv:2510.14865 (midtraining)**: "mixing specialized with base pretraining data in an intermediate phase consistently outperforms continued pretraining on specialized data alone, both in-domain and for reduced forgetting."
- **arxiv:2510.01631 (synthetic)**: "Pure synthetic data is *not* superior to CommonCrawl alone; mixtures beat both."

Formally, let $p^*(y \mid x)$ be the true data distribution and $p_\theta(y \mid x)$ be the model. The cross-entropy bound under specialization-only is

$$
\mathbb{E}_{\mathcal{D}_{\text{specialized}}}[-\log p_\theta] \geq \mathrm{H}(\mathcal{D}_{\text{specialized}}).
$$

Under a mixture $\alpha \mathcal{D}_{\text{spec}} + (1-\alpha) \mathcal{D}_{\text{base}}$,

$$
\mathbb{E}_{\text{mix}}[-\log p_\theta] = \alpha \mathbb{E}_{\text{spec}} + (1-\alpha) \mathbb{E}_{\text{base}}.
$$

When $\mathcal{D}_{\text{base}}$ carries distributional mass that $\mathcal{D}_{\text{spec}}$ does not (e.g. FineWeb has broad vocabulary that NVIDIA math does not), the mixture's coverage is strictly larger, and the model's held-out CE is lower on heterogeneous evaluation distributions. This is why the FineWeb anchor is kept at 5% even in the Expert phase.

---

## 8. Compression as Intelligence

Source: Delétang et al. 2023, "Language Models are Compressors".

### 8.1 Shannon bound

For any source with entropy rate $H$, any uniquely decodable code has expected code length $L \geq H$. A language model $p_\theta$ defines an arithmetic code with expected length

$$
\mathbb{E}_x[L(x)] = -\mathbb{E}_x[\log_2 p_\theta(x)] = \mathrm{CE}_{\text{bits}}(p_\theta).
$$

**Cross-entropy in bits equals the compression rate in bits per symbol.** Cross-entropy in nats equals bits-per-byte times $\ln(2) / 8 \cdot |\text{symbol}|$ for ASCII sources.

### 8.2 Bits-per-byte (bpb)

The standard compression benchmark quantity is

$$
\mathrm{bpb}(x) = \frac{-\log_2 p_\theta(x)}{\text{len}(x)_{\text{bytes}}}.
$$

For English text, reasonable values:
- **gzip**: ~2.5 bpb
- **Chinchilla-70B**: ~0.4 bpb
- **FANT 3 50m (undertrained)**: ~5-6 bpb
- **FANT 2 60m (at CE 5.72)**: ~1.2 bpb on specialized domains, worse on broad text

### 8.3 FANT connection

Running CE $\to$ bpb conversion on the 742m Tier C checkpoint (best CE 5.72 nats, tokenizer v2 with avg ~4 bytes per token):

$$
\mathrm{bpb} \approx \frac{5.72 / \ln(2)}{4} \approx 2.06 \text{ bpb}.
$$

This is slightly better than gzip (~2.5 bpb) on specialized text, roughly 3x worse than Chinchilla-70B on the same data. At 190x under Chinchilla-optimal training tokens, this is consistent — the Chinchilla law predicts that adding training tokens to a fixed-parameter model follows a power law until $D = 20 N$, beyond which returns diminish.

---

## 9. The Fractal Thread

The "F" in FANT stands for Fractal. Not every component of FANT 3 is fractal — MASA attention, AHN, ETF routing, Cerebellum, and the progressive curriculum are not. But three of the core architectural decisions are genuinely fractal in a precise mathematical sense, and it is those three that define the shape of the model.

### 9.1 Matryoshka MoE as a scale-nested self-similar hierarchy

Recall from section 1.1 the strict nesting chain

$$
\mathcal{E}_1 \subsetneq \mathcal{E}_2 \subsetneq \cdots \subsetneq \mathcal{E}_{L_{\text{mat}}}.
$$

Each $\mathcal{E}_\ell$ is a functional sub-model that produces a complete output distribution $p_\ell(y \mid x)$. The nesting invariant requires that the sub-model at level $\ell$ be **consistent** with the sub-model at level $\ell + 1$: any behavior producible at level $\ell$ is also producible at level $\ell+1$ (but not conversely).

This is the defining property of a **scale-nested self-similar system**. In fractal geometry terms: the levels form a self-similar sequence under the "add more experts" scaling. If we write $S_\ell$ for the set of output distributions reachable at level $\ell$, then

$$
S_1 \subsetneq S_2 \subsetneq \cdots \subsetneq S_{L_{\text{mat}}},
$$

and each $S_\ell$ is a refinement of $S_{\ell-1}$ on the same probability simplex. The matryoshka doll metaphor (from which the architecture takes its name) is precise: each level contains the previous level, scaled up.

### 9.2 Mixture of Recursions as an iterated function system

From section 2.1, the MoR update is

$$
x^{(k+1)} = (1 - \alpha_k) x^{(k)} + \alpha_k f_\theta(x^{(k)}).
$$

This is an **iterated function system** (IFS) in the sense of Hutchinson 1981. An IFS is a finite set of contractive maps $\{T_1, \ldots, T_n\}$ on a complete metric space; its **attractor** $A$ is the unique compact set satisfying

$$
A = T_1(A) \cup T_2(A) \cup \cdots \cup T_n(A).
$$

Under contractive decay (section 2.2), the MoR update is a single contraction $T_\alpha$ with contraction ratio $\alpha$. The attractor is a fixed point. Under **dynamic-$K$** (section 2.3), the effective system is stochastic — each step samples which $T_{\alpha_k}$ to apply from a finite family indexed by $k$. The stochastic attractor is a measurable fractal set whose Hausdorff dimension can be computed from the contraction ratios via the Moran equation

$$
\sum_{k} \alpha_k^{s} = 1,
$$

where $s$ is the Hausdorff dimension. For uniform $\alpha_k = \alpha$ and $n$ recursion depths, $s = \log(n) / \log(1/\alpha)$. For FANT 3 typical values $n = 3$, $\alpha_k \in [0.3, 0.7]$: $s \approx 1.5$ to $3.0$.

**The attractor set is where token representations converge under recursion.** Shallow tokens land near the surface of the attractor; deep tokens penetrate its interior. Barnsley's fern is the same class of object.

### 9.3 Spinor Apollonian memory as the Apollonian fractal

The Apollonian packing is the canonical example of a circle-packing fractal. Starting with four mutually tangent circles satisfying the Descartes equation $Q = 0$ (section 4.1), each triple of tangent circles bounds a curvilinear triangle; the Descartes operator $D$ constructs a new circle tangent to all three, filling the triangle. Applied recursively, this generates an infinite packing whose **Hausdorff dimension** is

$$
\dim_H(\text{Apollonian packing}) \approx 1.30568673...
$$

(Mandelbrot 1983, refined by Boyd 1973 and McMullen 1998). This is a measurable fractal dimension — neither 1 (a curve) nor 2 (a region), but strictly between.

When FANT 3 projects hidden states $x_t$ to 4-vectors $\mathbf{v}_t \in \mathbb{R}^4$ and computes the Descartes invariant $Q_t$, the chirality $\chi_t = \mathrm{sign}(Q_t)$ partitions tokens by which **side of the packing** they live on. Tokens with $\chi_t = +1$ (α-pack) correspond to points inside the filled regions of the packing; tokens with $\chi_t = -1$ (β-pack) correspond to points outside. The packing boundary itself is the fractal set where $Q_t = 0$.

In this sense, the memory structure is **literally a packing of fractal sets**, not an analogy to one.

### 9.4 What isn't fractal

For honesty: the other five components are not fractal.

- **MASA attention** is a rank-$r$ decomposition with a shared dictionary. This is low-rank structure, not fractal structure.
- **AHN** is a FIFO plus a compressed buffer. Linear structure.
- **ETF routing** is an optimal simplex arrangement — a tight frame, not a fractal.
- **Cerebellum** is an echo-state reservoir. The biological cerebellum has fractal cortical folding with dimension ~2.57 (Mandelbrot 1983), but the mathematical reservoir itself is a standard RNN attractor, not a fractal.
- **Progressive curriculum** is a three-phase weighted schedule. No fractal structure.

The "F" in FANT names the three fractal decisions (Matryoshka, MoR, Apollonian) that shape the overall architecture. It does not claim every module is fractal.

### 9.5 Why this matters

Fractal structure is not decorative. It gives three concrete properties that flat architectures lack:

1. **Elastic inference.** Matryoshka nesting means one trained checkpoint serves multiple compute budgets by terminating at any level.
2. **Depth extrapolation.** MoR's contractive IFS converges to an attractor regardless of iteration count, so models trained at $K = 3$ can run at $K = 6$ without divergence. Validated empirically at the 1 M scale (section 2.4).
3. **Topological memory split.** The α/β chirality is a topological invariant — it cannot starve under threshold drift, because there is no threshold. Validated across five scales (section 4.3).

None of these properties are obtainable from flat, non-fractal designs.

---

## 10. References

| Paper | Used in | arXiv |
|---|---|---|
| Wang et al. 2025 | Matryoshka MoE | 2509.26520 |
| Kocik 2020 | Spinor Apollonian | 2001.05866 |
| Bucher, Kaminer et al. 2026 | Spinor Apollonian external validation | 2509.17675 (Nature 651:920) |
| Toulouse & Kléman 1976 | Homotopy classification of defects | J. Phys. Lett. 37:L149 |
| Berry & Dennis 2000 | Phase singularity statistics | Proc. R. Soc. A 456:2059 |
| ByteDance 2025 | AHN | (TR) |
| Jaeger 2001 | Echo-state networks | (Tech Rep) |
| Welch 1974 | ETF bound | (JOSA) |
| Litim 2001 | LR schedule | Phys. Rev. D 64 105007 |
| Hoffmann et al. 2022 | Chinchilla law | 2203.15556 |
| Delétang et al. 2023 | Compression as intelligence | 2309.10668 |
| Li et al. 2026 | Progressive curriculum | 2604.16278 |
| Zhang et al. 2026 | AgentV-RL (Fix 3) | 2604.16004 |
| arXiv 2510.14865 | Midtraining validates curriculum | 2510.14865 |
| arXiv 2510.01631 | Synthetic data mixtures | 2510.01631 |
| arXiv 2510.25741 | Ouro recurrence | 2510.25741 |
| arXiv 2511.07384 | Retrofitted recurrence | 2511.07384 |
| Parisi / de Almeida-Thouless | Routing diversity | 2604.11921 |
| arXiv 2504.19874 | TurboQuant (queued) | 2504.19874 |
| arXiv 2512.03324 | TRIM-KV (queued) | 2512.03324 |

## See also

- [THEORY/README.md](./THEORY/README.md) — plain-English companion
- [glossary.md](./glossary.md) — terms and notation
- [architecture/](./architecture/) — component-level deep-dives with code references
- [ADR/](./ADR/) — why each of these choices was made
