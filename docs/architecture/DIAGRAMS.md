# FANT 3 Architecture Diagrams

All diagrams use Mermaid syntax and render in GitHub. Open any `.md` file in GitHub to see them as visual flow charts.

---

## Diagram 1: Full Forward Pass

This diagram shows a single forward call through `FANT3Model.forward()` for the `fant3_742m` preset (16 layers: 2 dense prefix, 11 shared middle via MoR, 3 suffix MoE).

```mermaid
flowchart TD
    A["input_ids ∈ ℤ^(B×T)"] --> B["tok_emb: Embedding(vocab_size, dim)\n(B, T) → (B, T, dim)"]

    B --> C0["DenseBlock 0\nMASAAttention + RMSNorm + DenseSwiGLU + RMSNorm\nresidual connections throughout"]
    C0 --> C1["DenseBlock 1\n(same structure, different per-layer\nMASA coefficients)"]

    C1 --> D["MoRShared\n— MoRDepthRouter chooses depth 1 or 2 per token\n— Shared MoEBlock applied 1..2 times\n— Active mask written back after each pass"]

    D --> E0["Suffix MoEBlock 0\nMASAAttention + RMSNorm + MatryoshkaMoEFFN + RMSNorm\n(distinct weights from shared middle block)"]
    E0 --> E1["Suffix MoEBlock 1"]
    E1 --> E2["Suffix MoEBlock 2\n← Memory retrieval from SpinorApollonianMemory\n   augments attention in retrieval layers"]

    E2 --> F{"cerebellum_enabled?"}
    F -- "Yes (default)" --> G["CerebellumModule\ncereb_in_proj → reservoir loop → purkinje\n+= sigmoid(cereb_gate) × output"]
    F -- "No" --> H
    G --> H{"ahn_enabled?"}

    H -- "Yes (default)" --> I["ArtificialHippocampusNetwork\nshort-term window + compressed long-term memory\n+= sigmoid(ahn_gate) × output"]
    H -- "No" --> J
    I --> J["RMSNorm (final_norm)"]

    J --> K["lm_head: Linear(dim, vocab_size)\n(weight tied to tok_emb.weight)"]
    K --> L["logits ∈ ℝ^(B×T×vocab_size)"]

    L --> M{"targets provided?"}
    M -- "Yes" --> N["CrossEntropy loss\n(ignore_index=-100 for padding)"]
    M -- "No" --> O["return logits only"]

    style A fill:#dde,stroke:#aac
    style L fill:#ded,stroke:#aac
    style N fill:#fdd,stroke:#aac
```

---

## Diagram 2: MoR (Mixture of Recursions) Recursion Flow

This diagram zooms into `MoRShared.forward()` showing how per-token depth routing works. The shared `MoEBlock` is called `max_depth` (2) times; only tokens that need more compute receive subsequent passes.

```mermaid
flowchart TD
    A["x ∈ ℝ^(B×T×dim)\nfrom dense prefix blocks"]

    A --> B["MoRDepthRouter\nfc1: (dim → mor_router_dim) + SiLU\nfc2: (mor_router_dim → n_recursion_depths)\nsoftmax → argmax → depth ∈ {1, 2} per token"]

    B --> C["depth tensor shaped (B, T)\nValue 1: token gets 1 pass\nValue 2: token gets 2 passes"]

    A --> D["Pass 1: call shared MoEBlock(current)\n→ next_state"]
    C --> E["active mask: depth >= 1  (all tokens True on pass 1)\ncurrent = where(active, next_state, current)"]
    D --> E

    E --> F["Pass 2: call shared MoEBlock(current)\n→ next_state"]
    E --> G["active mask: depth >= 2  (only depth-2 tokens True)\ncurrent = where(active, next_state, current)"]
    F --> G

    G --> H["Output: current ∈ ℝ^(B×T×dim)\nDepth-1 tokens: processed by shared block once\nDepth-2 tokens: processed by shared block twice"]

    H --> I["return (current, router_info)"]

    note1["Note: gradient checkpointing wraps\neach pass call independently\nwhen use_gradient_checkpointing=True"]

    style A fill:#dde,stroke:#aac
    style H fill:#ded,stroke:#aac
    style note1 fill:#fff,stroke:#ccc,stroke-dasharray:5 5
```

---

## Diagram 3: Matryoshka MoE (Mixture of Experts) Routing

This diagram shows `MatryoshkaMoEFFN.forward()`. Each token independently selects a megapool and a nesting level that determines how many experts in that pool are activated.

```mermaid
flowchart TD
    A["x ∈ ℝ^(B×T×dim)\nfrom MASAAttention + RMSNorm"]

    A --> B["Flatten: (B×T, dim)"]

    B --> C["megapool_proj: Linear(dim, n_megapools)\n+ megapool_bias buffer\n→ softmax → mp_probs\n→ argmax → mp_idx ∈ {0..3}"]

    B --> D["level_proj: Linear(dim, n_matryoshka_levels)\n+ level_bias buffer\n→ softmax → lv_probs\n→ argmax → lv_idx ∈ {0..1}"]

    C --> E["Dispatch: group tokens by (mp_idx, lv_idx)"]
    D --> E

    E --> F0["Level 0 tokens (depth=1)\nActivate expert 0 in their megapool\nexpert_id = mp_idx × n_per_megapool + 0\nSwiGLU(x, W_up[id], W_down[id])"]

    E --> F1["Level 1 tokens (depth=2)\nActivate experts {0, 1} in their megapool\nUniform weight 1/2 each\nSwiGLU aggregated sum"]

    F0 --> G["Aggregate: scatter results back to (B×T, dim)\nusing active-level mask"]
    F1 --> G

    G --> H["Shared expert (always active)\nLinear(dim, 2×shared_hidden) → SiLU gate\n→ Linear(shared_hidden, dim)\n+= sigmoid(shared_gate) × shared_output"]

    H --> I["Reshape: (B, T, dim)\nreturn (out, router_info)"]

    style A fill:#dde,stroke:#aac
    style I fill:#ded,stroke:#aac
```

---

## Diagram 4: SpinorApollonianMemory Store and Retrieve

This diagram shows the full lifecycle of the spinor-based long-term memory: how embeddings are classified and stored (during Phase 4+ training), and how queries retrieve from both packs.

```mermaid
flowchart TD
    subgraph STORE ["store(embeddings, hidden_preRMSnorm)"]
        S1["hidden_preRMSnorm ∈ ℝ^(N×dim)\n(pre-RMSNorm hiddens carry richer signal\nthan post-norm embeddings)"]
        S1 --> S2["proj_spinor: Linear(dim, 2, bias=False)\nh → s = (s₀, s₁) ∈ ℝ²"]
        S2 --> S3["curvature = s₀² + s₁²\n(Clifford Euclidean norm, for monitoring)"]
        S2 --> S4["chirality = sign(s₁)\n> 0 → α pack (instance, recent)\n≤ 0 → β pack (schema, stable)"]
        S4 --> S5["α FIFO buffer\ncapacity = alpha_cap (10000)\nFIFO eviction of oldest entry"]
        S4 --> S6["β FIFO buffer\ncapacity = beta_cap (10000)\nFIFO eviction of oldest entry"]
    end

    subgraph RETRIEVE ["retrieve(query, top_k, pool)"]
        R1["query ∈ ℝ^(B×T×dim)"]
        R1 --> R2["proj_spinor(query) → q_spinors ∈ ℝ^(N×2)"]
        R1 --> R3["Gather pool: α + β or one pack\n→ pool_emb (M, dim), pool_sp (M, 2)"]
        R2 --> R4["Score = 0.7 × cos_sim(q_emb, pool_emb)\n+ 0.3 × clifford_bilinear(q_sp, pool_sp)\nclifford_bilinear(a,b) = a₀b₀ − a₁b₁"]
        R3 --> R4
        R4 --> R5["top_k selection over M candidates\n→ values (B,T,k,dim), scores (B,T,k)"]
    end

    S5 & S6 --> RETRIEVE
    STORE --> S5
    STORE --> S6

    style STORE fill:#f0f4ff,stroke:#88a
    style RETRIEVE fill:#f0fff4,stroke:#8a8
```

---

## Diagram 5: Scale-Aware Config and Training Recipe Selection

This diagram shows how the Colab notebook maps a `TARGET_SCALE` string to a model config and a matching training recipe.

```mermaid
flowchart TD
    A["TARGET_SCALE string\ne.g. '20m', '50m', '150m', '742m', '1b'"]

    A --> B{{"Match scale"}}

    B -- "smoke" --> C1["fant3_smoke()\ndim=512, n_layers=8\n~40M stored\nrecipe: B=1 T=512 steps=200 lr=2e-4"]
    B -- "20m" --> C2["fant3_20m()\ndim=320, n_layers=10\n~23.5M stored\nrecipe: B=2 T=512 steps=5000 lr=2e-4"]
    B -- "50m" --> C3["fant3_50m()\ndim=384, n_layers=12\n~50.8M stored\nrecipe: B=2 T=1024 steps=5000 lr=2e-4"]
    B -- "150m" --> C4["fant3_smoke() with adjustments\n(intermediate scale)\nrecipe: B=2 T=512 steps=2500 lr=1.5e-4"]
    B -- "742m" --> C5["fant3_742m()\ndim=1024, n_layers=16\n~770.9M stored\nrecipe: B=1 T=1024 accum=8 steps=10000\nlr=1.5e-4 warmup=1500\ngc=True auto-enabled"]
    B -- "1b" --> C6["FANT3Config() defaults (fant3_1b)\ndim=1024, n_layers=20\n~986.6M stored\nrecipe: B=2 T=1024 accum=4 steps=12000\nlr=1.2e-4 warmup=1800 gc=True"]

    C1 & C2 & C3 & C4 & C5 & C6 --> D["FANT3Model(cfg) instantiated\nParam count verified\n(stored vs active distinction)"]

    D --> E["8-source training mix\nFineWeb 30% + Sonnet 20% + Opus 15%\n+ Kimi 10% + NVIDIA 8% + Numina 7%\n+ FineTome 5% + Superior 5%"]

    D --> F["Optimizer: AdamW 8-bit (bnb)\nGradient checkpointing (if gc=True)\nLR schedule: linear warmup → cosine decay"]

    style A fill:#dde,stroke:#aac
    style D fill:#ded,stroke:#aac
```

---

## Diagram 6: Data Pipeline

This diagram shows how raw data flows from HuggingFace dataset streams through format extraction, decontamination, tokenization, and into the model forward pass.

```mermaid
flowchart TD
    A1["HuggingFace streaming datasets\n(8 sources in 11-source MIX v3)"]
    A2["NVIDIA Cascade-2 (chat/code/math)"]
    A3["Kimi K2.5 / Opus 4.6 / Sonnet 4.6\ndistillation traces (JSONL cache)"]

    A1 & A2 & A3 --> B["InterleavedMultiDatasetStream\nWeighted interleaving by source weight\n(FineWeb 30%, Sonnet 20%, …)"]

    B --> C["formats.py — format extractor\nSupports 6 schema variants:\n· PROBLEM_THINK_SOLUTION\n· PROBLEM_SOLUTION\n· CHAT / MESSAGES\n· FINEWEB (plain text)\n· PROBLEM_ANSWER\n· JSONL (cached traces)"]

    C --> D["Output text in canonical format:\n<|problem|> … <|think|> … <|answer|> … <|eos|>\nor plain text for FineWeb"]

    D --> E["decontaminate.py — 13-gram SHA-1 filter\nCheck against ngram_hashes.json\n(457910 hashes from GSM8K + MATH-500 + MMLU)\nContaminated samples → skip"]

    E --> F["tokenizer_v2.json (BPE, vocab_size=32768)\nTrained on 82K docs, 6-source mix\n10–18% compression gain over v1"]

    F --> G["token IDs (B, T)\ntargets = ids shifted left by 1\ntargets[ids == PAD_ID] = -100"]

    G --> H["FANT3Model.forward(input_ids, targets)\nCE loss computed, gradients flow\n(ignore_index=-100 masks padding)"]

    style A1 fill:#ffe,stroke:#aa8
    style A2 fill:#ffe,stroke:#aa8
    style A3 fill:#ffe,stroke:#aa8
    style H fill:#ded,stroke:#8a8
```
