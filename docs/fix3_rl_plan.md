# Fix 3 — RL Post-Pretrain Plan

Status: **deferred (paper-trail only)**. Not blocking current pretrain.

## Context

Fix 3 was scoped during the FANT 3 launch as "GSPO RL to close the SFT→RL
gap after pretrain converges." During the 2026-04-24 paper review against
two April-2026 arxiv papers, the plan was updated to incorporate findings
from arxiv:2604.16004 **AgentV-RL** (Fudan + ByteDance Seed).

## Source papers

1. **GSPO / G2RPO-A** — the family of group-relative policy optimization
   with asymmetric clipping and dynamic zero-variance filtering. Already
   implemented in `fant3/training/` under the N6 code path. This is the
   algorithm half.
2. **arxiv:2604.16004 AgentV-RL** — "Scaling Reward Modeling with Agentic
   Verifier." Qwen3-4B verifier with Forward + Backward Plan-Validate-Verdict
   agents, Python tool use, GRPO training. Beats Llama3.1-70B ORM by +25.2pp
   on MATH500 at N=128 Best-of-N. This is the reward half.

## The gap our original Fix 3 left

Original Fix 3 plan: direct outcome-reward GRPO. Reward = binary match
against gold answer extracted from `<|answer|>...<|/answer|>` tags.

AgentV-RL shows this is leaving up to +25.2pp on the table at MATH-class
benchmarks. The missing ingredient is a trained **agentic verifier** that
interleaves reasoning with tool-grounded checks and produces verdicts with
far higher accuracy than pattern-match or single-pass LLM-as-judge.

## Updated plan

Add a verifier stage between pretrain and RL:

### Stage 0 — pretrain (current)

Already in motion. Progressive curriculum (arxiv:2604.16278) via
`--curriculum deepinsight_3phase` reaches steady CE on reasoning data.

### Stage 1 — verifier distillation

- **Teacher:** AgentV-RL's released Qwen3-4B verifier
  (forward-only variant to halve cost; per paper ablation, forward-only is
  "competitive")
- **Student:** our FANT 3 pretrain checkpoint (1B stored / 100M active)
- **Data:** rollouts from our student on ~10K math/code problems, each
  labeled by the Qwen3-4B verifier
- **Training:** rejection-sampling SFT on verdicts matching ground truth
  (paper's recipe: 15K samples, SFT then GRPO)
- **Artifacts:** separate verifier head trained from the pretrain backbone
  (tied embeddings; add a classification head on top of `<|verdict|>` token)

### Stage 2 — GSPO with verifier-shaped reward

- **Algorithm:** GSPO / G2RPO-A (existing N6 code)
- **Reward composition:**
  - `r_outcome` = +1 if gold match, -1 otherwise (existing)
  - `r_verifier` = +1/-1 from the trained verifier on the rollout
  - `r_total = (r_outcome + r_verifier) / 2` (simple average; per paper
    ablation, forward-only verifier contributes most of the gain)
- **Filter:** dynamic zero-variance group filter (DAPO / Yu et al. 2025),
  already supported by N6 code
- **Clip:** asymmetric `clip(1-ε_low, 1+ε_high)` (existing)

## Design decisions

### Why forward-only instead of forward+backward?

Paper ablation shows forward-only and backward-only are both "competitive";
full bidirectional is best by only a small margin. Forward-only halves
verifier inference cost (8349 tok × 11.3 rounds ≈ 94K tokens per rollout
vs ~47K for forward-only). At our inference budget, accept the ~2pp margin
loss for 2× throughput. Revisit if throughput ends up not being the
bottleneck.

### Why not train our own verifier from scratch?

The paper demonstrated a 4B verifier beats a 70B ORM. Training our own 1B
verifier would reproduce that result but burn weeks. Distilling from the
released 4B is faster and anchors us to a known-good ceiling.

### Why not Python tool-use?

Paper ablation: "tool-free variant already significantly beats base." The
agentic framework (Plan-Validate-Verdict loop) dominates the gain, not the
Python interpreter. Tool integration requires a sandboxed execution path
in our training harness we don't currently have. Defer until we have a
concrete throughput justification.

### Why N=1 inference instead of N=128?

The paper's +25.2pp headline result was at N=128 Best-of-N. Our deployment
budget is N=1–8. We do not yet know how the gain scales down. **Before
committing to the full Stage 1+2 plan, run a cheap ablation:** fine-tune a
tiny verifier on a few-hundred-sample label set, measure gain at N=1, N=8,
N=32 on a MATH subset. If the gain collapses to <5pp at N=1–8, reconsider
whether verifier RL is worth the complexity vs simpler outcome-reward GRPO.

## Blocking items

- Current pretrain run (50m-unlimited, per
  `project_50m_unlimited_2026_04_23.md`) must complete to a usable
  checkpoint first.
- Progressive curriculum test run (this repo, `--curriculum
  deepinsight_3phase`) must show non-regression vs legacy_2phase before
  we pivot the main run.
- Verifier N=1 ablation (see above) — tiny verifier on held-out MATH,
  measure gain-vs-cost at our inference budget.

## Risks

1. **Verifier reward-hacking.** The paper flags this as mitigated by tool
   grounding. Without tools we lose that mitigation. Mitigation: keep the
   outcome reward in the total (weight 0.5) so gaming the verifier alone
   still under-scores the rollout.
2. **Verifier quality floor.** Qwen3-4B is a stronger-than-ours verifier;
   distilling into a 1B FANT student may cap below the teacher. Mitigation:
   measure student-verifier agreement with ground truth on a held-out set
   before using it as a reward signal. Reject the plan if agreement <85%.
3. **Inference cost.** 47K tokens per verified rollout × 50K rollouts =
   2.35B tokens of verifier inference. At our RunPod A100 rate
   (~100 tok/sec for 1B model) that's ~6.5 hours just to label the RL
   dataset. Budget that explicitly.

## Open questions

- How to integrate the verifier head into the FANT 3 model architecture?
  Options: (a) separate 1B verifier model, (b) shared backbone with
  verification-specific adapter heads, (c) train one 1B model that
  alternates roles via prompt. Paper used option (a).
- Does the verifier need its own tokenizer? Our tokenizer_v2 has
  `<|think|>` / `<|answer|>` but no `<|verdict|>` / `<|critic|>` markers.
  Adding them is a vocab change; may need tokenizer_v3.

## When to revisit

- When a pretrain checkpoint is available (post current 50m run
  completion)
- Or: when a test of the progressive curriculum shows the data-mix lever
  has plateaued and further gains require RL

## References

- Memory: `research_agentv_rl_2026_04_24.md` — paper summary with numbers
- Memory: `project_post_paper_review_actions_2026_04_24.md` — action ranking
- Memory: `project_fant3_implementation_2026_04_16.md` — FANT 3 design with
  Fix 3 scope
