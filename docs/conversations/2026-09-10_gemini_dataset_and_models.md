# Conversation: MoE Models & Dataset Strategy

**Date:** September 10, 2026  
**Participants:** Steve (user), Gemini (Claude)  
**Topic:** Selecting appropriate MoE models and benchmarking datasets for expert prefetching research  
**Source:** Gemini conversation (22 pages)

---

## Key Decisions Documented

### Models Discussed

#### Primary Target: Mixtral 8x7B

**Specification:**
- 8 total experts per layer
- Top-2 routing (2 active experts per token)
- Well-documented, accessible, manageable memory footprint
- **Status:** Primary target for Phase 1 MVP

**Rationale:** Fast iteration, clear documentation, allows proof-of-concept before scaling

---

#### Qwen Models with MoE

**Qwen1.5-MoE** (lighter alternative)
- Lower overhead than Qwen3
- Good for initial testing and development

**Qwen2.5-Coder-32B / Qwen3-Coder** (dense alternative)
- Code-specialized, no MoE routing complexity
- Lightweight deployment if MoE not required

**Qwen3-VL (Vision-Language MoE)**
- 128 total experts, activating 8 per token
- **Major differences:**
  1. **Much larger expert pool** (128 vs 8) → harder prediction, higher potential speedup
  2. **Multimodal routing** → visual tokens + text tokens create new analysis dimension
- **Impact:** Could test hypothesis that image/video inputs route to dedicated expert subsets separate from text
- **Implementation:** Phase 1 hooks stay identical; only data loading changes to handle images

---

#### DeepSeek Models

**DeepSeek-V3** (extreme scale)
- 257 experts per layer (256 routed + 1 shared always-on)
- Top-8 routing (8 of 256 selected)
- Purely text-focused (no multimodal inputs)
- **Use case:** Final production test after MVP validation
- **Challenge:** Massive memory overhead for initial POC

**DeepSeek-Coder-V2-Lite** ⭐ (recommended alternative)
- 64 routed experts
- Top-6 routing (6 of 64 selected)
- 16B total params, 2.4B active per token
- **Advantage:** Sweet spot between complexity and hardware constraints
  - Runs on single 40GB GPU
  - Granular enough (64 experts) to prove predictor works
  - Significant speedup potential (predicting 6/64)
  - Strong performance on code/reasoning tasks
- **Status:** Recommended as "Goldilocks" option if MVP succeeds

**DeepSeek-R1 / DeepSeek-R1-Distill-Llama-70B**
- Dense (non-MoE) but includes reasoning trace generation
- "Thinking mode" for multi-step logical chain generation

---

#### Snowflake Arctic

**Specification:**
- 128 total experts
- Top-2 routing (2 of 128 selected)
- Enterprise-focused, tuned for coding/SQL/logic
- **Use case:** Final scale test, rigorous predictor validation
- **Challenge:** Massive memory requirements; better for production than POC

---

### Model Selection Framework

**The Decision Matrix:**

| Tier | Model | Experts | Active | Use Case | Constraints |
|------|-------|---------|--------|----------|------------|
| **🥇 MVP** | Mixtral 8x7B | 8 | 2 | Proof-of-concept, baseline | Simple but less impressive speedup |
| **⭐ Recommended** | DeepSeek-V2-Lite | 64 | 6 | Strong POC, then scale | 40GB GPU, best balance |
| **📈 Next** | Qwen3-MoE | 128 | 8 | Scaling test post-MVP | More memory |
| **🚀 Final** | Snowflake Arctic | 128 | 2 | Production validation | Massive memory, post-MVP |
| **🔬 Extreme** | DeepSeek-V3 | 256 | 8 | Ultimate scale test | Prohibitive memory |

---

### Migration Path: Mixtral → DeepSeek

**You can absolutely develop on Mixtral and port to DeepSeek with minimal changes.**

#### What Stays Identical

1. **Hook mechanism:** Forward hook on linear gating layer (logits before top-k selection)
2. **Parquet schema:** Same columns (prompt_id, token_idx, layer_idx, expert_probs, selected_experts)
3. **Prefill determinism:** Both produce consistent routing for static inputs (single-pass collection sufficient)
4. **Pipeline logic:** Data storage, validation, analysis all portable

#### What Changes When Switching to DeepSeek-V2-Lite

1. **Module Paths**
   - Mixtral: `model.layers[i].block_sparse_moe.gate`
   - DeepSeek: `model.layers[i].mlp.gate`
   - (Simple find-replace in hook attachment code)

2. **Array Sizing**
   - Mixtral: 8 experts, select 2
   - DeepSeek-V2-Lite: 64 experts, select 6
   - Resize probability arrays and selected-expert indices

3. **Shared Experts**
   - DeepSeek isolates "shared experts" (always activated for common knowledge)
   - These bypass the router entirely
   - **Telemetry only tracks 64 routed experts** (gate-evaluated ones)

---

## Dataset Selection & Sourcing

### GSM8K (Grade School Math)

**Overview**
- 8,500 crowdsourced elementary/middle-school arithmetic problems
- Focus: addition, subtraction, multiplication, division, percentages, rates, fractions
- Ground truth includes step-by-step solutions with intermediate calculations marked (`<<...>>`)

**Slicing Strategies** (no pre-assigned metadata)
- **By step count (complexity):** Count `\n` or `<<...>>` blocks in answer field
  - Simple: 2–3 steps
  - Hard: 6–8 steps
- **By operator type:** Parse calculator annotations for division, decimals (.), percentages (%)
- **By keyword/entity:** Regex filters for money ($), rates/speed (mph, hours), fractions (half, third)
- **By LLM tagging:** Pass questions through LLM to auto-tag with math categories
  - Rates & Speed
  - Ratios
  - Basic Multi-step Arithmetic
  - Proportions

**Variants Available**
- **GSM8K Socratic:** Official OpenAI split with explicit sub-question breakdowns
- **GSM-Hard:** Replaces small integers with larger numbers (true reasoning test, not memorization)
- **GSM8K-Zero / Extraction Subsets:** Problems answerable without multi-step arithmetic (tests over-thinking)

**Recommendation for This Project**
- Use full ~8.5k for Phase 1 telemetry collection
- Optional: narrow to arithmetic-only for stronger domain separation signal vs. MBPP
- Domain characteristics: problems about buying items, calculating distances, multi-step reasoning

---

### MBPP (Mostly Basic Programming Problems)

**Overview**
- 974 total crowdsourced Python coding problems
  - 500 training problems
  - 427 test problems (standard for leaderboards)
  - Validation split
- **Standard reference:** MBPP-Sanitized (427 hand-verified, clarified prompts, valid test cases)
- Entry-level: focuses on function-level code synthesis, avoids multi-file projects or complex OOP

**Domain Composition**
- ~58% math/arithmetic
- ~43% list processing  
- ~19% string manipulation

**Problem Structure**
```json
{
  "task_id": 1,
  "text": "Write a function to find the minimum cost path...",
  "code": "def solution(...): ...",
  "test_list": [
    "assert solution(...) == expected",
    "assert solution(...) == expected",
    "assert solution(...) == expected"
  ]
}
```

**Modern Variant: MBPP+**
- EvalPlus framework generates ~35x more unit tests per problem
- Catches edge-case bugs; frontier models typically drop 10–15 percentage points vs. standard MBPP
- Use standard MBPP for comparison, but MBPP+ for rigorous validation

**Recommendation for This Project**
- Use MBPP-Sanitized 427 as primary benchmark (standard for leaderboards)
- Provides clear domain separation from GSM8K arithmetic
- Programming domain: list operations, string manipulation, basic algorithms

---

### Benchmark Sourcing Options

**Standard Datasets** (recommended for comparison)
- GSM8K: Available on Hugging Face, Kaggle, academic repositories
- MBPP: Housed in EvalPlus repo, Hugging Face datasets
- Both have well-established evaluation harnesses

**Synthetic Prompts** (optional expansion)
- Could generate additional prompts for specific domains
- Useful for testing hypothesis about expert specialization
- Example: synthetic "pure arithmetic" vs. "pure string manipulation" prompts

**Wide vs. Narrow Domains** (strategy decision)
- **Narrow domain approach** (current): Limit to arithmetic + Python
  - Pros: Clean domain separation, easier validation
  - Cons: Less generalization testing
- **Wide domain approach** (future): MMLU, conversational datasets
  - Start with narrow proof-of-concept, expand later

---

## Token Budget & Cardinality

### Target: ~10,000 Tokens

**Calculation**
- GSM8K: ~8,500 examples
- MBPP: ~974 examples  
- **Total examples:** ~9,500
- Average tokens per example varies by prompt/problem length

**Key Insight:** Token count will vary naturally by prompt length.

**Strategy:** Average probabilities per expert rather than collecting raw logits to handle variable-length sequences cleanly.

### Per-Token Breakdown

For each token in a sequence:
- Capture expert probabilities at each of 48 layers (Qwen) or 24 (Mixtral)
- Store running sum + count (for averaging repeated tokens later)
- Finalize: running_sum / count = average probability per expert

---

## Validation Strategy

### Randomized Runs (5x)

**Approach:**
1. Run data collection 5 times with randomized prompt orderings
2. Compute Jensen-Shannon divergence across runs
3. Target: <0.05 JSD (indicates determinism is holding)

**Why 5 runs?**
- Enough to catch systematic issues
- Validates determinism without excessive compute
- Confirms model settings are frozen

### Consistency Checks

**Same-prompt validation:**
- Same prompt subset across runs should produce bitwise-identical expert routing (in eval mode)
- Subset size: 200 random prompts (10% of data)

**Cross-domain consistency:**
- Shared tokens between GSM8K and MBPP should show routing correlation
- Example: punctuation, common words, operators (+, -, etc.)
- Validates that domain-specific signal is genuine, not noise

---

## Open Questions & Next Steps

### Dataset Decisions

1. **GSM8K Filtering:** Should we narrow to arithmetic-only (easier domain split)? Or keep full set for diversity?
2. **MBPP Coverage:** Use sanitized 427 (standard) or all 974 (broader)?
3. **Token Budget Confirmation:** Is ~10k tokens sufficient, or scale to 20k+?

### Model Decisions

1. **Starting Model:** Confirm Mixtral 8x7B as MVP target?
2. **Secondary Target:** DeepSeek-V2-Lite for post-MVP scaling?
3. **Multimodal Experiment:** Later (Qwen3-VL) or out of scope?

### Implementation

1. **Test Coverage:** Which Mixtral → DeepSeek paths need adjustment?
2. **Parquet Schema:** Finalize column definitions with expert count flexibility?
3. **Validation Baseline:** What's the minimum JSD threshold before proceeding to Phase 2?

---

## References

- **MBPP Paper:** Austin et al., "Program Synthesis with Large Language Models" (Google Research, 2021)
- **GSM8K Paper:** Cobbe et al. (OpenAI, 2021)
- **MBPP+ (EvalPlus):** https://github.com/evalplus/evalplus
- **DeepSeek Models:** https://github.com/deepseek-ai
- **Mixtral Paper:** Jiang et al., "Mixtral of Experts" (2024)
- **Snowflake Arctic:** https://www.snowflake.com/en/research/arctic/

---

**Status:** Decisions documented, framework for model selection established  
**Last Updated:** 2026-09-11  
**Notes:** This conversation establishes the foundation for Phase 1 implementation decisions. Key takeaway: start with Mixtral 8x7B, port seamlessly to DeepSeek-V2-Lite post-MVP.
