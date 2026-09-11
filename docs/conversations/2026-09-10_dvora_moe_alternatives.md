# Conversation: MoE Alternatives for MBPP

**Date:** September 10, 2026  
**Participants:** Dvora, Gemini (Claude)  
**Topic:** Exploring alternative MoE and dense models for code-focused benchmarking (MBPP)  
**Source:** Gemini conversation

---

## Context

This conversation explores alternatives to Qwen3 MoE specifically for evaluating code generation and mathematical reasoning performance on MBPP (Mostly Basic Python Problems), without requiring multimodal (vision) capabilities.

---

## MoE Alternatives for MBPP

### Top Sparse MoE Choices

**DeepSeek-V3 / DeepSeek-Coder-V2-Lite**
- Explicitly optimized for code synthesis and mathematical reasoning
- Micro-expert routing architecture
- Elite pass@1 scores on MBPP and HumanEval
- Minimal active parameters (efficient routing)

**Mixtral 8x22B Instruct**
- Strong general-purpose sparse MoE (larger variant of 8x7B)
- Solid reasoning capabilities for algorithmic Python problems
- Better suited to larger workloads than base Mixtral 8x7B

**DBRX Instruct (Databricks)**
- 132B total parameters / 36B active per token
- Trained with heavy emphasis on programming and data tasks
- Balanced general and specialized performance

---

### Dense & Code-Specialized Alternatives

**Qwen2.5-Coder-32B / Qwen3-Coder**
- Stay in Qwen ecosystem without MoE routing complexity
- Dedicated Coder variants often outperform general MoE on pure Python generation
- Lightweight deployment profile
- Specialized pre-training for programming tasks

**DeepSeek-R1 / DeepSeek-R1-Distill-Llama-70B**
- Reasoning trace generation ("thinking mode")
- Strong multi-step logical chain generation before output
- Ideal for validating intermediate reasoning quality, not just final answers

**CodeLlama-70B / Llama 3.3 70B Instruct**
- Strong non-MoE options
- Good baseline for standardized Python benchmark comparison
- Established performance baselines across open-weight models

---

### Model Comparison Table

| Model | Architecture | Primary Strength on MBPP | Active Params | Notes |
|-------|-------------|--------------------------|---------------|-------|
| DeepSeek-Coder-V2 | Sparse MoE | High pass@1 rate, Python syntax | Efficient | Code-optimized |
| Qwen3-Coder-32B | Dense | Lightweight, specialized | 32B | No MoE routing |
| DeepSeek-R1 Distill | Dense (Reasoning) | Multi-step chain gen | Variable | Thinking mode |
| Mixtral 8x22B | Sparse MoE | Balanced general + structured | 47B (8x22B variant) | Larger than base |
| DBRX | Sparse MoE | Programming emphasis | 36B active (132B total) | Data tasks tuned |

---

## Detailed Dataset Analysis

### GSM8K Filtering & Variants

**Raw GSM8K Structure**
- No pre-assigned difficulty tags, math topics, or complexity metadata
- Contains only: question + answer fields
- Requires programmatic engineering for meaningful subsets

**Slicing Strategies (DIY)**

1. **By Step Count (Complexity)**
   - Count `\n` or `<<...>>` blocks in answer field
   - Simple: 2–3 steps
   - Hard: 6–8 steps
   - Directly measures reasoning depth

2. **By Operator Type**
   - Parse calculator annotations: `<<10+5=15>>`
   - Filter for: division, decimals (.), percentages (%)
   - Isolates specific computational domains

3. **By Keyword/Entity Matching**
   - Regex filters: money ($), rates/speed (mph, hours), fractions (half, third)
   - Captures problem archetypes
   - Useful for targeted domain isolation

4. **LLM/Classifier Tagging**
   - Pass questions through LLM for auto-tagging
   - Math categories: Rates & Speed, Ratios, Basic Multi-step Arithmetic, Proportions
   - Export sub-datasets by category

**Pre-Made Variants (No Custom Parsing)**

- **GSM8K Socratic (OpenAI Official):** Intermediate steps broken down with explicit sub-questions
- **GSM-Hard:** Replaces small integers with larger numbers; tests true reasoning vs. memorization
- **GSM8K-Zero / Extraction Subsets:** Problems answerable without multi-step arithmetic; tests over-thinking

---

### MBPP Dataset Deep Dive

**Overview**
- 974 total crowdsourced Python coding problems (Google Research, 2021)
- Measures ability to convert natural language descriptions → correct, executable Python
- Entry-level: solvable by introductory CS students

**Domain Composition**
- ~58% math/arithmetic
- ~43% list processing
- ~19% string manipulation
- Avoids multi-file projects and complex OOP patterns

**Problem Structure Example**
```json
{
  "task_id": 1,
  "text": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n).",
  "code": "def min_cost(cost, m, n): ...",
  "test_list": [
    "assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8",
    "assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12",
    "assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16"
  ]
}
```

**Dataset Splits**
- **Full MBPP (974):** 500 train, 427 test, validation
- **MBPP-Sanitized (427):** Hand-verified, clarified prompts, corrected test cases
  - Standard for modern LLM evaluation leaderboards
  - Preferred over full 974 (original had ambiguities and flaws)

**MBPP vs. HumanEval (OpenAI)**

| Feature | MBPP | HumanEval |
|---------|------|-----------|
| Creator | Google Research (2021) | OpenAI (2021) |
| Size | 974 total / 427 sanitized | 164 problems |
| Prompt Style | Short natural language | Function docstrings + I/O examples |
| Setup | Raw task instructions | Explicit function header & signatures |
| Focus | Control-flow & stdlib usage | Algorithmic logic & puzzle solving |

**MBPP+ (EvalPlus)**
- Extends MBPP-Sanitized with ~35x more unit tests per problem
- Automated test generation + fuzzing to catch edge cases and "hallucinated correctness"
- Frontier models see 10–15 percentage point performance drop on MBPP+ vs. standard MBPP
- Good final validation benchmark after POC

---

## Dataset Strategy for This Project

### Recommended Approach

**GSM8K**
- Full 8.5k examples for Phase 1 telemetry
- Optional: narrow to arithmetic-only via step-count filtering for stronger domain separation
- Characteristics: multi-step deduction, numeric constants, natural language reasoning

**MBPP**
- Use MBPP-Sanitized (427) as primary benchmark
- Standard for leaderboards, cleaner than full 974
- Clear domain separation from math: syntax, structure, list/string operations
- MBPP+ for rigorous final validation only

### Why This Combination

- **Arithmetic (GSM8K)** and **Code Syntax (MBPP)** are sufficiently distinct domains
- Both at "entry-level" difficulty → model behavior comparable
- No multimodal complexity (pure text)
- Well-established evaluation baselines
- Can start narrow, expand to other domains (general knowledge, conversational) post-MVP

---

## Decision Summary

### For MBPP Specifically

| Question | Decision |
|----------|----------|
| Use Qwen3 MoE? | Not required for pure text; alternatives equally valid |
| MoE or Dense? | MoE (DeepSeek-Coder-V2-Lite) OR Dense (Qwen3-Coder) depending on scale focus |
| Dataset variant? | MBPP-Sanitized (427) as standard; MBPP+ for final rigor |
| GSM8K subset? | Full 8.5k for breadth; optional arithmetic-only for domain purity |

### Key Takeaways

1. **Code-optimized MoE models (DeepSeek variants)** outperform general-purpose MoE on MBPP
2. **Dense Coder variants** are lightweight alternative if routing complexity unwanted
3. **MBPP-Sanitized is the standard**—use it for consistency with published leaderboards
4. **GSM8K filtering strategies** are straightforward (step count, operators, keywords)
5. **Combination provides clean domain separation** for prefetching research

---

## References

- **MBPP Paper:** Austin et al., "Program Synthesis with Large Language Models" (Google Research, 2021)
- **GSM8K Paper:** Cobbe et al. (OpenAI, 2021)
- **MBPP+ Framework:** https://github.com/evalplus/evalplus
- **DeepSeek Models:** https://github.com/deepseek-ai
- **Qwen Models:** https://github.com/QwenLM

---

**Status:** Alternatives documented, dataset strategies clarified  
**Last Updated:** 2026-09-11  
**Note:** Complements the earlier conversation about model selection with Dvora's focus on MBPP-specific optimizations
