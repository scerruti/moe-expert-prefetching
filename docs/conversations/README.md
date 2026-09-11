# Conversations & Meeting Notes

This folder documents the project's intellectual history: conversations with AI systems, design discussions, and meeting notes. Organized chronologically and by topic for easy reference and traceability.

## Conversations

### Dataset & Model Selection

- [**2026-09-10: Gemini – Dataset and Models**](2026-09-10_gemini_dataset_and_models.md)
  - Discussion of GSM8K vs. MBPP datasets
  - MoE model options (Mixtral, DeepSeek, Qwen variants)
  - Token budget and randomization strategy
  - Model migration path (Mixtral → DeepSeek with code portability)
  - **Key decision:** Mixtral 8x7B primary target, DeepSeek-V2-Lite for post-MVP scaling

- [**2026-09-10: Dvora – MoE Alternatives for MBPP**](2026-09-10_dvora_moe_alternatives.md)
  - MBPP-specific model analysis
  - Code-optimized MoE vs. dense alternatives
  - GSM8K filtering strategies (step count, operators, keywords)
  - MBPP-Sanitized (427) vs. full MBPP (974) comparison
  - **Key decision:** MBPP-Sanitized is standard for leaderboards

## Slide Decks

- [**2026-09-10: Speculative Expert Prefetching Overview**](slides_2026-09-10_moe_prefetching_overview.md)
  - High-level project summary (Marp format)
  - Five-phase roadmap visualization
  - Success metrics and bottleneck analysis

## Meeting Notes

*To be added as team syncs occur.*

## References

Supporting materials and background research:

- [McNair Spatial Reasoning Paper](../references/mcnair_spatial_reasoning.pdf) – Prior work on Qwen3-VL expert routing analysis
- [Gemini MoE Background](../references/gemini_moe_background.pdf) – LLM conversation notes on MoE fundamentals

---

## How to Add New Conversations

1. Create a new markdown file: `YYYY-MM-DD_topic_brief.md`
2. Include header with date, participants, and topic
3. Document key decisions, options, and rationale
4. Link to any supporting references
5. Update this README with a one-line entry

---

**Last Updated:** 2026-09-11
