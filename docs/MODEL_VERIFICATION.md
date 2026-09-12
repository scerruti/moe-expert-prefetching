# Model Architecture Verification

## Qwen3.8-27B – NOT MoE ❌

**Status:** Confirmed NOT Mixture-of-Experts

**Evidence:**
- **Architecture type:** "Qwen3_5ForConditionalGeneration" (dense model)
- **Model type:** Causal Language Model with Vision Encoder
- **Hidden layout:** 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
- **Parameters:** 27B (all active)
- **MoE mention:** NONE in architecture documentation

**Source:** https://huggingface.co/Qwen/Qwen3.8-27B (model card, config.json)

**Implications:**
- ✅ Good for general performance (strong coding/reasoning scores)
- ✅ Vision-language capable
- ❌ NOT suitable for this project (requires MoE routing analysis)
- ❌ Defeats the purpose of expert prefetching research

---

## Confirmed MoE Models – Available Options

### Option 1: Mixtral 8x7B ✅ (Verified MoE)
- **Status:** Dense alternative already selected for MVP
- **Expert count:** 8 experts, top-2 routing
- **Use case:** Software validation/proof-of-concept
- **Confirmed MoE:** Yes

### Option 2: DeepSeek-Coder-V2-Lite ✅ (Candidate)
- **Status:** Code-optimized MoE
- **Need:** Verify actual architecture is MoE before committing

### Option 3: Qwen3 MoE (30B or 235B) ✅ (Candidate)
- **Status:** Confirmed MoE in conversation notes
- **Expert count:** 128 experts with top-2 activation
- **Need:** Verify this is truly MoE (not confused with dense Qwen3.8)

---

## Recommendation

**Stick with current plan:**
1. **MVP Validation:** Mixtral 8x7B on Colab (confirmed MoE, well-documented)
2. **Research Model:** Choose between:
   - DeepSeek-Coder-V2-Lite (verify MoE status)
   - Qwen3 MoE 30B (verify is MoE, not dense variant)
   - Qwen3 MoE 235B (larger scale)

**Do NOT use Qwen3.8-27B for this project** – it's dense, not sparse MoE.

---

**Last Updated:** 2026-09-12
