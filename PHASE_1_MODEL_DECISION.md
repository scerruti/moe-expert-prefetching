# Phase 1 Model Selection Decision

**Date:** October 30, 2026  
**Decision:** YAGNI - Pragmatic single-model approach

## Context

**Model Selection History:**
- Sept 11, 2026: Started with Mixtral 8x7B for initial validation
- Sept 22, 2026: Waleed identified Mixtral OOMs on Colab 40GB
- Sept 27, 2026: Confirmed Mixtral + 4-bit fails due to memory fragmentation during loading
- Oct 1, 2026: Decided on **Qwen1.5-MoE-A2.7B** — successfully loads and validates on Colab A100
- Nov 1, 2026: Switched from 4-bit to **fp16** (see Amendment below)

## Decision

**Phase 1 is locked to: Qwen1.5-MoE-A2.7B in fp16 on Colab A100** (4-bit until the Nov 1 amendment)

### What This Means

✅ **Locked for Phase 1:**
- Model: Qwen1.5-MoE-A2.7B
- GPU: Colab A100 40GB
- Precision: fp16, unquantized (was 4-bit; see Amendment)
- Router gate path: `layer.mlp.gate`

❌ **Remove/Don't Support:**
- Multi-model abstraction (YAGNI)
- Mixtral support in hooks (abandoned at Oct 1)
- Vision models (Qwen3-VL out of scope)

### Rationale

YAGNI principle: We have one model that works. Don't abstract for hypothetical future models.
- Phase 1 scope is clear and resource-constrained
- Qwen1.5-MoE-A2.7B is the only model we've validated works on our target GPU
- Adding abstraction complexity buys us nothing today
- If Phase 2 needs a different model, we build for Phase 2 at that time

### Code Changes Required

1. **ModelLoader:** Keep as-is (already there, doesn't hurt, tests use it)
2. **pytorch_forward_hooks.py:** Hardcode for Qwen1.5's `mlp.gate` (remove abstraction)
3. **test_pytorch_forward_hooks.py:** Use Qwen toy models instead of Mixtral

### What Stays General

- Data collection (model-agnostic)
- Determinism validation (model-agnostic)
- Tokenization pipeline (any HF tokenizer)
- Parquet storage (generic)

These don't need change.

## Decision Made By

Stephen Cerruti (with team validation via Phase 1 collection success)

## Amendment: fp16 instead of 4-bit (Nov 1, 2026)

**Decision:** Phase 1 loads Qwen1.5-MoE-A2.7B in fp16, unquantized (#103).

**Why:**
- Under 4-bit loading, bitsandbytes also quantized the router gate (`mlp.gate`) and the shared-expert gate, so the routing decisions we measure came from 4-bit weights.
- fp16 fits: loading with `torch_dtype=torch.float16` uses 28.7 GB of the A100's 42.4 GB (checked on Colab, Nov 1).
- The unquantized OOM that led to 4-bit was most likely an fp32 load. On Oct 30, `67b6f1b` (08:46) removed `torch_dtype` from the loader, and without it transformers 4.45 loads in fp32 (~57 GB). 4-bit was enabled for the pipeline 2.5 hours later (`e79c64c`, 11:20). The loader passes `torch_dtype` again.

**Confirmed (Nov 1, Colab A100):** a batch of the 8 longest prompts (about 180 tokens each) peaks at 30.5 GB of 42.4 GB, so batch size 8 fits. The default is now **batch size 32** (#106): with the unused LM head skipped, the 32 longest prompts peak at 30.4 GB, and collection runs at about 31 ms per prompt.

## Amendment: Phase 1 finishes on Qwen1.5; Phase 2 makes the model swappable (Nov 21, 2026)

**What this corrects.** The October 30 decision locked Phase 1 to Qwen1.5-MoE-A2.7B and removed multi-model support as YAGNI. Choosing Qwen was right: Mixtral ran out of memory on Colab, and Qwen worked. But the decision also departed from `SYSTEM_DESIGN.md`, which planned a pipeline that moves between MoE models. That departure was never stated as a goal change, and it wasn't intended as one.

**Decision.**
- **Phase 1 finishes as it is,** locked to Qwen1.5-MoE-A2.7B, including the official collection. Nothing in Phase 1 changes.
- **Phase 2 corrects course:** the pipeline becomes able to swap in different MoE models. Qwen1.5 serves as the small-model proof of concept, and findings can then be reproduced on other MoE models, including larger ones.

**Why.** Findings that hold across models are worth more than findings on one. It also supports research on separating experts from their models and combining them in new ways, which needs a proof of concept on a small MoE that can then be reproduced on a larger one.

**What's model-specific today.** More than the router hook. Collection and validation still assume every decoder layer is an MoE layer: extraction expects a router event from every layer, and #59's row count is tokens × all decoder layers. A model with dense layers between its MoE layers fails even with the right gate hook. Waleed showed this with a tiny Qwen2MoE model with `decoder_sparse_step=2` (#127). So "What Stays General" above is wrong about data collection and determinism validation. The Phase 2 work covers:

- the router hook: the gate path (`layer.mlp.gate`), the top-k check, and the router's output format;
- which layers are routed, recorded as metadata;
- coverage and row-count checks based on the routed layers, not every decoder layer;
- each model's routing semantics, such as how it normalises the top-k probabilities;
- a memory check for each model added.

Tokenization and storage are model-agnostic. The work is planned in the Phase 2 reorganisation (#112).

**Follow-up:** update `SYSTEM_DESIGN.md` to match (#110).

## See Also

- [CONTRIBUTING.md](CONTRIBUTING.md) — Acceptance criteria must be proven
- [requirements.txt](requirements.txt) — Qwen1.5-MoE-A2.7B with transformers 4.45.0
