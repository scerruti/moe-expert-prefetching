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

**Still to confirm:** a full forward pass at batch size 8 fits in the remaining memory.

## See Also

- [CONTRIBUTING.md](CONTRIBUTING.md) — Acceptance criteria must be proven
- [requirements.txt](requirements.txt) — Qwen1.5-MoE-A2.7B with transformers 4.45.0
