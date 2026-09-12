# Phase 1 Review: Telemetry Hooks & Deterministic Data Collection

**Date:** 2026-09-12  
**Status:** Design Complete → Implementation Ready  
**Objective:** Establish baseline by capturing expert routing data across GSM8K and MBPP datasets.

---

## Executive Summary

Phase 1 is well-specified in SYSTEM_DESIGN.md with clear objectives, data schemas, and validation steps. The infrastructure is designed but **not yet implemented**. All necessary decisions have been made; development can begin immediately.

---

## ✅ What's in Place (Design & Context)

### Documentation
- **SYSTEM_DESIGN.md** (320 lines): Complete technical specification covering all 5 phases with mathematical formulations, dataset details, storage schema, and implementation notes.
- **CONTRIBUTORS.md**: Team identified (4 members from UCSD/San Diego schools).
- **PRIOR_WORK_McNAIR.md**: Contextual background on concurrent research.
- **Conversation Logs** (`docs/conversations/`):
  - `slides_2026-09-10_moe_prefetching_overview.md` – High-level overview
  - `2026-09-10_gemini_dataset_and_models.md` – Dataset/model research decisions
  - `2026-09-10_dvora_moe_alternatives.md` – Alternative architectures explored

### Key Design Decisions (Already Made)
1. **Model:** Mixtral 8x7B (primary) with optional DeepSeek-Coder-V2-Lite (code-optimized)
2. **Datasets:** GSM8K (8.5k math problems) + MBPP (974 code problems)
3. **Storage:** Parquet format (columnar, compressed)
4. **Data Collection:** PyTorch forward hooks on router layers
5. **Focus:** Prefill phase only (deterministic routing, no generation noise)
6. **Validation:** 5 randomized runs for consistency verification

---

## ❌ What's Missing (Implementation Artifacts)

### Directory Structure
```
moe/
├── phase_1/
│   ├── scripts/                # Phase 1 implementation code
│   │   ├── data_collection.py  [MISSING]
│   │   ├── validation.py       [MISSING]
│   │   └── dataset_loading.py  [MISSING]
│   └── docs/                   # This folder
│
├── data/                       # Outputs (TBD)
│   ├── gsm8k_routing_traces/   [MISSING]
│   ├── mbpp_routing_traces/    [MISSING]
│   └── validation_reports/     [MISSING]
│
└── requirements.txt            [MISSING]
```

### Phase 1 Implementation Components (Priority Order)

#### 1. **Environment Setup** (CRITICAL)
- [ ] `requirements.txt` – Python dependencies:
  - `torch>=2.0`
  - `transformers>=4.30`
  - `datasets>=2.10`
  - `pyarrow>=12.0`
  - `tqdm`
  - (optional) `duckdb`, `polars` for downstream analysis
- [ ] GPU environment detection (A100/H100)
- [ ] Model downloading & caching

#### 2. **Dataset Loading** (CRITICAL)
- [ ] GSM8K loader (Hugging Face datasets API)
- [ ] MBPP loader (Hugging Face or EvalPlus)
- [ ] Token counting utility (to estimate memory footprint)
- [ ] Prompt assembly (text input for prefill, no generation)

#### 3. **Model & Hook Infrastructure** (CRITICAL)
- [ ] Model loading (Mixtral 8x7B, model.eval() mode)
- [ ] Tokenizer setup (padding, max_length handling)
- [ ] Forward hook registration on `block_sparse_moe.gate` layers
- [ ] Routing data extraction from hooks:
  - Pre-softmax logits
  - Post-softmax probabilities
  - Top-k indices

#### 4. **Data Collection Pipeline** (CRITICAL)
- [ ] Main collection loop:
  ```python
  for run in range(5):  # 5 randomized runs
      for dataset in ['gsm8k', 'mbpp']:
          for prompt in shuffled_dataset[dataset]:
              # Tokenize
              # Forward pass (capture routing)
              # Store running averages
  ```
- [ ] Running sum/count tracking (averaging repeated tokens)
- [ ] Memory-efficient buffering (avoid OOM with 9.5k+ examples)
- [ ] Progress tracking (tqdm)

#### 5. **Storage & Serialization** (CRITICAL)
- [ ] Parquet schema implementation using PyArrow:
  - `sequence_id` (string)
  - `token_id` (int32)
  - `token_str` (string)
  - `token_position` (int32)
  - `layer_index` (int32)
  - `expert_probabilities` (float32 array, size E)
  - `selected_experts` (int16 array, size k)
  - `selected_probs` (float32 array, size k)
  - `dataset_source` (string)
- [ ] Incremental batch writing (avoid memory spike)
- [ ] Compression (snappy or zstd)

#### 6. **Validation Testing** (IMPORTANT)
- [ ] **Determinism Check:** Run same 5 prompts 5×; verify expert probabilities are bitwise identical
- [ ] **Randomization Consistency:** Randomize prompt order; rerun; verify same routing traces
- [ ] **Baseline Comparison:** Dense model comparison to ensure meaningful sparse routing divergence
- [ ] **Data Integrity:** Parquet schema validation, no NaN/inf values
- [ ] **Sample Statistics:** Histograms of expert activation frequencies per layer

#### 7. **Logging & Telemetry** (IMPORTANT)
- [ ] Run metadata (model name, dataset, run #, timestamp, git commit)
- [ ] Memory usage tracking (peak RAM, GPU memory)
- [ ] Execution time per dataset/run
- [ ] Log file output (structured JSON or plain text)

#### 8. **Documentation** (IMPORTANT)
- [ ] `phase_1/README.md` – How to run Phase 1
- [ ] Inline code comments (especially hook logic)
- [ ] Data schema documentation (columnar descriptions)
- [ ] Known issues & limitations

---

## Implementation Roadmap

### Week 1: Foundation
1. Set up Python environment & requirements.txt
2. Implement dataset loaders (GSM8K, MBPP)
3. Model loading + tokenizer setup
4. Hook registration on router layers

### Week 2: Data Collection
5. Main collection loop with running average tracking
6. Parquet serialization (batch writing)
7. Memory efficiency optimization
8. First test run on small subset (100 prompts)

### Week 3: Validation & Refinement
9. Determinism checks (same prompt, 5 runs)
10. Randomization consistency (different orderings)
11. Baseline comparison (dense model)
12. Full data collection (GSM8K + MBPP, all 5 runs)

### Week 4: Analysis & Wrap-up
13. Aggregate statistics & reporting
14. Data quality assurance
15. Documentation cleanup
16. Commit Phase 1 artifacts

---

## Key Metrics to Capture

### Per Dataset/Run
- Total prompts processed
- Total tokens processed
- Execution time (elapsed wall clock)
- Peak GPU memory, RAM usage
- Expert activation frequencies per layer

### Data Quality
- % of unique tokens vs. total tokens (repetition rate)
- Expert probability ranges (min/max across experts per token)
- Top-k coverage (% of top-2 experts that match across runs)

### Validation Results
- Determinism check: % prompts with bitwise-identical probabilities across 5 runs (target: 100%)
- Randomization consistency: % tokens with identical top-k experts across orderings (target: 100%)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| OOM with 9.5k examples × 128 experts | HIGH | Batch processing, incremental Parquet writes |
| Stochasticity in model forward (unexpected) | MEDIUM | Enforce `model.eval()`, verify determinism early |
| Hook registration fails on model architecture | MEDIUM | Test hook on toy model first; layer path may vary |
| GPU instance timeout mid-run | MEDIUM | Checkpoint after each dataset; resume capability |
| Parquet schema mismatch / corrupted files | MEDIUM | Validate schema on first batch before full run |

---

## Deliverables (End of Phase 1)

1. **Routing Traces (Parquet):**
   - `data/gsm8k_routing_traces.parquet` (5 runs, all 8.5k examples)
   - `data/mbpp_routing_traces.parquet` (5 runs, all 974 examples)

2. **Validation Report:** `data/validation_reports/phase_1_validation.md`
   - Determinism verification
   - Randomization consistency results
   - Data quality statistics
   - Any anomalies or deviations

3. **Implementation Code:** `phase_1/scripts/data_collection.py`
   - Fully commented, reproducible
   - Optional: `phase_1/scripts/validation.py` for standalone checks

4. **Metadata:** `data/phase_1_metadata.json`
   - Model name, tokenizer, layer paths
   - Dataset versions (URLs, commit hashes)
   - Run timestamps, execution times, resource usage
   - Git commit hash for reproducibility

---

## Next Steps

1. **Immediate:** Create `requirements.txt` and test environment setup
2. **This week:** Implement dataset loaders and model setup
3. **By end of week:** First test run on 50-100 prompts
4. **Target completion:** 3 weeks from start

---

## Questions to Resolve

1. **Model Precision:** Use fp16 or fp32 for routing probabilities? (Recommendation: fp32 for determinism verification)
2. **Batch Size:** What batch size fits in A100/H100 HBM for forward passes? (Depends on seq_len; start conservative: 4)
3. **Prompt Lengths:** What's the max sequence length for GSM8K/MBPP? (Affects tokenization, memory)
4. **Hook Placement:** Exact layer path for Mixtral? (Verify: `model.layers[i].block_sparse_moe.gate`)
5. **Randomization Strategy:** Shuffle dataset or shuffle prompt ordering within fixed dataset? (Recommendation: shuffle order, keep seed for reproducibility)

---

## Success Criteria

- [ ] Phase 1 code runs without errors on full datasets
- [ ] All 9.5k examples tokenized and forwarded through model
- [ ] Expert routing data stored in Parquet with no corruption
- [ ] Determinism check: 100% of prompts show identical routing across 5 runs
- [ ] Validation report confirms data quality and readiness for Phase 2
- [ ] Code committed with clear README for reproduction
