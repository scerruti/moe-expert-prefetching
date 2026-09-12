# Phase 1 Implementation Checklist

**Status:** Ready to begin  
**Target Duration:** 3-4 weeks  
**Priority:** CRITICAL (foundation for all downstream phases)

---

## Setup & Environment

- [ ] Create `requirements.txt` with all dependencies
- [ ] Test Python environment (3.10+ required)
- [ ] Verify GPU access (CUDA, torch install)
- [ ] Create data directory structure (`data/gsm8k/`, `data/mbpp/`, etc.)
- [ ] Create phase_1/scripts directory structure
- [ ] Git branch for Phase 1 implementation

---

## Dataset Loading

- [ ] Implement GSM8K loader
  - [ ] Load from Hugging Face datasets
  - [ ] Extract problem text + solution
  - [ ] Tokenize and prepare for model input
  - [ ] Test on small sample (10 examples)
  
- [ ] Implement MBPP loader
  - [ ] Load from Hugging Face datasets (or EvalPlus variant)
  - [ ] Extract problem text
  - [ ] Tokenize and prepare
  - [ ] Test on small sample (10 examples)

- [ ] Implement dataset utilities
  - [ ] Prompt assembly (clean text, no extra tokens)
  - [ ] Token counting (estimate memory footprint)
  - [ ] Shuffling with seed for reproducibility
  - [ ] Batch iterator

---

## Model & Tokenizer Setup

- [ ] Load Mixtral 8x7B model
  - [ ] Download model (HF transformers)
  - [ ] Verify architecture (8 experts, top-2 routing)
  - [ ] Check model device placement

- [ ] Set up tokenizer
  - [ ] Load Mixtral tokenizer
  - [ ] Test encoding/decoding
  - [ ] Verify padding, max_length handling

- [ ] Prepare model for data collection
  - [ ] Set `model.eval()` (no dropout)
  - [ ] Disable generation randomness (if applicable)
  - [ ] Verify no gradient computation needed

---

## Forward Hook Implementation

- [ ] Research Mixtral layer structure
  - [ ] Find exact module path for router gates
  - [ ] Identify pre-softmax and post-softmax layers
  - [ ] Document layer indices and names

- [ ] Implement hook function
  - [ ] Extract pre-softmax logits from input/output
  - [ ] Compute softmax probabilities
  - [ ] Extract top-k indices
  - [ ] Handle batched inputs (multiple sequences)

- [ ] Implement hook registration
  - [ ] Register on all MoE router layers
  - [ ] Store hooks in list for later cleanup
  - [ ] Test hook fires correctly on forward pass

- [ ] Implement hook data extraction
  - [ ] Store outputs in thread-safe buffer
  - [ ] No blocking operations during forward
  - [ ] Prepare data format for downstream storage

---

## Data Collection Pipeline

- [ ] Implement main collection loop
  - [ ] Outer loop: 5 randomized runs
  - [ ] Inner loop: iterate over datasets (GSM8K, MBPP)
  - [ ] Innermost loop: iterate over prompts

- [ ] Implement token processing
  - [ ] Tokenize prompt
  - [ ] Forward through model (capture hooks)
  - [ ] Extract token IDs, positions, token strings

- [ ] Implement running average tracking
  - [ ] For each unique token: maintain sum of probabilities
  - [ ] Track count of occurrences
  - [ ] Finalize: divide sum by count after all runs

- [ ] Implement progress tracking
  - [ ] tqdm progress bar for outer loops
  - [ ] Estimated time remaining
  - [ ] Checkpoint every N prompts

- [ ] Implement error handling
  - [ ] Skip malformed prompts (log warning)
  - [ ] Handle OOM gracefully (flush buffer, resume)
  - [ ] Catch CUDA errors

---

## Parquet Storage & Serialization

- [ ] Define PyArrow schema
  - [ ] `sequence_id` (string)
  - [ ] `token_id` (int32)
  - [ ] `token_str` (string)
  - [ ] `token_position` (int32)
  - [ ] `layer_index` (int32)
  - [ ] `expert_probabilities` (float32 array)
  - [ ] `selected_experts` (int16 array)
  - [ ] `selected_probs` (float32 array)
  - [ ] `dataset_source` (string)

- [ ] Implement batch writer
  - [ ] Convert in-memory data to PyArrow Table
  - [ ] Handle variable-length arrays (experts)
  - [ ] Test schema on small batch

- [ ] Implement file I/O
  - [ ] Incremental batch writing (avoid memory spike)
  - [ ] Parquet compression (snappy or zstd)
  - [ ] File naming convention (dataset, run #)
  - [ ] Error recovery (verify file integrity)

- [ ] Test serialization
  - [ ] Write sample batch to Parquet
  - [ ] Read back and verify schema
  - [ ] Check data types and values

---

## Validation Testing

- [ ] Implement determinism check
  - [ ] Select 5 sample prompts
  - [ ] Run each 5 times (bitwise identical test)
  - [ ] Report % success (target: 100%)

- [ ] Implement randomization consistency check
  - [ ] Run dataset collection 2x with different random orderings
  - [ ] Compare routing for same tokens
  - [ ] Report % consistency (target: 100%)

- [ ] Implement baseline comparison (optional for Phase 1)
  - [ ] Load dense model variant
  - [ ] Compare expert probabilities (should show divergence)
  - [ ] Document findings

- [ ] Implement data quality checks
  - [ ] No NaN or inf values
  - [ ] Expert probabilities sum to ~1.0
  - [ ] Top-k indices are valid (0 to E-1)
  - [ ] Token IDs match tokenizer vocab size

- [ ] Validation report generation
  - [ ] Automated report with pass/fail per check
  - [ ] Include statistics (activation frequencies, etc.)
  - [ ] Highlight any anomalies

---

## Logging & Telemetry

- [ ] Implement structured logging
  - [ ] Run start/end timestamps
  - [ ] Model name, tokenizer version
  - [ ] Dataset versions (URLs/commits)
  - [ ] Layer paths used

- [ ] Track resource usage
  - [ ] Peak GPU memory
  - [ ] Peak CPU memory
  - [ ] Execution time per run/dataset
  - [ ] Throughput (tokens/sec)

- [ ] Create metadata JSON
  - [ ] Model config (vocab size, num_experts, k)
  - [ ] Dataset config (sizes, sources)
  - [ ] Run info (times, random seeds)
  - [ ] Git commit hash

---

## Documentation

- [ ] Create `phase_1/README.md`
  - [ ] How to install dependencies
  - [ ] How to run Phase 1 collection
  - [ ] Expected outputs
  - [ ] Known limitations

- [ ] Inline code documentation
  - [ ] Docstrings for main functions
  - [ ] Comments on non-obvious logic (hooks, storage)
  - [ ] Example usage

- [ ] Data schema documentation
  - [ ] Detailed description of Parquet columns
  - [ ] Data type justification
  - [ ] Query examples

- [ ] Troubleshooting guide
  - [ ] Common errors and fixes
  - [ ] Memory optimization tips
  - [ ] GPU compatibility notes

---

## Testing & Validation

- [ ] Unit tests (if applicable)
  - [ ] Test dataset loaders on small samples
  - [ ] Test hook registration and firing
  - [ ] Test Parquet schema and I/O

- [ ] Integration test
  - [ ] Run full pipeline on 10 GSM8K examples
  - [ ] Verify all outputs generated
  - [ ] Check data format and completeness

- [ ] Small-scale test
  - [ ] Run on 100 GSM8K + 100 MBPP examples
  - [ ] Verify determinism on subset
  - [ ] Estimate memory usage for full run

- [ ] Full-scale validation
  - [ ] Run complete Phase 1 on all datasets
  - [ ] All 5 runs completed successfully
  - [ ] All validation checks pass

---

## Code Quality

- [ ] Code review
  - [ ] Style: PEP 8 compliance
  - [ ] Clarity: Well-named variables, functions
  - [ ] Efficiency: No unnecessary copies or loops

- [ ] Performance
  - [ ] Profile memory usage
  - [ ] Optimize hot paths
  - [ ] Document any trade-offs

- [ ] Reproducibility
  - [ ] All random seeds set and logged
  - [ ] Environment requirements documented
  - [ ] Git commit hash recorded with results

---

## Final Deliverables

- [ ] Parquet files
  - [ ] `data/gsm8k_routing_traces.parquet`
  - [ ] `data/mbpp_routing_traces.parquet`
  - [ ] File sizes reasonable (check compression)

- [ ] Validation report
  - [ ] `data/validation_reports/phase_1_validation.md`
  - [ ] All checks documented with results
  - [ ] Recommendations for Phase 2

- [ ] Implementation code
  - [ ] `phase_1/scripts/data_collection.py` (main)
  - [ ] `phase_1/scripts/validation.py` (validation)
  - [ ] `phase_1/scripts/dataset_loading.py` (utilities)
  - [ ] All code commented and documented

- [ ] Metadata
  - [ ] `data/phase_1_metadata.json`
  - [ ] Execution logs (optional)
  - [ ] Resource usage summary

- [ ] Version control
  - [ ] All code committed with clear messages
  - [ ] Branch merged to main
  - [ ] Tag release (e.g., `phase-1-v1.0`)

---

## Sign-off

- [ ] All checklist items completed
- [ ] Code reviewed and tested
- [ ] Documentation complete
- [ ] Phase 1 complete and ready for Phase 2 handoff
- [ ] Meeting with team to review findings

**Completion Target:** 2026-10-03 (3 weeks from 2026-09-12)
