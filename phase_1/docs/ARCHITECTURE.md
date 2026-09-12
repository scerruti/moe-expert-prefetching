# Phase 1 Architecture: Quick Reference

This document provides a high-level architectural overview for Phase 1 implementation.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 1 Data Collection                   │
└─────────────────────────────────────────────────────────────┘

Input: GSM8K (8.5k) + MBPP (974 examples)
       ↓
   ┌───────────────────────────────────────┐
   │  Load & Prepare Datasets              │
   │  - Tokenize prompts                   │
   │  - Pad/truncate sequences             │
   │  - Shuffle for 5 randomized runs      │
   └───────────────────────────────────────┘
       ↓
   ┌───────────────────────────────────────┐
   │  Load Mixtral 8x7B Model              │
   │  - model.eval()                       │
   │  - Register forward hooks             │
   │  - Prepare for inference              │
   └───────────────────────────────────────┘
       ↓
   ┌───────────────────────────────────────┐
   │  Forward Pass with Hook Capture       │
   │  ├─ Layer 0: extract P(e|x)           │
   │  ├─ Layer 1: extract P(e|x)           │
   │  └─ Layer N: extract P(e|x)           │
   │                                       │
   │  For each token:                      │
   │  - Pre-softmax logits (E,)            │
   │  - Softmax probabilities (E,)         │
   │  - Top-2 indices (2,)                 │
   └───────────────────────────────────────┘
       ↓
   ┌───────────────────────────────────────┐
   │  Running Averages (Per Token)         │
   │  - sum_probs[token][expert] +=        │
   │  - count[token] += 1                  │
   │  (accumulated across all 5 runs)      │
   └───────────────────────────────────────┘
       ↓
   ┌───────────────────────────────────────┐
   │  Serialize to Parquet                 │
   │  - Batch write (avoid memory spike)   │
   │  - Compression: snappy/zstd           │
   │  - Schema: columnar format            │
   └───────────────────────────────────────┘
       ↓
   Output: Parquet files (GSM8K, MBPP)
           + Validation report
           + Metadata JSON
```

---

## Data Flow Diagram

### Per-Prompt Processing

```
Prompt Text
    ↓
[Tokenizer]  →  Token IDs: [101, 2054, 2003, ...]
    ↓
[Model.forward()] with Hooks Registered
    ↓
Hook Fires at Each MoE Layer:
  ├─ Layer 0 (gate)
  │   ├─ input[batch=1, seq_len, hidden_dim]
  │   ├─ output[batch=1, seq_len, num_experts=128]
  │   ├─ softmax → probabilities[seq_len, 128]
  │   └─ topk → top_indices[seq_len, 2]
  │
  ├─ Layer 1 (gate)
  │   └─ ... (repeat for all layers)
  │
  └─ Layer N (gate)
       └─ probabilities[seq_len, 128], indices[seq_len, 2]
    ↓
[Extract & Buffer]
  ├─ sequence_id = hash(prompt)
  ├─ For each token at position t:
  │   ├─ token_id = tokenizer.encode(token)[0]
  │   ├─ token_str = tokenizer.decode(token)
  │   ├─ token_position = t
  │   └─ For each layer l:
  │       ├─ expert_probabilities[l] = probs[t, :]
  │       ├─ selected_experts[l] = topk_indices[t]
  │       └─ selected_probs[l] = probs[t, topk_indices]
  │
  └─ Aggregate into batch buffer
    ↓
[Parquet Writer] (every 100 prompts)
  → Flush buffer to disk
```

---

## Hook Architecture

### Hook Registration

```python
def register_hooks(model):
    """Register forward hooks on all MoE router layers."""
    hooks = []
    for layer_idx, layer in enumerate(model.layers):
        gate = layer.block_sparse_moe.gate  # Module path (Mixtral)
        hook = gate.register_forward_hook(
            lambda module, input, output, 
                   layer_id=layer_idx: 
            capture_routing(module, input, output, layer_id)
        )
        hooks.append(hook)
    return hooks
```

### Hook Execution

```python
def capture_routing(module, input, output, layer_id):
    """
    Called on each forward pass through a MoE router.
    
    Args:
        module: The router gate module
        input: (x,) tuple with input tensor [batch, seq_len, hidden_dim]
        output: Pre-softmax logits [batch, seq_len, num_experts]
        layer_id: Index of this MoE layer (0 to num_layers-1)
    
    Operations:
        1. Softmax over expert dimension
        2. Extract top-k indices
        3. Store to thread-safe buffer (no blocking)
    """
    x = input[0]                    # [batch, seq_len, hidden_dim]
    logits = output                 # [batch, seq_len, num_experts]
    probs = torch.softmax(logits, dim=-1)  # [batch, seq_len, num_experts]
    
    # Top-k selection
    top_probs, top_indices = torch.topk(probs, k=2, dim=-1)
    
    # Store to global buffer (sequence-id → layer → token → data)
    # No tensor operations; just enqueue to buffer
    buffer.append({
        'layer_id': layer_id,
        'probs': probs.cpu().numpy(),
        'top_indices': top_indices.cpu().numpy(),
        'top_probs': top_probs.cpu().numpy(),
    })
```

---

## Storage Schema

### Parquet Structure (PyArrow)

```
Schema (Tab-Separated for Clarity):
┌─────────────────┬───────┬──────────────────────────────────┐
│ Column          │ Type  │ Notes                            │
├─────────────────┼───────┼──────────────────────────────────┤
│ sequence_id     │ str   │ Hash of prompt (32 char hex)     │
│ token_id        │ i32   │ Tokenizer vocab ID               │
│ token_str       │ str   │ Decoded token (readable)         │
│ token_position  │ i32   │ 0-indexed position in sequence   │
│ layer_index     │ i32   │ MoE layer number (0 to 31)       │
│ expert_probs    │ [f32] │ Softmax over 128 experts         │
│ top_experts     │ [i16] │ Indices of top-2 experts         │
│ top_probs       │ [f32] │ Probabilities of top-2 experts   │
│ dataset_source  │ str   │ "gsm8k" or "mbpp"                │
└─────────────────┴───────┴──────────────────────────────────┘

Example Row:
  sequence_id = "abc123def456..."
  token_id = 2054
  token_str = "math"
  token_position = 5
  layer_index = 0
  expert_probs = [0.002, 0.001, ..., 0.004, ..., 0.001]  (128 values)
  top_experts = [37, 82]
  top_probs = [0.15, 0.12]
  dataset_source = "gsm8k"
```

### File Organization

```
phase_1/
├── docs/                           # This folder
├── scripts/                        # Implementation code
│   ├── data_collection.py
│   ├── validation.py
│   └── dataset_loading.py
│
data/
├── gsm8k_routing_traces.parquet
│   └─ Rows: ~850k (8.5k prompts × 100 avg tokens × 1 record/token)
│   └─ Parquet partition by run # (optional)
│
├── mbpp_routing_traces.parquet
│   └─ Rows: ~100k (974 prompts × 100 avg tokens)
│
├── validation_reports/
│   └── phase_1_validation.md
│       ├─ Determinism check results
│       ├─ Randomization consistency
│       ├─ Data quality stats
│       └─ Sign-off
│
└── phase_1_metadata.json
    ├─ model: "mixtral-8x7b"
    ├─ num_layers: 32
    ├─ num_experts: 128
    ├─ top_k: 2
    ├─ datasets: {gsm8k: 8500, mbpp: 974}
    ├─ runs: 5
    ├─ execution_times: {...}
    └─ git_commit: "..."
```

---

## Key Implementation Details

### 1. Determinism Under `model.eval()`

During the prefill phase (no generation), expert routing is **bitwise deterministic** if:
- Model is in `eval()` mode (no dropout)
- Same input → same output every time
- No stochasticity in routing

**Verification:** Run same 5 prompts 5 times; expert probabilities must be identical to machine epsilon.

### 2. Token Averaging

For tokens appearing multiple times across the dataset:

```
For each unique (token, layer) pair:
  sum_prob[token][layer] += observed_probabilities
  count[token][layer] += 1

Final average:
  avg_prob[token][layer] = sum_prob[token][layer] / count[token][layer]
```

This is stored in Parquet **after all 5 runs complete**.

### 3. Memory-Efficient Streaming

Parquet is written in batches (every 100-1000 prompts) to avoid OOM:

```python
batch_buffer = []
for prompt in dataset:
    # Process prompt, append rows to batch_buffer
    batch_buffer.extend(extracted_rows)
    
    if len(batch_buffer) >= batch_size:
        write_parquet_batch(batch_buffer)
        batch_buffer = []
```

### 4. Hook Cleanup

After data collection, remove hooks to avoid memory overhead:

```python
for hook in hooks:
    hook.remove()  # Unregister hook
```

---

## Validation Checkpoints

### Determinism Check
- Input: Same 5 prompts
- Process: Run through model 5 times
- Output: Table of probabilities per run
- Pass Criterion: Bitwise identical across all runs

### Randomization Check
- Input: Full dataset, 2 different random orderings
- Process: Collect routing for both orderings
- Output: Comparison of top-k selections
- Pass Criterion: 100% match for same tokens

### Data Quality Check
- No NaN/inf values
- Expert probabilities sum to 1.0 (within epsilon)
- Top-k indices are valid (0 ≤ idx < 128)
- No missing sequences

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Throughput | 50-100 tokens/sec | Depends on GPU, batch size |
| Memory (peak) | <80GB | A100 HBM limit |
| Total runtime | 12-24 hours | For 9.5k prompts × 5 runs |
| Determinism | 100% | Zero variance across runs |
| Data compression ratio | 10:1+ | Parquet snappy compression |

---

## Error Handling Strategy

```
Try:
  ├─ Load dataset
  ├─ Load model
  ├─ Register hooks
  ├─ Forward pass (capture hooks)
  ├─ Write Parquet
  Catch CUDA OOM:
    └─ Flush buffer, reduce batch size, resume from checkpoint
  Catch Tokenization Error:
    └─ Skip prompt, log warning, continue
  Catch Hook Fire Failure:
    └─ Verify layer path, retry with adjusted path
  Finally:
    └─ Remove hooks, free GPU memory
```

---

## Testing Strategy

### Phase 1a: Unit Tests (Day 1-2)
- [ ] Dataset loader on 10 examples
- [ ] Model loading and eval mode
- [ ] Hook registration fires correctly
- [ ] Parquet schema creation

### Phase 1b: Integration Test (Day 3-4)
- [ ] Full pipeline on 100 GSM8K + 100 MBPP examples
- [ ] All outputs generated
- [ ] Parquet file readable
- [ ] Validation checks pass

### Phase 1c: Small-Scale Test (Day 5-6)
- [ ] Full pipeline on 500 examples
- [ ] Determinism check (same 5 prompts, 5 runs)
- [ ] Memory usage profiling
- [ ] Execution time estimation

### Phase 1d: Full-Scale Validation (Day 7+)
- [ ] All 9.5k examples, 5 runs
- [ ] All validation checks pass
- [ ] Metadata and logging complete
- [ ] Ready for Phase 2 analysis
