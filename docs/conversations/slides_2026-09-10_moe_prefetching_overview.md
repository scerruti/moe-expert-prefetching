---
marp: true
theme: default
paginate: true
header: "Sparse MoE Architecture Analysis"
footer: "Activation Density & Token Routing"
style: |
  section {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  }
  h1 {
    color: #00529B;
  }
  h2 {
    color: #333333;
  }
---

<!-- _class: lead -->
# Speculative Expert Prefetching
## Optimizing Mixture-of-Experts via Lookahead Routing
**Research Collaboration: UC San Diego & e3 Civic High**

---

## 1. The Bottleneck in MoE

* **The Architecture**: Models like Mixtral 8x7B route tokens to a subset of experts (e.g., top-2 out of 8).
* **The Problem**: Transferring expert weights from host memory to accelerator HBM creates GPU pipeline stalls.
* **The Goal**: Accelerate processing by prefetching expert weights before they are required for computation.

---

## 2. Core Hypothesis: Determinism

* **Evaluation Mode**: During the prefill phase with static input sequences, routing decisions are mathematically invariant.
* **No Stochastic Noise**: Probability distributions $P(e \mid x)$ are generated deterministically prior to top-$k$ selection.
* **The Advantage**: We can reliably capture consistent routing patterns from a single pass per dataset without randomized variance.

---

## 3. Dataset Benchmarks & Domain Strategy

We are comparing probability distributions across distinct cognitive domains to evaluate structural versus semantic routing:

* **GSM8K (Math)**: 
  * Chain-of-thought deduction, arithmetic, numeric constants.
  * Hypothesis: Concentrated activation in algorithmic/computational experts.
* **MBPP (Python)**: 
  * Code syntax, indentations, logic structures.
  * Hypothesis: Specialized routing for syntactic tokens distinct from numeric paths.

---

## 4. Telemetry & Infrastructure

* **Target Hardware**: Cloud GPU environment (e.g., RunPod).
* **Framework**: PyTorch.
* **Extraction Mechanism**: Forward hooks registered on `block_sparse_moe.gate` linear layers to intercept pre-softmax logits and normalized probabilities.
* **Storage Engine**: Columnar datastore via Apache Parquet (PyArrow) for low-overhead nesting of expert probability float arrays.

---

## 5. Five-Phase Implementation Roadmap

1. **Deterministic Data Collection**: Hooking PyTorch models to capture layer-by-layer router output.
2. **Statistical Verification**: Calculating Jensen-Shannon Divergence across shared tokens (GSM8K vs. MBPP).
3. **Speculative Predictor**: Training an auxiliary lookahead model to forecast $\hat{\mathcal{K}}(x_{t+n}, l)$.
4. **Cache Simulation**: Modeling a Least Recently Predicted (LRP) eviction policy.
5. **Pipeline Profiling**: Measuring empirical hardware performance.

---

## 6. Success Metrics

Our evaluation will hinge on three primary performance indicators:

* **Prediction Recall@$(k+m)$**: The frequency with which our lookahead predictor successfully includes the true top-$k$ experts within a preloaded candidate pool.
* **Cache Hit Rate**: The percentage of execution steps where required expert parameters are already resident in local memory.
* **Pipeline Stall Reduction**: The overall decrease in GPU idle time spent waiting on PCIe memory transfers.