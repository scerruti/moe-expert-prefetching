# System Design Document: Speculative Expert Prefetching via Lookahead Routing in Mixture-of-Experts (MoE)

## 1. Architectural Overview & Mathematical Formulation

### 1.1 Router Mechanics & Top-k Routing Formulation

In a sparse Mixture-of-Experts (MoE) layer $l$, an input representation token $x \in \mathbb{R}^d$ is routed dynamically to a subset $k$ of $E$ total feed-forward network (FFN) experts $\{E_1, E_2, \dots, E_E\}$.

Let $W_g^{(l)} \in \mathbb{R}^{d \times E}$ denote the router parameter matrix at layer $l$. The gating logits are calculated via affine transformation:

$$H(x) = x W_g^{(l)}$$

A standard softmax normalization produces the probability distribution across all experts:

$$P(e_i \mid x) = \frac{\exp(H(x)_i)}{\sum_{j=1}^{E} \exp(H(x)_j)}$$

The discrete top-$k$ selection picks the $k$ highest routing probabilities:

$$\mathcal{K}(x) = \operatorname{TopK}\left(P(e \mid x), k\right)$$

The final expert layer output is the probability-weighted sum of selected expert representations:

$$y = \sum_{i \in \mathcal{K}(x)} P(e_i \mid x) E_i(x)$$

### 1.2 Determinism During the Fill (Prefill) Phase

Under evaluation mode (`model.eval()`) with static input embeddings and frozen model parameters:
* The feed-forward projections and self-attention operations over static sequences are deterministic.
* Router gating logits $H(x)$ and probability distributions $P(e \mid x)$ are mathematically invariant across repeated passes.
* Expert activations are deterministic functions of the input token sequence:

$$P(e \mid x_{t}, l, \text{run}_1) \equiv P(e \mid x_{t}, l, \text{run}_n)$$

**Implication:** Stochastic variation exists only during generation phases utilizing randomized decoding strategies (temperature $T > 0$, top-$p$, top-$k$ token sampling). During the prefill/evaluation phase, single-run telemetry per dataset suffices for distribution profiling.

---

## 2. Dataset Benchmarks & Domain Strategy

| Domain | Dataset | Size | Context Profile | Routing Hypothesis |
| :--- | :--- | :--- | :--- | :--- |
| **Mathematical Reasoning** | GSM8K | 8.5k examples | Chain-of-thought arithmetic deduction, numeric constants | High activation entropy in early layers; specialization in algorithmic/computational experts in deep layers. |
| **Code Generation** | MBPP | 974 examples | Python syntax, indentation, logic structures | Concentrated expert allocation on syntactic tokens; domain separation from numeric reasoning paths. |

### Comparative Analysis: Domain Separation vs. Invariant Structure

* **Domain-Specific Specialization:** Measures whether technical tokens (`def`, `import`, `=`, `+`, numeric digits) map to dedicated sub-networks distinct from math tokens.
* **Structural Invariance:** Common grammatical tokens (articles, punctuation, whitespace, conjunctions) are tracked across datasets to establish baseline router behavior independent of high-level semantic intent.
* **Shared Token Analysis:** Identify common tokens appearing in both GSM8K and MBPP; compare their probability distributions across the two domains to isolate context-dependent routing divergence.

### Dataset Specifics

**GSM8K (Grade School Math 8K):**
* 8,500 crowdsourced arithmetic word problems at elementary/middle school level.
* Focus: Basic mathematical concepts (addition, subtraction, multiplication, division, percentages, rates, fractions).
* Ground truth includes step-by-step solutions with intermediate calculations marked (e.g., `<<10+5=15>>`).
* Can be sliced by step count (complexity), operator type (division, decimals), or keyword matching (money, rates/speed, fractions).
* Variants available: GSM-Hard (larger numbers for true reasoning test), GSM8K-Socratic (explicit sub-question breakdowns).

**MBPP (Mostly Basic Programming Problems):**
* 974 total crowdsourced Python coding problems (500 train, 427 test, validation split).
* **Standard for leaderboards:** MBPP-Sanitized (427 hand-verified problems with clarified prompts and valid test cases).
* Domain breakdown: ~58% math/arithmetic, ~43% list processing, ~19% string manipulation.
* Each problem: `{task_id, text (prompt), code (reference impl), test_list (3 assertions)}`.
* Entry-level CS: focuses on function-level code synthesis, avoiding multi-file projects or complex OOP patterns.
* **Modern variant:** MBPP+ (EvalPlus) generates ~35x more unit tests per problem to catch edge-case bugs; frontier models typically drop 10–15 percentage points on MBPP+.

**Token Cardinality:** Processing all tokens within ~9.5k examples during prefill (not just prompt text); exact token count varies by vocabulary size and natural prompt lengths in each dataset.

**Randomization Strategy:** Run the data collection 5 times with randomized prompt orderings to validate consistency. For validation, run a subset (e.g., 200 random prompts) against repeated randomized orderings to confirm deterministic routing.

---

## 3. Five-Phase Implementation Roadmap

### Phase 1: Telemetry Hooks & Deterministic Data Collection

**Objective:** Capture pre-softmax logits, post-softmax routing distributions, and top-$k$ assignments directly from model forward passes.

**Mechanism:** PyTorch forward hooks registered on router linear projection layers (`block_sparse_moe.gate`):

```python
def router_hook(module, input, output):
    # Extract logits and normalized probabilities
    probs = torch.softmax(output, dim=-1)
    # Store to tensor buffer (no activation, deterministic)
    # Save running averages for repeated tokens
```

**Data Collection Process (Pseudocode):**

```python
# Load target MoE model (Mixtral 8x7B for MVP validation, or confirmed-MoE model for research)
# and tokenizer using Hugging Face Transformers
# Initialize empty dictionary storing expert probability averages

for prompt in dataset:
    # Tokenize input; run through model (no generation, prefill phase only)
    # Capture output from each MoE router layer before top-k selection
    # For each token: Update running sum of probabilities for each expert
    # Store count of how many samples we've seen that token
    
# Finalize storage: running sum / count = average probability per expert
```

**Storage Schema:** Apache Parquet (via PyArrow) with the following columnar structure:

| Column | Type | Description |
| :--- | :--- | :--- |
| `sequence_id` | string (hash of prompt) | Unique identifier for the sequence/prompt |
| `token_id` | int32 | Tokenizer-specific ID for the token |
| `token_str` | string | Human-readable token string |
| `token_position` | int32 | Position within the sequence (0-indexed) |
| `layer_index` | int32 | Which MoE layer (0 to num_layers-1) |
| `expert_probabilities` | float32[E] | Softmax probabilities across all E experts |
| `selected_experts` | int16[k] | Indices of top-$k$ activated experts |
| `selected_probs` | float32[k] | Probabilities for the selected experts |
| `dataset_source` | string | "gsm8k" or "mbpp" |

**Key Properties:**
* Parquet handles nested float and int arrays cleanly for storing probability vectors and expert indices.
* High compression efficiency; data can be queried directly using DuckDB or Polars for fast downstream slicing.
* Columnar format facilitates statistical aggregation (e.g., averaging probabilities per expert across prompts).

**Validation Steps:**
1. Run the same prompt through the model 5 times; verify expert probabilities are bitwise identical.
2. For a subset of 200 prompts, randomize their order within each dataset and repeat collection; confirm consistency.
3. Cross-check tensor values against a dense model baseline (dense vs. sparse probabilities) to ensure meaningful routing divergence.

---

### Phase 2: Statistical Verification & Domain Cross-Comparison

**Objective:** Verify router distribution determinism and profile cross-domain probability shifts.

**Run Strategy:**

* **Sampling Subsets:** Stratified random sample of 200 prompts per dataset to minimize memory footprint while enabling cross-run analysis.
* **Consistency Check:** Verify zero variance across duplicated runs on sample subset:
  $$\Delta P = \max \left| P(e \mid x)_{\text{run}_1} - P(e \mid x)_{\text{run}_2} \right| = 0$$

**Distribution Shift Metrics:**

* **Jensen-Shannon Divergence (JSD):** Calculate JSD between domain-averaged distributions:
  $$D_{JS}(P_{\text{GSM8K}} \| P_{\text{MBPP}}) = \frac{1}{2} D_{KL}(P_{\text{GSM8K}} \| M) + \frac{1}{2} D_{KL}(P_{\text{MBPP}} \| M)$$
  where $M = \frac{1}{2}(P_{\text{GSM8K}} + P_{\text{MBPP}})$.

* **Shared Token Analysis:** Identify common tokens across both datasets (e.g., "def", "+", "if", punctuation). For each shared token, compare its probability distribution across the two domains to isolate context-dependent vs. invariant routing behavior.

* **Layer-wise Divergence:** Compute JSD at each layer to identify which layers exhibit strongest domain-specific specialization vs. general linguistic processing.

---

### Phase 3: Speculative Predictor Architecture

**Objective:** Train a lightweight auxiliary model to forecast future expert activations ahead of actual forward execution.

**Predictor Architecture:**

* **Sliding-Window Lightweight Transformer:** Consumes hidden representations from layer $l - m$ (or $l - n$ tokens) to predict upcoming top-$k$ expert activations.
* **Input:** Hidden state at token position $t - n$ through $t$ (past context window of $n$ tokens, default $n = 3$).
* **Output:** Multi-label classification target $\mathcal{K}(x_{t+1}, l)$ (which experts are selected at position $t+1$ for layer $l$).

**Loss Function:** Asymmetric multi-label cross-entropy:

$$\mathcal{L} = -\sum_{i=1}^{E} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

where $y_i = 1$ if $i \in \mathcal{K}(x_{t+1}, l)$, else 0.

**Training Regime:**
* Use collected routing traces as training data (GSM8K + MBPP combined).
* Split: 80% train, 10% validation, 10% test.
* Optimize to minimize false negatives (recall-focused) to ensure prefetched experts cover true top-$k$ selections.

---

### Phase 4: Cache Simulation & Prefetching Pipeline

**Objective:** Simulate asynchronous hardware memory pipeline; measure prefetch effectiveness under realistic latency constraints.

**Prefetch Execution Rules:**

* **Prediction Horizon:** $H$ = number of tokens/layers ahead the prediction occurs.
* **Transfer Latency Model:** Weights for expert $E_i$ must finish loading before layer $l$ begins execution.
* **Replacement Policy:** Least Recently Predicted (LRP) eviction policy alongside static caching for consistently activated experts.

**Simulation Pipeline:**

1. For each token in sequence:
   - Predictor forecasts top-$(k+m)$ candidates $(m$ = buffer size, typically 2–4).
   - Initiate async weight transfer for preload window $H$.
   - Compare actual routing against preloaded candidates; record hit/miss.
   - Measure GPU idle time waiting for expert weights (baseline vs. prefetch).

---

### Phase 5: Pipeline Profiling & Validation Metrics

**Objective:** Benchmark the speedup, accuracy, and pipeline stall reductions achieved by speculative prefetching.

**Primary Evaluation Metrics:**

* **Prediction Recall@$(k+m)$:** Fraction of actual activated experts contained within the top $(k+m)$ preloaded candidates.
  $$\text{Recall@}(k+m) = \frac{|\mathcal{K}_{\text{true}} \cap \mathcal{K}_{\text{preload}}|}{|\mathcal{K}_{\text{true}}|}$$
  Goal: >95% recall with $k + m \leq 4$ (prefetch 4 of 128 experts in Qwen3, or 4 of 8 in Mixtral).

* **Cache Hit Rate:** Ratio of required expert layers already resident in accelerator memory at execution time.
  $$\text{Hit Rate} = \frac{\text{\# experts resident}}{k \times \text{\# layers}}$$

* **Pipeline Stall Reduction:** Percentage decrease in GPU idle time waiting on memory transfers vs. standard dense model:
  $$\text{Stall Reduction} = \frac{\text{Stall}_{\text{baseline}} - \text{Stall}_{\text{prefetch}}}{\text{Stall}_{\text{baseline}}} \times 100\%$$

---

## 4. Model & Infrastructure

### Target Model

**MVP / Software Validation:** **Mixtral 8x7B** (8 experts, top-2 routing, well-documented sparse architecture)
- Used to validate telemetry hooks, data collection pipeline, and Phase 1-5 infrastructure
- Can run locally on Colab with manageable memory footprint
- **Note:** Mixtral architecture well-understood and reliable for code/algorithm validation, but not necessarily the model used for the final research

**Research Model (Candidates):** 
- **Qwen3.8-27B:** ❌ NOT MoE (confirmed dense model with Gated DeltaNet + Attention + FFN). **Do not use for this project.**
- **DeepSeek-Coder-V2-Lite:** Sparse MoE, explicitly optimized for code synthesis and mathematical reasoning (preferred if confirmed MoE)
- **Qwen3 MoE (30B or 235B):** 128 routed experts with top-2 activation (candidate if confirmed as MoE variant, not the dense Qwen3.8)

**See docs/MODEL_VERIFICATION.md for architecture confirmation details.**

**Other Variants Considered:**
* **DBRX Instruct (Databricks):** 132B total / 36B active; programming & data tasks emphasis.
* **DeepSeek-V3:** 256 experts per layer (extreme scale, heavy memory overhead; post-MVP).
* **Snowflake Arctic:** 128 experts with top-2 routing; enterprise tuning.

**Strategy:**
1. Validate all software (telemetry, hooks, data pipeline) on **Mixtral 8x7B** (Colab) to prove code works
2. Confirm MoE status of **Qwen3.8-27B** before committing to research runs
3. If Qwen3.8-27B confirmed MoE: migrate telemetry to that model on RunPod for actual research
4. If Qwen3.8-27B not MoE: select next best confirmed-MoE model and proceed
5. No code changes needed for model swap—only module path adjustments for router layer hooks

### Environment

**MVP Validation (Colab):**
- **Hardware:** Google Colab GPU (T4 or A100 if available)
- **Purpose:** Validate telemetry hooks, data collection pipeline, and Phase 1-5 infrastructure on Mixtral 8x7B
- **Note:** Approach requires validation to confirm Colab can handle data collection workload without running out of memory or compute time

**Research Runs (RunPod):**
- **Hardware:** Cloud GPU instance (A100 or H100 via RunPod, or equivalent)
- **HBM:** 80GB (A100) or 141GB (H100) sufficient to hold model weights + routing telemetry without memory strain
- **Purpose:** Run telemetry on confirmed-MoE research model (pending Qwen3.8-27B verification)

**Framework & Dependencies:**
* `pytorch` + `torch.cuda` for tensor operations and hooks
* `transformers` (Hugging Face) for model loading and tokenizer
* `accelerate` for distributed loading on multi-GPU setups (if needed)
* `pyarrow` for Parquet I/O
* `duckdb` or `polars` for downstream data querying and slicing

---

## 5. Analysis Goals & Actionable Outputs

1. **Empirical Distribution Baseline:** Quantify whether expert assignment is highly domain-specific or if common tokens share routing pathways regardless of context.
2. **Trained Lookahead Predictor:** Small auxiliary model (MLP/lightweight RNN) trained to predict top-$k$ expert activations from earlier-layer hidden states.
3. **Prefetch Simulation Metrics:** Concrete benchmarks measuring:
   - Recall@$(k+m)$: How often preloaded candidates cover true top-$k$ selections.
   - Cache hit rates under realistic memory constraints.
   - GPU pipeline stall reduction (time savings from speculative prefetching).

---

## 6. Timeline & Scope

### Scope

An empirical research investigation to establish a proof-of-concept for speculative expert prefetching, focused on deterministic routing behavior and domain-specific expertise allocation.

### Milestones

1. **Complete Phase 1 telemetry collection** and confirm single-run deterministic outputs across 5 randomized runs on GSM8K and MBPP.
2. **Perform cross-domain divergence analysis** between math and code prompts; identify shared tokens and their routing consistency.
3. **Train and benchmark the lookahead predictor** against collected routing traces; measure recall@$(k+m)$ and cache hit metrics.

### Optional Extensions (Post-MVP)

* Integrate additional datasets (MMLU for general knowledge, conversational datasets).
* Validate against production inference pipelines (not just simulation).
* Extend to multimodal routing (Qwen3-VL) to test visual + text token handling.

---

## 7. Expected Outcomes & Visualization

### Data Structure Diagram

Data collected per token: `(sequence_id, token_id, token_str, token_position, layer_index, expert_probabilities[E], selected_experts[k], dataset_source)`

Stored in columnar Parquet format for efficient aggregation and querying.

### Key Artifacts

1. **Telemetry Parquet Files:** Raw routing traces (one per dataset, partitioned by layer).
2. **Statistical Analysis Report:** Jensen-Shannon divergence, shared token analysis, layer-wise specialization profiles.
3. **Trained Predictor Checkpoint:** Saved weights for lookahead model (PyTorch `.pt` file).
4. **Simulation Results:** Recall@$(k+m)$, cache hit rates, pipeline stall reduction metrics (CSV/plots).
5. **Slide Deck:** High-level findings formatted in Marp (Markdown presentation) for presentation/sharing.

---

## 8. Implementation Notes

### Critical Design Decisions

* **"Fill" Phase Only:** Data collection focuses on the prefill phase (prompt processing), not generation. This avoids stochastic decoding noise.
* **No Sparse Memory Formats:** Initial implementation uses dense tensors for simplicity; can revisit sparse storage/computation later if needed.
* **Single Pass Suffices:** Due to determinism, one forward pass per prompt is sufficient; no need for repeated sampling per prompt.
* **Averaged Probabilities:** For repeated tokens, track running sum and count; divide at the end to get average expert probabilities per token.

### Known Challenges

* **Data Scale:** 9.5k prompts × multiple layers × 128 experts (Qwen3) = large probability matrices. Addressed by Parquet compression + columnar storage.
* **Model Switching:** Migrating from Mixtral to Qwen3 requires updating module path names (e.g., `block_sparse_moe.gate` → `model.layers[i].block_sparse_moe.gate`). Core hook logic unchanged.
* **Token Determinism Across Runs:** Assumes model settings (temperature, top-p, etc.) remain frozen; any stochasticity (e.g., from dropout or generation strategies) invalidates the assumption. Enforce `model.eval()` and disable generation-based randomness.

---

## 9. References & Resources

* **Original Mixtral Paper:** [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
* **Qwen Models:** [Qwen GitHub](https://github.com/QwenLM)
* **Parquet Format:** [Apache Arrow Parquet](https://parquet.apache.org/)
* **Lookahead Decoding Inspiration:** Speculative decoding and prefetching concepts adapted from recent work on efficient LLM inference.

---

**Project Status:** Pre-implementation (design finalized, awaiting Phase 1 telemetry hook development).  
**Last Updated:** 2026-09-11
