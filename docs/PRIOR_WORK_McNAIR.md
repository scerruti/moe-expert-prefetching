# Prior Work: McNair Research on Qwen3-VL Spatial Reasoning

## Overview

This project builds on insights from concurrent research conducted by Waleed Alghaiti and Taylor Berg-Kirkpatrick at UCSD, published as *"Uncovering the Limits of Spatial Reasoning in Vision-Language Models"*. Their work analyzed expert routing patterns in Qwen3-VL-30B on the LEGOLite spatial reasoning benchmark through four phases of telemetry, clustering, and ablation experiments.

**Critical Connection:** Both projects use PyTorch forward hooks to capture MoE routing telemetry during the prefill phase. The McNair findings validate our core assumptions and provide proven methodologies for Phase 1-5 execution.

---

## Key Findings Relevant to Speculative Prefetching

### 1. Determinism of Expert Routing (Phase 1-2)

**Finding:** Expert routing is **bitwise deterministic** under evaluation mode with static inputs.

**Evidence:**
- Jensen-Shannon divergence between identical image/prompt runs: **0.0215** (highly consistent)
- Same image → identical expert ranking across multiple runs
- Two full inference passes on 400 questions showed ~1 percentage point accuracy variance (Table 3)

**Implication for Your Project:** Single-run telemetry per dataset is sufficient. No need for repeated sampling per prompt; determinism assumption holds. Running 5 randomized passes validates consistency across prompt orderings.

### 2. Modality-Specific Expert Routing (Phase 1)

**Finding:** Image and text tokens route to **completely different expert subsets**.

**Evidence:**
- Image vs text token expert routing: 0/10 overlap (completely disjoint)
- Jensen-Shannon divergence (image || text): 0.0213-0.0217
- Image-token entropy: 6.954 bits (spread across more experts)
- Text-token entropy: 6.389 bits (concentrated on fewer experts)

**Implication for Your Project:** Domain separation is strong. GSM8K (text-heavy math) and MBPP (code) will show distinct routing patterns—exactly what your Phase 2 cross-domain analysis should reveal. Shared-token analysis will be noisy but meaningful.

### 3. Weak Signal in Localization (Phase 3)

**Finding:** Spatial reasoning errors are **distributed broadly across layers**, not concentrated in specific components.

**Evidence:**
- Z-score layer ranking: max score only **0.055** (far below z > 2 threshold for outliers)
- K-medoids clustering: silhouette scores all **< 0.25** (weak/overlapping clusters)
- McNemar's test: p = 0.25 (ablation does not significantly improve accuracy)
- Holdout validation: -2 percentage point drop (not +3.5 points from test set)

**Implication for Your Project:** Prefetching won't improve *accuracy*, but it will improve *latency*. Expert routing is distributed, so your lookahead predictor should expect non-trivial entropy. Don't expect to identify a small "critical" expert set via clustering alone.

---

## Reusable Methodologies

### Layer Scoring via Z-Score (Phase 3, Section 7.2)

The McNair team scored layers by how much their expert activations correlated with wrong answers:

$$z_e = \frac{\text{observed\_wrong}_e - n_e \cdot \hat{p}}{\sqrt{n_e \cdot \hat{p}(1-\hat{p})}}$$

**Where:**
- $n_e$ = # questions expert $e$ was activated on
- $\hat{p}$ = base error rate (0.765 in their case)
- $\text{observed\_wrong}_e$ = # of those $n_e$ questions expert $e$ got wrong

**Recommended Use:** In Phase 2 analysis, score layers by routing entropy divergence or prediction difficulty. Layers with high divergence between domains are good prefetch candidates.

### Expert Fingerprinting & Clustering (Phase 3, Section 7.3)

**Approach:**
1. Build binary activation matrix: 128 experts × 200 questions (1 = expert activated, 0 = not)
2. Compute pairwise Jaccard distance (not Euclidean—sparse binary data):
   $$d_J(A, B) = 1 - \frac{|A \cap B|}{|A \cup B|}$$
3. Run k-medoids clustering (k ∈ {2, 4, 6, 8})
4. Compute silhouette scores; keep clusters with score > 0.25

**Key Insight:** Jaccard distance is better than Euclidean for sparse binary activation patterns. Their silhouette scores were weak (0.147–0.236), suggesting experts don't form tight, separable clusters.

**Recommended Use:** Use this for Phase 3 (predictor training). Identify clusters of experts that co-activate; this can inform your multi-label classification loss weighting.

### Ablation Design with Random Controls (Phase 3, Section 7.4)

**Three-Part Framework:**

**Part A:** Test intervention types on worst layer:
- Full block ablation (skip entire layer)
- MoE-only ablation (zero MoE output, keep attention)
- Expert-level ablation (zero flagged "bad" cluster experts)

**Part B:** Scale best intervention across 1, 3, 5 layers

**Part C:** Random-ablation control (ablate same number of randomly chosen layers; repeat 10 trials)

**Statistical Tests:**
- McNemar's test on paired per-question outcomes
- Z-score vs random: $z = \frac{\text{targeted\_accuracy} - \text{random\_mean}}{\text{random\_std}}$
- 95% CI: $p \pm 1.96\sqrt{p(1-p)/n}$
- Holdout validation on held-out 200 questions (never touched during model selection)

**Recommended Use:** Adapt this exact framework for Phase 5 validation. Compare Recall@(k+m) and pipeline stall reduction against random-prefetch baselines.

---

## Data & Artifacts

| Component | McNair's Approach | Your Project |
|---|---|---|
| **Model** | Qwen3-VL-30B (48 layers, 128 experts, top-8) | Mixtral 8x7B (24 layers, 8 experts, top-2) |
| **Dataset** | LEGOLite (400 questions, 4 spatial reasoning categories) | GSM8K + MBPP (~9.5k examples) |
| **Telemetry** | Per-question expert routing (binary top-8 activations) | Per-token probability distributions (averaged) |
| **Storage** | JSON (results.json) | Parquet (columnar, compressed) |
| **Key Data File** | `data/phase2/runpod_second/results.json` | `<your_parquet_files>` (Phase 1 output) |

---

## Implementation Notes

### PyTorch Hook Registration

McNair's Phase 1 notebook (Google Colab) shows direct hook attachment to Qwen3's `QwenVLMoeTextTopKRouter.forward()`. For Mixtral, attach to `block_sparse_moe.gate` (similar structure).

**Critical Detail:** Router output is log-softmax, not softmax. Convert with `torch.exp()` if needed, and validate that post-softmax probabilities sum to ~1.0 per token.

### Dataset Considerations

- **Prompt diversity:** Both GSM8K and MBPP have natural variation in token lengths. Average your probabilities per expert rather than collecting raw logits (handles variable-length sequences cleanly).
- **Determinism validation:** Same as McNair—run the same prompt subset 5 times, compare routing distributions via JS divergence. Should be near-zero.
- **Category annotation:** GSM8K has no pre-assigned difficulty labels. Use operator type (division, decimals, percentages) or step-count heuristics to stratify.

---

## Lessons for Your Phases

| Phase | McNair Finding | Your Application |
|---|---|---|
| **Phase 1** | Routing is deterministic, single-run capture suffices | Validate with 5 random runs; JS divergence should be <0.05 |
| **Phase 2** | Domain separation is strong but entropy varies by layer | Expect 10-20% JSD between GSM8K and MBPP; analyze per-layer |
| **Phase 3** | Clustering is weak (silhouette < 0.25) | Use for feature extraction; don't expect tight expert groups |
| **Phase 4** | No localized effect exists | Simulate prefetch based on probabilistic predictions, not hard thresholds |
| **Phase 5** | McNemar's test + random controls are gold standard | Use identical statistical framework; include holdout validation |

---

## Reference

**Full Paper:** Alghaiti, W. & Berg-Kirkpatrick, T. "Uncovering the Limits of Spatial Reasoning in Vision-Language Models." arXiv preprint, 2025.

**Project Structure:** Organized into Phase 0 (baseline), Phase 1 (routing analysis), Phase 2 (full telemetry), Phase 3 (statistical analysis & ablation). Each phase has standalone scripts and intermediate outputs for auditability.

**Key Reproducibility Assets:**
- `scripts/phase3/step2/` — Z-score layer scoring (Python, CPU-only)
- `scripts/phase3/step4/` — K-medoids clustering (Jaccard distance, silhouette scoring)
- `scripts/phase3/step6/` — Statistical analysis (McNemar, CIs, plots)

---

**Last Updated:** 2026-09-11
