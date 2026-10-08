# Phase 2 Detailed Checklist

**Last Updated:** 2026-10-02  
**Scope:** Granular task breakdown for all 11 Phase 2 GitHub issues

---

## #40: Data Loading & Preparation

**Issue:** https://github.com/scerruti/moe-expert-prefetching/issues/40  
**Duration:** 2-3 days  
**Status:** Not started

### Code Setup
- [ ] Create `phase_2/scripts/data_loader.py`
- [ ] Import dependencies: polars, pandas, pyarrow, logging
- [ ] Set up logging configuration for data loading operations
- [ ] Create `RoutingDataLoader` class skeleton

### Schema Validation
- [ ] Define expected Parquet schema (9 columns) as constant
- [ ] Implement `validate_schema()` method
  - [ ] Check column names match: sequence_id, token_id, token_str, token_position, layer_index, expert_probs, selected_experts, selected_probs, dataset_source
  - [ ] Check data types (int32, float32, string, etc.)
  - [ ] Validate value ranges (layer_index: 0-31, expert_probs: 0-1, etc.)
- [ ] Log all schema validation results
- [ ] Raise informative errors for schema mismatches

### Data Quality Analysis
- [ ] Implement `quality_report()` method
  - [ ] Count null/missing values per column
  - [ ] Compute value ranges (min/max) for numeric columns
  - [ ] Check for NaN/inf values in probability columns
  - [ ] Verify expert probabilities sum to 1.0 (within epsilon 1e-5)
  - [ ] Count unique sequences, tokens, layers
  - [ ] Generate statistics for each dataset (GSM8K, MBPP)
- [ ] Export quality report as Markdown (`data_quality_report.md`)
  - [ ] Table format: column, null_count, range, notes
  - [ ] Pass/fail status per check
  - [ ] Recommendations if issues found

### Summary Statistics
- [ ] Implement `summary_stats()` method
  - [ ] Total row count per dataset
  - [ ] Tokens per layer (mean, min, max)
  - [ ] Experts per layer (mean, min, max)
  - [ ] Unique prompts/sequences
  - [ ] Average sequence length
  - [ ] Expert activation frequencies
- [ ] Export as CSV: `summary_statistics.csv`
- [ ] Create human-readable stats table for reporting

### Memory Profiling
- [ ] Implement memory usage tracking during load
  - [ ] Peak RAM usage (using psutil)
  - [ ] Per-dataset memory footprint
  - [ ] Estimated memory for full dataset
- [ ] Export profile: `memory_profile.txt`
- [ ] Estimate if dataset fits in standard GPU HBM

### Unified Loader Interface
- [ ] Implement `load()` method with Polars lazy evaluation
- [ ] Add filtering methods:
  - [ ] `filter_by_dataset(name: str)` – GSM8K or MBPP
  - [ ] `filter_by_layer(layer_idx: int)`
  - [ ] `filter_by_token(token_id: int)`
  - [ ] `sample(n: int, stratify: bool)` – Stratified sampling
- [ ] Add export methods:
  - [ ] `to_pandas()`, `to_polars()`, `to_arrow()`
- [ ] Implement streaming/chunked processing for large analyses
- [ ] Add caching to avoid reloading

### Testing & Documentation
- [ ] Write docstrings for all methods (NumPy style)
- [ ] Test on sample data (100 rows)
- [ ] Document data assumptions and edge cases
- [ ] Add usage examples in module docstring

### Deliverables
- [ ] `phase_2/scripts/data_loader.py` – Complete implementation
- [ ] `phase_2/data/data_quality_report.md` – QA report
- [ ] `phase_2/data/summary_statistics.csv` – Statistics table
- [ ] `phase_2/data/memory_profile.txt` – Memory usage
- [ ] Tests pass for schema validation and quality checks

---

## #41: Determinism Verification

**Issue:** https://github.com/scerruti/moe-expert-prefetching/issues/41  
**Duration:** 2-3 days  
**Depends on:** #40

### Determinism Analysis Setup
- [ ] Create `phase_2/scripts/determinism.py`
- [ ] Import scipy.spatial.distance (KL divergence), numpy, pandas

### Sampling Strategy
- [ ] Implement `stratified_sample(loader, sample_size=200)`
  - [ ] Stratify by: dataset (GSM8K/MBPP), token_position (early/mid/late), layer
  - [ ] Ensure representative coverage across dimensions
  - [ ] Document sampling rationale
- [ ] Extract prompts for each sample
- [ ] Get probability distributions for same 5 prompts across all runs

### Deviation Computation
- [ ] For each sampled prompt, compute max deviation across 5 runs
  - [ ] Per token: max|P_run1 - P_run2| ... max|P_run1 - P_run5|
  - [ ] Per layer: aggregate deviation per layer
  - [ ] Per expert: max deviation per expert ID
  - [ ] Global: max deviation across all tokens/layers/experts
- [ ] Implement checks for different distance metrics:
  - [ ] L-infinity norm (max absolute difference)
  - [ ] L2 norm (Euclidean distance)
  - [ ] KL divergence between distributions
- [ ] Document which metric used in final report

### Pass/Fail Criteria
- [ ] Define thresholds for pass criterion (default: <1e-6 max deviation)
- [ ] Implement `pass_fail_determination(max_dev, threshold)`
- [ ] Handle edge cases (missing data, NaN values)
- [ ] Log reasoning if fails

### Statistical Summary
- [ ] Compute statistics by layer:
  - [ ] Mean, median, std, min, max deviation
  - [ ] Per-layer pass/fail status
- [ ] Compute statistics by expert:
  - [ ] Which experts have highest variance?
  - [ ] Are certain experts more non-deterministic?
- [ ] Generate comparison tables (GSM8K vs MBPP)
- [ ] Export: `determinism_statistics.csv`

### Visualization
- [ ] Create plot: max deviation per layer
- [ ] Create plot: max deviation per expert (top-20)
- [ ] Create boxplot: deviation distribution per layer
- [ ] Export as PNG

### Report Generation
- [ ] Write `determinism_verification.md` report with:
  - [ ] Executive summary (pass/fail)
  - [ ] Methodology (sampling, metrics, thresholds)
  - [ ] Results tables
  - [ ] Visualizations embedded
  - [ ] Conclusions and implications
  - [ ] Recommendations (e.g., "safe to proceed to Phase 3")

### Testing & Documentation
- [ ] Unit tests for deviation computation
- [ ] Test on toy dataset (10 prompts, 2 runs)
- [ ] Document any non-determinism findings
- [ ] Add edge case handling (missing runs, NaN values)

### Deliverables
- [ ] `phase_2/scripts/determinism.py` – Complete implementation
- [ ] `phase_2/reports/determinism_verification.md` – Report
- [ ] `phase_2/data/determinism_statistics.csv` – Statistics
- [ ] PNG plots (deviation per layer, per expert)

---

## #42: Baseline Distributions Analysis

**Issue:** https://github.com/scerruti/moe-expert-prefetching/issues/42  
**Duration:** 3-4 days  
**Depends on:** #40
**Blocks:** #43, #45, #46, #47

### Distribution Computation Setup
- [ ] Create `phase_2/scripts/baseline_distributions.py`
- [ ] Import numpy, scipy.stats, polars, pandas

### Per-Layer, Per-Domain Aggregation
- [ ] Implement `compute_distributions(loader, by_layer=True, by_domain=True)`
- [ ] For each (layer, domain) pair:
  - [ ] Collect all expert probabilities from matching rows
  - [ ] Compute: mean, median, std, min, max
  - [ ] Compute quantiles: q25, q50 (median), q75, q95
  - [ ] Count non-zero probabilities per expert (utilization)
  - [ ] Track number of observations (sample size)

### Token-level Averaging
- [ ] Implement token averaging for repeated tokens:
  - [ ] For each unique (token_id, layer_index) pair:
    - [ ] Collect all probability distributions
    - [ ] Average probabilities across occurrences
    - [ ] Compute standard deviation of averaged values
  - [ ] Store count of how many times token appeared
- [ ] Export token-level averages separately

### Data Organization
- [ ] Organize results as nested dict:
  ```
  distributions[layer_idx][dataset_name] = {
    'mean': array([num_experts]),
    'median': array([num_experts]),
    'std': array([num_experts]),
    'min': array([num_experts]),
    'max': array([num_experts]),
    'q25': array([num_experts]),
    'q75': array([num_experts]),
    'q95': array([num_experts]),
    'utilization': array([num_experts]),  # fraction > 0
    'n_samples': int,
  }
  ```
- [ ] Save to JSON: `baseline_distributions.json` (human-readable)
- [ ] Save to Parquet (optional, for future analysis)

### Distribution Comparison Tables
- [ ] Generate side-by-side comparison tables for each layer:
  - [ ] GSM8K vs MBPP mean probabilities
  - [ ] Difference (MBPP - GSM8K)
  - [ ] Ratio (MBPP / GSM8K) with safeguards for division by zero
- [ ] Export: `distributions_by_layer.csv`
  - [ ] Format: layer_idx, expert_id, mean_gsm8k, mean_mbpp, diff, ratio
  - [ ] Sort by absolute difference (highest divergence first)

### Layer-wise Aggregate Statistics
- [ ] For each layer, compute:
  - [ ] Overall mean probability (uniform would be 1/num_experts)
  - [ ] Probability distribution skewness (are some experts heavily used?)
  - [ ] Distribution kurtosis (concentration)
- [ ] Create summary table: `layer_aggregate_statistics.csv`
  - [ ] Columns: layer_idx, mean_prob_gsm8k, mean_prob_mbpp, skewness_gsm8k, skewness_mbpp, ...

### Validation Checks
- [ ] Verify all probabilities are in [0, 1]
- [ ] Verify expert probabilities sum to 1.0 per token (within 1e-5)
- [ ] Check for NaN/inf values (should be zero)
- [ ] Log any validation failures

### Visualization Preparation
- [ ] Generate distribution plots per layer (for later visualization #49):
  - [ ] Histogram: expert probability distribution GSM8K vs MBPP
  - [ ] Density plot: smooth distribution curves
- [ ] Identify candidate layers for main report
  - [ ] Early layer (e.g., layer 0): expected to be general
  - [ ] Mid layer (e.g., layer 12): expected to show divergence
  - [ ] Late layer (e.g., layer 23): expected to be integrative

### Testing & Documentation
- [ ] Unit tests:
  - [ ] Verify mean computation correctness
  - [ ] Verify probability sums to 1.0
  - [ ] Verify quantile calculations
- [ ] Test on small sample (100 rows)
- [ ] Document mathematical formulas used
- [ ] Add edge cases (single sample, identical values)

### Deliverables
- [ ] `phase_2/scripts/baseline_distributions.py` – Complete implementation
- [ ] `phase_2/data/baseline_distributions.json` – Full statistical results
- [ ] `phase_2/reports/distributions_by_layer.csv` – Comparison tables
- [ ] `phase_2/reports/layer_aggregate_statistics.csv` – Summary statistics
- [ ] Tests pass for statistical correctness

---

## #43: Layer-wise Specialization Summaries

**Issue:** https://github.com/scerruti/moe-expert-prefetching/issues/43  
**Duration:** 2-3 days  
**Depends on:** #42

### Top Expert Ranking
- [ ] Create `phase_2/scripts/layer_summaries.py`
- [ ] For each layer and domain:
  - [ ] Rank experts by mean probability (highest to lowest)
  - [ ] Extract top-10 experts
  - [ ] Record their mean probabilities
  - [ ] Note: top-k in actual routing is 2, but we track broader activation
- [ ] Export: `top_experts_by_layer.csv`
  - [ ] Columns: layer_idx, rank, expert_id, mean_prob_gsm8k, mean_prob_mbpp

### Expert Utilization Analysis
- [ ] Define utilization threshold (e.g., mean probability > 0.001)
- [ ] For each layer and domain:
  - [ ] Count experts above threshold (active experts)
  - [ ] Compute utilization rate = active_experts / total_experts
  - [ ] Identify "dead" experts (never activated)
- [ ] Export: `expert_utilization.csv`
  - [ ] Columns: layer_idx, utilization_gsm8k, utilization_mbpp, dead_experts_count

### Shannon Entropy Computation
- [ ] Implement entropy calculation: $H = -\sum p_i \log(p_i)$
  - [ ] Use natural log (ln)
  - [ ] Handle p_i = 0 gracefully (0 * log(0) → 0)
- [ ] For each layer and domain:
  - [ ] Compute Shannon entropy of expert probability distribution
  - [ ] Normalize to [0, 1] if desired (divide by log(num_experts))
- [ ] Interpretation:
  - [ ] High entropy (≈0.8-1.0): routing is spread across many experts
  - [ ] Low entropy (≈0.1-0.3): routing is concentrated on few experts
- [ ] Export: `entropy_analysis.csv`
  - [ ] Columns: layer_idx, entropy_gsm8k, entropy_mbpp, entropy_diff

### Comparison Tables
- [ ] Generate comparison tables (GSM8K vs MBPP) for each layer
- [ ] Highlight layers with largest divergence in:
  - [ ] Expert utilization
  - [ ] Entropy
  - [ ] Top-expert consistency
- [ ] Export comprehensive summaries: `layer_profiles.csv`

### Per-Dataset Specialization Profiles
- [ ] Create layer profiles for GSM8K:
  - [ ] Layer type (early/general, mid/specialized, late/integration)
  - [ ] Top 5 active experts
  - [ ] Utilization rate
  - [ ] Entropy score
- [ ] Create same for MBPP
- [ ] Compare and note differences

### Visualization Prep
- [ ] Identify plots for later visualization:
  - [ ] Entropy comparison: GSM8K vs MBPP by layer
  - [ ] Expert utilization by layer
  - [ ] Top-expert consistency heatmap

### Statistical Anomalies
- [ ] Identify outliers:
  - [ ] Layers with very high/low entropy
  - [ ] Layers with expert utilization <20% or >80%
  - [ ] Layers with many dead experts
- [ ] Document anomalies and possible explanations

### Testing & Documentation
- [ ] Unit tests for entropy calculation
- [ ] Verify utilization rate is in [0, 1]
- [ ] Test edge cases (all experts equally used, one expert only)
- [ ] Document entropy interpretation

### Deliverables
- [ ] `phase_2/scripts/layer_summaries.py` – Implementation
- [ ] `phase_2/data/layer_summaries.json` – Complete layer profiles
- [ ] `phase_2/reports/layer_profiles.csv` – Human-readable summaries
- [ ] `phase_2/reports/entropy_analysis.csv` – Entropy statistics
- [ ] `phase_2/reports/top_experts_by_layer.csv` – Expert rankings

---

## #44: Shared Token Identification

**Issue:** https://github.com/scerruti/moe-expert-prefetching/issues/44  
**Duration:** 1-2 days  
**Depends on:** #40
**Blocks:** #45

### Vocabulary Overlap Computation
- [ ] Create `phase_2/scripts/token_analysis.py`
- [ ] Extract unique token_id values from each dataset
  - [ ] GSM8K vocabulary
  - [ ] MBPP vocabulary
  - [ ] Get decoded token strings (token_str)
- [ ] Find intersection: shared tokens = vocab_gsm8k ∩ vocab_mbpp
- [ ] Compute overlap statistics:
  - [ ] Total shared tokens count
  - [ ] Overlap percentage: (shared / min(vocab_size)) * 100
  - [ ] Jaccard similarity: shared / union
- [ ] Log vocabulary sizes and overlap details

### Shared Token Inventory
- [ ] For each shared token:
  - [ ] Store token_id, token_str
  - [ ] Count occurrences in GSM8K
  - [ ] Count occurrences in MBPP
  - [ ] Total occurrences across datasets
  - [ ] Frequency rank in each dataset
- [ ] Classify tokens:
  - [ ] Common: appears in both datasets with similar frequency
  - [ ] Domain-biased: appears much more in one dataset
  - [ ] Rare: low frequency in both
- [ ] Export: `shared_tokens.csv`
  - [ ] Columns: token_id, token_str, count_gsm8k, count_mbpp, total_count, frequency_rank_gsm8k, frequency_rank_mbpp, classification

### Stratified Sampling
- [ ] Stratify shared tokens by frequency:
  - [ ] High frequency (top 20%): most common tokens
  - [ ] Medium frequency (middle 60%): moderate frequency
  - [ ] Low frequency (bottom 20%): rare tokens
- [ ] Purpose: ensure analysis covers all frequency ranges
- [ ] Document stratification ratios in report

### Token Quality Checks
- [ ] Identify special tokens:
  - [ ] Whitespace/punctuation (if tokenizer includes these)
  - [ ] Rare/OOV tokens
  - [ ] Control tokens
- [ ] Flag tokens that might need special handling in analysis
- [ ] Document any token exclusions

### Vocabulary Statistics
- [ ] Generate comprehensive vocabulary report:
  - [ ] GSM8K vocab size, unique tokens
  - [ ] MBPP vocab size, unique tokens
  - [ ] Total shared tokens
  - [ ] Vocab overlap percentage
  - [ ] Top-20 most common shared tokens
  - [ ] Bottom-20 rarest shared tokens
- [ ] Export: `vocabulary_statistics.txt` or Markdown

### Visualization Prep
- [ ] Prepare data for token frequency plots (later visualization):
  - [ ] Frequency distribution: shared tokens ranked by count
  - [ ] Scatter plot: frequency in GSM8K vs MBPP
  - [ ] Venn diagram data: overlap visualization

### Testing & Documentation
- [ ] Unit tests:
  - [ ] Verify intersection correctness
  - [ ] Verify frequency counts match Parquet data
  - [ ] Check for duplicate tokens
- [ ] Test on sample data
- [ ] Document token ID mapping (if any tokenizer changes needed)

### Deliverables
- [ ] `phase_2/scripts/token_analysis.py` – Implementation (part 1)
- [ ] `phase_2/data/shared_tokens.csv` – Shared token inventory
- [ ] `phase_2/reports/token_overlap_analysis.md` – Vocabulary analysis report
- [ ] `phase_2/reports/vocabulary_statistics.txt` – Statistics summary

---

## #45: Cross-Domain Token Divergence Analysis

**Issue:** https://github.com/scerruti/moe-expert-prefetching/issues/45  
**Duration:** 3 days  
**Depends on:** #44, #42
**Blocks:** None (feeds into final analysis)

### KL Divergence Computation
- [ ] Extend `phase_2/scripts/token_analysis.py` (part 2)
- [ ] For each shared token and layer:
  - [ ] Extract probability distribution from GSM8K data
  - [ ] Extract probability distribution from MBPP data
  - [ ] Compute KL(GSM8K || MBPP) = sum p_gsm8k * log(p_gsm8k / p_mbpp)
  - [ ] Handle zero probabilities: use small epsilon (1e-10)
  - [ ] Also compute symmetric KL divergence if needed
- [ ] Store results: `token_divergence_per_layer`

### Token-level Divergence Aggregation
- [ ] For each shared token, aggregate KL divergence across layers:
  - [ ] Mean KL divergence across layers
  - [ ] Max KL divergence (which layer most divergent?)
  - [ ] Layer-wise KL ranking (which layers specialize per token?)
- [ ] Compute specialization score:
  - [ ] High KL (> 0.1) = domain-specific token
  - [ ] Low KL (< 0.01) = universal/invariant token
  - [ ] Moderate KL = mixed specialization
- [ ] Export: `token_divergence.csv`
  - [ ] Columns: token_id, token_str, mean_kl, max_kl, max_kl_layer, specialization_score, specialization_type

### Token Ranking
- [ ] Rank shared tokens by KL divergence (highest to lowest)
- [ ] Identify:
  - [ ] Top 20 most domain-specific tokens (high KL)
  - [ ] Top 20 most universal tokens (low KL)
  - [ ] Tokens with interesting per-layer patterns
- [ ] Create ranking tables: `token_divergence_rankings.csv`

### Per-Layer Token Analysis
- [ ] For each layer, identify:
  - [ ] Which shared tokens have highest divergence?
  - [ ] Are certain tokens consistently specialized across layers?
  - [ ] Are there layer-specific token specializations?
- [ ] Create layer-wise rankings: `token_divergence_by_layer.csv`

### Semantic Analysis (Optional)
- [ ] Manually inspect tokens with high divergence:
  - [ ] Are they domain-specific tokens (e.g., "def" for code, "number" for math)?
  - [ ] Are they common tokens with context-dependent routing?
  - [ ] Group high-divergence tokens by semantic category
- [ ] Document findings in analysis report

### Validation Checks
- [ ] Verify all KL divergences are non-negative (should be ≥ 0)
- [ ] Check for NaN/inf values (handle edge cases)
- [ ] Verify probability sums before KL computation
- [ ] Log any computational issues

### Visualization Prep
- [ ] Prepare data for:
  - [ ] Scatter plot: KL divergence vs token frequency
  - [ ] Bar chart: top 20 most/least divergent tokens
  - [ ] Heatmap: token × layer KL divergence

### Statistical Summary
- [ ] Compute aggregate statistics:
  - [ ] Mean KL across all shared tokens
  - [ ] Median, std, min, max KL
  - [ ] Fraction of tokens classified as domain-specific
- [ ] Generate summary report: `token_divergence_summary.txt`

### Testing & Documentation
- [ ] Unit tests:
  - [ ] Verify KL computation correctness (vs. scipy.spatial.distance.entropy)
  - [ ] Edge case: identical distributions (KL = 0)
  - [ ] Edge case: very different distributions
- [ ] Test on subset of tokens
- [ ] Document KL divergence interpretation for non-technical readers

### Deliverables
- [ ] `phase_2/scripts/token_analysis.py` – Complete (part 2)
- [ ] `phase_2/data/token_divergence.csv` – Full divergence results
- [ ] `phase_2/reports/cross_domain_token_analysis.md` – Analysis report
- [ ] `phase_2/reports/token_divergence_rankings.csv` – Token rankings
- [ ] `phase_2/reports/token_divergence_by_layer.csv` – Per-layer analysis

---

## #46: Jensen-Shannon Divergence Computation

**Issue:** https://github.com/scerruti/moe-expert-prefetching/issues/46  
**Duration:** 3-4 days  
**Depends on:** #42
**Blocks:** #48

### Global JSD Computation
- [ ] Create `phase_2/scripts/jsd_analysis.py`
- [ ] Compute domain-averaged expert probability distribution:
  - [ ] Average all expert probabilities from GSM8K across all tokens/layers
  - [ ] Result: single distribution [num_experts]
  - [ ] Similarly for MBPP
- [ ] Compute JS divergence between the two distributions:
  - [ ] Use scipy.spatial.distance.jensenshannon or implement manually
  - [ ] Formula: JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M), M = (P+Q)/2
- [ ] Result: single global JSD value
- [ ] Interpretation: global domain difference in routing

### Per-Layer JSD Computation
- [ ] For each layer independently:
  - [ ] Compute domain-averaged distribution for that layer only
  - [ ] Average across all tokens in that layer (per domain)
  - [ ] Compute JSD between GSM8K and MBPP distributions
  - [ ] Result: array[num_layers] of JSD values
- [ ] Identify which layers have highest JSD (most specialization)
- [ ] Identify which layers have lowest JSD (most general)

### Bootstrap Confidence Intervals
- [ ] Implement bootstrap resampling (n=1000):
  - [ ] For each bootstrap iteration:
    - [ ] Resample tokens with replacement (per layer, per domain)
    - [ ] Compute JSD on resampled data
  - [ ] Collect bootstrap JSD values
  - [ ] Compute 95% confidence interval: percentiles [2.5, 97.5]
- [ ] Per-layer bootstrap intervals: result is array[num_layers, 2]
- [ ] Global bootstrap interval: single interval
- [ ] Export: `jsd_bootstrap_ci.csv`
  - [ ] Columns: layer_idx, jsd_mean, ci_lower_95, ci_upper_95

### Per-Expert JSD Computation
- [ ] For each expert across all layers:
  - [ ] Extract activation pattern: prob distribution across layers × domains
  - [ ] Compute JSD(GSM8K || MBPP) for this expert's activation pattern
  - [ ] Result: array[num_experts] of expert-level JSD values
- [ ] Rank experts by JSD:
  - [ ] High JSD experts: domain-specific
  - [ ] Low JSD experts: universal experts
- [ ] Export: `jsd_per_expert.csv`
  - [ ] Columns: expert_id, jsd_value, specialization_type

### Statistical Significance Testing
- [ ] For each per-layer JSD, compute p-value:
  - [ ] Null hypothesis: JSD = 0 (no domain difference)
  - [ ] Alternative: JSD > 0 (significant difference)
  - [ ] Use bootstrap p-value: fraction of bootstrap samples ≤ 0 (should be ~0 for real data)
  - [ ] Or use permutation test: shuffle domain labels, compute JSD
- [ ] Correct for multiple comparisons (Bonferroni):
  - [ ] Adjust significance level: 0.05 / num_layers
- [ ] Export: `jsd_significance.csv`
  - [ ] Columns: layer_idx, jsd_value, p_value, significant (yes/no), ci_lower, ci_upper

### Interpretation & Classification
- [ ] Classify layers by JSD magnitude:
  - [ ] Highly specialized: JSD > 0.15
  - [ ] Moderately specialized: JSD 0.05-0.15
  - [ ] General: JSD < 0.05
- [ ] Create classification table: `layer_jsd_classification.csv`
- [ ] Identify candidate layers for Phase 3 predictor focus

### Validation Checks
- [ ] Verify JSD is symmetric (or document if using asymmetric KL)
- [ ] Verify JSD values are in [0, log(2)] for uniform distributions
- [ ] Check for NaN/inf (handle edge cases)
- [ ] Log all computational details for reproducibility

### Visualization Prep
- [ ] Prepare data for:
  - [ ] Line plot: JSD by layer with confidence intervals
  - [ ] Heatmap: expert × layer JSD values
  - [ ] Bar chart: top 20 most/least specialized experts

### Results Export
- [ ] JSON summary: `jsd_analysis.json`
  ```json
  {
    "global_jsd": 0.125,
    "global_jsd_ci": [0.120, 0.130],
    "per_layer_jsd": [...],
    "per_layer_ci": [...],
    "per_expert_jsd": [...],
    "methodology": "..."
  }
  ```
- [ ] CSV results: `jsd_significance.csv`, `jsd_per_expert.csv`

### Documentation & Report
- [ ] Write detailed methodology section:
  - [ ] JSD formula
  - [ ] Bootstrap procedure
  - [ ] Interpretation guidelines
- [ ] Create findings summary: `jsd_findings.md`
  - [ ] Which layers are most specialized?
  - [ ] Which experts are domain-specific?
  - [ ] Statistical confidence in findings

### Testing & Documentation
- [ ] Unit tests:
  - [ ] Verify JSD formula correctness (vs. scipy)
  - [ ] Verify bootstrap produces reasonable distributions
  - [ ] Edge case: identical distributions (JSD = 0)
  - [ ] Edge case: completely different distributions
- [ ] Test on small sample (5 layers, 100 tokens)
- [ ] Document all assumptions and limitations

### Deliverables
- [ ] `phase_2/scripts/jsd_analysis.py` – Complete implementation
- [ ] `phase_2/data/jsd_analysis.json` – Full JSD results
- [ ] `phase_2/reports/jsd_significance.csv` – Statistical results
- [ ] `phase_2/reports/jsd_per_expert.csv` – Expert rankings
- [ ] `phase_2/reports/jsd_findings.md` – Interpretation and findings

---

## #47: Expert-level Divergence & Specialization

**Issue:** https://github.com/scerruti/moe-expert-prefetching/issues/47  
**Duration:** 2-3 days  
**Depends on:** #42

### Expert Activation Analysis
- [ ] Create `phase_2/scripts/expert_profiling.py`
- [ ] For each expert:
  - [ ] Count how many tokens activate this expert as top-2 (GSM8K)
  - [ ] Count how many tokens activate this expert as top-2 (MBPP)
  - [ ] Compute activation frequency: activation_count / total_tokens
  - [ ] Separate by layer if needed
- [ ] Export: `expert_activation_frequency.csv`
  - [ ] Columns: expert_id, activation_freq_gsm8k, activation_freq_mbpp, ratio

### Expert Probability Profile
- [ ] For each expert, compute across all layers:
  - [ ] Mean probability in GSM8K (average over all occurrences)
  - [ ] Mean probability in MBPP
  - [ ] Std of probabilities (variability)
  - [ ] Max probability observed
  - [ ] Min probability observed
- [ ] Store as: `expert_prob_profiles.csv`

### Cross-Domain Divergence Per Expert
- [ ] For each expert:
  - [ ] Extract probability distribution across layers (GSM8K)
  - [ ] Extract probability distribution across layers (MBPP)
  - [ ] Compute JS divergence between the two distributions
  - [ ] Interpretation: high = expert is domain-specific, low = universal
- [ ] Export: `expert_jsd_profiles.csv`
  - [ ] Columns: expert_id, jsd_value, specialization_score

### Expert Classification
- [ ] Classify each expert:
  - [ ] Universal: low divergence, activated in both domains
  - [ ] Math-specific: high divergence, more active in GSM8K
  - [ ] Code-specific: high divergence, more active in MBPP
  - [ ] Underutilized: low activation frequency in both domains
- [ ] Create classification table: `expert_classification.csv`
  - [ ] Columns: expert_id, classification, justification, jsd_value, activation_freq

### Expert Clustering
- [ ] Perform k-means clustering on expert profiles:
  - [ ] Features: [activation_freq_gsm8k, activation_freq_mbpp, jsd_value, mean_prob]
  - [ ] Normalize features to [0, 1]
  - [ ] Try k = 3-5 clusters (universal, math-specific, code-specific, etc.)
  - [ ] Use silhouette analysis to find optimal k
- [ ] Interpret cluster meanings:
  - [ ] Cluster 1: universal experts (low specialization)
  - [ ] Cluster 2: math-focused experts
  - [ ] Cluster 3: code-focused experts
  - [ ] Etc.
- [ ] Assign experts to clusters: `expert_clustering.csv`
  - [ ] Columns: expert_id, cluster_id, cluster_name

### Expert Patterns
- [ ] Analyze patterns within clusters:
  - [ ] Average characteristics per cluster
  - [ ] Layer distribution per cluster (do certain clusters activate certain layers?)
  - [ ] Token interaction patterns
- [ ] Document findings: `expert_clustering_report.md`

### Top/Bottom Experts Ranking
- [ ] Rank experts by different criteria:
  - [ ] By activation frequency (most/least active)
  - [ ] By specialization (most/least domain-specific)
  - [ ] By consistency (most/least variable)
- [ ] Create ranking tables for reporting

### Validation & Quality Checks
- [ ] Verify all experts are represented (no missing experts)
- [ ] Check for NaN/inf in profiles
- [ ] Verify clustering assignments are sensible
- [ ] Log any computational issues

### Visualization Prep
- [ ] Prepare data for:
  - [ ] Scatter plot: activation frequency vs JSD (2D embedding of experts)
  - [ ] Bubble chart: add cluster size as bubble size
  - [ ] Heatmap: expert × layer activation patterns
  - [ ] Radar chart: expert profile (multi-dimensional)

### Integration with Other Analyses
- [ ] Link expert profiles to:
  - [ ] Shared token analysis (#45): which experts activate on domain-specific tokens?
  - [ ] Layer specialization (#48): which experts are in specialized layers?
- [ ] Document interdependencies

### Testing & Documentation
- [ ] Unit tests:
  - [ ] Verify activation frequency computation
  - [ ] Verify clustering produces valid partitions
  - [ ] Test edge cases (all experts equal, one expert dominates)
- [ ] Test on subset of experts
- [ ] Document expert numbering (make sure IDs match Parquet)

### Deliverables
- [ ] `phase_2/scripts/expert_profiling.py` – Complete implementation
- [ ] `phase_2/data/expert_profiles.csv` – Comprehensive expert data
- [ ] `phase_2/data/expert_clustering.csv` – Cluster assignments
- [ ] `phase_2/reports/expert_specialization.md` – Full analysis report
- [ ] `phase_2/reports/expert_classification.csv` – Classification results

---

## #48: Layer Specialization Indexing

**Issue:** https://github.com/scerruti/moe-expert-prefetching/issues/48  
**Duration:** 2-3 days  
**Depends on:** #41 (Determinism), #46 (JSD)

### Specialization Index Computation
- [ ] Create `phase_2/scripts/layer_specialization.py`
- [ ] Compute index for each layer:
  - [ ] Component 1 (JSD-based): Use per-layer JSD from #46 (weight: 0.5)
  - [ ] Component 2 (utilization divergence): Difference in expert utilization between domains (weight: 0.25)
  - [ ] Component 3 (top-expert consistency): Jaccard similarity of top-5 experts between domains (weight: 0.25)
    - [ ] High consistency (Jaccard ≈ 1) → low contribution to specialization
    - [ ] Low consistency (Jaccard < 0.5) → high contribution to specialization
- [ ] Formula: $I_{spec}(l) = 0.5 \cdot JSD + 0.25 \cdot |util_{gsm8k} - util_{mbpp}| + 0.25 \cdot (1 - Jaccard)$
- [ ] Normalize to [0, 100] scale for interpretation
- [ ] Export: `layer_specialization_index.csv`
  - [ ] Columns: layer_idx, specialization_index, component_jsd, component_utilization, component_consistency

### Layer Classification
- [ ] Classify layers into categories:
  - [ ] Early layers (0-7): General linguistic processing (low specialization expected)
  - [ ] Mid layers (8-19): Domain-specific expert allocation (high specialization expected)
  - [ ] Late layers (20+): Integration and refinement (moderate specialization)
- [ ] Compute thresholds based on distribution:
  - [ ] If specialization index is high in early layers → unexpected, flag
  - [ ] If specialization index is low in mid layers → unexpected, flag
  - [ ] Document any anomalies
- [ ] Create classification table: `layer_classification.csv`
  - [ ] Columns: layer_idx, specialization_index, classification, confidence

### Layer Grouping
- [ ] Group layers with similar specialization patterns:
  - [ ] Cluster 1: Highly general layers
  - [ ] Cluster 2: Moderately specialized layers
  - [ ] Cluster 3: Highly specialized layers
- [ ] Use clustering algorithm (k-means with k=3)
- [ ] Document groupings: `layer_grouping_analysis.md`

### Anomaly Detection
- [ ] Identify layers with unexpected specialization:
  - [ ] Early layer with high specialization (unusual)
  - [ ] Mid layer with low specialization (unusual)
  - [ ] Layer with very high/low divergence in other measures
- [ ] Investigate anomalies:
  - [ ] Could indicate interesting structure (document for Phase 3)
  - [ ] Could indicate data quality issues (investigate)
- [ ] Export anomaly report: `layer_anomalies.md`

### Phase 3 Recommendations
- [ ] Based on layer specialization, recommend which layers to focus on for Phase 3 predictor training:
  - [ ] High specialization layers: ROI for predictor likely higher
  - [ ] Low specialization layers: routing may be too complex to predict
  - [ ] Moderate specialization layers: sweet spot for predictor training?
- [ ] Rank layers by recommendation priority
- [ ] Create prioritization table: `phase_3_layer_recommendations.csv`
  - [ ] Columns: layer_idx, specialization_index, recommended_priority, reasoning

### Cross-Model Comparison (if DeepSeek available)
- [ ] If DeepSeek Phase 1 data available:
  - [ ] Compute specialization index for DeepSeek
  - [ ] Compare layer-wise specialization patterns with Qwen
  - [ ] Identify consistent patterns across models
  - [ ] Document differences and explanations
- [ ] Create comparison table: `qwen_vs_deepseek_specialization.csv`

### Visualization Prep
- [ ] Prepare data for:
  - [ ] Line plot: specialization index by layer with classification colors
  - [ ] Stacked bar chart: components of specialization index by layer
  - [ ] Heatmap: specialization components by layer
  - [ ] Scatter plot: model comparison (Qwen vs DeepSeek if available)

### Report Generation
- [ ] Create comprehensive report: `layer_classification.md`
  - [ ] Executive summary: key findings
  - [ ] Methods: specialization index formula, classification logic
  - [ ] Results: per-layer index scores, classifications
  - [ ] Anomalies: unexpected patterns and explanations
  - [ ] Phase 3 recommendations: prioritized layer list
  - [ ] Appendices: detailed tables, formulas

### Validation & Quality Checks
- [ ] Verify specialization index is in [0, 100]
- [ ] Check for NaN/inf values
- [ ] Verify classification makes intuitive sense
- [ ] Spot-check a few layers manually
- [ ] Log all computational steps

### Testing & Documentation
- [ ] Unit tests:
  - [ ] Verify index computation correctness
  - [ ] Verify classification logic
  - [ ] Test edge cases (all components = 0, all = max)
- [ ] Test on subset of layers
- [ ] Document assumptions about layer numbering
- [ ] Document formula and rationale for weights

### Deliverables
- [ ] `phase_2/scripts/layer_specialization.py` – Complete implementation
- [ ] `phase_2/reports/layer_specialization_index.csv` – Index scores
- [ ] `phase_2/reports/layer_classification.csv` – Classification results
- [ ] `phase_2/reports/layer_classification.md` – Detailed report
- [ ] `phase_2/reports/phase_3_layer_recommendations.csv` – Recommendations
- [ ] `phase_2/reports/layer_anomalies.md` – Anomaly analysis (if any)

---

## #49: Visualizations & Chart Publishing

**Issue:** https://github.com/scerruti/moe-expert-prefetching/issues/49  
**Duration:** 3-4 days  
**Depends on:** All prior analyses (#41-48)

### Visualization Setup
- [ ] Create `phase_2/scripts/visualization.py`
- [ ] Import: matplotlib, seaborn, plotly
- [ ] Set up consistent style:
  - [ ] Color palette: accessible, publication-ready
  - [ ] Font: readable (10-12pt for legends, 14-16pt for labels)
  - [ ] Figure size: optimized for paper (8.5" × 11" considerations)
  - [ ] DPI: 300 for PNG (publication quality)
- [ ] Create helper functions for consistent formatting

### Distribution Plots (3+ figures)
- [ ] **Figure 1: Distribution Comparison (Selected Layers)**
  - [ ] Show GSM8K vs MBPP probability distributions for 3 layers (early, mid, late)
  - [ ] Format: histogram or density plot, overlaid
  - [ ] Legend: clear labeling of domains
  - [ ] Export: PNG, SVG, PDF
- [ ] **Figure 2: Expert Activation Heatmap (per Layer)**
  - [ ] X-axis: expert ID (all 64 or top 32)
  - [ ] Y-axis: layer index
  - [ ] Color: mean probability value
  - [ ] Create two versions: GSM8K and MBPP side-by-side
  - [ ] Export: PNG, SVG
- [ ] **Figure 3: Layer Entropy Comparison**
  - [ ] X-axis: layer index
  - [ ] Y-axis: Shannon entropy
  - [ ] Two lines: GSM8K and MBPP
  - [ ] Shaded region: confidence intervals (from #48)
  - [ ] Export: PNG, SVG

### Divergence & Specialization Plots (4+ figures)
- [ ] **Figure 4: Per-Layer JSD with Confidence Intervals**
  - [ ] X-axis: layer index
  - [ ] Y-axis: JSD value
  - [ ] Points: layer JSD with error bars (bootstrap CI)
  - [ ] Color: specialization level (low/medium/high)
  - [ ] Background: layer classification zones (early/mid/late)
  - [ ] Export: PNG, SVG
- [ ] **Figure 5: JSD Heatmap (Layer × Expert)**
  - [ ] X-axis: expert ID
  - [ ] Y-axis: layer index
  - [ ] Color: per-expert JSD value (from #47)
  - [ ] Annotations: top divergence values labeled
  - [ ] Export: PNG, SVG
- [ ] **Figure 6: Specialization Index by Layer**
  - [ ] Bar chart: specialization index per layer
  - [ ] Color: classification (early/mid/late)
  - [ ] Y-axis: specialization index [0-100]
  - [ ] Annotations: anomalies flagged
  - [ ] Export: PNG, SVG
- [ ] **Figure 7: Specialization Components Stacked Bar**
  - [ ] X-axis: layer index
  - [ ] Y-axis: specialization index [0-100]
  - [ ] Stacked bars: contributions from JSD, utilization, consistency
  - [ ] Legend: component labels
  - [ ] Export: PNG, SVG

### Token Analysis Plots (3+ figures)
- [ ] **Figure 8: Shared Token Divergence Scatter**
  - [ ] X-axis: token frequency
  - [ ] Y-axis: KL divergence
  - [ ] Points: individual shared tokens
  - [ ] Color: specialization type (domain-specific / universal / mixed)
  - [ ] Size: optional (layer range or activation count)
  - [ ] Export: PNG, SVG
- [ ] **Figure 9: Top Domain-Specific Tokens**
  - [ ] Bar chart: top 20 shared tokens by KL divergence
  - [ ] Color: GSM8K vs MBPP activation color-coded
  - [ ] Y-axis: KL divergence value
  - [ ] X-axis: token strings (or indices with legend)
  - [ ] Export: PNG, SVG
- [ ] **Figure 10: Token Frequency vs Divergence**
  - [ ] Scatter plot with different point sizes/colors
  - [ ] Identify clusters: frequent universal tokens, rare specialized tokens, etc.
  - [ ] Export: PNG, SVG

### Expert Analysis Plots (2+ figures)
- [ ] **Figure 11: Expert Clustering Scatter (2D Projection)**
  - [ ] X-axis: activation frequency (average across domains)
  - [ ] Y-axis: JSD divergence
  - [ ] Points: individual experts, colored by cluster
  - [ ] Legend: cluster names (universal / math-specific / code-specific)
  - [ ] Annotations: label interesting/notable experts
  - [ ] Export: PNG, SVG
- [ ] **Figure 12: Expert Activation Heatmap (Top Experts)**
  - [ ] X-axis: GSM8K vs MBPP (two columns)
  - [ ] Y-axis: top 20 experts (by JSD)
  - [ ] Color: activation frequency or mean probability
  - [ ] Annotations: cluster labels
  - [ ] Export: PNG, SVG

### Composite/Summary Figures (2+ figures)
- [ ] **Figure 13: Layer Specialization Landscape**
  - [ ] 3-panel figure:
    - [ ] Panel A: Specialization index by layer (line plot with zones)
    - [ ] Panel B: Specialization components breakdown (stacked bar)
    - [ ] Panel C: JSD confidence intervals (line with error bars)
  - [ ] Unified title and caption
  - [ ] Export: PNG, SVG (suitable for publication)
- [ ] **Figure 14: Summary Statistics Table as Figure**
  - [ ] Create a publication-ready table image:
    - [ ] Model stats (Qwen1.5-MoE-A2.7B)
    - [ ] Dataset stats (GSM8K, MBPP, shared tokens)
    - [ ] Key findings (global JSD, top specialized layers, etc.)
  - [ ] Export: PNG

### Interactive Dashboard
- [ ] Create HTML dashboard with plotly:
  - [ ] Tab 1: Distribution analysis
    - [ ] Dropdown: select layer to visualize
    - [ ] Plot: distribution comparison GSM8K vs MBPP
    - [ ] Statistics: display summary stats for selected layer
  - [ ] Tab 2: Divergence analysis
    - [ ] Plot: JSD by layer (interactive line chart)
    - [ ] Click to see per-expert divergence for layer
    - [ ] Dropdown: select divergence metric
  - [ ] Tab 3: Specialization
    - [ ] Plot: specialization index by layer
    - [ ] Hover: show component breakdown
    - [ ] Click: drill into layer details
  - [ ] Tab 4: Tokens
    - [ ] Plot: scatter of shared tokens
    - [ ] Hover: show token string, divergence, frequency
    - [ ] Filter: by frequency range, specialization type
  - [ ] Tab 5: Experts
    - [ ] Plot: expert clustering scatter
    - [ ] Hover: show expert ID, cluster, characteristics
    - [ ] Filter: by cluster, activation range
  - [ ] Export button on each plot (PNG)
  - [ ] Summary statistics card (showing key metrics)
- [ ] Save: `phase_2/reports/dashboard.html`
- [ ] Test: verify all plots interactive and responsive

### Figure Quality & Consistency
- [ ] All figures use consistent color scheme:
  - [ ] GSM8K: blue (or designated color)
  - [ ] MBPP: orange/red (or designated color)
  - [ ] Specialization levels: green (low) → yellow → red (high)
- [ ] All figures have:
  - [ ] Clear titles
  - [ ] Axis labels with units
  - [ ] Legends (positioned appropriately)
  - [ ] Captions (5-10 sentences explaining findings)
  - [ ] Source data notation (e.g., "Based on Phase 1 analysis")
- [ ] All text readable (font size ≥ 10pt in final output)
- [ ] No overlapping elements or clutter
- [ ] Colorblind-accessible palette (test with simulator)

### Export & Archiving
- [ ] Create directory structure:
  ```
  phase_2/reports/figures/
  ├── distribution_comparison_layer_*.png
  ├── expert_activation_heatmap_gsm8k.png
  ├── expert_activation_heatmap_mbpp.png
  ├── layer_entropy_comparison.png
  ├── per_layer_jsd_with_ci.png
  ├── jsd_heatmap_layer_expert.png
  ├── specialization_index_by_layer.png
  ├── specialization_components_stacked.png
  ├── shared_token_divergence_scatter.png
  ├── top_domain_specific_tokens.png
  ├── token_frequency_vs_divergence.png
  ├── expert_clustering_scatter.png
  ├── expert_activation_heatmap_top.png
  ├── layer_specialization_landscape_composite.png
  ├── summary_statistics_table.png
  ├── *_vector.svg (vector versions)
  └── README.txt (figure descriptions)
  ```
- [ ] Batch convert PNG to PDF (if needed for print)
- [ ] Generate figure index: `figures_README.md`
  - [ ] List of all figures
  - [ ] Captions and descriptions
  - [ ] Which section of report each figure appears in

### Visualization Testing & Validation
- [ ] Check all figures render correctly (no missing data)
- [ ] Verify color palettes are accessible (colorblind-friendly)
- [ ] Proofread all labels, titles, captions
- [ ] Ensure consistent figure numbering and references
- [ ] Spot-check data in plots against raw CSV files
- [ ] Test dashboard in different browsers (Chrome, Firefox, Safari)
- [ ] Test dashboard on mobile (ensure responsive)

### Documentation
- [ ] Document visualization code:
  - [ ] High-level functions for each figure type
  - [ ] Parameters and customization options
  - [ ] Color scheme definitions
  - [ ] Font and DPI settings
- [ ] Create reproducibility guide: `visualization_reproducibility.md`
  - [ ] How to regenerate all figures
  - [ ] Required data files and formats
  - [ ] Dependencies and versions
  - [ ] Expected output sizes and quality

### Deliverables
- [ ] `phase_2/scripts/visualization.py` – Complete implementation
- [ ] `phase_2/reports/figures/` (12+ PNG/SVG figures, high quality)
- [ ] `phase_2/reports/dashboard.html` – Interactive dashboard
- [ ] `phase_2/reports/figures_README.md` – Figure index and captions
- [ ] `visualization_reproducibility.md` – How to regenerate plots

---

## #50: Comprehensive Report & Documentation

**Issue:** https://github.com/scerruti/moe-expert-prefetching/issues/50  
**Duration:** 2-3 days  
**Depends on:** #49 (Visualizations)

### Report Structure & Outline
- [ ] Create report skeleton: `phase_2/reports/PHASE_2_ANALYSIS_REPORT.md`
- [ ] Outline sections:
  - [ ] Title page & metadata
  - [ ] Executive summary
  - [ ] Table of contents
  - [ ] Introduction & motivation
  - [ ] Methods & data
  - [ ] Results
  - [ ] Discussion & implications
  - [ ] Recommendations for Phase 3
  - [ ] Appendices

### Executive Summary (1 page max)
- [ ] Key findings (3-5 bullet points):
  - [ ] Global domain divergence (JSD value)
  - [ ] Most specialized layers and experts
  - [ ] Shared token patterns
  - [ ] Implications for speculative prefetching
  - [ ] Confidence in findings
- [ ] Actionable recommendations:
  - [ ] Which layers to focus on in Phase 3
  - [ ] Which experts likely benefit from prediction
  - [ ] Token-level prioritization if any

### Introduction (2-3 pages)
- [ ] Context: MoE models and expert routing
- [ ] Motivation: why domain-specific routing matters for prefetching
- [ ] Research questions:
  - [ ] Q1: Is expert routing deterministic?
  - [ ] Q2: Do different domains (math vs code) activate different experts?
  - [ ] Q3: Which layers specialize vs generalize?
  - [ ] Q4: Which tokens are universally routed?
- [ ] Contributions of Phase 2 analysis

### Methods (2 pages)
- [ ] Data sources: Phase 1 Parquet files (GSM8K, MBPP, Qwen1.5-MoE-A2.7B)
- [ ] Analysis components:
  - [ ] Determinism verification (section referencing #41)
  - [ ] Baseline distributions (section referencing #42)
  - [ ] Token analysis (section referencing #44, #45)
  - [ ] Divergence metrics (section referencing #46)
  - [ ] Specialization indexing (section referencing #48)
- [ ] Statistical methods:
  - [ ] KL divergence
  - [ ] Jensen-Shannon divergence
  - [ ] Bootstrap confidence intervals
  - [ ] K-means clustering
- [ ] Software: Python, scipy, polars, matplotlib, plotly

### Results Section

#### Determinism Verification (1 page + Figure)
- [ ] Summary: routing is/isn't bitwise deterministic across runs
- [ ] Key numbers: max deviation, fraction passing criterion
- [ ] Interpretation: safe to proceed / concerns exist
- [ ] Embed Figure 1 (determinism plot)
- [ ] Reference issue #41

#### Domain Distribution Analysis (2 pages + Figures)
- [ ] Overview: how expert activations differ between GSM8K and MBPP
- [ ] Layer-wise patterns:
  - [ ] Early layers: mostly general (low JSD)
  - [ ] Mid layers: most specialized (high JSD)
  - [ ] Late layers: integration phase (moderate JSD)
- [ ] Key statistics:
  - [ ] Global JSD value and confidence interval
  - [ ] Per-layer JSD range
  - [ ] Layer classification breakdown (how many general/specialized/integrative)
- [ ] Embed Figures: JSD by layer (Figure 4), specialization index (Figure 6)
- [ ] Reference issues #42, #43, #46, #48

#### Expert Characterization (1.5 pages + Figures)
- [ ] Expert population breakdown:
  - [ ] Universal experts: always active
  - [ ] Domain-specific experts: math-only or code-only
  - [ ] Underutilized experts: rarely active
- [ ] Clustering results:
  - [ ] How many clusters identified
  - [ ] Cluster characteristics
  - [ ] Cluster sizes
- [ ] Notable experts:
  - [ ] Most universal experts (best for all domains)
  - [ ] Most specialized experts (high-value targets for prediction)
- [ ] Embed Figures: expert clustering (Figure 11), expert activation (Figure 12)
- [ ] Reference issue #47

#### Shared Token Analysis (1.5 pages + Figures)
- [ ] Vocabulary overlap:
  - [ ] Total shared tokens
  - [ ] Overlap percentage
  - [ ] Token frequency distribution
- [ ] Token specialization patterns:
  - [ ] Domain-specific tokens (high KL divergence)
    - [ ] Example: coding keywords, math operators
  - [ ] Universal tokens (low KL divergence)
    - [ ] Example: common words, punctuation
  - [ ] Mixed tokens (moderate divergence)
- [ ] Per-layer token specialization:
  - [ ] Which layers show token-level domain effects
  - [ ] Interaction patterns
- [ ] Embed Figures: token divergence scatter (Figure 8), top tokens (Figure 9)
- [ ] Reference issues #44, #45

#### Layer Specialization Profile (1.5 pages + Figures)
- [ ] Specialization index overview:
  - [ ] Formula and components
  - [ ] Scale and interpretation
- [ ] Layer classification:
  - [ ] Early layers (0-7): general linguistic processing
  - [ ] Mid layers (8-19): domain-specific specialization hotspot
  - [ ] Late layers (20+): integration and refinement
- [ ] Anomalies and interesting patterns:
  - [ ] Any unexpected specialization in early/late layers
  - [ ] Layers with high/low consistency with other metrics
- [ ] Embed Figures: specialization landscape (Figure 13), components breakdown
- [ ] Reference issue #48

#### Cross-Model Insights (if DeepSeek data available)
- [ ] Comparison with DeepSeek-V2-Lite (if available)
  - [ ] Similar specialization patterns?
  - [ ] Differences in layer-wise divergence
  - [ ] Implications for Phase 3 generalizability
- [ ] If not available: note as future work

### Discussion (2 pages)
- [ ] Interpretation of findings:
  - [ ] Why do we see these specialization patterns?
  - [ ] What does it mean for expert prefetching?
  - [ ] How predictable is expert routing?
- [ ] Limitations:
  - [ ] Determinism verified only on sample
  - [ ] Analysis based on prefill phase only
  - [ ] Single model (Qwen) for now
- [ ] Implications for Phase 3 predictor design:
  - [ ] Which layers to prioritize for training
  - [ ] Expected prediction difficulty by layer
  - [ ] Potential predictor accuracy bounds
  - [ ] Token-level prediction strategies
- [ ] Broader impact:
  - [ ] Relevance to other MoE models
  - [ ] Potential for hardware optimization

### Recommendations for Phase 3 (1 page)
- [ ] Layer prioritization:
  - [ ] Top 5 layers to focus on for predictor training (from #48 recommendations)
  - [ ] Layers to skip or deprioritize
- [ ] Architectural considerations:
  - [ ] Should predictor be per-layer or shared?
  - [ ] Input features (tokens, positions, hidden states)?
  - [ ] Output: predict top-2 or top-(k+m) experts?
- [ ] Expected outcome:
  - [ ] Based on specialization index, estimated prediction accuracy
  - [ ] Potential speedup from prefetching
- [ ] Risk factors:
  - [ ] Layers or tokens that may be hard to predict
  - [ ] Robustness concerns (generalization to new prompts)

### Appendices (3-5 pages)
- [ ] Appendix A: Detailed statistics tables
  - [ ] Per-layer statistics (mean, std, JSD, specialization index)
  - [ ] Per-expert profiles (top 20 universal, top 20 specialized)
  - [ ] Token divergence rankings (top 20 most/least divergent)
- [ ] Appendix B: Mathematical formulas
  - [ ] JSD formula with explanation
  - [ ] Specialization index formula with weights
  - [ ] Confidence interval computation
- [ ] Appendix C: Methodological details
  - [ ] Bootstrap procedure (n=1000, random seed)
  - [ ] Clustering algorithm (k-means, k=3-5)
  - [ ] Thresholds and cutoffs (why these values?)
- [ ] Appendix D: Data availability
  - [ ] Phase 1 outputs used (Parquet file names, sizes)
  - [ ] Reproducibility: how to regenerate this analysis
- [ ] Appendix E: Figures gallery
  - [ ] All 12+ figures with detailed captions
  - [ ] High-resolution versions available

### Report Formatting & Style
- [ ] Use Markdown formatting for editability
- [ ] Consistent heading hierarchy
- [ ] Numbered sections and cross-references
- [ ] References to GitHub issues (#40, #41, etc.)
- [ ] Figure captions below figures (not above)
- [ ] Table captions above tables (standard)
- [ ] Numbered equations with explanations
- [ ] Bullet points for lists (not numbered unless sequence matters)
- [ ] Bold for key terms on first mention

### PDF Export
- [ ] Use pandoc to convert Markdown → PDF:
  ```bash
  pandoc PHASE_2_ANALYSIS_REPORT.md \
    --template eisvogel \
    --from markdown \
    --to pdf \
    -o PHASE_2_ANALYSIS_REPORT.pdf
  ```
- [ ] Configure pandoc template:
  - [ ] Custom title page
  - [ ] Table of contents
  - [ ] Page numbers
  - [ ] Margins and spacing
  - [ ] Figure placement
- [ ] Review PDF for:
  - [ ] Correct page breaks
  - [ ] Figure placement and sizing
  - [ ] Table formatting
  - [ ] Cross-references
  - [ ] No orphaned text

### Documentation Suite
- [ ] Create `phase_2/README.md`:
  - [ ] Overview of Phase 2 analysis
  - [ ] Key findings (bullet list)
  - [ ] Files and directory structure
  - [ ] How to read the report
  - [ ] Links to GitHub issues
  - [ ] Instructions for reproducing analysis
- [ ] Update main project `/phase_2/README.md` (brief version)
- [ ] Create `phase_2/REPRODUCIBILITY.md`:
  - [ ] Step-by-step guide to regenerate entire analysis
  - [ ] Software dependencies and versions
  - [ ] Data requirements (which Phase 1 Parquet files)
  - [ ] Expected runtime and resources
  - [ ] Troubleshooting section

### Integration & Cross-references
- [ ] Link all issues (#40-#49) from report introduction
- [ ] Reference figures from appropriate sections
- [ ] Cross-link analysis scripts
- [ ] Ensure consistency with ARCHITECTURE.md and GITHUB_ISSUES.md
- [ ] Update SYSTEM_DESIGN.md if any changes to planned analysis

### Quality Assurance
- [ ] Proofread entire report:
  - [ ] Grammar and spelling
  - [ ] Technical accuracy
  - [ ] Consistency of terminology
  - [ ] Numbering and cross-references
- [ ] Verify all figure captions are complete and accurate
- [ ] Verify all tables have titles and units
- [ ] Check all citations and references
- [ ] Review numbers and statistics for typos
- [ ] Ensure consistent notation (what does "e" mean? expert? always explain)

### Final Review
- [ ] Read report as if unfamiliar with analysis (spot-check understanding)
- [ ] Verify key takeaways are clear
- [ ] Confirm recommendations are actionable for Phase 3
- [ ] Check that findings are well-supported by data
- [ ] Ensure report is publication-ready quality

### Deliverables
- [ ] `phase_2/reports/PHASE_2_ANALYSIS_REPORT.md` – Main report (Markdown)
- [ ] `phase_2/reports/PHASE_2_ANALYSIS_REPORT.pdf` – PDF export
- [ ] `phase_2/README.md` – Directory overview and key findings
- [ ] `phase_2/REPRODUCIBILITY.md` – Reproducibility guide
- [ ] `phase_2/figures_README.md` – Figure index and captions (if separate)
- [ ] All supporting data files and scripts from #40-#49

---

## Cross-Component Dependencies & Timing

```
Week 1 (Oct 7-13):
├─ #40: Data Loading (2-3 days) ← START HERE
│   ├─ then #41: Determinism (2-3 days, parallel start #42, #44)
│   ├─ #42: Baseline Dist (3-4 days, parallel)
│   └─ #44: Shared Tokens (1-2 days, parallel)

Week 2 (Oct 14-20):
├─ #43: Layer Summaries (2-3 days) ← depends on #42
├─ #45: Token Divergence (3 days) ← depends on #44, #42
├─ #46: JSD Analysis (3-4 days) ← depends on #42, parallel
└─ #47: Expert Divergence (2-3 days) ← depends on #42, parallel

Week 3 (Oct 21-27):
├─ #48: Layer Specialization (2-3 days) ← depends on #41, #46
├─ #49: Visualizations (3-4 days) ← depends on all #41-48
└─ (can start once most analyses done)

Week 4 (Oct 28-Nov 4):
└─ #50: Final Report (2-3 days) ← depends on #49
```

---

## Acceptance Criteria Template

For each GitHub issue, acceptance criteria should include:

- [ ] Code written and tested (unit tests passing)
- [ ] All acceptance criteria checked off (from GitHub issue)
- [ ] Output files generated and validated
- [ ] Documentation complete (docstrings, comments)
- [ ] Results reviewed for sanity (spot-checks, comparisons)
- [ ] Ready for code review
- [ ] PR created and linked to issue
- [ ] Code review passed (feedback addressed)
- [ ] Merged to feature branch
- [ ] Listed as complete in STATUS.md

---

**Checklist Status:** ✅ Ready for implementation

