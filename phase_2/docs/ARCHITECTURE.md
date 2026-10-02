# Phase 2 Architecture: Statistical Analysis Pipeline

**Last Updated:** 2026-10-02  
**Model:** Qwen1.5-MoE-A2.7B (MVP), DeepSeek-V2-Lite (optional research extension)

---

## System Overview

Phase 2 transforms Phase 1's raw routing traces into actionable statistical insights through a modular analysis pipeline.

```
Phase 1 Outputs (Parquet Files)
  ├─ gsm8k_routing_traces.parquet
  └─ mbpp_routing_traces.parquet
      │
      ↓
  ┌──────────────────────────────────────────────┐
  │  Phase 2: Statistical Analysis Pipeline      │
  │                                              │
  │  [#40] Data Loading ──→ Unified Loader      │
  │         │                                    │
  │         ├─→ [#41] Determinism Verification  │
  │         │          → binwise comparison      │
  │         │                                    │
  │         ├─→ [#42] Baseline Distributions    │
  │         │          → mean/std/quantiles      │
  │         │          │                         │
  │         │          ├─→ [#43] Layer Summaries│
  │         │          ├─→ [#46] JSD Analysis   │
  │         │          └─→ [#47] Expert Prof.   │
  │         │                                    │
  │         ├─→ [#44] Shared Token Identification
  │         │          → vocab overlap           │
  │         │          │                         │
  │         │          └─→ [#45] Token Divergence
  │         │                                    │
  │         └─→ [#48] Layer Specialization      │
  │                   ← (uses #41, #46 results) │
  │                                              │
  │         ┌──────────────────────────────────┐
  │         │    All Results/Analyses ↓        │
  │         │  [#49] Visualizations & Publishing
  │         │  [#50] Comprehensive Report      │
  └─────────└──────────────────────────────────┘
      │
      ↓
  Publication-Ready Outputs
  ├─ Statistical Reports (Markdown, PDF)
  ├─ Visualizations (PNG, SVG, HTML dashboard)
  ├─ Data Artifacts (CSV, JSON, Parquet)
  └─ Phase 3 Recommendations
```

---

## Component Architectures

### Component 1: Data Foundation (#40)

**Module:** `phase_2/scripts/data_loader.py`

```python
class RoutingDataLoader:
    """Unified interface for loading Phase 1 Parquet files."""
    
    def __init__(self, parquet_path: str, model_name: str):
        self.parquet_path = parquet_path
        self.model_name = model_name  # "qwen" or "deepseek"
        self.data = None
        
    def load(self) -> pl.DataFrame:
        """Load Parquet file using Polars."""
        # Load with optimizations (lazy evaluation, streaming)
        
    def validate_schema(self) -> dict:
        """Verify schema against SYSTEM_DESIGN.md spec."""
        # Check 9 required columns, types, ranges
        
    def quality_report(self) -> dict:
        """Generate data quality statistics."""
        # Missing values, null counts, range checks
        
    def summary_stats(self) -> pd.DataFrame:
        """Compute aggregate statistics."""
        # Tokens per layer, experts per layer, etc.
```

**Outputs:**
- `data_quality_report.md` – Human-readable quality assessment
- `summary_statistics.csv` – Aggregate stats table
- `memory_profile.txt` – Peak RAM usage during load

---

### Component 2: Determinism Verification (#41)

**Module:** `phase_2/scripts/determinism.py`

```python
def verify_determinism(loader: RoutingDataLoader, 
                      sample_size: int = 200) -> dict:
    """
    Verify bitwise determinism across 5 Phase 1 runs.
    
    Strategy:
    1. Stratified random sample (200 prompts per dataset)
    2. For each prompt, compare probabilities across 5 runs
    3. Compute max deviation per token/layer/expert
    4. Pass criterion: max_deviation < 1e-6
    
    Returns:
        dict with keys:
        - pass_fail: bool (True if max_deviation < threshold)
        - max_deviation: float
        - stats_by_layer: dict (mean, median, max deviation per layer)
        - stats_by_expert: dict (per-expert deviation statistics)
    """
    pass

def plot_determinism(stats: dict) -> None:
    """Visualization: max deviation per layer, per expert."""
    pass
```

**Outputs:**
- `determinism_verification.md` – Pass/fail report with findings
- `determinism_statistics.csv` – Detailed deviation table

---

### Component 3: Baseline Distributions (#42)

**Module:** `phase_2/scripts/baseline_distributions.py`

```python
def compute_distributions(loader: RoutingDataLoader) -> dict:
    """
    Compute per-layer, per-expert probability distributions.
    
    For each layer and domain (GSM8K/MBPP):
    1. Aggregate expert probabilities across all prompts
    2. Calculate: mean, median, std, min, max, quantiles (25/50/75/95)
    3. Handle token averaging (same token, multiple prompts)
    
    Returns:
        dict[layer_idx][domain] = {
            'mean': array([128]),      # mean prob per expert
            'std': array([128]),       # std dev
            'quantiles': dict,         # q25, q50, q75, q95
            'token_counts': dict,      # how many tokens per expert
        }
    """
    pass

def distribution_comparison(distributions: dict, 
                           layer_idx: int) -> pd.DataFrame:
    """
    Compare distributions GSM8K vs MBPP for a single layer.
    Returns table with mean/std/quantiles side-by-side.
    """
    pass
```

**Outputs:**
- `baseline_distributions.json` – Complete statistical tensors
- `distributions_by_layer.csv` – Human-readable summaries

---

### Component 4: Layer-wise Summaries (#43)

**Module:** `phase_2/scripts/layer_summaries.py`

```python
def layer_statistics(distributions: dict) -> dict:
    """
    Per-layer summaries: top experts, utilization, entropy.
    
    For each layer:
    1. Top-k activated experts (by mean probability)
    2. Expert utilization rate (fraction > threshold)
    3. Shannon entropy of routing distribution
    4. Per-domain comparison
    
    Returns:
        dict[layer_idx] = {
            'top_experts_gsm8k': list,        # [expert_id, ...]
            'top_experts_mbpp': list,
            'utilization_gsm8k': float,       # 0-1
            'utilization_mbpp': float,
            'entropy_gsm8k': float,           # Shannon entropy
            'entropy_mbpp': float,
        }
    """
    pass
```

**Outputs:**
- `layer_summaries.json` – Full layer profiles
- `layer_profiles.csv` – Spreadsheet-friendly format
- `entropy_analysis.csv` – Entropy ranking by layer

---

### Component 5 & 6: Token Analysis (#44, #45)

**Module:** `phase_2/scripts/token_analysis.py`

```python
def identify_shared_tokens(loader: RoutingDataLoader) -> dict:
    """
    Build vocabulary overlap between GSM8K and MBPP.
    
    Returns:
        dict with keys:
        - 'shared_tokens': list of token_ids
        - 'shared_token_strs': list of decoded token strings
        - 'gsm8k_vocab_size': int
        - 'mbpp_vocab_size': int
        - 'overlap_percentage': float
        - 'shared_token_frequencies': dict[token_id -> (gsm8k_count, mbpp_count)]
    """
    pass

def token_divergence_analysis(loader: RoutingDataLoader,
                             shared_tokens: dict) -> pd.DataFrame:
    """
    KL divergence per shared token across domains.
    
    For each shared token:
    1. Extract probability distributions (GSM8K vs MBPP) per layer
    2. Compute KL divergence(GSM8K || MBPP) per layer
    3. Aggregate across layers (mean, max)
    4. Rank tokens by divergence
    
    Returns:
        DataFrame with columns:
        - token_id, token_str
        - kl_divergence_mean, kl_divergence_max
        - specialization_score (high = domain-specific)
    """
    pass
```

**Outputs:**
- `shared_tokens.csv` – Shared token inventory with frequencies
- `token_overlap_analysis.md` – Summary statistics and insights
- `token_divergence.csv` – Ranked tokens by cross-domain divergence

---

### Component 7: Jensen-Shannon Divergence (#46)

**Module:** `phase_2/scripts/jsd_analysis.py`

```python
def jensen_shannon_divergence(distributions: dict) -> dict:
    """
    Compute JSD between domain-averaged distributions.
    
    Global Level:
    1. Average all expert probabilities across GSM8K
    2. Average all expert probabilities across MBPP
    3. Compute JS divergence between these distributions
    
    Per-Layer Level:
    1. Repeat (1-3) for each layer independently
    2. Compute bootstrap confidence intervals (n=1000)
    3. Statistical significance testing
    
    Per-Expert Level:
    1. For each expert, compute divergence across domains
    2. Rank experts by specialization
    
    Returns:
        dict with keys:
        - 'global_jsd': float
        - 'global_ci_95': (lower, upper)
        - 'per_layer_jsd': array([num_layers])
        - 'per_layer_ci_95': array([num_layers, 2])
        - 'per_expert_jsd': array([num_experts])
    """
    pass

def jsd_significance_test(distributions: dict,
                         n_bootstrap: int = 1000) -> pd.DataFrame:
    """
    Bootstrap-based significance testing for JSD values.
    
    Returns DataFrame with:
    - layer_idx, jsd_value, ci_lower, ci_upper, p_value
    """
    pass
```

**Outputs:**
- `jsd_analysis.json` – Complete JSD results with confidence intervals
- `jsd_significance.csv` – Significance testing results
- `jsd_per_expert.csv` – Expert-level divergence rankings

---

### Component 8: Expert Profiling (#47)

**Module:** `phase_2/scripts/expert_profiling.py`

```python
def expert_profiles(distributions: dict) -> pd.DataFrame:
    """
    Characterize each expert's routing behavior.
    
    For each expert across all layers:
    1. Activation frequency (% of tokens where expert is top-2)
    2. Mean probability across domains
    3. Domain-specific probability divergence
    4. Specialization classification (universal, domain-specific, etc.)
    
    Returns DataFrame with columns:
    - expert_id
    - activation_freq_gsm8k, activation_freq_mbpp
    - mean_prob_gsm8k, mean_prob_mbpp
    - jsd_gsm8k_vs_mbpp
    - specialization_type (categorical)
    """
    pass

def expert_clustering(expert_profiles: pd.DataFrame,
                     n_clusters: int = 5) -> dict:
    """
    Cluster experts by routing behavior using k-means.
    
    Returns:
        dict[cluster_id] = {
            'experts': list,
            'profile': str,  # e.g., "universal", "math-specific", etc.
            'characteristics': dict,
        }
    """
    pass
```

**Outputs:**
- `expert_profiles.csv` – Expert-level statistics
- `expert_specialization.md` – Clustering results and profiles

---

### Component 9: Layer Specialization (#48)

**Module:** `phase_2/scripts/layer_specialization.py`

```python
def layer_specialization_index(distributions: dict,
                              jsd_results: dict,
                              determinism: dict) -> pd.DataFrame:
    """
    Compute specialization index per layer.
    
    Specialization Index combines:
    1. JSD between domains (high = specialized)
    2. Expert utilization divergence (high = specialized)
    3. Top-expert consistency (low = specialized)
    
    Classification:
    - Early layers (0-5): mostly general linguistic processing
    - Mid layers (6-20): domain-specific expert allocation
    - Late layers (21+): integration and refinement
    
    Returns DataFrame with columns:
    - layer_idx
    - specialization_index (0-100)
    - classification (early/mid/late)
    - top_specialized_experts (list)
    - recommendations_for_phase_3
    """
    pass
```

**Outputs:**
- `layer_specialization_index.csv` – Specialization scores per layer
- `layer_classification.md` – Classification report with Phase 3 recommendations

---

### Component 10: Visualizations (#49)

**Module:** `phase_2/scripts/visualization.py`

```python
def create_visualizations(loader: RoutingDataLoader,
                         all_results: dict) -> None:
    """
    Generate 10+ publication-ready charts.
    
    Charts (organized by analysis type):
    
    Distribution Plots:
    1. Distribution comparison (GSM8K vs MBPP) per selected layers
    2. Expert activation probability density plots
    3. Layer-wise entropy comparison
    
    Heatmaps:
    4. Layer × Expert heatmap (mean probability per domain)
    5. JSD heatmap (layer × expert divergence)
    6. Specialization index heatmap
    
    Scatter/Ranking:
    7. Shared token divergence scatter plot
    8. Expert specialization scatter (activation vs divergence)
    9. Layer specialization bar chart
    
    Composite:
    10. Specialization landscape (layer classification)
    11. Expert clustering visualization
    12. Layer heatmap with annotations
    
    Export formats:
    - PNG (300 dpi for publications)
    - SVG (vector, for editing)
    - Interactive HTML (plotly for exploration)
    """
    pass

def create_dashboard(all_results: dict) -> None:
    """
    Generate interactive HTML dashboard with:
    - Tabs for each analysis component
    - Drill-down capabilities
    - Export buttons (PNG)
    - Summary statistics cards
    """
    pass
```

**Outputs:**
- `phase_2/reports/figures/` – PNG, SVG, PDF charts
- `phase_2/reports/dashboard.html` – Interactive dashboard
- `phase_2/scripts/visualization.py` – Reproducible plotting code

---

### Component 11: Final Report (#50)

**Module:** `phase_2/scripts/report_generator.py`

```python
def generate_report(all_results: dict, model_name: str) -> str:
    """
    Synthesize all findings into comprehensive Markdown report.
    
    Report Structure:
    1. Title & Metadata
    2. Executive Summary (1 page)
       - Key findings
       - Implications for Phase 3
    3. Introduction & Methodology
    4. Data Overview
    5. Determinism Verification (with results)
    6. Domain Distribution Analysis
    7. Shared Token Analysis
    8. Divergence Metrics (JSD, KL)
    9. Layer Specialization Profiles
    10. Expert Characterization
    11. Visualizations & Figures
    12. Recommendations for Phase 3
    13. Appendices (detailed tables, confidence intervals)
    
    Export:
    - Markdown (.md) – primary format
    - PDF – via pandoc
    - HTML – via markdown → html
    """
    pass
```

**Outputs:**
- `PHASE_2_ANALYSIS_REPORT.md` – Main report
- `PHASE_2_ANALYSIS_REPORT.pdf` – PDF export
- `README.md` (phase_2/) – Overview and results summary

---

## Data Flow & Dependencies

```
Phase 1 Parquet Files
    ↓
[#40] Data Loader
    ├─ Validates schema
    ├─ Generates QA report
    └─ Outputs unified loader interface
        ↓
    ├─→ [#41] Determinism Verification
    │   ├─ Samples 200 prompts per dataset
    │   ├─ Compares across 5 runs
    │   └─ Produces: determinism_statistics.csv
    │
    ├─→ [#42] Baseline Distributions
    │   ├─ Aggregates probabilities per layer
    │   ├─ Computes mean/std/quantiles
    │   └─ Produces: baseline_distributions.json
    │       ├─→ [#43] Layer Summaries
    │       ├─→ [#46] JSD Analysis
    │       └─→ [#47] Expert Profiling
    │
    ├─→ [#44] Shared Token ID
    │   ├─ Builds vocabulary overlap
    │   └─ Produces: shared_tokens.csv
    │       ├─→ [#45] Token Divergence
    │
    └─→ [#48] Layer Specialization
        ├─ Uses #41, #46 results
        └─ Produces: layer_specialization_index.csv
            ↓
        [#49] Visualizations
        ├─ All results fed to visualization engine
        ├─ Produces: PNG, SVG, HTML
        └─ Outputs: dashboard.html
            ↓
        [#50] Final Report
        ├─ Integrates all findings
        ├─ Produces: PHASE_2_ANALYSIS_REPORT.md/pdf
        └─ Recommendations for Phase 3
```

---

## Statistical Methods

### Jensen-Shannon Divergence

$$D_{JS}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)$$

where $M = \frac{1}{2}(P + Q)$ and $D_{KL}$ is Kullback-Leibler divergence.

**Interpretation:**
- JS = 0: Identical distributions (no domain difference)
- JS > 0.1: Significant domain-specific routing
- JS > 0.2: Very strong specialization

### Specialization Index

$$I_{spec}(l) = w_1 \cdot D_{JS}(l) + w_2 \cdot \Delta_{util}(l) + w_3 \cdot (1 - C_{top}(l))$$

where:
- $D_{JS}(l)$ – JSD between domains at layer $l$
- $\Delta_{util}(l)$ – Expert utilization divergence
- $C_{top}(l)$ – Top-expert consistency (Jaccard similarity)
- $w_1, w_2, w_3$ – Weights (default: 0.5, 0.25, 0.25)

---

## Software Stack

### Required Packages
```
scipy>=1.8.0          # KL/JS divergence, statistical tests
polars>=0.19.0        # Fast Parquet I/O and querying
duckdb>=0.8.0         # SQL-based analysis (optional)
scikit-learn>=1.2.0   # Entropy, clustering
numpy>=1.20.0         # Numerical operations
pandas>=1.5.0         # Data wrangling
matplotlib>=3.5.0     # Static visualizations
seaborn>=0.12.0       # Statistical plotting
plotly>=5.0.0         # Interactive visualizations
```

### Development
```
jupyter>=1.0.0        # Interactive notebooks
black>=23.0.0         # Code formatting
pytest>=7.0.0         # Testing framework
```

---

## Code Organization

```
phase_2/
├── docs/
│   ├── ARCHITECTURE.md          # This file
│   ├── GITHUB_ISSUES.md         # Issue tracking
│   ├── CHECKLIST.md             # Granular tasks
│   └── STATUS.md                # Status tracking
│
├── scripts/
│   ├── __init__.py
│   ├── data_loader.py           # #40
│   ├── determinism.py           # #41
│   ├── baseline_distributions.py# #42
│   ├── layer_summaries.py       # #43
│   ├── token_analysis.py        # #44, #45
│   ├── jsd_analysis.py          # #46
│   ├── expert_profiling.py      # #47
│   ├── layer_specialization.py  # #48
│   ├── visualization.py         # #49
│   └── report_generator.py      # #50
│
├── notebooks/
│   └── exploratory/             # Draft analysis notebooks
│
├── data/
│   ├── data_quality_report.md
│   ├── summary_statistics.csv
│   ├── baseline_distributions.json
│   ├── layer_summaries.json
│   ├── shared_tokens.csv
│   ├── determinism_statistics.csv
│   ├── token_divergence.csv
│   ├── jsd_analysis.json
│   ├── jsd_significance.csv
│   ├── expert_profiles.csv
│   └── layer_specialization_index.csv
│
└── reports/
    ├── determinism_verification.md
    ├── distributions_by_layer.csv
    ├── token_overlap_analysis.md
    ├── cross_domain_token_analysis.md
    ├── expert_specialization.md
    ├── layer_classification.md
    ├── figures/
    │   ├── distribution_comparison_layer*.png
    │   ├── jsd_heatmap.png
    │   ├── specialization_landscape.png
    │   ├── expert_clustering.png
    │   └── ... (10+ charts)
    ├── dashboard.html
    ├── PHASE_2_ANALYSIS_REPORT.md
    ├── PHASE_2_ANALYSIS_REPORT.pdf
    └── README.md
```

---

## Model-Specific Considerations

### Qwen1.5-MoE-A2.7B (MVP)
- **Expert count:** 64 routed experts
- **Top-k routing:** top-2
- **Layers:** 24 MoE layers
- **Expected JSD range:** 0.05-0.2 (moderate specialization)

### DeepSeek-V2-Lite (Optional Research)
- **Expert count:** 128 routed experts
- **Top-k routing:** top-6
- **Layers:** 27 MoE layers (estimated)
- **Expected JSD range:** 0.08-0.25 (stronger specialization likely)
- **Note:** Include cross-model comparison if data available

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Data loading time | <5 min | Both datasets, Parquet I/O |
| Determinism check | <10 min | 200 prompt sample |
| Baseline distributions | <30 min | All layers, aggregations |
| JSD computation | <20 min | Bootstrap included |
| Visualizations | <30 min | 10+ charts, all formats |
| Total pipeline runtime | <3 hours | End-to-end, optimized |
| Memory peak | <16GB | Single machine sufficient |

---

## Quality Assurance

### Validation Checkpoints

1. **Data Quality:** Schema validation, null counts, value ranges
2. **Determinism:** Max deviation < 1e-6 for sample subset
3. **Distribution Consistency:** Probabilities sum to 1.0 (within epsilon)
4. **Statistical Tests:** Bootstrap confidence intervals, p-values
5. **Visualization Quality:** Charts readable, labels clear, colors accessible
6. **Report Completeness:** All sections present, no missing figures

### Testing Strategy

```python
# pytest tests/
tests/
├── test_data_loader.py        # Schema, QA checks
├── test_determinism.py        # Variance computation
├── test_distributions.py      # Statistical aggregation
├── test_jsd.py                # Divergence calculation
├── test_visualization.py      # Chart generation
└── conftest.py                # Fixtures for sample data
```

---

## Next Phase (Phase 3) Preparation

Phase 2 outputs that feed Phase 3 (Predictor Training):

1. **Layer Specialization Report (#48)**
   - Identifies which layers to focus on for predictor training
   - Ranking: layers with highest specialization get most attention

2. **Baseline Distributions (#42)**
   - Probability distributions serve as baseline for predictor evaluation
   - Used to compute expected activation patterns

3. **Shared Token Analysis (#45)**
   - Token-level predictions can prioritize high-divergence tokens
   - Universal tokens may be easier to predict

4. **Expert Profiles (#47)**
   - Clustering results can guide expert grouping for predictor architecture
   - Universal experts may need less prediction overhead

---

**Status:** ✅ Architecture finalized and ready for implementation

