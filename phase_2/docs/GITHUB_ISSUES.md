# Phase 2 GitHub Issues & Work Tracking

**Created:** 2026-10-02  
**Status:** 11 issues created with dependency tracking  
**Target Completion:** 2026-11-04 (4 weeks)

---

## Quick Navigation

All Phase 2 issues are tracked on GitHub with proper labeling and blocking relationships.  
**View all issues:** https://github.com/scerruti/moe-expert-prefetching/issues?q=label:phase-2

---

## Issue Dependency Map

```
Phase 1 Outputs (Qwen1.5-MoE-A2.7B Parquet)
      │
      ├─→ #40: Data Loading ────────────────────────────────┐
      │   (CRITICAL: foundation for all analyses)            │
      │                                                       │
      ├─→ #41: Determinism Verification ←──────────────────┤
      │   (Depends on #40)                                   │
      │                                                       │
      ├─→ #42: Baseline Distributions ←───────────────────┤
      │   (Depends on #40; CRITICAL: blocks 4,6,7,8)         │
      │   │                                                   │
      │   ├─→ #43: Layer-wise Summaries ←──────────────────┤
      │   │   (Depends on #42)                               │
      │   │                                                   │
      │   ├─→ #45: Cross-Domain Token Divergence ←──────────┤
      │   │   (Also depends on #44)                          │
      │   │                                                   │
      │   ├─→ #46: JSD Analysis ←─────────────────────────┤
      │   │   (Depends on #42)                               │
      │   │                                                   │
      │   └─→ #47: Expert-level Divergence ←────────────────┤
      │       (Depends on #42)                               │
      │                                                       │
      ├─→ #44: Shared Token Identification ←────────────────┤
      │   (Depends on #40; also needed by #45)               │
      │                                                       │
      ├─→ #48: Layer Specialization ←─────────────────────┤
      │   (Depends on #41 and #46)                           │
      │                                                       │
      ├─→ #49: Visualizations ←──────────────────────────────┤
      │   (Depends on all analyses: #41-48)                  │
      │                                                       │
      └─→ #50: Comprehensive Report ←──────────────────────┤
          (Depends on #49; final synthesis)
```

---

## Issues by Priority & Status

### 🔴 CRITICAL PRIORITY (Blocks Other Work)

#### #40: Data Loading & Preparation
- **Status:** Ready to start (no blockers)
- **Assignee:** @me
- **Estimated Duration:** 2-3 days
- **Labels:** `phase-2/data-foundation`, `type/analysis`, `priority/critical`, `blocker/none`
- **Description:** Load Phase 1 Parquet files for Qwen1.5-MoE-A2.7B. Validate schema, perform data quality checks, generate summary statistics. Create unified data loader interface.
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/40
- **Outputs:**
  - `phase_2/data/data_quality_report.md`
  - `phase_2/data/summary_statistics.csv`
  - `phase_2/scripts/data_loader.py`

#### #42: Baseline Distributions Analysis
- **Status:** Blocked by #40
- **Assignee:** @me
- **Estimated Duration:** 3-4 days
- **Labels:** `phase-2/analysis`, `type/analysis`, `priority/critical`, `blocker/depends-on`
- **Description:** Compute per-layer, per-expert probability distributions for each domain (GSM8K vs MBPP). Calculate mean, median, std, quantiles. Critical blocker for #43, #45, #46, #47.
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/42
- **Blocks:** #43, #45, #46, #47
- **Outputs:**
  - `phase_2/data/baseline_distributions.json`
  - `phase_2/reports/distributions_by_layer.csv`

#### #46: Jensen-Shannon Divergence Computation
- **Status:** Blocked by #42
- **Assignee:** @me
- **Estimated Duration:** 3-4 days
- **Labels:** `phase-2/analysis`, `type/analysis`, `priority/critical`, `blocker/depends-on`
- **Description:** Compute JSD between domain-averaged distributions. Global and per-layer analysis, statistical significance testing, per-expert divergence. Critical for #48.
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/46
- **Blocks:** #48
- **Outputs:**
  - `phase_2/data/jsd_analysis.json`
  - `phase_2/reports/jsd_significance.csv`

#### #48: Layer Specialization Indexing
- **Status:** Blocked by #41, #46
- **Assignee:** @me
- **Estimated Duration:** 2-3 days
- **Labels:** `phase-2/analysis`, `type/analysis`, `priority/critical`, `blocker/depends-on`
- **Description:** Classify layers by specialization pattern (general vs. domain-specific). Compute specialization index, identify which layers specialize by domain.
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/48
- **Blocks:** #49
- **Outputs:**
  - `phase_2/reports/layer_specialization_index.csv`
  - `phase_2/reports/layer_classification.md`

#### #49: Visualizations & Chart Publishing
- **Status:** Blocked by #41-48 (all analyses)
- **Assignee:** @me
- **Estimated Duration:** 3-4 days
- **Labels:** `phase-2/visualizations`, `type/visualization`, `priority/critical`, `blocker/depends-on`
- **Description:** Create 10+ publication-ready charts (distributions, heatmaps, scatter plots, bar charts). Export as PNG/SVG and interactive HTML dashboard.
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/49
- **Blocks:** #50
- **Outputs:**
  - `phase_2/reports/figures/` (PNG, SVG)
  - `phase_2/reports/dashboard.html`
  - `phase_2/scripts/visualization.py`

#### #50: Comprehensive Report & Documentation
- **Status:** Blocked by #49
- **Assignee:** @me
- **Estimated Duration:** 2-3 days
- **Labels:** `phase-2/documentation`, `type/documentation`, `priority/critical`, `blocker/depends-on`
- **Description:** Synthesize all findings into publication-ready report. Executive summary, per-model analysis, domain cross-comparison, recommendations for Phase 3.
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/50
- **Outputs:**
  - `phase_2/reports/PHASE_2_ANALYSIS_REPORT.md`
  - `phase_2/reports/PHASE_2_ANALYSIS_REPORT.pdf`
  - `phase_2/README.md`

#### #41: Determinism Verification
- **Status:** Blocked by #40
- **Assignee:** @me
- **Estimated Duration:** 2-3 days
- **Labels:** `phase-2/verification`, `type/analysis`, `priority/critical`, `blocker/depends-on`
- **Description:** Verify expert routing is bitwise identical across Phase 1's 5 runs. Stratified sample of 200 prompts per dataset, compute max deviation per token/layer/expert.
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/41
- **Blocks:** #48
- **Outputs:**
  - `phase_2/reports/determinism_verification.md`
  - `phase_2/data/determinism_statistics.csv`

---

### 🟠 HIGH PRIORITY (Support/Analysis Tasks)

#### #43: Layer-wise Specialization Summaries
- **Status:** Blocked by #42
- **Assignee:** @me
- **Estimated Duration:** 2-3 days
- **Labels:** `phase-2/analysis`, `type/analysis`, `priority/high`, `blocker/depends-on`
- **Description:** Layer-by-layer summaries: top-activated experts, utilization rates, entropy measures. Compare GSM8K vs MBPP per layer.
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/43
- **Outputs:**
  - `phase_2/data/layer_summaries.json`
  - `phase_2/reports/layer_profiles.csv`
  - `phase_2/reports/entropy_analysis.csv`

#### #44: Shared Token Identification
- **Status:** Can start anytime (parallel track)
- **Assignee:** @me
- **Estimated Duration:** 1-2 days
- **Labels:** `phase-2/tokens`, `type/analysis`, `priority/high`, `blocker/none`
- **Description:** Identify tokens appearing in both GSM8K and MBPP. Build vocabulary overlap, count occurrences, stratify by frequency.
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/44
- **Blocks:** #45
- **Outputs:**
  - `phase_2/data/shared_tokens.csv`
  - `phase_2/reports/token_overlap_analysis.md`

#### #45: Cross-Domain Token Divergence Analysis
- **Status:** Blocked by #44, #42
- **Assignee:** @me
- **Estimated Duration:** 3 days
- **Labels:** `phase-2/tokens`, `type/analysis`, `priority/high`, `blocker/depends-on`
- **Description:** For shared tokens, compute routing probability divergence between GSM8K and MBPP. KL divergence per token/layer, identify high vs. low divergence tokens.
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/45
- **Outputs:**
  - `phase_2/data/token_divergence.csv`
  - `phase_2/reports/cross_domain_token_analysis.md`

#### #47: Expert-level Divergence & Specialization
- **Status:** Blocked by #42
- **Assignee:** @me
- **Estimated Duration:** 2-3 days
- **Labels:** `phase-2/analysis`, `type/analysis`, `priority/high`, `blocker/depends-on`
- **Description:** Characterize each expert's routing across domains. JS divergence per expert, specialization ranking, identify universal experts, clustering analysis.
- **Link:** https://github.com/scerruti/moe-expert-prefetching/issues/47
- **Outputs:**
  - `phase_2/data/expert_profiles.csv`
  - `phase_2/reports/expert_specialization.md`

---

## Labels Reference

### Component Labels (Blue)
- `phase-2/data-foundation` – Data loading and preparation
- `phase-2/verification` – Determinism and validation
- `phase-2/analysis` – Statistical analyses
- `phase-2/tokens` – Token-level analysis
- `phase-2/visualizations` – Visualizations and charts
- `phase-2/documentation` – Documentation and reporting

### Type Labels (Purple)
- `type/analysis` – Statistical/analytical work
- `type/visualization` – Visualization and charting
- `type/documentation` – Documentation

### Priority Labels (Red/Orange)
- `priority/critical` – Critical path (blocks other work)
- `priority/high` – Should be done soon (parallel track)

### Blocker Labels (Green/Yellow)
- `blocker/none` – No blockers, ready to start (green)
- `blocker/depends-on` – Has dependencies (yellow)

---

## How to Pick Up Work

### Step 1: Find Work
1. Go to GitHub Issues: https://github.com/scerruti/moe-expert-prefetching/issues
2. Filter by `phase-2` and `blocker/none` to see work ready to start
3. Pick an issue to work on

### Step 2: Create a Branch
```bash
gh issue develop <issue-number> --checkout
# Or manually:
git checkout -b feature/phase-2-<component-name>
# Example: feature/phase-2-data-loading
```

### Step 3: Work & Commit
- Check off tasks in the issue as you complete them
- Commit frequently with clear messages
- Link commits to the issue:
  ```bash
  git commit -m "Implement data loading and validation - fixes #40"
  ```

### Step 4: Create a PR
When done, create a pull request linking to the issue:
```bash
gh pr create --title "Phase 2: [Component Name]" \
  --body "Closes #<issue-number>" \
  --label "phase-2/<component>"
```

### Step 5: PR Review
- Address any review comments
- Ensure all checks pass
- Merge when approved

---

## Current Status

| Issue | Title | Status | Assignee | % Complete |
|-------|-------|--------|----------|-----------|
| #40 | Data Loading | Not Started | @me | 0% |
| #41 | Determinism Verification | Blocked | @me | 0% |
| #42 | Baseline Distributions | Blocked | @me | 0% |
| #43 | Layer-wise Summaries | Blocked | @me | 0% |
| #44 | Shared Token Identification | Not Started | @me | 0% |
| #45 | Cross-Domain Token Divergence | Blocked | @me | 0% |
| #46 | JSD Analysis | Blocked | @me | 0% |
| #47 | Expert-level Divergence | Blocked | @me | 0% |
| #48 | Layer Specialization | Blocked | @me | 0% |
| #49 | Visualizations | Blocked | @me | 0% |
| #50 | Comprehensive Report | Blocked | @me | 0% |

**Overall Progress:** 0/11 issues started

---

## Recommended Work Order

For **single developer** with moderate parallelism:
1. **#40** (Data Loading) – 2-3 days
2. **Parallel (Week 1):**
   - #41 (Determinism) – 2-3 days
   - #42 (Baseline Dist) – 3-4 days
   - #44 (Shared Tokens) – 1-2 days
3. **Sequential (Week 2-3):**
   - #43 (Layer Summaries) – 2-3 days (after #42)
   - #45 (Token Divergence) – 3 days (after #44, #42)
   - #46 (JSD) – 3-4 days (after #42)
   - #47 (Expert Divergence) – 2-3 days (after #42)
4. **#48** (Layer Specialization) – 2-3 days (after #41, #46)
5. **#49** (Visualizations) – 3-4 days (after all analyses)
6. **#50** (Report) – 2-3 days (after #49)

**Total: ~4 weeks (Oct 7 - Nov 4)**

---

## FAQ

### Q: An issue says "Blocked by #X" – can I still start?
**A:** Not recommended. Wait for the blocking issue to complete. However, you can review the implementation or prepare infrastructure offline.

### Q: What if I find a sub-issue not listed?
**A:** Add it as a checkbox in the existing issue. If significant, comment on the issue and we can create a new one.

### Q: How do I know if my work is done?
**A:** Check all the boxes in the "Acceptance Criteria" section. If they're all checked ✅, you're done!

### Q: What if I get blocked?
**A:** Comment on the GitHub issue with details, mention the maintainer, and pick up a different issue if possible.

### Q: Can I work on issues in a different order?
**A:** Only if they don't have blocking dependencies. Check the "Blocked by" section in the issue description.

### Q: How do I handle optional DeepSeek analysis?
**A:** DeepSeek work is optional if Phase 1 data becomes available. Focus on Qwen MVP first; add DeepSeek comparison if time permits.

---

## Support & Questions

- **Issues:** Comment on the GitHub issue
- **Slack/Email:** Tag the project lead
- **Documentation:** See phase_2/docs/ folder
- **Troubleshooting:** Check ARCHITECTURE.md for technical details

---

**Good luck! 🚀 Pick up an issue and get started!**

