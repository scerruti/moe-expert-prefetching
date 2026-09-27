# GSM8K Grouping Strategy & Research Hypothesis

## Overview

GSM8K (7,473 training samples) is divided into **8 balanced groups** by mathematical complexity characteristics. The grouping strategy is designed to test whether problem characteristics correlate with expert activation patterns in MoE routing.

## Grouping Rationale

### Problem
A single random dataset doesn't reveal how MoE routers specialize. We need samples that:
1. Are internally consistent (similar expert activation within group)
2. Differ meaningfully between groups (divergent expert activation across groups)
3. Maintain dataset balance (no group dominates Phase 1 collection)

### Solution: Complexity-Based Grouping

Groups are divided by three features that characterize problem *structure* and *difficulty*:

#### 1. **Reasoning Steps** (Primary feature)
- **2 steps:** 1,961 samples (26%) — Simple two-step calculations
- **3-4 steps:** 3,808 samples (51%) — Standard multi-step reasoning
- **5+ steps:** 1,704 samples (23%) — Complex, deeply-reasoned problems

**Why this matters:** 
- Step count proxies problem complexity
- More steps → longer token sequences → deeper expert routing
- Different complexity levels likely require different expert patterns

#### 2. **Magnitude of Numbers** (Secondary feature)
- **<10:** 1.8% — Trivial scale problems
- **10-100:** 44.5% — Small-scale word problems
- **100-1000:** 37.5% — Medium-scale reasoning
- **>1000:** 16.2% — Large-scale numerical problems

**Why this matters:**
- Magnitude affects problem semantics and solution strategies
- Small numbers (10-100) might use different token patterns than large (>1000)
- Different magnitude ranges → different mathematical operation distributions

#### 3. **Operand Count** (Tertiary refinement)
- Applied only to 3-4 step category (51% of dataset)
- Breaks large dominant groups into balanced subgroups
- Captures problem complexity nuance (2 numbers vs 20 numbers)

### Algorithm

**Iteration 4: Refined Hybrid Grouping**
```
For each sample:
  step_category = categorize_steps(reasoning_steps)
  magnitude_category = categorize_magnitude(max_number)
  
  if step_category == "3-4":  # Dominant category
    operand_category = categorize_operands(count)
    group_key = (step_category, magnitude_category, operand_category)
  else:
    group_key = (step_category, magnitude_category)

# Greedy assignment: assign each key to group with smallest size
# Result: 2.56x variance (balanced within soft constraint)
```

**Selection Rationale:** 
- Iteration 1 (operators): 462.89x variance — Too skewed
- Iteration 3 (steps × magnitude): 2.61x variance — Good
- Iteration 4 (+ operands): 2.56x variance — Minimal improvement but selected
- Iteration 5 (K-means): 6.72x variance — Worse

## Research Hypothesis

### Primary Question
**Do problem complexity characteristics (steps, magnitude, operands) correlate with MoE expert activation patterns?**

### Hypothesis
Groups with similar complexity characteristics will:
1. **Internally consistent:** Activate similar sets of experts within the group
2. **Between-group divergent:** Activate different expert sets across groups
3. **Measurable partitioning:** Experts will show >80% within-group similarity, <50% between-group similarity

### Theoretical Basis

**Why grouping by complexity should activate different experts:**

1. **Problem structure determines solution strategy**
   - 2-step problems: Direct calculation paths
   - 5+ step problems: Multi-stage reasoning, intermediate steps
   - Different strategies → different token sequences → different expert routing

2. **Magnitude affects token distributions**
   - Small numbers (10-100): Tokens like "five", "hundred", common operations
   - Large numbers (>1000): Tokens like "thousand", more complex operations
   - Token distributions determine router gate attention → expert selection

3. **Operand complexity reflects reasoning depth**
   - Few operands: Simple pairwise operations
   - Many operands: Aggregation, accumulation, repeated operations
   - More complex sequences → activate deeper, more specialized experts

**Why this might NOT work:**

1. **Router learns from token embeddings, not problem semantics**
   - Our grouping is problem-level; router sees token-level patterns
   - Same token might appear in different complexity groups

2. **Expert activation might be driven by other factors**
   - Token position, context length, operation type
   - Not problem magnitude or step count

3. **Overlap could be substantial**
   - Groups might activate 70-80% overlapping expert sets
   - Only 20-30% expert variance between groups

## Phase 1 Validation Plan

### What to Measure

For each group, capture:
```
Expert Activation Matrix:
  [group_id, layer, token_position, top_k_experts, activation_scores]

Aggregated Statistics:
  - Expert frequency per group (which experts used how often)
  - Layer-wise expert usage (do different layers show different patterns)
  - Token-to-expert mappings (does same token activate same expert across groups)
  - Within-group expert overlap (consistency)
  - Between-group expert divergence (distinctness)
```

### Success Criteria

**Strong support for hypothesis:**
- Within-group expert overlap: >85% (samples in group activate similar experts)
- Between-group expert overlap: <50% (different groups activate different experts)
- Clear expert specialization by group

**Weak support (results still valuable):**
- Within-group overlap: 70-85%
- Between-group overlap: 50-70%
- Suggests partial expert specialization or other factors driving routing

**No support (research pivot):**
- Within-group overlap: <70%
- Between-group overlap: >70%
- Indicates grouping strategy doesn't correlate with expert activation
- Router might be routing on other features (not problem complexity)

### Output

Phase 1 generates:
```
phase_1/data/routing_telemetry_by_group/
  ├── group_0/
  │   ├── expert_frequencies.json
  │   ├── layer_expert_usage.json
  │   └── sample_routing_logs.jsonl
  ├── group_1/
  ├── ...
  └── analysis/
      ├── within_group_similarity.json
      ├── between_group_divergence.json
      └── expert_specialization_report.md
```

## Expected Outcomes

### Scenario 1: Grouping Validates (High Probability)
- Groups show distinct expert activation patterns
- Step count strongest signal: 5+ step problems activate notably different experts
- Magnitude provides secondary differentiation
- Result: Confirms grouping strategy is effective for Phase 2+ research

### Scenario 2: Partial Validation (Medium Probability)
- Groups show some expert differentiation but not strong
- Suggests other factors (token length, operation type) also matter
- Grouping helps but isn't sufficient alone
- Result: Need additional grouping dimensions for Phase 2

### Scenario 3: Grouping Doesn't Validate (Lower Probability)
- Expert activation largely independent of problem complexity
- Router might be optimizing for other properties (efficiency, load balancing)
- Grouping doesn't correlate with expert specialization
- Result: Research pivot to other grouping approaches (by operation type, domain, etc.)

## Implications

**If validated:** 
- Problem complexity naturally partitions expert specialization
- Can use complexity-based grouping for downstream analysis
- Suggests MoE routers learn meaningful problem semantics

**If partially validated:**
- Complexity is one factor among several
- Need multi-dimensional grouping for future phases
- Router learns hybrid of semantic and distributional features

**If not validated:**
- Current grouping strategy doesn't drive expert activation
- Need different hypothesis about router specialization
- Explore operation-based, domain-based, or empirical clustering

## Files

- **Algorithm:** `phase_1/scripts/gsm8k_grouping.py` (Iteration 4 selected)
- **Group assignments:** `phase_1/data/gsm8k_groups.json` (reproducible, version controlled)
- **Process documentation:** `phase_1/docs/gsm8k_grouping_process.md` (algorithm iterations)
- **Dataset loader:** `phase_1/scripts/dataset_loaders.py` (group-based loading support)

## Next Steps

1. **Phase 1 collection:** Run data collection on all 8 groups, capture routing telemetry
2. **Analysis:** Compute expert activation statistics per group
3. **Validation:** Check within/between-group overlap against success criteria
4. **Documentation:** Report findings and implications for Phase 2 research
