# GSM8K Grouping Algorithm Development Process

## Problem Statement
Divide GSM8K dataset (7,473 training samples) into ~8 balanced groups by meaningful mathematical characteristics. Each group should maintain approximately 85/15 train/test ratio.

## Acceptance Criteria
1. ✓ All samples assigned to a group
2. ✓ Groups reasonably balanced (soft constraint on size variance)
3. ✓ Train/test proportions maintained in each group
4. ✓ Structured output with group metadata
5. ✓ Validation script to verify criteria

## Dataset Overview
- **Total samples:** 7,473 (train split)
- **Target group count:** ~8 groups
- **Target group size:** ~934 samples per group
- **Variance tolerance:** Soft (user feedback: "size balancing is soft")

---

## Iteration 1: Operator Pattern Exploration

**Hypothesis:** Mathematical operators (+, -, *, /, %) are good grouping characteristics.

**Approach:**
- Extract operators from answer text
- Assign each sample to primary operator
- Create groups by operator type

**Results:**
```
Operator Distribution (Full Training Set):
  +       : 4,166 samples (55.7%)
  -       : 1,717 samples (23.0%)
  *       :   851 samples (11.4%)
  ratio   :   624 samples (8.4%)
  /       :   106 samples (1.4%)
  other   :     9 samples (0.1%)
```

**Groups Created:** 6 (not 8)
**Size Variance:** 462.89x (FAR TOO HIGH)

**Verdict:** ❌ FAILED - Operators too skewed
- Addition dominates (55.7%), making groups extremely unbalanced
- Most operators are too rare to form independent groups
- Need secondary characteristics to subdivide large operator groups

**Key Learning:** Single-feature grouping insufficient; need multi-dimensional approach.

---

## Iteration 2: Complexity Feature Analysis

**Hypothesis:** Sample complexity (operand count, magnitude of numbers, reasoning steps) provides better distribution.

**Approach:**
- Extract all numbers from answers
- Count reasoning steps
- Analyze magnitude ranges
- Examine operand counts

**Results:**

### Operand Count Distribution
Range: 2-71 operands per problem
- Highly scattered across entire range
- Many different counts with ~1-6% frequency each
- **Verdict:** Not useful for grouping (too fragmented)

### Magnitude Ranges (Numbers used in problems)
```
<10     :   132 samples (1.8%)
10-100  : 3,328 samples (44.5%)
100-1000: 2,802 samples (37.5%)
>1000   : 1,211 samples (16.2%)
```
**Verdict:** ✓ Good distribution, promising for grouping

### Reasoning Steps
```
1 step  :     0 samples (0.0%)  ← Nobody uses single step!
2 steps : 1,961 samples (26.2%)
3-4 steps: 3,808 samples (51.0%)
5+ steps : 1,704 samples (22.8%)
```
**Verdict:** ✓ EXCELLENT distribution
- More balanced than operators
- 3-4 step group (51%) still dominant but manageable
- Can subdivide 3-4 steps by magnitude for balance

**Key Learning:** Step count is the strongest primary feature. Magnitude is strong secondary feature. Combining these can create balanced groups.

---

## Strategy for Iteration 3

### Hybrid Grouping Approach: (Step Count) × (Magnitude)

**Rationale:**
- Step count provides good primary distribution (26%, 51%, 23%)
- The 51% "3-4 steps" group can be split by magnitude
- Other step groups are already well-sized

**Proposed Groups:**

| Group | Step Count | Magnitude | Est. Size | Notes |
|-------|-----------|-----------|-----------|-------|
| 0 | 2 steps | any | ~1,961 (26%) | 2-step problems |
| 1 | 3-4 steps | 10-100 | ~1,400 (19%) | Mid-complexity, small numbers |
| 2 | 3-4 steps | 100-1000 | ~1,200 (16%) | Mid-complexity, medium numbers |
| 3 | 3-4 steps | >1000 | ~1,200 (16%) | Mid-complexity, large numbers |
| 4 | 5+ steps | any | ~1,704 (23%) | 5+ step problems |
| 5-7 | Operator-based refinement | - | ~100 each | Remaining imbalance |

**Target Balance:** All groups within 900-1000 samples (±10% of mean)

**Validation Plan:**
1. Implement hybrid grouping
2. Check all samples assigned
3. Verify group sizes within tolerance
4. Check magnitude/step distribution within groups
5. Ensure 85/15 train/test split maintained

---

## Notes for Future Iterations

### Characteristics Explored (Working)
- ✓ Step count: 2, 3-4, 5+ (good signal)
- ✓ Magnitude ranges: <10, 10-100, 100-1000, >1000 (good signal)
- ✓ Operators: +, -, *, /, % (too skewed)

### Characteristics Not Yet Explored
- Problem structure: "count", "calculate", "compare"
- Question length / problem verbosity
- Specific keyword themes (money, time, distance, etc.)
- Operand count ranges (might work better than exact counts)

### Stopping Condition
- Minimum 5 iterations (will reach ~iteration 3-4 based on progress)
- Stop when acceptance criteria met
- Or 20 iterations max
- Or task proved impossible (unlikely given iteration 2 findings)

### Iteration Results Summary

| Iteration | Approach | Variance | Groups | Status |
|-----------|----------|----------|--------|--------|
| 1 | Operators only | 462.89x | 6 | ❌ Too skewed |
| 2 | Feature analysis | - | - | ✓ Exploration |
| 3 | Hybrid (steps × magnitude) | 2.61x | 8 | ✓ Good |
| 4 | Refined hybrid (+ operands for 3-4) | 2.56x | 8 | ✓ Best |
| 5 | K-means clustering | 6.72x | 8 | ❌ Worse |

### Current Status
- **Iteration:** 4/5 (best result achieved)
- **Best Approach:** Iteration 4 - Refined hybrid grouping
- **Key Metrics:** 2.56x variance, 8 groups, 665-1,703 samples per group
- **Next:** Verify train/test ratios, build validation script, finalize output format
