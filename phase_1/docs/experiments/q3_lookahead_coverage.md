# Question 3: can a prefetch predictor absorb batch-dependent routing changes?

**Status:** preliminary (experiment branch `experiment/q3-lookahead-coverage`). Discussed in #114 (question 3); target wording in #110.

## Question

About 2% of top-4 routing decisions change when the same prompts are batched differently (#114), and a change at one layer cascades into later layers. Can a predictor that prefetches extra experts (*m* beyond the top *k* = 4), predicting *H* layers ahead from the token's live hidden state, still cover the experts actually used?

## Method

For each determinism-subset ordering compared with ordering 0, per (prompt, token, layer):

- **changed:** the two orderings' top-4 sets differ.
- **coverage(*m*):** each ordering's top 4 is inside the other's top-(4 + *m*).
- **first flip:** the earliest layer at which this token's top 4 changed, for this pair of orderings.
- **unforeseeable at lookahead *H*:** a change at layer *l* whose token first flipped inside (*l* − *H*, *l*]. A predictor reading the live state at layer *l* − *H* saw an unchanged state. If the first flip is at or before *l* − *H*, the predictor already sees the diverged state, so the change is **foreseeable**.

## Data

- **Trial collection** at `38ef1da`: Qwen1.5-MoE-A2.7B, fp16, batch size 32, Colab A100 40 GB. Stratified 200-prompt subset per dataset, 5 orderings. **Before** the `position_ids` fix (`1d9d995`), which lowered the change rate by about 9%, so these numbers are slightly pessimistic.
- `determinism/` folder: [Drive, `trial_38ef1da`](https://drive.google.com/drive/folders/1F5ObENu9J_J5yH7ZIr-J4qMPwDqnyfyP?usp=drive_link).

## Reproduce

```bash
python phase_1/scripts/lookahead_coverage.py <collection output>/determinism
```

## Results

```text
1,462,560 compared decisions; top-4 changed: 29,573 (2.02%)

1. Over-prediction: share of CHANGED decisions covered by predicting top-(4+m)
             changed   m=1   m=2   m=4   m=8
  all layers   2.02% 81.6% 94.5% 99.1% 99.9%
  layers 0-7   0.95% 89.8% 98.0% 99.8% 99.9%
 layers 8-15   1.51% 88.5% 96.8% 99.3% 99.9%
layers 16-23   3.61% 76.5% 92.5% 98.8% 99.8% 

2. Lookahead: share of ALL decisions that changed unforeseeably (first flip within the last H layers)
             all changes   H=1   H=2   H=4   H=8
  all layers       2.02% 1.00% 1.11% 1.29% 1.56%
  layers 0-7       0.95% 0.73% 0.81% 0.89% 0.95%
 layers 8-15       1.51% 0.80% 0.90% 1.05% 1.30%
layers 16-23       3.61% 1.47% 1.63% 1.94% 2.45% 

3. Both: share of ALL decisions still missed with lookahead H and top-(4+m)
    unforeseeable missed, m=1 missed, m=2 missed, m=4 missed, m=8
H=1         1.00%      0.060%      0.005%      0.000%      0.000%
H=2         1.11%      0.089%      0.012%      0.002%      0.000%
H=4         1.29%      0.134%      0.026%      0.003%      0.001%
H=8         1.56%      0.218%      0.053%      0.008%      0.001% 

   Same, deepest layers only (16-23):
    unforeseeable missed, m=1 missed, m=2 missed, m=4 missed, m=8
H=1         1.47%      0.125%      0.013%      0.001%      0.000%
H=2         1.63%      0.165%      0.021%      0.003%      0.001%
H=4         1.94%      0.246%      0.047%      0.006%      0.002%
H=8         2.45%      0.432%      0.107%      0.014%      0.002%
```

The 29,573 changed decisions (2.02%) match `batch_variance.py` and the validator on the same data.

## Reading the results

1. **Over-prediction alone** covers most changed decisions: 81.6% with *m* = 1, 94.5% with *m* = 2, 99.1% with *m* = 4. It's lower in the deepest layers (92.5% at *m* = 2).
2. **About half of all changes are foreseeable** for a live-state predictor. Only 1.00% of decisions change unforeseeably at *H* = 1, rising to 1.56% at *H* = 8, and 2.45% in layers 16–23.
3. **Together:** the unforeseeable changes are almost all near-ties. With *m* = 2, only 0.005% (*H* = 1) to 0.053% (*H* = 8) of all decisions are still missed; at most 0.107% in the deepest layers.

**Conclusion:** batch-dependent routing is not a practical obstacle for a predictor that reads live hidden state with a small buffer (*m* = 2: 6 of 60 experts per layer, +50% transfer over the minimum). The predictor's own accuracy is the open question for Phase 3.

## Caveats

- This is a **ceiling from batch effects alone**. It assumes a perfect predictor of one batch composition, not a real predictor's error.
- It assumes the predictor runs **inside the same forward pass**, reading live state. A predictor working from token identity alone would face the full 2%.
- "Foreseeable" approximates "already diverged at the input layer" by "flipped at or before it".
- One GPU type (A100 40 GB), fp16, batch size 32, and data from before the `position_ids` fix.
- **For Phase 3 evaluation**, use labels from the same collection run as the inputs. Comparing across runs counts batch effects as predictor errors.
