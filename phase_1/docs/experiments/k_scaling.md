# Does the prefetch buffer scale with k?

**Status:** preliminary (experiment branch `experiment/k-scaling`). Related: #114 (question 3), #110.

## Question

For Qwen (k = 4), k + 2 and 1.5k are both 6. Does the buffer stay roughly constant as k grows, or grow in proportion?

## Method

Each row stores all 60 expert probabilities, so a hypothetical top-k cut-off can be applied to the same routing. For each pair of orderings and each k, we find the smallest m such that each ordering's top-k is inside the other's top-(k+m).

**The main limit:** this measures how crowded the ranking is around rank k. It doesn't simulate a model built with that k: hidden states and cascades still come from top-4 routing, so only k = 4 is a real measurement. It also covers batch effects only, not a predictor's error.

## Data

Trial collection at `38ef1da` (1,462,560 comparisons; fp16, batch 32, A100 40 GB; before the `position_ids` fix): [Drive, `trial_38ef1da`](https://drive.google.com/drive/folders/1F5ObENu9J_J5yH7ZIr-J4qMPwDqnyfyP?usp=drive_link).

```bash
python phase_1/scripts/k_scaling.py <collection output>/determinism
```

## Results

"m for 95%" is the smallest m covering 95% of changed decisions. "Missed at m=2" is a share of all decisions.

```text
all layers:
   k  changed  median gap  m for 90%  m for 95%  m for 99%  (k+m95)/k  missed at m=2
   1   0.49%      0.0538          1          2          3       3.00        0.009%
   2   0.86%      0.0241          2          2          4       2.00        0.029%
   3   1.38%      0.0108          2          2          4       1.67        0.065%
   4   2.02%      0.0050          2          3          4       1.75        0.115%
   6   3.23%      0.0019          2          3          5       1.50        0.249%
   8   4.25%      0.0011          2          3          6       1.38        0.404%
  12   5.87%      0.0006          3          4          6       1.33        0.699%
  16   6.87%      0.0004          3          4          7       1.25        0.900%

layers 16-23:
   k  changed  median gap  m for 90%  m for 95%  m for 99%  (k+m95)/k  missed at m=2
   1   0.87%      0.0538          2          2          4       3.00        0.023%
   2   1.60%      0.0241          2          2          4       2.00        0.071%
   3   2.53%      0.0108          2          3          4       2.00        0.154%
   4   3.61%      0.0050          2          3          5       1.75        0.275%
   6   5.70%      0.0019          3          3          6       1.50        0.599%
   8   7.63%      0.0011          3          4          6       1.50        0.932%
  12  10.23%      0.0006          3          4          7       1.33        1.569%
  16  11.93%      0.0004          3          4          7       1.25        2.057%

```

Check: k = 4 reproduces the 2.02% change rate from `batch_variance.py`.

## Reading the results

- **Neither simple rule fits.** From k = 1 to 16, m for 95% only goes from 2 to 4, far below proportional growth (1.5k would allow m = 8 at k = 16), and (k+m)/k falls from 3.0 to 1.25.
- **A fixed m doesn't hold its quality.** Misses at m = 2 rise about 100-fold, from 0.009% to 0.900%, because changes become more common as the median gap at the cut-off shrinks from 0.054 to 0.0004.
- **So the buffer seems to depend on how crowded the ranking is at the cut-off,** not on k directly. For Qwen, state a measured m; don't generalise it as a rule.

## Are changes really at near-ties?

The gap between the 4th and 5th expert (k = 4, all layers), from the same data:

| | Median | 99th percentile | Within 0.02 | Within 0.001 |
|---|---|---|---|---|
| All decisions | 0.0050 | 0.0758 | 84.4% | 17.7% |
| Changed decisions | 0.0002 | 0.0043 | 100% | 83.9% |
| First flip per token | 0.0001 | 0.0013 | 100% | 97.8% |

The validator's near-tie window (0.02) includes 84% of all decisions, so passing it is weak evidence. The gaps themselves are strong evidence: changed decisions, and first flips especially, sit at far smaller gaps than decisions in general. A tighter window would make the check meaningful.
