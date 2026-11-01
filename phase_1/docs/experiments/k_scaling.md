# Does the prefetch buffer scale with k?

**Status:** preliminary (experiment branch `experiment/k-scaling`). Related: #114 (question 3), #110.

## Question

For Qwen (k = 4), k + 2 and 1.5k are both 6. Does the buffer stay roughly constant as k grows, or grow in proportion?

## Method

Each row stores all 60 expert probabilities, so a hypothetical top-k cut-off can be applied to the same routing. For each pair of orderings and each k, we find the smallest m such that each ordering's top-k is inside the other's top-(k+m).

Every k from 1 to 40 is measured, stopping well short of the 60-expert ceiling. m is reported as a fraction, interpolating where coverage of changed decisions crosses each target, because whole-number m can't separate candidate curves.

**The main limit:** this measures how crowded the ranking is around rank k. It doesn't simulate a model built with that k: hidden states and cascades still come from top-4 routing, so only k = 4 is a real measurement. It also covers batch effects only, not a predictor's error.

## Data

Trial collection at `38ef1da` (1,462,560 comparisons; fp16, batch 32, A100 40 GB; before the `position_ids` fix): [Drive, `trial_38ef1da`](https://drive.google.com/drive/folders/1F5ObENu9J_J5yH7ZIr-J4qMPwDqnyfyP?usp=drive_link).

```bash
python phase_1/scripts/k_scaling.py <collection output>/determinism
```

## Results

m95 is the buffer covering 95% of changed decisions. "Missed at m=2" is a share of all decisions.

```text
all layers:
   k  changed  median gap  m90.frac  m95.frac  m99.frac  (k+m95)/k  missed at m=2
   1   0.49%     0.05382      0.98      1.53      2.85       2.53        0.009%
   2   0.86%     0.02411      1.29      1.83      3.44       1.92        0.029%
   3   1.38%     0.01077      1.56      1.98      3.75       1.66        0.065%
   4   2.02%     0.00501      1.68      2.20      3.95       1.55        0.115%
   5   2.65%     0.00291      1.78      2.39      4.30       1.48        0.174%
   6   3.23%     0.00195      1.85      2.59      4.65       1.43        0.249%
   7   3.76%     0.00144      1.91      2.73      4.95       1.39        0.322%
   8   4.25%     0.00114      1.97      2.85      5.07       1.36        0.404%
   9   4.71%     0.00093      2.03      2.90      5.38       1.32        0.479%
  10   5.19%     0.00082      2.10      2.94      5.38       1.29        0.551%
  11   5.42%     0.00070      2.24      3.10      5.52       1.28        0.621%
  12   5.87%     0.00063      2.29      3.15      5.57       1.26        0.699%
  13   6.14%     0.00054      2.35      3.23      5.66       1.25        0.758%
  14   6.48%     0.00051      2.35      3.26      5.79       1.23        0.793%
  15   6.71%     0.00046      2.38      3.30      5.88       1.22        0.843%
  16   6.87%     0.00042      2.45      3.42      6.09       1.21        0.900%
  17   7.08%     0.00039      2.53      3.53      6.13       1.21        0.974%
  18   7.30%     0.00037      2.55      3.56      6.13       1.20        1.023%
  19   7.50%     0.00034      2.58      3.58      6.27       1.19        1.061%
  20   7.74%     0.00032      2.58      3.59      6.23       1.18        1.094%
  21   7.70%     0.00030      2.61      3.62      6.25       1.17        1.123%
  22   7.93%     0.00029      2.60      3.62      6.25       1.16        1.139%
  23   8.02%     0.00027      2.59      3.60      6.21       1.16        1.153%
  24   7.96%     0.00026      2.59      3.59      6.17       1.15        1.143%
  25   8.16%     0.00025      2.56      3.57      6.13       1.14        1.140%
  26   8.09%     0.00024      2.58      3.56      6.11       1.14        1.155%
  27   8.15%     0.00023      2.55      3.52      6.02       1.13        1.143%
  28   8.32%     0.00022      2.52      3.48      5.95       1.12        1.135%
  29   8.25%     0.00021      2.50      3.49      5.97       1.12        1.114%
  30   8.08%     0.00021      2.51      3.48      5.93       1.12        1.098%
  31   8.04%     0.00020      2.56      3.57      5.97       1.12        1.126%
  32   8.10%     0.00019      2.55      3.55      5.92       1.11        1.118%
  33   8.04%     0.00019      2.57      3.50      5.84       1.11        1.144%
  34   7.92%     0.00018      2.49      3.37      5.74       1.10        1.089%
  35   7.93%     0.00018      2.41      3.31      5.69       1.09        1.021%
  36   7.77%     0.00017      2.38      3.29      5.57       1.09        0.970%
  37   7.50%     0.00017      2.38      3.22      5.47       1.09        0.949%
  38   7.42%     0.00017      2.31      3.14      5.41       1.08        0.892%
  39   7.37%     0.00016      2.29      3.10      5.28       1.08        0.881%
  40   7.24%     0.00016      2.24      3.00      5.16       1.07        0.837%
  fit m90 k=1-40: linear: 1.82+0.023x R2=0.466 LOO=0.298  sqrt: 1.38+0.211x R2=0.641 LOO=0.247  log2: 1.18+0.276x R2=0.814 LOO=0.174
  fit m90 k=1-16: linear: 1.24+0.084x R2=0.914 LOO=0.147  sqrt: 0.69+0.454x R2=0.976 LOO=0.081  log2: 0.94+0.365x R2=0.989 LOO=0.050
  fit m95 k=1-40: linear: 2.55+0.029x R2=0.422 LOO=0.426  sqrt: 1.96+0.279x R2=0.598 LOO=0.357  log2: 1.69+0.369x R2=0.770 LOO=0.264
  fit m95 k=1-16: linear: 1.72+0.117x R2=0.936 LOO=0.168  sqrt: 0.97+0.629x R2=0.988 LOO=0.070  log2: 1.33+0.500x R2=0.978 LOO=0.119
  fit m99 k=1-40: linear: 4.68+0.040x R2=0.323 LOO=0.717  sqrt: 3.80+0.396x R2=0.497 LOO=0.622  log2: 3.34+0.544x R2=0.693 LOO=0.479
  fit m99 k=1-16: linear: 3.23+0.196x R2=0.926 LOO=0.302  sqrt: 1.96+1.055x R2=0.982 LOO=0.142  log2: 2.56+0.841x R2=0.978 LOO=0.185

layers 16-23:
   k  changed  median gap  m90.frac  m95.frac  m99.frac  (k+m95)/k  missed at m=2
   1   0.87%     0.07847      1.09      1.71      3.64       2.71        0.023%
   2   1.60%     0.03042      1.49      1.95      3.70       1.97        0.071%
   3   2.53%     0.01304      1.74      2.28      3.98       1.76        0.154%
   4   3.61%     0.00616      1.86      2.55      4.37       1.64        0.275%
   5   4.69%     0.00347      1.94      2.74      4.74       1.55        0.417%
   6   5.70%     0.00239      2.08      2.90      5.20       1.48        0.599%
   7   6.70%     0.00177      2.19      2.99      5.54       1.43        0.747%
   8   7.63%     0.00135      2.33      3.19      5.70       1.40        0.932%
   9   8.20%     0.00109      2.47      3.35      5.99       1.37        1.108%
  10   9.09%     0.00096      2.51      3.41      5.93       1.34        1.263%
  11   9.55%     0.00081      2.61      3.57      6.10       1.32        1.413%
  12  10.23%     0.00072      2.67      3.63      6.15       1.30        1.569%
  13  10.72%     0.00061      2.71      3.68      6.28       1.28        1.700%
  14  11.15%     0.00057      2.74      3.74      6.54       1.27        1.793%
  15  11.49%     0.00050      2.78      3.80      6.62       1.25        1.905%
  16  11.93%     0.00045      2.84      3.89      6.84       1.24        2.057%
  17  12.17%     0.00042      2.91      3.98      6.83       1.23        2.210%
  18  12.52%     0.00040      2.93      4.05      6.88       1.22        2.330%
  19  12.81%     0.00037      2.95      4.04      6.87       1.21        2.400%
  20  13.12%     0.00035      2.96      4.05      6.85       1.20        2.472%
  21  13.08%     0.00033      2.96      4.12      6.91       1.20        2.510%
  22  13.26%     0.00031      2.99      4.14      6.97       1.19        2.555%
  23  13.57%     0.00029      2.97      4.06      6.93       1.18        2.586%
  24  13.55%     0.00028      2.96      4.06      6.88       1.17        2.581%
  25  13.76%     0.00026      2.96      4.08      6.83       1.16        2.593%
  26  13.58%     0.00025      2.96      4.02      6.88       1.15        2.583%
  27  13.63%     0.00024      2.93      3.97      6.78       1.15        2.564%
  28  13.76%     0.00023      2.91      3.95      6.75       1.14        2.551%
  29  13.89%     0.00022      2.88      3.95      6.73       1.14        2.502%
  30  13.40%     0.00021      2.92      3.96      6.74       1.13        2.463%
  31  13.56%     0.00020      2.96      4.14      6.81       1.13        2.556%
  32  13.53%     0.00019      2.97      4.07      6.69       1.13        2.552%
  33  13.41%     0.00019      2.95      3.94      6.61       1.12        2.570%
  34  13.36%     0.00018      2.85      3.83      6.41       1.11        2.442%
  35  13.30%     0.00017      2.81      3.81      6.37       1.11        2.317%
  36  13.10%     0.00017      2.79      3.77      6.23       1.10        2.183%
  37  12.62%     0.00017      2.78      3.76      6.20       1.10        2.146%
  38  12.55%     0.00016      2.75      3.71      6.00       1.10        2.053%
  39  12.39%     0.00015      2.74      3.67      5.93       1.09        2.049%
  40  12.16%     0.00015      2.68      3.56      5.80       1.09        1.926%
  fit m90 k=1-40: linear: 2.07+0.028x R2=0.501 LOO=0.342  sqrt: 1.54+0.256x R2=0.677 LOO=0.278  log2: 1.31+0.334x R2=0.849 LOO=0.187
  fit m90 k=1-16: linear: 1.37+0.104x R2=0.920 LOO=0.173  sqrt: 0.69+0.561x R2=0.981 LOO=0.087  log2: 1.01+0.450x R2=0.988 LOO=0.067
  fit m95 k=1-40: linear: 2.87+0.036x R2=0.470 LOO=0.473  sqrt: 2.16+0.336x R2=0.646 LOO=0.389  log2: 1.85+0.439x R2=0.815 LOO=0.275
  fit m95 k=1-16: linear: 1.91+0.139x R2=0.937 LOO=0.199  sqrt: 1.01+0.746x R2=0.990 LOO=0.080  log2: 1.45+0.592x R2=0.979 LOO=0.142
  fit m99 k=1-40: linear: 5.20+0.046x R2=0.351 LOO=0.781  sqrt: 4.22+0.452x R2=0.524 LOO=0.670  log2: 3.73+0.608x R2=0.702 LOO=0.522
  fit m99 k=1-16: linear: 3.62+0.216x R2=0.941 LOO=0.285  sqrt: 2.25+1.153x R2=0.975 LOO=0.193  log2: 2.96+0.903x R2=0.937 LOO=0.382

```

Check: k = 4 reproduces the 2.02% change rate from `batch_variance.py`, and its m95 of 2.20 agrees with 94.5% coverage at m = 2.

## Reading the results

- **Neither simple rule fits.** m95 is 1.53 at k = 1 and 3.42 at k = 16, far below proportional growth, and (k+m95)/k falls from 2.53 to 1.21. The 1.5 ratio at k = 4 (1.55) is a coincidence of that point.
- **For k = 1 to 16, a square-root curve describes it best:** m95 ≈ 0.97 + 0.63√k (R² 0.988, leave-one-out error 0.07; logarithmic 0.12, linear 0.17). Square root also wins for m99 and in layers 16-23. It's a description of this range, chosen after seeing the data, not a law.
- **The curve doesn't continue.** m95 levels off at about 3.6 around k = 20 to 24, then falls to 3.0 at k = 40; the change rate peaks near 8% at k = 30. No single form fits k = 1 to 40. Part of this may be structural: choosing the top k of 60 also chooses the bottom 60 − k, so mid-ranking cut-offs may be where sets change most. This hasn't been tested.
- **Ranking crowding isn't the whole explanation.** At k = 4, layers 16-23 have a *larger* median gap than all layers (0.0062 against 0.0050) but change about 1.8 times as often. Disturbances also grow with depth, which fits the cascade finding.
- **For Qwen (k = 4):** m90 = 1.68, m95 = 2.20, m99 = 3.95; in layers 16-23, 1.86, 2.55 and 4.37. So m = 2 falls just short of 95% of changed decisions from batch effects alone, and doesn't include predictor error.

## Are changes really at near-ties?

The gap between the 4th and 5th expert (k = 4, all layers), from the same data:

| | Median | 99th percentile | Within 0.02 | Within 0.001 |
|---|---|---|---|---|
| All decisions | 0.0050 | 0.0758 | 84.4% | 17.7% |
| Changed decisions | 0.0002 | 0.0043 | 100% | 83.9% |
| First flip per token | 0.0001 | 0.0013 | 100% | 97.8% |

The validator's near-tie window (0.02) includes 84% of all decisions, so passing it is weak evidence. The gaps themselves are strong evidence: changed decisions, and first flips especially, sit at far smaller gaps than decisions in general. A tighter window would make the check meaningful.
