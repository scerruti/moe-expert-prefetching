# Determinism check: what it requires and why

**Status:** decided November 2026. Implemented in #60. Discussion: #114, #105.

## The problem

Phase 1 records which experts Qwen1.5-MoE-A2.7B's router picks for every token at every layer. Running the same prompts in different batch orders changes about **2% of those choices** (#114). The cause is the GPU's arithmetic: results round slightly differently depending on the shape of each batch. The original check required every probability to match within 0.01 across orderings. It fails on correct data, because once a token's experts change at one layer, its later layers can legitimately differ by up to 0.14.

## Decisions

1. **Accept the batch differences.** Taylor's view: this is a common problem in ML research and not a concern. We document it, and we don't collect an extra run with one prompt at a time.
2. **Same batch, same result.** Running an identical batch twice must give bitwise-identical routing. This is already shown (#114).
3. **Drop the 0.01 rule.** Real runs break it for legitimate reasons.
4. **Keep a near-tie check across batch orders.** For each token, find the **first layer where its experts changed**. The 4th and 5th experts' probabilities there must be within the **near-tie window**: 0.005 for Qwen1.5. Use the smaller of the two orderings' gaps.
5. **The window is a per-model setting,** worked out from a pilot run for each model (see below).
6. **Report the change statistics with every result:** how many decisions changed, and their gaps.

## Why keep the near-tie check (decision 4)

- **It separates noise from error.** Rounding can only swap two experts whose scores were already nearly equal. A swap between experts that weren't close needs another cause, such as a bug. The check can fail, so it tests the explanation rather than describing the noise. It would have caught the wrong-rows bug (#99).
- **It supports a methods-section claim:** "All first changes occurred where the top-4 boundary was within 0.005, consistent with floating-point rounding." The claim is backed by a check, not asserted.
- **The evidence supports it.** In the trial, changed decisions sat at tiny gaps, often exact ties, and the differences came in whole fp16 rounding steps (#114).
- **It protects future runs.** New hardware, the transformers upgrade (#124) and other models (#112) could each introduce a bug that looks like noise.
- **It costs almost nothing:** a comparison over data we already collect.

**The alternative considered:** check only that identical batches match, and report the cross-batch changes with no pass or fail. That's consistent with Taylor's view too, but it can't tell noise from a bug.

## Why first changes only

Once a token's experts change, its hidden state changes, so its later layers can differ by much more than rounding. These are cascades (#114). In the trial:

| Within the window of 0.005 | Share |
|---|---|
| First change per token | **100%** |
| All changes | 99.4% (the rest are cascades) |
| All decisions, changed or not | 50.8% |

Checking every change would fail for a legitimate reason. Checking first changes tests exactly the claim that the *cause* is rounding.

## How the window is set, for any model

1. **Pilot run:** a few batch orderings of a small prompt subset, for that model.
2. **Measure:** for each token's first change, the gap between the 4th and 5th experts.
3. **Set the window** just above the largest pilot gap, with a margin. Record the value and the pilot run it came from.
4. **Apply it to the real collection,** never to the pilot that set it.

Qwen1.5's 0.005 comes from the trial collection at `38ef1da`, and is applied to the official collection.

## Limitations

- **Calibration:** Qwen's window was chosen from the trial data. It's honest only if the official collection, a separate run, is what gets checked.
- **Units:** probability gaps depend on how many experts a model has, so a fixed probability window doesn't transfer between models. Phase 1 uses probability units, as the trial measured. For other models (#112), defining the window in the router's raw scores, or in fp16 rounding steps, would measure closeness the same way everywhere.
- **Gap definition matters:** using one ordering's gap instead of the smaller of the two puts only 96.4% of changes within 0.005. The check must state its definition.
- **Trial data:** it predates the `position_ids` fix, so the official collection re-establishes these numbers.

## Evidence

- #114: the batch-sensitivity finding, near-tie and rounding-step analysis, and the control run showing identical batches match bitwise.
- `phase_1/docs/experiments/k_scaling.md` (branch `experiment/k-scaling`): the near-tie table and gap definitions.
- #124: the transformers 5.23 pre-test, where changes across versions match batch-order changes (about 2%).
