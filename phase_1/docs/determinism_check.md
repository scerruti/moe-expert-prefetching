# Determinism check: what it requires and why

**Status:** decided November 2026. Implementation pending in #60. Discussion: #114, #105, #128.

## The problem

Phase 1 records which experts the router picks for every token at every layer. On Qwen1.5-MoE-A2.7B, running the same prompts in different batch orders changes about **2% of those choices** (#114). The cause is the GPU's arithmetic: results round slightly differently depending on the shape of each batch. The original check required every probability to match within 0.01 across orderings. It fails on correct data, because once one choice changes, later choices can legitimately differ by up to 0.14.

## Decisions

1. **Accept the batch differences.** Taylor's view: this is a common problem in ML research and not a concern. We document it, and we don't collect an extra run with one prompt at a time.
2. **Same batch, same result.** Running an identical batch twice must give bitwise-identical routing. This is already shown (#114).
3. **Drop the 0.01 rule.** Real runs break it for legitimate reasons.
4. **Keep a near-tie check across batch orders, as a diagnostic.** For each **root change** (defined below), the gap between the router's kth and (k+1)th experts must be within the model's **near-tie window**, where k is the model's number of experts per token (k = 4 for Qwen1.5). Use the smaller of the two orderings' gaps. A root change outside the window is a **warning** to investigate, not proof of a bug.
5. **The window is computed for each model** by a fixed procedure (below), implemented in #60. Nobody picks the number.
6. **Report the change statistics with every result:** how many decisions changed, how many were root changes, and their gaps.

## Root changes

A change is a **root change** if, in the same sequence, nothing changed at a lower layer on the same token or any earlier token. Every other change is a **knock-on**.

Why not just each token's first changed layer, as the first draft said? Waleed pointed out (#128) that a change can travel *between* tokens: the model's attention lets every later token read earlier tokens, so a change at token 3, layer 5 alters the hidden state of token 10 at layer 6 and above. Token 10's first change can then be a knock-on, with a gap far from a near-tie. The trial data confirms it: **76.5%** of first changes per token (11,174 of 14,603) came after an earlier token had already changed at a lower layer.

A root change has no earlier change that could explain it, so rounding is the only expected cause. That's what the near-tie check tests.

**Credit:** the cross-token mechanism, and the point that small differences grow with depth, are Waleed's (#128).

## Why keep the near-tie check (decision 4)

- **It separates noise from error.** Rounding can only swap two experts whose scores were already nearly equal. A root change between experts that weren't close needs another cause, such as a bug. The check can fail, so it tests the explanation rather than describing the noise. It would have caught the wrong-rows bug (#99).
- **It supports a methods-section claim:** "All root changes occurred where the top-k boundary was within the computed window, consistent with floating-point rounding." The claim is backed by a check, not asserted.
- **The evidence supports it.** In the trial, changed decisions sat at tiny gaps, often exact ties, and the differences came in whole fp16 rounding steps (#114).
- **It protects future runs.** New hardware, the transformers upgrade (#124) and other models (#112) could each introduce a bug that looks like noise.
- **It costs almost nothing:** a comparison over data we already collect.

**Why a warning and not a failure:** the check is empirical. It shows root changes behave like rounding; it doesn't prove each one is rounding. A rare out-of-window root change could still be noise, for example from rounding inside the attention calculation that the router gap doesn't capture. So it flags the run for a person to look at, and the run's report says how many warnings there were.

**The alternative considered:** check only that identical batches match, and report the cross-batch changes with no pass or fail. That's consistent with Taylor's view too, but it can't tell noise from a bug.

## Why root changes only

Knock-on changes come from a changed hidden state, not from rounding at that point, so their gaps can be large. In the trial collection (`38ef1da`):

| Changes counted | Number | Share within 0.001 | Largest gap |
|---|---|---|---|
| Root changes | 3,429 | **100%** | **0.00035** |
| First change per token | 14,603 | 97.85% | 0.00446 |

Checking every change, or every first change, would raise warnings for legitimate reasons. Checking root changes tests exactly the claim that the *cause* is rounding. Knock-ons are reported, with no pass or fail.

## How the window is computed, for any model

1. **Pilot run.** Use the same settings as the real collection: the same software versions, hardware, precision, batch size and batch orderings, on a separate set of prompts.
2. **Find the root changes.** Keep adding pilot prompts until there are **at least 99** root changes.
3. **Window = 3 × the largest root-change gap** in the pilot.
4. **Record** the window, the pilot run it came from, and the root-change count.
5. **Apply it to the real collection,** never to the pilot that set it. Root changes outside the window are warnings. Knock-ons are reported.

**Where 99 comes from.** The project's target is that, if the real collection behaves like the pilot, fewer than **1%** of its root changes should fall outside the pilot's largest gap. If the pilot has n root changes, the chance a new one beats all of them is about 1 in (n + 1). For 1%, n + 1 ≥ 100, so n ≥ 99. The 1% target is a project-wide choice; the count follows from it, and the same rule applies to every model.

**Where 3 comes from.** The pilot uses different prompts from the real collection and may miss the rarest near-ties. A 3× margin covers that, while staying far below the gaps of real errors. It's a project-wide choice, set once and not tuned per model.

**Qwen1.5's window.** Using the trial collection as its pilot: 3,429 root changes (well above 99), largest gap 0.00035036, so the window is **3 × 0.00035 ≈ 0.00105**. This was computed by hand, with a throwaway analysis script; #60 adds the code that computes it, and a test that reproduces these figures from the trial data. It applies to the official Phase 1 collection. If the official collection's settings differ from the trial's (for example after the transformers upgrade, #124), a new pilot recomputes it.

## Limitations

- **Empirical, not proof.** The check shows root changes behave like rounding. A warning means "look at this", not "this is a bug".
- **Root changes are an approximation.** Calling a change a root change assumes nothing earlier explains it. Attention-layer rounding can still differ between orderings without changing any router choice, so a few root changes may be partly knock-on. This is why the margin exists and why the result is a warning.
- **Noise grows with depth.** Small differences compound through the layers (Waleed, #128), so deeper root changes may sit at somewhat larger gaps. The window is taken over all layers; if warnings cluster in the deepest layers, a per-layer window is the next step.
- **Calibration.** Qwen's window comes from the trial data. It's honest only if the official collection, a separate run, is what gets checked.
- **Units.** Probability gaps depend on how many experts a model has, so a window doesn't transfer between models. That's why each model computes its own. For other models (#112), measuring the gap in the router's raw scores, or in fp16 rounding steps, may make windows easier to compare.
- **Gap definition matters.** Using one ordering's gap instead of the smaller of the two changes the results. The check must state its definition.
- **Trial data** predates the `position_ids` fix, so the official pilot re-establishes these numbers.

## Evidence

- #114: the batch-sensitivity finding, near-tie and rounding-step analysis, and the control run showing identical batches match bitwise.
- #128: Waleed's review, which found the cross-token knock-ons and the depth effect.
- `phase_1/docs/experiments/k_scaling.md` (branch `experiment/k-scaling`): the near-tie table and gap definitions.
- #124: the transformers 5.23 pre-test, where changes across versions match batch-order changes (about 2%).
- The root-change figures above come from `data/phase_1_trial_38ef1da/determinism`, comparing every pair of orderings. They were computed by hand; until #60 lands, they can't yet be reproduced from code in the repo.
