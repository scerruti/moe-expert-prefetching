## What this changes and why

<!-- One or two sentences. Link the issue(s): "Refs #NN" while work is in progress, "Fixes #NN" when this PR resolves it. -->

## Acceptance criteria and proof

Each criterion needs a test that **checks values**, not just that output exists, and a record of where it ran.

| Acceptance criterion | Test that proves it | Where it ran |
|---|---|---|
| <!-- e.g. Every row records 4 selected experts --> | <!-- e.g. `test_router_topk.py::test_selected_experts_has_qwen_top_k` --> | <!-- CPU CI / `gpu-tests` on Colab at <commit> --> |

## Reviewer guide

<!-- For a 30-minute walkthrough: where to start, what to look at closely, what can be skimmed, and any decision you need from the reviewer. -->

## Checklist

- [ ] CPU tests pass in CI.
- [ ] GPU tests (`pytest -m gpu`) ran on Colab at this PR's final commit, if this PR changes GPU code paths.
- [ ] No test assertion was loosened to make it pass, or the reason is explained above (CONTRIBUTING.md).
