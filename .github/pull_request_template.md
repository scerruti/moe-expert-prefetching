## What this changes and why

<!-- One or two sentences. Link the issue(s): "Refs #NN" while work is in progress, "Fixes #NN" when this PR resolves it. -->

## Acceptance criteria and proof

Each criterion needs a test that **checks values**, not just that output exists, and a record of where it ran.

| Acceptance criterion | Test that proves it | Where it ran |
|---|---|---|
| <!-- e.g. Every row records 4 selected experts --> | <!-- e.g. `test_router_topk.py::test_selected_experts_has_qwen_top_k` --> | <!-- CPU CI / `gpu-tests` on Colab at <commit> --> |

## Reviewer guide

<!-- For a 30-minute walkthrough: where to start, what to look at closely, what can be skimmed, and any decision you need from the reviewer. -->

## Failure behaviour

How can this change fail, and what happens when it does? It must fail loudly, release what it acquired, and never report a pass for a check that couldn't run (CONTRIBUTING.md, "Failure Behaviour").

| Failure mode | What happens | Test that forces it |
|---|---|---|
| <!-- e.g. the dataset fails to load partway --> | <!-- e.g. hooks removed, the original error raised --> | <!-- e.g. `test_collection_scope.py::test_hooks_are_removed_when_dataset_loading_fails` --> |

<!-- Documentation-only PR? That means Markdown or other prose files only: no code, tests, notebooks, CI workflows, dependency files or configuration. Write one row: "None: documentation only". -->

## Checklist

- [ ] CPU tests pass in CI.
- [ ] GPU tests (`pytest -m gpu`) ran on Colab at this PR's final commit, if this PR changes GPU code paths or dependencies (`requirements.txt` or a lock file).
- [ ] At least one test forces a failure and checks what's left (CONTRIBUTING.md), or not applicable because the PR is documentation only.
- [ ] No test assertion was loosened to make it pass, or the reason is explained above (CONTRIBUTING.md).
