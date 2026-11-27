# Contributing to MOE Expert Prefetching

## PR Acceptance Criteria

### Acceptance Criteria Must Be PROVEN, Not Stated

**Do not claim "Acceptance Criteria Met" in the PR unless you have:**

1. **Actually run the acceptance tests** — Run pytest on all test files related to your feature
2. **Included test output** — Paste the passing test results in the PR description
3. **Verified integration** — Test that your changes work with dependent code

### Example PR Description Format

```markdown
## Summary
[description of changes]

## Acceptance Criteria

### Tests Run and Passing
```
pytest phase_1/scripts/test_forward_pass.py -v
================ 6 passed in 2.34s ================
```

### Tests Verified
- ✅ test_hook_registration_and_firing() 
- ✅ test_routing_data_format()
- ✅ test_token_layer_routing_extraction()
- ✅ test_no_memory_leaks()
- ✅ test_end_to_end_pipeline()
- ✅ test_integration_with_tokenizer()

### Code Review Checklist
- [ ] All acceptance tests run and pass
- [ ] No new warnings or errors from linters
- [ ] Documentation updated if needed
- [ ] Code follows project style
```

## Testing Requirements

**Before submitting a PR:**

1. Run all relevant tests: `pytest phase_1/scripts/test_*.py`
2. Check that tests actually pass (don't assume)
3. Include test output in PR description as proof
4. If tests don't exist, write them first

## Lesson Learned

**Issue**: Forward Pass PR (#58) claimed "Acceptance Criteria Met" but tests were never run. This allowed the `device` parameter bug to slip through.

**Prevention**: Acceptance criteria must be PROVEN with actual test execution, not just written and assumed to pass.

## Tests Must Not Be Weakened to Pass

**Do not loosen a test's assertion to make it pass** (for example, changing `== 4` to `>= 2`, widening a tolerance, or replacing a value check with a presence check) unless the PR states why the old assertion was wrong.

**Why:** in Phase 1, the router hook test was loosened until it accepted top-2 experts (`4763418`), while the model routes top-4 (#100). Other tests checked only that data existed (`routing_events > 0`), and three bugs that corrupted the data passed every test (#99–#101).

**What a test should do:** compare against an independent reference, not the code's own output. Examples: running each prompt alone versus batched (`test_extraction_rows.py`), or the model's own router logits versus the hook's selection (`test_router_topk.py`).

## Failure Behaviour

Code is usually designed and tested for when things go right. In Phase 1, reviews kept finding what happens when they go wrong: things left behind after a failure (#58, #59, #118, #121), and failures that passed silently (#111, #59). Waleed and Dvora found most of them by forcing a failure and looking at what was left. These rules apply to all new and changed code (#131).

1. **Never fail silently.** Don't drop or skip data, don't `except: pass`, and don't log an error and carry on. Raise, or record an error that makes the result fail validation.
   - *Example:* #59 silently dropped invalid routing rows: with 4 bad rows injected, 20 of 24 rows were saved and the run reported success. Bad rows now fail the pass.
   - **A check that couldn't run is not a pass.** If a check is missing what it needs, report that it didn't run. #60's validator says "not run" when there's no repeated ordering or no window record, and its report is then "incomplete", never "passed". #129's startup check stops pytest when the installed packages don't match `requirements.txt`, rather than running the tests anyway.
2. **Check completeness against what should exist,** not against what arrived. Count the rows you expect, then compare.
   - *Example:* #59 checks the rows written against the tokenizer's token count × the model's layers, so missing rows fail (`test_missing_rows_fail_the_pass`). Before #60 (reported in #105), the determinism check passed when it had fewer than two runs to compare.
3. **Release what you acquire on every exit path:** hooks, files, GPU memory and temporary state. Use `with` or `try`/`finally`. Cleanup must not hide the original error: if cleanup also fails, log that and re-raise the first error.
   - *Example:* `a6d642a` (#59) removes the router hooks even when the collection fails partway. Before it, a failed dataset load left 2 hooks attached to the model.
4. **Test the failure.** Every PR names its likely failure modes, and at least one test forces a failure and checks what's left: data, state, hooks, the error raised.
   - *Example:* `test_hooks_are_removed_when_dataset_loading_fails` forces the failure Waleed found and counts the hooks left. It fails on the code before `a6d642a`.

**Deliberate exceptions** need their reason in the code, on the line. For example, cleanup that logs a second error so it doesn't hide the first one.

**Enforcement:** the PR template asks for the failure modes and the test that forces one. Reviewers try at least one forced failure. Once #123 and #125 land, CI flags the common silent-failure patterns (`except: pass`, catching every exception, logging without the traceback).

## Continuous Integration

- **CPU tests run automatically** on every pull request and on pushes to `main` and `integration/phase-1-validation` (`.github/workflows/cpu-tests.yml`), with `pytest -m "not gpu"` over the whole repository.
- **GPU tests** carry the `gpu` marker (any test using the `qwen_model` fixture gets it automatically). Load the model only through that fixture: without a GPU, `ModelLoader.load()` refuses to start, so a test that loads it directly fails in CI. They run on Colab with `pytest -m gpu` at the PR's final commit (#113).
- **A dependency change is a GPU-path change.** A PR that changes `requirements.txt` or a lock file runs the GPU tests at its final commit, even if no model code changed. CI runs only on the CPU, and a version change can change routing results while every CPU check passes: moving transformers from 4.45 to 5.23 changed about 2% of routing decisions (#126). #123 will make CI enforce this.
- **If the CPU check fails for a reason that isn't your code,** check whether the Hugging Face Hub is down: the tests download datasets and the tokenizer from it. Say so on the PR. Re-run the check once the Hub is back, or ask an admin to merge past it, noting why on the PR.
- **Every PR fills in the acceptance-criteria table** in the PR template: each criterion, the test that proves it, and where it ran.
