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

## Continuous Integration

- **CPU tests run automatically** on every pull request and on pushes to `main` and `integration/phase-1-validation` (`.github/workflows/cpu-tests.yml`), with `pytest -m "not gpu"`.
- **GPU tests** carry the `gpu` marker (any test using the `qwen_model` fixture gets it automatically). They run on Colab with `pytest -m gpu` at the PR's final commit (#113).
- **Every PR fills in the acceptance-criteria table** in the PR template: each criterion, the test that proves it, and where it ran.
