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

