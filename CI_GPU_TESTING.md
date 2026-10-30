# GPU Testing in CI/CD

Design and implementation plan for automated GPU testing in GitHub Actions.

## Current State

**CPU Tests (Fast, Every PR):**
- 30 tests on `ubuntu-latest`
- ~2 minutes
- No cost
- Validates: data collection, determinism, basic pipeline logic

**GPU Tests (Expensive, Manual):**
- 70 tests on Colab A100 40GB
- ~7 minutes
- ~$0.40 per run (at $8.88/hour)
- Validates: full pipeline with real model, tokenization, hooks, forward pass

## Proposed Strategy

### Option A: GitHub Actions with GPU Runner (Recommended for MVP)

**When:** On merge to `main` only (not every PR, too expensive)

**Setup:**
```yaml
# .github/workflows/gpu-tests.yml
name: GPU Tests (A100)
on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  gpu-tests:
    runs-on: [self-hosted, gpu, a100]  # Self-hosted runner with A100
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v3
      - name: Run GPU tests
        run: |
          python -m pytest phase_1/scripts/test_*.py \
            -v --timeout=300 \
            --tb=short \
            --junit-xml=gpu-test-results.xml
      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: gpu-test-results
          path: gpu-test-results.xml
```

**Requirements:**
- Self-hosted runner on A100 (AWS, GCP, or on-premise)
- Runner registration with GitHub
- Cost: $0.40 per run (7 min @ $8.88/hr)

### Option B: Colab Integration (Current Workflow)

**When:** Manual trigger, or scheduled weekly

**Current Process:**
1. User runs `bash run_gpu_tests.sh` on Colab A100
2. Results saved to Google Drive
3. User manually verifies and reports

**Challenges:**
- Manual process
- No automatic reporting to GitHub
- Hard to track history
- Depends on manual trigger

### Option C: Hybrid (Best for Phase 1)

**Phase 1 Approach:**
- CPU tests: Auto-run on every PR (GitHub Actions free tier)
- GPU tests: Manual + optional scheduled weekly

**Benefits:**
- Fast feedback on every PR (CPU tests pass/fail)
- GPU tests only when needed
- Minimal cost
- Simple setup

## Acceptance Criteria for GPU Tests

**Success Metrics:**
```
PASS:  36-40 tests pass (data collection, pipeline, tokenizer)
SKIP:  5 tests skip (CPU detection on GPU runner - expected)
FAIL:  0-2 tests fail (acceptable: hook architecture issues, Issue #90)
```

**Blocking Failures:**
- test_full_pipeline.py fails (core pipeline broken)
- test_data_collection.py fails (data flow broken)
- CUDA OOM (memory management broken)

**Non-Blocking Failures:**
- test_pytorch_forward_hooks.py fails (architectural mismatch, Issue #90)
- Hook output format issues (data still collected, just validation off)

## GitHub Actions Workflow

### CPU Tests (Every PR)

```yaml
name: Acceptance Tests (CPU)
on: [pull_request, push]

jobs:
  cpu-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: |
          python -m pytest phase_1/scripts/test_*.py \
            -v --tb=short \
            -k "not (test_model_loading or test_forward_pass or test_pytorch_forward_hooks)"
```

**What it tests:** Data collection, determinism, row validation
**Time:** ~2 minutes
**Cost:** Free (GitHub Actions free tier)

### GPU Tests (Manual or Scheduled)

**Option 1: Manual trigger**
```yaml
name: Acceptance Tests (GPU A100)
on: workflow_dispatch

jobs:
  gpu-tests:
    runs-on: [self-hosted, gpu, a100]
    # ... run full test suite ...
```

**Option 2: Scheduled weekly**
```yaml
on:
  schedule:
    - cron: '0 2 * * 1'  # Monday 2am UTC
```

## Implementation Roadmap

**Phase 1 (Current - Manual):**
- ✅ CPU tests auto-run on PR
- ✅ GPU tests documented (GPU_TESTING.md)
- ✅ Test fixtures for memory management
- 🟡 GPU tests run manually on Colab

**Phase 1.5 (Week of Nov 1):**
- [ ] Set up self-hosted runner with A100
- [ ] Add GPU test workflow to GitHub Actions
- [ ] Configure result reporting
- [ ] Document for team

**Phase 2 (Post-Phase-1):**
- [ ] Cost optimization (spot instances, shared runner)
- [ ] Multiple GPU types (A100, H100, RTX)
- [ ] Performance regression detection
- [ ] Model update validation

## Cost Analysis

**Current (Manual Colab):**
- ~$0.40 per run (7 min @ $8.88/hr)
- Variable (depends on when user runs)
- Human effort to verify results

**GitHub Actions (Self-Hosted A100):**
- ~$0.40 per run (7 min @ $8.88/hr)
- Weekly = ~$2/week = ~$8/month
- Automated reporting and history

**Cost Reduction Strategies:**
1. Run only on main/release (not every PR)
2. Use spot instances (70% cheaper)
3. Share runner with other projects
4. Cache model weights

## Reporting & Monitoring

**What to Track:**
- Pass/fail rate over time
- Flaky tests (pass sometimes, fail others)
- Performance regressions
- Memory usage trends

**Where to Report:**
- GitHub Actions Artifacts (test XML)
- GitHub Checks API (pass/fail status on commit)
- Optional: Slack notification on failure
- Optional: Dashboard (GitHub Pages)

## Security Considerations

**Self-Hosted Runner:**
- Runner has access to GitHub secrets
- Can read private data
- Must be on trusted infrastructure
- Consider: Ephemeral runners (spin up/down per run)

**Sensitive Data:**
- No API keys in logs
- No model weights in artifacts
- Drive access (Colab): needs separate auth

## Next Steps

1. **Decide:** Which option (A, B, or C)?
2. **Setup:** Install and register self-hosted runner (if A or C)
3. **Test:** Verify workflow passes with known good commit
4. **Document:** Add to project setup guide
5. **Monitor:** Track costs and reliability

## See Also

- [GPU_TESTING.md](GPU_TESTING.md) — Manual GPU testing on Colab
- [Issue #81](https://github.com/scerruti/moe-expert-prefetching/issues/81) — CI/CD setup proposal
- [PHASE_1_MODEL_DECISION.md](PHASE_1_MODEL_DECISION.md) — Model selection
