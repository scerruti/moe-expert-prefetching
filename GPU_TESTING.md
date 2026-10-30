# GPU Testing Guide for Phase 1

This document describes how to run and validate acceptance tests on GPU (Colab A100).

## Quick Start

**In Colab notebook cell:**
```python
# Clone repo and pull latest
!git clone https://github.com/scerruti/moe-expert-prefetching.git /content/moe
%cd /content/moe
!git checkout integration/phase-1-validation

# Run all tests with output saved to Drive
!bash run_gpu_tests.sh
```

Or specify a custom output directory:
```python
!bash run_gpu_tests.sh /content/drive/MyDrive/my-test-results
```

## Test Execution

### What Runs

**70 total tests** across 6 test files:

| File | Tests | GPU Required | Status |
|------|-------|--------------|--------|
| test_data_collection.py | 15 | ❌ No | Always run |
| test_determinism_validation.py | 9 | ❌ No | Always run |
| test_full_pipeline.py | 9 | ✅ Yes | Skip on CPU, run on GPU |
| test_forward_pass.py | 7 | ✅ Yes | Skip on CPU, run on GPU |
| test_pytorch_forward_hooks.py | 14 | ✅ Yes | Skip on CPU, run on GPU |
| test_tokenizer.py | 16 | ✅ Yes | Skip on CPU, run on GPU |

### Expected Results on A100

**Colab A100 (40GB):**
- ✅ **36-40 tests pass** — data collection, determinism, full pipeline
- ⏭️ **5 tests skip** — CPU-only detection (expected, not failures)
- ❌ **0-5 tests fail** — hook tests depend on model architecture

**Colab A40 (24GB):**
- May hit CUDA OOM on tokenizer tests
- Use `run_gpu_tests.sh` which handles cleanup between tests

**Local CPU:**
```bash
bash run_tests.sh  # Runs only CPU-safe tests (30 pass)
```

## Expected Failures & Workarounds

### Issue 1: Determinism Validation Tests Fail

**Symptom:**
```
FAILED test_determinism_validation.py::test_data_quality_check_valid
TypeError: DataQualityCheckResult.__init__() missing 1 required positional argument: 'passed'
```

**Cause:** Code didn't pull fixes from remote

**Fix:**
```bash
git pull origin integration/phase-1-validation
```

### Issue 2: Hook Tests Fail (Expected)

**Symptom:**
```
FAILED test_pytorch_forward_hooks.py::TestRouterHooks::test_hooks_fire_on_all_layers
ValueError: not enough values to unpack (expected 3, got 2)
```

**Cause:** Hook output format doesn't match model expectations  
**Status:** Known issue, Issue #90 (architecture mismatch)  
**Impact:** Non-critical — forward pass works correctly in full pipeline

### Issue 3: CUDA Out of Memory (Fixed)

**Symptom:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.00 MiB.
GPU 0 has a total capacity of 39.49 GiB of which 2.62 MiB is free.
```

**Cause:** Tests load model repeatedly without cleanup  
**Fix:** Fixture in `conftest.py` cleans up after each test (commit 409bc3f)  
**Result:** Should not occur with current code

## Output Interpretation

### Test Summary Line

```
======================== 36 passed, 5 skipped in 423.47s (0:07:03) =============
```

| Part | Meaning |
|------|---------|
| 36 passed | Tests that ran and succeeded |
| 5 skipped | GPU tests on CPU (expected with skipif) |
| 423.47s | Total runtime |

### Success Criteria for Phase 1

✅ **Phase 1 is validated if:**
1. `test_full_pipeline.py` passes (9 tests)
2. `test_data_collection.py` passes (15 tests)
3. `test_tokenizer.py` passes (11+ tests — 5 per model load cycle)
4. No CUDA OOM errors

🟡 **Non-blocking issues:**
- Hook tests fail (doesn't affect production pipeline)
- Some tokenizer tests fail (depends on GPU memory)

## Results File

When using `run_gpu_tests.sh`, results are saved to Google Drive:

**Default location:**
```
/content/drive/MyDrive/moe-test-results/acceptance_tests_YYYYMMDD_HHMMSS.txt
```

**Custom location:**
```bash
bash run_gpu_tests.sh /content/drive/MyDrive/phase-1-results
```

### Viewing Results

**In Colab:**
```python
# Read latest results
!ls -lh /content/drive/MyDrive/moe-test-results/ | tail -5
!tail -100 /content/drive/MyDrive/moe-test-results/acceptance_tests_*.txt
```

**Download from Google Drive:**
- Open https://drive.google.com
- Navigate to `moe-test-results` folder
- Right-click file → Download

## Memory Management

### GPU Memory Profile

**A100 40GB:**
- Initial free: ~35GB
- After model load (4-bit): ~8GB free
- After cleanup: ~33GB free (recovered)

**Why tests were failing (before fix):**
- Each test loaded model (~27GB 4-bit)
- Models weren't deleted between tests
- After 10+ tests: no free memory left
- Solution: `conftest.py` fixture cleans up after each test

### Fixture Cleanup

The `qwen_model` fixture (in `conftest.py`) automatically:
```python
@pytest.fixture(scope="function")
def qwen_model():
    model, tokenizer, config = loader.load()
    yield model, tokenizer, config
    del model  # Free memory
    torch.cuda.empty_cache()  # Return to GPU
```

Each test gets:
- Fresh model load
- Automatic cleanup
- Clean GPU state for next test

## Troubleshooting

### Problem: "No module named 'moe_expert_prefetching'"

**Solution:**
```bash
cd /content/moe
python -m pytest phase_1/scripts/test_*.py -v
```

### Problem: Tests timeout or hang

**Solution:** Use timeout wrapper
```bash
pip install pytest-timeout
python -m pytest phase_1/scripts/test_*.py --timeout=300 -v
```

### Problem: Still getting CUDA OOM despite fixture

**Check:** Model really being deleted
```python
# In a test
def test_check_cleanup(qwen_model):
    import torch
    model, tokenizer, config = qwen_model
    print(f"Before cleanup: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    # Test runs
    # After test: fixture cleanup should free memory
```

## CI/CD Integration

**Future work:** Set up GitHub Actions to run:
- CPU tests on every PR (fast, 2 min)
- GPU tests on merge to main (optional, 10 min on Colab VM)

See Issue #81 for CI setup proposal.

## See Also

- [PHASE_1_MODEL_DECISION.md](PHASE_1_MODEL_DECISION.md) — Model selection rationale
- [CONTRIBUTING.md](CONTRIBUTING.md) — Acceptance criteria requirements
- [Phase 1 Collection Notebook](https://colab.research.google.com/...) — Full data collection (3-4 hours)
