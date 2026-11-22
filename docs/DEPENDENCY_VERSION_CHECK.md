# Issue #107: dependency version checks

Base: main at `cddaf9a745b57ba323b47bc6d048193eadbab6be`.

## Scope and current status

Pytest now stops with exit code 4 before collecting tests when an exact `==` pin is missing or mismatched. This strict policy was selected on November 21, 2026. It applies to full-suite runs, selected files and runs launched from nested directories. There is no warning-only mode or automatic package installation.

Accelerate is also declared explicitly: `requirements.txt` now pins `accelerate==1.16.0`. The existing pin test automatically checks the new pin without changing its assertion. No other dependency pin or loader behavior was changed.

## Why Accelerate is required

`ModelLoader.load()` passes `device_map` to Transformers. Transformers 4.45.0's [loading implementation](https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/modeling_utils.py) requires Accelerate for that path. [Accelerate's dispatch documentation](https://huggingface.co/docs/accelerate/en/package_reference/big_modeling) describes its model placement and checkpoint loading functions.

The [PyPI metadata](https://pypi.org/pypi/accelerate/1.16.0/json) identified 1.16.0 as the current stable release on November 21, 2026. Its declared Python and torch requirements are compatible with Python 3.13 and torch 2.15.0. Compatibility was also exercised, rather than inferred from version ranges alone.

## Regression evidence

`test_accelerate_loading.py` creates a tiny Qwen2MoE model and saves a local checkpoint. It loads that checkpoint with `device_map="cpu"` and `low_cpu_mem_usage=True`, checks CPU placement, and compares output logits exactly against the original model. It downloads no weights and needs no GPU.

- Before the dependency was installed, the test failed with the expected ImportError requiring Accelerate.
- With Accelerate 1.16.0 and the unchanged scientific pins, the test passed. The loaded model's output matched with zero absolute and relative tolerance.
- The original eight pin tests passed before changes.
- `uv pip check` passed for the isolated 56-package CPU environment.
- A strict advisory query for Accelerate 1.16.0 returned no known findings. This was a single-package query, not a claim that the entire environment is free of vulnerabilities. Existing Transformers findings remain outside this fix.

The final CPU suite passed: **85 passed, eight GPU tests deselected**, in 14.50 seconds. This includes the existing 75 CPU tests, the new Accelerate pin case, the new device-map integration test and eight startup regression cases. Ten existing tokenizer return-value warnings and one local cache-variable deprecation warning remain. No existing assertion was changed.

The environment uses Python 3.13.16 on macOS, with bitsandbytes excluded. The user also ran the prepared source snapshot on Colab and reported **85 passed, 8 deselected, 10 warnings in 33.79 seconds**. That notebook embeds the same code and requirements as this change; it is not a run checked out at the final PR commit. The eight real-model GPU tests were intentionally excluded. No real-model CUDA run has been performed, so these results do not establish GPU memory or speed behavior.

## Startup design

- `dependency_versions.py` owns exact-pin parsing and metadata comparison. Both the root startup hook and the existing parameterized pin test use it, so their source of truth stays aligned.
- The root `pytest_sessionstart` hook validates before collection. It aggregates missing/mismatched packages into one diagnostic and exits with pytest's usage-error code, 4. Missing, empty or duplicate exact-pin inputs also fail validation.
- Version lookup uses the standard library's `importlib.metadata`. It does not import torch, Transformers or model-loading code.
- CPU/CUDA local suffixes such as `+cpu` and `+cu126` remain allowed when the public version matches. This preserves the existing pin-test contract and its original assertion.
- Only exact `==` pins are enforced here. Existing minimum-version requirements such as bitsandbytes and pytest remain outside this equality check; dependency installation still resolves those constraints.
- A minimal root `pytest.ini` anchors configuration discovery. A regression reproduced that a nested-directory invocation otherwise bypassed the root guard. No test paths are excluded by the config.
- Phase 1 fixture imports move into the collection hook or fixture that needs them. An incompatible scientific library therefore cannot hide the intended pin diagnostic during initial conftest loading. GPU marking and fixture cleanup behavior are preserved.

[Pytest's startup hook](https://docs.pytest.org/en/stable/reference/reference.html#pytest.hookspec.pytest_sessionstart) runs before collection. The eight subprocess regression cases use isolated requirement files and distribution metadata to test real pytest startup without uninstalling packages from the working environment. They cover missing and mismatched pins, matching versions, CPU/CUDA suffixes, malformed pin inputs and nested invocation before scientific imports. A marker written at test-module import proves that bad environments stop before collection.

## Verification

| Check | Result |
|---|---|
| Startup regression suite before implementation | Five failures and three passes, demonstrating the bypasses |
| Startup regression suite after implementation | Eight passed |
| Full CPU suite, `python -m pytest -m "not gpu" -q` | 85 passed, eight GPU tests deselected |
| Colab CPU source snapshot, user-reported | 85 passed, eight GPU tests deselected, 10 warnings in 33.79 seconds |
| Actual older environment without Accelerate, selecting only `test_shared_model.py` | Exits 4 with `accelerate==1.16.0: not installed`, before test collection |
| Invocation from `phase_1/scripts`, selecting `test_verify_environment.py` | Passed |
| Existing test assertions | Unchanged; the old parser moved to the shared helper |
| `git diff --check` | Passed |

The fixture subprocesses test the error path with controlled metadata, while the actual older environment independently confirms a missing real dependency is refused. Local checks do not replace hosted Linux or real-model CUDA validation.

## Integration notes

This work is isolated on `fix/107-session-version-check`. It does not depend on #59 or #123 merging. It changes neither model-loading behavior nor the other scientific dependency pins.

After this lands, Part A (#123) must regenerate its dependency locks for Accelerate and combine its root test-classification hook with this startup hook. Its pytest settings should extend the existing `pytest.ini`, rather than introduce a second configuration file whose settings pytest would ignore. Those are follow-up adjustments to Part A, not prerequisites for this fix.
