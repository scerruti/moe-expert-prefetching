"""
Keeping one model shared across GPU tests clean (#118).

conftest.py loads Qwen1.5-MoE-A2.7B once per test session. A test that leaves
hooks on any module, or leaves the model in training mode (dropout on), would
change the results of every test after it. reset_after_test() undoes both and
reports the training-mode case, so the test that caused it fails, not a later one.
"""
import torch

# The per-module hook registries in torch.nn.Module (torch 2.15)
HOOK_ATTRIBUTES = (
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_backward_hooks",
    "_backward_pre_hooks",
)


def reset_after_test(model: torch.nn.Module) -> None:
    """Remove every module hook, put the model back in eval mode, and raise if it had left eval mode."""
    for module in model.modules():
        for attribute in HOOK_ATTRIBUTES:
            hooks = getattr(module, attribute, None)
            if hooks:
                hooks.clear()
    if model.training:
        model.eval()  # so the tests after this one still run on an eval-mode model
        raise AssertionError(
            "A test left the shared model in training mode (dropout on). "
            "Call model.eval() before the test ends."
        )
