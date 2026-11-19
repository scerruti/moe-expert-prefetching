#!/usr/bin/env bash
# Run from a trusted checkout after auditing and installing requirements/gpu.lock.
set -euo pipefail

expected_sha="${1:?Pass the full reviewed commit SHA}"
actual_sha="$(git rev-parse HEAD)"
if [[ "$actual_sha" != "$expected_sha" ]] || [[ ! "$expected_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "The checkout does not match the reviewed SHA" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing GPU attestation from a modified checkout" >&2
  exit 1
fi
mkdir -p reports
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; print(torch.cuda.get_device_name(0))'
python -m pip freeze > reports/gpu-environment.txt
printf '%s\n' "$actual_sha" > reports/tested-sha.txt
python -m pip check
python -m pytest phase_1/scripts/test_pinned_versions.py -q
python scripts/run_tests.py gpu 2>&1 | tee reports/gpu.log
