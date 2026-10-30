#!/bin/bash
# Run Phase 1 acceptance tests
# Requirements: pytest, torch, transformers, bitsandbytes (for GPU tests)

set -e

echo "Running Phase 1 Acceptance Tests..."
echo "===================================="

# Run all Phase 1 tests
python -m pytest phase_1/scripts/test_*.py -v --tb=short

echo ""
echo "===================================="
echo "✅ All tests passed!"
