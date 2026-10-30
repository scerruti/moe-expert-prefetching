#!/bin/bash
# Run all acceptance tests on GPU and save output to Google Drive

set -e

# Determine output path (Google Drive mounted in Colab)
OUTPUT_DIR="${1:-/content/drive/MyDrive/moe-test-results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/acceptance_tests_${TIMESTAMP}.txt"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

echo "Installing requirements..."
pip install -q -r requirements.txt

echo "Running acceptance tests..."
echo "Output will be saved to: $OUTPUT_FILE"
echo ""

# Run tests and capture output
python -m pytest phase_1/scripts/test_full_pipeline.py phase_1/scripts/test_determinism_validation.py -v --tb=short 2>&1 | tee "$OUTPUT_FILE"

echo ""
echo "✅ Tests complete. Results saved to: $OUTPUT_FILE"
