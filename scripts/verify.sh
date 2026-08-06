#!/usr/bin/env bash
set -eo pipefail

echo "======================================================"
echo " Running LEA Verification Suite"
echo "======================================================"

export PYTHONPATH="src:${PYTHONPATH}"

echo "Step 1: Running unit tests..."
pytest tests/unit -v --tb=short

echo "Step 2: Running integration tests..."
pytest tests/integration -v --tb=short

echo "Step 3: Running CLI doctor check..."
python3 -m lea doctor

echo "Step 4: Running GPU LLM inference tests..."
pytest tests/gpu -v --tb=short

echo "======================================================"
echo " All LEA verification checks completed successfully!"
echo "======================================================"
