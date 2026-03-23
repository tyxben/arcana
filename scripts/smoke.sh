#!/usr/bin/env bash
# Smoke test runner — exits 0 if all pass, 1 otherwise.
# Usage: bash scripts/smoke.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Arcana Smoke Tests ==="
echo ""

# Run smoke integration tests (skips tests that need missing keys)
uv run pytest tests/integration/test_smoke.py -v --tb=short "$@"

exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    echo ""
    echo "=== All smoke tests passed ==="
else
    echo ""
    echo "=== Smoke tests FAILED (exit $exit_code) ==="
fi

exit "$exit_code"
