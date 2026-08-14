#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== Local smoke test ==="
PYTHON_BIN="${PYTHON:-python}"
"$PYTHON_BIN" -m pytest tests/ -q
kvq list-policies
kvq memory --policy document_naive --nbits 4
echo "OK"
