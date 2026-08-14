#!/bin/bash
# Lightweight quality benchmark — run one policy with PPL + error breakdown
# Usage: bash run_quality_one.sh <policy> <nbits>

set -euo pipefail
POLICY="${1:-no_quant}"
BITS="${2:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=== $POLICY @ ${BITS}bit ==="
kvq quality \
  --model "TinyLlama/TinyLlama_v1.1" \
  --policy "$POLICY" --nbits "$BITS" \
  --text-file "data/wikitext2_test.txt" \
  --max-tokens 128 --num-windows 2 \
  --batch-size 1 --warmup-steps 1 \
  --error-breakdown \
  --output "results/quick_test.jsonl" 2>&1
