#!/bin/bash
# Run remaining 6 quality benchmarks (no_quant and document_naive already done)
# Usage: bash run_quality_batch.sh <run_dir>
set -euo pipefail
RUN_DIR="${1:-results/paper_v1_$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "$RUN_DIR"
OUT="$RUN_DIR/quality.jsonl"

POLICIES=(
  "per_head:4"
  "per_channel:4"
  "group:4"
  "kvquant_int3:3"
  "polar_int3:3"
  "turbo_int3:3"
)

for entry in "${POLICIES[@]}"; do
  policy="${entry%%:*}"
  bits="${entry##*:}"
  echo "=== $policy @ ${bits}bit ==="
  kvq quality \
    --model "TinyLlama/TinyLlama_v1.1" \
    --policy "$policy" --nbits "$bits" \
    --text-file "data/wikitext2_test.txt" \
    --max-tokens 128 --num-windows 2 \
    --warmup-steps 1 --error-breakdown \
    --output "$OUT" 2>&1 | tail -2
  echo ""
done
echo "Done. Results: $OUT"
