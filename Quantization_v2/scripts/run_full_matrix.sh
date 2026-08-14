#!/bin/bash
# Full experiment matrix: all 8 policies x all bit-widths (16, 4, 3, 2, 1)
# + kernel latency + memory + tasks
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

RUN="paper_full_$(date +%Y%m%d_%H%M%S)"
OUT="results/$RUN"
mkdir -p "$OUT"

echo "=== Full Experiment Matrix ==="
echo "Run: $RUN  Output: $OUT/"
echo "Policies: all 8"
echo "Bits: 16 4 3 2 1"

# Entry list: policy_name:bits
ENTRIES=(
  "no_quant:16"
  "document_naive:4" "document_naive:3" "document_naive:2" "document_naive:1"
  "per_head:4" "per_head:3" "per_head:2" "per_head:1"
  "per_channel:4" "per_channel:3" "per_channel:2" "per_channel:1"
  "group:4" "group:3" "group:2" "group:1"
  "kvquant_int3:4" "kvquant_int3:3" "kvquant_int3:2" "kvquant_int3:1"
  "polar_int3:4" "polar_int3:3" "polar_int3:2" "polar_int3:1"
  "turbo_int3:4" "turbo_int3:3" "turbo_int3:2" "turbo_int3:1"
)

# ---- Quality ----
echo ""
echo "=== Quality ==="
for entry in "${ENTRIES[@]}"; do
  policy="${entry%%:*}"
  bits="${entry##*:}"
  echo -n "  $policy @ ${bits}bit ... "
  kvq quality \
    --model "TinyLlama/TinyLlama_v1.1" \
    --policy "$policy" --nbits "$bits" \
    --text-file "data/wikitext2_test.txt" \
    --max-tokens 128 --num-windows 2 \
    --warmup-steps 1 --error-breakdown \
    --output "$OUT/quality.jsonl" 2>/dev/null && echo "OK" || echo "FAILED"
done

# ---- Kernel Latency ----
echo ""
echo "=== Kernel Latency ==="
for entry in "${ENTRIES[@]}"; do
  policy="${entry%%:*}"
  bits="${entry##*:}"
  if [ "$bits" = "16" ]; then continue; fi
  echo -n "  $policy @ ${bits}bit ... "
  kvq latency \
    --model "TinyLlama/TinyLlama_v1.1" \
    --policy "$policy" --nbits "$bits" \
    --iters 50 --backend triton \
    --output "$OUT/kernel.jsonl" 2>/dev/null && echo "OK" || echo "FAILED"
done

# ---- Memory ----
echo ""
echo "=== Memory ==="
for entry in "${ENTRIES[@]}"; do
  policy="${entry%%:*}"
  bits="${entry##*:}"
  echo -n "  $policy @ ${bits}bit ... "
  kvq memory \
    --model "TinyLlama/TinyLlama_v1.1" \
    --policy "$policy" --nbits "$bits" \
    --batch-size 8 --seq-len 4096 --memory-budget-gb 40 \
    --output "$OUT/memory.jsonl" 2>/dev/null && echo "OK" || echo "FAILED"
done

# ---- Tasks ----
echo ""
echo "=== Tasks ==="
for entry in "${ENTRIES[@]}"; do
  policy="${entry%%:*}"
  bits="${entry##*:}"
  echo -n "  $policy @ ${bits}bit ... "
  kvq tasks \
    --model "TinyLlama/TinyLlama_v1.1" \
    --policy "$policy" --nbits "$bits" \
    --dataset needle --dataset-path "data/needle_retrieval.sample.jsonl" \
    --max-examples 5 --max-new-tokens 20 \
    --output "$OUT/tasks.jsonl" 2>/dev/null && echo "OK" || echo "FAILED"
done

echo ""
echo "=== Generate Reports ==="
REPORT_INPUTS=()
for f in "$OUT"/quality.jsonl "$OUT"/kernel.jsonl "$OUT"/memory.jsonl "$OUT"/tasks.jsonl; do
  if [ -f "$f" ]; then REPORT_INPUTS+=("$f"); fi
done

kvq report --input "${REPORT_INPUTS[@]}" --output "$OUT/report.md" --title "KV Cache Quantization - Full Matrix (A100)" 2>/dev/null
kvq validate --input "${REPORT_INPUTS[@]}" 2>/dev/null || true

echo ""
echo "Done! Total rows: $(wc -l < "$OUT/quality.jsonl" 2>/dev/null || echo 0) quality, $(wc -l < "$OUT/kernel.jsonl" 2>/dev/null || echo 0) kernel, $(wc -l < "$OUT/memory.jsonl" 2>/dev/null || echo 0) memory, $(wc -l < "$OUT/tasks.jsonl" 2>/dev/null || echo 0) tasks"
echo "Report: $OUT/report.md"
echo "Results: $OUT/"
