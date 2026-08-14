#!/bin/bash
# Run remaining quality experiments: 22 configs
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
OUT="results/paper_full_20260705_222212/quality.jsonl"

for entry in \
  "kvquant_int3:4" "kvquant_int3:3" "kvquant_int3:2" "kvquant_int3:1" \
  "polar_int3:4" "polar_int3:3" "polar_int3:2" "polar_int3:1" \
  "turbo_int3:4" "turbo_int3:3" "turbo_int3:2" "turbo_int3:1" \
  "per_channel:4" "per_channel:3" "per_channel:2" "per_channel:1" \
  "group:4" "group:3" "group:2" "group:1" \
  "per_head:2" "per_head:1"
do
  policy="${entry%%:*}"
  bits="${entry##*:}"
  echo -n "$policy @ ${bits}bit ... "
  kvq quality \
    --model "TinyLlama/TinyLlama_v1.1" \
    --policy "$policy" --nbits "$bits" \
    --text-file "data/wikitext2_test.txt" \
    --max-tokens 128 --num-windows 2 \
    --warmup-steps 1 --error-breakdown \
    --output "$OUT" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'PPL={d[\"ppl\"]:.1f}')" || echo "FAIL"
done
echo "Done: $(wc -l < "$OUT") rows in $OUT"
