#!/bin/bash
# Single policy, complete pipeline: kvquant_int3 @ 4/3/2 bit
# Quality + Kernel Latency + Memory + Tasks + Report
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

RUN="paper_single_$(date +%Y%m%d_%H%M%S)"
OUT="results/$RUN"
mkdir -p "$OUT"

echo "=== Single Policy Pipeline: kvquant_int3 ==="
echo "Run: $RUN"
echo "Bits: 16 (baseline), 4, 3, 2"
echo ""

# ---- 1. Quality (PPL + Error + KL + Top-k) ----
echo "=== 1/4: Quality (PPL) ==="
for bits in 16 4 3 2; do
  policy="no_quant"
  [[ "$bits" != "16" ]] && policy="kvquant_int3"
  echo -n "  $policy @ ${bits}bit ... "
  kvq quality --model TinyLlama/TinyLlama_v1.1 \
    --policy "$policy" --nbits "$bits" \
    --text-file data/wikitext2_test.txt \
    --max-tokens 128 --num-windows 2 --warmup-steps 1 \
    --error-breakdown --output "$OUT/quality.jsonl" \
    2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'PPL={d[\"ppl\"]:.1f}')" || echo "FAIL"
done

# ---- 2. Kernel Latency (Triton) ----
echo ""
echo "=== 2/4: Kernel Latency ==="
for bits in 4 3 2; do
  echo -n "  kvquant_int3 @ ${bits}bit ... "
  kvq latency --model TinyLlama/TinyLlama_v1.1 \
    --policy kvquant_int3 --nbits "$bits" \
    --iters 50 --backend triton --output "$OUT/kernel.jsonl" \
    2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'TPOT={d[\"tpot_ms\"]:.3f}ms baseline={d[\"baseline_tpot_ms\"]:.3f}ms')" || echo "FAIL"
done

# ---- 3. Memory ----
echo ""
echo "=== 3/4: Memory ==="
kvq memory --model TinyLlama/TinyLlama_v1.1 --policy no_quant --nbits 16 \
  --batch-size 8 --seq-len 4096 --memory-budget-gb 40 --output "$OUT/memory.jsonl"
for bits in 4 3 2; do
  echo -n "  kvquant_int3 @ ${bits}bit ... "
  kvq memory --model TinyLlama/TinyLlama_v1.1 --policy kvquant_int3 --nbits "$bits" \
    --batch-size 8 --seq-len 4096 --memory-budget-gb 40 --output "$OUT/memory.jsonl" \
    2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Comp={d[\"compression_ratio\"]:.1f}x Eff={d[\"effective_bits_per_element\"]:.2f}bits')" || echo "FAIL"
done

# ---- 4. Tasks (Needle) ----
echo ""
echo "=== 4/4: Tasks (Needle) ==="
for bits in 16 4 3 2; do
  policy="no_quant"
  [[ "$bits" != "16" ]] && policy="kvquant_int3"
  echo -n "  $policy @ ${bits}bit ... "
  kvq tasks --model TinyLlama/TinyLlama_v1.1 \
    --policy "$policy" --nbits "$bits" \
    --dataset needle --dataset-path data/needle_retrieval.sample.jsonl \
    --max-examples 5 --max-new-tokens 20 --output "$OUT/tasks.jsonl" \
    2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Acc={d[\"accuracy_recovery\"]*100:.0f}%')" || echo "FAIL"
done

# ---- Report ----
echo ""
echo "=== Report ==="
REPORT_INPUTS=()
for f in "$OUT"/quality.jsonl "$OUT"/kernel.jsonl "$OUT"/memory.jsonl "$OUT"/tasks.jsonl; do
  [[ -f "$f" ]] && REPORT_INPUTS+=("$f")
done
kvq report --input "${REPORT_INPUTS[@]}" --output "$OUT/report.md" --title "kvquant_int3 - A100" 2>/dev/null
echo "Report: $OUT/report.md"
echo "Total rows: $(wc -l < "$OUT/quality.jsonl") quality + $(wc -l < "$OUT/kernel.jsonl") kernel + $(wc -l < "$OUT/memory.jsonl") memory + $(wc -l < "$OUT/tasks.jsonl") tasks"
echo "Results: $OUT/"
