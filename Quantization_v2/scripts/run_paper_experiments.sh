#!/bin/bash
# 论文评测一键入口 — Paper-Ready Experiments
#
# 固定配置以保证可复现性。输出到 results/paper_v1/。
#
# 运行:
#   bash scripts/run_paper_experiments.sh
#
# 需要: CUDA GPU (至少 8GB VRAM)
# 模型: TinyLlama/TinyLlama_v1.1 (轻量, 快速迭代)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

# ── Fixed paper configuration ─────────────────────────────────────
MODEL="TinyLlama/TinyLlama_v1.1"
RUN="paper_v1_$(date +%Y%m%d_%H%M%S)"
OUT="$PROJECT_ROOT/results/$RUN"
REPORT="$PROJECT_ROOT/reports/${RUN}.md"
DATA="$PROJECT_ROOT/data/wikitext2_test.txt"
TASK_DATA="$PROJECT_ROOT/data/needle_retrieval.sample.jsonl"
VLLM_ROOT="${VLLM_ROOT:-/Users/chen/vs_code/vllm-kvquant}"

# Paper policies: 1 baseline + 7 quantized
# no_quant (fp16) → baseline
# document_naive (4-bit): global min-max
# per_head (4-bit): per-head scale
# per_channel (4-bit): per-channel scale
# group (4-bit, group=64): grouped quantization
# kvquant_int3 (3-bit): per-token-head INT3 (vLLM fork target)
# polar_int3 (3-bit): Hadamard rotation + INT3
# turbo_int3 (3-bit): PolarQuant + 1-bit QJL residual
POLICIES="no_quant,document_naive,per_head,per_channel,group,kvquant_int3,polar_int3,turbo_int3"

PARAMS=(
  # (policy, nbits)
  "no_quant:16"
  "document_naive:4"
  "per_head:4"
  "per_channel:4"
  "group:4"
  "kvquant_int3:3"
  "polar_int3:3"
  "turbo_int3:3"
)

MAX_TOKENS=256
WINDOWS=4
BATCH_SIZE=1
WARMUP=3
ITERS=100
TASK_EXAMPLES=5
TASK_MAX_NEW_TOKENS=20
MEM_BATCH=8
MEM_SEQ=4096
MEM_BUDGET=24

mkdir -p "$OUT" "$(dirname "$REPORT")"

# ── Header ────────────────────────────────────────────────────────
echo "============================================================"
echo "  KVQuant Paper Experiments v1"
echo "  Model: $MODEL"
echo "  Policies: $POLICIES"
echo "  Run: $RUN"
echo "  GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo cpu)"
echo "============================================================"

# Save experiment manifest
python3 -c "
import json, torch, platform, datetime
info = {
    'experiment': 'paper_v1',
    'run': '$RUN',
    'date': datetime.datetime.now().isoformat(),
    'model': '$MODEL',
    'policies': '$POLICIES',
    'hardware': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    'cuda': torch.version.cuda if torch.cuda.is_available() else None,
    'torch': torch.__version__,
    'python': platform.python_version(),
    'max_tokens': $MAX_TOKENS,
    'windows': $WINDOWS,
    'kernel_iters': $ITERS,
    'memory_batch_size': $MEM_BATCH,
    'memory_seq_len': $MEM_SEQ,
    'task_examples': $TASK_EXAMPLES,
}
with open('$OUT/manifest.json', 'w') as f:
    json.dump(info, f, indent=2, default=str)
print(json.dumps(info, indent=2))
"

# ── 1. Quality (PPL + Error + Logit Fidelity) ───────────────────
echo ""
echo "=== Dimension 1/4: Quality (PPL + Error Breakdown) ==="

for entry in "${PARAMS[@]}"; do
  policy="${entry%%:*}"
  bits="${entry##*:}"
  echo "  [$policy @ ${bits}bit]"

  if [ "$policy" = "no_quant" ]; then
    kvq quality --model "$MODEL" --policy "$policy" --nbits "$bits" \
      --text-file "$DATA" --max-tokens "$MAX_TOKENS" --num-windows "$WINDOWS" \
      --batch-size "$BATCH_SIZE" --warmup-steps "$WARMUP" \
      --output "$OUT/quality.jsonl" 2>&1 | tail -1 || echo "  FAILED"
  else
    kvq quality --model "$MODEL" --policy "$policy" --nbits "$bits" \
      --text-file "$DATA" --max-tokens "$MAX_TOKENS" --num-windows "$WINDOWS" \
      --batch-size "$BATCH_SIZE" --warmup-steps "$WARMUP" --error-breakdown \
      --output "$OUT/quality.jsonl" 2>&1 | tail -1 || echo "  FAILED"
  fi
done

# ── 2. Kernel Latency ───────────────────────────────────────────
echo ""
echo "=== Dimension 2/4: Kernel Latency ==="

for entry in "${PARAMS[@]}"; do
  policy="${entry%%:*}"
  bits="${entry##*:}"
  if [ "$bits" = "16" ]; then continue; fi

  echo "  [$policy @ ${bits}bit]"
  kvq latency --model "$MODEL" --policy "$policy" --nbits "$bits" \
    --iters "$ITERS" --backend triton \
    --output "$OUT/kernel.jsonl" 2>&1 | tail -1 || echo "  FAILED"
done

# ── 3. Memory ───────────────────────────────────────────────────
echo ""
echo "=== Dimension 3/4: Memory Analysis ==="

for entry in "${PARAMS[@]}"; do
  policy="${entry%%:*}"
  bits="${entry##*:}"
  if [ "$bits" = "16" ] && [ "$policy" != "no_quant" ]; then continue; fi

  echo "  [$policy @ ${bits}bit]"
  kvq memory --model "$MODEL" --policy "$policy" --nbits "$bits" \
    --batch-size "$MEM_BATCH" --seq-len "$MEM_SEQ" \
    --memory-budget-gb "$MEM_BUDGET" \
    --output "$OUT/memory.jsonl" 2>&1 | tail -1 || echo "  FAILED"
done

# ── 4. Task Quality (exact match) ───────────────────────────────
echo ""
echo "=== Dimension 4/4: Task Quality ==="

for entry in "${PARAMS[@]}"; do
  policy="${entry%%:*}"
  bits="${entry##*:}"
  echo "  [$policy @ ${bits}bit]"
  kvq tasks --model "$MODEL" --policy "$policy" --nbits "$bits" \
    --dataset needle --dataset-path "$TASK_DATA" \
    --max-examples "$TASK_EXAMPLES" --max-new-tokens "$TASK_MAX_NEW_TOKENS" \
    --output "$OUT/tasks.jsonl" 2>&1 | tail -1 || echo "  FAILED"
done

# ── 5. Report ───────────────────────────────────────────────────
echo ""
echo "=== Generate Paper Report ==="

REPORT_INPUTS=()
for f in quality kernel memory tasks; do
  if [ -f "$OUT/$f.jsonl" ]; then
    REPORT_INPUTS+=("$OUT/$f.jsonl")
  fi
done

kvq report \
  --input "${REPORT_INPUTS[@]}" \
  --output "$REPORT" \
  --title "KV Cache Quantization — Paper Experiments v1 ($MODEL)"

# Generate LaTeX tables
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from kvquant.latex_export import export_to_latex
export_to_latex(
    inputs=['$OUT/quality.jsonl', '$OUT/kernel.jsonl', '$OUT/memory.jsonl', '$OUT/tasks.jsonl'],
    output='$OUT/paper_tables.tex',
)
" 2>/dev/null && echo "LaTeX tables → $OUT/paper_tables.tex" || echo "[skip] LaTeX export unavailable"

# ── 6. Validate ─────────────────────────────────────────────────
echo ""
echo "=== Evidence Validation ==="
kvq validate --input "$OUT"/*.jsonl || echo "(some checks failed — review before submitting)"

echo ""
echo "============================================================"
echo "  Paper experiments complete!"
echo "  Report:   $REPORT"
echo "  Results:  $OUT/"
echo "  Manifest: $OUT/manifest.json"
echo "============================================================"
