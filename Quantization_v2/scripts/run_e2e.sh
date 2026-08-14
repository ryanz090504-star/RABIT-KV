#!/bin/bash
# KVQuant 端到端 GPU 实验脚本 (v2)
#
# 运行:
#   MODEL=TinyLlama/TinyLlama_v1.1 bash scripts/run_e2e.sh
#   POLICIES="kvquant_int3,turbo_int3,polar_int3" bash scripts/run_e2e.sh
#   RUN_TASKS=1 RUN_VLLM_DEPLOY=1 bash scripts/run_e2e.sh
#
# 论文证据边界:
#   - quality: quality_reference_not_deploy_speedup
#   - kernel latency: kernel_latency_not_deploy_speedup
#   - vLLM deploy: deploy_latency (需要 vLLM fork 的 packed attention kernel)
#   - tasks: quality_task_not_deploy_speedup
#   - memory: memory_estimate_not_runtime_peak

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

# ── User-configurable ─────────────────────────────────────────────
MODEL="${MODEL:-TinyLlama/TinyLlama_v1.1}"
RUN="${RUN:-e2e_$(date +%Y%m%d_%H%M%S)}"
OUT="$PROJECT_ROOT/results/$RUN"
REPORT="$PROJECT_ROOT/reports/${RUN}.md"
DATA="${DATA:-data/wikitext2_test.txt}"
POLICIES="${POLICIES:-document_naive,per_head,per_channel,group,kvquant_int3,polar_int3,turbo_int3,outlier_residual}"
BITS="${BITS:-16 4 3}"
MAX_TOKENS="${MAX_TOKENS:-256}"
WINDOWS="${WINDOWS:-4}"
ITERS="${ITERS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RUN_TASKS="${RUN_TASKS:-0}"
RUN_VLLM_DEPLOY="${RUN_VLLM_DEPLOY:-0}"
VLLM_ROOT="${VLLM_ROOT:-/Users/chen/vs_code/vllm-kvquant}"
TASK_DATA="${TASK_DATA:-data/needle_retrieval.sample.jsonl}"
TASK_DATASET="${TASK_DATASET:-needle}"
TASK_EXAMPLES="${TASK_EXAMPLES:-5}"
TASK_MAX_NEW_TOKENS="${TASK_MAX_NEW_TOKENS:-20}"
MEM_BATCH_SIZE="${MEM_BATCH_SIZE:-8}"
MEM_SEQ_LEN="${MEM_SEQ_LEN:-4096}"
MEM_BUDGET_GB="${MEM_BUDGET_GB:-24}"

mkdir -p "$OUT" "$(dirname "$REPORT")"

# ── Environment info ──────────────────────────────────────────────
echo "============================================"
echo "  KVQuant E2E — $RUN"
echo "  Model: $MODEL"
echo "  Policies: $POLICIES"
echo "  Bits: $BITS"
echo "  GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo cpu)"
echo "  CUDA: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo N/A)"
echo "============================================"

# Save env info
python3 -c "
import json, torch, platform
info = {
    'run': '$RUN',
    'model': '$MODEL',
    'policies': '$POLICIES',
    'bits': '$BITS',
    'hardware': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
    'python_version': platform.python_version(),
    'platform': platform.platform(),
}
with open('$OUT/env.json', 'w') as f:
    json.dump(info, f, indent=2)
"

IFS=',' read -ra POLICY_ARRAY <<<"$POLICIES"

# ── Step 1: Quality (PPL) ────────────────────────────────────────
echo ""
echo "=== Step 1: Quality (PPL) ==="

# Baseline (16-bit / no_quant)
echo "  -> Baseline (no_quant, fp16 reference)"
kvq quality \
  --model "$MODEL" --policy no_quant --nbits 16 \
  --text-file "$DATA" --max-tokens "$MAX_TOKENS" --num-windows "$WINDOWS" \
  --batch-size "$BATCH_SIZE" --error-breakdown \
  --output "$OUT/quality.jsonl" || echo "  [skip] baseline failed"

for policy in "${POLICY_ARRAY[@]}"; do
  for bits in $BITS; do
    echo "  -> $policy @ ${bits}bit"
    kvq quality \
      --model "$MODEL" --policy "$policy" --nbits "$bits" \
      --text-file "$DATA" --max-tokens "$MAX_TOKENS" --num-windows "$WINDOWS" \
      --batch-size "$BATCH_SIZE" --error-breakdown \
      --output "$OUT/quality.jsonl" || echo "  [skip] $policy/${bits}bit failed"
  done
done

# ── Step 2: Kernel latency diagnosis ──────────────────────────────
echo ""
echo "=== Step 2: Kernel Latency Diagnostic ==="

for policy in "${POLICY_ARRAY[@]}"; do
  for bits in $BITS; do
    # Skip 16-bit for quantized policies
    if [ "$bits" = "16" ]; then continue; fi
    echo "  -> $policy @ ${bits}bit (Triton kernel)"
    kvq latency \
      --model "$MODEL" --policy "$policy" --nbits "$bits" \
      --iters "$ITERS" --backend triton \
      --output "$OUT/kernel.jsonl" || echo "  [skip] $policy/${bits}bit kernel failed"
  done
done

# ── Step 3: Memory analysis ───────────────────────────────────────
echo ""
echo "=== Step 3: Memory Analysis ==="

for policy in "${POLICY_ARRAY[@]}"; do
  for bits in $BITS; do
    if [ "$bits" = "16" ]; then continue; fi
    echo "  -> $policy @ ${bits}bit"
    kvq memory \
      --model "$MODEL" --policy "$policy" --nbits "$bits" \
      --batch-size "$MEM_BATCH_SIZE" --seq-len "$MEM_SEQ_LEN" \
      --memory-budget-gb "$MEM_BUDGET_GB" \
      --output "$OUT/memory.jsonl" || echo "  [skip] $policy/${bits}bit memory failed"
  done
done

# Baseline memory
kvq memory \
  --model "$MODEL" --policy no_quant --nbits 16 \
  --batch-size "$MEM_BATCH_SIZE" --seq-len "$MEM_SEQ_LEN" \
  --memory-budget-gb "$MEM_BUDGET_GB" \
  --output "$OUT/memory.jsonl" || echo "  [skip] baseline memory failed"

# ── Step 4: Task quality (optional) ──────────────────────────────
if [[ "$RUN_TASKS" == "1" ]]; then
  echo ""
  echo "=== Step 4: Task Quality (reference generate, not deploy) ==="

  kvq tasks \
    --model "$MODEL" --policy no_quant --nbits 16 \
    --dataset "$TASK_DATASET" --dataset-path "$TASK_DATA" \
    --max-examples "$TASK_EXAMPLES" --max-new-tokens "$TASK_MAX_NEW_TOKENS" \
    --output "$OUT/tasks.jsonl" || echo "  [skip] baseline tasks failed"

  for policy in "${POLICY_ARRAY[@]}"; do
    for bits in $BITS; do
      if [ "$bits" = "16" ]; then continue; fi
      echo "  -> $policy @ ${bits}bit"
      kvq tasks \
        --model "$MODEL" --policy "$policy" --nbits "$bits" \
        --dataset "$TASK_DATASET" --dataset-path "$TASK_DATA" \
        --max-examples "$TASK_EXAMPLES" --max-new-tokens "$TASK_MAX_NEW_TOKENS" \
        --output "$OUT/tasks.jsonl" || echo "  [skip] $policy/${bits}bit tasks failed"
    done
  done
fi

# ── Step 5: vLLM deploy latency (optional, needs vLLM fork) ──────
if [[ "$RUN_VLLM_DEPLOY" == "1" ]]; then
  echo ""
  echo "=== Step 5: vLLM Deploy Latency ==="

  if [ -d "$VLLM_ROOT" ]; then
    for policy in "${POLICY_ARRAY[@]}"; do
      for bits in $BITS; do
        if [ "$bits" = "16" ]; then continue; fi
        echo "  -> $policy @ ${bits}bit (vLLM serving)"
        kvq latency \
          --model "$MODEL" --policy "$policy" --nbits "$bits" \
          --backend vllm --vllm-root "$VLLM_ROOT" \
          --kv-cache-dtype "kvquant_k3" \
          --output "$OUT/deploy.jsonl" || echo "  [skip] $policy/${bits}bit vLLM deploy not ready"
      done
    done
  else
    echo "  [skip] vLLM root not found: $VLLM_ROOT"
  fi
fi

# ── Step 6: Report ─────────────────────────────────────────────────
echo ""
echo "=== Step 6: Generate Report ==="

REPORT_INPUTS=()
for f in "$OUT"/quality.jsonl "$OUT"/kernel.jsonl "$OUT"/memory.jsonl; do
  if [ -f "$f" ]; then
    REPORT_INPUTS+=("$f")
  fi
done

if [ -f "$OUT/tasks.jsonl" ]; then
  REPORT_INPUTS+=("$OUT/tasks.jsonl")
fi
if [ -f "$OUT/deploy.jsonl" ]; then
  REPORT_INPUTS+=("$OUT/deploy.jsonl")
fi

if [ ${#REPORT_INPUTS[@]} -gt 0 ]; then
  kvq report \
    --input "${REPORT_INPUTS[@]}" \
    --output "$REPORT" \
    --title "KV Cache Quantization E2E — $MODEL"
  echo "  Report: $REPORT"
else
  echo "  [skip] no result files found"
fi

# ── Step 7: Validate ───────────────────────────────────────────────
echo ""
echo "=== Step 7: Validate ==="

VALIDATE_INPUTS=("$OUT"/*.jsonl)
if [ ${#VALIDATE_INPUTS[@]} -gt 0 ]; then
  kvq validate --input "${VALIDATE_INPUTS[@]}" || echo "(validation issues — see above)"
else
  echo "  [skip] no JSONL files to validate"
fi

echo ""
echo "Done! Report: $REPORT"
echo "Results: $OUT/"
