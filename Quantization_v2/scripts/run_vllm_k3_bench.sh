#!/bin/bash
# Run the vLLM deploy-latency entrypoint for the future kvquant_k3 fork.
#
# Required:
#   VLLM_ROOT=/path/to/editable/vllm-fork bash scripts/run_vllm_k3_bench.sh
#
# Until the fork implements kvquant_k3 cache allocation, write quantization,
# and packed-cache attention, kvq intentionally raises NotImplementedError.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${VLLM_ROOT:?Set VLLM_ROOT to the editable vLLM fork root}"

MODEL="${MODEL:-TinyLlama/TinyLlama_v1.1}"
POLICY="${POLICY:-kvquant_int3}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
OUT="${OUT:-results/vllm_k3_deploy.jsonl}"

export PYTHONPATH="$PROJECT_ROOT:$VLLM_ROOT:${PYTHONPATH:-}"

kvq latency \
  --backend vllm \
  --model "$MODEL" \
  --policy "$POLICY" \
  --nbits 3 \
  --kv-cache-dtype kvquant_k3 \
  --vllm-root "$VLLM_ROOT" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --output "$OUT"
