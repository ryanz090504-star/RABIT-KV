#!/bin/bash
# Check environment and run a quick sanity test (no GPU required for basic tests).
# Usage: bash scripts/check_env.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== KVQuant Environment Check ==="

echo -n "Python: "
python --version 2>&1 || { echo "NOT FOUND — install Python 3.10+"; exit 1; }

echo -n "PyTorch: "
python -c "import torch; print(torch.__version__)" 2>&1 || echo "NOT INSTALLED (pip install torch)"

echo -n "CUDA available: "
python -c "import torch; print(torch.cuda.is_available())" 2>&1 || echo "False"

echo -n "vLLM: "
python -c "import vllm; print(vllm.__version__)" 2>&1 || echo "NOT INSTALLED (pip install vllm for GPU)"

echo -n "kvquant: "
python -c "import kvquant; print('OK')" 2>&1 || echo "NOT INSTALLED (pip install -e .)"

echo ""
echo "=== Running unit tests ==="
python -m pytest tests/ -v --tb=short 2>&1 | tail -10 || echo "(tests may need GPU)"

echo ""
echo "=== Available policies ==="
python -c "from kvquant.policies import list_policies; print(', '.join(list_policies()))" 2>&1 || echo "N/A"

echo ""
echo "Done. If all checks passed, run: bash scripts/run_single_pipeline.sh"
