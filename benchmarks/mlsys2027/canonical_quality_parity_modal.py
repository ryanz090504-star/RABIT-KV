"""
CPU-ONLY Modal app for the canonical-quality-v2 parity tests (no GPU, no model). Self-contained: no sibling-module
import at module level. The repository files the tests need are copied to /repo with their repo-relative paths; the
full result dict is RETURNED through the Modal function-call result and written locally by the entrypoint to
CANONICAL_PARITY_RESULT_PATH (never transported through the log stream).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
FILES = ["benchmarks/mlsys2027/canonical_rabit_quality.py", "benchmarks/mlsys2027/canonical_quality_parity_tests.py",
         "vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py", "benchmarks/quality/hotpotqa.py"]
RESULT_PATH_ENV = "CANONICAL_PARITY_RESULT_PATH"

app = modal.App("rabit-kv-canonical-quality-v2-parity")
image = (modal.Image.debian_slim(python_version="3.11")
         .run_commands("python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu")
         .pip_install("transformers==4.48.2"))
for rel in FILES:
    image = image.add_local_file(str(ROOT / rel), f"/repo/{rel}", copy=True)


@app.function(image=image, cpu=8, memory=16384, timeout=2400)
def parity() -> dict:
    sys.path.insert(0, "/repo/benchmarks/mlsys2027")
    import canonical_quality_parity_tests as t  # noqa: PLC0415

    return t.run(Path("/repo"))


@app.local_entrypoint()
def main():
    res = parity.remote()
    Path(os.environ[RESULT_PATH_ENV]).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("CANONICAL_PARITY=" + json.dumps({"passed": res["passed"], "negative_control_passed": res["negative_control_passed"],
                                            "torch": res["torch_version"], "python": res["python_version"],
                                            **{g: G["summary"] for g, G in res["geometries"].items()}}), flush=True)
    if not (res["passed"] and res["negative_control_passed"]):
        raise SystemExit(1)
