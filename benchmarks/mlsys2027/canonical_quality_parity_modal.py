"""
CPU-ONLY Modal app for the canonical-quality-v2 parity tests (no GPU, no model). Self-contained: no sibling-module
import at module level. Parity Attempt 2 harness fix: repository paths are resolved ONLY in local context
(modal.is_local()); the remote import never derives repository ancestry from __file__ and uses the fixed paths
/repo/<repo-relative path> to which the needed files are copied. The full result dict is RETURNED through the Modal
function-call result and written locally by the entrypoint to CANONICAL_PARITY_RESULT_PATH (never via the log stream).
"""

from __future__ import annotations

import json
import os
import sys

import modal

REMOTE_REPO = "/repo"
FILES = ["benchmarks/mlsys2027/canonical_rabit_quality.py", "benchmarks/mlsys2027/canonical_quality_parity_tests.py",
         "vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py", "benchmarks/quality/hotpotqa.py"]
RESULT_PATH_ENV = "CANONICAL_PARITY_RESULT_PATH"

app = modal.App("rabit-kv-canonical-quality-v2-parity")
image = (modal.Image.debian_slim(python_version="3.11")
         .run_commands("python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu")
         .pip_install("transformers==4.48.2"))
if modal.is_local():  # local repository layout is consulted ONLY when building the app locally
    from pathlib import Path

    _local_root = Path(__file__).resolve().parents[2]
    for _rel in FILES:
        image = image.add_local_file(str(_local_root / _rel), f"{REMOTE_REPO}/{_rel}", copy=True)


@app.function(image=image, cpu=8, memory=16384, timeout=1200)
def parity() -> dict:
    from pathlib import Path

    sys.path.insert(0, f"{REMOTE_REPO}/benchmarks/mlsys2027")
    import canonical_quality_parity_tests as t  # noqa: PLC0415

    return t.run(Path(REMOTE_REPO))


@app.local_entrypoint()
def main():
    res = parity.remote()
    with open(os.environ[RESULT_PATH_ENV], "w", encoding="utf-8") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")
    print("CANONICAL_PARITY=" + json.dumps({"passed": res["passed"], "negative_control_passed": res["negative_control_passed"],
                                            "torch": res["torch_version"], "python": res["python_version"],
                                            **{g: G["summary"] for g, G in res["geometries"].items()}}), flush=True)
    if not (res["passed"] and res["negative_control_passed"]):
        raise SystemExit(1)
