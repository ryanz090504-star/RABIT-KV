"""
MLSys 2027 Experiment 14 -- CPU-only Modal app: verify every Qwen2.5-7B-Instruct snapshot directory in the shared
model volume against the FROZEN Model-B identity (exp14_model_snapshot.py). No GPU, no download, no model load:
it only hashes files already in the volume. Used by the quality part after its five runs (the canonical quality
scripts download without a revision and may not be modified). Emits EXP14_MODEL_SCAN=<json>; raises on failure.
Launched only by benchmarks/mlsys2027/run_experiment14_second_model.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

SNAPSHOT_LOCAL = Path(__file__).resolve().parent / "exp14_model_snapshot.py"
SNAPSHOT_REMOTE = "/opt/exp14/exp14_model_snapshot.py"

app = modal.App("rabit-kv-mlsys2027-exp14-model-scan")
model_cache = modal.Volume.from_name("modelscope-llama31-cache", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").add_local_file(str(SNAPSHOT_LOCAL), SNAPSHOT_REMOTE, copy=True)


@app.function(image=image, cpu=4, timeout=3600, volumes={"/model_cache": model_cache})
def scan() -> dict:
    model_cache.reload()
    sys.path.insert(0, str(Path(SNAPSHOT_REMOTE).parent))
    import exp14_model_snapshot as ms

    res = ms.scan_volume("/model_cache")
    print("EXP14_MODEL_SCAN=" + json.dumps(res, sort_keys=True), flush=True)
    return res


@app.local_entrypoint()
def main():
    res = scan.remote()
    if not res["passed"]:
        raise SystemExit("Model-B snapshot scan FAILED")
