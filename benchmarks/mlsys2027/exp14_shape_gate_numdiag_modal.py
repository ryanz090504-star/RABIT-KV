"""
MLSys 2027 Experiment 14 -- Modal app for the NON-EVIDENCE shape-gate numerical diagnosis (one H100; no model, no
engine, no download). Reuses the Exp14 image object from exp14_deployment_modal.py (imported, NOT modified) and adds
only exp14_shape_gate_numdiag.py. Launched only by benchmarks/mlsys2027/run_exp14_shape_gate_numdiag.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp14_deployment_modal import image as exp14_image  # noqa: E402  (frozen Exp14 image; read-only reuse)

DIAG_LOCAL = Path(__file__).resolve().parent / "exp14_shape_gate_numdiag.py"
DIAG_REMOTE = "/opt/exp14/exp14_shape_gate_numdiag.py"

app = modal.App("rabit-kv-mlsys2027-exp14-shape-gate-numdiag")
image = exp14_image.add_local_file(str(DIAG_LOCAL), DIAG_REMOTE, copy=True)


@app.function(image=image, gpu="H100", timeout=3600)
def numdiag() -> int:
    p = subprocess.run([sys.executable, DIAG_REMOTE], text=True, capture_output=True)
    print(p.stdout, flush=True)
    if p.stderr:
        print(p.stderr[-8000:], flush=True)
    return p.returncode


@app.local_entrypoint()
def main():
    rc = numdiag.remote()
    if rc:
        raise SystemExit(rc)
