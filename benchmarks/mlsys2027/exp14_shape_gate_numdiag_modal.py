"""
MLSys 2027 Experiment 14 -- SELF-CONTAINED Modal app for the NON-EVIDENCE shape-gate numerical diagnosis (Attempt 2;
one H100; no model, no engine, no download). Post-failure harness fix: Attempt 1's wrapper imported
exp14_deployment_modal at module level and every container failed with ModuleNotFoundError before any diagnostic ran.

This module has NO dependency on sibling Exp14 Python modules: the frozen Exp14 `image = (...)` expression is copied
TEXTUALLY from exp14_deployment_modal.py (the offline test proves equality with the same extractor used for Exp4 /
Exp13 / Exp14), and the only files appended to the image are the unchanged diagnostic payload
(exp14_shape_gate_numdiag.py) and the frozen exp14_shape_gate.py it imports for the frozen seeds / reference helpers.
Launched only by benchmarks/mlsys2027/run_exp14_shape_gate_numdiag.py (hard wall-clock limit + app stop).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

BASE_COMMIT = "f329ce405b12623fb8b1cf1830f12e5a712523be"
SNAP = Path(os.environ.get("EXP14_VLLM_SNAPSHOT", "/nonexistent/EXP14_VLLM_SNAPSHOT-not-set.zip"))
DIAG_LOCAL = Path(__file__).resolve().parent / "exp14_shape_gate_numdiag.py"
DIAG_REMOTE = "/opt/exp14/exp14_shape_gate_numdiag.py"
SHAPE_GATE_LOCAL = Path(__file__).resolve().parent / "exp14_shape_gate.py"
SHAPE_GATE_REMOTE = "/opt/exp14/exp14_shape_gate.py"
FUNCTION_TIMEOUT_S = 2700  # Modal-side backstop (45 min); the local runner enforces the hard wall-clock limit
DIAG_SUBPROCESS_TIMEOUT_S = 2400

if modal.is_local() and not SNAP.is_file():
    raise RuntimeError(
        "EXP14_VLLM_SNAPSHOT is not set to an existing snapshot zip. Launch via "
        "benchmarks/mlsys2027/run_exp14_shape_gate_numdiag.py."
    )

app = modal.App("rabit-kv-mlsys2027-exp14-shape-gate-numdiag")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "curl", "libnuma1", "libnuma-dev")
    .pip_install(
        "pip>=25", "cmake>=3.26.1", "ninja", "packaging>=24.2",
        "setuptools>=77.0.3,<81.0.0", "setuptools-scm>=8.0",
        "setuptools-rust>=1.9.0", "wheel", "jinja2",
    )
    .run_commands(
        "python -m pip install torch==2.11.0 torchvision==0.26.0 "
        "torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130"
    )
    .add_local_file(
        str(SNAP), "/tmp/rabit2_final_fast_decode_append_snapshot.zip", copy=True
    )
    .run_commands(
        "rm -rf /root/vllm-kvquant && mkdir -p /root/vllm-kvquant && "
        "python -m zipfile -e /tmp/rabit2_final_fast_decode_append_snapshot.zip "
        "/root/vllm-kvquant"
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "VLLM_TARGET_DEVICE": "cuda",
        "VLLM_USE_PRECOMPILED": "1",
        "VLLM_PRECOMPILED_WHEEL_COMMIT": BASE_COMMIT,
        "VLLM_PRECOMPILED_WHEEL_VARIANT": "cu130",
        "VLLM_MAIN_CUDA_VERSION": "13.0",
        "VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX": "1",
        "SETUPTOOLS_SCM_PRETEND_VERSION": "0.10.0+kvquant",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM": "0.10.0+kvquant",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "VLLM_USE_V1": "1",
    })
    .run_commands(
        "cd /root/vllm-kvquant && python use_existing_torch.py --prefix",
        "cd /root/vllm-kvquant && rm -rf build dist *.egg-info /tmp/vllm.log && "
        "(python -m pip install -e . --no-build-isolation > /tmp/vllm.log 2>&1 || "
        "(tail -n 250 /tmp/vllm.log; exit 1))",
    )
    .pip_install("pytest", "modelscope")
)

image = (
    image.add_local_file(str(DIAG_LOCAL), DIAG_REMOTE, copy=True)
    .add_local_file(str(SHAPE_GATE_LOCAL), SHAPE_GATE_REMOTE, copy=True)
)


@app.function(image=image, gpu="H100", timeout=FUNCTION_TIMEOUT_S)
def numdiag() -> int:
    p = subprocess.run([sys.executable, DIAG_REMOTE], text=True, capture_output=True, timeout=DIAG_SUBPROCESS_TIMEOUT_S)
    print(p.stdout, flush=True)
    if p.stderr:
        print(p.stderr[-8000:], flush=True)
    return p.returncode


@app.local_entrypoint()
def main():
    rc = numdiag.remote()
    if rc:
        raise SystemExit(rc)
