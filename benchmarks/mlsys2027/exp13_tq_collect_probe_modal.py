"""
MLSys 2027 Experiment 13 -- TurboQuant test COLLECTION probe (NON-EVIDENCE).

Runs, in the unchanged Exp4-verbatim image on one H100, ONLY:
  * `pytest -q -p no:cacheprovider --confcutdir=/root/vllm-kvquant/tests/quantization --collect-only
     tests/quantization/test_turboquant.py` (cwd /root/vllm-kvquant) -- no test is executed;
  * whether scipy is importable (importlib.util.find_spec), and the value of the test module's GPGPU_AVAILABLE
    expression (torch.cuda.is_available() or torch.xpu.is_available()).
Purpose: the authoritative collected node-ID list and the frozen allowed-skip / GPU-only sets for the Attempt-2
TurboQuant gate. No benchmark, no engine, no source / image change.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

BASE_COMMIT = "f329ce405b12623fb8b1cf1830f12e5a712523be"
SNAP = Path(os.environ.get("EXP13_VLLM_SNAPSHOT", "/nonexistent/EXP13_VLLM_SNAPSHOT-not-set.zip"))
if modal.is_local() and not SNAP.is_file():
    raise RuntimeError("EXP13_VLLM_SNAPSHOT is not set. Launch via benchmarks/mlsys2027/run_exp13_tq_collect_probe.py.")

app = modal.App("rabit-kv-mlsys2027-exp13-tq-collect-probe")

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

COLLECT_CMD = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "--confcutdir=/root/vllm-kvquant/tests/quantization",
               "--collect-only", "tests/quantization/test_turboquant.py"]


@app.function(image=image, gpu="H100", timeout=1800)
def collect() -> None:
    import importlib.metadata as md
    import importlib.util

    spec = importlib.util.find_spec("scipy")
    try:
        scipy_version = md.version("scipy")
    except md.PackageNotFoundError:
        scipy_version = None
    gpgpu = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available() or "
                            "torch.xpu.is_available())"], capture_output=True, text=True)
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], capture_output=True,
                          text=True).stdout.strip()
    print("EXP13_TQCOLLECT_ENV=" + json.dumps({"scipy_importable": spec is not None, "scipy_version": scipy_version,
                                               "gpgpu_available_expr": gpgpu.stdout.strip(),
                                               "gpgpu_stderr_tail": gpgpu.stderr[-300:], "gpus": gpus}), flush=True)
    cmd = [sys.executable, *COLLECT_CMD]
    p = subprocess.run(cmd, cwd="/root/vllm-kvquant", capture_output=True, text=True)
    print("EXP13_TQCOLLECT_CMD=" + json.dumps({"cmd": cmd, "cwd": "/root/vllm-kvquant", "returncode": p.returncode}),
          flush=True)
    print("EXP13_TQCOLLECT_STDOUT_BEGIN\n" + p.stdout + "EXP13_TQCOLLECT_STDOUT_END", flush=True)
    print("EXP13_TQCOLLECT_STDERR_BEGIN\n" + p.stderr[-4000:] + "EXP13_TQCOLLECT_STDERR_END", flush=True)


@app.local_entrypoint()
def main():
    collect.remote()
