"""
MLSys 2027 Experiment 13 -- Modal app for the TurboQuant FEASIBILITY PROBE (NON-EVIDENCE).

One container, one H100: records the environment and idle GPU state, downloads the same model snapshot as
Experiment 4, and runs exp13_turboquant_probe_worker.py ONCE under the hard watchdog (exp3_watchdog.py). No
latency is measured, no BF16 / FP8 / RABIT leg is run, nothing is retried, and no setting is varied.

The `image = (...)` base expression is copied VERBATIM from the accepted exp4_deployment_modal.py (extracted
programmatically; test/launcher check equality); only the probe worker and the watchdog are appended. The
vllm-kvquant snapshot is a `git archive` of the committed tree (EXP13_VLLM_SNAPSHOT), exactly as in Experiment 4.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

MODEL = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
BASE_COMMIT = "f329ce405b12623fb8b1cf1830f12e5a712523be"
SNAP = Path(os.environ.get("EXP13_VLLM_SNAPSHOT", "/nonexistent/EXP13_VLLM_SNAPSHOT-not-set.zip"))
WORKER_LOCAL = Path(__file__).resolve().parent / "exp13_turboquant_probe_worker.py"
WORKER_REMOTE = "/opt/exp13/exp13_turboquant_probe_worker.py"
WATCHDOG_LOCAL = Path(__file__).resolve().parent / "exp3_watchdog.py"
WATCHDOG_REMOTE = "/opt/exp13/exp3_watchdog.py"
PROBE_TIMEOUT_S = 1200

if modal.is_local() and not SNAP.is_file():
    raise RuntimeError("EXP13_VLLM_SNAPSHOT is not set to an existing snapshot zip. Launch via "
                       "benchmarks/mlsys2027/run_exp13_turboquant_probe.py.")

app = modal.App("rabit-kv-mlsys2027-exp13-turboquant-feasibility-probe")
model_cache = modal.Volume.from_name("modelscope-llama31-cache", create_if_missing=True)

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
    image.add_local_file(str(WORKER_LOCAL), WORKER_REMOTE, copy=True)
    .add_local_file(str(WATCHDOG_LOCAL), WATCHDOG_REMOTE, copy=True)
)


def _emit(tag: str, payload) -> None:
    print(f"EXP13_PROBE_{tag}={json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _nvsmi(query: str, fields: str) -> list[dict]:
    out = subprocess.run(["nvidia-smi", f"--{query}={fields}", "--format=csv,noheader,nounits"],
                         text=True, capture_output=True, check=True).stdout
    return [dict(zip(fields.split(","), [v.strip() for v in ln.split(",")])) for ln in out.strip().splitlines()
            if ln.strip()]


@app.function(image=image, gpu="H100", timeout=3600, volumes={"/model_cache": model_cache})
def probe() -> None:
    import importlib.metadata as md

    from modelscope import snapshot_download

    gpus = _nvsmi("query-gpu", "index,name,uuid,driver_version,memory.total,memory.used")
    apps = _nvsmi("query-compute-apps", "pid,process_name,used_memory")
    versions = {}
    for pkg in ("vllm", "torch", "triton", "transformers", "modelscope"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    _emit("ENVIRONMENT", {"gpus": gpus, "compute_apps_before": apps, "python": sys.version.split()[0],
                          "packages": versions,
                          "vllm_precompiled_wheel_commit": os.environ.get("VLLM_PRECOMPILED_WHEEL_COMMIT")})
    if apps:
        raise RuntimeError(f"GPU has compute processes before the probe: {apps}")
    os.environ["MODELSCOPE_CACHE"] = "/model_cache"
    model_dir = snapshot_download(MODEL, cache_dir="/model_cache")
    _emit("MODEL", {"model": MODEL, "snapshot_dir": model_dir})

    sys.path.insert(0, str(Path(WATCHDOG_REMOTE).parent))
    from exp3_watchdog import run_with_watchdog

    cmd = [sys.executable, WORKER_REMOTE, "--model-dir", model_dir]
    _emit("WORKER_START", {"cmd": cmd, "timeout_s": PROBE_TIMEOUT_S})
    meta = run_with_watchdog(cmd, "[probe] ", PROBE_TIMEOUT_S, "turboquant_probe")
    _emit("WORKER_EXIT", meta)
    print("EXP13_PROBE_MODAL_DONE", flush=True)


@app.local_entrypoint()
def main():
    probe.remote()
