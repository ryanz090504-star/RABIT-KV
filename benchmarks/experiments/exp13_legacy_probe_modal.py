"""
MLSys 2027 Experiment 13 -- Modal app for the LEGACY-RUNNER feasibility probe (NON-EVIDENCE).

One container, one H100, VLLM_USE_V2_MODEL_RUNNER=0 for every process. Phases, each a fresh process under the hard
watchdog (exp3_watchdog.py), with a clean-GPU check (no compute process; memory.used within tolerance of the idle
baseline) before every process, and the decision rule applied in order:
  1. TurboQuant turboquant_k3v4_nc engine probe          -> stop if startup fails
  2. RABIT frozen physical correctness gate (UNCHANGED exp3_correctness_gate.py; kernel-level exactness + RABIT
     pytest suites) and the RABIT rabit_kv2 engine probe  -> stop if either fails
  3. BF16 and native FP8 E4M3 startup-only engine probes
No latency is measured, nothing is retried, no setting is varied, no source is modified.

The `image = (...)` base expression is copied VERBATIM from the accepted exp4_deployment_modal.py (the launcher checks
equality); only the probe worker, the gate and the watchdog are appended.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

MODEL = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
BASE_COMMIT = "f329ce405b12623fb8b1cf1830f12e5a712523be"
SNAP = Path(os.environ.get("EXP13_VLLM_SNAPSHOT", "/nonexistent/EXP13_VLLM_SNAPSHOT-not-set.zip"))
WORKER_LOCAL = Path(__file__).resolve().parent / "exp13_legacy_probe_worker.py"
WORKER_REMOTE = "/opt/exp13/exp13_legacy_probe_worker.py"
GATE_LOCAL = Path(__file__).resolve().parent / "exp3_correctness_gate.py"
GATE_REMOTE = "/opt/exp13/exp3_correctness_gate.py"
WATCHDOG_LOCAL = Path(__file__).resolve().parent / "exp3_watchdog.py"
WATCHDOG_REMOTE = "/opt/exp13/exp3_watchdog.py"
PROCESS_TIMEOUT_S = 1200
GPU_CLEAN_TOLERANCE_MIB = 256
GPU_CLEAN_MAX_WAIT_S = 60

if modal.is_local() and not SNAP.is_file():
    raise RuntimeError("EXP13_VLLM_SNAPSHOT is not set to an existing snapshot zip. Launch via "
                       "benchmarks/mlsys2027/run_exp13_legacy_probe.py.")

app = modal.App("rabit-kv-mlsys2027-exp13-legacy-runner-probe")
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
    .add_local_file(str(GATE_LOCAL), GATE_REMOTE, copy=True)
    .add_local_file(str(WATCHDOG_LOCAL), WATCHDOG_REMOTE, copy=True)
)


def _emit(tag: str, payload) -> None:
    print(f"EXP13_LPROBE_{tag}={json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _nvsmi(query: str, fields: str) -> list[dict]:
    out = subprocess.run(["nvidia-smi", f"--{query}={fields}", "--format=csv,noheader,nounits"],
                         text=True, capture_output=True, check=True).stdout
    return [dict(zip(fields.split(","), [v.strip() for v in ln.split(",")])) for ln in out.strip().splitlines()
            if ln.strip()]


def _mem() -> list[int]:
    return [int(g["memory.used"]) for g in _nvsmi("query-gpu", "memory.used")]


def _require_clean(label: str, baseline: list[int]) -> None:
    deadline, readings = time.time() + GPU_CLEAN_MAX_WAIT_S, []
    while True:
        apps, mem = _nvsmi("query-compute-apps", "pid,process_name,used_memory"), _mem()
        readings.append({"compute_apps": apps, "memory_used_mib": mem})
        clean = not apps and all(u <= b + GPU_CLEAN_TOLERANCE_MIB for u, b in zip(mem, baseline))
        if clean or time.time() >= deadline:
            break
        time.sleep(2)
    _emit("PRE_PROCESS_GPU_STATE", {"label": label, "clean": clean, "baseline_mib": baseline, "readings": readings})
    if not clean:
        raise RuntimeError(f"GPU not clean before {label}")


@app.function(image=image, gpu="H100", timeout=7200, volumes={"/model_cache": model_cache})
def probe() -> None:
    import importlib.metadata as md

    from modelscope import snapshot_download

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"  # inherited by every child process
    gpus = _nvsmi("query-gpu", "index,name,uuid,driver_version,memory.total,memory.used")
    apps = _nvsmi("query-compute-apps", "pid,process_name,used_memory")
    versions = {}
    for pkg in ("vllm", "torch", "triton", "transformers", "modelscope", "pytest"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    _emit("ENVIRONMENT", {"gpus": gpus, "compute_apps_before": apps, "python": sys.version.split()[0],
                          "packages": versions, "VLLM_USE_V2_MODEL_RUNNER": os.environ["VLLM_USE_V2_MODEL_RUNNER"],
                          "vllm_precompiled_wheel_commit": os.environ.get("VLLM_PRECOMPILED_WHEEL_COMMIT")})
    if apps:
        raise RuntimeError(f"GPU has compute processes before the probe: {apps}")
    baseline = _mem()
    os.environ["MODELSCOPE_CACHE"] = "/model_cache"
    model_dir = snapshot_download(MODEL, cache_dir="/model_cache")
    _emit("MODEL", {"model": MODEL, "snapshot_dir": model_dir})

    sys.path.insert(0, str(Path(WATCHDOG_REMOTE).parent))
    from exp3_watchdog import run_with_watchdog

    def run(label: str, cmd: list[str]) -> int:
        _require_clean(label, baseline)
        _emit("PROCESS_START", {"label": label, "cmd": cmd, "timeout_s": PROCESS_TIMEOUT_S})
        meta = run_with_watchdog(cmd, f"[{label}] ", PROCESS_TIMEOUT_S, label)
        _emit("PROCESS_EXIT", {"label": label, **meta})
        return 1 if meta["timed_out"] else meta["returncode"]

    def engine(dtype: str) -> int:
        return run(dtype, [sys.executable, WORKER_REMOTE, "--model-dir", model_dir, "--kv-cache-dtype", dtype])

    phases = []
    rc = engine("turboquant_k3v4_nc")
    phases.append({"phase": "turboquant_k3v4_nc", "returncode": rc})
    if rc == 0:
        rc = run("rabit_gate", [sys.executable, GATE_REMOTE])
        phases.append({"phase": "rabit_gate", "returncode": rc})
        if rc == 0:
            rc = engine("rabit_kv2")
            phases.append({"phase": "rabit_kv2", "returncode": rc})
            if rc == 0:
                for d in ("bfloat16", "fp8_e4m3"):
                    phases.append({"phase": d, "returncode": engine(d)})
    _emit("PHASES", phases)
    print("EXP13_LPROBE_MODAL_DONE", flush=True)


@app.local_entrypoint()
def main():
    probe.remote()
