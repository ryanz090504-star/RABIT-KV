"""
Stage3C cliff DIAGNOSTIC -- Modal app (one H100 container). DIAGNOSTIC evidence
only; NOT Experiment 5 evidence.

Derived from exp5_deployment_modal.py (NOT modified): the `image = (...)`
expression is the canonical one (verified by AST) and the clean-state /
watchdog helpers are the Experiment 3 ones (verified by AST, modulo tag prefix).
The only image additions: the diagnostic worker and the UNCHANGED Experiment 3
correctness gate and watchdog files.

Inside ONE container / ONE physical GPU:
  1. idle GPU baseline;
  2. frozen RABIT-KV correctness gate (exp3_correctness_gate.py);
  3. one fresh engine process per dtype series, in --series order
     (bfloat16 control first, then rabit_kv2). Each series: one unmeasured
     conditioning request, then one measured request per point, each under the
     worker's 600 s per-request SIGALRM guard, with the process-group watchdog
     (the series process runs under the Experiment 3 watchdog, SERIES_TIMEOUT_S,
     whole-group kill and reap) and this Modal function timeout as hard
     process-level backstops.
Stage3C per-layer markers are once-per-layer-per-engine: the unmeasured
conditioning request consumes them, so they are unavailable for the measured
diagnostic points.
Before every series the GPU must be clean (no compute process, memory within
GPU_CLEAN_TOLERANCE_MIB of idle). ANY failure (gate, clean state, request cap,
request failure, non-zero exit, watchdog, surviving process) stops the whole
diagnostic; nothing is retried.

Launched only by benchmarks/mlsys2027/run_stage3c_cliff_diagnostic.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

MODEL = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
BASE_COMMIT = "f329ce405b12623fb8b1cf1830f12e5a712523be"
SNAP = Path(os.environ.get("S3C_VLLM_SNAPSHOT", "/nonexistent/S3C_VLLM_SNAPSHOT-not-set.zip"))
WORKER_LOCAL = Path(__file__).resolve().parent / "stage3c_diag_worker.py"
WORKER_REMOTE = "/opt/s3c/stage3c_diag_worker.py"
GATE_LOCAL = Path(__file__).resolve().parent / "exp3_correctness_gate.py"
GATE_REMOTE = "/opt/s3c/exp3_correctness_gate.py"
WATCHDOG_LOCAL = Path(__file__).resolve().parent / "exp3_watchdog.py"
WATCHDOG_REMOTE = "/opt/s3c/exp3_watchdog.py"
RABIT_KV2_REMOTE = "/root/vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py"
EXPECTED_RABIT_SHA256_LF = "7e628c94eebb9fe689bf416ea61f748c0f909a82d0f229c463edd1a0df92e6ae"
GPU_CLEAN_TOLERANCE_MIB = 256
GPU_CLEAN_MAX_WAIT_S = 60
GPU_CLEAN_POLL_S = 2
# Hard per-process watchdogs (process-group kill; see exp3_watchdog.py). The
# function timeout below is only a final backstop.
GATE_TIMEOUT_S = 600
LEG_TIMEOUT_S = 900  # unused here; kept identical to Experiment 3 (verified)
REQUEST_CAP_S = 600
SERIES_TIMEOUT_S = 7500  # >= (1 conditioning + 11 points) x REQUEST_CAP_S + engine start

if modal.is_local() and not SNAP.is_file():
    raise RuntimeError(
        "S3C_VLLM_SNAPSHOT is not set to an existing snapshot zip. Launch via "
        "benchmarks/mlsys2027/run_stage3c_cliff_diagnostic.py."
    )

app = modal.App("rabit-kv-mlsys2027-stage3c-cliff-diagnostic")
model_cache = modal.Volume.from_name(
    "modelscope-llama31-cache", create_if_missing=True
)

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
    print(f"{tag}={json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _sha256_file(path: Path, normalize_lf: bool = False) -> str:
    if normalize_lf:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _gpu_query() -> list[dict]:
    fields = "index,name,uuid,driver_version,memory.total,memory.used,clocks.max.sm,power.limit"
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        rows.append(dict(zip(fields.split(","), vals)))
    return rows


def _compute_apps() -> list[dict]:
    fields = "pid,process_name,used_memory"
    out = subprocess.run(
        ["nvidia-smi", f"--query-compute-apps={fields}", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    ).stdout
    return [dict(zip(fields.split(","), [v.strip() for v in line.split(",")]))
            for line in out.strip().splitlines() if line.strip()]


def _gpu_state() -> dict:
    return {
        "t": time.time(),
        "memory_used_mib": [int(g["memory.used"]) for g in _gpu_query()],
        "compute_apps": _compute_apps(),
    }


def _require_clean(label: str, baseline_mib: list[int]) -> None:
    """Before every leg: no compute process on the GPU and memory.used within
    GPU_CLEAN_TOLERANCE_MIB of the idle baseline. Polls (bounded) while a
    previous process releases memory; this is a wait, never a re-run."""
    readings = []
    deadline = time.time() + GPU_CLEAN_MAX_WAIT_S
    while True:
        s = _gpu_state()
        readings.append(s)
        clean = not s["compute_apps"] and all(
            u <= b + GPU_CLEAN_TOLERANCE_MIB for u, b in zip(s["memory_used_mib"], baseline_mib)
        )
        if clean or time.time() >= deadline:
            break
        time.sleep(GPU_CLEAN_POLL_S)
    _emit("S3C_PRE_LEG_GPU_STATE", {
        "leg": label, "clean": clean, "baseline_memory_used_mib": baseline_mib,
        "tolerance_mib": GPU_CLEAN_TOLERANCE_MIB, "max_wait_s": GPU_CLEAN_MAX_WAIT_S,
        "readings": readings,
    })
    if not clean:
        raise RuntimeError(
            f"GPU not back to idle baseline before {label}: last reading {readings[-1]}; "
            f"baseline {baseline_mib} MiB + {GPU_CLEAN_TOLERANCE_MIB} MiB tolerance"
        )


def _run_guarded(cmd: list[str], prefix: str, timeout_s: int, label: str) -> dict:
    """Run under the hard watchdog (own process group, group kill on timeout)."""
    sys.path.insert(0, str(Path(WATCHDOG_REMOTE).parent))
    from exp3_watchdog import run_with_watchdog

    meta = run_with_watchdog(cmd, prefix, timeout_s, label)
    _emit("S3C_PROCESS_EXIT", meta)
    if meta["timed_out"]:
        _emit("S3C_WATCHDOG_TIMEOUT", meta)
        raise RuntimeError(
            f"{label} exceeded its {timeout_s}s watchdog; process group {meta['pgid']} killed "
            f"({meta['signals_sent']}); aborting the whole experiment, no retry"
        )
    if meta["group_processes_remaining"]:
        raise RuntimeError(f"{label}: processes survived group kill: {meta['group_processes_remaining']}")
    return meta


@app.function(
    image=image,
    gpu="H100",
    timeout=16800,
    volumes={"/model_cache": model_cache},
)
def diagnose(series: str, points: str, conditioning_prompt: int) -> None:
    import importlib.metadata as md

    from modelscope import snapshot_download

    dtypes = [x.strip() for x in series.split(",") if x.strip()]
    rabit_sha = _sha256_file(Path(RABIT_KV2_REMOTE), normalize_lf=True)
    if rabit_sha != EXPECTED_RABIT_SHA256_LF:
        raise RuntimeError(f"rabit_kv2.py in image is not the frozen source: {rabit_sha}")

    baseline = _gpu_state()
    _emit("S3C_GPU_BASELINE", {**baseline, "tolerance_mib": GPU_CLEAN_TOLERANCE_MIB,
                               "max_wait_s": GPU_CLEAN_MAX_WAIT_S})
    if baseline["compute_apps"]:
        raise RuntimeError(f"GPU has compute processes before the diagnostic: {baseline['compute_apps']}")

    versions = {}
    for pkg in ("vllm", "torch", "triton", "transformers", "modelscope", "pytest"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    _emit("S3C_ENVIRONMENT", {
        "gpus": _gpu_query(), "python": sys.version.split()[0], "packages": versions,
        "rabit_kv2_sha256_lf": rabit_sha,
        "vllm_precompiled_wheel_commit": os.environ.get("VLLM_PRECOMPILED_WHEEL_COMMIT"),
        "series": dtypes, "points": [int(p) for p in points.split(",")],
        "conditioning_prompt_tokens": conditioning_prompt, "request_cap_s": REQUEST_CAP_S,
        "series_timeout_s": SERIES_TIMEOUT_S,
    })

    gate_cmd = [sys.executable, GATE_REMOTE]
    _emit("S3C_GATE_START", {"cmd": gate_cmd, "timeout_s": GATE_TIMEOUT_S})
    code = _run_guarded(gate_cmd, "[gate] ", GATE_TIMEOUT_S, "gate")["returncode"]
    _emit("S3C_GATE_EXIT", {"returncode": code})
    if code != 0:
        raise RuntimeError(f"correctness gate failed with exit code {code}; no series run")

    os.environ["MODELSCOPE_CACHE"] = "/model_cache"
    model_dir = snapshot_download(MODEL, cache_dir="/model_cache")
    try:
        model_cache.commit()
    except Exception:  # noqa: BLE001
        pass
    mdir = Path(model_dir)
    files = {}
    for p in sorted(mdir.iterdir()):
        if p.is_file() and (p.suffix in {".json", ".safetensors"} or p.name == "tokenizer.model"):
            files[p.name] = {"bytes": p.stat().st_size, "sha256": _sha256_file(p)}
    _emit("S3C_MODEL", {"model": MODEL, "snapshot_dir": model_dir, "files": files})

    for k, dtype in enumerate(dtypes, start=1):
        label = f"series_{dtype}"
        _require_clean(label, baseline["memory_used_mib"])
        cmd = [sys.executable, WORKER_REMOTE, "--kv-cache-dtype", dtype, "--model-dir", model_dir,
               "--points", points, "--conditioning-prompt", str(conditioning_prompt),
               "--request-cap-s", str(REQUEST_CAP_S), "--series", label]
        _emit("S3C_SERIES_START", {"series": label, "index": k, "kv_cache_dtype": dtype, "cmd": cmd,
                                   "timeout_s": SERIES_TIMEOUT_S})
        meta = _run_guarded(cmd, f"[series{k}:{dtype}] ", SERIES_TIMEOUT_S, label)
        _emit("S3C_SERIES_EXIT", {"series": label, "index": k, "kv_cache_dtype": dtype,
                                  "returncode": meta["returncode"]})
        if meta["returncode"] != 0:
            _emit("S3C_STOPPED", {"series": label, "returncode": meta["returncode"]})
            raise RuntimeError(f"series {label} failed with exit code {meta['returncode']}; diagnostic stopped")

    _emit("S3C_POST_RUN_GPU_STATE", _gpu_state())
    print("S3C_DIAGNOSTIC_COMPLETE", flush=True)


@app.local_entrypoint()
def main(series: str, points: str, conditioning_prompt: int):
    diagnose.remote(series=series, points=points, conditioning_prompt=conditioning_prompt)
