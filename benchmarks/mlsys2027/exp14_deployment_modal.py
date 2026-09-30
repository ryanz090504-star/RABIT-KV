"""
MLSys 2027 Experiment 14 -- Modal app: Model B (Qwen2.5-7B-Instruct) matched BF16 vs RABIT-KV physical capacity and
single-request latency in ONE H100 container (generalization of the FROZEN RABIT policy K3/V2/G32/R4/META8g64).

Derived MECHANICALLY from the accepted exp13_deployment_modal.py (NOT modified); the `image = (...)` expression is
verbatim (the runner verifies equality with Exp4 / Exp13). The vllm-kvquant snapshot is a `git archive` of the
committed tree at the execution commit (unchanged since the fa4321de3 / PR #47609 backport, local commit 611a4ff).

Inside ONE container / ONE physical GPU:
  1. idle GPU baseline;
  2. the frozen RABIT-KV correctness gate (exp3_correctness_gate.py, unchanged; Llama shape), then the Exp14 Model-B
     SHAPE gate (exp14_shape_gate.py: Qwen2.5 shape 28/4/128 with the Llama shape as positive control); each in its own
     fresh process under the hard watchdog; either failing aborts the experiment; nothing is retried;
  3. the counterbalanced legs A1 B1 B2 A2 (A = bfloat16, B = rabit_kv2), each a fresh worker / engine process (vLLM
     default multiprocess engine core, default V2 model runner), with the clean-GPU check before EVERY leg.
Gate lines are relayed with "[gate] " / "[shapegate] " prefixes and leg lines with "[leg<k>:<kv_cache_dtype>] ".
Launched only by benchmarks/mlsys2027/run_experiment14_second_model.py.
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

MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_COMMIT = "f329ce405b12623fb8b1cf1830f12e5a712523be"
SNAP = Path(os.environ.get("EXP14_VLLM_SNAPSHOT", "/nonexistent/EXP14_VLLM_SNAPSHOT-not-set.zip"))
WORKER_LOCAL = Path(__file__).resolve().parent / "exp14_engine_worker.py"
WORKER_REMOTE = "/opt/exp14/exp14_engine_worker.py"
GATE_LOCAL = Path(__file__).resolve().parent / "exp3_correctness_gate.py"
GATE_REMOTE = "/opt/exp14/exp3_correctness_gate.py"
WATCHDOG_LOCAL = Path(__file__).resolve().parent / "exp3_watchdog.py"
WATCHDOG_REMOTE = "/opt/exp14/exp3_watchdog.py"
SHAPE_GATE_LOCAL = Path(__file__).resolve().parent / "exp14_shape_gate.py"
SHAPE_GATE_REMOTE = "/opt/exp14/exp14_shape_gate.py"
PROBE_WORKER_LOCAL = Path(__file__).resolve().parent / "exp14_probe_worker.py"  # non-evidence probe only
PROBE_WORKER_REMOTE = "/opt/exp14/exp14_probe_worker.py"
RABIT_KV2_REMOTE = "/root/vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py"
EXPECTED_RABIT_SHA256_LF = "7e628c94eebb9fe689bf416ea61f748c0f909a82d0f229c463edd1a0df92e6ae"
GPU_CLEAN_TOLERANCE_MIB = 256
GPU_CLEAN_MAX_WAIT_S = 60
GPU_CLEAN_POLL_S = 2
# Hard per-process watchdogs (process-group kill; see exp3_watchdog.py). The
# function timeout below is only a final backstop.
GATE_TIMEOUT_S = 600
SHAPE_GATE_TIMEOUT_S = 900
LEG_TIMEOUT_S = 900

if modal.is_local() and not SNAP.is_file():
    raise RuntimeError(
        "EXP14_VLLM_SNAPSHOT is not set to an existing snapshot zip. Launch via "
        "benchmarks/mlsys2027/run_experiment14_second_model.py."
    )

app = modal.App("rabit-kv-mlsys2027-exp14-second-model")
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
    .add_local_file(str(SHAPE_GATE_LOCAL), SHAPE_GATE_REMOTE, copy=True)
    .add_local_file(str(PROBE_WORKER_LOCAL), PROBE_WORKER_REMOTE, copy=True)
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
    _emit("EXP14_PRE_LEG_GPU_STATE", {
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
    _emit("EXP14_PROCESS_EXIT", meta)
    if meta["timed_out"]:
        _emit("EXP14_WATCHDOG_TIMEOUT", meta)
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
    timeout=14400,
    volumes={"/model_cache": model_cache},
)
def mirrored(legs: str, warmups: int, reps_per_leg: int, probe: bool = False) -> None:
    """probe=True: NON-EVIDENCE feasibility probe -- identical gates / provenance / GPU-clean checks, but each
    "leg" runs exp14_probe_worker.py (engine init + sanity generations, no timing) instead of the measured worker."""
    import importlib.metadata as md

    from modelscope import snapshot_download

    # legs: "A1=bfloat16,B1=rabit_kv2,B2=rabit_kv2,A2=bfloat16"
    plan = [tuple(x.strip().split("=", 1)) for x in legs.split(",") if x.strip()]
    leg_dtypes = [d for _, d in plan]

    rabit_sha = _sha256_file(Path(RABIT_KV2_REMOTE), normalize_lf=True)
    if rabit_sha != EXPECTED_RABIT_SHA256_LF:
        raise RuntimeError(f"rabit_kv2.py in image is not the frozen source: {rabit_sha}")

    # Idle baseline BEFORE any process touches the GPU.
    baseline = _gpu_state()
    _emit("EXP14_GPU_BASELINE", {**baseline, "tolerance_mib": GPU_CLEAN_TOLERANCE_MIB,
                                "max_wait_s": GPU_CLEAN_MAX_WAIT_S})
    if baseline["compute_apps"]:
        raise RuntimeError(f"GPU has compute processes before the experiment: {baseline['compute_apps']}")

    versions = {}
    for pkg in ("vllm", "torch", "triton", "transformers", "modelscope", "pytest"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    _emit(
        "EXP14_ENVIRONMENT",
        {
            "gpus": _gpu_query(),
            "python": sys.version.split()[0],
            "packages": versions,
            "rabit_kv2_sha256_lf": rabit_sha,
            "vllm_precompiled_wheel_commit": os.environ.get("VLLM_PRECOMPILED_WHEEL_COMMIT"),
            "leg_labels": [label for label, _ in plan],
            "leg_dtypes": leg_dtypes,
            "warmups": warmups,
            "reps_per_leg": reps_per_leg,
            "mode": "feasibility_probe_non_evidence" if probe else "measured",
        },
    )

    # Correctness gate: must pass before any measurement. Not timed.
    gate_cmd = [sys.executable, GATE_REMOTE]
    _emit("EXP14_GATE_START", {"cmd": gate_cmd, "timeout_s": GATE_TIMEOUT_S})
    code = _run_guarded(gate_cmd, "[gate] ", GATE_TIMEOUT_S, "gate")["returncode"]
    _emit("EXP14_GATE_EXIT", {"returncode": code})
    if code != 0:
        raise RuntimeError(f"correctness gate failed with exit code {code}; no measurement run")

    # Model-B SHAPE gate (Qwen2.5 28/4/128 fallback dispatch; Llama 32/8/128 positive control). Not timed.
    shape_cmd = [sys.executable, SHAPE_GATE_REMOTE]
    _emit("EXP14_SHAPE_GATE_START", {"cmd": shape_cmd, "timeout_s": SHAPE_GATE_TIMEOUT_S})
    code = _run_guarded(shape_cmd, "[shapegate] ", SHAPE_GATE_TIMEOUT_S, "shape_gate")["returncode"]
    _emit("EXP14_SHAPE_GATE_EXIT", {"returncode": code})
    if code != 0:
        raise RuntimeError(f"Model-B shape gate failed with exit code {code}; no measurement run")

    os.environ["MODELSCOPE_CACHE"] = "/model_cache"
    model_dir = snapshot_download(MODEL, cache_dir="/model_cache")
    try:
        model_cache.commit()
    except Exception:  # noqa: BLE001
        pass

    # Checkpoint provenance: every leg loads exactly this directory.
    mdir = Path(model_dir)
    files = {}
    for p in sorted(mdir.iterdir()):
        if p.is_file() and (p.suffix in {".json", ".safetensors"} or p.name == "tokenizer.model"):
            files[p.name] = {"bytes": p.stat().st_size, "sha256": _sha256_file(p)}
    _emit("EXP14_MODEL", {"model": MODEL, "snapshot_dir": model_dir, "files": files})

    for k, (label, dtype) in enumerate(plan, start=1):
        _require_clean(label, baseline["memory_used_mib"])
        if probe:
            cmd = [sys.executable, PROBE_WORKER_REMOTE, "--kv-cache-dtype", dtype, "--model-dir", model_dir]
        else:
            cmd = [
                sys.executable, WORKER_REMOTE,
                "--kv-cache-dtype", dtype,
                "--model-dir", model_dir,
                "--warmups", str(warmups),
                "--reps", str(reps_per_leg),
                "--leg", label,
            ]
        _emit("EXP14_LEG_START", {"leg": label, "index": k, "kv_cache_dtype": dtype, "cmd": cmd,
                                 "timeout_s": LEG_TIMEOUT_S})
        code = _run_guarded(cmd, f"[leg{k}:{dtype}] ", LEG_TIMEOUT_S, label)["returncode"]
        _emit("EXP14_LEG_EXIT", {"leg": label, "index": k, "kv_cache_dtype": dtype, "returncode": code})
        if code != 0:
            raise RuntimeError(f"leg {label} ({dtype}) failed with exit code {code}; not continuing")

    _emit("EXP14_POST_RUN_GPU_STATE", _gpu_state())
    print("EXP14_MIRRORED_COMPLETE", flush=True)


@app.local_entrypoint()
def main(legs: str, warmups: int, reps_per_leg: int, probe: bool = False):
    mirrored.remote(legs=legs, warmups=warmups, reps_per_leg=reps_per_leg, probe=probe)
