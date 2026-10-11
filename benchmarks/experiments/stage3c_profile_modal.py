"""
Stage3C COMPONENT PROFILE diagnostic -- Modal app (one H100 container).
Component attribution only; NOT a latency benchmark, NOT paper performance
evidence, NOT Experiment 5 evidence.

Derived from stage3c_tile32_bench_modal.py (NOT modified): canonical image and
clean-state / watchdog helpers (AST-verified).

Inside ONE container / ONE physical GPU:
  1. idle GPU baseline;
  2. frozen RABIT-KV correctness gate (exp3_correctness_gate.py, unchanged);
  3. tile32 correctness tests (test_rabit2_stage3c_tile32.py, unchanged);
  4. profiler tests (test_rabit2_stage3c_profile.py): profiling OFF is a no-op,
     profiling ON is bit-exact to OFF for reference and tile32, patches restored;
  5. unless --correctness-only: one fresh engine process per series, in --series
     order ("label=dtype:impl"), with VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE=1;
     each series: one unmeasured conditioning request, then one profiled request
     per point, each under the worker's 600 s SIGALRM guard, with the
     process-group watchdog and this Modal function timeout as hard backstops.
ANY failure stops the whole run; nothing is retried.
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
SNAP = Path(os.environ.get("S3P_VLLM_SNAPSHOT", "/nonexistent/S3P_VLLM_SNAPSHOT-not-set.zip"))
WORKER_LOCAL = Path(__file__).resolve().parent / "stage3c_profile_worker.py"
WORKER_REMOTE = "/opt/s3p/stage3c_profile_worker.py"
GATE_LOCAL = Path(__file__).resolve().parent / "exp3_correctness_gate.py"
GATE_REMOTE = "/opt/s3p/exp3_correctness_gate.py"
WATCHDOG_LOCAL = Path(__file__).resolve().parent / "exp3_watchdog.py"
WATCHDOG_REMOTE = "/opt/s3p/exp3_watchdog.py"
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
SERIES_TIMEOUT_S = 3600  # >= (1 conditioning + 4 points) x REQUEST_CAP_S + engine start
TILE32_TEST_TIMEOUT_S = 1800
TILE32_TEST_FILE = "/root/vllm-kvquant/tests/quantization/test_rabit2_stage3c_tile32.py"
PROFILE_TEST_TIMEOUT_S = 1800
PROFILE_TEST_FILE = "/root/vllm-kvquant/tests/quantization/test_rabit2_stage3c_profile.py"

if modal.is_local() and not SNAP.is_file():
    raise RuntimeError(
        "S3P_VLLM_SNAPSHOT is not set to an existing snapshot zip. Launch via "
        "benchmarks/mlsys2027/run_stage3c_profile_diagnostic.py."
    )

app = modal.App("rabit-kv-mlsys2027-stage3c-component-profile")
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
    timeout=12600,
    volumes={"/model_cache": model_cache},
)
def benchmark(series: str, points: str, conditioning_prompt: int, correctness_only: bool) -> None:
    import importlib.metadata as md

    from modelscope import snapshot_download

    plan = []
    for item in (x.strip() for x in series.split(",") if x.strip()):
        label, spec = item.split("=", 1)
        dtype, impl = spec.split(":")
        plan.append((label, dtype, impl))
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
        "series": [label for label, _, _ in plan], "series_dtypes": [d for _, d, _ in plan],
        "series_impls": [i for _, _, i in plan], "points": [int(p) for p in points.split(",")],
        "correctness_only": correctness_only, "tile32_test_timeout_s": TILE32_TEST_TIMEOUT_S,
        "profile_test_timeout_s": PROFILE_TEST_TIMEOUT_S, "component_profiling": True,
        "conditioning_prompt_tokens": conditioning_prompt, "request_cap_s": REQUEST_CAP_S,
        "series_timeout_s": SERIES_TIMEOUT_S,
    })

    gate_cmd = [sys.executable, GATE_REMOTE]
    _emit("S3C_GATE_START", {"cmd": gate_cmd, "timeout_s": GATE_TIMEOUT_S})
    code = _run_guarded(gate_cmd, "[gate] ", GATE_TIMEOUT_S, "gate")["returncode"]
    _emit("S3C_GATE_EXIT", {"returncode": code})
    if code != 0:
        raise RuntimeError(f"correctness gate failed with exit code {code}; no series run")

    # tile32 correctness vs the reference Stage3C path (exact equality required).
    t_cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-rA",
             "--confcutdir=/root/vllm-kvquant/tests/quantization", TILE32_TEST_FILE]
    _emit("S3C_TILE_TESTS_START", {"cmd": t_cmd, "timeout_s": TILE32_TEST_TIMEOUT_S})
    code = _run_guarded(t_cmd, "[tile32-tests] ", TILE32_TEST_TIMEOUT_S, "tile32_tests")["returncode"]
    _emit("S3C_TILE_TESTS_EXIT", {"returncode": code})
    if code != 0:
        raise RuntimeError(f"tile32 correctness tests failed with exit code {code}; no benchmark run")

    # Profiler: OFF is a no-op; ON is bit-exact to OFF (reference and tile32).
    p_cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-rA",
             "--confcutdir=/root/vllm-kvquant/tests/quantization", PROFILE_TEST_FILE]
    _emit("S3C_PROFILE_TESTS_START", {"cmd": p_cmd, "timeout_s": PROFILE_TEST_TIMEOUT_S})
    code = _run_guarded(p_cmd, "[profile-tests] ", PROFILE_TEST_TIMEOUT_S, "profile_tests")["returncode"]
    _emit("S3C_PROFILE_TESTS_EXIT", {"returncode": code})
    if code != 0:
        raise RuntimeError(f"profiler tests failed with exit code {code}; no profiling run")
    if correctness_only:
        _emit("S3C_POST_RUN_GPU_STATE", _gpu_state())
        print("S3C_CORRECTNESS_ONLY_COMPLETE", flush=True)
        return

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

    for k, (label, dtype, impl) in enumerate(plan, start=1):
        _require_clean(label, baseline["memory_used_mib"])
        cmd = [sys.executable, WORKER_REMOTE, "--kv-cache-dtype", dtype, "--model-dir", model_dir,
               "--points", points, "--conditioning-prompt", str(conditioning_prompt),
               "--request-cap-s", str(REQUEST_CAP_S), "--series", label, "--stage3c-impl", impl]
        _emit("S3C_SERIES_START", {"series": label, "index": k, "kv_cache_dtype": dtype, "impl": impl,
                                   "cmd": cmd, "timeout_s": SERIES_TIMEOUT_S})
        meta = _run_guarded(cmd, f"[series{k}:{label}] ", SERIES_TIMEOUT_S, label)
        _emit("S3C_SERIES_EXIT", {"series": label, "index": k, "kv_cache_dtype": dtype, "impl": impl,
                                  "returncode": meta["returncode"]})
        if meta["returncode"] != 0:
            _emit("S3C_STOPPED", {"series": label, "returncode": meta["returncode"]})
            raise RuntimeError(f"series {label} failed with exit code {meta['returncode']}; diagnostic stopped")

    _emit("S3C_POST_RUN_GPU_STATE", _gpu_state())
    print("S3C_PROFILE_COMPLETE", flush=True)


@app.local_entrypoint()
def main(series: str, points: str, conditioning_prompt: int, correctness_only: bool = False):
    benchmark.remote(series=series, points=points, conditioning_prompt=conditioning_prompt,
                     correctness_only=correctness_only)
