"""
RABIT-KV performance-risk PROFILING DIAGNOSTIC -- Modal app (ONE H100 container, all legs sequentially).
DIAGNOSTIC attribution only; NOT paper performance evidence.

Derived from exp6_modal.py (NOT modified): the image recipe is identical (AST-verified by the tests) so the served
vLLM build is the accepted one; the same clean-state check and process-group watchdog are used. Inside the container:
  1. hardware guard (exactly one NVIDIA H100) and idle GPU baseline;
  2. source identity: LF-normalized SHA-256 of rabit_kv2.py, triton_attn.py and rabit_kv2_stage3c_shared_decode.py in
     the image must equal the hashes of the accepted serving evidence (nothing in the vLLM tree is edited to profile);
  3. frozen RABIT-KV correctness gate (exp3_correctness_gate.py, unchanged);
  4. the legs in the given order, each ONE fresh engine process (perf_profile_worker.py) under the watchdog after a
     GPU clean-state check. A failed leg is recorded and the run continues; nothing is retried.
The complete remote stdout is tee'd to the Modal Volume LOG_VOLUME at /<run_id>/remote_session.log (committed after
every leg) and /<run_id>/DONE.json is written when the function ends; the local entrypoint only spawns the call.
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
SNAP = Path(os.environ.get("PERF_VLLM_SNAPSHOT", "/nonexistent/PERF_VLLM_SNAPSHOT-not-set.zip"))
HERE = Path(__file__).resolve().parent
REMOTE_DIR = "/opt/perf"
LOCAL_FILES = ("perf_profile_worker.py", "perf_profile_plugin.py", "exp6_workload.py", "exp3_correctness_gate.py",
               "exp3_watchdog.py")
VLLM_REMOTE = "/root/vllm-kvquant"
IDENTITY_FILES = {
    "rabit_kv2.py": "vllm/v1/attention/ops/rabit_kv2.py",
    "triton_attn.py": "vllm/v1/attention/backends/triton_attn.py",
    "rabit_kv2_stage3c_shared_decode.py": "vllm/v1/attention/ops/rabit_kv2_stage3c_shared_decode.py",
}
GPU_CLEAN_TOLERANCE_MIB = 256
GPU_CLEAN_MAX_WAIT_S = 60
GPU_CLEAN_POLL_S = 2
GATE_TIMEOUT_S = 600
MODAL_BACKSTOP_S = 21600  # must equal the run() decorator timeout
LOG_VOLUME = "rabit-kv-mlsys2027-perf-profile-logs"
LOG_MOUNT = "/session_logs"

if modal.is_local() and not SNAP.is_file():
    raise RuntimeError(
        "PERF_VLLM_SNAPSHOT is not set to an existing snapshot zip. Launch via "
        "benchmarks/mlsys2027/run_perf_profile_diagnostic.py."
    )

app = modal.App("rabit-kv-mlsys2027-perf-risk-profile")
model_cache = modal.Volume.from_name(
    "modelscope-llama31-cache", create_if_missing=True
)
session_logs = modal.Volume.from_name(LOG_VOLUME, create_if_missing=True)

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

for _name in LOCAL_FILES:
    image = image.add_local_file(str(HERE / _name), f"{REMOTE_DIR}/{_name}", copy=True)


def _emit(tag: str, payload) -> None:
    print(f"{tag}={json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _gpu_query() -> list[dict]:
    fields = "index,name,uuid,driver_version,memory.total,memory.used"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         text=True, capture_output=True, check=True).stdout
    return [dict(zip(fields.split(","), [v.strip() for v in line.split(",")])) for line in out.strip().splitlines()]


def _compute_apps() -> list[dict]:
    fields = "pid,process_name,used_memory"
    out = subprocess.run(["nvidia-smi", f"--query-compute-apps={fields}", "--format=csv,noheader,nounits"],
                         text=True, capture_output=True, check=True).stdout
    return [dict(zip(fields.split(","), [v.strip() for v in line.split(",")]))
            for line in out.strip().splitlines() if line.strip()]


def _gpu_state() -> dict:
    return {"t": time.time(), "memory_used_mib": [int(g["memory.used"]) for g in _gpu_query()],
            "compute_apps": _compute_apps()}


def _require_clean(label: str, baseline_mib: list[int]) -> None:
    """Before every leg: no compute process on the GPU and memory.used within GPU_CLEAN_TOLERANCE_MIB of the idle
    baseline. Polls (bounded) while a previous process releases memory; this is a wait, never a re-run."""
    deadline = time.time() + GPU_CLEAN_MAX_WAIT_S
    while True:
        s = _gpu_state()
        clean = not s["compute_apps"] and all(
            u <= b + GPU_CLEAN_TOLERANCE_MIB for u, b in zip(s["memory_used_mib"], baseline_mib))
        if clean or time.time() >= deadline:
            break
        time.sleep(GPU_CLEAN_POLL_S)
    _emit("PERFRUN_PRE_LEG_GPU_STATE", {"leg": label, "clean": clean, "baseline_memory_used_mib": baseline_mib, **s})
    if not clean:
        raise RuntimeError(f"GPU not back to idle baseline before {label}: {s}")


class _Tee:
    """Writes every line to the original stream AND the remote session log file."""

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, text):
        self._stream.write(text)
        self._fh.write(text)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return False


def _commit_logs() -> None:
    try:
        sys.stdout.flush()
        session_logs.commit()
    except Exception as exc:  # noqa: BLE001  (never let log persistence change the run)
        print(f"PERFRUN_LOG_COMMIT_ERROR={json.dumps({'error': repr(exc)[:500]})}", flush=True)


def parse_legs(legs: str) -> list[dict]:
    """label=case:dtype:mode:conditioning:measured:timeout_s,..."""
    plan = []
    for item in (x.strip() for x in legs.split(",") if x.strip()):
        label, spec = item.split("=", 1)
        case, dtype, mode, cond, meas, timeout_s = spec.split(":")
        plan.append({"label": label, "case": case, "dtype": dtype, "mode": mode, "conditioning": int(cond),
                     "measured": int(meas), "timeout_s": int(timeout_s)})
    return plan


@app.function(
    image=image,
    gpu="H100!:1",
    timeout=21600,
    volumes={"/model_cache": model_cache, LOG_MOUNT: session_logs},
)
def run(legs: str, identity: str, run_id: str) -> None:
    import traceback

    if not run_id or "/" in run_id or ".." in run_id:
        raise RuntimeError(f"invalid run_id {run_id!r}")
    run_dir = Path(LOG_MOUNT) / run_id
    if run_dir.exists():
        raise RuntimeError(f"session-log directory {run_dir} already exists; run ids are never reused")
    run_dir.mkdir(parents=True)
    fh = (run_dir / "remote_session.log").open("w", encoding="utf-8")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(orig_out, fh), _Tee(orig_err, fh)
    done = {"run_id": run_id, "status": "exception", "error": None}
    try:
        _body(legs, identity)
        done["status"] = "complete"
    except BaseException as exc:
        done["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = orig_out, orig_err
        fh.close()
        done["finished_unix"] = time.time()
        (run_dir / "DONE.json").write_text(json.dumps(done, sort_keys=True) + "\n", encoding="utf-8")
        try:
            session_logs.commit()
        except Exception:  # noqa: BLE001
            pass


def _body(legs: str, identity: str) -> None:
    import importlib.metadata as md

    sys.path.insert(0, REMOTE_DIR)
    from exp3_watchdog import run_with_watchdog
    from modelscope import snapshot_download

    plan = parse_legs(legs)
    budget = GATE_TIMEOUT_S + sum(p["timeout_s"] for p in plan)
    if not budget < MODAL_BACKSTOP_S:
        raise RuntimeError(f"static watchdog budget {budget} s does not fit under the {MODAL_BACKSTOP_S} s backstop")
    gpus = _gpu_query()
    if len(gpus) != 1 or "H100" not in gpus[0]["name"]:
        raise RuntimeError(f"hardware guard: expected exactly one NVIDIA H100, got {gpus}")
    expected = json.loads(identity)
    actual = {name: _sha256_lf(Path(VLLM_REMOTE) / rel) for name, rel in IDENTITY_FILES.items()}
    _emit("PERFRUN_SOURCE_IDENTITY", {"expected": expected, "actual": actual, "match": actual == expected})
    if actual != expected:
        raise RuntimeError("vLLM serving sources in the image are not the accepted ones")
    baseline = _gpu_state()
    if baseline["compute_apps"]:
        raise RuntimeError(f"GPU has compute processes before the run: {baseline['compute_apps']}")
    versions = {}
    for pkg in ("vllm", "torch", "triton", "transformers", "modelscope"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    _emit("PERFRUN_ENVIRONMENT", {"gpus": gpus, "python": sys.version.split()[0], "packages": versions,
                                  "baseline": baseline, "static_watchdog_budget_s": budget,
                                  "legs": [p["label"] for p in plan],
                                  "harness_sha256_lf": {n: _sha256_lf(Path(REMOTE_DIR) / n) for n in LOCAL_FILES}})
    gate_cmd = [sys.executable, f"{REMOTE_DIR}/exp3_correctness_gate.py"]
    _emit("PERFRUN_GATE_START", {"cmd": gate_cmd, "timeout_s": GATE_TIMEOUT_S})
    meta = run_with_watchdog(gate_cmd, "[gate] ", GATE_TIMEOUT_S, "gate")
    _emit("PERFRUN_GATE_EXIT", {"returncode": meta["returncode"], "timed_out": meta["timed_out"]})
    _commit_logs()
    if meta["returncode"] != 0 or meta["timed_out"]:
        raise RuntimeError("correctness gate failed; no leg run")
    os.environ["MODELSCOPE_CACHE"] = "/model_cache"
    model_dir = snapshot_download(MODEL, cache_dir="/model_cache")

    failed = []
    for k, p in enumerate(plan, start=1):
        _require_clean(p["label"], baseline["memory_used_mib"])
        cmd = [sys.executable, f"{REMOTE_DIR}/perf_profile_worker.py", "--case", p["case"],
               "--kv-cache-dtype", p["dtype"], "--mode", p["mode"], "--model-dir", model_dir, "--label", p["label"],
               "--conditioning", str(p["conditioning"]), "--measured", str(p["measured"])]
        _emit("PERFRUN_LEG_START", {"index": k, **p, "cmd": cmd, "unix_time": time.time()})
        meta = run_with_watchdog(cmd, f"[leg:{p['label']}] ", p["timeout_s"], p["label"])
        _emit("PERFRUN_LEG_EXIT", {"label": p["label"], "index": k, "returncode": meta["returncode"],
                                   "timed_out": meta["timed_out"], "elapsed_s": meta.get("elapsed_s"),
                                   "unix_time": time.time()})
        if meta["group_processes_remaining"]:
            raise RuntimeError(f"{p['label']}: processes survived group kill; stopping")
        if meta["returncode"] != 0 or meta["timed_out"]:
            failed.append(p["label"])
        _commit_logs()
    _require_clean("post_run", baseline["memory_used_mib"])
    _emit("PERFRUN_COMPLETE", {"legs": len(plan), "failed_legs": failed})


@app.local_entrypoint()
def main(legs: str, identity: str, run_id: str, launch_record: str, git_commit: str):
    # Runs LOCALLY. Spawn (never .remote): the FunctionCall does not depend on this process staying alive.
    record_path = Path(launch_record)
    if record_path.exists():
        raise RuntimeError(f"launch record {record_path} already exists; a FunctionCall was already spawned")
    call = run.spawn(legs=legs, identity=identity, run_id=run_id)
    record = {"function_call_id": call.object_id, "app_id": app.app_id, "app_name": app.name, "run_id": run_id,
              "git_commit": git_commit, "launch_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PERFRUN_LAUNCH_RECORD={json.dumps(record, sort_keys=True)}", flush=True)
