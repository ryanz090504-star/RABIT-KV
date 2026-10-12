"""
MLSys 2027 Experiment 6 -- concurrency / throughput scaling, Modal app (one H100
container per prompt-length sweep).

Derived from exp6_smoke_modal.py (NOT modified): canonical image and clean-state /
watchdog helpers (AST-verified by run_experiment6_concurrency.py). Execution path
(reviewed amendment): vLLM's normal multi-request API via exp6_worker.py.

Inside ONE container / ONE physical GPU, for ONE prompt length:
  1. idle GPU baseline;
  2. frozen RABIT-KV correctness gate (exp3_correctness_gate.py, unchanged);
  3. the 36 pre-registered points in the given order (3 trials; per-trial dtype
     order BF16->RABIT, RABIT->BF16, BF16->RABIT; ascending concurrency
     1,4,8,16,32,64 within each dtype). Each point is ONE fresh engine process
     (exp6_worker.py: 256-request shadow conditioning, 2 warmup, 256 measured)
     under the process-group watchdog, after a GPU clean-state check. Every
     point is attempted independently: a failed point (engine / request
     failure, OOM, watchdog) is recorded and the sweep continues with the
     next point -- only a GPU that is not back to its idle baseline stops the run;
  4. post-run GPU state.
Nothing is retried.

REMOTE-LOG CAPTURE (harness-only, after the infrastructure-aborted L2048
attempt 3): the app is launched with `modal run --detach`, so the sweep keeps
running if the local client disconnects. The COMPLETE remote stdout (every
line the sweep and its watchdogged children print) is tee'd to the Modal Volume
SESSION_LOG_VOLUME at /<run_id>/remote_session.log, committed after the gate
and after every point; /<run_id>/DONE.json is written when the sweep function
ends (normally or by exception). The local runner downloads both and analyzes
the remote log. No protocol / workload / gate change.

L8192 EXECUTION AMENDMENT (infrastructure only; exp6_l8192_execution_amendment.json):
L8192 runs ONE trial (12 points) per Modal function with a 3600 s point watchdog
(static budget 600 + 12 x 3600 = 43800 s < 45000 s backstop). L2048 keeps its
historical execution (36 points, 1200 s). Only these two (points, watchdog)
configurations are accepted.

INDEPENDENT FUNCTIONCALL (orchestration only, after the L8192 trial-1 attempt-1
InputCancellation): the local entrypoint no longer blocks on a synchronous
`sweep.remote(...)`. It calls `sweep.spawn(...)`, immediately writes the launch
record (FunctionCall ID fc-..., App ID, run id, trial, git commit, launch UTC) to
the local path given by --launch-record, and returns; with `modal run --detach`
the spawned H100 FunctionCall continues independently of the local caller. The
runner monitors it by ID (modal.FunctionCall.from_id) and never cancels it.
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
SNAP = Path(os.environ.get("EXP6_VLLM_SNAPSHOT", "/nonexistent/EXP6_VLLM_SNAPSHOT-not-set.zip"))
WORKER_LOCAL = Path(__file__).resolve().parent / "exp6_worker.py"
WORKER_REMOTE = "/opt/exp6/exp6_worker.py"
WORKLOAD_LOCAL = Path(__file__).resolve().parent / "exp6_workload.py"
WORKLOAD_REMOTE = "/opt/exp6/exp6_workload.py"
GATE_LOCAL = Path(__file__).resolve().parent / "exp3_correctness_gate.py"
GATE_REMOTE = "/opt/exp6/exp3_correctness_gate.py"
WATCHDOG_LOCAL = Path(__file__).resolve().parent / "exp3_watchdog.py"
WATCHDOG_REMOTE = "/opt/exp6/exp3_watchdog.py"
RABIT_KV2_REMOTE = "/root/vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py"
EXPECTED_RABIT_SHA256_LF = "7e628c94eebb9fe689bf416ea61f748c0f909a82d0f229c463edd1a0df92e6ae"
GPU_CLEAN_TOLERANCE_MIB = 256
GPU_CLEAN_MAX_WAIT_S = 60
GPU_CLEAN_POLL_S = 2
# Hard per-process watchdogs (process-group kill; see exp3_watchdog.py). The
# function timeout below is only a final backstop.
GATE_TIMEOUT_S = 600
LEG_TIMEOUT_S = 900  # unused here; kept identical to Experiment 3 (verified)
REQUEST_CAP_S = 600  # unused here; kept identical (verified)
POINT_TIMEOUT_S = 1200  # per point: engine start + 256 shadow + 2 warmup + 256 measured requests
POINTS_PER_SWEEP = 36
L8192_POINT_TIMEOUT_S = 3600  # L8192 execution amendment (infrastructure only)
TRIAL_POINTS = 12  # L8192: one trial (2 dtypes x 6 C) per Modal run
ALLOWED_EXECUTIONS = ((POINTS_PER_SWEEP, POINT_TIMEOUT_S), (TRIAL_POINTS, L8192_POINT_TIMEOUT_S))
MODAL_BACKSTOP_S = 45000  # must equal the sweep() decorator timeout

if modal.is_local() and not SNAP.is_file():
    raise RuntimeError(
        "EXP6_VLLM_SNAPSHOT is not set to an existing snapshot zip. Launch via "
        "benchmarks/mlsys2027/run_experiment6_concurrency.py."
    )

app = modal.App("rabit-kv-mlsys2027-exp6-concurrency-scaling")
model_cache = modal.Volume.from_name(
    "modelscope-llama31-cache", create_if_missing=True
)
SESSION_LOG_VOLUME = "rabit-kv-mlsys2027-exp6-session-logs"
SESSION_LOG_MOUNT = "/session_logs"
session_logs = modal.Volume.from_name(SESSION_LOG_VOLUME, create_if_missing=True)

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
    .add_local_file(str(WORKLOAD_LOCAL), WORKLOAD_REMOTE, copy=True)
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


def _commit_session_logs() -> None:
    try:
        sys.stdout.flush()
        session_logs.commit()
    except Exception as exc:  # noqa: BLE001  (never let log persistence change the sweep)
        print(f"S3C_SESSION_LOG_COMMIT_ERROR={json.dumps({'error': repr(exc)[:500]})}", flush=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=45000,
    volumes={"/model_cache": model_cache, SESSION_LOG_MOUNT: session_logs},
)
def sweep(points: str, prompt_set_sha256: str, stage3c_impl: str, query_block: int, run_id: str,
          point_timeout_s: int = POINT_TIMEOUT_S, expected_points: int = POINTS_PER_SWEEP) -> None:
    import traceback

    if not run_id or "/" in run_id or ".." in run_id:
        raise RuntimeError(f"invalid run_id {run_id!r}")
    run_dir = Path(SESSION_LOG_MOUNT) / run_id
    if run_dir.exists():
        raise RuntimeError(f"session-log directory {run_dir} already exists; run ids are never reused")
    run_dir.mkdir(parents=True)
    fh = (run_dir / "remote_session.log").open("w", encoding="utf-8")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(orig_out, fh), _Tee(orig_err, fh)
    done = {"run_id": run_id, "status": "exception", "error": None}
    try:
        _emit("S3C_SESSION_LOG", {"run_id": run_id, "volume": SESSION_LOG_VOLUME,
                                  "path": f"/{run_id}/remote_session.log"})
        _sweep_body(points, prompt_set_sha256, stage3c_impl, query_block, int(point_timeout_s), int(expected_points))
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


def _sweep_body(points: str, prompt_set_sha256: str, stage3c_impl: str, query_block: int, point_timeout_s: int,
                expected_points: int) -> None:
    import importlib.metadata as md

    if (expected_points, point_timeout_s) not in ALLOWED_EXECUTIONS:
        raise RuntimeError(f"execution ({expected_points} points, {point_timeout_s} s) is not a pinned configuration")
    budget = GATE_TIMEOUT_S + expected_points * point_timeout_s
    if not budget < MODAL_BACKSTOP_S:
        raise RuntimeError(f"static watchdog budget {budget} s does not fit under the {MODAL_BACKSTOP_S} s backstop")

    from modelscope import snapshot_download

    # points: "label=dtype:prompt_tokens:concurrency:trial,..." in the pre-registered order
    plan = []
    for item in (x.strip() for x in points.split(",") if x.strip()):
        label, spec = item.split("=", 1)
        dtype, prompt_tokens, conc, trial = spec.split(":")
        plan.append({"label": label, "dtype": dtype, "prompt_tokens": int(prompt_tokens),
                     "concurrency": int(conc), "trial": int(trial)})
    if len(plan) != expected_points:
        raise RuntimeError(f"expected {expected_points} points, got {len(plan)}")
    rabit_sha = _sha256_file(Path(RABIT_KV2_REMOTE), normalize_lf=True)
    if rabit_sha != EXPECTED_RABIT_SHA256_LF:
        raise RuntimeError(f"rabit_kv2.py in image is not the frozen source: {rabit_sha}")
    baseline = _gpu_state()
    _emit("S3C_GPU_BASELINE", {**baseline, "tolerance_mib": GPU_CLEAN_TOLERANCE_MIB,
                               "max_wait_s": GPU_CLEAN_MAX_WAIT_S})
    if baseline["compute_apps"]:
        raise RuntimeError(f"GPU has compute processes before the sweep: {baseline['compute_apps']}")
    versions = {}
    for pkg in ("vllm", "torch", "triton", "transformers", "modelscope", "pytest"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    _emit("S3C_ENVIRONMENT", {"gpus": _gpu_query(), "python": sys.version.split()[0], "packages": versions,
                              "rabit_kv2_sha256_lf": rabit_sha, "stage3c_impl": stage3c_impl,
                              "query_block": int(query_block), "point_timeout_s": point_timeout_s,
                              "expected_points": expected_points, "static_watchdog_budget_s": budget,
                              "modal_backstop_s": MODAL_BACKSTOP_S,
                              "point_labels": [p["label"] for p in plan], "prompt_set_sha256": prompt_set_sha256})
    gate_cmd = [sys.executable, GATE_REMOTE]
    _emit("S3C_GATE_START", {"cmd": gate_cmd, "timeout_s": GATE_TIMEOUT_S})
    code = _run_guarded(gate_cmd, "[gate] ", GATE_TIMEOUT_S, "gate")["returncode"]
    _emit("S3C_GATE_EXIT", {"returncode": code})
    _commit_session_logs()
    if code != 0:
        raise RuntimeError(f"correctness gate failed with exit code {code}; no point run")
    os.environ["MODELSCOPE_CACHE"] = "/model_cache"
    model_dir = snapshot_download(MODEL, cache_dir="/model_cache")
    try:
        model_cache.commit()
    except Exception:  # noqa: BLE001
        pass
    sys.path.insert(0, str(Path(WATCHDOG_REMOTE).parent))
    from exp3_watchdog import run_with_watchdog

    failed = []
    for k, p in enumerate(plan, start=1):
        _require_clean(p["label"], baseline["memory_used_mib"])  # raises (stops) if the GPU is not clean
        cmd = [sys.executable, WORKER_REMOTE, "--kv-cache-dtype", p["dtype"], "--model-dir", model_dir,
               "--prompt-tokens", str(p["prompt_tokens"]), "--concurrency", str(p["concurrency"]),
               "--trial", str(p["trial"]), "--label", p["label"], "--prompt-set-sha256", prompt_set_sha256]
        if p["dtype"] == "rabit_kv2":
            cmd += ["--stage3c-impl", stage3c_impl, "--query-block", str(int(query_block))]
        _emit("S3C_SERIES_START", {"series": p["label"], "index": k, **p, "cmd": cmd, "timeout_s": point_timeout_s})
        # Same watchdog as _run_guarded, but a timed-out / failed point is recorded instead of stopping the sweep.
        meta = run_with_watchdog(cmd, f"[pt{k}:{p['label']}] ", point_timeout_s, p["label"])
        _emit("S3C_PROCESS_EXIT", meta)
        if meta["timed_out"]:
            _emit("S3C_WATCHDOG_TIMEOUT", meta)
        if meta["group_processes_remaining"]:
            _emit("S3C_STOPPED", {"series": p["label"], "reason": "processes survived group kill"})
            raise RuntimeError(f"{p['label']}: processes survived group kill; stopping")
        _emit("S3C_SERIES_EXIT", {"series": p["label"], "index": k, "returncode": meta["returncode"],
                                  "timed_out": meta["timed_out"]})
        if meta["returncode"] != 0 or meta["timed_out"]:
            failed.append(p["label"])
        _commit_session_logs()
    _require_clean("post_run", baseline["memory_used_mib"])
    _emit("S3C_POST_RUN_GPU_STATE", _gpu_state())
    _emit("S3C_SWEEP_COMPLETE", {"points": len(plan), "failed_points": failed})
    print("S3C_SWEEP_DONE", flush=True)


@app.local_entrypoint()
def main(points: str, prompt_set_sha256: str, stage3c_impl: str, query_block: int, run_id: str,
         launch_record: str, git_commit: str, trial: int = 0,
         point_timeout_s: int = POINT_TIMEOUT_S, expected_points: int = POINTS_PER_SWEEP):
    # Runs LOCALLY. Spawn (never .remote): the FunctionCall has its own durable handle and does not depend on this
    # process staying alive. The launch record is persisted before this entrypoint returns.
    record_path = Path(launch_record)
    if record_path.exists():
        raise RuntimeError(f"launch record {record_path} already exists; a FunctionCall was already spawned")
    call = sweep.spawn(points=points, prompt_set_sha256=prompt_set_sha256, stage3c_impl=stage3c_impl,
                       query_block=query_block, run_id=run_id, point_timeout_s=point_timeout_s,
                       expected_points=expected_points)
    record = {"function_call_id": call.object_id, "app_id": app.app_id, "app_name": app.name, "run_id": run_id,
              "trial": trial or None, "git_commit": git_commit,
              "launch_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "invocation": "sweep.spawn (modal run --detach); monitored by FunctionCall.from_id; never cancelled"}
    tmp = record_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(record_path)
    print(f"EXP6_LAUNCH_RECORD={json.dumps(record, sort_keys=True)}", flush=True)
