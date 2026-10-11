"""
MLSys 2027 Experiment 5 -- Modal app: BF16 vs RABIT-KV context-length scaling
(real engine, one H100 container).

Derived from exp4_deployment_modal.py / exp3_deployment_modal.py (NOT
modified). The `image = (...)` expression is copied verbatim from the
canonical embedded runner in benchmarks/performance/benchmark_deployment.py
(RUNNER_Z); run_experiment5_context_scaling.py verifies AST equality before
any run. The only image additions are appended as a final layer: the
Experiment 5 worker and the UNCHANGED Experiment 3 correctness gate and
watchdog files.

Inside ONE container / ONE physical GPU:
  1. the idle GPU baseline (memory.used, compute processes) is recorded;
  2. the frozen RABIT-KV correctness gate (exp3_correctness_gate.py) runs once
     in its own fresh process and must pass;
  3. two UNMEASURED conditioning cells (conditioning_A512 = bfloat16,
     conditioning_B512 = rabit_kv2: fresh engine, exact 512-token prompt,
     5 full-shape warmups, zero measured reps) absorb the first-engine /
     container effect; any conditioning failure stops the experiment;
  4. the 12 official cells run sequentially, each a fresh worker/engine
     process, in the order given by --legs ("label=dtype:context:prompt:role").
Before EVERY cell the GPU must have no compute process and memory.used within
GPU_CLEAN_TOLERANCE_MIB of the idle baseline, else the sweep stops. Gate and
every cell run under the hard watchdog (exp3_watchdog.py: own process group,
whole-group kill on timeout); a timeout or a surviving process stops the
sweep; nothing is retried.

After every cell, _verify_cell() re-checks the cell's own output inside the
container (fail-closed). The sweep CONTINUES past a failed cell ONLY for a
workload-level failure of an official measured cell: the worker reported
EXP5_WORKLOAD_FAILURE of kind request_oom / request_execution_failure with
exit code WORKLOAD_FAILURE_EXIT after a successful engine initialization, all
methodology checks on what the cell emitted passed, the whole process group
was reaped with no child/orphan process, and a fresh GPU clean-state check
passes. It STOPS immediately for everything else: engine initialization
failure, dtype/config mismatch, prompt token-count or prompt hash mismatch,
JIT during measurement, an incomplete cell, any conditioning failure, a
rejected request / programming error, or any other non-zero exit.
Gate lines are relayed with a "[gate] " prefix and cell lines with
"[leg<k>:<kv_cache_dtype>] " so the local runner can split the stream.

Launched only by benchmarks/mlsys2027/run_experiment5_context_scaling.py.
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
SNAP = Path(os.environ.get("EXP5_VLLM_SNAPSHOT", "/nonexistent/EXP5_VLLM_SNAPSHOT-not-set.zip"))
WORKER_LOCAL = Path(__file__).resolve().parent / "exp5_engine_worker.py"
WORKER_REMOTE = "/opt/exp5/exp5_engine_worker.py"
GATE_LOCAL = Path(__file__).resolve().parent / "exp3_correctness_gate.py"
GATE_REMOTE = "/opt/exp5/exp3_correctness_gate.py"
WATCHDOG_LOCAL = Path(__file__).resolve().parent / "exp3_watchdog.py"
WATCHDOG_REMOTE = "/opt/exp5/exp3_watchdog.py"
RABIT_KV2_REMOTE = "/root/vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py"
EXPECTED_RABIT_SHA256_LF = "7e628c94eebb9fe689bf416ea61f748c0f909a82d0f229c463edd1a0df92e6ae"
GPU_CLEAN_TOLERANCE_MIB = 256
GPU_CLEAN_MAX_WAIT_S = 60
GPU_CLEAN_POLL_S = 2
# Hard per-process watchdogs (process-group kill; see exp3_watchdog.py). The
# function timeout below is only a final backstop.
GATE_TIMEOUT_S = 600
LEG_TIMEOUT_S = 900
# Must equal exp5_engine_worker.WORKLOAD_FAILURE_EXIT (verified by the runner).
WORKLOAD_FAILURE_EXIT = 75
CONTINUABLE_FAILURE_KINDS = ("request_oom", "request_execution_failure")
OUTPUT_TOKENS = 32

if modal.is_local() and not SNAP.is_file():
    raise RuntimeError(
        "EXP5_VLLM_SNAPSHOT is not set to an existing snapshot zip. Launch via "
        "benchmarks/mlsys2027/run_experiment5_context_scaling.py."
    )

app = modal.App("rabit-kv-mlsys2027-exp5-context-scaling")
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
    _emit("EXP5_PRE_LEG_GPU_STATE", {
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
    _emit("EXP5_PROCESS_EXIT", meta)
    if meta["timed_out"]:
        _emit("EXP5_WATCHDOG_TIMEOUT", meta)
        raise RuntimeError(
            f"{label} exceeded its {timeout_s}s watchdog; process group {meta['pgid']} killed "
            f"({meta['signals_sent']}); aborting the whole experiment, no retry"
        )
    if meta["group_processes_remaining"]:
        raise RuntimeError(f"{label}: processes survived group kill: {meta['group_processes_remaining']}")
    return meta


class _Tee:
    """Mirror everything written to stdout into a buffer (cell verification)."""

    def __init__(self, stream):
        self.stream, self.chunks = stream, []

    def write(self, s):
        self.chunks.append(s)
        return self.stream.write(s)

    def flush(self):
        self.stream.flush()

    def lines(self) -> list[str]:
        return "".join(self.chunks).splitlines()


def _parse_cell(lines: list[str], prefix: str) -> dict:
    out = {"tags": {}, "samples": [], "warmups": [], "markers": [], "jit_in_measurement": 0, "failures": []}
    phase = None
    for raw in lines:
        if not raw.startswith(prefix):
            continue
        s = raw[len(prefix):].strip()
        if s.startswith("EXP5_SAMPLE "):
            out["samples"].append(json.loads(s[len("EXP5_SAMPLE "):]))
        elif s.startswith("EXP5_WARMUP "):
            out["warmups"].append(json.loads(s[len("EXP5_WARMUP "):]))
        elif s.startswith("EXP5_") and "={" in s:
            tag, payload = s.split("=", 1)
            if tag == "EXP5_WORKLOAD_FAILURE":
                out["failures"].append(json.loads(payload))
            elif tag != "EXP5_GPU_MEMORY":
                out["tags"][tag] = json.loads(payload)
        elif s in ("EXP5_WARMUP_BEGIN", "EXP5_WARMUP_END", "EXP5_MEASUREMENT_BEGIN", "EXP5_MEASUREMENT_END",
                   "EXP5_WORKER_COMPLETE"):
            out["markers"].append(s)
            phase = "measure" if s == "EXP5_MEASUREMENT_BEGIN" else (None if s.endswith("_END") else phase)
        elif phase == "measure" and "Triton kernel JIT compilation during inference" in s:
            out["jit_in_measurement"] += 1
    return out


def _verify_cell(lines: list[str], prefix: str, cell: dict, meta: dict, ref: dict) -> dict:
    """Fail-closed verdict for one finished cell: ok / workload_failure / stop.
    `ref` carries state across cells (reference config, per-dtype KV record and
    capacity, per-context prompt hash) and is updated only by accepted cells."""
    reasons: list[str] = []
    try:
        c = _parse_cell(lines, prefix)
    except (ValueError, KeyError) as exc:
        return {"verdict": "stop", "reasons": [f"unparseable cell output: {type(exc).__name__}: {exc}"],
                "workload_failure": None}
    t = c["tags"]
    label, dtype, role = cell["label"], cell["dtype"], cell["role"]
    if t.get("EXP5_LEG") != {"leg": label, "kv_cache_dtype": dtype, "role": role,
                             "context_point": cell["context"], "prompt_tokens": cell["prompt"]}:
        reasons.append(f"cell identity mismatch: {t.get('EXP5_LEG')}")
    need = ("EXP5_REQUESTED_ENGINE_KWARGS", "EXP5_EFFECTIVE_ENGINE_CONFIG", "EXP5_KV_DTYPE", "EXP5_CAPACITY",
            "EXP5_WORKLOAD")
    missing = [x for x in need if x not in t]
    if missing:
        return {"verdict": "stop", "reasons": reasons + [f"engine initialization failure (missing {missing})"],
                "workload_failure": None}
    req = dict(t["EXP5_REQUESTED_ENGINE_KWARGS"])
    if req.pop("kv_cache_dtype", None) != dtype:
        reasons.append("requested kv_cache_dtype mismatch")
    eff, kv, cap, wl = (t["EXP5_EFFECTIVE_ENGINE_CONFIG"], t["EXP5_KV_DTYPE"], t["EXP5_CAPACITY"],
                        t["EXP5_WORKLOAD"])
    if ref.get("requested") is not None and req != ref["requested"]:
        reasons.append("requested engine kwargs differ from the first cell (config mismatch)")
    if ref.get("effective") is not None and eff != ref["effective"]:
        reasons.append("effective engine config differs from the first cell (config mismatch)")
    if not (eff.get("compilation_mode") == "NONE" and eff.get("cudagraph_mode") == "NONE"
            and eff.get("enforce_eager") is True and str(eff.get("attention_backend", "")).endswith("TRITON_ATTN")
            and eff.get("calculate_kv_scales") is False and eff.get("hf_quantization_config") is None
            and eff.get("quantization") is None):
        reasons.append("effective engine config violates the fixed protocol (eager/Triton/no compile/no scales)")
    if not (kv.get("requested_kv_cache_dtype") == dtype == kv.get("engine_cache_dtype")):
        reasons.append(f"KV dtype mismatch: {kv}")
    if dtype in ref.get("kv", {}) and kv != ref["kv"][dtype]:
        reasons.append("resolved KV dtype record differs from the earlier cell of this dtype")
    if dtype in ref.get("capacity", {}) and cap != ref["capacity"][dtype]:
        reasons.append("allocator capacity differs from the earlier cell of this dtype (config mismatch)")
    want_reps = 0 if role == "conditioning" else cell["reps"]
    if not (wl.get("role") == role and wl.get("context_point") == cell["context"]
            and wl.get("prompt_tokens") == cell["prompt"] and wl.get("output_tokens") == OUTPUT_TOKENS
            and wl.get("max_tokens") == OUTPUT_TOKENS and wl.get("temperature") == 0.0
            and wl.get("ignore_eos") is True and wl.get("warmups") == cell["warmups"]
            and wl.get("reps") == want_reps):
        reasons.append(f"workload record mismatch: {wl}")
    h = wl.get("prompt_token_ids_sha256")
    if cell["context"] in ref.get("prompt_hash", {}) and h != ref["prompt_hash"][cell["context"]]:
        reasons.append("prompt hash differs from the other cell(s) at this context")
    for r in c["warmups"] + c["samples"]:
        if r.get("prompt_tokens") != cell["prompt"]:
            reasons.append(f"prompt token-count mismatch: {r.get('prompt_tokens')} != {cell['prompt']}")
            break
        if r.get("output_tokens") != OUTPUT_TOKENS:
            reasons.append(f"output token-count mismatch: {r.get('output_tokens')}")
            break
        if r.get("prompt_token_ids_sha256") != h:
            reasons.append("per-request prompt hash mismatch")
            break
    if c["jit_in_measurement"]:
        reasons.append(f"Triton JIT during measurement ({c['jit_in_measurement']} lines)")
    if reasons:
        return {"verdict": "stop", "reasons": reasons, "workload_failure": None}

    code = meta.get("returncode")
    complete = (c["markers"] == ["EXP5_WARMUP_BEGIN", "EXP5_WARMUP_END", "EXP5_MEASUREMENT_BEGIN",
                                 "EXP5_MEASUREMENT_END", "EXP5_WORKER_COMPLETE"]
                and len(c["warmups"]) == cell["warmups"] and len(c["samples"]) == want_reps and not c["failures"])
    if code == 0 and complete:
        ref.setdefault("requested", req)
        ref.setdefault("effective", eff)
        ref.setdefault("kv", {}).setdefault(dtype, kv)
        ref.setdefault("capacity", {}).setdefault(dtype, cap)
        ref.setdefault("prompt_hash", {}).setdefault(cell["context"], h)
        return {"verdict": "ok", "reasons": [], "workload_failure": None}
    if code == 0:
        return {"verdict": "stop", "reasons": ["exit 0 but incomplete cell output"], "workload_failure": None}
    fail = c["failures"][0] if len(c["failures"]) == 1 else None
    reaped = (not meta.get("timed_out") and meta.get("group_processes_after_leader_exit") == []
              and meta.get("group_processes_remaining") == [])
    if (code == WORKLOAD_FAILURE_EXIT and fail is not None and fail.get("kind") in CONTINUABLE_FAILURE_KINDS
            and role == "measured" and reaped):
        return {"verdict": "workload_failure", "reasons": [f"workload-level failure: {fail.get('kind')}"],
                "workload_failure": fail}
    if role == "conditioning":
        why = "conditioning cell failed"
    elif not reaped:
        why = "process group not cleanly reaped (child/orphan process)"
    else:
        why = f"non-continuable worker failure (exit {code}, reported {fail})"
    return {"verdict": "stop", "reasons": [why], "workload_failure": fail}


def _run_cells(plan: list[dict], model_dir: str, baseline_mib: list[int]) -> list[str]:
    """Conditioning + official cells in plan order; returns cells that failed at
    workload level (and were continued past). Raises (stops the sweep) on
    anything else."""
    ref: dict = {}
    failed_cells: list[str] = []
    for k, cell in enumerate(plan, start=1):
        label, dtype = cell["label"], cell["dtype"]
        _require_clean(label, baseline_mib)
        reps = 0 if cell["role"] == "conditioning" else cell["reps"]
        cmd = [
            sys.executable, WORKER_REMOTE,
            "--kv-cache-dtype", dtype,
            "--context-point", str(cell["context"]),
            "--prompt-tokens", str(cell["prompt"]),
            "--model-dir", model_dir,
            "--warmups", str(cell["warmups"]),
            "--reps", str(reps),
            "--leg", label,
        ] + (["--conditioning"] if cell["role"] == "conditioning" else [])
        _emit("EXP5_LEG_START", {"leg": label, "index": k, "kv_cache_dtype": dtype, "role": cell["role"],
                                 "context_point": cell["context"], "prompt_tokens": cell["prompt"], "cmd": cmd,
                                 "timeout_s": LEG_TIMEOUT_S})
        prefix = f"[leg{k}:{dtype}] "
        tee = _Tee(sys.stdout)
        sys.stdout = tee
        try:
            meta = _run_guarded(cmd, prefix, LEG_TIMEOUT_S, label)  # raises on timeout / surviving process
        finally:
            sys.stdout = tee.stream
        code = meta["returncode"]
        _emit("EXP5_LEG_EXIT", {"leg": label, "index": k, "kv_cache_dtype": dtype, "role": cell["role"],
                                "context_point": cell["context"], "returncode": code})
        verdict = _verify_cell(tee.lines(), prefix, cell, meta, ref)
        _emit("EXP5_CELL_VERDICT", {"leg": label, "index": k, **verdict})
        if verdict["verdict"] == "stop":
            _emit("EXP5_SWEEP_STOPPED", {"leg": label, "index": k, "reasons": verdict["reasons"]})
            raise RuntimeError(f"cell {label} stopped the sweep: {verdict['reasons']}")
        if verdict["verdict"] == "workload_failure":
            # Continue ONLY after the group was reaped (checked in _verify_cell)
            # and a fresh clean-state check passes (raises otherwise).
            _require_clean(f"{label}:post_failure", baseline_mib)
            failed_cells.append(label)
            _emit("EXP5_CELL_FAILED", {"leg": label, "index": k, "kv_cache_dtype": dtype,
                                       "context_point": cell["context"], "returncode": code,
                                       "workload_failure": verdict["workload_failure"]})
    return failed_cells


@app.function(
    image=image,
    gpu="H100",
    timeout=14400,
    volumes={"/model_cache": model_cache},
)
def sweep(legs: str, warmups: int, reps_per_leg: int) -> None:
    import importlib.metadata as md

    from modelscope import snapshot_download

    # legs: "conditioning_A512=bfloat16:512:512:conditioning,...,A512=bfloat16:512:512:measured,..."
    plan = []
    for item in (x.strip() for x in legs.split(",") if x.strip()):
        label, spec = item.split("=", 1)
        dtype, context, prompt_tokens, role = spec.split(":")
        if role not in ("conditioning", "measured"):
            raise RuntimeError(f"unknown cell role {role!r}")
        plan.append({"label": label, "dtype": dtype, "context": int(context), "prompt": int(prompt_tokens),
                     "role": role, "warmups": warmups, "reps": reps_per_leg})

    rabit_sha = _sha256_file(Path(RABIT_KV2_REMOTE), normalize_lf=True)
    if rabit_sha != EXPECTED_RABIT_SHA256_LF:
        raise RuntimeError(f"rabit_kv2.py in image is not the frozen source: {rabit_sha}")

    # Idle baseline BEFORE any process touches the GPU.
    baseline = _gpu_state()
    _emit("EXP5_GPU_BASELINE", {**baseline, "tolerance_mib": GPU_CLEAN_TOLERANCE_MIB,
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
        "EXP5_ENVIRONMENT",
        {
            "gpus": _gpu_query(),
            "python": sys.version.split()[0],
            "packages": versions,
            "rabit_kv2_sha256_lf": rabit_sha,
            "vllm_precompiled_wheel_commit": os.environ.get("VLLM_PRECOMPILED_WHEEL_COMMIT"),
            "leg_labels": [c["label"] for c in plan],
            "leg_roles": [c["role"] for c in plan],
            "leg_dtypes": [c["dtype"] for c in plan],
            "leg_context_points": [c["context"] for c in plan],
            "leg_prompt_tokens": [c["prompt"] for c in plan],
            "warmups": warmups,
            "reps_per_leg": reps_per_leg,
        },
    )

    # Correctness gate: must pass before any cell. Not timed.
    gate_cmd = [sys.executable, GATE_REMOTE]
    _emit("EXP5_GATE_START", {"cmd": gate_cmd, "timeout_s": GATE_TIMEOUT_S})
    code = _run_guarded(gate_cmd, "[gate] ", GATE_TIMEOUT_S, "gate")["returncode"]
    _emit("EXP5_GATE_EXIT", {"returncode": code})
    if code != 0:
        raise RuntimeError(f"correctness gate failed with exit code {code}; no cell run")

    os.environ["MODELSCOPE_CACHE"] = "/model_cache"
    model_dir = snapshot_download(MODEL, cache_dir="/model_cache")
    try:
        model_cache.commit()
    except Exception:  # noqa: BLE001
        pass

    # Checkpoint provenance: every cell loads exactly this directory.
    mdir = Path(model_dir)
    files = {}
    for p in sorted(mdir.iterdir()):
        if p.is_file() and (p.suffix in {".json", ".safetensors"} or p.name == "tokenizer.model"):
            files[p.name] = {"bytes": p.stat().st_size, "sha256": _sha256_file(p)}
    _emit("EXP5_MODEL", {"model": MODEL, "snapshot_dir": model_dir, "files": files})

    failed_cells = _run_cells(plan, model_dir, baseline["memory_used_mib"])

    _emit("EXP5_POST_RUN_GPU_STATE", _gpu_state())
    _emit("EXP5_SWEEP_COMPLETE", {"cells": len(plan), "failed_cells": failed_cells})


@app.local_entrypoint()
def main(legs: str, warmups: int, reps_per_leg: int):
    sweep.remote(legs=legs, warmups=warmups, reps_per_leg=reps_per_leg)
