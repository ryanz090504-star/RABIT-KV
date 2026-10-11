"""
RABIT-KV MLSys 2027 -- Experiment 3 runner: matched BF16 vs RABIT-KV physical
capacity and latency on the real vLLM engine (H100, Modal).

This is a PHYSICAL real-engine experiment (allocator capacity, TTFT, TPOT,
wall time), not a fake-quant quality experiment.

Design (one `modal run` of benchmarks/mlsys2027/exp3_deployment_modal.py):
  * one container, one physical H100, one image, one model snapshot;
  * idle GPU baseline recorded first;
  * RABIT-KV correctness gate (benchmarks/mlsys2027/exp3_correctness_gate.py,
    the canonical regression() verbatim) must pass before any measurement;
  * counterbalanced ABBA legs, A = bfloat16, B = rabit_kv2: A1, B1, B2, A2,
    each a fresh worker/engine process (exp3_engine_worker.py) with 5
    full-shape warmups (excluded) + 15 measured reps -> 30 per dtype;
  * gate and every leg run under a hard watchdog (exp3_watchdog.py): own
    process group, whole-group kill on timeout (gate 600 s, leg 900 s), and a
    timeout aborts the whole experiment;
  * before EVERY leg: no GPU compute process and memory back within a small
    tolerance of the idle baseline, else hard fail;
  * engine arguments byte-identical except kv_cache_dtype.

Before any run this script proves by AST that the worker's engine arguments,
the Modal image and the correctness gate are identical to the canonical
embedded runner in benchmarks/performance/benchmark_deployment.py (except
kv_cache_dtype). After the run it evaluates every integrity check with an
explicit state -- passed / failed / not_run / not_evaluated -- and only a run
in which EVERY check is "passed" produces a summary. Every terminal path
(success, Modal non-zero exit, gate failure, watchdog timeout, parser
failure, integrity/config failure, local exception) runs the protected-path
post-check and persists it in manifest.json.

This script does NOT modify benchmark_deployment.py, vllm-kvquant or any
result; it writes ONLY top-level files under results/mlsys2027/deployment/
and never touches archived attempts (results/mlsys2027/deployment/failed_attempt_*/).

Usage:
    python benchmarks/mlsys2027/run_experiment3_deployment.py --dry-run
    python benchmarks/mlsys2027/run_experiment3_deployment.py
"""

from __future__ import annotations

import argparse
import ast
import base64
import datetime as dt
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER_SCRIPT = Path(__file__).resolve()
MODAL_APP = ROOT / "benchmarks" / "mlsys2027" / "exp3_deployment_modal.py"
WORKER = ROOT / "benchmarks" / "mlsys2027" / "exp3_engine_worker.py"
GATE = ROOT / "benchmarks" / "mlsys2027" / "exp3_correctness_gate.py"
WATCHDOG = ROOT / "benchmarks" / "mlsys2027" / "exp3_watchdog.py"
CANONICAL_DEPLOYMENT = ROOT / "benchmarks" / "performance" / "benchmark_deployment.py"
CANONICAL_CAPACITY = ROOT / "results" / "performance" / "capacity.json"
CANONICAL_LATENCY = ROOT / "results" / "performance" / "latency.json"
RABIT_KV2 = ROOT / "vllm-kvquant" / "vllm" / "v1" / "attention" / "ops" / "rabit_kv2.py"
EXPECTED_RABIT_SHA256_LF = "7e628c94eebb9fe689bf416ea61f748c0f909a82d0f229c463edd1a0df92e6ae"

OUT_DIR = ROOT / "results" / "mlsys2027" / "deployment"
ARCHIVE_GLOB = "failed_attempt_*"
SESSION_LOG = OUT_DIR / "modal_session.log"
GATE_LOG = OUT_DIR / "correctness_gate.log"
MANIFEST = OUT_DIR / "manifest.json"
CONFIG_DIFF = OUT_DIR / "matched_config_diff.json"
INTEGRITY = OUT_DIR / "integrity_check.json"
SUMMARY = OUT_DIR / "matched_capacity_latency_summary.json"

PROTECTED_PATHS = [
    ROOT / "results" / "quality",
    ROOT / "results" / "performance",
    ROOT / "results" / "summary.json",
    ROOT / "vllm-kvquant",
    ROOT / "benchmarks" / "performance",
    ROOT / "results" / "mlsys2027" / "quality_frontier",
    ROOT / "results" / "mlsys2027" / "multilingual_frontier",
]
MUST_BE_COMMITTED = [RUNNER_SCRIPT, MODAL_APP, WORKER, GATE, WATCHDOG]

# ABBA counterbalanced plan: A = bfloat16, B = rabit_kv2.
A, B = "bfloat16", "rabit_kv2"
DTYPES = [A, B]
LEGS = [(1, "A1", A), (2, "B1", B), (3, "B2", B), (4, "A2", A)]
LOG_NAME = {A: "bf16_deployment.log", B: "rabit_kv2_deployment.log"}
WARMUPS_PER_LEG = 5
REPS_PER_LEG = 15
REPS_PER_DTYPE = 30
CONTEXT_TOKENS = 2048
OUTPUT_TOKENS = 32

# Must equal the Modal app constants (verified by AST).
GPU_CLEAN_TOLERANCE_MIB = 256
GPU_CLEAN_MAX_WAIT_S = 60
GATE_TIMEOUT_S = 600
LEG_TIMEOUT_S = 900

# Config comparison: only these kv_cache_dtype-induced fields may differ, and
# only between dtypes (never between the two legs of one dtype).
DTYPE_INDUCED_ALLOWLIST = [
    "requested.kv_cache_dtype",
    "kv_dtype.requested_kv_cache_dtype",
    "kv_dtype.engine_cache_dtype",
    "kv_dtype.resolved_kv_torch_dtype",
]

# Llama-3.1-8B: 32 layers x (K,V) x 8 KV heads x 128 head_dim x 2 bytes.
BF16_BYTES_PER_TOKEN = 32 * 2 * 8 * 128 * 2
BYTES_PER_TOKEN_REL_TOL = 0.01

EXPECTED_EFFECTIVE = {
    "model_dtype": "torch.bfloat16",
    "max_model_len": 32768,
    "enforce_eager": True,
    "block_size": 32,
    "gpu_memory_utilization": 0.82,
    "enable_prefix_caching": False,
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 32,
    "enable_chunked_prefill": True,
    "attention_backend": "AttentionBackendEnum.TRITON_ATTN",
    "tensor_parallel_size": 1,
    "log_stats": True,
    "quantization": None,
}
EXPECTED_KV = {
    A: {"engine_cache_dtype": "bfloat16", "resolved_kv_torch_dtype": "torch.bfloat16"},
    B: {"engine_cache_dtype": "rabit_kv2", "resolved_kv_torch_dtype": "torch.uint8"},
}
EXPECTED_WORKLOAD = {
    "context_tokens": CONTEXT_TOKENS, "output_tokens": OUTPUT_TOKENS, "warmups": WARMUPS_PER_LEG,
    "reps": REPS_PER_LEG, "temperature": 0.0, "ignore_eos": True, "max_tokens": OUTPUT_TOKENS,
}
CANONICAL_GATE_PYTEST_PASSED = 105  # informational (results/performance/deployment.log)

# vllm/config/compilation.py member values (name -> value), for normalization
# of raw representations without importing vllm locally.
COMPILATION_MODE_VALUES = {"NONE": 0, "STOCK_TORCH_COMPILE": 1, "DYNAMO_TRACE_ONCE": 2, "VLLM_COMPILE": 3}
CUDAGRAPH_MODE_VALUES = {"NONE": 0, "PIECEWISE": 1, "FULL": 2, "FULL_DECODE_ONLY": (2, 0),
                         "FULL_AND_PIECEWISE": (2, 1)}

CAPACITY_LABEL = "PHYSICAL vLLM allocator KV capacity (num_gpu_blocks x block_size) from the real engine"
LATENCY_LABEL = (
    "Real-engine single-request latency: TTFT = frontend first_token_latency; "
    "TPOT = (last_token_ts - first_token_ts)/(n-1) from engine-core timestamps; "
    "wall = perf_counter around llm.generate. p90 = statistics.quantiles(n=10, "
    "method='inclusive')[8]. Headline statistics are over the pooled 30 samples per dtype."
)

PASSED, FAILED, NOT_RUN, NOT_EVALUATED = "passed", "failed", "not_run", "not_evaluated"


# ---------------------------------------------------------------- utilities
def run_git(*args: str) -> str:
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout.strip()


def sha256(path: Path) -> str:
    """LF-normalized SHA-256 (equals committed git content under any autocrlf)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_raw(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    """Repo-relative path; never raises (a failure path must not crash here)."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def assert_protected_paths_clean(context: str) -> None:
    status = run_git("status", "--short", "--", *[rel(p) for p in PROTECTED_PATHS])
    if status:
        raise RuntimeError(
            f"CRITICAL: protected paths changed ({context}). Investigate immediately:\n" + status
        )


def uncommitted_experiment_files() -> str:
    return run_git("status", "--short", "--", *[rel(p) for p in MUST_BE_COMMITTED])


def archived_attempts_digest() -> dict:
    """SHA-256 of every file in archived attempt folders (must never change)."""
    out = {}
    if OUT_DIR.is_dir():
        for d in sorted(OUT_DIR.glob(ARCHIVE_GLOB)):
            for f in sorted(p for p in d.rglob("*") if p.is_file()):
                out[f.relative_to(OUT_DIR).as_posix()] = sha256_raw(f)
    return out


def top_level_output_files() -> list[str]:
    if not OUT_DIR.is_dir():
        return []
    return sorted(p.name for p in OUT_DIR.iterdir() if p.is_file())


def normalize_mode(value, table: dict) -> str:
    """Normalize a compilation/CUDA-graph mode representation to its member
    NAME without relying on Enum.__str__. Accepts a member name ("NONE"),
    "Class.NAME", repr "<Class.NAME: v>", or a raw value (0, "0", "(2, 0)")."""
    if isinstance(value, dict):  # {"repr": ..., "str": ...}
        names = {normalize_mode(v, table) for v in value.values()}
        return names.pop() if len(names) == 1 else f"INCONSISTENT:{sorted(names)}"
    if isinstance(value, (int, tuple)) and not isinstance(value, bool):
        for name, v in table.items():
            if v == value:
                return name
        return f"UNRECOGNIZED:{value!r}"
    s = str(value).strip()
    m = re.fullmatch(r"<\w+\.(\w+): .*>", s)
    if m:
        s = m.group(1)
    elif re.fullmatch(r"\w+\.\w+", s):
        s = s.split(".")[-1]
    if s in table:
        return s
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return f"UNRECOGNIZED:{value!r}"
    return normalize_mode(parsed, table) if isinstance(parsed, (int, tuple)) else f"UNRECOGNIZED:{value!r}"


# ------------------------------------------------ canonical equivalence (AST)
def canonical_runner_source() -> str:
    tree = ast.parse(CANONICAL_DEPLOYMENT.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "RUNNER_Z":
            return zlib.decompress(base64.b64decode(node.value.value)).decode("utf-8")
    raise RuntimeError("RUNNER_Z not found in canonical benchmark_deployment.py")


def _module_assign(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return node.value
    raise RuntimeError(f"module-level assignment {name!r} not found")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"function {name!r} not found")


def canonical_llm_kwargs(src: str) -> dict:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LLM":
            return {kw.arg: "<model>" if kw.arg == "model" else ast.literal_eval(kw.value)
                    for kw in node.keywords}
    raise RuntimeError("LLM(...) call not found in canonical runner")


def verify_canonical_equivalence() -> dict:
    """Prove worker/image/gate equal the canonical deployment runner except
    kv_cache_dtype, that the KV-spec RPC is gone, and that the clean-state and
    watchdog constants match this runner."""
    csrc = canonical_runner_source()
    ctree = ast.parse(csrc)
    mtree = ast.parse(MODAL_APP.read_text(encoding="utf-8"))
    wsrc = WORKER.read_text(encoding="utf-8")
    wtree = ast.parse(wsrc)
    gtree = ast.parse(GATE.read_text(encoding="utf-8"))

    canon_kwargs = canonical_llm_kwargs(csrc)
    worker_base = ast.literal_eval(_module_assign(wtree, "BASE_ENGINE_KWARGS"))
    canon_wo = {k: v for k, v in canon_kwargs.items() if k not in ("model", "kv_cache_dtype")}
    if canon_wo != worker_base:
        raise RuntimeError(f"Worker BASE_ENGINE_KWARGS differ from canonical:\n{canon_wo}\n{worker_base}")
    if canon_kwargs.get("kv_cache_dtype") != "rabit_kv2":
        raise RuntimeError("Canonical runner kv_cache_dtype is not rabit_kv2")

    if ast.dump(_module_assign(ctree, "image")) != ast.dump(_module_assign(mtree, "image")):
        raise RuntimeError("Experiment 3 Modal image expression differs from the canonical runner's")
    for const in ("MODEL", "BASE_COMMIT"):
        if ast.literal_eval(_module_assign(ctree, const)) != ast.literal_eval(_module_assign(mtree, const)):
            raise RuntimeError(f"{const} differs from the canonical runner")

    cfn, gfn = _function(ctree, "regression"), _function(gtree, "regression")
    if not (ast.dump(ast.Module(body=cfn.body, type_ignores=[])) ==
            ast.dump(ast.Module(body=gfn.body, type_ignores=[])) and ast.dump(cfn.args) == ast.dump(gfn.args)):
        raise RuntimeError("exp3_correctness_gate.regression() is not verbatim the canonical regression()")

    for const, expected in (("GPU_CLEAN_TOLERANCE_MIB", GPU_CLEAN_TOLERANCE_MIB),
                            ("GPU_CLEAN_MAX_WAIT_S", GPU_CLEAN_MAX_WAIT_S),
                            ("GATE_TIMEOUT_S", GATE_TIMEOUT_S),
                            ("LEG_TIMEOUT_S", LEG_TIMEOUT_S)):
        if ast.literal_eval(_module_assign(mtree, const)) != expected:
            raise RuntimeError(f"Modal app {const} differs from runner")

    # Attempt-1 root cause must stay removed: no engine RPC from the worker.
    for node in ast.walk(wtree):
        if isinstance(node, ast.Attribute) and node.attr == "collective_rpc":
            raise RuntimeError("worker still calls collective_rpc (attempt-1 hang cause)")
    if "EXP3_KV_SPEC" in wsrc:
        raise RuntimeError("worker still emits EXP3_KV_SPEC")

    for snippet in (
        "m.first_token_latency * 1000.0",
        "(m.last_token_ts - m.first_token_ts) / (n - 1) * 1000.0",
        "(time.perf_counter() - t0) * 1000.0",
        'tok.encode(" the", add_special_tokens=False)[-1]',
        "SamplingParams(temperature=0.0, max_tokens=",
    ):
        if snippet not in csrc or snippet not in wsrc:
            raise RuntimeError(f"timing/workload definition not shared with canonical: {snippet}")

    return {
        "canonical_source": f"{rel(CANONICAL_DEPLOYMENT)}#RUNNER_Z (zlib+base64, decoded in memory)",
        "canonical_runner_sha256": hashlib.sha256(csrc.encode("utf-8")).hexdigest(),
        "canonical_llm_kwargs": canon_kwargs,
        "worker_base_engine_kwargs": worker_base,
        "llm_kwargs_equal_except_kv_cache_dtype_and_model_path": True,
        "modal_image_expression_ast_equal": True,
        "model_and_base_commit_equal": True,
        "correctness_gate_regression_ast_equal_to_canonical": True,
        "worker_has_no_engine_rpc": True,
        "timing_definitions_shared": True,
    }


def requested_kwargs(dtype: str, model_dir: str = "<modelscope snapshot dir>") -> dict:
    base = ast.literal_eval(_module_assign(ast.parse(WORKER.read_text(encoding="utf-8")), "BASE_ENGINE_KWARGS"))
    return {"model": model_dir, **base, "kv_cache_dtype": dtype}


# --------------------------------------------------------------- preflight
def preflight(dry_run: bool) -> dict:
    branch = run_git("branch", "--show-current")
    head = run_git("rev-parse", "HEAD")
    assert_protected_paths_clean("preflight, before any run")

    rabit_sha = sha256(RABIT_KV2)
    if rabit_sha != EXPECTED_RABIT_SHA256_LF:
        raise RuntimeError(f"rabit_kv2.py is not the frozen committed content: {rabit_sha}")

    equivalence = verify_canonical_equivalence()

    uncommitted = uncommitted_experiment_files()
    if uncommitted and not dry_run:
        raise RuntimeError(
            "Refusing to run: Experiment 3 code has uncommitted changes, so recorded SHAs "
            "would not match a commit:\n" + uncommitted
        )
    leftovers = top_level_output_files()
    if leftovers and not dry_run:
        raise RuntimeError(
            f"Refusing to run: {rel(OUT_DIR)}/ already contains files from a previous attempt "
            f"({leftovers}). Archive them under {ARCHIVE_GLOB.replace('*', 'N')}/ first; this "
            "runner never overwrites earlier evidence."
        )

    return {
        "git_branch": branch,
        "git_head": head,
        "vllm_kvquant_tree": run_git("rev-parse", "HEAD:vllm-kvquant"),
        "rabit_kv2_sha256": rabit_sha,
        "runner_script_sha256": sha256(RUNNER_SCRIPT),
        "modal_app_sha256": sha256(MODAL_APP),
        "worker_sha256": sha256(WORKER),
        "correctness_gate_sha256": sha256(GATE),
        "watchdog_sha256": sha256(WATCHDOG),
        "canonical_benchmark_deployment_sha256": sha256(CANONICAL_DEPLOYMENT),
        "sha256_basis": "LF-normalized bytes (CRLF -> LF); equals committed git content",
        "canonical_equivalence": equivalence,
        "protected_paths_baseline_status": "clean",
        "archived_attempts_sha256": archived_attempts_digest(),
        "existing_top_level_output_files": leftovers or None,
        "uncommitted_experiment_files": uncommitted or None,
    }


# ------------------------------------------------------------ run plumbing
def make_console_encoding_safe() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def console_write(text: str) -> None:
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        sys.stdout.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))
    sys.stdout.flush()


def stream_command(cmd: list[str], log_path: Path, extra_env: dict) -> int:
    env = os.environ.copy()
    env.update(extra_env)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    console_write("\n" + "=" * 118 + "\n")
    console_write(f"RUNNING: {' '.join(cmd)}\nLOG: {log_path}\n" + "=" * 118 + "\n")
    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", env=env, bufsize=1)
        assert p.stdout is not None
        try:
            for line in p.stdout:
                log.write(line)  # log first, unchanged
                log.flush()
                console_write(line)
        except BaseException:
            p.kill()
            p.wait()
            raise
        return p.wait()


def build_snapshot() -> dict:
    """git archive of the committed vllm-kvquant tree (LF, no untracked or
    ignored files), written outside the repository."""
    out = Path(tempfile.mkdtemp(prefix="exp3_vllm_snapshot_")) / "vllm_kvquant_snapshot.zip"
    run_git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(out), "HEAD:vllm-kvquant")
    return {"path": str(out), "sha256": sha256_raw(out), "bytes": out.stat().st_size,
            "source": "git archive --format=zip HEAD:vllm-kvquant (core.autocrlf=false)"}


def build_command() -> list[str]:
    return [sys.executable, "-m", "modal", "run", str(MODAL_APP),
            "--legs", ",".join(d for _, _, d in LEGS),
            "--warmups", str(WARMUPS_PER_LEG), "--reps-per-leg", str(REPS_PER_LEG)]


def worker_command(label: str, dtype: str, model_dir: str = "<modelscope snapshot dir>") -> list[str]:
    return ["python", "/opt/exp3/exp3_engine_worker.py", "--kv-cache-dtype", dtype,
            "--model-dir", model_dir, "--warmups", str(WARMUPS_PER_LEG),
            "--reps", str(REPS_PER_LEG), "--leg", label]


GATE_COMMAND_DOC = ["python", "/opt/exp3/exp3_correctness_gate.py"]


# ------------------------------------------------------------------ parsing
TAG = re.compile(r"^(EXP3_[A-Z_]+)=(\{.*\})\s*$")
ROWLINE = re.compile(r"^(EXP3_SAMPLE|EXP3_WARMUP) (\{.*\})\s*$")
KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
KV_MEM = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
PYTEST_SUMMARY = re.compile(r"^(\d+) passed(?:, (\d+) warnings?)?(?:.* in ([\d.]+)s)?")
PYTEST_EXIT = re.compile(r"^pytest exit=(-?\d+)$")
JIT = "Triton kernel JIT compilation during inference"


def demux(session_text: str) -> tuple[dict, list[str], list[str]]:
    legs = {k: [] for k, _, _ in LEGS}
    gate, top = [], []
    prefixes = {f"[leg{k}:{d}] ": k for k, _, d in LEGS}
    for line in session_text.splitlines():
        if line.startswith("[gate] "):
            gate.append(line[len("[gate] "):])
            continue
        for prefix, k in prefixes.items():
            if line.startswith(prefix):
                legs[k].append(line[len(prefix):])
                break
        else:
            top.append(line)
    return legs, gate, top


def parse_worker(lines: list[str]) -> dict:
    out: dict = {"tags": {}, "samples": [], "warmups": [], "markers": [], "jit_during_measurement": [],
                 "kv_log_tokens": None, "kv_log_gib": None, "line_count": len(lines)}
    phase = None
    for i, line in enumerate(lines, start=1):
        m = TAG.match(line)
        if m:
            out["tags"][m.group(1)] = json.loads(m.group(2))
            continue
        m = ROWLINE.match(line)
        if m:
            (out["samples"] if m.group(1) == "EXP3_SAMPLE" else out["warmups"]).append(json.loads(m.group(2)))
            continue
        s = line.strip()
        if s in ("EXP3_WARMUP_BEGIN", "EXP3_WARMUP_END", "EXP3_MEASUREMENT_BEGIN",
                 "EXP3_MEASUREMENT_END", "EXP3_WORKER_COMPLETE"):
            out["markers"].append(s)
            phase = "measure" if s == "EXP3_MEASUREMENT_BEGIN" else (None if s.endswith("_END") else phase)
            continue
        if phase == "measure" and JIT in line:
            out["jit_during_measurement"].append({"line": i, "text": line.strip()})
        m = KV_TOKENS.search(line)
        if m:
            out["kv_log_tokens"] = int(m.group(1).replace(",", ""))
        m = KV_MEM.search(line)
        if m:
            out["kv_log_gib"] = float(m.group(1))
    return out


def parse_gate(lines: list[str]) -> dict:
    out: dict = {"line_count": len(lines), "begin": None, "result": None, "pytest_passed": None,
                 "pytest_warnings": None, "pytest_seconds": None, "pytest_exit": None,
                 "regression_passed_line": False}
    for line in lines:
        s = line.strip()
        m = TAG.match(s)
        if m and m.group(1) == "EXP3_GATE_BEGIN":
            out["begin"] = json.loads(m.group(2))
        elif m and m.group(1) == "EXP3_GATE_RESULT":
            out["result"] = json.loads(m.group(2))
        m = PYTEST_SUMMARY.match(s)
        if m:
            out["pytest_passed"] = int(m.group(1))
            out["pytest_warnings"] = int(m.group(2)) if m.group(2) else 0
            out["pytest_seconds"] = float(m.group(3)) if m.group(3) else None
        m = PYTEST_EXIT.match(s)
        if m:
            out["pytest_exit"] = int(m.group(1))
        if s == "RABIT-2 FINAL TARGETED REGRESSION PASSED":
            out["regression_passed_line"] = True
    return out


def parse_top(lines: list[str]) -> dict:
    out: dict = {"pre_leg": {}, "leg_exit": {}, "leg_start": {}, "process_exit": {}, "watchdog_timeouts": [],
                 "abba_complete": False}
    for line in lines:
        s = line.strip()
        m = TAG.match(s)
        if not m:
            if s == "EXP3_ABBA_COMPLETE":
                out["abba_complete"] = True
            continue
        tag, payload = m.group(1), json.loads(m.group(2))
        if tag == "EXP3_PRE_LEG_GPU_STATE":
            out["pre_leg"][payload["leg"]] = payload
        elif tag == "EXP3_LEG_EXIT":
            out["leg_exit"][payload["leg"]] = payload
        elif tag == "EXP3_LEG_START":
            out["leg_start"][payload["leg"]] = payload
        elif tag == "EXP3_PROCESS_EXIT":
            out["process_exit"][payload["label"]] = payload
        elif tag == "EXP3_WATCHDOG_TIMEOUT":
            out["watchdog_timeouts"].append(payload)
        else:
            out[tag] = payload
    return out


# ---------------------------------------------------------- config diff
def flatten(prefix: str, d: dict) -> dict:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(flatten(key, v))
        else:
            flat[key] = v
    return flat


def leg_config(p: dict) -> dict:
    t = p["tags"]
    return {
        **flatten("requested", t.get("EXP3_REQUESTED_ENGINE_KWARGS", {})),
        **flatten("effective", t.get("EXP3_EFFECTIVE_ENGINE_CONFIG", {})),
        **flatten("workload", t.get("EXP3_WORKLOAD", {})),
        **flatten("kv_dtype", t.get("EXP3_KV_DTYPE", {})),
    }


CONFIG_SECTIONS = ("EXP3_REQUESTED_ENGINE_KWARGS", "EXP3_EFFECTIVE_ENGINE_CONFIG", "EXP3_WORKLOAD", "EXP3_KV_DTYPE")


def config_diff(parsed: dict) -> dict:
    """All four legs: every non-allowlisted field identical across all legs;
    allowlisted (kv_cache_dtype-induced) fields identical within each dtype.
    Not evaluated at all unless every leg reported every config section."""
    dtype_of = {label: d for _, label, d in LEGS}
    missing = {label: [s for s in CONFIG_SECTIONS if s not in parsed[k]["tags"]] for k, label, _ in LEGS}
    missing = {label: v for label, v in missing.items() if v}
    base = {
        "rule": ("All non-dtype engine/workload fields must be identical across all four ABBA legs. "
                 "Only DTYPE_INDUCED_ALLOWLIST fields may differ, and only between dtypes "
                 "(never between the two legs of the same dtype). Capacity/latency are outcomes, "
                 "compared separately. Not evaluated unless all four legs reported their config."),
        "legs": dtype_of,
        "dtype_induced_allowlist": DTYPE_INDUCED_ALLOWLIST,
    }
    if missing:
        return {**base, "status": NOT_EVALUATED, "matched": None,
                "missing_config_sections_by_leg": missing,
                "configs": {label: leg_config(parsed[k]) for k, label, _ in LEGS}}
    configs = {label: leg_config(parsed[k]) for k, label, _ in LEGS}
    keys = sorted(set().union(*configs.values()))
    violations, differing_across_dtypes = [], []
    for key in keys:
        vals = {label: cfg.get(key, "<missing>") for label, cfg in configs.items()}
        if len({json.dumps(v, sort_keys=True, default=str) for v in vals.values()}) == 1:
            continue
        if key not in DTYPE_INDUCED_ALLOWLIST:
            violations.append({"field": key, "reason": "non-dtype field differs between legs", "values": vals})
            continue
        within_ok = True
        for d in DTYPES:
            if len({json.dumps(vals[l], sort_keys=True, default=str) for l in vals if dtype_of[l] == d}) != 1:
                within_ok = False
                violations.append({"field": key, "reason": f"allowlisted field differs within dtype {d}",
                                   "values": vals})
        if within_ok:
            differing_across_dtypes.append(key)
    return {**base, "status": PASSED if not violations else FAILED, "matched": not violations,
            "configs": configs, "fields_compared": len(keys),
            "fields_differing_between_dtypes": differing_across_dtypes, "violations": violations}


# ---------------------------------------------------------- checks / stats
def stats(rows: list[dict], key: str) -> dict:
    v = [r[key] for r in rows]
    return {
        "median": statistics.median(v),
        "p90": statistics.quantiles(v, n=10, method="inclusive")[8],
        "mean": statistics.mean(v),
        "stdev": statistics.stdev(v),
        "min": min(v),
        "max": max(v),
        "n": len(v),
    }


def gpu_leg_clean(pre: dict, baseline: dict) -> bool:
    """Independent re-check of the Modal app's clean-state decision."""
    if not pre or not pre.get("readings") or not baseline:
        return False
    last = pre["readings"][-1]
    return (not last["compute_apps"]
            and pre.get("tolerance_mib") == GPU_CLEAN_TOLERANCE_MIB
            and all(u <= b + GPU_CLEAN_TOLERANCE_MIB
                    for u, b in zip(last["memory_used_mib"], baseline["memory_used_mib"])))


def integrity(parsed: dict, gate: dict, top: dict, diff: dict) -> dict:
    """Every check has an explicit state. Only 'passed' counts as passed;
    all_ok requires every check to be 'passed'."""
    checks: list[dict] = []

    def add(name: str, category: str, state, observed=None) -> None:
        if isinstance(state, bool):
            state = PASSED if state else FAILED
        checks.append({"check": name, "category": category, "state": state, "observed": observed})

    env = top.get("EXP3_ENVIRONMENT", {})
    gpus = env.get("gpus", [])
    baseline = top.get("EXP3_GPU_BASELINE", {})
    add("ABBA completed", "completion", top.get("abba_complete", False))
    add("no watchdog timeout", "watchdog", not top["watchdog_timeouts"], top["watchdog_timeouts"] or None)
    add("exactly one GPU, H100", "environment",
        (len(gpus) == 1 and "H100" in gpus[0].get("name", "")) if env else NOT_EVALUATED,
        [g.get("name") for g in gpus] or None)
    add("leg order is ABBA", "environment",
        env.get("leg_dtypes") == [d for _, _, d in LEGS] if env else NOT_EVALUATED, env.get("leg_dtypes"))
    add("rabit_kv2.py in image is frozen source", "environment",
        env.get("rabit_kv2_sha256_lf") == EXPECTED_RABIT_SHA256_LF if env else NOT_EVALUATED,
        env.get("rabit_kv2_sha256_lf"))
    add("idle baseline recorded with no compute process", "gpu_clean",
        bool(baseline) and not baseline.get("compute_apps") and "memory_used_mib" in baseline,
        baseline.get("memory_used_mib"))

    # Correctness gate.
    gate_ran = "EXP3_GATE_START" in top
    gstate = (lambda ok: ok) if gate_ran else (lambda ok: NOT_RUN)
    add("gate: process exit 0", "gate", gstate(top.get("EXP3_GATE_EXIT", {}).get("returncode") == 0),
        top.get("EXP3_GATE_EXIT"))
    add("gate: result passed", "gate", gstate((gate.get("result") or {}).get("passed") is True), gate.get("result"))
    add("gate: pytest exit 0", "gate", gstate(gate.get("pytest_exit") == 0), gate.get("pytest_exit"))
    add("gate: pytest passed count parsed", "gate", gstate(isinstance(gate.get("pytest_passed"), int)),
        gate.get("pytest_passed"))
    add("gate: canonical 'REGRESSION PASSED' line", "gate", gstate(gate.get("regression_passed_line")))
    add("gate: ran against frozen rabit_kv2.py", "gate",
        gstate((gate.get("begin") or {}).get("rabit_kv2_sha256_lf") == EXPECTED_RABIT_SHA256_LF))
    if not top["leg_start"]:
        add("gate completed before first leg", "gate", NOT_EVALUATED if gate_ran else NOT_RUN,
            "no leg started")
    else:
        add("gate completed before first leg", "gate", "EXP3_GATE_EXIT" in top)

    model = top.get("EXP3_MODEL", {})
    add("model snapshot hashed", "model", bool(model.get("files")) if model else NOT_RUN)

    for k, label, d in LEGS:
        p = parsed[k]
        t = p["tags"]
        started = label in top["leg_start"]

        def leg(name: str, category: str, ok, observed=None, _started=started) -> None:
            add(f"{label}: {name}", category, ok if _started else NOT_RUN, observed if _started else None)

        def measured(name: str, category: str, ok, observed=None, _p=p, _started=started) -> None:
            """Only evaluable once the leg actually entered measurement."""
            if not _started:
                add(f"{label}: {name}", category, NOT_RUN)
            elif "EXP3_MEASUREMENT_BEGIN" not in _p["markers"]:
                add(f"{label}: {name}", category, NOT_EVALUATED, "measurement phase never started")
            else:
                add(f"{label}: {name}", category, ok, observed)

        pre = top["pre_leg"].get(label)
        add(f"{label}: GPU clean before leg (no compute process, within {GPU_CLEAN_TOLERANCE_MIB} MiB of baseline)",
            "gpu_clean", gpu_leg_clean(pre, baseline) if pre else NOT_RUN,
            pre["readings"][-1] if pre and pre.get("readings") else None)
        leg("worker exit 0", "leg", (top["leg_exit"].get(label) or {}).get("returncode") == 0,
            top["leg_exit"].get(label))
        leg("worker reports leg/dtype", "leg", t.get("EXP3_LEG") == {"leg": label, "kv_cache_dtype": d}, t.get("EXP3_LEG"))
        leg("RABIT frozen-source markers", "leg", bool(t.get("EXP3_RABIT_MARKERS")) and all(t["EXP3_RABIT_MARKERS"].values()))
        req = t.get("EXP3_REQUESTED_ENGINE_KWARGS", {})
        leg("requested kwargs as planned", "config", req == requested_kwargs(d, req.get("model", "<missing>")))
        leg("model path is the hashed snapshot", "model",
            bool(model) and req.get("model") == model.get("snapshot_dir"), req.get("model"))
        eff = t.get("EXP3_EFFECTIVE_ENGINE_CONFIG", {})
        bad = {x: eff.get(x, "<missing>") for x, v in EXPECTED_EFFECTIVE.items() if eff.get(x, "<missing>") != v}
        leg("effective engine config as planned (eager, Triton, block/len/batch limits, ...)", "config",
            bool(eff) and not bad, bad or None)
        cm = {"name": normalize_mode(eff.get("compilation_mode"), COMPILATION_MODE_VALUES),
              "raw": normalize_mode(eff["compilation_mode_raw"], COMPILATION_MODE_VALUES)
              if "compilation_mode_raw" in eff else None}
        leg("compilation mode = NONE (torch.compile off)", "config",
            cm["name"] == "NONE" and cm["raw"] in (None, "NONE"), cm)
        gm = {"name": normalize_mode(eff.get("cudagraph_mode"), CUDAGRAPH_MODE_VALUES),
              "raw": normalize_mode(eff["cudagraph_mode_raw"], CUDAGRAPH_MODE_VALUES)
              if "cudagraph_mode_raw" in eff else None}
        leg("CUDA graph mode = NONE", "config", gm["name"] == "NONE" and gm["raw"] in (None, "NONE"), gm)
        wl = t.get("EXP3_WORKLOAD", {})
        leg("workload as planned", "workload",
            all(wl.get(x) == v for x, v in EXPECTED_WORKLOAD.items()) and bool(wl.get("prompt_token_ids_sha256")), wl or None)
        kv = t.get("EXP3_KV_DTYPE", {})
        leg("resolved KV dtype", "kv_dtype", bool(kv) and all(kv.get(x) == v for x, v in EXPECTED_KV[d].items()), kv or None)
        cap = t.get("EXP3_CAPACITY", {})
        leg("capacity = num_gpu_blocks x block_size", "capacity",
            bool(cap) and cap["capacity_tokens"] == cap["num_gpu_blocks"] * cap["block_size"], cap or None)
        leg("capacity matches engine log 'GPU KV cache size'", "capacity",
            bool(cap) and p["kv_log_tokens"] == cap.get("capacity_tokens"), p["kv_log_tokens"])
        if d == A:
            if started and cap and p["kv_log_gib"]:
                bpt = p["kv_log_gib"] * 2**30 / cap["capacity_tokens"]
                add(f"{label}: physical bytes/token consistent with 2-byte BF16 KV", "kv_dtype",
                    abs(bpt / BF16_BYTES_PER_TOKEN - 1) <= BYTES_PER_TOKEN_REL_TOL, round(bpt, 1))
            else:
                leg("physical bytes/token consistent with 2-byte BF16 KV", "kv_dtype", False)
        measured(f"warmups == {WARMUPS_PER_LEG}", "measurement", len(p["warmups"]) == WARMUPS_PER_LEG, len(p["warmups"]))
        measured(f"measured reps == {REPS_PER_LEG}", "measurement",
                 [r["rep"] for r in p["samples"]] == list(range(REPS_PER_LEG)), len(p["samples"]))
        measured(f"every request {CONTEXT_TOKENS} prompt / {OUTPUT_TOKENS} output tokens", "measurement",
                 bool(p["samples"]) and all(r["prompt_tokens"] == CONTEXT_TOKENS and r["output_tokens"] == OUTPUT_TOKENS
                                            for r in p["samples"] + p["warmups"]))
        leg("marker order", "leg", p["markers"] == ["EXP3_WARMUP_BEGIN", "EXP3_WARMUP_END", "EXP3_MEASUREMENT_BEGIN",
                                                  "EXP3_MEASUREMENT_END", "EXP3_WORKER_COMPLETE"], p["markers"])
        measured("no Triton JIT compilation during measurement", "measurement",
                 "EXP3_MEASUREMENT_END" in p["markers"] and not p["jit_during_measurement"],
                 p["jit_during_measurement"] or None)

    # Cross-leg checks: evaluated only when every required leg has the data.
    add("config: all non-dtype fields identical across ABBA legs", "config",
        diff["status"], diff.get("violations") or diff.get("missing_config_sections_by_leg"))
    hashes = {label: parsed[k]["tags"].get("EXP3_WORKLOAD", {}).get("prompt_token_ids_sha256") for k, label, _ in LEGS}
    add("prompt token hash identical across all legs", "workload",
        (len(set(hashes.values())) == 1) if all(hashes.values()) else NOT_EVALUATED, hashes)
    for d in DTYPES:
        caps = {label: parsed[k]["tags"].get("EXP3_CAPACITY") for k, label, dd in LEGS if dd == d}
        add(f"{d}: duplicate capacity measurements identical", "capacity",
            (len({json.dumps(c, sort_keys=True) for c in caps.values()}) == 1)
            if all(caps.values()) else NOT_EVALUATED, caps)
        legs_measured = all("EXP3_MEASUREMENT_END" in parsed[k]["markers"] for k, _, dd in LEGS if dd == d)
        n = sum(len(parsed[k]["samples"]) for k, _, dd in LEGS if dd == d)
        add(f"{d}: pooled measured samples == {REPS_PER_DTYPE}", "measurement",
            (n == REPS_PER_DTYPE) if legs_measured else NOT_EVALUATED, n)

    counts = {s: sum(1 for c in checks if c["state"] == s) for s in (PASSED, FAILED, NOT_RUN, NOT_EVALUATED)}
    return {"state_semantics": "passed / failed / not_run (leg never started) / not_evaluated "
                               "(required data or legs missing); only 'passed' counts as passed",
            "checks": checks, "counts": counts, "all_ok": counts[PASSED] == len(checks),
            "failed_categories": sorted({c["category"] for c in checks if c["state"] == FAILED}),
            "non_passed_categories": sorted({c["category"] for c in checks if c["state"] != PASSED})}


def build_summary(parsed: dict, gate: dict, top: dict) -> dict:
    per_leg = {}
    for k, label, d in LEGS:
        p = parsed[k]
        per_leg[label] = {
            "index": k, "kv_cache_dtype": d,
            "capacity": p["tags"]["EXP3_CAPACITY"],
            "available_kv_cache_memory_gib_logged": p["kv_log_gib"],
            "tpot_ms": stats(p["samples"], "tpot_ms"),
            "ttft_ms": stats(p["samples"], "ttft_ms"),
            "wall_ms": stats(p["samples"], "wall_ms"),
            "warmup_samples_excluded": p["warmups"],
            "measured_samples": p["samples"],
            "pre_leg_gpu_state": top["pre_leg"].get(label),
            "process": top["process_exit"].get(label),
        }
    pooled = {}
    for d in DTYPES:
        legs = [(label, parsed[k]) for k, label, dd in LEGS if dd == d]
        rows = [dict(r, leg=label) for label, p in legs for r in p["samples"]]
        cap = legs[0][1]["tags"]["EXP3_CAPACITY"]
        gib = legs[0][1]["kv_log_gib"]
        pooled[d] = {
            "legs": [label for label, _ in legs],
            "resolved_kv_dtype": legs[0][1]["tags"]["EXP3_KV_DTYPE"],
            "capacity_tokens": cap["capacity_tokens"],
            "num_gpu_blocks": cap["num_gpu_blocks"],
            "block_size": cap["block_size"],
            "capacity_identical_across_legs": True,
            "available_kv_cache_memory_gib_logged": gib,
            "physical_bytes_per_token_implied": round(gib * 2**30 / cap["capacity_tokens"], 1) if gib else None,
            "headline": {
                "tpot_ms_median": statistics.median(r["tpot_ms"] for r in rows),
                "tpot_ms_p90": statistics.quantiles([r["tpot_ms"] for r in rows], n=10, method="inclusive")[8],
                "ttft_ms_median": statistics.median(r["ttft_ms"] for r in rows),
                "wall_ms_median": statistics.median(r["wall_ms"] for r in rows),
            },
            "pooled_tpot_ms": stats(rows, "tpot_ms"),
            "pooled_ttft_ms": stats(rows, "ttft_ms"),
            "pooled_wall_ms": stats(rows, "wall_ms"),
            "pooled_measured_samples": rows,
        }
    a, b = pooled[A], pooled[B]

    def delta(key: str) -> dict:
        x, y = a["headline"][key], b["headline"][key]
        return {"bf16": x, "rabit_kv2": y, "signed_delta_ms_rabit_minus_bf16": y - x,
                "signed_delta_pct_vs_bf16": (y / x - 1) * 100}

    canon_cap = json.loads(CANONICAL_CAPACITY.read_text(encoding="utf-8-sig"))
    canon_lat = json.loads(CANONICAL_LATENCY.read_text(encoding="utf-8-sig"))
    return {
        "experiment": "MLSys 2027 Experiment 3 -- matched BF16 vs RABIT-KV physical capacity and latency",
        "capacity_label": CAPACITY_LABEL,
        "latency_label": LATENCY_LABEL,
        "design": {"legs": [{"index": k, "leg": label, "kv_cache_dtype": d} for k, label, d in LEGS],
                   "warmups_per_leg_excluded": WARMUPS_PER_LEG, "measured_reps_per_leg": REPS_PER_LEG,
                   "measured_reps_per_dtype": REPS_PER_DTYPE, "context_tokens": CONTEXT_TOKENS,
                   "output_tokens": OUTPUT_TOKENS, "same_container_same_gpu": True,
                   "fresh_process_per_leg": True, "gate_timeout_s": GATE_TIMEOUT_S,
                   "leg_timeout_s": LEG_TIMEOUT_S},
        "correctness_gate": {
            "command": (gate.get("begin") or {}).get("pytest_command"),
            "wrapper": GATE_COMMAND_DOC,
            "passed": (gate.get("result") or {}).get("passed"),
            "pytest_exit": gate.get("pytest_exit"),
            "pytest_passed": gate.get("pytest_passed"),
            "pytest_warnings": gate.get("pytest_warnings"),
            "canonical_pytest_passed_informational": CANONICAL_GATE_PYTEST_PASSED,
            "rabit_kv2_sha256_lf": (gate.get("begin") or {}).get("rabit_kv2_sha256_lf"),
            "process": top["process_exit"].get("gate"),
        },
        "gpu_clean_state": {"baseline": top.get("EXP3_GPU_BASELINE"),
                            "tolerance_mib": GPU_CLEAN_TOLERANCE_MIB,
                            "pre_leg": top["pre_leg"], "post_run": top.get("EXP3_POST_RUN_GPU_STATE")},
        "pooled_per_dtype": pooled,
        "per_leg": per_leg,
        "signed_deltas_rabit_minus_bf16_pooled": {
            "note": "Signed deltas only (rabit_kv2 - bf16), pooled 30 samples per dtype. No speedup claim.",
            "tpot_median": delta("tpot_ms_median"),
            "tpot_p90": delta("tpot_ms_p90"),
            "ttft_median": delta("ttft_ms_median"),
            "wall_median": delta("wall_ms_median"),
            "capacity_tokens": {"bf16": a["capacity_tokens"], "rabit_kv2": b["capacity_tokens"],
                                "signed_delta_tokens": b["capacity_tokens"] - a["capacity_tokens"],
                                "ratio_rabit_over_bf16": b["capacity_tokens"] / a["capacity_tokens"]},
        },
        "canonical_reference_informational_only": {
            "note": ("Frozen canonical values, NOT newly generated and NOT used for any delta above. "
                     "The canonical bf16 capacity (393,024) is a derived historical figure with no "
                     "measurement log in the repository; it is NOT reused. The bf16 capacity above is "
                     "directly measured."),
            "rabit_kv2_capacity_tokens": canon_cap.get("rabit_kv_capacity_tokens"),
            "bf16_capacity_tokens_derived_not_reused": canon_cap.get("bf16_capacity_tokens"),
            "rabit_kv2_tpot_ms_median_5reps": canon_lat.get("tpot_ms_median"),
            "rabit_kv2_ttft_ms_median_5reps": canon_lat.get("ttft_ms_median"),
            "rabit_kv2_wall_ms_median_5reps": canon_lat.get("wall_ms_median"),
            "sources": [rel(CANONICAL_CAPACITY), rel(CANONICAL_LATENCY)],
        },
        "environment": top.get("EXP3_ENVIRONMENT"),
        "model": top.get("EXP3_MODEL"),
    }


def analyze(session_text: str, write: bool) -> tuple[dict, dict, dict | None]:
    leg_lines, gate_lines, top_lines = demux(session_text)
    parsed = {k: parse_worker(leg_lines[k]) for k, _, _ in LEGS}
    gate = parse_gate(gate_lines)
    top = parse_top(top_lines)
    diff = config_diff(parsed)
    integ = integrity(parsed, gate, top, diff)
    integ["correctness_gate"] = gate
    integ["gpu_clean_state"] = {"baseline": top.get("EXP3_GPU_BASELINE"), "tolerance_mib": GPU_CLEAN_TOLERANCE_MIB,
                                "max_wait_s": GPU_CLEAN_MAX_WAIT_S, "pre_leg": top["pre_leg"]}
    integ["processes"] = {"exits": top["process_exit"], "watchdog_timeouts": top["watchdog_timeouts"]}
    # A summary exists ONLY for a complete run in which every check passed.
    summary = build_summary(parsed, gate, top) if integ["all_ok"] else None
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        GATE_LOG.write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
        for d in DTYPES:
            parts = []
            for k, label, dd in LEGS:
                if dd == d:
                    parts.append(f"===== EXP3 LEG {label} (index {k}, {d}) =====")
                    parts.extend(leg_lines[k])
            (OUT_DIR / LOG_NAME[d]).write_text("\n".join(parts) + "\n", encoding="utf-8")
        CONFIG_DIFF.write_text(json.dumps(diff, indent=2, default=str) + "\n", encoding="utf-8")
        INTEGRITY.write_text(json.dumps(integ, indent=2, default=str) + "\n", encoding="utf-8")
        if summary is not None:
            SUMMARY.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return diff, integ, summary


# ------------------------------------------------------------ manifest/run
def write_manifest(manifest: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def finalize(manifest: dict, status: str, failure: dict | None) -> None:
    """Single terminal path: protected-path post-check + archived-attempt
    integrity, persisted in manifest.json. The original failure reason is
    kept even if the post-check itself fails."""
    manifest["status"] = status
    manifest["completed_utc"] = now()
    if failure is not None:
        manifest.setdefault("failure", failure)
    try:
        assert_protected_paths_clean(f"post-run ({status})")
    except Exception as exc:  # noqa: BLE001
        manifest["protected_paths_post_run_status"] = "check_failed"
        manifest["protected_paths_check_error"] = {"type": type(exc).__name__, "message": str(exc)}
        manifest["status"] = "failed"
    else:
        manifest["protected_paths_post_run_status"] = "clean"
    archived_now = archived_attempts_digest()
    archived_before = manifest["provenance"].get("archived_attempts_sha256", {})
    manifest["archived_attempts_unchanged"] = archived_now == archived_before
    if not manifest["archived_attempts_unchanged"]:
        manifest["status"] = "failed"
        manifest.setdefault("failure", {"stage": "archived_attempt_modified",
                                        "reason": "an archived failed_attempt_* file changed during the run"})
    write_manifest(manifest)


def classify_failure(code: int, integ: dict) -> dict:
    """Stage from checks that actually FAILED (not_run / not_evaluated are
    consequences of an earlier stop, never a cause)."""
    cats = integ["failed_categories"]
    info = {"modal_returncode": code, "failed_categories": cats,
            "non_passed_categories": integ["non_passed_categories"]}
    for stage, cat in (("watchdog_timeout", "watchdog"), ("correctness_gate", "gate"), ("gpu_clean", "gpu_clean")):
        if cat in cats:
            return {"stage": stage, **info}
    if code != 0:
        return {"stage": "modal_nonzero_exit", **info}
    if "config" in cats:
        return {"stage": "config_diff", **info}
    return {"stage": "integrity", **info}


def run(manifest: dict) -> int:
    snapshot = build_snapshot()
    manifest["vllm_kvquant_snapshot"] = snapshot
    command = build_command()
    row = {"name": "abba", "command": command, "log": rel(SESSION_LOG),
           "gate_command": GATE_COMMAND_DOC,
           "worker_commands": {label: worker_command(label, d) for _, label, d in LEGS},
           "started_utc": now(), "completed_utc": None, "returncode": None, "status": "running"}
    manifest["runs"].append(row)
    write_manifest(manifest)

    code = stream_command(command, SESSION_LOG, {"EXP3_VLLM_SNAPSHOT": snapshot["path"]})
    row.update(returncode=code, completed_utc=now())

    manifest["stage"] = "parse"
    diff, integ, summary = analyze(SESSION_LOG.read_text(encoding="utf-8", errors="replace"), write=True)
    row["integrity_counts"] = integ["counts"]
    manifest["gpu_clean_state"] = integ["gpu_clean_state"]
    manifest["correctness_gate"] = integ["correctness_gate"]
    manifest["processes"] = integ["processes"]
    manifest["integrity_counts"] = integ["counts"]
    manifest["config_diff_status"] = diff["status"]
    manifest.pop("stage", None)

    if code != 0 or not integ["all_ok"]:
        failure = classify_failure(code, integ)
        row["status"] = failure["stage"]
        finalize(manifest, "failed", failure)
        raise SystemExit(
            f"\nEXPERIMENT 3 STOPPED ({failure['stage']}): modal exit={code}; integrity {integ['counts']}; "
            f"non-passed categories {integ['non_passed_categories']}. Logs and {rel(INTEGRITY)} preserved."
        )

    row["status"] = "passed"
    finalize(manifest, "passed", None)
    if manifest["status"] != "passed":
        raise SystemExit(f"\nEXPERIMENT 3 FAILED at post-run checks: {manifest.get('failure')}")
    print("\n" + "=" * 118 + "\nEXPERIMENT 3: ABBA MATCHED RUN PASSED\n"
          f"Summary: {SUMMARY}\n" + "=" * 118)
    return 0


def main(argv: list[str] | None = None) -> int:
    make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Preflight + canonical-equivalence proof + planned commands and config diff. "
                         "No Modal/GPU, no files written.")
    args = ap.parse_args(argv)

    print("RABIT-KV MLSys 2027 -- Experiment 3: matched BF16 vs RABIT-KV physical capacity/latency")
    print(f"ABBA legs (one container, one GPU, fresh process per leg): "
          f"{[f'{label}={d}' for _, label, d in LEGS]}")
    print(f"Per leg: {WARMUPS_PER_LEG} full-shape warmups (excluded) + {REPS_PER_LEG} measured reps "
          f"-> {REPS_PER_DTYPE} measured reps per dtype. Workload: {CONTEXT_TOKENS} ctx / {OUTPUT_TOKENS} out")
    print(f"Watchdogs: gate {GATE_TIMEOUT_S}s, each leg {LEG_TIMEOUT_S}s (own process group, group kill, abort)")
    print()

    prov = preflight(args.dry_run)
    print("Preflight OK.")
    for k in ("git_branch", "git_head", "vllm_kvquant_tree", "rabit_kv2_sha256", "runner_script_sha256",
              "modal_app_sha256", "worker_sha256", "correctness_gate_sha256", "watchdog_sha256",
              "canonical_benchmark_deployment_sha256"):
        print(f"  {k:<40} {prov[k]}")
    print("  canonical equivalence: LLM kwargs equal except kv_cache_dtype/model path; image AST equal; "
          "gate regression() AST equal; no engine RPC in worker; timing definitions shared; "
          "clean-state + watchdog constants match")
    print(f"  archived attempts protected: {len(prov['archived_attempts_sha256'])} files hashed")
    if prov["uncommitted_experiment_files"]:
        print("  WARNING (dry-run only): a real run would refuse until these are committed:")
        for line in prov["uncommitted_experiment_files"].splitlines():
            print(f"    {line}")
    if prov["existing_top_level_output_files"]:
        print(f"  WARNING (dry-run only): a real run would refuse; previous-attempt files present: "
              f"{prov['existing_top_level_output_files']}")

    plan = {label: flatten("requested", requested_kwargs(d)) for _, label, d in LEGS}
    keys = sorted(set().union(*plan.values()))
    differing = [x for x in keys if len({json.dumps(p.get(x)) for p in plan.values()}) > 1]
    print("\nPlanned requested-config comparison across ABBA legs "
          "(post-run the same check covers effective config, workload and KV dtype):")
    print(json.dumps({"fields_compared": len(keys), "differing_fields": differing,
                      "dtype_induced_allowlist": DTYPE_INDUCED_ALLOWLIST,
                      "only_allowlisted_fields_differ": set(differing) <= set(DTYPE_INDUCED_ALLOWLIST)},
                     indent=2))
    if not set(differing) <= set(DTYPE_INDUCED_ALLOWLIST):
        raise SystemExit("Planned configs differ in a non-dtype field -- refusing.")

    print("\nLocal command (one Modal run):\n  " + " ".join(build_command()))
    print("Step 0 (in container): idle GPU baseline; no compute process allowed")
    print(f"Step 1 (in container): correctness gate, fresh process group, watchdog {GATE_TIMEOUT_S}s, "
          "must pass before any leg:\n  " + " ".join(GATE_COMMAND_DOC))
    for k, label, d in LEGS:
        print(f"Step {k + 1} (in container): GPU clean check, then leg {label} ({d}), watchdog {LEG_TIMEOUT_S}s:\n  "
              + " ".join(worker_command(label, d)))
    print(f"Outputs: {rel(OUT_DIR)}/ ({SESSION_LOG.name}, {GATE_LOG.name}, {', '.join(LOG_NAME.values())}, "
          f"{MANIFEST.name}, {CONFIG_DIFF.name}, {INTEGRITY.name}, {SUMMARY.name})")

    if args.dry_run:
        print("\n--dry-run: no Modal/GPU commands executed, no snapshot built, no files written.")
        return 0

    manifest = {
        "experiment": "Experiment 3 -- matched BF16 vs RABIT-KV physical capacity and latency (ABBA)",
        "plan_reference": "docs/MLSYS_EXPERIMENT_PLAN.md",
        "model": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "gpu": "NVIDIA H100 80GB (Modal)",
        "legs": [{"index": k, "leg": label, "kv_cache_dtype": d} for k, label, d in LEGS],
        "protocol": {"warmups_per_leg_excluded": WARMUPS_PER_LEG, "measured_reps_per_leg": REPS_PER_LEG,
                     "measured_reps_per_dtype": REPS_PER_DTYPE, "context_tokens": CONTEXT_TOKENS,
                     "output_tokens": OUTPUT_TOKENS, "same_container_same_gpu": True,
                     "fresh_process_per_leg": True, "dtype_induced_allowlist": DTYPE_INDUCED_ALLOWLIST,
                     "gpu_clean_tolerance_mib": GPU_CLEAN_TOLERANCE_MIB,
                     "gpu_clean_max_wait_s": GPU_CLEAN_MAX_WAIT_S,
                     "gate_timeout_s": GATE_TIMEOUT_S, "leg_timeout_s": LEG_TIMEOUT_S},
        "capacity_label": CAPACITY_LABEL,
        "latency_label": LATENCY_LABEL,
        "started_utc": now(), "completed_utc": None, "status": "running",
        "protected_paths_post_run_status": "pending",
        "provenance": prov, "runs": [],
    }
    write_manifest(manifest)
    return execute(manifest)


def execute(manifest: dict) -> int:
    """Run and guarantee a finalized manifest on every terminal path."""
    try:
        return run(manifest)
    except SystemExit:
        raise
    except (Exception, KeyboardInterrupt) as exc:
        t = now()
        error = {"type": type(exc).__name__, "message": str(exc)}
        for row in manifest["runs"]:
            if row.get("status") == "running":
                row.update(status="runner_failed", completed_utc=t, error=error)
        manifest["runner_error"] = error
        stage = manifest.pop("stage", "local_runner")
        finalize(manifest, "failed", {"stage": "parser_failure" if stage == "parse" else "local_runner_exception",
                                      **error})
        ce = manifest.get("protected_paths_check_error")
        extra = f"\nPROTECTED-PATH CHECK ALSO FAILED: {ce['type']}: {ce['message']}" if ce else ""
        raise SystemExit(f"\nEXPERIMENT 3 RUNNER FAILED: {type(exc).__name__}: {exc}{extra}\n"
                         f"Manifest marked failed; partial logs preserved under {rel(OUT_DIR)}/. Not retried.") from exc


if __name__ == "__main__":
    raise SystemExit(main())
