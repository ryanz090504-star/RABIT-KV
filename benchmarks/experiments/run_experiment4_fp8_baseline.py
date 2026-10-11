"""
RABIT-KV MLSys 2027 -- Experiment 4 runner: matched BF16 vs native FP8 vs
RABIT-KV physical capacity and latency on the real vLLM engine (H100, Modal).

PHYSICAL capacity + latency only. No FP8 quality claim is made or implied;
a real FP8 quality comparison needs a separate real-engine quality harness.

Design (one `modal run` of benchmarks/mlsys2027/exp4_deployment_modal.py):
  * one container, one physical H100, one image, one model snapshot;
  * idle GPU baseline recorded first;
  * frozen RABIT-KV correctness gate (exp3_correctness_gate.py, unchanged)
    runs once and must pass before any measurement;
  * mirrored six-leg order A1 B1 C1 C2 B2 A2 with A = bfloat16,
    B = fp8_e4m3 (native per-tensor FP8 KV cache), C = rabit_kv2; each leg a
    fresh worker/engine process (exp4_engine_worker.py) with 5 full-shape
    warmups (excluded) + 15 measured reps -> 30 measured samples per dtype;
  * ALL THREE dtypes are measured fresh in this session. No Experiment 3
    sample (latency or capacity) is read, reused or pooled;
  * gate and every leg under the unchanged Experiment 3 watchdog
    (exp3_watchdog.py): own process group, whole-group kill on timeout
    (gate 600 s, leg 900 s), no retry, a timeout aborts the experiment;
  * before EVERY leg: no GPU compute process and memory back within 256 MiB
    of the idle baseline, else hard fail;
  * engine arguments identical except kv_cache_dtype.

Before any run this script proves by AST that the worker's engine kwargs,
workload constants, prompt construction and timed region equal the frozen
Experiment 3 worker (and the canonical embedded runner in
benchmarks/performance/benchmark_deployment.py), that the Modal image equals
the canonical one, that the correctness gate and watchdog are the unchanged
committed Experiment 3 files, and that the clean-state / watchdog helpers of
the Modal app equal Experiment 3's. After the run every integrity check has
an explicit state -- passed / failed / not_run / not_evaluated -- and only a
run in which EVERY check passed produces a summary. Every terminal path runs
the protected-path post-check and persists it in manifest.json.

This script never modifies Experiment 1-3 code or evidence, vllm-kvquant or
any canonical result; it writes ONLY top-level files under
results/mlsys2027/fp8_baseline/ and never touches archived attempts there.

Usage:
    python benchmarks/mlsys2027/run_experiment4_fp8_baseline.py --dry-run
    python benchmarks/mlsys2027/run_experiment4_fp8_baseline.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment3_deployment as r3  # noqa: E402  (frozen; helpers only, never modified)
from run_experiment3_deployment import (  # noqa: E402
    COMPILATION_MODE_VALUES, CUDAGRAPH_MODE_VALUES, FAILED, NOT_EVALUATED, NOT_RUN, PASSED,
    _function, _module_assign, canonical_llm_kwargs, canonical_runner_source, console_write,
    flatten, make_console_encoding_safe, normalize_mode, now, rel, run_git, sha256, sha256_raw,
    stats, stream_command,
)

ROOT = r3.ROOT
HERE = ROOT / "benchmarks" / "mlsys2027"
RUNNER_SCRIPT = Path(__file__).resolve()
MODAL_APP = HERE / "exp4_deployment_modal.py"
WORKER = HERE / "exp4_engine_worker.py"
GATE = HERE / "exp3_correctness_gate.py"          # reused unchanged
WATCHDOG = HERE / "exp3_watchdog.py"              # reused unchanged
EXP3_WORKER = HERE / "exp3_engine_worker.py"      # reference for AST equivalence
EXP3_MODAL_APP = HERE / "exp3_deployment_modal.py"
EXP3_RUNNER = HERE / "run_experiment3_deployment.py"
CANONICAL_DEPLOYMENT = r3.CANONICAL_DEPLOYMENT
RABIT_KV2 = r3.RABIT_KV2
EXPECTED_RABIT_SHA256_LF = r3.EXPECTED_RABIT_SHA256_LF

OUT_DIR = ROOT / "results" / "mlsys2027" / "fp8_baseline"
ARCHIVE_GLOB = "failed_attempt_*"
SESSION_LOG = OUT_DIR / "modal_session.log"
GATE_LOG = OUT_DIR / "correctness_gate.log"
MANIFEST = OUT_DIR / "manifest.json"
CONFIG_DIFF = OUT_DIR / "matched_config_diff.json"
INTEGRITY = OUT_DIR / "integrity_check.json"
SUMMARY = OUT_DIR / "matched_capacity_latency_summary.json"

EXP3_EVIDENCE_DIR = ROOT / "results" / "mlsys2027" / "deployment"
PROTECTED_PATHS = [
    *r3.PROTECTED_PATHS,
    # All Experiment 3 evidence: run #1, failed_attempt_1, replication_1,
    # replication_comparison.json.
    EXP3_EVIDENCE_DIR,
    # Frozen Experiment 1-3 code, including the gate/watchdog reused here.
    *sorted(HERE.glob("*exp1*")), *sorted(HERE.glob("*exp2*")), *sorted(HERE.glob("*exp3*")),
    *sorted(HERE.glob("run_experiment[123]_*")),
]
MUST_BE_COMMITTED = [RUNNER_SCRIPT, MODAL_APP, WORKER, GATE, WATCHDOG]

# Mirrored plan: A = bfloat16, B = native FP8 (e4m3), C = rabit_kv2.
A, B, C = "bfloat16", "fp8_e4m3", "rabit_kv2"
DTYPES = [A, B, C]
LETTER = {A: "A", B: "B", C: "C"}
LEGS = [(1, "A1", A), (2, "B1", B), (3, "C1", C), (4, "C2", C), (5, "B2", B), (6, "A2", A)]
LOG_NAME = {A: "bf16_deployment.log", B: "fp8_e4m3_deployment.log", C: "rabit_kv2_deployment.log"}
SHORT = {A: "bf16", B: "fp8", C: "rabit"}
WARMUPS_PER_LEG = 5
REPS_PER_LEG = 15
REPS_PER_DTYPE = 30
CONTEXT_TOKENS = 2048
OUTPUT_TOKENS = 32

# Must equal the Modal app constants AND Experiment 3 (verified by AST).
GPU_CLEAN_TOLERANCE_MIB = 256
GPU_CLEAN_MAX_WAIT_S = 60
GATE_TIMEOUT_S = 600
LEG_TIMEOUT_S = 900
MODAL_FUNCTION_TIMEOUT_S = 7200  # backstop only: >= gate + 6 x leg watchdog budgets

# Config comparison: only these kv_cache_dtype-induced fields may differ, and
# only between dtypes (never between the two legs of one dtype). Every one is
# a direct function of the requested dtype string (audit in the plan).
DTYPE_INDUCED_ALLOWLIST = [
    "requested.kv_cache_dtype",
    "kv_dtype.requested_kv_cache_dtype",
    "kv_dtype.engine_cache_dtype",
    "kv_dtype.resolved_kv_torch_dtype",
    "kv_dtype.kv_quant_mode",
    "kv_dtype.fp8_storage_view_dtype",
]

# Llama-3.1-8B: 32 layers x (K,V) x 8 KV heads x 128 head_dim.
KV_ELEMENTS_PER_TOKEN = 32 * 2 * 8 * 128
# Element-size consistency cross-check (NOT a capacity-ratio expectation):
# BF16 stores 2 bytes/element, native FP8 stores 1 byte/element with no
# per-token metadata (FP8_PER_TENSOR). RABIT-KV bytes/token are reported only.
EXPECTED_BYTES_PER_ELEMENT = {A: 2, B: 1}
BYTES_PER_TOKEN_REL_TOL = 0.01

EXPECTED_EFFECTIVE = {
    **r3.EXPECTED_EFFECTIVE,
    "calculate_kv_scales": False,
    "kv_cache_dtype_skip_layers": [],
    "hf_quantization_config": None,
}
EXPECTED_KV = {
    A: {"engine_cache_dtype": "bfloat16", "resolved_kv_torch_dtype": "torch.bfloat16",
        "kv_quant_mode": "NONE", "fp8_storage_view_dtype": None},
    B: {"engine_cache_dtype": "fp8_e4m3", "resolved_kv_torch_dtype": "torch.uint8",
        "kv_quant_mode": "FP8_PER_TENSOR", "fp8_storage_view_dtype": "torch.float8_e4m3fn"},
    C: {"engine_cache_dtype": "rabit_kv2", "resolved_kv_torch_dtype": "torch.uint8",
        "kv_quant_mode": "RABIT_KV2", "fp8_storage_view_dtype": None},
}
EXPECTED_WORKLOAD = {
    "context_tokens": CONTEXT_TOKENS, "output_tokens": OUTPUT_TOKENS, "warmups": WARMUPS_PER_LEG,
    "reps": REPS_PER_LEG, "temperature": 0.0, "ignore_eos": True, "max_tokens": OUTPUT_TOKENS,
}

CAPACITY_LABEL = "PHYSICAL vLLM allocator KV capacity (num_gpu_blocks x block_size) from the real engine"
LATENCY_LABEL = r3.LATENCY_LABEL.replace("30 samples per dtype", "30 samples per dtype (this session only)")
SCOPE_NOTE = (
    "Physical capacity and latency only. FP8 = vLLM's native per-tensor FP8 (e4m3) KV cache with "
    "default 1.0 KV scales (unquantized checkpoint, calculate_kv_scales=False). No FP8 quality "
    "claim is made: a real FP8 quality comparison requires a separate real-engine quality harness. "
    "No Experiment 3 sample is used; all three dtypes are measured in this session."
)
NATIVE_FP8_SOURCE_AUDIT = {
    "requested_dtype": "fp8_e4m3 (CacheDType literal; 'fp8' is an exact alias on CUDA)",
    "storage": "STR_DTYPE_TO_TORCH_DTYPE['fp8_e4m3'] = torch.uint8, viewed by TritonAttentionImpl "
               "as current_platform.fp8_dtype() = torch.float8_e4m3fn",
    "kv_quant_mode": "get_kv_quant_mode('fp8_e4m3') = FP8_PER_TENSOR",
    "kv_scales": "checkpoint has no quantization_config -> no BaseKVCacheMethod; "
                 "set_default_quant_scales() sets k/v/q scales to 1.0; calculate_kv_scales=False",
    "checkpoint_override": "resolve_kv_cache_dtype_string only rewrites 'auto'; an explicit "
                           "fp8_e4m3 cannot be overridden by the checkpoint",
    "query": "on CUDA the attention layer also quantizes the query to FP8 (QuantFP8 static "
             "per-tensor, q_scale 1.0) for fp8/fp8_e4m3 KV caches; this is part of the native "
             "FP8 path being measured",
    "hardware": "Triton FP8 KV requires SM89+; H100 is SM90",
}

PAIRS = [(B, A), (C, A), (C, B)]  # (x, y): report x - y and x / y


# ---------------------------------------------------------------- utilities
def assert_protected_paths_clean(context: str) -> None:
    status = run_git("status", "--short", "--", *[rel(p) for p in PROTECTED_PATHS])
    if status:
        raise RuntimeError(
            f"CRITICAL: protected paths changed ({context}). Investigate immediately:\n" + status
        )


def uncommitted_experiment_files() -> str:
    return run_git("status", "--short", "--", *[rel(p) for p in MUST_BE_COMMITTED])


def _digest_dir(d: Path) -> dict:
    return {f.relative_to(d).as_posix(): sha256_raw(f) for f in sorted(p for p in d.rglob("*") if p.is_file())}


def archived_attempts_digest() -> dict:
    out = {}
    if OUT_DIR.is_dir():
        for d in sorted(OUT_DIR.glob(ARCHIVE_GLOB)):
            for k, v in _digest_dir(d).items():
                out[f"{d.name}/{k}"] = v
    return out


def exp3_evidence_digest() -> dict:
    """Raw-byte SHA-256 of every Experiment 3 evidence file (must never change)."""
    return _digest_dir(EXP3_EVIDENCE_DIR) if EXP3_EVIDENCE_DIR.is_dir() else {}


def top_level_output_files() -> list[str]:
    if not OUT_DIR.is_dir():
        return []
    return sorted(p.name for p in OUT_DIR.iterdir() if p.is_file())


def _const(tree: ast.Module, name: str):
    return ast.literal_eval(_module_assign(tree, name))


def _dump(node: ast.AST, rename: tuple[str, str] | None = None) -> str:
    s = ast.dump(node)
    return s.replace(*rename) if rename else s


def _return_dict(fn: ast.FunctionDef) -> dict[str, str]:
    for node in fn.body:
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            return {ast.literal_eval(k): ast.dump(v) for k, v in zip(node.value.keys, node.value.values)}
    raise RuntimeError(f"{fn.name}() has no dict return")


def _nested_function(fn: ast.FunctionDef, name: str) -> ast.FunctionDef:
    for node in ast.walk(fn):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"nested function {name!r} not found in {fn.name}()")


def _assign_stmts(fn: ast.FunctionDef, targets: tuple[str, ...]) -> list[str]:
    out = []
    for node in fn.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in targets:
            out.append(ast.dump(node))
    return out


# ------------------------------------------------ equivalence (AST)
MODAL_SHARED_FUNCTIONS = ("_emit", "_sha256_file", "_gpu_query", "_compute_apps", "_gpu_state",
                          "_require_clean", "_run_guarded")
PROMPT_TARGETS = ("tok", "bos", "filler", "prompt", "sp")


def verify_equivalence() -> dict:
    """Prove Experiment 4 measures exactly like the frozen Experiment 3 (and the
    canonical runner), except kv_cache_dtype."""
    csrc = canonical_runner_source()
    ctree = ast.parse(csrc)
    w4src = WORKER.read_text(encoding="utf-8")
    w4, w3 = ast.parse(w4src), ast.parse(EXP3_WORKER.read_text(encoding="utf-8"))
    m4, m3 = ast.parse(MODAL_APP.read_text(encoding="utf-8")), ast.parse(EXP3_MODAL_APP.read_text(encoding="utf-8"))
    rtree = ast.parse(EXP3_RUNNER.read_text(encoding="utf-8"))

    # 1. Engine kwargs: canonical == Experiment 3 worker == Experiment 4 worker.
    canon_kwargs = canonical_llm_kwargs(csrc)
    canon_wo = {k: v for k, v in canon_kwargs.items() if k not in ("model", "kv_cache_dtype")}
    b3, b4 = _const(w3, "BASE_ENGINE_KWARGS"), _const(w4, "BASE_ENGINE_KWARGS")
    if not (canon_wo == b3 == b4):
        raise RuntimeError(f"BASE_ENGINE_KWARGS differ (canonical / exp3 / exp4):\n{canon_wo}\n{b3}\n{b4}")

    # 2. Workload constants and allowed dtypes.
    for const in ("CONTEXT_TOKENS", "OUTPUT_TOKENS"):
        if not (_const(w3, const) == _const(w4, const) == globals()[const]):
            raise RuntimeError(f"{const} differs from Experiment 3")
    if tuple(_const(w4, "ALLOWED_KV_CACHE_DTYPES")) != tuple(DTYPES):
        raise RuntimeError("worker ALLOWED_KV_CACHE_DTYPES != runner DTYPES")
    for const, mine in (("WARMUPS_PER_LEG", WARMUPS_PER_LEG), ("REPS_PER_LEG", REPS_PER_LEG),
                        ("CONTEXT_TOKENS", CONTEXT_TOKENS), ("OUTPUT_TOKENS", OUTPUT_TOKENS),
                        ("GPU_CLEAN_TOLERANCE_MIB", GPU_CLEAN_TOLERANCE_MIB),
                        ("GPU_CLEAN_MAX_WAIT_S", GPU_CLEAN_MAX_WAIT_S),
                        ("GATE_TIMEOUT_S", GATE_TIMEOUT_S), ("LEG_TIMEOUT_S", LEG_TIMEOUT_S)):
        if _const(rtree, const) != mine:
            raise RuntimeError(f"{const} differs from the Experiment 3 runner")

    # 3. Prompt construction and timed region identical to Experiment 3.
    main3, main4 = _function(w3, "main"), _function(w4, "main")
    if _assign_stmts(main3, PROMPT_TARGETS) != _assign_stmts(main4, PROMPT_TARGETS) or \
            len(_assign_stmts(main4, PROMPT_TARGETS)) != len(PROMPT_TARGETS):
        raise RuntimeError("prompt/SamplingParams construction differs from Experiment 3")
    one3, one4 = _nested_function(main3, "one"), _nested_function(main4, "one")
    timed3 = [ast.dump(s) for s in one3.body[:5]]
    timed4 = [ast.dump(s) for s in one4.body[:5]]
    if timed3 != timed4:
        raise RuntimeError("timed region / metric inputs of one() differ from Experiment 3")
    r3d, r4d = _return_dict(one3), _return_dict(one4)
    if any(r4d.get(k) != v for k, v in r3d.items()):
        raise RuntimeError("an Experiment 3 sample field is computed differently")
    if set(r4d) - set(r3d) != {"output_token_ids_sha256"}:
        raise RuntimeError(f"unexpected extra sample fields: {sorted(set(r4d) - set(r3d))}")
    if ast.dump(one3.args) != ast.dump(one4.args):
        raise RuntimeError("one() signature differs")
    if _dump(_function(w3, "enum_name")) != _dump(_function(w4, "enum_name")):
        raise RuntimeError("enum_name() differs from Experiment 3")

    # 4. No engine RPC (Experiment 3 attempt-1 hang cause).
    for node in ast.walk(w4):
        if isinstance(node, ast.Attribute) and node.attr == "collective_rpc":
            raise RuntimeError("worker calls collective_rpc (Experiment 3 attempt-1 hang cause)")

    # 5. Modal image / model / commit equal canonical; clean-state and watchdog
    #    helpers equal Experiment 3's (modulo EXP3_ -> EXP4_ tag names).
    if ast.dump(_module_assign(ctree, "image")) != ast.dump(_module_assign(m4, "image")):
        raise RuntimeError("Experiment 4 Modal image expression differs from the canonical runner's")
    for const in ("MODEL", "BASE_COMMIT"):
        if _const(ctree, const) != _const(m4, const):
            raise RuntimeError(f"{const} differs from the canonical runner")
    for const, mine in (("GPU_CLEAN_TOLERANCE_MIB", GPU_CLEAN_TOLERANCE_MIB),
                        ("GPU_CLEAN_MAX_WAIT_S", GPU_CLEAN_MAX_WAIT_S), ("GATE_TIMEOUT_S", GATE_TIMEOUT_S),
                        ("LEG_TIMEOUT_S", LEG_TIMEOUT_S), ("GPU_CLEAN_POLL_S", _const(m3, "GPU_CLEAN_POLL_S")),
                        ("EXPECTED_RABIT_SHA256_LF", EXPECTED_RABIT_SHA256_LF)):
        if not (_const(m4, const) == _const(m3, const) == mine):
            raise RuntimeError(f"Modal app {const} differs from Experiment 3 / runner")
    for fn in MODAL_SHARED_FUNCTIONS:
        if _dump(_function(m3, fn), ("EXP3_", "EXP4_")) != _dump(_function(m4, fn)):
            raise RuntimeError(f"Modal helper {fn}() differs from Experiment 3")
    backstop = None
    for node in ast.walk(m4):
        if isinstance(node, ast.keyword) and node.arg == "timeout":
            backstop = ast.literal_eval(node.value)
    if backstop != MODAL_FUNCTION_TIMEOUT_S or backstop < GATE_TIMEOUT_S + len(LEGS) * LEG_TIMEOUT_S:
        raise RuntimeError(f"Modal function backstop timeout {backstop} too small for the watchdog budget")

    # 6. Gate: canonical regression() verbatim (the file itself is committed and protected).
    gtree = ast.parse(GATE.read_text(encoding="utf-8"))
    cfn, gfn = _function(ctree, "regression"), _function(gtree, "regression")
    if not (ast.dump(ast.Module(body=cfn.body, type_ignores=[])) ==
            ast.dump(ast.Module(body=gfn.body, type_ignores=[])) and ast.dump(cfn.args) == ast.dump(gfn.args)):
        raise RuntimeError("exp3_correctness_gate.regression() is not verbatim the canonical regression()")

    return {
        "canonical_source": f"{rel(CANONICAL_DEPLOYMENT)}#RUNNER_Z (zlib+base64, decoded in memory)",
        "canonical_runner_sha256": hashlib.sha256(csrc.encode("utf-8")).hexdigest(),
        "worker_base_engine_kwargs": b4,
        "engine_kwargs_equal_canonical_exp3_exp4_except_kv_cache_dtype_and_model_path": True,
        "workload_and_protocol_constants_equal_exp3": True,
        "prompt_construction_ast_equal_exp3": True,
        "timed_region_and_sample_fields_ast_equal_exp3": True,
        "extra_sample_fields": ["output_token_ids_sha256"],
        "worker_has_no_engine_rpc": True,
        "modal_image_expression_ast_equal_canonical": True,
        "modal_clean_state_and_watchdog_helpers_ast_equal_exp3": list(MODAL_SHARED_FUNCTIONS),
        "modal_function_backstop_timeout_s": backstop,
        "correctness_gate_regression_ast_equal_to_canonical": True,
        "gate_and_watchdog_are_unchanged_exp3_files": [rel(GATE), rel(WATCHDOG)],
    }


def requested_kwargs(dtype: str, model_dir: str = "<modelscope snapshot dir>") -> dict:
    base = _const(ast.parse(WORKER.read_text(encoding="utf-8")), "BASE_ENGINE_KWARGS")
    return {"model": model_dir, **base, "kv_cache_dtype": dtype}


# --------------------------------------------------------------- preflight
def preflight(dry_run: bool) -> dict:
    branch = run_git("branch", "--show-current")
    head = run_git("rev-parse", "HEAD")
    assert_protected_paths_clean("preflight, before any run")

    rabit_sha = sha256(RABIT_KV2)
    if rabit_sha != EXPECTED_RABIT_SHA256_LF:
        raise RuntimeError(f"rabit_kv2.py is not the frozen committed content: {rabit_sha}")

    equivalence = verify_equivalence()

    uncommitted = uncommitted_experiment_files()
    if uncommitted and not dry_run:
        raise RuntimeError(
            "Refusing to run: Experiment 4 code has uncommitted changes, so recorded SHAs "
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
        "exp3_worker_sha256": sha256(EXP3_WORKER),
        "exp3_modal_app_sha256": sha256(EXP3_MODAL_APP),
        "exp3_runner_sha256": sha256(EXP3_RUNNER),
        "canonical_benchmark_deployment_sha256": sha256(CANONICAL_DEPLOYMENT),
        "sha256_basis": "LF-normalized bytes (CRLF -> LF); equals committed git content",
        "equivalence": equivalence,
        "protected_paths": [rel(p) for p in PROTECTED_PATHS],
        "protected_paths_baseline_status": "clean",
        "archived_attempts_sha256": archived_attempts_digest(),
        "exp3_evidence_sha256_raw": exp3_evidence_digest(),
        "existing_top_level_output_files": leftovers or None,
        "uncommitted_experiment_files": uncommitted or None,
    }


# ------------------------------------------------------------ run plumbing
def build_snapshot() -> dict:
    out = Path(tempfile.mkdtemp(prefix="exp4_vllm_snapshot_")) / "vllm_kvquant_snapshot.zip"
    run_git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(out), "HEAD:vllm-kvquant")
    return {"path": str(out), "sha256": sha256_raw(out), "bytes": out.stat().st_size,
            "source": "git archive --format=zip HEAD:vllm-kvquant (core.autocrlf=false)"}


def legs_arg() -> str:
    return ",".join(f"{label}={d}" for _, label, d in LEGS)


def build_command() -> list[str]:
    return [sys.executable, "-m", "modal", "run", str(MODAL_APP),
            "--legs", legs_arg(), "--warmups", str(WARMUPS_PER_LEG), "--reps-per-leg", str(REPS_PER_LEG)]


def worker_command(label: str, dtype: str, model_dir: str = "<modelscope snapshot dir>") -> list[str]:
    return ["python", "/opt/exp4/exp4_engine_worker.py", "--kv-cache-dtype", dtype,
            "--model-dir", model_dir, "--warmups", str(WARMUPS_PER_LEG),
            "--reps", str(REPS_PER_LEG), "--leg", label]


GATE_COMMAND_DOC = ["python", "/opt/exp4/exp3_correctness_gate.py"]


# ------------------------------------------------------------------ parsing
TAG = re.compile(r"^(EXP4_[A-Z_]+)=(\{.*\})\s*$")
GATE_TAG = re.compile(r"^(EXP3_GATE_[A-Z_]+)=(\{.*\})\s*$")  # unchanged Experiment 3 gate
ROWLINE = re.compile(r"^(EXP4_SAMPLE|EXP4_WARMUP) (\{.*\})\s*$")
KV_TOKENS = r3.KV_TOKENS
KV_MEM = r3.KV_MEM
PYTEST_SUMMARY = r3.PYTEST_SUMMARY
PYTEST_EXIT = r3.PYTEST_EXIT
JIT = r3.JIT
MARKERS = ["EXP4_WARMUP_BEGIN", "EXP4_WARMUP_END", "EXP4_MEASUREMENT_BEGIN", "EXP4_MEASUREMENT_END",
           "EXP4_WORKER_COMPLETE"]


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
            (out["samples"] if m.group(1) == "EXP4_SAMPLE" else out["warmups"]).append(json.loads(m.group(2)))
            continue
        s = line.strip()
        if s in MARKERS:
            out["markers"].append(s)
            phase = "measure" if s == "EXP4_MEASUREMENT_BEGIN" else (None if s.endswith("_END") else phase)
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
        m = GATE_TAG.match(s)
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
                 "complete": False}
    for line in lines:
        s = line.strip()
        m = TAG.match(s)
        if not m:
            if s == "EXP4_MIRRORED_COMPLETE":
                out["complete"] = True
            continue
        tag, payload = m.group(1), json.loads(m.group(2))
        if tag == "EXP4_PRE_LEG_GPU_STATE":
            out["pre_leg"][payload["leg"]] = payload
        elif tag == "EXP4_LEG_EXIT":
            out["leg_exit"][payload["leg"]] = payload
        elif tag == "EXP4_LEG_START":
            out["leg_start"][payload["leg"]] = payload
        elif tag == "EXP4_PROCESS_EXIT":
            out["process_exit"][payload["label"]] = payload
        elif tag == "EXP4_WATCHDOG_TIMEOUT":
            out["watchdog_timeouts"].append(payload)
        else:
            out[tag] = payload
    return out


# ---------------------------------------------------------- config diff
CONFIG_SECTIONS = ("EXP4_REQUESTED_ENGINE_KWARGS", "EXP4_EFFECTIVE_ENGINE_CONFIG", "EXP4_WORKLOAD", "EXP4_KV_DTYPE")


def leg_config(p: dict) -> dict:
    t = p["tags"]
    return {
        **flatten("requested", t.get("EXP4_REQUESTED_ENGINE_KWARGS", {})),
        **flatten("effective", t.get("EXP4_EFFECTIVE_ENGINE_CONFIG", {})),
        **flatten("workload", t.get("EXP4_WORKLOAD", {})),
        **flatten("kv_dtype", t.get("EXP4_KV_DTYPE", {})),
    }


def config_diff(parsed: dict) -> dict:
    """All six legs: every non-allowlisted field identical across all legs;
    allowlisted (kv_cache_dtype-induced) fields identical within each dtype.
    Not evaluated at all unless every leg reported every config section."""
    dtype_of = {label: d for _, label, d in LEGS}
    missing = {label: [s for s in CONFIG_SECTIONS if s not in parsed[k]["tags"]] for k, label, _ in LEGS}
    missing = {label: v for label, v in missing.items() if v}
    base = {
        "rule": ("All non-dtype engine/workload fields must be identical across all six legs "
                 "(A1 B1 C1 C2 B2 A2). Only DTYPE_INDUCED_ALLOWLIST fields may differ, and only "
                 "between dtypes (never between the two legs of the same dtype). Capacity/latency "
                 "are outcomes, compared separately. Not evaluated unless all six legs reported "
                 "their config."),
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
def gpu_leg_clean(pre: dict, baseline: dict) -> bool:
    """Independent re-check of the Modal app's clean-state decision."""
    if not pre or not pre.get("readings") or not baseline:
        return False
    last = pre["readings"][-1]
    return (not last["compute_apps"]
            and pre.get("tolerance_mib") == GPU_CLEAN_TOLERANCE_MIB
            and all(u <= b + GPU_CLEAN_TOLERANCE_MIB
                    for u, b in zip(last["memory_used_mib"], baseline["memory_used_mib"])))


def implied_bytes_per_token(gib: float | None, capacity_tokens: int | None) -> float | None:
    if not gib or not capacity_tokens:
        return None
    return gib * 2**30 / capacity_tokens


def integrity(parsed: dict, gate: dict, top: dict, diff: dict) -> dict:
    """Every check has an explicit state. Only 'passed' counts as passed;
    all_ok requires every check to be 'passed'."""
    checks: list[dict] = []

    def add(name: str, category: str, state, observed=None) -> None:
        if isinstance(state, bool):
            state = PASSED if state else FAILED
        checks.append({"check": name, "category": category, "state": state, "observed": observed})

    env = top.get("EXP4_ENVIRONMENT", {})
    gpus = env.get("gpus", [])
    baseline = top.get("EXP4_GPU_BASELINE", {})
    add("all six legs completed (A1 B1 C1 C2 B2 A2)", "completion", top.get("complete", False))
    add("no watchdog timeout", "watchdog", not top["watchdog_timeouts"], top["watchdog_timeouts"] or None)
    add("exactly one GPU, H100", "environment",
        (len(gpus) == 1 and "H100" in gpus[0].get("name", "")) if env else NOT_EVALUATED,
        [g.get("name") for g in gpus] or None)
    add("leg order is A1 B1 C1 C2 B2 A2", "environment",
        (env.get("leg_labels") == [label for _, label, _ in LEGS]
         and env.get("leg_dtypes") == [d for _, _, d in LEGS]) if env else NOT_EVALUATED,
        {"labels": env.get("leg_labels"), "dtypes": env.get("leg_dtypes")} if env else None)
    add("rabit_kv2.py in image is frozen source", "environment",
        env.get("rabit_kv2_sha256_lf") == EXPECTED_RABIT_SHA256_LF if env else NOT_EVALUATED,
        env.get("rabit_kv2_sha256_lf"))
    add("idle baseline recorded with no compute process", "gpu_clean",
        bool(baseline) and not baseline.get("compute_apps") and "memory_used_mib" in baseline,
        baseline.get("memory_used_mib"))

    # Correctness gate (frozen RABIT-KV gate, run once).
    gate_ran = "EXP4_GATE_START" in top
    gstate = (lambda ok: ok) if gate_ran else (lambda ok: NOT_RUN)
    add("gate: process exit 0", "gate", gstate(top.get("EXP4_GATE_EXIT", {}).get("returncode") == 0),
        top.get("EXP4_GATE_EXIT"))
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
        add("gate completed before first leg", "gate", "EXP4_GATE_EXIT" in top)

    model = top.get("EXP4_MODEL", {})
    add("model snapshot hashed", "model", bool(model.get("files")) if model else NOT_RUN)

    for k, label, d in LEGS:
        p = parsed[k]
        t = p["tags"]
        started = label in top["leg_start"]

        def leg(name: str, category: str, ok, observed=None, _started=started) -> None:
            add(f"{label}: {name}", category, ok if _started else NOT_RUN, observed if _started else None)

        def measured(name: str, category: str, ok, observed=None, _p=p, _started=started) -> None:
            if not _started:
                add(f"{label}: {name}", category, NOT_RUN)
            elif "EXP4_MEASUREMENT_BEGIN" not in _p["markers"]:
                add(f"{label}: {name}", category, NOT_EVALUATED, "measurement phase never started")
            else:
                add(f"{label}: {name}", category, ok, observed)

        pre = top["pre_leg"].get(label)
        add(f"{label}: GPU clean before leg (no compute process, within {GPU_CLEAN_TOLERANCE_MIB} MiB of baseline)",
            "gpu_clean", gpu_leg_clean(pre, baseline) if pre else NOT_RUN,
            pre["readings"][-1] if pre and pre.get("readings") else None)
        leg("worker exit 0", "leg", (top["leg_exit"].get(label) or {}).get("returncode") == 0,
            top["leg_exit"].get(label))
        leg("worker reports leg/dtype", "leg", t.get("EXP4_LEG") == {"leg": label, "kv_cache_dtype": d},
            t.get("EXP4_LEG"))
        leg("RABIT frozen-source markers", "leg",
            bool(t.get("EXP4_RABIT_MARKERS")) and all(t["EXP4_RABIT_MARKERS"].values()))
        req = t.get("EXP4_REQUESTED_ENGINE_KWARGS", {})
        leg("requested kwargs as planned", "config", req == requested_kwargs(d, req.get("model", "<missing>")))
        leg("model path is the hashed snapshot", "model",
            bool(model) and req.get("model") == model.get("snapshot_dir"), req.get("model"))
        eff = t.get("EXP4_EFFECTIVE_ENGINE_CONFIG", {})
        bad = {x: eff.get(x, "<missing>") for x, v in EXPECTED_EFFECTIVE.items() if eff.get(x, "<missing>") != v}
        leg("effective engine config as planned (eager, Triton, limits, no KV-scale/dtype override source)",
            "config", bool(eff) and not bad, bad or None)
        cm = {"name": normalize_mode(eff.get("compilation_mode"), COMPILATION_MODE_VALUES),
              "raw": normalize_mode(eff["compilation_mode_raw"], COMPILATION_MODE_VALUES)
              if "compilation_mode_raw" in eff else None}
        leg("compilation mode = NONE (torch.compile off)", "config",
            cm["name"] == "NONE" and cm["raw"] in (None, "NONE"), cm)
        gm = {"name": normalize_mode(eff.get("cudagraph_mode"), CUDAGRAPH_MODE_VALUES),
              "raw": normalize_mode(eff["cudagraph_mode_raw"], CUDAGRAPH_MODE_VALUES)
              if "cudagraph_mode_raw" in eff else None}
        leg("CUDA graph mode = NONE", "config", gm["name"] == "NONE" and gm["raw"] in (None, "NONE"), gm)
        wl = t.get("EXP4_WORKLOAD", {})
        leg("workload as planned", "workload",
            all(wl.get(x) == v for x, v in EXPECTED_WORKLOAD.items()) and bool(wl.get("prompt_token_ids_sha256")),
            wl or None)
        kv = t.get("EXP4_KV_DTYPE", {})
        leg("resolved KV dtype", "kv_dtype",
            bool(kv) and kv.get("requested_kv_cache_dtype") == d
            and all(kv.get(x, "<missing>") == v for x, v in EXPECTED_KV[d].items()), kv or None)
        cap = t.get("EXP4_CAPACITY", {})
        leg("capacity = num_gpu_blocks x block_size", "capacity",
            bool(cap) and cap["capacity_tokens"] == cap["num_gpu_blocks"] * cap["block_size"], cap or None)
        leg("capacity matches engine log 'GPU KV cache size'", "capacity",
            bool(cap) and p["kv_log_tokens"] == cap.get("capacity_tokens"), p["kv_log_tokens"])
        leg("available KV cache memory logged", "capacity", bool(p["kv_log_gib"]), p["kv_log_gib"])
        if d in EXPECTED_BYTES_PER_ELEMENT:
            want = KV_ELEMENTS_PER_TOKEN * EXPECTED_BYTES_PER_ELEMENT[d]
            bpt = implied_bytes_per_token(p["kv_log_gib"], cap.get("capacity_tokens") if cap else None)
            leg(f"physical bytes/token consistent with {EXPECTED_BYTES_PER_ELEMENT[d]}-byte KV elements "
                f"({want} B/token +/-{BYTES_PER_TOKEN_REL_TOL:.0%})", "kv_dtype",
                bpt is not None and abs(bpt / want - 1) <= BYTES_PER_TOKEN_REL_TOL,
                round(bpt, 1) if bpt else None)
        measured(f"warmups == {WARMUPS_PER_LEG}", "measurement", len(p["warmups"]) == WARMUPS_PER_LEG,
                 len(p["warmups"]))
        measured(f"measured reps == {REPS_PER_LEG}", "measurement",
                 [r["rep"] for r in p["samples"]] == list(range(REPS_PER_LEG)), len(p["samples"]))
        measured(f"every request {CONTEXT_TOKENS} prompt / {OUTPUT_TOKENS} output tokens", "measurement",
                 bool(p["samples"]) and all(r["prompt_tokens"] == CONTEXT_TOKENS and r["output_tokens"] == OUTPUT_TOKENS
                                            for r in p["samples"] + p["warmups"]))
        measured("every request recorded generated-token hash", "measurement",
                 bool(p["samples"]) and all(isinstance(r.get("output_token_ids_sha256"), str)
                                            and len(r["output_token_ids_sha256"]) == 64
                                            for r in p["samples"] + p["warmups"]))
        leg("marker order", "leg", p["markers"] == MARKERS, p["markers"])
        measured("no Triton JIT compilation during measurement", "measurement",
                 "EXP4_MEASUREMENT_END" in p["markers"] and not p["jit_during_measurement"],
                 p["jit_during_measurement"] or None)

    # Cross-leg checks: evaluated only when every required leg has the data.
    add("config: all non-dtype fields identical across all six legs", "config",
        diff["status"], diff.get("violations") or diff.get("missing_config_sections_by_leg"))
    hashes = {label: parsed[k]["tags"].get("EXP4_WORKLOAD", {}).get("prompt_token_ids_sha256")
              for k, label, _ in LEGS}
    add("prompt token hash identical across all six legs", "workload",
        (len(set(hashes.values())) == 1) if all(hashes.values()) else NOT_EVALUATED, hashes)
    for d in DTYPES:
        caps = {label: parsed[k]["tags"].get("EXP4_CAPACITY") for k, label, dd in LEGS if dd == d}
        l1, l2 = list(caps)
        add(f"{d}: duplicate capacity identical ({l1} == {l2})", "capacity",
            (len({json.dumps(c, sort_keys=True) for c in caps.values()}) == 1)
            if all(caps.values()) else NOT_EVALUATED, caps)
        legs_measured = all("EXP4_MEASUREMENT_END" in parsed[k]["markers"] for k, _, dd in LEGS if dd == d)
        n = sum(len(parsed[k]["samples"]) for k, _, dd in LEGS if dd == d)
        add(f"{d}: pooled measured samples == {REPS_PER_DTYPE}", "measurement",
            (n == REPS_PER_DTYPE) if legs_measured else NOT_EVALUATED, n)

    counts = {s: sum(1 for c in checks if c["state"] == s) for s in (PASSED, FAILED, NOT_RUN, NOT_EVALUATED)}
    return {"state_semantics": "passed / failed / not_run (leg never started) / not_evaluated "
                               "(required data or legs missing); only 'passed' counts as passed",
            "checks": checks, "counts": counts, "all_ok": counts[PASSED] == len(checks),
            "failed_categories": sorted({c["category"] for c in checks if c["state"] == FAILED}),
            "non_passed_categories": sorted({c["category"] for c in checks if c["state"] != PASSED})}


def p90(values: list[float]) -> float:
    return statistics.quantiles(values, n=10, method="inclusive")[8]


def latency_wording(x: str, y: str, delta: float) -> str:
    if delta > 0:
        return f"{SHORT[x]} slower than {SHORT[y]}"
    if delta < 0:
        return f"{SHORT[x]} faster than {SHORT[y]}"
    return f"{SHORT[x]} equal to {SHORT[y]}"


def order_effect(first: list[dict], second: list[dict], l1: str, l2: str) -> dict:
    out = {"first_leg": l1, "second_leg": l2}
    for key in ("tpot_ms", "ttft_ms", "wall_ms"):
        a, b = [r[key] for r in first], [r[key] for r in second]
        ma, mb = statistics.median(a), statistics.median(b)
        out[key] = {
            "first_median": ma, "second_median": mb,
            "median_drift_pct": (mb / ma - 1) * 100,
            "first_range": [min(a), max(a)], "second_range": [min(b), max(b)],
            "ranges_overlap": max(min(a), min(b)) <= min(max(a), max(b)),
        }
    return out


def build_summary(parsed: dict, gate: dict, top: dict) -> dict:
    per_leg = {}
    for k, label, d in LEGS:
        p = parsed[k]
        cap = p["tags"]["EXP4_CAPACITY"]
        per_leg[label] = {
            "index": k, "kv_cache_dtype": d,
            "kv_dtype": p["tags"]["EXP4_KV_DTYPE"],
            "capacity": cap,
            "available_kv_cache_memory_gib_logged": p["kv_log_gib"],
            "physical_bytes_per_token_implied": round(implied_bytes_per_token(p["kv_log_gib"], cap["capacity_tokens"]), 1),
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
        cap = legs[0][1]["tags"]["EXP4_CAPACITY"]
        gib = legs[0][1]["kv_log_gib"]
        hashes = sorted({r["output_token_ids_sha256"] for _, p in legs for r in p["samples"] + p["warmups"]})
        pooled[d] = {
            "legs": [label for label, _ in legs],
            "resolved_kv_dtype": legs[0][1]["tags"]["EXP4_KV_DTYPE"],
            "capacity_tokens": cap["capacity_tokens"],
            "num_gpu_blocks": cap["num_gpu_blocks"],
            "block_size": cap["block_size"],
            "duplicate_capacity_identical": True,
            "available_kv_cache_memory_gib_logged": gib,
            "physical_bytes_per_token_implied": round(implied_bytes_per_token(gib, cap["capacity_tokens"]), 1),
            "headline": {
                "tpot_ms_median": statistics.median(r["tpot_ms"] for r in rows),
                "tpot_ms_p90": p90([r["tpot_ms"] for r in rows]),
                "ttft_ms_median": statistics.median(r["ttft_ms"] for r in rows),
                "wall_ms_median": statistics.median(r["wall_ms"] for r in rows),
            },
            "pooled_tpot_ms": stats(rows, "tpot_ms"),
            "pooled_ttft_ms": stats(rows, "ttft_ms"),
            "pooled_wall_ms": stats(rows, "wall_ms"),
            "pooled_measured_samples": rows,
            "generated_token_hashes_informational": {
                "distinct_hashes_over_warmups_and_samples": hashes,
                "deterministic_within_dtype": len(hashes) == 1,
                "note": "Functional evidence only (greedy, same prompt). Not a quality metric; "
                        "hashes are not compared across dtypes.",
            },
        }

    capacity = {}
    deltas = {}
    for x, y in PAIRS:
        name = f"{SHORT[x]}_vs_{SHORT[y]}"
        cx, cy = pooled[x]["capacity_tokens"], pooled[y]["capacity_tokens"]
        capacity[f"{SHORT[x]}_over_{SHORT[y]}"] = {
            SHORT[x]: cx, SHORT[y]: cy, "ratio": cx / cy, "signed_delta_tokens": cx - cy,
            "wording": (f"{SHORT[x]} holds {cx / cy:.4f}x the physical KV tokens of {SHORT[y]}"),
        }
        deltas[name] = {}
        for metric, key in (("tpot_median", "tpot_ms_median"), ("tpot_p90", "tpot_ms_p90"),
                            ("ttft_median", "ttft_ms_median"), ("wall_median", "wall_ms_median")):
            vx, vy = pooled[x]["headline"][key], pooled[y]["headline"][key]
            deltas[name][metric] = {
                SHORT[x]: vx, SHORT[y]: vy,
                f"signed_delta_ms_{SHORT[x]}_minus_{SHORT[y]}": vx - vy,
                f"signed_delta_pct_vs_{SHORT[y]}": (vx / vy - 1) * 100,
                "direction": latency_wording(x, y, vx - vy),
            }

    order_effects = {}
    for d in DTYPES:
        (k1, l1, _), (k2, l2, _) = [leg for leg in LEGS if leg[2] == d]
        order_effects[f"{l1}_vs_{l2}"] = {"kv_cache_dtype": d,
                                         **order_effect(parsed[k1]["samples"], parsed[k2]["samples"], l1, l2)}

    return {
        "experiment": "MLSys 2027 Experiment 4 -- matched BF16 vs native FP8 vs RABIT-KV physical "
                      "capacity and latency (mirrored A-B-C-C-B-A)",
        "scope": SCOPE_NOTE,
        "fp8_quality_evaluated": False,
        "experiment3_samples_used": False,
        "capacity_label": CAPACITY_LABEL,
        "latency_label": LATENCY_LABEL,
        "native_fp8_source_audit": NATIVE_FP8_SOURCE_AUDIT,
        "design": {"legs": [{"index": k, "leg": label, "kv_cache_dtype": d} for k, label, d in LEGS],
                   "letters": {LETTER[d]: d for d in DTYPES},
                   "warmups_per_leg_excluded": WARMUPS_PER_LEG, "measured_reps_per_leg": REPS_PER_LEG,
                   "measured_reps_per_dtype": REPS_PER_DTYPE, "context_tokens": CONTEXT_TOKENS,
                   "output_tokens": OUTPUT_TOKENS, "same_container_same_gpu": True,
                   "fresh_process_per_leg": True, "gate_timeout_s": GATE_TIMEOUT_S,
                   "leg_timeout_s": LEG_TIMEOUT_S},
        "correctness_gate": {
            "scope": "frozen RABIT-KV correctness gate, run once before any leg",
            "command": (gate.get("begin") or {}).get("pytest_command"),
            "wrapper": GATE_COMMAND_DOC,
            "passed": (gate.get("result") or {}).get("passed"),
            "pytest_exit": gate.get("pytest_exit"),
            "pytest_passed": gate.get("pytest_passed"),
            "pytest_warnings": gate.get("pytest_warnings"),
            "rabit_kv2_sha256_lf": (gate.get("begin") or {}).get("rabit_kv2_sha256_lf"),
            "process": top["process_exit"].get("gate"),
        },
        "gpu_clean_state": {"baseline": top.get("EXP4_GPU_BASELINE"),
                            "tolerance_mib": GPU_CLEAN_TOLERANCE_MIB,
                            "pre_leg": top["pre_leg"], "post_run": top.get("EXP4_POST_RUN_GPU_STATE")},
        "pooled_per_dtype": pooled,
        "per_leg": per_leg,
        "capacity_ratios": capacity,
        "signed_latency_deltas_pooled": {
            "note": "Signed deltas (x - y) over pooled 30 samples per dtype from this session only. "
                    "Direction wording follows the measured sign.",
            **deltas,
        },
        "order_effects": order_effects,
        "environment": top.get("EXP4_ENVIRONMENT"),
        "model": top.get("EXP4_MODEL"),
    }


def analyze(session_text: str, write: bool) -> tuple[dict, dict, dict | None]:
    leg_lines, gate_lines, top_lines = demux(session_text)
    parsed = {k: parse_worker(leg_lines[k]) for k, _, _ in LEGS}
    gate = parse_gate(gate_lines)
    top = parse_top(top_lines)
    diff = config_diff(parsed)
    integ = integrity(parsed, gate, top, diff)
    integ["correctness_gate"] = gate
    integ["gpu_clean_state"] = {"baseline": top.get("EXP4_GPU_BASELINE"), "tolerance_mib": GPU_CLEAN_TOLERANCE_MIB,
                                "max_wait_s": GPU_CLEAN_MAX_WAIT_S, "pre_leg": top["pre_leg"]}
    integ["processes"] = {"exits": top["process_exit"], "watchdog_timeouts": top["watchdog_timeouts"]}
    summary = build_summary(parsed, gate, top) if integ["all_ok"] else None
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        GATE_LOG.write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
        for d in DTYPES:
            parts = []
            for k, label, dd in LEGS:
                if dd == d:
                    parts.append(f"===== EXP4 LEG {label} (index {k}, {d}) =====")
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
    """Single terminal path: protected-path post-check, archived-attempt and
    Experiment 3 evidence integrity, persisted in manifest.json."""
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
    prov = manifest["provenance"]
    manifest["archived_attempts_unchanged"] = archived_attempts_digest() == prov.get("archived_attempts_sha256", {})
    if not manifest["archived_attempts_unchanged"]:
        manifest["status"] = "failed"
        manifest.setdefault("failure", {"stage": "archived_attempt_modified",
                                        "reason": "an archived failed_attempt_* file changed during the run"})
    manifest["exp3_evidence_unchanged"] = exp3_evidence_digest() == prov.get("exp3_evidence_sha256_raw", {})
    if not manifest["exp3_evidence_unchanged"]:
        manifest["status"] = "failed"
        manifest.setdefault("failure", {"stage": "exp3_evidence_modified",
                                        "reason": "an Experiment 3 evidence file changed during the run"})
    write_manifest(manifest)


def classify_failure(code: int, integ: dict) -> dict:
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
    row = {"name": "mirrored_a_b_c_c_b_a", "command": command, "log": rel(SESSION_LOG),
           "gate_command": GATE_COMMAND_DOC,
           "worker_commands": {label: worker_command(label, d) for _, label, d in LEGS},
           "started_utc": now(), "completed_utc": None, "returncode": None, "status": "running"}
    manifest["runs"].append(row)
    write_manifest(manifest)

    code = stream_command(command, SESSION_LOG, {"EXP4_VLLM_SNAPSHOT": snapshot["path"]})
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
            f"\nEXPERIMENT 4 STOPPED ({failure['stage']}): modal exit={code}; integrity {integ['counts']}; "
            f"non-passed categories {integ['non_passed_categories']}. Logs and {rel(INTEGRITY)} preserved."
        )

    row["status"] = "passed"
    finalize(manifest, "passed", None)
    if manifest["status"] != "passed":
        raise SystemExit(f"\nEXPERIMENT 4 FAILED at post-run checks: {manifest.get('failure')}")
    print("\n" + "=" * 118 + "\nEXPERIMENT 4: MIRRORED MATCHED RUN PASSED\n"
          f"Summary: {SUMMARY}\n" + "=" * 118)
    return 0


def main(argv: list[str] | None = None) -> int:
    make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Preflight + equivalence proof + planned commands and config diff. "
                         "No Modal/GPU, no files written.")
    args = ap.parse_args(argv)

    print("RABIT-KV MLSys 2027 -- Experiment 4: matched BF16 vs native FP8 vs RABIT-KV physical capacity/latency")
    print(f"Mirrored legs (one container, one GPU, fresh process per leg): "
          f"{[f'{label}={d}' for _, label, d in LEGS]}")
    print(f"Per leg: {WARMUPS_PER_LEG} full-shape warmups (excluded) + {REPS_PER_LEG} measured reps "
          f"-> {REPS_PER_DTYPE} measured reps per dtype. Workload: {CONTEXT_TOKENS} ctx / {OUTPUT_TOKENS} out")
    print(f"Watchdogs: gate {GATE_TIMEOUT_S}s, each leg {LEG_TIMEOUT_S}s (own process group, group kill, abort); "
          f"Modal backstop {MODAL_FUNCTION_TIMEOUT_S}s")
    print("Scope: physical capacity + latency only; no FP8 quality claim; no Experiment 3 samples reused.")
    print()

    prov = preflight(args.dry_run)
    print("Preflight OK.")
    for k in ("git_branch", "git_head", "vllm_kvquant_tree", "rabit_kv2_sha256", "runner_script_sha256",
              "modal_app_sha256", "worker_sha256", "correctness_gate_sha256", "watchdog_sha256",
              "exp3_worker_sha256", "exp3_modal_app_sha256", "canonical_benchmark_deployment_sha256"):
        print(f"  {k:<40} {prov[k]}")
    print("  equivalence: engine kwargs canonical == exp3 == exp4 (except kv_cache_dtype/model path); workload "
          "and protocol constants == exp3; prompt construction and timed region AST == exp3; image AST == "
          "canonical; clean-state/watchdog helpers AST == exp3; gate regression() AST == canonical; no engine RPC")
    print(f"  protected paths: {len(prov['protected_paths'])} (incl. all Experiment 3 evidence); "
          f"Experiment 3 evidence files hashed: {len(prov['exp3_evidence_sha256_raw'])}; "
          f"archived Exp4 attempts hashed: {len(prov['archived_attempts_sha256'])}")
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
    print("\nPlanned requested-config comparison across the six legs "
          "(post-run the same check covers effective config, workload and KV dtype):")
    print(json.dumps({"fields_compared": len(keys), "differing_fields": differing,
                      "dtype_induced_allowlist": DTYPE_INDUCED_ALLOWLIST,
                      "only_allowlisted_fields_differ": set(differing) <= set(DTYPE_INDUCED_ALLOWLIST)},
                     indent=2))
    if not set(differing) <= set(DTYPE_INDUCED_ALLOWLIST):
        raise SystemExit("Planned configs differ in a non-dtype field -- refusing.")

    print("\nLocal command (one Modal run):\n  " + " ".join(build_command()))
    print("Step 0 (in container): idle GPU baseline; no compute process allowed")
    print(f"Step 1 (in container): frozen RABIT-KV correctness gate, fresh process group, watchdog "
          f"{GATE_TIMEOUT_S}s, must pass before any leg:\n  " + " ".join(GATE_COMMAND_DOC))
    for k, label, d in LEGS:
        print(f"Step {k + 1} (in container): GPU clean check, then leg {label} ({d}), watchdog {LEG_TIMEOUT_S}s:\n  "
              + " ".join(worker_command(label, d)))
    print(f"Outputs: {rel(OUT_DIR)}/ ({SESSION_LOG.name}, {GATE_LOG.name}, {', '.join(LOG_NAME.values())}, "
          f"{MANIFEST.name}, {CONFIG_DIFF.name}, {INTEGRITY.name}, {SUMMARY.name})")

    if args.dry_run:
        print("\n--dry-run: no Modal/GPU commands executed, no snapshot built, no files written.")
        return 0

    manifest = {
        "experiment": "Experiment 4 -- matched BF16 vs native FP8 vs RABIT-KV physical capacity and latency "
                      "(mirrored A1 B1 C1 C2 B2 A2)",
        "plan_reference": "docs/MLSYS_EXPERIMENT_PLAN.md",
        "model": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "gpu": "NVIDIA H100 80GB (Modal)",
        "scope": SCOPE_NOTE,
        "native_fp8_source_audit": NATIVE_FP8_SOURCE_AUDIT,
        "legs": [{"index": k, "leg": label, "kv_cache_dtype": d} for k, label, d in LEGS],
        "protocol": {"warmups_per_leg_excluded": WARMUPS_PER_LEG, "measured_reps_per_leg": REPS_PER_LEG,
                     "measured_reps_per_dtype": REPS_PER_DTYPE, "context_tokens": CONTEXT_TOKENS,
                     "output_tokens": OUTPUT_TOKENS, "same_container_same_gpu": True,
                     "fresh_process_per_leg": True, "dtype_induced_allowlist": DTYPE_INDUCED_ALLOWLIST,
                     "gpu_clean_tolerance_mib": GPU_CLEAN_TOLERANCE_MIB,
                     "gpu_clean_max_wait_s": GPU_CLEAN_MAX_WAIT_S,
                     "gate_timeout_s": GATE_TIMEOUT_S, "leg_timeout_s": LEG_TIMEOUT_S,
                     "modal_function_backstop_timeout_s": MODAL_FUNCTION_TIMEOUT_S,
                     "retries": 0},
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
        raise SystemExit(f"\nEXPERIMENT 4 RUNNER FAILED: {type(exc).__name__}: {exc}{extra}\n"
                         f"Manifest marked failed; partial logs preserved under {rel(OUT_DIR)}/. Not retried.") from exc


if __name__ == "__main__":
    raise SystemExit(main())
