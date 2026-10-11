"""
RABIT-KV MLSys 2027 -- Experiment 5 runner: BF16 vs RABIT-KV context-length
scaling on the real vLLM engine (H100, Modal).

PHYSICAL real-engine latency at each context point, plus clearly separated
memory quantities:
  * MEASURED  -- engine-reported allocator capacity (num_gpu_blocks x
    block_size), engine-logged available KV memory and model-load memory,
    device-wide nvidia-smi memory.used (idle before the cell, after engine
    init, after the measured reps);
  * DERIVED   -- live paged-KV usage of the request at its peak:
    ceil((prompt + output - 1) / block_size) blocks x (available KV bytes /
    num_gpu_blocks). No engine RPC / probe is used to observe live KV usage.
Allocator capacity is a property of the fixed engine configuration and does
not vary with request context; it is recorded per cell only as a
consistency check (it must be identical across every context of a dtype).

Design (one `modal run` of benchmarks/mlsys2027/exp5_deployment_modal.py):
  * one container, one physical H100, one image, one model snapshot;
  * frozen RABIT-KV correctness gate (exp3_correctness_gate.py, unchanged)
    once, before any cell;
  * 12 cells = context grid {512, 2048, 4096, 8192, 16384, 32768} x
    {A = bfloat16, B = rabit_kv2}, ascending context, dtype order alternating
    per context (A B | B A | A B | B A | A B | B A), each a fresh
    worker/engine process with 5 full-shape warmups (excluded) + 15 measured
    reps;
  * prompt = context point, except where prompt + 32 output tokens would
    exceed max_model_len = 32768: the 32768 point uses a 32736-token prompt
    so prompt + output = 32768 (this fork rejects a prompt equal to
    max_model_len and stops generation when prompt + output reaches it);
  * no Experiment 3/4 sample is read or reused;
  * gate and every cell under the unchanged Experiment 3 watchdog (gate 600 s,
    cell 900 s, whole-group kill, no retry; a timeout aborts the sweep);
    GPU clean state (no compute process, within 256 MiB of idle) before every
    cell, else abort; a cell whose worker exits non-zero (e.g. OOM) is kept as
    a failed cell and the sweep continues;
  * engine arguments identical except kv_cache_dtype; workload identical
    except the context point / prompt.

Before any run this script proves by AST that the worker's engine kwargs,
output length, prompt construction and timed region equal the frozen
Experiment 3 worker (and the canonical runner), that the Modal image equals
the canonical one, that the Modal clean-state / watchdog helpers equal
Experiment 3's, and that the gate is the canonical regression(). After the
run every integrity check is passed / failed / not_run / not_evaluated; only
a sweep in which EVERY check passed produces a summary. Every terminal path
runs the protected-path post-check and persists it in manifest.json.

This script never modifies Experiment 1-4 code or evidence, vllm-kvquant or
any canonical result; it writes ONLY top-level files under
results/mlsys2027/context_scaling/ and never touches archived attempts there.

Usage:
    python benchmarks/mlsys2027/run_experiment5_context_scaling.py --dry-run
    python benchmarks/mlsys2027/run_experiment5_context_scaling.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment3_deployment as r3  # noqa: E402  (frozen; helpers only, never modified)
from run_experiment3_deployment import (  # noqa: E402
    COMPILATION_MODE_VALUES, CUDAGRAPH_MODE_VALUES, FAILED, NOT_EVALUATED, NOT_RUN, PASSED,
    _function, _module_assign, canonical_llm_kwargs, canonical_runner_source,
    flatten, make_console_encoding_safe, normalize_mode, now, rel, run_git, sha256, sha256_raw,
    stats, stream_command,
)

ROOT = r3.ROOT
HERE = ROOT / "benchmarks" / "mlsys2027"
RUNNER_SCRIPT = Path(__file__).resolve()
MODAL_APP = HERE / "exp5_deployment_modal.py"
WORKER = HERE / "exp5_engine_worker.py"
GATE = HERE / "exp3_correctness_gate.py"          # reused unchanged
WATCHDOG = HERE / "exp3_watchdog.py"              # reused unchanged
EXP3_WORKER = HERE / "exp3_engine_worker.py"      # reference for AST equivalence
EXP3_MODAL_APP = HERE / "exp3_deployment_modal.py"
EXP3_RUNNER = HERE / "run_experiment3_deployment.py"
CANONICAL_DEPLOYMENT = r3.CANONICAL_DEPLOYMENT
RABIT_KV2 = r3.RABIT_KV2
EXPECTED_RABIT_SHA256_LF = r3.EXPECTED_RABIT_SHA256_LF

OUT_DIR = ROOT / "results" / "mlsys2027" / "context_scaling"
ARCHIVE_GLOB = "failed_attempt_*"
SESSION_LOG = OUT_DIR / "modal_session.log"
GATE_LOG = OUT_DIR / "correctness_gate.log"
MANIFEST = OUT_DIR / "manifest.json"
CONFIG_DIFF = OUT_DIR / "matched_config_diff.json"
INTEGRITY = OUT_DIR / "integrity_check.json"
SUMMARY = OUT_DIR / "context_scaling_summary.json"

EVIDENCE_DIRS = [ROOT / "results" / "mlsys2027" / "deployment", ROOT / "results" / "mlsys2027" / "fp8_baseline"]
PROTECTED_PATHS = [
    *r3.PROTECTED_PATHS,
    *EVIDENCE_DIRS,  # all Experiment 3 and Experiment 4 evidence
    *sorted(HERE.glob("*exp1*")), *sorted(HERE.glob("*exp2*")), *sorted(HERE.glob("*exp3*")),
    *sorted(HERE.glob("*exp4*")), *sorted(HERE.glob("run_experiment[1234]_*")),
]
MUST_BE_COMMITTED = [RUNNER_SCRIPT, MODAL_APP, WORKER, GATE, WATCHDOG]

A, B = "bfloat16", "rabit_kv2"
DTYPES = [A, B]
LETTER = {A: "A", B: "B"}
SHORT = {A: "bf16", B: "rabit_kv2"}
CONTEXT_GRID = [512, 2048, 4096, 8192, 16384, 32768]
OUTPUT_TOKENS = 32
MAX_MODEL_LEN = 32768
BLOCK_SIZE = 32
MAX_NUM_BATCHED_TOKENS = 16384


def prompt_tokens_for(context_point: int) -> int:
    """Prompt = context point, unless prompt + output would exceed max_model_len
    (vLLM rejects prompt == max_model_len and caps prompt + output at it)."""
    return context_point if context_point + OUTPUT_TOKENS <= MAX_MODEL_LEN else MAX_MODEL_LEN - OUTPUT_TOKENS


CONDITIONING_CONTEXT = 512
# Unmeasured conditioning cells run first (after gate + idle baseline) so that
# no official cell is the first engine process in the container.
# (index, label, dtype, context point, prompt tokens)
CONDITIONING = [(1, f"conditioning_A{CONDITIONING_CONTEXT}", A, CONDITIONING_CONTEXT, CONDITIONING_CONTEXT),
                (2, f"conditioning_B{CONDITIONING_CONTEXT}", B, CONDITIONING_CONTEXT, CONDITIONING_CONTEXT)]


def _build_legs() -> list[tuple[int, str, str, int, int]]:
    legs, k = [], len(CONDITIONING)
    for i, c in enumerate(CONTEXT_GRID):
        order = [A, B] if i % 2 == 0 else [B, A]
        for d in order:
            k += 1
            legs.append((k, f"{LETTER[d]}{c}", d, c, prompt_tokens_for(c)))
    return legs


# Official measured cells: (index, label, dtype, context point, prompt tokens)
LEGS = _build_legs()
ALL_CELLS = CONDITIONING + LEGS
ROLE = {**{label: "conditioning" for _, label, *_ in CONDITIONING}, **{label: "measured" for _, label, *_ in LEGS}}
LOG_NAME = {label: f"{SHORT[d]}_ctx{c}.log" for _, label, d, c, _ in LEGS}
LOG_NAME.update({label: f"conditioning_{SHORT[d]}_ctx{c}.log" for _, label, d, c, _ in CONDITIONING})
WARMUPS_PER_LEG = 5
REPS_PER_LEG = 15
WORKLOAD_FAILURE_EXIT = 75
CONTINUABLE_FAILURE_KINDS = ("request_oom", "request_execution_failure")

GPU_CLEAN_TOLERANCE_MIB = 256
GPU_CLEAN_MAX_WAIT_S = 60
GATE_TIMEOUT_S = 600
LEG_TIMEOUT_S = 900
WATCHDOG_BUDGET_S = GATE_TIMEOUT_S + len(ALL_CELLS) * LEG_TIMEOUT_S  # 600 + 14 x 900 = 13200
MODAL_FUNCTION_TIMEOUT_S = 14400  # backstop only: must exceed the total watchdog budget


def display_label(context_point: int, prompt: int) -> str:
    """Plot/table label: the 32768 point is a model-limit point, not a 32768-token prompt."""
    if prompt == context_point:
        return f"{context_point} ({prompt} input + {OUTPUT_TOKENS} output)"
    return f"{context_point // 1024}K model-limit point ({prompt} input + {OUTPUT_TOKENS} output)"

# Intended differences, two classes. Dtype-induced fields may differ only
# between dtypes and must be constant across all contexts of a dtype;
# context-induced fields may differ only between contexts and must be
# identical for both dtypes at one context. Everything else: identical.
DTYPE_INDUCED_ALLOWLIST = [
    "requested.kv_cache_dtype",
    "kv_dtype.requested_kv_cache_dtype",
    "kv_dtype.engine_cache_dtype",
    "kv_dtype.resolved_kv_torch_dtype",
    "kv_dtype.kv_quant_mode",
]
CONTEXT_INDUCED_ALLOWLIST = [
    "workload.context_point",
    "workload.prompt_tokens",
    "workload.prompt_token_ids_sha256",
]

KV_ELEMENTS_PER_TOKEN = 32 * 2 * 8 * 128  # Llama-3.1-8B: layers x (K,V) x KV heads x head_dim
BF16_BYTES_PER_TOKEN = KV_ELEMENTS_PER_TOKEN * 2
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
    B: {"engine_cache_dtype": "rabit_kv2", "resolved_kv_torch_dtype": "torch.uint8",
        "kv_quant_mode": "RABIT_KV2", "fp8_storage_view_dtype": None},
}


def expected_workload(context_point: int, role: str = "measured") -> dict:
    return {"role": role, "context_point": context_point, "prompt_tokens": prompt_tokens_for(context_point),
            "output_tokens": OUTPUT_TOKENS, "warmups": WARMUPS_PER_LEG,
            "reps": 0 if role == "conditioning" else REPS_PER_LEG,
            "temperature": 0.0, "ignore_eos": True, "max_tokens": OUTPUT_TOKENS}


CAPACITY_LABEL = ("MEASURED allocator capacity (num_gpu_blocks x block_size) at engine start; a property of the "
                  "fixed engine configuration, NOT of the request context")
LIVE_KV_LABEL = ("derived_live_paged_kv_bytes is DERIVED, never directly measured live memory: "
                 "ceil((prompt_tokens + output_tokens - 1) / block_size) blocks x bytes/block, where bytes/block is "
                 "derived from the physical allocator budget (engine-logged available KV bytes / num_gpu_blocks). "
                 "It excludes RABIT-KV per-sequence state outside the paged pool (and any other memory outside the "
                 "paged KV pool), which is not observed.")
GPU_MEMORY_LABEL = ("MEASURED device-wide nvidia-smi memory.used (idle before the cell, after engine init, after "
                    "the measured reps); dominated by the gpu_memory_utilization pre-allocation, so it is not a "
                    "per-request KV measurement")
LATENCY_LABEL = (
    "Real-engine single-request latency: TTFT = frontend first_token_latency; TPOT = (last_token_ts - "
    "first_token_ts)/(n-1) from engine-core timestamps; wall = perf_counter around llm.generate. p90 = "
    "statistics.quantiles(n=10, method='inclusive')[8]. 15 measured samples per (dtype, context) cell, this "
    "session only."
)
SCOPE_NOTE = (
    "Context-length scaling of BF16 vs RABIT-KV on the real engine. 32768 is a nominal / model-limit context "
    "point (= max_model_len), NOT a 32768-token prompt: its actual prompt is 32736 tokens + 32 output tokens. It "
    "is the maximum TESTED context point, not a demonstrated maximum feasible context. Context scaling is "
    "distinct from allocator capacity. No Experiment 3/4 samples are reused. Two unmeasured conditioning cells "
    "run first and contribute no statistics. Measured and derived memory quantities are reported separately "
    "and labeled."
)
PASSED_STATES = (PASSED, FAILED, NOT_RUN, NOT_EVALUATED)
OOM_PATTERNS = ("CUDA out of memory", "OutOfMemoryError", "torch.OutOfMemoryError")
CHUNKED_MARKER = "RABIT2_STAGE3C_CHUNKED_PREFILL_ACTIVE"


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


def prior_evidence_digest() -> dict:
    """Raw-byte SHA-256 of every Experiment 3 and 4 evidence file (must never change)."""
    out = {}
    for d in EVIDENCE_DIRS:
        if d.is_dir():
            for k, v in _digest_dir(d).items():
                out[f"{d.name}/{k}"] = v
    return out


def top_level_output_files() -> list[str]:
    if not OUT_DIR.is_dir():
        return []
    return sorted(p.name for p in OUT_DIR.iterdir() if p.is_file())


def _const(tree: ast.Module, name: str):
    return ast.literal_eval(_module_assign(tree, name))


def _nested_function(fn: ast.FunctionDef, name: str) -> ast.FunctionDef:
    for node in ast.walk(fn):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"nested function {name!r} not found in {fn.name}()")


def _return_dict(fn: ast.FunctionDef) -> dict[str, str]:
    for node in fn.body:
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            return {ast.literal_eval(k): ast.dump(v) for k, v in zip(node.value.keys, node.value.values)}
    raise RuntimeError(f"{fn.name}() has no dict return")


def _assign(fn: ast.FunctionDef, target: str) -> ast.Assign:
    for node in fn.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == target:
            return node
    raise RuntimeError(f"assignment to {target!r} not found in {fn.name}()")


# ------------------------------------------------ equivalence (AST)
MODAL_SHARED_FUNCTIONS = ("_emit", "_sha256_file", "_gpu_query", "_compute_apps", "_gpu_state",
                          "_require_clean", "_run_guarded")
SHARED_PROMPT_TARGETS = ("tok", "bos", "filler", "sp")


def verify_equivalence() -> dict:
    csrc = canonical_runner_source()
    ctree = ast.parse(csrc)
    w5, w3 = ast.parse(WORKER.read_text(encoding="utf-8")), ast.parse(EXP3_WORKER.read_text(encoding="utf-8"))
    m5, m3 = ast.parse(MODAL_APP.read_text(encoding="utf-8")), ast.parse(EXP3_MODAL_APP.read_text(encoding="utf-8"))
    rtree = ast.parse(EXP3_RUNNER.read_text(encoding="utf-8"))

    canon_kwargs = canonical_llm_kwargs(csrc)
    canon_wo = {k: v for k, v in canon_kwargs.items() if k not in ("model", "kv_cache_dtype")}
    b3, b5 = _const(w3, "BASE_ENGINE_KWARGS"), _const(w5, "BASE_ENGINE_KWARGS")
    if not (canon_wo == b3 == b5):
        raise RuntimeError(f"BASE_ENGINE_KWARGS differ (canonical / exp3 / exp5):\n{canon_wo}\n{b3}\n{b5}")
    if not (b5["max_model_len"] == MAX_MODEL_LEN and b5["block_size"] == BLOCK_SIZE
            and b5["max_num_batched_tokens"] == MAX_NUM_BATCHED_TOKENS):
        raise RuntimeError("runner MAX_MODEL_LEN / BLOCK_SIZE / MAX_NUM_BATCHED_TOKENS differ from engine kwargs")
    if not (_const(w3, "OUTPUT_TOKENS") == _const(w5, "OUTPUT_TOKENS") == OUTPUT_TOKENS):
        raise RuntimeError("OUTPUT_TOKENS differs from Experiment 3")
    if tuple(_const(w5, "ALLOWED_KV_CACHE_DTYPES")) != tuple(DTYPES):
        raise RuntimeError("worker ALLOWED_KV_CACHE_DTYPES != runner DTYPES")
    for const, mine in (("WARMUPS_PER_LEG", WARMUPS_PER_LEG), ("REPS_PER_LEG", REPS_PER_LEG),
                        ("OUTPUT_TOKENS", OUTPUT_TOKENS),
                        ("GPU_CLEAN_TOLERANCE_MIB", GPU_CLEAN_TOLERANCE_MIB),
                        ("GPU_CLEAN_MAX_WAIT_S", GPU_CLEAN_MAX_WAIT_S),
                        ("GATE_TIMEOUT_S", GATE_TIMEOUT_S), ("LEG_TIMEOUT_S", LEG_TIMEOUT_S)):
        if _const(rtree, const) != mine:
            raise RuntimeError(f"{const} differs from the Experiment 3 runner")

    main3, main5 = _function(w3, "main"), _function(w5, "main")
    for target in SHARED_PROMPT_TARGETS:
        if ast.dump(_assign(main3, target)) != ast.dump(_assign(main5, target)):
            raise RuntimeError(f"statement '{target} = ...' differs from Experiment 3")
    ctx_name = "Name(id='CONTEXT_TOKENS', ctx=Load())"
    arg_attr = "Attribute(value=Name(id='args', ctx=Load()), attr='prompt_tokens', ctx=Load())"
    p3 = ast.dump(_assign(main3, "prompt"))
    if ctx_name not in p3 or p3.replace(ctx_name, arg_attr) != ast.dump(_assign(main5, "prompt")):
        raise RuntimeError("prompt construction differs from Experiment 3 (other than the prompt length)")
    one3, one5 = _nested_function(main3, "one"), _nested_function(main5, "one")
    if [ast.dump(x) for x in one3.body[:5]] != [ast.dump(x) for x in one5.body[:5]] \
            or len(one3.body) != len(one5.body) or ast.dump(one3.args) != ast.dump(one5.args):
        raise RuntimeError("timed region / metric inputs of one() differ from Experiment 3")
    r3d, r5d = _return_dict(one3), _return_dict(one5)
    if any(r5d.get(k) != v for k, v in r3d.items()):
        raise RuntimeError("an Experiment 3 sample field is computed differently")
    if set(r5d) - set(r3d) != {"prompt_token_ids_sha256", "output_token_ids_sha256"}:
        raise RuntimeError(f"unexpected extra sample fields: {sorted(set(r5d) - set(r3d))}")
    if ast.dump(_function(w3, "enum_name")) != ast.dump(_function(w5, "enum_name")):
        raise RuntimeError("enum_name() differs from Experiment 3")
    for node in ast.walk(w5):
        if isinstance(node, ast.Attribute) and node.attr == "collective_rpc":
            raise RuntimeError("worker calls collective_rpc (Experiment 3 attempt-1 hang cause)")

    if ast.dump(_module_assign(ctree, "image")) != ast.dump(_module_assign(m5, "image")):
        raise RuntimeError("Experiment 5 Modal image expression differs from the canonical runner's")
    for const in ("MODEL", "BASE_COMMIT"):
        if _const(ctree, const) != _const(m5, const):
            raise RuntimeError(f"{const} differs from the canonical runner")
    for const, mine in (("GPU_CLEAN_TOLERANCE_MIB", GPU_CLEAN_TOLERANCE_MIB),
                        ("GPU_CLEAN_MAX_WAIT_S", GPU_CLEAN_MAX_WAIT_S), ("GATE_TIMEOUT_S", GATE_TIMEOUT_S),
                        ("LEG_TIMEOUT_S", LEG_TIMEOUT_S), ("GPU_CLEAN_POLL_S", _const(m3, "GPU_CLEAN_POLL_S")),
                        ("EXPECTED_RABIT_SHA256_LF", EXPECTED_RABIT_SHA256_LF)):
        if not (_const(m5, const) == _const(m3, const) == mine):
            raise RuntimeError(f"Modal app {const} differs from Experiment 3 / runner")
    for fn in MODAL_SHARED_FUNCTIONS:
        if ast.dump(_function(m3, fn)).replace("EXP3_", "EXP5_") != ast.dump(_function(m5, fn)):
            raise RuntimeError(f"Modal helper {fn}() differs from Experiment 3")
    backstop = None
    for dec in _function(m5, "sweep").decorator_list:
        for kw in getattr(dec, "keywords", []):
            if kw.arg == "timeout":
                backstop = ast.literal_eval(kw.value)
    if not (WATCHDOG_BUDGET_S == GATE_TIMEOUT_S + len(ALL_CELLS) * LEG_TIMEOUT_S
            and backstop == MODAL_FUNCTION_TIMEOUT_S and backstop > WATCHDOG_BUDGET_S):
        raise RuntimeError(f"Modal function backstop timeout {backstop} does not exceed the total watchdog budget "
                           f"{GATE_TIMEOUT_S} + {len(ALL_CELLS)} x {LEG_TIMEOUT_S} = {WATCHDOG_BUDGET_S} s")
    for const, mine in (("WORKLOAD_FAILURE_EXIT", WORKLOAD_FAILURE_EXIT), ("OUTPUT_TOKENS", OUTPUT_TOKENS)):
        if not (_const(w5, const) == _const(m5, const) == mine):
            raise RuntimeError(f"{const} differs between worker / Modal app / runner")
    if tuple(_const(m5, "CONTINUABLE_FAILURE_KINDS")) != CONTINUABLE_FAILURE_KINDS:
        raise RuntimeError("Modal app CONTINUABLE_FAILURE_KINDS differ from the runner's")

    gtree = ast.parse(GATE.read_text(encoding="utf-8"))
    cfn, gfn = _function(ctree, "regression"), _function(gtree, "regression")
    if not (ast.dump(ast.Module(body=cfn.body, type_ignores=[])) ==
            ast.dump(ast.Module(body=gfn.body, type_ignores=[])) and ast.dump(cfn.args) == ast.dump(gfn.args)):
        raise RuntimeError("exp3_correctness_gate.regression() is not verbatim the canonical regression()")

    return {
        "canonical_source": f"{rel(CANONICAL_DEPLOYMENT)}#RUNNER_Z (zlib+base64, decoded in memory)",
        "canonical_runner_sha256": hashlib.sha256(csrc.encode("utf-8")).hexdigest(),
        "worker_base_engine_kwargs": b5,
        "engine_kwargs_equal_canonical_exp3_exp5_except_kv_cache_dtype_and_model_path": True,
        "output_tokens_and_protocol_constants_equal_exp3": True,
        "prompt_construction_equal_exp3_except_length": True,
        "timed_region_and_sample_fields_ast_equal_exp3": True,
        "extra_sample_fields": ["prompt_token_ids_sha256", "output_token_ids_sha256"],
        "worker_has_no_engine_rpc": True,
        "modal_image_expression_ast_equal_canonical": True,
        "modal_clean_state_and_watchdog_helpers_ast_equal_exp3": list(MODAL_SHARED_FUNCTIONS),
        "modal_function_backstop_timeout_s": backstop,
        "total_watchdog_budget_s": WATCHDOG_BUDGET_S,
        "workload_failure_exit_and_continuable_kinds_equal": True,
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
            "Refusing to run: Experiment 5 code has uncommitted changes, so recorded SHAs "
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
        "prior_evidence_sha256_raw": prior_evidence_digest(),
        "existing_top_level_output_files": leftovers or None,
        "uncommitted_experiment_files": uncommitted or None,
    }


# ------------------------------------------------------------ run plumbing
def build_snapshot() -> dict:
    out = Path(tempfile.mkdtemp(prefix="exp5_vllm_snapshot_")) / "vllm_kvquant_snapshot.zip"
    run_git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(out), "HEAD:vllm-kvquant")
    return {"path": str(out), "sha256": sha256_raw(out), "bytes": out.stat().st_size,
            "source": "git archive --format=zip HEAD:vllm-kvquant (core.autocrlf=false)"}


def legs_arg() -> str:
    return ",".join(f"{label}={d}:{c}:{p}:{ROLE[label]}" for _, label, d, c, p in ALL_CELLS)


def build_command() -> list[str]:
    return [sys.executable, "-m", "modal", "run", str(MODAL_APP),
            "--legs", legs_arg(), "--warmups", str(WARMUPS_PER_LEG), "--reps-per-leg", str(REPS_PER_LEG)]


def worker_command(label: str, dtype: str, context: int, prompt: int,
                   model_dir: str = "<modelscope snapshot dir>") -> list[str]:
    conditioning = ROLE[label] == "conditioning"
    return ["python", "/opt/exp5/exp5_engine_worker.py", "--kv-cache-dtype", dtype,
            "--context-point", str(context), "--prompt-tokens", str(prompt),
            "--model-dir", model_dir, "--warmups", str(WARMUPS_PER_LEG),
            "--reps", str(0 if conditioning else REPS_PER_LEG), "--leg", label] +         (["--conditioning"] if conditioning else [])


GATE_COMMAND_DOC = ["python", "/opt/exp5/exp3_correctness_gate.py"]


# ------------------------------------------------------------------ parsing
TAG = re.compile(r"^(EXP5_[A-Z_]+)=(\{.*\})\s*$")
GATE_TAG = re.compile(r"^(EXP3_GATE_[A-Z_]+)=(\{.*\})\s*$")  # unchanged Experiment 3 gate
ROWLINE = re.compile(r"^(EXP5_SAMPLE|EXP5_WARMUP) (\{.*\})\s*$")
KV_TOKENS = r3.KV_TOKENS
KV_MEM = r3.KV_MEM
MODEL_LOAD = re.compile(r"Model loading took ([\d.]+) GiB")
KV_USAGE = re.compile(r"GPU KV cache usage: ([\d.]+)%")
JIT = r3.JIT
MARKERS = ["EXP5_WARMUP_BEGIN", "EXP5_WARMUP_END", "EXP5_MEASUREMENT_BEGIN", "EXP5_MEASUREMENT_END",
           "EXP5_WORKER_COMPLETE"]


def demux(session_text: str) -> tuple[dict, list[str], list[str]]:
    legs = {k: [] for k, *_ in ALL_CELLS}
    gate, top = [], []
    prefixes = {f"[leg{k}:{d}] ": k for k, _, d, _, _ in ALL_CELLS}
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
                 "kv_log_tokens": None, "kv_log_gib": None, "model_load_gib": None, "gpu_memory": {},
                 "kv_usage_logged_pct": [], "oom_lines": [], "chunked_prefill_marker": False,
                 "workload_failures": [], "line_count": len(lines)}
    phase = None
    for i, line in enumerate(lines, start=1):
        m = TAG.match(line)
        if m:
            payload = json.loads(m.group(2))
            if m.group(1) == "EXP5_GPU_MEMORY":
                out["gpu_memory"][payload["phase"]] = payload["memory_used_mib"]
            elif m.group(1) == "EXP5_WORKLOAD_FAILURE":
                out["workload_failures"].append(payload)
            else:
                out["tags"][m.group(1)] = payload
            continue
        m = ROWLINE.match(line)
        if m:
            (out["samples"] if m.group(1) == "EXP5_SAMPLE" else out["warmups"]).append(json.loads(m.group(2)))
            continue
        s = line.strip()
        if s in MARKERS:
            out["markers"].append(s)
            phase = "measure" if s == "EXP5_MEASUREMENT_BEGIN" else (None if s.endswith("_END") else phase)
            continue
        if phase == "measure" and JIT in line:
            out["jit_during_measurement"].append({"line": i, "text": s})
        if any(p in line for p in OOM_PATTERNS):
            out["oom_lines"].append({"line": i, "text": s[:300]})
        if CHUNKED_MARKER in line:
            out["chunked_prefill_marker"] = True
        for rx, key, conv in ((KV_TOKENS, "kv_log_tokens", lambda g: int(g.replace(",", ""))),
                              (KV_MEM, "kv_log_gib", float), (MODEL_LOAD, "model_load_gib", float)):
            m = rx.search(line)
            if m:
                out[key] = conv(m.group(1))
        m = KV_USAGE.search(line)
        if m:
            out["kv_usage_logged_pct"].append({"line": i, "phase": phase or "outside_measurement",
                                               "pct": float(m.group(1))})
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
        m = r3.PYTEST_SUMMARY.match(s)
        if m:
            out["pytest_passed"] = int(m.group(1))
            out["pytest_warnings"] = int(m.group(2)) if m.group(2) else 0
            out["pytest_seconds"] = float(m.group(3)) if m.group(3) else None
        m = r3.PYTEST_EXIT.match(s)
        if m:
            out["pytest_exit"] = int(m.group(1))
        if s == "RABIT-2 FINAL TARGETED REGRESSION PASSED":
            out["regression_passed_line"] = True
    return out


def parse_top(lines: list[str]) -> dict:
    out: dict = {"pre_leg": {}, "leg_exit": {}, "leg_start": {}, "process_exit": {}, "watchdog_timeouts": [],
                 "cell_failed": {}, "verdicts": {}, "sweep_stopped": None, "sweep_complete": None}
    for line in lines:
        m = TAG.match(line.strip())
        if not m:
            continue
        tag, payload = m.group(1), json.loads(m.group(2))
        if tag == "EXP5_PRE_LEG_GPU_STATE":
            out["pre_leg"][payload["leg"]] = payload
        elif tag == "EXP5_LEG_EXIT":
            out["leg_exit"][payload["leg"]] = payload
        elif tag == "EXP5_LEG_START":
            out["leg_start"][payload["leg"]] = payload
        elif tag == "EXP5_PROCESS_EXIT":
            out["process_exit"][payload["label"]] = payload
        elif tag == "EXP5_WATCHDOG_TIMEOUT":
            out["watchdog_timeouts"].append(payload)
        elif tag == "EXP5_CELL_FAILED":
            out["cell_failed"][payload["leg"]] = payload
        elif tag == "EXP5_CELL_VERDICT":
            out["verdicts"][payload["leg"]] = payload
        elif tag == "EXP5_SWEEP_STOPPED":
            out["sweep_stopped"] = payload
        elif tag == "EXP5_SWEEP_COMPLETE":
            out["sweep_complete"] = payload
        else:
            out[tag] = payload
    return out


# ---------------------------------------------------------- config diff
CONFIG_SECTIONS = ("EXP5_REQUESTED_ENGINE_KWARGS", "EXP5_EFFECTIVE_ENGINE_CONFIG", "EXP5_WORKLOAD", "EXP5_KV_DTYPE")


def leg_config(p: dict) -> dict:
    t = p["tags"]
    return {
        **flatten("requested", t.get("EXP5_REQUESTED_ENGINE_KWARGS", {})),
        **flatten("effective", t.get("EXP5_EFFECTIVE_ENGINE_CONFIG", {})),
        **flatten("workload", t.get("EXP5_WORKLOAD", {})),
        **flatten("kv_dtype", t.get("EXP5_KV_DTYPE", {})),
    }


def _uniq(values) -> int:
    return len({json.dumps(v, sort_keys=True, default=str) for v in values})


def config_diff(parsed: dict) -> dict:
    """All 12 cells. Non-allowlisted fields identical everywhere; dtype-induced
    fields constant within a dtype; context-induced fields constant within a
    context. Not evaluated unless every cell reported every config section."""
    meta = {label: {"dtype": d, "context_point": c} for _, label, d, c, _ in LEGS}
    missing = {label: [s for s in CONFIG_SECTIONS if s not in parsed[k]["tags"]] for k, label, *_ in LEGS}
    missing = {label: v for label, v in missing.items() if v}
    base = {
        "rule": ("All non-allowlisted engine/workload fields must be identical across all 12 cells. "
                 "DTYPE_INDUCED_ALLOWLIST fields may differ only between dtypes and must be identical across "
                 "every context of one dtype. CONTEXT_INDUCED_ALLOWLIST fields may differ only between context "
                 "points and must be identical for both dtypes at one context point. Capacity/latency/memory "
                 "are outcomes, compared separately. Not evaluated unless all 12 cells reported their config."),
        "cells": meta,
        "dtype_induced_allowlist": DTYPE_INDUCED_ALLOWLIST,
        "context_induced_allowlist": CONTEXT_INDUCED_ALLOWLIST,
    }
    configs = {label: leg_config(parsed[k]) for k, label, *_ in LEGS}
    if missing:
        return {**base, "status": NOT_EVALUATED, "matched": None, "missing_config_sections_by_cell": missing,
                "configs": configs}
    keys = sorted(set().union(*configs.values()))
    violations, dtype_fields, context_fields = [], [], []
    for key in keys:
        vals = {label: cfg.get(key, "<missing>") for label, cfg in configs.items()}
        if _uniq(vals.values()) == 1:
            continue
        if key in DTYPE_INDUCED_ALLOWLIST:
            bad = [d for d in DTYPES if _uniq(v for l, v in vals.items() if meta[l]["dtype"] == d) != 1]
            if bad:
                violations.append({"field": key, "reason": f"dtype-induced field varies within dtype {bad}",
                                   "values": vals})
            else:
                dtype_fields.append(key)
        elif key in CONTEXT_INDUCED_ALLOWLIST:
            bad = [c for c in CONTEXT_GRID if _uniq(v for l, v in vals.items() if meta[l]["context_point"] == c) != 1]
            if bad:
                violations.append({"field": key, "reason": f"context-induced field differs between dtypes at "
                                                           f"context {bad}", "values": vals})
            else:
                context_fields.append(key)
        else:
            violations.append({"field": key, "reason": "non-allowlisted field differs between cells", "values": vals})
    return {**base, "status": PASSED if not violations else FAILED, "matched": not violations,
            "configs": configs, "fields_compared": len(keys),
            "fields_differing_between_dtypes": dtype_fields,
            "fields_differing_between_contexts": context_fields, "violations": violations}


# ---------------------------------------------------------- checks / stats
def gpu_leg_clean(pre: dict, baseline: dict) -> bool:
    if not pre or not pre.get("readings") or not baseline:
        return False
    last = pre["readings"][-1]
    return (not last["compute_apps"]
            and pre.get("tolerance_mib") == GPU_CLEAN_TOLERANCE_MIB
            and all(u <= b + GPU_CLEAN_TOLERANCE_MIB
                    for u, b in zip(last["memory_used_mib"], baseline["memory_used_mib"])))


def derived_live_kv(prompt: int, cap: dict, kv_gib: float | None) -> dict:
    resident = prompt + OUTPUT_TOKENS - 1  # the last sampled token is never written to the KV cache
    blocks = math.ceil(resident / cap["block_size"])
    bpb = kv_gib * 2**30 / cap["num_gpu_blocks"] if kv_gib else None
    return {
        "kind": "DERIVED",
        "is_directly_measured": False,
        "formula": "ceil((prompt_tokens + output_tokens - 1) / block_size) x derived_bytes_per_block",
        "derived_bytes_per_block_source": ("physical allocator budget: engine-logged available KV bytes / "
                                           "num_gpu_blocks"),
        "excludes": "RABIT-KV per-sequence state outside the paged pool (and any memory outside the paged KV pool)",
        "prompt_tokens": prompt,
        "output_tokens": OUTPUT_TOKENS,
        "block_size": cap["block_size"],
        "resident_tokens_at_peak": resident,
        "derived_live_blocks": blocks,
        "derived_live_fraction_of_allocator_blocks": blocks / cap["num_gpu_blocks"],
        "derived_bytes_per_block": bpb,
        "derived_live_paged_kv_bytes": blocks * bpb if bpb else None,
        "derived_live_paged_kv_mib": blocks * bpb / 2**20 if bpb else None,
        "derived_bytes_per_block_uncertainty_note": "engine logs available KV memory rounded to 0.01 GiB",
    }


def feasibility(p: dict, started: bool, exit_info: dict | None, prompt: int, verdict: dict | None = None) -> dict:
    return {
        "cell_started": started,
        "in_container_verdict": (verdict or {}).get("verdict"),
        "workload_failure": p["workload_failures"][0] if p["workload_failures"] else None,
        "engine_initialized": "EXP5_CAPACITY" in p["tags"],
        "oom_detected": bool(p["oom_lines"]),
        "oom_lines": p["oom_lines"][:5],
        "worker_returncode": (exit_info or {}).get("returncode"),
        "all_requests_completed": "EXP5_MEASUREMENT_END" in p["markers"] and len(p["samples"]) == REPS_PER_LEG,
        "prompt_tokens_processed": sorted({r["prompt_tokens"] for r in p["samples"] + p["warmups"]}),
        "prompt_tokens_intended": prompt,
        "output_produced": bool(p["samples"]) and all(r["output_tokens"] == OUTPUT_TOKENS for r in p["samples"]),
    }


CONFIG_SECTIONS_NO_WORKLOAD = ("EXP5_REQUESTED_ENGINE_KWARGS", "EXP5_EFFECTIVE_ENGINE_CONFIG", "EXP5_KV_DTYPE")


def conditioning_checks(parsed: dict, top: dict, baseline: dict, add) -> None:
    """Conditioning cells: unmeasured, exact 512-token workload, 5 warmups, zero
    reps, same engine config / KV dtype / capacity / prompt as the official 512
    cell of the same dtype, GPU clean before and after. Their latency values are
    never used."""
    official = {(d, c): k for k, _, d, c, _ in LEGS}
    for k, label, d, c, prompt in CONDITIONING:
        p = parsed[k]
        t = p["tags"]
        started = label in top["leg_start"]

        def chk(name: str, ok, observed=None, _started=started, _label=label) -> None:
            add(f"{_label}: {name}", "conditioning", ok if _started else NOT_RUN, observed if _started else None)

        pre = top["pre_leg"].get(label)
        add(f"{label}: GPU clean before conditioning cell", "gpu_clean",
            gpu_leg_clean(pre, baseline) if pre else NOT_RUN,
            pre["readings"][-1] if pre and pre.get("readings") else None)
        chk("worker exit 0", (top["leg_exit"].get(label) or {}).get("returncode") == 0, top["leg_exit"].get(label))
        chk("in-container verdict ok", (top["verdicts"].get(label) or {}).get("verdict") == "ok",
            top["verdicts"].get(label))
        chk("worker reports conditioning cell identity",
            t.get("EXP5_LEG") == {"leg": label, "kv_cache_dtype": d, "role": "conditioning", "context_point": c,
                                  "prompt_tokens": prompt}, t.get("EXP5_LEG"))
        wl = t.get("EXP5_WORKLOAD", {})
        chk("workload = exact 512-token prompt, 32 output, 5 warmups, ZERO measured reps",
            all(wl.get(x) == v for x, v in expected_workload(c, "conditioning").items()), wl or None)
        chk(f"exactly {WARMUPS_PER_LEG} warmups and no measured sample",
            len(p["warmups"]) == WARMUPS_PER_LEG and p["samples"] == [],
            {"warmups": len(p["warmups"]), "samples": len(p["samples"])})
        chk("every warmup: exact prompt/output token counts and prompt hash",
            bool(p["warmups"]) and all(r["prompt_tokens"] == prompt and r["output_tokens"] == OUTPUT_TOKENS
                                       and r.get("prompt_token_ids_sha256") == wl.get("prompt_token_ids_sha256")
                                       for r in p["warmups"]))
        chk("marker order", p["markers"] == MARKERS, p["markers"])
        ref = parsed[official[(d, c)]]["tags"]
        if not started:
            state = NOT_RUN
        elif not all(s in ref for s in (*CONFIG_SECTIONS_NO_WORKLOAD, "EXP5_CAPACITY", "EXP5_WORKLOAD")):
            state = NOT_EVALUATED
        else:
            state = (all(t.get(s) == ref[s] for s in CONFIG_SECTIONS_NO_WORKLOAD)
                     and t.get("EXP5_CAPACITY") == ref["EXP5_CAPACITY"]
                     and wl.get("prompt_token_ids_sha256") == ref["EXP5_WORKLOAD"].get("prompt_token_ids_sha256"))
        add(f"{label}: engine config, KV dtype, capacity and prompt identical to the official cell of this dtype "
            f"at {c}", "conditioning", state)
    # GPU clean state after the last conditioning cell = the pre-state of the first official cell.
    first = LEGS[0][1]
    pre = top["pre_leg"].get(first)
    add(f"GPU clean after conditioning (pre-state of {first})", "conditioning",
        gpu_leg_clean(pre, baseline) if pre else NOT_RUN, pre["readings"][-1] if pre and pre.get("readings") else None)


def continuation_policy_checks(parsed: dict, top: dict, baseline: dict, add) -> None:
    """Every cell the sweep continued past must be a workload-level failure of a
    measured cell, after full reaping and a fresh clean-state check."""
    cells = {label: k for k, label, *_ in ALL_CELLS}
    for label in sorted(top["cell_failed"]):
        k = cells.get(label)
        p = parsed[k] if k else {"workload_failures": [], "tags": {}}
        meta = top["process_exit"].get(label) or {}
        wf = p["workload_failures"]
        post = top["pre_leg"].get(f"{label}:post_failure")
        verdict = top["verdicts"].get(label) or {}
        rc = (top["leg_exit"].get(label) or {}).get("returncode")
        ok = (ROLE.get(label) == "measured" and rc == WORKLOAD_FAILURE_EXIT
              and len(wf) == 1 and wf[0].get("kind") in CONTINUABLE_FAILURE_KINDS
              and "EXP5_CAPACITY" in p["tags"]
              and meta.get("timed_out") is False and meta.get("group_processes_after_leader_exit") == []
              and meta.get("group_processes_remaining") == []
              and verdict.get("verdict") == "workload_failure"
              and gpu_leg_clean(post, baseline))
        add(f"{label}: continuation after failure respected policy (workload-level failure of a measured cell, "
            f"engine initialized, group reaped, no orphan, GPU clean re-check passed)", "stop", ok,
            {"returncode": rc, "workload_failure": wf[:1],
             "group_after_exit": meta.get("group_processes_after_leader_exit"),
             "post_failure_clean": gpu_leg_clean(post, baseline), "verdict": verdict.get("verdict")})


def integrity(parsed: dict, gate: dict, top: dict, diff: dict) -> dict:
    checks: list[dict] = []

    def add(name: str, category: str, state, observed=None) -> None:
        if isinstance(state, bool):
            state = PASSED if state else FAILED
        checks.append({"check": name, "category": category, "state": state, "observed": observed})

    env = top.get("EXP5_ENVIRONMENT", {})
    gpus = env.get("gpus", [])
    baseline = top.get("EXP5_GPU_BASELINE", {})
    sc = top.get("sweep_complete")
    stopped = top.get("sweep_stopped")
    add("sweep was not stopped by the in-container cell verifier", "stop", not stopped, stopped)
    add(f"all {len(ALL_CELLS)} cell processes (2 conditioning + 12 measured) ran to the end of the sweep",
        "completion", bool(sc) and sc.get("cells") == len(ALL_CELLS), sc)
    add("no failed cell", "completion", (not (sc or {}).get("failed_cells") and not top["cell_failed"])
        if sc else NOT_EVALUATED, (sc or {}).get("failed_cells"))
    add("no watchdog timeout", "watchdog", not top["watchdog_timeouts"], top["watchdog_timeouts"] or None)
    add("exactly one GPU, H100", "environment",
        (len(gpus) == 1 and "H100" in gpus[0].get("name", "")) if env else NOT_EVALUATED,
        [g.get("name") for g in gpus] or None)
    add("cell order as planned (conditioning A512, B512; then ascending context, alternating dtype order)",
        "environment",
        (env.get("leg_labels") == [l for _, l, *_ in ALL_CELLS]
         and env.get("leg_roles") == [ROLE[l] for _, l, *_ in ALL_CELLS]
         and env.get("leg_dtypes") == [d for _, _, d, _, _ in ALL_CELLS]
         and env.get("leg_context_points") == [c for *_, c, _ in ALL_CELLS]
         and env.get("leg_prompt_tokens") == [p for *_, p in ALL_CELLS]) if env else NOT_EVALUATED,
        {k: env.get(k) for k in ("leg_labels", "leg_roles", "leg_dtypes", "leg_context_points", "leg_prompt_tokens")}
        if env else None)
    add("rabit_kv2.py in image is frozen source", "environment",
        env.get("rabit_kv2_sha256_lf") == EXPECTED_RABIT_SHA256_LF if env else NOT_EVALUATED,
        env.get("rabit_kv2_sha256_lf"))
    add("idle baseline recorded with no compute process", "gpu_clean",
        bool(baseline) and not baseline.get("compute_apps") and "memory_used_mib" in baseline,
        baseline.get("memory_used_mib"))

    gate_ran = "EXP5_GATE_START" in top
    gstate = (lambda ok: ok) if gate_ran else (lambda ok: NOT_RUN)
    add("gate: process exit 0", "gate", gstate(top.get("EXP5_GATE_EXIT", {}).get("returncode") == 0),
        top.get("EXP5_GATE_EXIT"))
    add("gate: result passed", "gate", gstate((gate.get("result") or {}).get("passed") is True), gate.get("result"))
    add("gate: pytest exit 0", "gate", gstate(gate.get("pytest_exit") == 0), gate.get("pytest_exit"))
    add("gate: pytest passed count parsed", "gate", gstate(isinstance(gate.get("pytest_passed"), int)),
        gate.get("pytest_passed"))
    add("gate: canonical 'REGRESSION PASSED' line", "gate", gstate(gate.get("regression_passed_line")))
    add("gate: ran against frozen rabit_kv2.py", "gate",
        gstate((gate.get("begin") or {}).get("rabit_kv2_sha256_lf") == EXPECTED_RABIT_SHA256_LF))
    if not top["leg_start"]:
        add("gate completed before first cell", "gate", NOT_EVALUATED if gate_ran else NOT_RUN, "no cell started")
    else:
        add("gate completed before first cell", "gate", "EXP5_GATE_EXIT" in top)
    model = top.get("EXP5_MODEL", {})
    add("model snapshot hashed", "model", bool(model.get("files")) if model else NOT_RUN)

    feas = {}
    for k, label, d, c, prompt in LEGS:
        p = parsed[k]
        t = p["tags"]
        started = label in top["leg_start"]
        feas[label] = feasibility(p, started, top["leg_exit"].get(label), prompt, top["verdicts"].get(label))

        def leg(name: str, category: str, ok, observed=None, _started=started) -> None:
            add(f"{label}: {name}", category, ok if _started else NOT_RUN, observed if _started else None)

        def measured(name: str, category: str, ok, observed=None, _p=p, _started=started) -> None:
            if not _started:
                add(f"{label}: {name}", category, NOT_RUN)
            elif "EXP5_MEASUREMENT_BEGIN" not in _p["markers"]:
                add(f"{label}: {name}", category, NOT_EVALUATED, "measurement phase never started")
            else:
                add(f"{label}: {name}", category, ok, observed)

        pre = top["pre_leg"].get(label)
        add(f"{label}: GPU clean before cell (no compute process, within {GPU_CLEAN_TOLERANCE_MIB} MiB of baseline)",
            "gpu_clean", gpu_leg_clean(pre, baseline) if pre else NOT_RUN,
            pre["readings"][-1] if pre and pre.get("readings") else None)
        leg("worker exit 0", "leg", (top["leg_exit"].get(label) or {}).get("returncode") == 0,
            top["leg_exit"].get(label))
        leg("no OOM", "feasibility", not p["oom_lines"], p["oom_lines"][:3] or None)
        leg("worker reports cell/dtype/context/prompt", "leg",
            t.get("EXP5_LEG") == {"leg": label, "kv_cache_dtype": d, "role": "measured", "context_point": c,
                                  "prompt_tokens": prompt},
            t.get("EXP5_LEG"))
        leg("RABIT frozen-source markers", "leg",
            bool(t.get("EXP5_RABIT_MARKERS")) and all(t["EXP5_RABIT_MARKERS"].values()))
        req = t.get("EXP5_REQUESTED_ENGINE_KWARGS", {})
        leg("requested kwargs as planned", "config", req == requested_kwargs(d, req.get("model", "<missing>")))
        leg("model path is the hashed snapshot", "model",
            bool(model) and req.get("model") == model.get("snapshot_dir"), req.get("model"))
        eff = t.get("EXP5_EFFECTIVE_ENGINE_CONFIG", {})
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
        wl = t.get("EXP5_WORKLOAD", {})
        leg("workload as planned (context point, exact prompt length, 32 output)", "workload",
            all(wl.get(x) == v for x, v in expected_workload(c).items()) and bool(wl.get("prompt_token_ids_sha256")),
            wl or None)
        kv = t.get("EXP5_KV_DTYPE", {})
        leg("resolved KV dtype", "kv_dtype",
            bool(kv) and kv.get("requested_kv_cache_dtype") == d
            and all(kv.get(x, "<missing>") == v for x, v in EXPECTED_KV[d].items()), kv or None)
        cap = t.get("EXP5_CAPACITY", {})
        leg("capacity = num_gpu_blocks x block_size", "capacity",
            bool(cap) and cap["capacity_tokens"] == cap["num_gpu_blocks"] * cap["block_size"], cap or None)
        leg("capacity matches engine log 'GPU KV cache size'", "capacity",
            bool(cap) and p["kv_log_tokens"] == cap.get("capacity_tokens"), p["kv_log_tokens"])
        leg("available KV cache memory and model-load memory logged", "memory",
            bool(p["kv_log_gib"]) and bool(p["model_load_gib"]),
            {"available_kv_gib": p["kv_log_gib"], "model_load_gib": p["model_load_gib"]})
        leg("GPU memory recorded after engine init and after measurement", "memory",
            set(p["gpu_memory"]) == {"after_engine_init", "after_measurement"}, p["gpu_memory"] or None)
        if d == A:
            bpt = p["kv_log_gib"] * 2**30 / cap["capacity_tokens"] if cap and p["kv_log_gib"] else None
            leg(f"physical bytes/token consistent with 2-byte BF16 KV ({BF16_BYTES_PER_TOKEN} B/token)", "kv_dtype",
                bpt is not None and abs(bpt / BF16_BYTES_PER_TOKEN - 1) <= BYTES_PER_TOKEN_REL_TOL,
                round(bpt, 1) if bpt else None)
        rows = p["samples"] + p["warmups"]
        measured(f"warmups == {WARMUPS_PER_LEG}", "measurement", len(p["warmups"]) == WARMUPS_PER_LEG,
                 len(p["warmups"]))
        measured(f"measured reps == {REPS_PER_LEG}", "measurement",
                 [r["rep"] for r in p["samples"]] == list(range(REPS_PER_LEG)), len(p["samples"]))
        measured(f"every request: engine saw exactly {prompt} prompt tokens", "workload",
                 bool(p["samples"]) and all(r["prompt_tokens"] == prompt for r in rows),
                 sorted({r["prompt_tokens"] for r in rows}))
        measured(f"every request: exactly {OUTPUT_TOKENS} output tokens", "workload",
                 bool(p["samples"]) and all(r["output_tokens"] == OUTPUT_TOKENS for r in rows),
                 sorted({r["output_tokens"] for r in rows}))
        measured("every request: engine prompt hash == planned prompt hash", "workload",
                 bool(p["samples"]) and all(r.get("prompt_token_ids_sha256") == wl.get("prompt_token_ids_sha256")
                                            for r in rows))
        measured("every request recorded generated-token hash", "measurement",
                 bool(p["samples"]) and all(isinstance(r.get("output_token_ids_sha256"), str)
                                            and len(r["output_token_ids_sha256"]) == 64 for r in rows))
        leg("marker order", "leg", p["markers"] == MARKERS, p["markers"])
        measured("no Triton JIT compilation during measurement", "measurement",
                 "EXP5_MEASUREMENT_END" in p["markers"] and not p["jit_during_measurement"],
                 p["jit_during_measurement"] or None)

    add("config: non-allowlisted fields identical across all 12 cells; allowlisted fields vary only by their class",
        "config", diff["status"], diff.get("violations") or diff.get("missing_config_sections_by_cell"))
    for c in CONTEXT_GRID:
        hashes = {label: parsed[k]["tags"].get("EXP5_WORKLOAD", {}).get("prompt_token_ids_sha256")
                  for k, label, d, cc, _ in LEGS if cc == c}
        add(f"context {c}: prompt token hash identical for both dtypes", "workload",
            (_uniq(hashes.values()) == 1) if all(hashes.values()) else NOT_EVALUATED, hashes)
    for d in DTYPES:
        caps = {label: parsed[k]["tags"].get("EXP5_CAPACITY") for k, label, dd, _, _ in LEGS if dd == d}
        add(f"{d}: allocator capacity identical across all six contexts (context-independent)", "capacity",
            (_uniq(caps.values()) == 1) if all(caps.values()) else NOT_EVALUATED, caps)

    conditioning_checks(parsed, top, baseline, add)
    continuation_policy_checks(parsed, top, baseline, add)

    counts = {s: sum(1 for ch in checks if ch["state"] == s) for s in PASSED_STATES}
    return {"state_semantics": "passed / failed / not_run (cell never started) / not_evaluated "
                               "(required data or cells missing); only 'passed' counts as passed",
            "checks": checks, "counts": counts, "all_ok": counts[PASSED] == len(checks),
            "failed_categories": sorted({ch["category"] for ch in checks if ch["state"] == FAILED}),
            "non_passed_categories": sorted({ch["category"] for ch in checks if ch["state"] != PASSED}),
            "feasibility": feas}


def p90(values: list[float]) -> float:
    return statistics.quantiles(values, n=10, method="inclusive")[8]


def direction(delta: float) -> str:
    return "slower" if delta > 0 else ("faster" if delta < 0 else "equal")


def build_summary(parsed: dict, gate: dict, top: dict) -> dict:
    cells = {}
    for k, label, d, c, prompt in LEGS:
        p = parsed[k]
        cap = p["tags"]["EXP5_CAPACITY"]
        cells[label] = {
            "index": k, "kv_cache_dtype": d, "context_point": c, "actual_prompt_tokens": prompt,
            "output_tokens": OUTPUT_TOKENS, "display_label": display_label(c, prompt), "log": LOG_NAME[label],
            "kv_dtype": p["tags"]["EXP5_KV_DTYPE"],
            "latency": {m: stats(p["samples"], m) for m in ("tpot_ms", "ttft_ms", "wall_ms")},
            "measured_samples": p["samples"],
            "warmup_samples_excluded": p["warmups"],
            "memory_measured": {
                "allocator_capacity": {**cap, "label": CAPACITY_LABEL},
                "available_kv_cache_memory_gib_logged": p["kv_log_gib"],
                "model_load_gib_logged": p["model_load_gib"],
                "gpu_memory_used_mib_idle_before_cell": (top["pre_leg"].get(label) or {}).get("readings", [{}])[-1]
                .get("memory_used_mib"),
                "gpu_memory_used_mib_after_engine_init": p["gpu_memory"].get("after_engine_init"),
                "gpu_memory_used_mib_after_measurement": p["gpu_memory"].get("after_measurement"),
                "engine_logged_kv_usage_pct_snapshots": p["kv_usage_logged_pct"],
                "label": GPU_MEMORY_LABEL,
            },
            "memory_derived": {**derived_live_kv(prompt, cap, p["kv_log_gib"]), "label": LIVE_KV_LABEL},
            "rabit_chunked_prefill_marker_logged": p["chunked_prefill_marker"],
            "prompt_exceeds_max_num_batched_tokens": prompt > MAX_NUM_BATCHED_TOKENS,
            "pre_cell_gpu_state": top["pre_leg"].get(label),
            "process": top["process_exit"].get(label),
        }
    per_context = {}
    for c in CONTEXT_GRID:
        la, lb = f"A{c}", f"B{c}"
        a, b = cells[la], cells[lb]
        first = la if a["index"] < b["index"] else lb
        entry = {"context_point": c, "actual_prompt_tokens": prompt_tokens_for(c), "output_tokens": OUTPUT_TOKENS,
                 "display_label": display_label(c, prompt_tokens_for(c)), "cells": {A: la, B: lb},
                 "first_cell": first, "latency_signed_deltas_rabit_minus_bf16": {}}
        for key, m, st in (("tpot_median", "tpot_ms", "median"), ("tpot_p90", "tpot_ms", "p90"),
                           ("ttft_median", "ttft_ms", "median"), ("wall_median", "wall_ms", "median")):
            x, y = a["latency"][m][st], b["latency"][m][st]
            entry["latency_signed_deltas_rabit_minus_bf16"][key] = {
                "bf16": x, "rabit_kv2": y, "signed_delta_ms": y - x, "signed_delta_pct": (y / x - 1) * 100,
                "direction": f"rabit_kv2 {direction(y - x)} than bf16"}
        da, db = a["memory_derived"], b["memory_derived"]
        entry["derived_live_paged_kv"] = {
            "kind": "DERIVED",
            "is_directly_measured": False,
            "bf16_derived_live_paged_kv_bytes": da["derived_live_paged_kv_bytes"],
            "rabit_kv2_derived_live_paged_kv_bytes": db["derived_live_paged_kv_bytes"],
            "derived_ratio_bf16_over_rabit_kv2": (da["derived_live_paged_kv_bytes"] / db["derived_live_paged_kv_bytes"]
                                                  if da["derived_live_paged_kv_bytes"]
                                                  and db["derived_live_paged_kv_bytes"] else None),
            "note": ("DERIVED live paged-KV ratio at matched context (excludes RABIT-KV per-sequence state outside "
                     "the paged pool); distinct from the allocator-capacity ratio."),
        }
        per_context[str(c)] = entry
    capacity = {d: cells[f"{LETTER[d]}{CONTEXT_GRID[0]}"]["memory_measured"]["allocator_capacity"] for d in DTYPES}
    return {
        "experiment": "MLSys 2027 Experiment 5 -- BF16 vs RABIT-KV context-length scaling (real engine)",
        "scope": SCOPE_NOTE,
        "experiment3_or_4_samples_used": False,
        "maximum_tested_context_point": max(CONTEXT_GRID),
        "maximum_feasible_context_demonstrated": False,
        "labels": {"capacity": CAPACITY_LABEL, "live_kv": LIVE_KV_LABEL, "gpu_memory": GPU_MEMORY_LABEL,
                   "latency": LATENCY_LABEL},
        "context_point_semantics": ("context_point is the nominal grid point; actual_prompt_tokens is the prompt "
                                    "the engine processed. 32768 is a nominal / model-limit point, NOT a "
                                    "32768-token prompt (32736 input + 32 output). Plots/tables must use "
                                    "actual_prompt_tokens or the display_label."),
        "conditioning": {
            "cells": {label: {"index": k, "kv_cache_dtype": d, "context_point": c, "actual_prompt_tokens": p,
                              "warmups": len(parsed[k]["warmups"]), "measured_reps": len(parsed[k]["samples"]),
                              "returncode": (top["leg_exit"].get(label) or {}).get("returncode"),
                              "in_container_verdict": (top["verdicts"].get(label) or {}).get("verdict"),
                              "pre_cell_gpu_state": top["pre_leg"].get(label),
                              "process": top["process_exit"].get(label), "log": LOG_NAME[label]}
                      for k, label, d, c, p in CONDITIONING},
            "note": ("UNMEASURED conditioning cells run after the gate and before the official sweep to absorb the "
                     "first-engine/container effect; their latency values are not part of any Experiment 5 "
                     "statistic."),
        },
        "design": {"cells": [{"index": k, "cell": label, "kv_cache_dtype": d, "context_point": c,
                              "actual_prompt_tokens": p, "role": ROLE[label]} for k, label, d, c, p in ALL_CELLS],
                   "context_grid": CONTEXT_GRID, "output_tokens": OUTPUT_TOKENS, "max_model_len": MAX_MODEL_LEN,
                   "prompt_rule": "prompt = context point unless prompt + output > max_model_len, then "
                                  "max_model_len - output",
                   "order": ("conditioning A512, B512 (unmeasured); then ascending context, dtype order "
                             "alternating per context (A B | B A | ...)"),
                   "warmups_per_cell_excluded": WARMUPS_PER_LEG, "measured_reps_per_cell": REPS_PER_LEG,
                   "fresh_process_per_cell": True, "gate_timeout_s": GATE_TIMEOUT_S,
                   "cell_timeout_s": LEG_TIMEOUT_S},
        "correctness_gate": {
            "scope": "frozen RABIT-KV correctness gate, run once before any cell",
            "command": (gate.get("begin") or {}).get("pytest_command"),
            "passed": (gate.get("result") or {}).get("passed"),
            "pytest_exit": gate.get("pytest_exit"), "pytest_passed": gate.get("pytest_passed"),
            "pytest_warnings": gate.get("pytest_warnings"),
            "rabit_kv2_sha256_lf": (gate.get("begin") or {}).get("rabit_kv2_sha256_lf"),
            "process": top["process_exit"].get("gate"),
        },
        "allocator_capacity_context_independent": capacity,
        "cells": cells,
        "per_context": per_context,
        "gpu_clean_state": {"baseline": top.get("EXP5_GPU_BASELINE"), "tolerance_mib": GPU_CLEAN_TOLERANCE_MIB,
                            "pre_cell": top["pre_leg"], "post_run": top.get("EXP5_POST_RUN_GPU_STATE")},
        "environment": top.get("EXP5_ENVIRONMENT"),
        "model": top.get("EXP5_MODEL"),
    }


def analyze(session_text: str, write: bool) -> tuple[dict, dict, dict | None]:
    leg_lines, gate_lines, top_lines = demux(session_text)
    parsed = {k: parse_worker(leg_lines[k]) for k, *_ in ALL_CELLS}
    gate = parse_gate(gate_lines)
    top = parse_top(top_lines)
    diff = config_diff(parsed)
    integ = integrity(parsed, gate, top, diff)
    integ["correctness_gate"] = gate
    integ["gpu_clean_state"] = {"baseline": top.get("EXP5_GPU_BASELINE"), "tolerance_mib": GPU_CLEAN_TOLERANCE_MIB,
                                "max_wait_s": GPU_CLEAN_MAX_WAIT_S, "pre_leg": top["pre_leg"]}
    integ["processes"] = {"exits": top["process_exit"], "watchdog_timeouts": top["watchdog_timeouts"],
                          "failed_cells": top["cell_failed"], "verdicts": top["verdicts"],
                          "sweep_stopped": top["sweep_stopped"]}
    summary = build_summary(parsed, gate, top) if integ["all_ok"] else None
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        GATE_LOG.write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
        for k, label, d, c, p in ALL_CELLS:
            header = (f"===== EXP5 CELL {label} (index {k}, {d}, role {ROLE[label]}, context {c}, "
                      f"prompt {p}) =====")
            (OUT_DIR / LOG_NAME[label]).write_text("\n".join([header, *leg_lines[k]]) + "\n", encoding="utf-8")
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
    manifest["prior_evidence_unchanged"] = prior_evidence_digest() == prov.get("prior_evidence_sha256_raw", {})
    if not manifest["prior_evidence_unchanged"]:
        manifest["status"] = "failed"
        manifest.setdefault("failure", {"stage": "prior_evidence_modified",
                                        "reason": "an Experiment 3/4 evidence file changed during the run"})
    write_manifest(manifest)


def classify_failure(code: int, integ: dict) -> dict:
    cats = integ["failed_categories"]
    info = {"modal_returncode": code, "failed_categories": cats,
            "non_passed_categories": integ["non_passed_categories"],
            "failed_cells": sorted(l for l, f in integ["feasibility"].items()
                                   if f["cell_started"] and not f["all_requests_completed"])}
    for stage, cat in (("watchdog_timeout", "watchdog"), ("correctness_gate", "gate"), ("gpu_clean", "gpu_clean"),
                       ("conditioning", "conditioning")):
        if cat in cats:
            return {"stage": stage, **info}
    stopped = next((c for c in integ["checks"] if c["check"].startswith("sweep was not stopped")
                    and c["state"] == FAILED), None)
    if stopped:
        return {"stage": "sweep_stopped_by_cell_verifier", "stop": stopped["observed"], **info}
    if "stop" in cats:
        return {"stage": "continuation_policy_violation", **info}
    if code != 0:
        return {"stage": "modal_nonzero_exit", **info}
    if "feasibility" in cats or "completion" in cats:
        return {"stage": "cell_failure", **info}
    if "config" in cats:
        return {"stage": "config_diff", **info}
    return {"stage": "integrity", **info}


def run(manifest: dict) -> int:
    snapshot = build_snapshot()
    manifest["vllm_kvquant_snapshot"] = snapshot
    command = build_command()
    row = {"name": "context_sweep", "command": command, "log": rel(SESSION_LOG),
           "gate_command": GATE_COMMAND_DOC,
           "worker_commands": {label: worker_command(label, d, c, p) for _, label, d, c, p in ALL_CELLS},
           "started_utc": now(), "completed_utc": None, "returncode": None, "status": "running"}
    manifest["runs"].append(row)
    write_manifest(manifest)

    code = stream_command(command, SESSION_LOG, {"EXP5_VLLM_SNAPSHOT": snapshot["path"]})
    row.update(returncode=code, completed_utc=now())

    manifest["stage"] = "parse"
    diff, integ, summary = analyze(SESSION_LOG.read_text(encoding="utf-8", errors="replace"), write=True)
    row["integrity_counts"] = integ["counts"]
    manifest["gpu_clean_state"] = integ["gpu_clean_state"]
    manifest["correctness_gate"] = integ["correctness_gate"]
    manifest["processes"] = integ["processes"]
    manifest["feasibility"] = integ["feasibility"]
    manifest["integrity_counts"] = integ["counts"]
    manifest["config_diff_status"] = diff["status"]
    manifest.pop("stage", None)

    if code != 0 or not integ["all_ok"]:
        failure = classify_failure(code, integ)
        row["status"] = failure["stage"]
        finalize(manifest, "failed", failure)
        raise SystemExit(
            f"\nEXPERIMENT 5 STOPPED ({failure['stage']}): modal exit={code}; integrity {integ['counts']}; "
            f"failed cells {failure['failed_cells']}; non-passed categories {integ['non_passed_categories']}. "
            f"Logs and {rel(INTEGRITY)} preserved; nothing retried."
        )

    row["status"] = "passed"
    finalize(manifest, "passed", None)
    if manifest["status"] != "passed":
        raise SystemExit(f"\nEXPERIMENT 5 FAILED at post-run checks: {manifest.get('failure')}")
    print("\n" + "=" * 118 + "\nEXPERIMENT 5: CONTEXT SWEEP PASSED\n" f"Summary: {SUMMARY}\n" + "=" * 118)
    return 0


FAILED_POLICY = 'continue past a failed measured cell ONLY for a workload-level failure (request OOM / request execution failure after successful engine init) after full process-group reaping, no orphan, and a fresh GPU clean-state check; stop immediately for everything else; the run is still marked failed and no summary is written'


def main(argv: list[str] | None = None) -> int:
    make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Preflight + equivalence proof + planned commands and config diff. "
                         "No Modal/GPU, no files written.")
    args = ap.parse_args(argv)

    print("RABIT-KV MLSys 2027 -- Experiment 5: BF16 vs RABIT-KV context-length scaling (real engine)")
    print(f"Conditioning cells (UNMEASURED, {WARMUPS_PER_LEG} warmups, 0 reps): "
          f"{[f'{label}={d}@{c}(prompt {p})' for _, label, d, c, p in CONDITIONING]}")
    print(f"Official measured cells (one container, one GPU, fresh process per cell): "
          f"{[f'{label}={d}@{c}(prompt {p})' for _, label, d, c, p in LEGS]}")
    print(f"Per official cell: {WARMUPS_PER_LEG} full-shape warmups (excluded) + {REPS_PER_LEG} measured reps; "
          f"{OUTPUT_TOKENS} output tokens; max_model_len {MAX_MODEL_LEN}")
    print(f"Watchdogs: gate {GATE_TIMEOUT_S}s, each of {len(ALL_CELLS)} cell processes {LEG_TIMEOUT_S}s "
          f"(own process group, group kill); total watchdog budget {WATCHDOG_BUDGET_S}s; Modal backstop "
          f"{MODAL_FUNCTION_TIMEOUT_S}s")
    print("Scope: 32768 is a nominal / model-limit context point (32736 input + 32 output), NOT a 32768-token "
          "prompt, and the maximum TESTED point (not a demonstrated maximum feasible context); live paged-KV "
          "memory is DERIVED (derived_live_paged_kv_bytes), not measured; no Experiment 3/4 samples reused.")
    print(f"Failed-cell policy: {FAILED_POLICY}")
    print()

    prov = preflight(args.dry_run)
    print("Preflight OK.")
    for k in ("git_branch", "git_head", "vllm_kvquant_tree", "rabit_kv2_sha256", "runner_script_sha256",
              "modal_app_sha256", "worker_sha256", "correctness_gate_sha256", "watchdog_sha256",
              "exp3_worker_sha256", "exp3_modal_app_sha256", "canonical_benchmark_deployment_sha256"):
        print(f"  {k:<40} {prov[k]}")
    print("  equivalence: engine kwargs canonical == exp3 == exp5 (except kv_cache_dtype/model path); output length "
          "and protocol constants == exp3; prompt construction == exp3 except length; timed region AST == exp3; "
          "image AST == canonical; clean-state/watchdog helpers AST == exp3; gate regression() AST == canonical; "
          "no engine RPC")
    print(f"  protected paths: {len(prov['protected_paths'])} (incl. all Experiment 3 and 4 evidence); prior "
          f"evidence files hashed: {len(prov['prior_evidence_sha256_raw'])}; archived Exp5 attempts hashed: "
          f"{len(prov['archived_attempts_sha256'])}")
    if prov["uncommitted_experiment_files"]:
        print("  WARNING (dry-run only): a real run would refuse until these are committed:")
        for line in prov["uncommitted_experiment_files"].splitlines():
            print(f"    {line}")
    if prov["existing_top_level_output_files"]:
        print(f"  WARNING (dry-run only): a real run would refuse; previous-attempt files present: "
              f"{prov['existing_top_level_output_files']}")

    plan = {label: {**flatten("requested", requested_kwargs(d)),
                    **flatten("workload", {"context_point": c, "prompt_tokens": p})} for _, label, d, c, p in LEGS}
    keys = sorted(set().union(*plan.values()))
    differing = [x for x in keys if _uniq(pp.get(x) for pp in plan.values()) > 1]
    allowed = set(DTYPE_INDUCED_ALLOWLIST) | set(CONTEXT_INDUCED_ALLOWLIST)
    print("\nPlanned requested-config comparison across the 12 cells "
          "(post-run the same check covers effective config, workload and KV dtype):")
    print(json.dumps({"fields_compared": len(keys), "differing_fields": differing,
                      "dtype_induced_allowlist": DTYPE_INDUCED_ALLOWLIST,
                      "context_induced_allowlist": CONTEXT_INDUCED_ALLOWLIST,
                      "only_allowlisted_fields_differ": set(differing) <= allowed}, indent=2))
    if not set(differing) <= allowed:
        raise SystemExit("Planned configs differ in a non-allowlisted field -- refusing.")

    print("\nLocal command (one Modal run):\n  " + " ".join(build_command()))
    print("Step 0 (in container): idle GPU baseline; no compute process allowed")
    print(f"Step 1 (in container): frozen RABIT-KV correctness gate, fresh process group, watchdog "
          f"{GATE_TIMEOUT_S}s, must pass before any cell:\n  " + " ".join(GATE_COMMAND_DOC))
    for k, label, d, c, p in ALL_CELLS:
        print(f"Step {k + 1} (in container): GPU clean check, then {ROLE[label]} cell {label} ({d}, context {c}, "
              f"prompt {p}), watchdog {LEG_TIMEOUT_S}s, in-container verdict:\n  "
              + " ".join(worker_command(label, d, c, p)))
    print(f"Outputs: {rel(OUT_DIR)}/ ({SESSION_LOG.name}, {GATE_LOG.name}, "
          f"{', '.join(LOG_NAME[l] for _, l, *_ in ALL_CELLS)}, {MANIFEST.name}, {CONFIG_DIFF.name}, "
          f"{INTEGRITY.name}, {SUMMARY.name})")

    if args.dry_run:
        print("\n--dry-run: no Modal/GPU commands executed, no snapshot built, no files written.")
        return 0

    manifest = {
        "experiment": "Experiment 5 -- BF16 vs RABIT-KV context-length scaling (real engine)",
        "plan_reference": "docs/MLSYS_EXPERIMENT_PLAN.md",
        "model": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "gpu": "NVIDIA H100 80GB (Modal)",
        "scope": SCOPE_NOTE,
        "cells": [{"index": k, "cell": label, "role": ROLE[label], "kv_cache_dtype": d, "context_point": c,
                   "actual_prompt_tokens": p, "output_tokens": OUTPUT_TOKENS} for k, label, d, c, p in ALL_CELLS],
        "protocol": {"warmups_per_cell_excluded": WARMUPS_PER_LEG, "measured_reps_per_cell": REPS_PER_LEG,
                     "context_grid": CONTEXT_GRID, "output_tokens": OUTPUT_TOKENS, "max_model_len": MAX_MODEL_LEN,
                     "same_container_same_gpu": True, "fresh_process_per_cell": True,
                     "dtype_induced_allowlist": DTYPE_INDUCED_ALLOWLIST,
                     "context_induced_allowlist": CONTEXT_INDUCED_ALLOWLIST,
                     "gpu_clean_tolerance_mib": GPU_CLEAN_TOLERANCE_MIB,
                     "gpu_clean_max_wait_s": GPU_CLEAN_MAX_WAIT_S,
                     "gate_timeout_s": GATE_TIMEOUT_S, "cell_timeout_s": LEG_TIMEOUT_S,
                     "total_watchdog_budget_s": WATCHDOG_BUDGET_S,
                     "modal_function_backstop_timeout_s": MODAL_FUNCTION_TIMEOUT_S,
                     "conditioning_cells": [label for _, label, *_ in CONDITIONING],
                     "retries": 0, "failed_cell_policy": FAILED_POLICY},
        "labels": {"capacity": CAPACITY_LABEL, "live_kv": LIVE_KV_LABEL, "gpu_memory": GPU_MEMORY_LABEL,
                   "latency": LATENCY_LABEL},
        "started_utc": now(), "completed_utc": None, "status": "running",
        "protected_paths_post_run_status": "pending",
        "provenance": prov, "runs": [],
    }
    write_manifest(manifest)
    return execute(manifest)


def execute(manifest: dict) -> int:
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
        raise SystemExit(f"\nEXPERIMENT 5 RUNNER FAILED: {type(exc).__name__}: {exc}{extra}\n"
                         f"Manifest marked failed; partial logs preserved under {rel(OUT_DIR)}/. Not retried.") from exc


if __name__ == "__main__":
    raise SystemExit(main())
