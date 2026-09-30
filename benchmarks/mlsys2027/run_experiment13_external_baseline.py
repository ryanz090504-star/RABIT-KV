"""
RABIT-KV MLSys 2027 -- Experiment 13 runner: matched BF16 / native FP8 / RABIT-KV / TurboQuant PHYSICAL capacity and
single-request decode latency in ONE H100 session (a matched, METHOD-NATIVE SYSTEM comparison; not a quantizer-kernel
comparison; no throughput claim; no quality claim).

  * one container, one physical H100, one image, one model snapshot, ONE patched vllm-kvquant snapshot for all four
    conditions (the official upstream vLLM fa4321de3 / PR #47609 backport, local commit 611a4ff);
  * default V2 model runner and vLLM's default multiprocess engine core for every leg (the accepted Exp4 topology);
  * the frozen RABIT correctness gate and the TurboQuant correctness gate (tests/quantization/test_turboquant.py) must
    pass before any leg;
  * mirrored 8-leg order A1 B1 C1 D1 D2 C2 B2 A2 (A = bfloat16, B = fp8_e4m3, C = rabit_kv2,
    D = turboquant_k3v4_nc), each a fresh engine process; 5 warmups + 30 measured reps per leg -> 60 samples per
    condition, pooled only within this session; stop at the first failure, no retry;
  * capacity = the live allocator's num_gpu_blocks x block_size (PRIMARY); derived bytes / token are reported
    separately and labelled theoretical.
No Exp3 / Exp4 sample is read, reused or pooled; they are historical context only.

Usage:
    python benchmarks/mlsys2027/run_experiment13_external_baseline.py --write-protocol   (once, before commit)
    python benchmarks/mlsys2027/run_experiment13_external_baseline.py --dry-run
    python benchmarks/mlsys2027/run_experiment13_external_baseline.py            (NOT until explicitly authorized)
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_exp13_turboquant_probe as p1  # noqa: E402  (image-expression extractor; read-only)
import run_experiment1_quality_frontier as e1  # noqa: E402  (accepted; read-only helpers)
import run_experiment12_variance as r12  # noqa: E402  (accepted; read-only: protected paths)

ROOT = e1.ROOT
RUNNER_SCRIPT = Path(__file__).resolve()
MODAL_APP = HERE / "exp13_deployment_modal.py"
WORKER = HERE / "exp13_engine_worker.py"
EXP4_MODAL = HERE / "exp4_deployment_modal.py"
EXP4_WORKER = HERE / "exp4_engine_worker.py"
GATE = HERE / "exp3_correctness_gate.py"
WATCHDOG = HERE / "exp3_watchdog.py"
PROTOCOL = HERE / "exp13_external_baseline_protocol.json"
OUT_DIR = ROOT / "results" / "mlsys2027" / "external_baseline" / "exp13"
SESSION_LOG = OUT_DIR / "modal_session.log"
MANIFEST = OUT_DIR / "manifest.json"
INTEGRITY = OUT_DIR / "integrity_check.json"
SUMMARY = OUT_DIR / "matched_capacity_latency_summary.json"
MUST_BE_COMMITTED = [RUNNER_SCRIPT, MODAL_APP, WORKER, GATE, WATCHDOG, PROTOCOL, HERE / "test_experiment13_external_baseline.py"]
EXP12_EVIDENCE_COMMIT = "4f767ab03d83e043b2871dd0cd4cf2f8dc862e6b"
PROTECTED_PATHS = [*r12.PROTECTED_PATHS, *r12.EXP11_FILES, r12.OUT_DIR,
                   ROOT / "results" / "mlsys2027" / "external_baseline" / "feasibility_probe",
                   ROOT / "results" / "mlsys2027" / "external_baseline" / "legacy_runner_probe",
                   ROOT / "results" / "mlsys2027" / "external_baseline" / "v2_runner_probe"]

BACKPORT_COMMIT = "611a4ffc96c70c81529978dc01a290e87ccf76e9"
UPSTREAM_FIX = "fa4321de3d894c50c5ca0766dffa352d3fb07423"
ATTN_UTILS = "vllm-kvquant/vllm/v1/worker/gpu/attn_utils.py"
UPSTREAM_ATTN_UTILS_BLOB = "5fcc9053bf4c65b5f0f110f62edae66f4c363c4c"
EXPECTED_RABIT_SHA256_LF = "7e628c94eebb9fe689bf416ea61f748c0f909a82d0f229c463edd1a0df92e6ae"

A, B, C, D = "bfloat16", "fp8_e4m3", "rabit_kv2", "turboquant_k3v4_nc"
CONDITIONS = [A, B, C, D]
SHORT = {A: "bf16", B: "fp8", C: "rabit", D: "turboquant"}
LEGS = [(1, "A1", A), (2, "B1", B), (3, "C1", C), (4, "D1", D), (5, "D2", D), (6, "C2", C), (7, "B2", B), (8, "A2", A)]
WARMUPS_PER_LEG = 5
REPS_PER_LEG = 30
SAMPLES_PER_CONDITION = 60
CONTEXT_TOKENS = 2048
OUTPUT_TOKENS = 32
REQUESTED_BLOCK_SIZE = 32
TQ_BOUNDARY_LAYERS = ["0", "1", "30", "31"]
# Fields that may differ between conditions (never between the two legs of one condition); each is a direct function
# of the requested kv_cache_dtype. The last four apply to TurboQuant only (method-required backend / boundary policy).
DTYPE_INDUCED_ALLOWLIST = [
    "requested.kv_cache_dtype", "kv_dtype.requested_kv_cache_dtype", "kv_dtype.engine_cache_dtype",
    "kv_dtype.resolved_kv_torch_dtype", "kv_dtype.kv_quant_mode", "kv_dtype.fp8_storage_view_dtype",
    "requested.attention_config", "effective.attention_backend", "effective.flash_attn_version",
    "effective.kv_cache_dtype_skip_layers",
]
THEORETICAL_BYTES_PER_TOKEN = {
    A: {"bytes": 32 * 2 * 8 * 128 * 2, "formula": "32 layers x (K,V) x 8 heads x 128 x 2 B (BF16)"},
    B: {"bytes": 32 * 2 * 8 * 128 * 1, "formula": "32 layers x (K,V) x 8 heads x 128 x 1 B (FP8 e4m3, per-tensor scales)"},
    C: {"bytes": 32 * 776, "formula": "32 layers x 776 B (RABIT physical page 24,832 B per 32-token block per layer)"},
    D: {"bytes": 28 * 8 * 118 + 4 * 8 * 128 * 2 * 2,
        "formula": "28 TurboQuant layers x 8 heads x 118 B slot (48 B 3-bit K codes + 2 B fp16 norm + 64 B 4-bit V "
                   "codes + 4 B fp16 scale/zero) + 4 BF16 boundary layers x 4,096 B"},
}
PAIRS = [(C, D), (C, A), (C, B), (D, A), (D, B), (B, A)]  # (x, y): x relative to y
CAPACITY_LABEL = "OBSERVED PHYSICAL vLLM allocator KV capacity (num_gpu_blocks x block_size) from the live engine"
CLAIM_BOUNDARY = ("A matched, method-native system comparison in ONE H100 session (default V2 runner, eager, "
                  "single request, 2048-token prompt, 32 output tokens). TurboQuant uses its method-required attention "
                  "backend (TURBOQUANT + FlashAttention v2 on its four BF16 boundary layers) while BF16 / FP8 / RABIT use "
                  "TRITON_ATTN; latency differences are therefore system-level, not quantizer-kernel-level. No "
                  "throughput claim (single request), no quality claim, no generalization beyond this configuration. "
                  "Exp3 / Exp4 absolute TPOT are not comparable as the same experiment (different snapshot / session).")
DISCLOSURE = ("TurboQuant was evaluated using our vendored vLLM snapshot (upstream base f329ce4) with the official "
              "upstream TurboQuant V2 cache-dtype integration fix from fa4321de3d894c50c5ca0766dffa352d3fb07423 "
              "(PR #47609; first released in vLLM v0.25.0) backported. The change only restores the intended cache-dtype "
              "dispatch for TurboQuant specs in the model runner's KV-cache reshape; TurboQuant's quantizer, packed "
              "cache format, boundary-layer policy, and kernels were unchanged. All four methods ran on the same "
              "patched snapshot.")

TAG = re.compile(r"^(EXP13_[A-Z_]+)=(\{.*\}|\[.*\]|null|true|false|-?\d+(?:\.\d+)?|\".*\")\s*$")
ROWLINE = re.compile(r"^(EXP13_SAMPLE|EXP13_WARMUP) (\{.*\})\s*$")
KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
KV_MEM = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
V2_RUNNER = "Using V2 Model Runner"
GATE_RESULT = re.compile(r"^EXP3_GATE_RESULT=(\{.*\})\s*$")
PYTEST_SUMMARY = re.compile(r"^=*\s*(\d+) passed(?:, (\d+) warnings?)?(?:.* in ([\d.]+)s)?")
PYTEST_FAILED = re.compile(r"\b(\d+) (failed|errors?)\b")


# ------------------------------------------------------------------------------------------ equivalence to Exp4
def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _const(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def _fn(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def verify_equivalence() -> dict:
    """The Exp13 worker / Modal image are the accepted Exp4 ones except for the documented differences."""
    w13, w4 = _module(WORKER), _module(EXP4_WORKER)
    checks = {
        "image_expression_identical_to_exp4": p1.image_expr(MODAL_APP) == p1.image_expr(EXP4_MODAL),
        "base_engine_kwargs_identical": _const(w13, "BASE_ENGINE_KWARGS") == _const(w4, "BASE_ENGINE_KWARGS"),
        "context_output_tokens_identical": all(_const(w13, k) == _const(w4, k) for k in ("CONTEXT_TOKENS", "OUTPUT_TOKENS")),
        "timed_region_identical": ast.dump(_fn(_fn(w13, "main"), "one")) == ast.dump(_fn(_fn(w4, "main"), "one")),
        "allowed_dtypes": _const(w13, "ALLOWED_KV_CACHE_DTYPES") == (*_const(w4, "ALLOWED_KV_CACHE_DTYPES"), D),
        "backend_pin_removed_only_for_turboquant": 'if args.kv_cache_dtype.startswith("turboquant_"):\n'
                                                   '        kwargs.pop("attention_config")' in WORKER.read_text(encoding="utf-8"),
    }
    return checks


def worker_command(label: str, dtype: str, model_dir: str = "<modelscope snapshot dir>") -> list[str]:
    return ["python", "/opt/exp13/exp13_engine_worker.py", "--kv-cache-dtype", dtype, "--model-dir", model_dir,
            "--warmups", str(WARMUPS_PER_LEG), "--reps", str(REPS_PER_LEG), "--leg", label]


def legs_arg() -> str:
    return ",".join(f"{label}={d}" for _, label, d in LEGS)


def build_protocol() -> dict:
    return {
        "experiment": 13,
        "type": "matched METHOD-NATIVE SYSTEM comparison: observed physical KV capacity + single-request decode latency "
                "(NOT a quality experiment, NOT a throughput benchmark, NOT a quantizer-kernel comparison)",
        "plan": "docs/MLSYS_EXPERIMENT_PLAN.md Experiment 13 (external baseline: BF16 / FP8 / TurboQuant / RABIT-KV)",
        "plan_deviation_approved": "BF16 / FP8 / RABIT are re-measured in the SAME session (not reused from Exp3 / Exp4); "
                                   "Exp3 / Exp4 are historical context only",
        "conditions": {
            "A": {"kv_cache_dtype": A, "role": "reference (uncompressed)", "attention_backend": "TRITON_ATTN"},
            "B": {"kv_cache_dtype": B, "role": "vLLM native FP8 E4M3 per-tensor KV", "attention_backend": "TRITON_ATTN"},
            "C": {"kv_cache_dtype": C, "role": "RABIT-KV final physical policy K3 / V2 / G32 / R4 / META8g64 (frozen "
                                              "source; rabit_kv2.py sha256 LF " + EXPECTED_RABIT_SHA256_LF + ")",
                  "attention_backend": "TRITON_ATTN"},
            "D": {"kv_cache_dtype": D, "role": "external baseline: vLLM TurboQuant preset turboquant_k3v4_nc (3-bit "
                                              "Lloyd-Max K + fp16 norm, 4-bit uniform V + fp16 scale/zero, packed "
                                              "118 B slot per head per token; layers 0, 1, 30, 31 BF16)",
                  "attention_backend": "method-required: TURBOQUANT (28 layers) + FlashAttention v2 (4 BF16 boundary layers)",
                  "boundary_layers": TQ_BOUNDARY_LAYERS}},
        "snapshot": {"vllm_kvquant": "the committed tree at the execution commit (identical for all four conditions)",
                     "upstream_base": "f329ce405b12623fb8b1cf1830f12e5a712523be",
                     "backport": {"local_commit": BACKPORT_COMMIT, "upstream_commit": UPSTREAM_FIX, "upstream_pr": 47609,
                                  "upstream_first_release": "v0.25.0", "file": ATTN_UTILS,
                                  "blob": UPSTREAM_ATTN_UTILS_BLOB},
                     "no_other_source_change": True},
        "engine": {"worker_base_engine_kwargs": _const(_module(WORKER), "BASE_ENGINE_KWARGS"),
                   "turboquant_only_difference": "attention_config pin removed (method-required backend selection)",
                   "model_runner": "default V2 (VLLM_USE_V2_MODEL_RUNNER unset); 'Using V2 Model Runner' required in every leg",
                   "process_topology": "vLLM default multiprocess engine core for all four (accepted Exp4 topology)"},
        "workload": {"model": "LLM-Research/Meta-Llama-3.1-8B-Instruct", "prompt": "[BOS] + ' the' x 2047 token ids",
                     "context_tokens": CONTEXT_TOKENS, "output_tokens": OUTPUT_TOKENS, "decoding": "greedy",
                     "ignore_eos": True, "requests": "single request per generate call"},
        "session": {"gpu": "one NVIDIA H100 80GB (Modal gpu='H100'; the GPU name is recorded and must be an H100 80GB HBM3)",
                    "legs": [{"index": k, "label": label, "kv_cache_dtype": d} for k, label, d in LEGS],
                    "order": "A B C D D C B A", "warmups_per_leg": WARMUPS_PER_LEG, "reps_per_leg": REPS_PER_LEG,
                    "samples_per_condition": SAMPLES_PER_CONDITION, "fresh_engine_process_per_leg": True,
                    "gates_before_legs": ["RABIT physical correctness gate (exp3_correctness_gate.py, unchanged)",
                                          "TurboQuant correctness gate: pytest tests/quantization/test_turboquant.py"],
                    "failure_rule": "any gate / leg failure, watchdog timeout or unclean GPU aborts the session; no retry; "
                                    "no substitution"},
        "metrics": {"capacity": {"primary": CAPACITY_LABEL, "ratios": [f"{SHORT[x]} / {SHORT[y]}" for x, y in PAIRS],
                                 "secondary_labelled_theoretical": THEORETICAL_BYTES_PER_TOKEN,
                                 "implied_bytes_per_token": "reported KV-cache GiB x 2^30 / observed capacity (derived, not primary)"},
                    "latency": {"per_sample": ["tpot_ms", "ttft_ms", "wall_ms", "output_tokens"],
                                "pooling": "the 60 measured samples of each condition (both legs) within this session only",
                                "summary": ["median TPOT", "p90 TPOT (nearest-rank on the sorted pooled samples)",
                                            "median TTFT", "median wall", "per-leg medians", "order effect: leg-2 median "
                                            "minus leg-1 median per condition", "pairwise median-TPOT delta %"]}},
        "integrity_gates": ["both correctness gates pass", "all 8 legs exit 0 with no watchdog timeout",
                            "GPU clean before every leg",
                            "every leg: 'Using V2 Model Runner', effective block_size 32, 5 warmups, 30 samples, "
                            "prompt 2048 and output 32 tokens in every sample",
                            "non-allowlisted engine / workload fields identical across all 8 legs",
                            "the two legs of each condition report identical capacity and identical config",
                            "TurboQuant legs: engine cache dtype turboquant_k3v4_nc, skip layers exactly [0, 1, 30, 31]",
                            "RABIT legs: frozen RABIT markers present and rabit_kv2.py sha matches"],
        "dtype_induced_allowlist": DTYPE_INDUCED_ALLOWLIST,
        "fairness": {
            "MATCHED": ["GPU model / container / session", "model weights and tokenizer", "patched vLLM snapshot",
                        "V2 model runner", "process topology", "prompt 2048 / output 32 / greedy / ignore_eos",
                        "max_model_len 32768", "max_num_seqs 32", "max_num_batched_tokens 16384",
                        "gpu_memory_utilization 0.82", "block_size 32", "enforce_eager", "chunked prefill on",
                        "prefix caching off", "warmup policy", "measurement code (timed region AST-identical to Exp4)",
                        "condition scheduling (mirrored A B C D D C B A)"],
            "METHOD_INHERENT": ["KV representation, precision, metadata / per-vector parameters, residual window, "
                                "and TurboQuant's four BF16 boundary layers"],
            "UNAVOIDABLY_DIFFERENT": ["attention backend: TRITON_ATTN (A, B, C) vs TURBOQUANT + FlashAttention v2 on "
                                      "the BF16 boundary layers (D); direction of the latency bias not determined"]},
        "claim_boundary": CLAIM_BOUNDARY,
        "disclosure": DISCLOSURE,
        "outputs": {"session_log": SESSION_LOG.relative_to(ROOT).as_posix(), "manifest": MANIFEST.relative_to(ROOT).as_posix(),
                    "integrity": INTEGRITY.relative_to(ROOT).as_posix(), "summary": SUMMARY.relative_to(ROOT).as_posix()},
        "not_reported": ["throughput", "quality", "multi-request / batched serving", "other GPUs or runners"],
    }


def load_protocol() -> dict:
    committed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if committed != json.loads(json.dumps(build_protocol())):
        raise RuntimeError("exp13_external_baseline_protocol.json differs from the regenerated protocol")
    return committed


# ------------------------------------------------------------------------------------------------- preflight
def protected_status() -> str:
    return e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in PROTECTED_PATHS])


def preflight(dry_run: bool) -> dict:
    status = protected_status()
    if status:
        raise RuntimeError("protected paths are not clean:\n" + status)
    if e1.run_git("diff", "--name-only", EXP12_EVIDENCE_COMMIT, "--", r12.OUT_DIR.relative_to(ROOT).as_posix()):
        raise RuntimeError("Exp12 accepted evidence differs from 4f767ab")
    if e1.run_git("rev-parse", f"HEAD:{ATTN_UTILS}") != UPSTREAM_ATTN_UTILS_BLOB:
        raise RuntimeError("snapshot does not contain the backported upstream attn_utils.py")
    e1.run_git("merge-base", "--is-ancestor", BACKPORT_COMMIT, "HEAD")  # raises if not an ancestor
    vdiff = e1.run_git("diff", "--name-only", BACKPORT_COMMIT, "HEAD", "--", "vllm-kvquant")
    if vdiff:
        raise RuntimeError("vllm-kvquant changed after the backport commit:\n" + vdiff)
    rabit = hashlib.sha256((ROOT / "vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if rabit != EXPECTED_RABIT_SHA256_LF:
        raise RuntimeError("rabit_kv2.py is not the frozen source")
    eq = verify_equivalence()
    if not all(eq.values()):
        raise RuntimeError(f"Exp13 harness is not equivalent to the accepted Exp4 harness: {eq}")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") is not None:
        raise RuntimeError("VLLM_USE_V2_MODEL_RUNNER must be unset (default V2 runner)")
    protocol = load_protocol()
    uncommitted = e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in MUST_BE_COMMITTED])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: Experiment 13 harness has uncommitted changes:\n" + uncommitted)
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding="utf-8")).get("status") == "passed":
        raise RuntimeError(f"{MANIFEST} already records a passed run; refusing to overwrite")
    return {"git_head": e1.run_git("rev-parse", "HEAD"), "vllm_kvquant_tree": e1.run_git("rev-parse", "HEAD:vllm-kvquant"),
            "backport_commit": BACKPORT_COMMIT, "rabit_kv2_sha256_lf": rabit, "equivalence": eq,
            "protocol_sha256": e1.sha256(PROTOCOL), "runner_sha256": e1.sha256(RUNNER_SCRIPT),
            "worker_sha256": e1.sha256(WORKER), "modal_sha256": e1.sha256(MODAL_APP), "protocol": protocol,
            "uncommitted_files": uncommitted or None}


# ---------------------------------------------------------------------------------------------------- parsing
def demux(session_text: str) -> tuple[dict, list[str], list[str], list[str]]:
    legs = {k: [] for k, _, _ in LEGS}
    gate, tqgate, top = [], [], []
    prefixes = {f"[leg{k}:{d}] ": k for k, _, d in LEGS}
    for line in session_text.splitlines():
        if line.startswith("[gate] "):
            gate.append(line[len("[gate] "):])
        elif line.startswith("[tqgate] "):
            tqgate.append(line[len("[tqgate] "):])
        else:
            for prefix, k in prefixes.items():
                if line.startswith(prefix):
                    legs[k].append(line[len(prefix):])
                    break
            else:
                top.append(line)
    return legs, gate, tqgate, top


def parse_worker(lines: list[str]) -> dict:
    out: dict = {"tags": {}, "samples": [], "warmups": [], "kv_log_tokens": None, "kv_log_gib": None,
                 "v2_runner": False, "line_count": len(lines)}
    for line in lines:
        s = line.strip()
        m = ROWLINE.match(s)
        if m:
            (out["samples"] if m.group(1) == "EXP13_SAMPLE" else out["warmups"]).append(json.loads(m.group(2)))
            continue
        m = TAG.match(s)
        if m:
            out["tags"].setdefault(m.group(1), json.loads(m.group(2)))
            continue
        if V2_RUNNER in line:
            out["v2_runner"] = True
        m = KV_TOKENS.search(line)
        if m:
            out["kv_log_tokens"] = int(m.group(1).replace(",", ""))
        m = KV_MEM.search(line)
        if m:
            out["kv_log_gib"] = float(m.group(1))
    return out


def parse_gates(gate: list[str], tqgate: list[str]) -> dict:
    res = next((json.loads(m.group(1)) for ln in gate if (m := GATE_RESULT.match(ln.strip()))), None)
    passed = next((int(m.group(1)) for ln in reversed(tqgate) if (m := PYTEST_SUMMARY.match(ln.strip()))), None)
    failed = [ln for ln in tqgate if PYTEST_FAILED.search(ln) and ("failed" in ln or "error" in ln.lower())]
    return {"rabit_gate_result": res, "rabit_gate_passed": bool(res and res.get("passed")),
            "tq_gate_pytest_passed": passed, "tq_gate_failure_lines": failed[:10]}


def parse_top(lines: list[str]) -> dict:
    out: dict = {"leg_exit": {}, "pre_leg": {}, "tq_gate_exit": None, "gate_exit": None, "timeouts": [],
                 "complete": False, "environment": None}
    for line in lines:
        s = line.strip()
        if s == "EXP13_MIRRORED_COMPLETE":
            out["complete"] = True
        m = TAG.match(s)
        if not m:
            continue
        tag, p = m.group(1), json.loads(m.group(2))
        if tag == "EXP13_LEG_EXIT":
            out["leg_exit"][p["leg"]] = p["returncode"]
        elif tag == "EXP13_PRE_LEG_GPU_STATE":
            out["pre_leg"][p["leg"]] = p.get("clean")
        elif tag == "EXP13_TQ_GATE_EXIT":
            out["tq_gate_exit"] = p["returncode"]
        elif tag == "EXP13_GATE_EXIT":
            out["gate_exit"] = p["returncode"]
        elif tag == "EXP13_WATCHDOG_TIMEOUT":
            out["timeouts"].append(p)
        elif tag == "EXP13_ENVIRONMENT":
            out["environment"] = p
    return out


def capacity(parsed_leg: dict) -> int | None:
    """Observed allocator capacity from the worker's EXP13_CAPACITY record (num_gpu_blocks x block_size) -- never
    from nominal bit widths."""
    c = parsed_leg["tags"].get("EXP13_CAPACITY")
    if not c or c.get("num_gpu_blocks") is None or c.get("block_size") is None:
        return None
    if c["capacity_tokens"] != c["num_gpu_blocks"] * c["block_size"]:
        return None
    return c["capacity_tokens"]


def leg_config(p: dict) -> dict:
    t = p["tags"]
    flat = {}
    for section, prefix in (("EXP13_REQUESTED_ENGINE_KWARGS", "requested"), ("EXP13_EFFECTIVE_ENGINE_CONFIG", "effective"),
                            ("EXP13_WORKLOAD", "workload"), ("EXP13_KV_DTYPE", "kv_dtype")):
        for k, v in (t.get(section) or {}).items():
            if (prefix, k) in (("requested", "model"), ("effective", "model")):
                continue
            flat[f"{prefix}.{k}"] = v
    return flat


def config_diff(parsed: dict) -> dict:
    configs = {label: leg_config(parsed[k]) for k, label, _ in LEGS}
    dtype_of = {label: d for _, label, d in LEGS}
    keys = sorted(set().union(*configs.values()))
    violations = []
    for key in keys:
        vals = {label: json.dumps(cfg.get(key), sort_keys=True) for label, cfg in configs.items()}
        if len(set(vals.values())) == 1:
            continue
        if key not in DTYPE_INDUCED_ALLOWLIST:
            violations.append({"field": key, "reason": "non-dtype field differs between legs", "values": vals})
            continue
        for d in CONDITIONS:
            if len({vals[label] for label in vals if dtype_of[label] == d}) != 1:
                violations.append({"field": key, "reason": "differs between the two legs of one condition",
                                   "condition": d})
    return {"violations": violations, "passed": not violations}


# ------------------------------------------------------------------------------------------------ statistics
def p90(values: list[float]) -> float:
    s = sorted(values)
    return s[max(0, math.ceil(0.9 * len(s)) - 1)]


def build_summary(parsed: dict, top: dict) -> dict:
    by = {d: [k for k, _, dd in LEGS if dd == d] for d in CONDITIONS}
    cap = {d: capacity(parsed[by[d][0]]) for d in CONDITIONS}
    lat = {}
    for d in CONDITIONS:
        samples = [s for k in by[d] for s in parsed[k]["samples"]]
        tpot = [s["tpot_ms"] for s in samples]
        legs = {label: statistics.median(s["tpot_ms"] for s in parsed[k]["samples"]) for k, label, dd in LEGS if dd == d}
        l1, l2 = [label for k, label, dd in LEGS if dd == d]
        lat[SHORT[d]] = {"n": len(samples), "median_tpot_ms": statistics.median(tpot), "p90_tpot_ms": p90(tpot),
                         "median_ttft_ms": statistics.median(s["ttft_ms"] for s in samples),
                         "median_wall_ms": statistics.median(s["wall_ms"] for s in samples),
                         "per_leg_median_tpot_ms": legs, "order_effect_ms": legs[l2] - legs[l1]}
    cap_out = {SHORT[d]: {"observed_capacity_tokens": cap[d],
                          "implied_bytes_per_token": (parsed[by[d][0]]["kv_log_gib"] * 2**30 / cap[d])
                          if cap[d] and parsed[by[d][0]]["kv_log_gib"] else None,
                          "theoretical_bytes_per_token": THEORETICAL_BYTES_PER_TOKEN[d]} for d in CONDITIONS}
    pairs = {f"{SHORT[x]}_vs_{SHORT[y]}": {
        "capacity_ratio": (cap[x] / cap[y]) if cap[x] and cap[y] else None,
        "median_tpot_delta_pct": 100.0 * (lat[SHORT[x]]["median_tpot_ms"] / lat[SHORT[y]]["median_tpot_ms"] - 1.0)}
        for x, y in PAIRS}
    return {"capacity_label": CAPACITY_LABEL, "capacity": cap_out, "latency": lat, "pairs": pairs,
            "claim_boundary": CLAIM_BOUNDARY, "disclosure": DISCLOSURE,
            "gpu": (top.get("environment") or {}).get("gpus")}


def integrity(parsed: dict, gates: dict, top: dict, diff: dict) -> dict:
    checks = {"rabit_gate_passed": gates["rabit_gate_passed"] and top["gate_exit"] == 0,
              "tq_gate_passed": top["tq_gate_exit"] == 0 and bool(gates["tq_gate_pytest_passed"])
                                and not gates["tq_gate_failure_lines"],
              "no_watchdog_timeout": not top["timeouts"], "session_complete": top["complete"],
              "gpu_is_h100_80gb": any("H100 80GB" in g.get("name", "") for g in ((top.get("environment") or {}).get("gpus") or []))}
    for k, label, d in LEGS:
        p = parsed[k]
        eff = p["tags"].get("EXP13_EFFECTIVE_ENGINE_CONFIG") or {}
        s = p["samples"]
        checks[f"{label}_exit_0"] = top["leg_exit"].get(label) == 0
        checks[f"{label}_gpu_clean_before"] = top["pre_leg"].get(label) is True
        checks[f"{label}_v2_runner"] = p["v2_runner"]
        checks[f"{label}_block_size_32"] = eff.get("block_size") == REQUESTED_BLOCK_SIZE
        checks[f"{label}_counts"] = len(p["warmups"]) == WARMUPS_PER_LEG and len(s) == REPS_PER_LEG
        checks[f"{label}_tokens"] = bool(s) and all(x["prompt_tokens"] == CONTEXT_TOKENS and
                                                    x["output_tokens"] == OUTPUT_TOKENS for x in s)
        checks[f"{label}_capacity_observed"] = capacity(p) is not None
        kvd = p["tags"].get("EXP13_KV_DTYPE") or {}
        checks[f"{label}_engine_cache_dtype"] = kvd.get("engine_cache_dtype") == d
        if d == D:
            checks[f"{label}_tq_boundary_layers"] = [str(x) for x in eff.get("kv_cache_dtype_skip_layers", [])] == TQ_BOUNDARY_LAYERS
        else:
            checks[f"{label}_no_skip_layers"] = eff.get("kv_cache_dtype_skip_layers") == []
        if d == C:
            checks[f"{label}_rabit_markers"] = all((p["tags"].get("EXP13_RABIT_MARKERS") or {}).values()) and \
                bool(p["tags"].get("EXP13_RABIT_MARKERS"))
    for d in CONDITIONS:
        ks = [k for k, _, dd in LEGS if dd == d]
        checks[f"{SHORT[d]}_capacity_identical_across_legs"] = len({capacity(parsed[k]) for k in ks}) == 1
    checks["config_diff_passed"] = diff["passed"]
    return {"checks": checks, "passed": all(checks.values())}


def analyze(session_text: str) -> dict:
    legs, gate, tqgate, top_lines = demux(session_text)
    parsed = {k: parse_worker(v) for k, v in legs.items()}
    gates, top = parse_gates(gate, tqgate), parse_top(top_lines)
    diff = config_diff(parsed)
    integ = integrity(parsed, gates, top, diff)
    summary = build_summary(parsed, top) if integ["passed"] else None
    return {"integrity": {**integ, "gates": gates, "config_diff": diff}, "summary": summary}


# ------------------------------------------------------------------------------------------------------ main
def build_snapshot() -> Path:
    out = Path(tempfile.mkdtemp(prefix="exp13_vllm_snapshot_")) / "vllm_kvquant_snapshot.zip"
    e1.run_git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(out), "HEAD:vllm-kvquant")
    return out


def main(argv: list[str] | None = None) -> int:
    e1.make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-protocol", action="store_true", help="write the protocol (pre-commit only)")
    a = ap.parse_args(argv)
    if a.write_protocol:
        if PROTOCOL.exists():
            raise SystemExit(f"{PROTOCOL.name} already exists; the frozen protocol is never overwritten")
        PROTOCOL.write_text(json.dumps(build_protocol(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {PROTOCOL.relative_to(ROOT).as_posix()}")
        return 0
    prov = preflight(a.dry_run)
    print("RABIT-KV MLSys 2027 -- Experiment 13 matched external baseline (BF16 / FP8 / RABIT / TurboQuant)")
    print("Preflight OK:", json.dumps({k: prov[k] for k in ("git_head", "vllm_kvquant_tree", "backport_commit",
                                                             "protocol_sha256")}))
    print("  legs:", legs_arg())
    if prov["uncommitted_files"]:
        print("  WARNING (dry-run only): uncommitted:\n    " + prov["uncommitted_files"].replace("\n", "\n    "))
    if a.dry_run:
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap = build_snapshot()
    os.environ["EXP13_VLLM_SNAPSHOT"] = str(snap)
    manifest = {"experiment": 13, "status": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "provenance": {**prov, "vllm_kvquant_snapshot_sha256": hashlib.sha256(snap.read_bytes()).hexdigest()}}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rc = e1.stream_command([sys.executable, "-m", "modal", "run", str(MODAL_APP), "--legs", legs_arg(),
                            "--warmups", str(WARMUPS_PER_LEG), "--reps-per-leg", str(REPS_PER_LEG)], SESSION_LOG)
    res = analyze(SESSION_LOG.read_text(encoding="utf-8", errors="replace"))
    INTEGRITY.write_text(json.dumps(res["integrity"], indent=2) + "\n", encoding="utf-8")
    ok = rc == 0 and res["integrity"]["passed"] and not protected_status()
    if ok:
        SUMMARY.write_text(json.dumps(res["summary"], indent=2) + "\n", encoding="utf-8")
    manifest.update(status="passed" if ok else "failed", modal_returncode=rc,
                    completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    modal_app_ids=sorted(set(re.findall(r"ap-[A-Za-z0-9]{20,}", SESSION_LOG.read_text(encoding="utf-8", errors="replace")))),
                    session_log_sha256=e1.sha256(SESSION_LOG))
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nEXPERIMENT 13 {manifest['status'].upper()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
