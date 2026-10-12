"""
RABIT-KV Stage3C cliff DIAGNOSTIC runner (pre-fix). DIAGNOSTIC evidence only:
NOT Experiment 5 evidence and never merged into Experiment 5 results.

Purpose: measure the onset and shape of the RABIT-KV non-initial chunked-prefill
(Stage3C) performance cliff around max_num_batched_tokens = 16384, with the
frozen engine configuration of Experiments 3-5 (unchanged):
max_num_batched_tokens 16384, max_model_len 32768, block_size 32,
gpu_memory_utilization 0.82, eager, Triton attention, torch.compile off, CUDA
graphs off, same model snapshot, 32 output tokens, BF16 control vs rabit_kv2
(no FP8).

Protocol (one `modal run` of benchmarks/mlsys2027/stage3c_diag_modal.py):
  * idle GPU baseline; frozen RABIT-KV correctness gate;
  * one fresh engine process per dtype series (bfloat16 control, then
    rabit_kv2); per series one UNMEASURED conditioning request at 16416 prompt
    tokens, then exactly ONE measured request per point:
      16384 (single chunk), 16385 (second chunk q_len 1), 16386, 16415, 16416,
      16417, 16896, 17408, 18432, 20480, 24576 (q_len 2..8192);
  * a 600 s per-request SIGALRM guard (exit 76 when the signal is delivered),
    with the process-group watchdog (each series process under the Experiment 3
    watchdog, whole-group kill and reap) and the Modal function timeout as hard
    process-level backstops; SIGALRM alone is not guaranteed to preempt native
    CUDA/C++ work. No retries; ANY failure stops the diagnostic and the partial
    evidence is kept.

Second-chunk q_len / context_len per point are DERIVED from the scheduler's
chunking rule (first chunk = max_num_batched_tokens tokens).

Per-layer marker contract: the Stage3C marker is logged once per attention-layer
object per engine process (triton_attn.py `_rabit2_logged_chunked`). With one
engine per dtype, the unmeasured conditioning request consumes those markers,
so measured_point_layer_markers_available is false. The conditioning markers
are used only to verify that the Stage3C path was entered and that all 32
layers were observed -- never as a per-layer latency distribution (their
intervals are below the 1 s log-timestamp resolution). Layer-level timing of
the severe 32K case is referenced only as EXTERNAL FROZEN REFERENCE evidence
from results/mlsys2027/context_scaling/failed_attempt_1/, never pooled with
this diagnostic.

Analysis is descriptive only: no complexity law is fitted from one sample per
point.

Usage:
    python benchmarks/mlsys2027/run_stage3c_cliff_diagnostic.py --dry-run
    python benchmarks/mlsys2027/run_stage3c_cliff_diagnostic.py
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
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment3_deployment as r3  # noqa: E402  (frozen; helpers only)
import run_experiment5_context_scaling as r5  # noqa: E402  (frozen; helpers only)
from run_experiment3_deployment import (  # noqa: E402
    FAILED, NOT_EVALUATED, NOT_RUN, PASSED, _function, _module_assign, canonical_llm_kwargs,
    canonical_runner_source, flatten, make_console_encoding_safe, now, rel, run_git, sha256, sha256_raw,
    stream_command,
)

ROOT = r3.ROOT
HERE = ROOT / "benchmarks" / "mlsys2027"
RUNNER_SCRIPT = Path(__file__).resolve()
MODAL_APP = HERE / "stage3c_diag_modal.py"
WORKER = HERE / "stage3c_diag_worker.py"
GATE = HERE / "exp3_correctness_gate.py"
WATCHDOG = HERE / "exp3_watchdog.py"
EXP3_WORKER = HERE / "exp3_engine_worker.py"
EXP3_MODAL_APP = HERE / "exp3_deployment_modal.py"
RABIT_KV2 = r3.RABIT_KV2
EXPECTED_RABIT_SHA256_LF = r3.EXPECTED_RABIT_SHA256_LF
FAILED_ATTEMPT_1 = ROOT / "results" / "mlsys2027" / "context_scaling" / "failed_attempt_1"

OUT_DIR = ROOT / "results" / "mlsys2027" / "diagnostics" / "stage3c_cliff"
SESSION_LOG = OUT_DIR / "modal_session.log"
GATE_LOG = OUT_DIR / "correctness_gate.log"
MANIFEST = OUT_DIR / "manifest.json"
INTEGRITY = OUT_DIR / "integrity_check.json"
ANALYSIS = OUT_DIR / "diagnostic_analysis.json"

EVIDENCE_DIRS = [ROOT / "results" / "mlsys2027" / "deployment", ROOT / "results" / "mlsys2027" / "fp8_baseline",
                 ROOT / "results" / "mlsys2027" / "context_scaling"]
PROTECTED_PATHS = [
    *r3.PROTECTED_PATHS, *EVIDENCE_DIRS,
    *sorted(HERE.glob("*exp1*")), *sorted(HERE.glob("*exp2*")), *sorted(HERE.glob("*exp3*")),
    *sorted(HERE.glob("*exp4*")), *sorted(HERE.glob("*exp5*")), *sorted(HERE.glob("run_experiment[12345]_*")),
]
MUST_BE_COMMITTED = [RUNNER_SCRIPT, MODAL_APP, WORKER, GATE, WATCHDOG]

A, B = "bfloat16", "rabit_kv2"
SERIES = [A, B]  # BF16 control first, then RABIT-KV
SERIES_LOG = {A: "bf16_series.log", B: "rabit_kv2_series.log"}
MAX_NUM_BATCHED_TOKENS = 16384
MAX_MODEL_LEN = 32768
BLOCK_SIZE = 32
OUTPUT_TOKENS = 32
CONDITIONING_PROMPT = 16416
POINTS = [16384, 16385, 16386, 16415, 16416, 16417, 16896, 17408, 18432, 20480, 24576]
REQUEST_CAP_S = 600
SERIES_TIMEOUT_S = 7500
GATE_TIMEOUT_S = 600
WATCHDOG_BUDGET_S = GATE_TIMEOUT_S + len(SERIES) * SERIES_TIMEOUT_S  # 15600
MODAL_FUNCTION_TIMEOUT_S = 16800
DTYPE_INDUCED_ALLOWLIST = ["requested.kv_cache_dtype", "kv_dtype.requested_kv_cache_dtype",
                           "kv_dtype.engine_cache_dtype", "kv_dtype.resolved_kv_torch_dtype",
                           "kv_dtype.kv_quant_mode", "capacity.num_gpu_blocks", "capacity.capacity_tokens"]
EXPECTED_EFFECTIVE = {
    "model_dtype": "torch.bfloat16", "max_model_len": MAX_MODEL_LEN, "enforce_eager": True,
    "block_size": BLOCK_SIZE, "gpu_memory_utilization": 0.82, "enable_prefix_caching": False,
    "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS, "max_num_seqs": 32, "enable_chunked_prefill": True,
    "attention_backend": "AttentionBackendEnum.TRITON_ATTN", "tensor_parallel_size": 1, "quantization": None,
    "compilation_mode": "NONE", "cudagraph_mode": "NONE", "calculate_kv_scales": False,
    "kv_cache_dtype_skip_layers": [], "hf_quantization_config": None,
}
EXPECTED_KV = {A: {"engine_cache_dtype": "bfloat16", "resolved_kv_torch_dtype": "torch.bfloat16",
                   "kv_quant_mode": "NONE"},
               B: {"engine_cache_dtype": "rabit_kv2", "resolved_kv_torch_dtype": "torch.uint8",
                   "kv_quant_mode": "RABIT_KV2"}}
SCOPE = ("DIAGNOSTIC evidence only (pre-fix Stage3C cliff characterization); NOT Experiment 5 evidence. One "
         "measured request per point; descriptive only, no complexity law fitted.")
Q_LEN_NOTE = ("second_chunk_q_len and context_len are DERIVED from the chunking rule (first chunk = "
              "max_num_batched_tokens = 16384 tokens, second chunk = prompt - 16384). Stage3C is entered only for "
              "context_len > 0 and q_len > 1 (triton_attn.py _forward_rabit_kv2).")
MEASURED_POINT_LAYER_MARKERS_AVAILABLE = False
NUM_LAYERS = 32  # Llama-3.1-8B attention layers (one Stage3C marker per layer per engine)
MARKER_NOTE = ("Stage3C per-layer markers are once-per-layer-per-engine. They are consumed by the unmeasured conditioning request and are therefore unavailable for the measured diagnostic points.")
REQUEST_GUARD_NOTE = ("600 s per-request SIGALRM guard, with the process-group watchdog and Modal function timeout as hard process-level backstops. Exit 76 on a delivered request-cap signal; no retries; partial evidence "
                      "preserved; any failure stops the diagnostic; the watchdog kills and reaps the whole process "
                      "group.")
PRIMARY_OUTPUTS = ["RABIT TTFT vs second_chunk_q_len", "BF16 TTFT vs prompt length", "RABIT - BF16 TTFT difference",
                   "per-chunk-token descriptive cost", "Stage3C onset between q_len 1 and 2",
                   "behavior around q_len 31 / 32 / 33"]


def second_chunk_q_len(prompt: int) -> int | None:
    return prompt - MAX_NUM_BATCHED_TOKENS if prompt > MAX_NUM_BATCHED_TOKENS else None


# ---------------------------------------------------------------- utilities
def assert_protected_paths_clean(context: str) -> None:
    status = run_git("status", "--short", "--", *[rel(p) for p in PROTECTED_PATHS])
    if status:
        raise RuntimeError(f"CRITICAL: protected paths changed ({context}):\n" + status)


def prior_evidence_digest() -> dict:
    out = {}
    for d in EVIDENCE_DIRS:
        if d.is_dir():
            for f in sorted(p for p in d.rglob("*") if p.is_file()):
                out[f"{d.name}/{f.relative_to(d).as_posix()}"] = sha256_raw(f)
    return out


def _const(tree, name):
    return ast.literal_eval(_module_assign(tree, name))


def _nested(fn, name):
    for node in ast.walk(fn):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"nested function {name!r} not found")


def _assigns(fn, target):
    return [ast.dump(n) for n in ast.walk(fn)
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == target]


def verify_equivalence() -> dict:
    csrc = canonical_runner_source()
    ctree = ast.parse(csrc)
    w, w3 = ast.parse(WORKER.read_text(encoding="utf-8")), ast.parse(EXP3_WORKER.read_text(encoding="utf-8"))
    m, m3 = ast.parse(MODAL_APP.read_text(encoding="utf-8")), ast.parse(EXP3_MODAL_APP.read_text(encoding="utf-8"))
    canon = {k: v for k, v in canonical_llm_kwargs(csrc).items() if k not in ("model", "kv_cache_dtype")}
    base = _const(w, "BASE_ENGINE_KWARGS")
    if not (canon == _const(w3, "BASE_ENGINE_KWARGS") == base):
        raise RuntimeError("BASE_ENGINE_KWARGS differ from canonical / Experiment 3")
    if not (base["max_num_batched_tokens"] == MAX_NUM_BATCHED_TOKENS and base["max_model_len"] == MAX_MODEL_LEN
            and base["block_size"] == BLOCK_SIZE and base["gpu_memory_utilization"] == 0.82
            and base["enforce_eager"] is True and base["attention_config"] == {"backend": "TRITON_ATTN"}):
        raise RuntimeError("frozen engine settings changed")
    if not (_const(w3, "OUTPUT_TOKENS") == _const(w, "OUTPUT_TOKENS") == OUTPUT_TOKENS):
        raise RuntimeError("OUTPUT_TOKENS differs from Experiment 3")
    if tuple(_const(w, "ALLOWED_KV_CACHE_DTYPES")) != tuple(SERIES):
        raise RuntimeError("worker dtypes != diagnostic series (FP8 must not be present)")
    main3, main = _function(w3, "main"), _function(w, "main")
    for target in ("tok", "bos", "filler", "sp"):
        if _assigns(main3, target) != _assigns(main, target) or len(_assigns(main, target)) != 1:
            raise RuntimeError(f"statement '{target} = ...' differs from Experiment 3")
    p3 = _assigns(main3, "prompt")[0].replace("Name(id='CONTEXT_TOKENS', ctx=Load())",
                                              "Name(id='prompt_tokens', ctx=Load())")
    if p3 not in _assigns(main, "prompt"):
        raise RuntimeError("prompt construction differs from Experiment 3 (other than its length)")
    one3, one = _nested(main3, "one"), _nested(main, "one")
    if [ast.dump(x) for x in one3.body[:5]] != [ast.dump(x) for x in one.body[:5]] \
            or ast.dump(one3.args) != ast.dump(one.args) or len(one3.body) != len(one.body):
        raise RuntimeError("timed region of one() differs from Experiment 3")
    r3d = {ast.literal_eval(k): ast.dump(v) for k, v in zip(one3.body[-1].value.keys, one3.body[-1].value.values)}
    rd = {ast.literal_eval(k): ast.dump(v) for k, v in zip(one.body[-1].value.keys, one.body[-1].value.values)}
    if any(rd.get(k) != v for k, v in r3d.items()) or set(rd) - set(r3d) != {"prompt_token_ids_sha256",
                                                                              "output_token_ids_sha256"}:
        raise RuntimeError("sample fields differ from Experiment 3")
    for node in ast.walk(w):
        if isinstance(node, ast.Attribute) and node.attr == "collective_rpc":
            raise RuntimeError("worker calls collective_rpc")
    if ast.dump(_module_assign(ctree, "image")) != ast.dump(_module_assign(m, "image")):
        raise RuntimeError("diagnostic Modal image differs from the canonical image")
    for const in ("MODEL", "BASE_COMMIT"):
        if _const(ctree, const) != _const(m, const):
            raise RuntimeError(f"{const} differs from the canonical runner")
    for const, mine in (("GPU_CLEAN_TOLERANCE_MIB", 256), ("GPU_CLEAN_MAX_WAIT_S", 60),
                        ("GATE_TIMEOUT_S", GATE_TIMEOUT_S), ("EXPECTED_RABIT_SHA256_LF", EXPECTED_RABIT_SHA256_LF)):
        if not (_const(m, const) == _const(m3, const) == mine):
            raise RuntimeError(f"Modal {const} differs from Experiment 3")
    if not (_const(m, "REQUEST_CAP_S") == REQUEST_CAP_S and _const(m, "SERIES_TIMEOUT_S") == SERIES_TIMEOUT_S
            and SERIES_TIMEOUT_S >= (1 + len(POINTS)) * REQUEST_CAP_S):
        raise RuntimeError("request cap / series timeout inconsistent")
    for fn in ("_emit", "_sha256_file", "_gpu_query", "_compute_apps", "_gpu_state", "_require_clean", "_run_guarded"):
        if ast.dump(_function(m3, fn)).replace("EXP3_", "S3C_") != ast.dump(_function(m, fn)):
            raise RuntimeError(f"Modal helper {fn}() differs from Experiment 3")
    backstop = None
    for dec in _function(m, "diagnose").decorator_list:
        for kw in getattr(dec, "keywords", []):
            if kw.arg == "timeout":
                backstop = ast.literal_eval(kw.value)
    if not (backstop == MODAL_FUNCTION_TIMEOUT_S > WATCHDOG_BUDGET_S):
        raise RuntimeError(f"Modal backstop {backstop} must exceed the watchdog budget {WATCHDOG_BUDGET_S}")
    gtree = ast.parse(GATE.read_text(encoding="utf-8"))
    if ast.dump(ast.Module(body=_function(ctree, "regression").body, type_ignores=[])) != \
            ast.dump(ast.Module(body=_function(gtree, "regression").body, type_ignores=[])):
        raise RuntimeError("gate regression() is not the canonical one")
    return {"engine_kwargs_equal_canonical_exp3": True, "timed_region_ast_equal_exp3": True,
            "prompt_construction_equal_exp3_except_length": True, "no_engine_rpc": True,
            "image_ast_equal_canonical": True, "helpers_ast_equal_exp3": True,
            "watchdog_budget_s": WATCHDOG_BUDGET_S, "modal_backstop_s": backstop}


def requested_kwargs(dtype, model_dir="<modelscope snapshot dir>"):
    return {"model": model_dir, **_const(ast.parse(WORKER.read_text(encoding="utf-8")), "BASE_ENGINE_KWARGS"),
            "kv_cache_dtype": dtype}


def preflight(dry_run: bool) -> dict:
    assert_protected_paths_clean("preflight")
    rabit_sha = sha256(RABIT_KV2)
    if rabit_sha != EXPECTED_RABIT_SHA256_LF:
        raise RuntimeError(f"rabit_kv2.py is not the frozen committed content: {rabit_sha}")
    eq = verify_equivalence()
    uncommitted = run_git("status", "--short", "--", *[rel(p) for p in MUST_BE_COMMITTED])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: diagnostic code has uncommitted changes:\n" + uncommitted)
    leftovers = sorted(p.name for p in OUT_DIR.iterdir()) if OUT_DIR.is_dir() else []
    if leftovers and not dry_run:
        raise RuntimeError(f"Refusing to run: {rel(OUT_DIR)} is not empty ({leftovers})")
    return {"git_branch": run_git("branch", "--show-current"), "git_head": run_git("rev-parse", "HEAD"),
            "vllm_kvquant_tree": run_git("rev-parse", "HEAD:vllm-kvquant"), "rabit_kv2_sha256": rabit_sha,
            "runner_script_sha256": sha256(RUNNER_SCRIPT), "modal_app_sha256": sha256(MODAL_APP),
            "worker_sha256": sha256(WORKER), "correctness_gate_sha256": sha256(GATE),
            "watchdog_sha256": sha256(WATCHDOG), "equivalence": eq,
            "protected_paths": [rel(p) for p in PROTECTED_PATHS], "prior_evidence_sha256_raw": prior_evidence_digest(),
            "uncommitted_diagnostic_files": uncommitted or None, "existing_output_files": leftovers or None}


def build_snapshot() -> dict:
    out = Path(tempfile.mkdtemp(prefix="s3c_vllm_snapshot_")) / "vllm_kvquant_snapshot.zip"
    run_git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(out), "HEAD:vllm-kvquant")
    return {"path": str(out), "sha256": sha256_raw(out), "bytes": out.stat().st_size}


def build_command() -> list[str]:
    return [sys.executable, "-m", "modal", "run", str(MODAL_APP), "--series", ",".join(SERIES),
            "--points", ",".join(map(str, POINTS)), "--conditioning-prompt", str(CONDITIONING_PROMPT)]


# ------------------------------------------------------------------ parsing
TAG = re.compile(r"^(S3C_[A-Z_]+)=(\{.*\})\s*$")
MARK = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[triton_attn\.py:\d+\] "
                  r"RABIT2_STAGE3C_CHUNKED_PREFILL_ACTIVE q_len=(\d+) context_len=(\d+)")
JIT = "Triton kernel JIT compilation during inference"


def _strict_tag(s: str):
    """Return (tag, payload) for a machine line; raise on a malformed S3C_ tag line (never ignore it)."""
    m = TAG.match(s)
    if m:
        return m.group(1), json.loads(m.group(2))
    if re.match(r"^S3C_[A-Z_]+=", s):
        raise ValueError(f"malformed S3C tag line: {s[:120]!r}")
    return None


def demux(text: str) -> tuple[dict, list[str], list[str]]:
    series = {d: [] for d in SERIES}
    gate, top = [], []
    prefixes = {f"[series{k}:{d}] ": d for k, d in enumerate(SERIES, start=1)}
    for line in text.splitlines():
        if line.startswith("[gate] "):
            gate.append(line[7:])
            continue
        for p, d in prefixes.items():
            if line.startswith(p):
                series[d].append(line[len(p):])
                break
        else:
            top.append(line)
    return series, gate, top


def parse_series(lines: list[str]) -> dict:
    out = {"tags": {}, "points": [], "begins": [], "timeouts": [], "failures": [], "complete": False,
           "init": {"jit": 0, "stage3c": [], "oom": 0}, "gpu_memory": {}}
    cur = None
    for line in lines:
        s = line.strip()
        tp = _strict_tag(s)
        if tp:
            tag, p = tp
            if tag == "S3C_POINT_BEGIN":
                cur = {"begin": p, "jit": 0, "stage3c": [], "oom": 0}
                out["begins"].append(cur)
            elif tag == "S3C_POINT":
                if cur is not None:
                    cur["row"] = p
                    out["points"].append(cur)
                cur = None
            elif tag == "S3C_REQUEST_TIMEOUT":
                out["timeouts"].append(p)
            elif tag == "S3C_REQUEST_FAILURE":
                out["failures"].append(p)
            elif tag == "S3C_GPU_MEMORY":
                out["gpu_memory"][p["phase"]] = p["memory_used_mib"]
            else:
                out["tags"][tag] = p
            continue
        if s == "S3C_SERIES_COMPLETE":
            out["complete"] = True
            continue
        bucket = cur if cur is not None else out["init"]
        if JIT in s:
            bucket["jit"] += 1
        if "CUDA out of memory" in s or "OutOfMemoryError" in s:
            bucket["oom"] += 1
        mk = MARK.search(s)
        if mk:
            bucket["stage3c"].append({"time": mk.group(1), "q_len": int(mk.group(2)),
                                      "context_len": int(mk.group(3))})
    return out


def parse_top(lines: list[str]) -> dict:
    out = {"pre": {}, "exit": {}, "start": {}, "proc": {}, "timeouts": [], "stopped": None, "complete": False}
    for line in lines:
        s = line.strip()
        if s == "S3C_DIAGNOSTIC_COMPLETE":
            out["complete"] = True
            continue
        tp = _strict_tag(s)
        if not tp:
            continue
        tag, p = tp
        if tag == "S3C_PRE_LEG_GPU_STATE":
            out["pre"][p["leg"]] = p
        elif tag == "S3C_SERIES_START":
            out["start"][p["kv_cache_dtype"]] = p
        elif tag == "S3C_SERIES_EXIT":
            out["exit"][p["kv_cache_dtype"]] = p
        elif tag == "S3C_PROCESS_EXIT":
            out["proc"][p["label"]] = p
        elif tag == "S3C_WATCHDOG_TIMEOUT":
            out["timeouts"].append(p)
        elif tag == "S3C_STOPPED":
            out["stopped"] = p
        else:
            out[tag] = p
    return out


def series_config(s: dict) -> dict:
    t = s["tags"]
    return {**flatten("requested", t.get("S3C_REQUESTED_ENGINE_KWARGS", {})),
            **flatten("effective", t.get("S3C_EFFECTIVE_ENGINE_CONFIG", {})),
            **flatten("kv_dtype", t.get("S3C_KV_DTYPE", {})), **flatten("capacity", t.get("S3C_CAPACITY", {}))}


def integrity(series: dict, gate: dict, top: dict) -> dict:
    checks = []

    def add(name, cat, state, observed=None):
        if isinstance(state, bool):
            state = PASSED if state else FAILED
        checks.append({"check": name, "category": cat, "state": state, "observed": observed})

    env = top.get("S3C_ENVIRONMENT", {})
    base = top.get("S3C_GPU_BASELINE", {})
    add("diagnostic completed (both series)", "completion", top["complete"])
    add("not stopped", "stop", top["stopped"] is None, top["stopped"])
    add("no series watchdog timeout", "watchdog", not top["timeouts"], top["timeouts"] or None)
    add("exactly one H100", "environment",
        (len(env.get("gpus", [])) == 1 and "H100" in env["gpus"][0].get("name", "")) if env else NOT_EVALUATED)
    add("plan as registered (series order, points, conditioning 16416, cap 600 s)", "environment",
        (env.get("series") == SERIES and env.get("points") == POINTS
         and env.get("conditioning_prompt_tokens") == CONDITIONING_PROMPT
         and env.get("request_cap_s") == REQUEST_CAP_S) if env else NOT_EVALUATED)
    add("rabit_kv2.py in image is frozen", "environment",
        env.get("rabit_kv2_sha256_lf") == EXPECTED_RABIT_SHA256_LF if env else NOT_EVALUATED)
    gate_ran = "S3C_GATE_START" in top
    add("gate passed", "gate", ((top.get("S3C_GATE_EXIT", {}).get("returncode") == 0
                                 and (gate.get("result") or {}).get("passed") is True and gate.get("pytest_exit") == 0
                                 and gate.get("regression_passed_line"))
                                if gate_ran else NOT_RUN))
    model = top.get("S3C_MODEL", {})
    for d in SERIES:
        s = series[d]
        t = s["tags"]
        label = f"series_{d}"
        started = d in top["start"]

        def chk(name, cat, ok, observed=None, _s=started):
            add(f"{d}: {name}", cat, ok if _s else NOT_RUN, observed if _s else None)

        pre = top["pre"].get(label)
        add(f"{d}: GPU clean before series", "gpu_clean", r5.gpu_leg_clean(pre, base) if pre else NOT_RUN)
        chk("series exit 0 and complete", "series", (top["exit"].get(d) or {}).get("returncode") == 0 and s["complete"],
            top["exit"].get(d))
        chk("no request timeout / failure", "request", not s["timeouts"] and not s["failures"],
            (s["timeouts"] + s["failures"]) or None)
        chk("requested kwargs as planned", "config",
            t.get("S3C_REQUESTED_ENGINE_KWARGS") == requested_kwargs(d, (t.get("S3C_REQUESTED_ENGINE_KWARGS") or {})
                                                                     .get("model", "<missing>")))
        chk("model path is the hashed snapshot", "model",
            bool(model) and (t.get("S3C_REQUESTED_ENGINE_KWARGS") or {}).get("model") == model.get("snapshot_dir"))
        eff = t.get("S3C_EFFECTIVE_ENGINE_CONFIG", {})
        chk("effective engine config frozen (eager/Triton/16384/32768/32/0.82/no compile/no graphs)", "config",
            bool(eff) and all(eff.get(k, "<missing>") == v for k, v in EXPECTED_EFFECTIVE.items()),
            {k: eff.get(k) for k, v in EXPECTED_EFFECTIVE.items() if eff.get(k, "<missing>") != v} or None)
        kv = t.get("S3C_KV_DTYPE", {})
        chk("resolved KV dtype", "kv_dtype", kv.get("requested_kv_cache_dtype") == d
            and all(kv.get(k) == v for k, v in EXPECTED_KV[d].items()), kv or None)
        cond = [p for p in s["points"] if p["begin"]["role"] == "conditioning"]
        chk("conditioning request completed (16416 prompt, 32 output)", "conditioning",
            len(cond) == 1 and cond[0]["row"]["prompt_tokens"] == CONDITIONING_PROMPT
            and cond[0]["row"]["output_tokens"] == OUTPUT_TOKENS)
        meas = [p for p in s["points"] if p["begin"]["role"] == "measured"]
        chk(f"exactly one measured request at each of the {len(POINTS)} points, in order", "measurement",
            [p["row"]["planned_prompt_tokens"] for p in meas] == POINTS, [p["row"]["planned_prompt_tokens"] for p in meas])
        chk("every request: engine prompt tokens == planned", "workload",
            all(p["row"]["prompt_tokens"] == p["row"]["planned_prompt_tokens"] for p in s["points"]) and bool(meas))
        chk("every request: 32 output tokens", "workload",
            all(p["row"]["output_tokens"] == OUTPUT_TOKENS for p in s["points"]) and bool(meas))
        chk("every request: engine prompt hash == planned hash", "workload",
            all(p["row"]["prompt_token_ids_sha256"] == p["row"]["planned_prompt_token_ids_sha256"]
                for p in s["points"]) and bool(meas))
        chk("no Triton JIT during any measured request", "jit", bool(meas) and all(p["jit"] == 0 for p in meas),
            {p["row"]["planned_prompt_tokens"]: p["jit"] for p in meas if p["jit"]} or None)
        chk("no OOM", "request", s["init"]["oom"] == 0 and all(p["oom"] == 0 for p in s["begins"]))
        if d == B:
            chk(f"Stage3C path entered on the conditioning request: {NUM_LAYERS} layer markers, q_len 32, "
                f"context_len 16384 (verification only, not timing)", "stage3c",
                bool(cond) and len(cond[0]["stage3c"]) == NUM_LAYERS
                and all(m["q_len"] == CONDITIONING_PROMPT - 16384 and m["context_len"] == 16384
                        for m in cond[0]["stage3c"]),
                len(cond[0]["stage3c"]) if cond else None)
        else:
            chk("no RABIT Stage3C marker in the BF16 series", "stage3c",
                not s["init"]["stage3c"] and all(not p["stage3c"] for p in s["begins"]))
    hashes = {}
    for d in SERIES:
        for p in series[d]["points"]:
            if p["begin"]["role"] == "measured":
                hashes.setdefault(p["row"]["planned_prompt_tokens"], set()).add(p["row"]["prompt_token_ids_sha256"])
    add("prompt hash identical for both dtypes at every point", "workload",
        (len(hashes) == len(POINTS) and all(len(v) == 1 for v in hashes.values()))
        if all(series[d]["points"] for d in SERIES) else NOT_EVALUATED)
    cfg = {d: series_config(series[d]) for d in SERIES}
    if all(cfg.values()):
        keys = sorted(set(cfg[A]) | set(cfg[B]))
        bad = [k for k in keys if cfg[A].get(k) != cfg[B].get(k) and k not in DTYPE_INDUCED_ALLOWLIST]
        add("config identical across dtypes except dtype-induced fields", "config", not bad, bad or None)
    else:
        add("config identical across dtypes except dtype-induced fields", "config", NOT_EVALUATED)
    counts = {s: sum(1 for c in checks if c["state"] == s) for s in (PASSED, FAILED, NOT_RUN, NOT_EVALUATED)}
    return {"checks": checks, "counts": counts, "all_ok": counts[PASSED] == len(checks),
            "failed_categories": sorted({c["category"] for c in checks if c["state"] == FAILED})}


def _intervals(marks: list[dict]) -> list[float]:
    ts = [datetime.strptime("2000-" + m["time"], "%Y-%m-%d %H:%M:%S") for m in marks]
    return [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]


def analysis(series: dict, integ: dict) -> dict:
    meas = {d: {p["row"]["planned_prompt_tokens"]: p for p in series[d]["points"] if p["begin"]["role"] == "measured"}
            for d in SERIES}
    rows = []
    for pt in POINTS:
        a, b = meas[A].get(pt), meas[B].get(pt)
        q = second_chunk_q_len(pt)
        row = {"prompt_tokens": pt, "second_chunk_q_len_derived": q,
               "context_len_derived": MAX_NUM_BATCHED_TOKENS if q else None,
               "stage3c_expected_by_source": bool(q and q > 1),
               "prompt_token_ids_sha256": (b or a or {}).get("row", {}).get("prompt_token_ids_sha256")}
        for d, p in ((A, a), (B, b)):
            key = "bf16" if d == A else "rabit_kv2"
            row[key] = None if p is None else {
                "ttft_ms": p["row"]["ttft_ms"], "tpot_ms": p["row"]["tpot_ms"], "wall_ms": p["row"]["wall_ms"],
                "output_tokens": p["row"]["output_tokens"], "actual_prompt_tokens": p["row"]["prompt_tokens"],
                "jit_lines": p["jit"], "stage3c_markers_observed": p["stage3c"],
                "gpu_memory_used_mib_after": p["row"]["gpu_memory_used_mib_after"]}
        if a and b:
            dlt = b["row"]["ttft_ms"] - a["row"]["ttft_ms"]
            row["rabit_minus_bf16_ttft_ms"] = dlt
            row["rabit_minus_bf16_ttft_ms_per_second_chunk_token"] = dlt / q if q and q > 1 else None
        rows.append(row)
    by_q = {r["second_chunk_q_len_derived"]: r for r in rows}

    def delta(q):
        return (by_q.get(q) or {}).get("rabit_minus_bf16_ttft_ms")

    cond_b = [p for p in series[B]["points"] if p["begin"]["role"] == "conditioning"]
    cond_marks = cond_b[0]["stage3c"] if cond_b else []
    fa_marks = []
    fa = FAILED_ATTEMPT_1 / "failure_analysis.json"
    if fa.is_file():
        fa_marks = json.loads(fa.read_text(encoding="utf-8"))["b32768"]["chunked_prefill_markers"]
    fa_iv = _intervals(fa_marks)
    return {
        "scope": SCOPE,
        "accepted_as_experiment5_evidence": False,
        "all_integrity_passed": integ["all_ok"],
        "integrity_counts": integ["counts"],
        "q_len_note": Q_LEN_NOTE,
        "measured_point_layer_markers_available": MEASURED_POINT_LAYER_MARKERS_AVAILABLE,
        "marker_note": MARKER_NOTE,
        "request_guard_note": REQUEST_GUARD_NOTE,
        "primary_outputs": PRIMARY_OUTPUTS,
        "points": rows,
        "onset": {
            "q_len_1_rabit_minus_bf16_ttft_ms": delta(1),
            "q_len_2_rabit_minus_bf16_ttft_ms": delta(2),
            "note": "Source predicts Stage3C only for q_len > 1 (q_len 1 takes the decode append path).",
        },
        "around_page_boundary": {str(q): delta(q) for q in (31, 32, 33)},
        "conditioning_stage3c_path_verification": {
            "purpose": "verification only: Stage3C path entered and all layers observed; NOT a latency distribution",
            "q_len_context_len": sorted({(m["q_len"], m["context_len"]) for m in cond_marks}),
            "layer_markers_observed": len(cond_marks),
            "layer_markers_expected": NUM_LAYERS,
            "stage3c_path_entered": bool(cond_marks),
            "all_layers_observed": len(cond_marks) == NUM_LAYERS,
            "note": ("Marker intervals of the q_len 32 conditioning request are below the 1 s log-timestamp "
                     "resolution and are deliberately not reported as per-layer timing."),
        },
        "external_frozen_reference_failed_attempt_1": {
            "label": ("EXTERNAL FROZEN REFERENCE evidence from results/mlsys2027/context_scaling/failed_attempt_1/ "
                      "(q_len 16352, context_len 16384, 32K model-limit point). Not produced by this diagnostic and "
                      "never pooled or merged with it."),
            "source": rel(fa), "layer_markers": len(fa_marks), "layer_intervals_s": fa_iv,
            "median_s": statistics.median(fa_iv) if fa_iv else None,
            "min_s": min(fa_iv) if fa_iv else None, "max_s": max(fa_iv) if fa_iv else None},
        "not_claimed": ["a complexity law (one sample per point)", "Experiment 5 results",
                        "an algorithmic limitation of low-bit KV quantization",
                        "per-layer timing of the measured diagnostic points"],
    }


def analyze(text: str, write: bool) -> tuple[dict, dict]:
    ser_lines, gate_lines, top_lines = demux(text)
    series = {d: parse_series(ser_lines[d]) for d in SERIES}
    gate = r5.parse_gate(gate_lines)
    top = parse_top(top_lines)
    integ = integrity(series, gate, top)
    an = analysis(series, integ)
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        GATE_LOG.write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
        for d in SERIES:
            (OUT_DIR / SERIES_LOG[d]).write_text("\n".join(ser_lines[d]) + "\n", encoding="utf-8")
        INTEGRITY.write_text(json.dumps(integ, indent=2, default=str) + "\n", encoding="utf-8")
        ANALYSIS.write_text(json.dumps(an, indent=2, default=str) + "\n", encoding="utf-8")
    return integ, an


def write_manifest(m: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def finalize(m: dict, status: str, failure: dict | None) -> None:
    m["status"], m["completed_utc"] = status, now()
    if failure:
        m.setdefault("failure", failure)
    try:
        assert_protected_paths_clean(f"post-run ({status})")
        m["protected_paths_post_run_status"] = "clean"
    except Exception as exc:  # noqa: BLE001
        m["protected_paths_post_run_status"] = "check_failed"
        m["protected_paths_check_error"] = {"type": type(exc).__name__, "message": str(exc)}
        m["status"] = "failed"
    m["prior_evidence_unchanged"] = prior_evidence_digest() == m["provenance"].get("prior_evidence_sha256_raw", {})
    if not m["prior_evidence_unchanged"]:
        m["status"] = "failed"
        m.setdefault("failure", {"stage": "prior_evidence_modified"})
    write_manifest(m)


def classify(code: int, integ: dict) -> str:
    cats = integ["failed_categories"]
    for stage, cat in (("watchdog_timeout", "watchdog"), ("correctness_gate", "gate"), ("gpu_clean", "gpu_clean"),
                       ("request_cap_or_failure", "request")):
        if cat in cats:
            return stage
    return "modal_nonzero_exit" if code != 0 else "integrity"


def run(m: dict) -> int:
    snap = build_snapshot()
    m["vllm_kvquant_snapshot"] = snap
    m["command"] = build_command()
    write_manifest(m)
    code = stream_command(m["command"], SESSION_LOG, {"S3C_VLLM_SNAPSHOT": snap["path"]})
    m["modal_returncode"] = code
    m["stage"] = "parse"
    integ, an = analyze(SESSION_LOG.read_text(encoding="utf-8", errors="replace"), write=True)
    m.pop("stage")
    m["integrity_counts"] = integ["counts"]
    if code != 0 or not integ["all_ok"]:
        stage = classify(code, integ)
        finalize(m, "failed", {"stage": stage, "failed_categories": integ["failed_categories"]})
        raise SystemExit(f"\nSTAGE3C DIAGNOSTIC STOPPED ({stage}); partial evidence kept in {rel(OUT_DIR)}/.")
    finalize(m, "completed", None)
    print(f"\nSTAGE3C DIAGNOSTIC COMPLETED (diagnostic evidence only). Analysis: {ANALYSIS}")
    return 0


def execute(m: dict) -> int:
    try:
        return run(m)
    except SystemExit:
        raise
    except (Exception, KeyboardInterrupt) as exc:
        stage = "parser_failure" if m.pop("stage", None) == "parse" else "local_runner_exception"
        finalize(m, "failed", {"stage": stage, "type": type(exc).__name__, "message": str(exc)})
        raise SystemExit(f"\nSTAGE3C DIAGNOSTIC RUNNER FAILED: {type(exc).__name__}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    print("RABIT-KV Stage3C cliff DIAGNOSTIC (pre-fix; not Experiment 5 evidence)")
    print(f"Series (fresh engine each): {SERIES}; conditioning request {CONDITIONING_PROMPT} (unmeasured)")
    print("Points (prompt tokens | second-chunk q_len, derived):")
    for p in POINTS:
        print(f"  {p:6d} | {second_chunk_q_len(p) if second_chunk_q_len(p) is not None else 'none (single chunk)'}")
    print(f"Request guard: {REQUEST_GUARD_NOTE}")
    print(f"  (series watchdog {SERIES_TIMEOUT_S}s; gate {GATE_TIMEOUT_S}s; watchdog budget {WATCHDOG_BUDGET_S}s; "
          f"Modal function timeout {MODAL_FUNCTION_TIMEOUT_S}s)")
    print(f"Per-layer markers: measured_point_layer_markers_available = {MEASURED_POINT_LAYER_MARKERS_AVAILABLE}. "
          f"{MARKER_NOTE}")
    print("Primary outputs: " + "; ".join(PRIMARY_OUTPUTS) + ". No complexity law.")
    prov = preflight(args.dry_run)
    print("Preflight OK:", json.dumps({k: prov[k] for k in ("git_head", "rabit_kv2_sha256", "runner_script_sha256",
                                                             "modal_app_sha256", "worker_sha256")}, indent=1))
    print("  equivalence:", json.dumps(prov["equivalence"]))
    print(f"  protected paths: {len(prov['protected_paths'])}; prior evidence files hashed: "
          f"{len(prov['prior_evidence_sha256_raw'])}")
    if prov["uncommitted_diagnostic_files"]:
        print("  WARNING (dry-run only): uncommitted diagnostic files:\n    "
              + prov["uncommitted_diagnostic_files"].replace("\n", "\n    "))
    print("Local command:\n  " + " ".join(build_command()))
    print(f"Outputs: {rel(OUT_DIR)}/ (modal_session.log, correctness_gate.log, {', '.join(SERIES_LOG.values())}, "
          f"manifest.json, integrity_check.json, diagnostic_analysis.json)")
    if args.dry_run:
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    m = {"diagnostic": "Stage3C cliff (pre-fix)", "scope": SCOPE, "series": SERIES, "points": POINTS,
         "conditioning_prompt_tokens": CONDITIONING_PROMPT, "request_cap_s": REQUEST_CAP_S,
         "series_timeout_s": SERIES_TIMEOUT_S, "modal_backstop_s": MODAL_FUNCTION_TIMEOUT_S,
         "request_guard_note": REQUEST_GUARD_NOTE,
         "measured_point_layer_markers_available": MEASURED_POINT_LAYER_MARKERS_AVAILABLE,
         "marker_note": MARKER_NOTE, "primary_outputs": PRIMARY_OUTPUTS,
         "started_utc": now(), "status": "running", "protected_paths_post_run_status": "pending",
         "provenance": prov}
    write_manifest(m)
    return execute(m)


if __name__ == "__main__":
    raise SystemExit(main())
