"""
RABIT-KV MLSys 2027 -- Experiment 6 concurrency CORRECTNESS SMOKE test runner.
Correctness only: proves concurrent multi-request cache state is correct before
any throughput work. NO performance claim is made from this run.

Upstream CLI audit (why this is not `vllm bench throughput`): the CLI accepts
--kv-cache-dtype rabit_kv2 (engine args; this fork's CacheDType includes it,
block size must be a multiple of 32), but its run_vllm() samples with
temperature=1.0 / top_p=1.0 and uses outputs only to count tokens -- it exposes
no deterministic per-request output token IDs. This smoke test therefore uses
vLLM's normal multi-request LLM.generate API; the later throughput sweep can
still use the upstream CLI.

One `modal run` of benchmarks/mlsys2027/exp6_smoke_modal.py: idle baseline ->
frozen correctness gate -> fresh bfloat16 engine -> fresh rabit_kv2 engine
(explicit shared_decode / QUERY_BLOCK from the committed tie-break evidence) ->
post-run clean check. Per engine (exp6_smoke_worker.py): unmeasured warmup; 4
distinct 2048-token TEST prompts each alone (reference); then all 4 in one
concurrent generate call (max_num_seqs 4). Pass criteria, per dtype:
  * request i concurrent output token IDs == request i single-stream output
    token IDs, for i = 0..3 (4/4); prompt hash of request i == test prompt i;
  * the 4 single-stream outputs are pairwise distinct (a swap would be detected);
  * TRUE concurrency: on the engine-core clock max(scheduled_ts) <
    min(last_token_ts) and max(first_token_ts) < min(last_token_ts) (all 4
    active, all 4 decoding simultaneously); the single phase is provably serial
    (last_token_ts[i] < scheduled_ts[i+1]);
  * selector / QB verified; profiling off; no JIT in the single / concurrent
    phases; no OOM; GPU clean before each engine and after the run.
Pre-registered note: vLLM does not guarantee batch-invariant numerics, so a
greedy divergence can in principle arise from batch-shape arithmetic; any
mismatch is reported with its first divergent position and read against the
BF16 control. Mismatches are never tolerated or re-run.

Usage:
    python benchmarks/mlsys2027/run_experiment6_smoke.py --dry-run
    python benchmarks/mlsys2027/run_experiment6_smoke.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment5_attempt2 as a2  # noqa: E402  (committed; accepted-implementation provenance)
import run_stage3c_cliff_diagnostic as rd  # noqa: E402  (committed; AST helpers)
import run_stage3c_shared_decode_benchmark as rsd  # noqa: E402  (committed; tie-break reader, protected paths)
from run_experiment3_deployment import (  # noqa: E402
    FAILED, NOT_EVALUATED, NOT_RUN, PASSED, _function, _module_assign, make_console_encoding_safe, now, rel,
    run_git, sha256, sha256_raw, stream_command,
)

ROOT = rsd.ROOT
HERE = ROOT / "benchmarks" / "mlsys2027"
RUNNER_SCRIPT = Path(__file__).resolve()
MODAL_APP = HERE / "exp6_smoke_modal.py"
WORKER = HERE / "exp6_smoke_worker.py"
OFFLINE_TESTS = HERE / "test_experiment6_smoke.py"
REF_MODAL_APP = rsd.MODAL_APP  # stage3c_shared_decode_bench_modal.py (canonical image + helpers)
EXP5_WORKER = HERE / "exp5_engine_worker.py"
OUT_DIR = ROOT / "results" / "mlsys2027" / "concurrency_smoke"
SESSION_LOG = OUT_DIR / "modal_session.log"
GATE_LOG = OUT_DIR / "correctness_gate.log"
MANIFEST = OUT_DIR / "manifest.json"
INTEGRITY = OUT_DIR / "integrity_check.json"
ANALYSIS = OUT_DIR / "smoke_analysis.json"
ATTEMPT2_DIR = a2.ATTEMPT_DIR
EVIDENCE_DIRS = [*rsd.EVIDENCE_DIRS, *a2.STAGE3C_EVIDENCE, ATTEMPT2_DIR, a2.FAILED_ATTEMPT_1]
PROTECTED_PATHS = [*rsd.PROTECTED_PATHS, *a2.STAGE3C_EVIDENCE, ATTEMPT2_DIR, a2.FAILED_ATTEMPT_1,
                   rsd.RUNNER_SCRIPT, rsd.MODAL_APP, rsd.WORKER, a2.RUNNER_SCRIPT, a2.MODAL_APP, a2.WORKER]
MUST_BE_COMMITTED = [RUNNER_SCRIPT, MODAL_APP, WORKER, OFFLINE_TESTS, rsd.GATE, rsd.WATCHDOG]
DTYPES = ("bfloat16", "rabit_kv2")
PREFIX = {d: f"[smoke{k}:{d}] " for k, d in enumerate(DTYPES, start=1)}
GATE_TIMEOUT_S, SMOKE_TIMEOUT_S, MODAL_FUNCTION_TIMEOUT_S = 600, 1800, 5400
WATCHDOG_BUDGET_S = GATE_TIMEOUT_S + len(DTYPES) * SMOKE_TIMEOUT_S
PROMPT_TOKENS, OUTPUT_TOKENS, CONCURRENCY = 2048, 32, 4
MARKERS = ["EXP6S_WARMUP_BEGIN", "EXP6S_WARMUP_END", "EXP6S_SINGLE_BEGIN", "EXP6S_SINGLE_END",
           "EXP6S_CONCURRENT_BEGIN", "EXP6S_CONCURRENT_END", "EXP6S_WORKER_COMPLETE"]
TAG = re.compile(r"^(EXP6S_[A-Z0-9_]+)=(\{.*\})\s*$")
TOP_TAG = re.compile(r"^(S3C_[A-Z0-9_]+)=(\{.*\})\s*$")
JIT = "Triton kernel JIT compilation during inference"
OOM = ("CUDA out of memory", "OutOfMemoryError")
SCOPE = ("Experiment 6 concurrency CORRECTNESS SMOKE test only (4 concurrent 2048-token requests, 32 greedy "
         "outputs, max_num_seqs 4); NOT a throughput or latency result; no performance claim.")


# ------------------------------------------------------------------ preflight
def assert_protected_paths_clean(context: str) -> None:
    status = run_git("status", "--short", "--", *[rel(p) for p in PROTECTED_PATHS])
    if status:
        raise RuntimeError(f"CRITICAL: protected paths changed ({context}):\n" + status)


def prior_evidence_digest() -> dict:
    out = {}
    for d in EVIDENCE_DIRS:
        if d.is_dir():
            for f in sorted(p for p in d.rglob("*") if p.is_file()):
                out[f"{rel(d)}/{f.relative_to(d).as_posix()}"] = sha256_raw(f)
    return out


def final_config() -> dict:
    fin = rsd.final_query_block(require_committed=True)
    accepted = json.loads(a2.ACCEPTED_SHARED_DECODE_MANIFEST.read_text(encoding="utf-8"))["provenance"]
    return {"impl": a2.FINAL_STAGE3C_IMPL, "query_block": fin["final_query_block"],
            "query_block_source": fin["tiebreak_evidence"],
            "accepted_shared_decode_module_sha256": accepted["shared_decode_module_sha256"],
            "accepted_triton_attn_sha256": accepted["triton_attn_sha256"],
            "accepted_rabit_kv2_sha256": accepted["rabit_kv2_sha256"]}


def verify_equivalence(cfg: dict) -> dict:
    w, w5 = (ast.parse(p.read_text(encoding="utf-8")) for p in (WORKER, EXP5_WORKER))
    m, mr = (ast.parse(p.read_text(encoding="utf-8")) for p in (MODAL_APP, REF_MODAL_APP))
    if rd._const(w, "BASE_ENGINE_KWARGS") != rd._const(w5, "BASE_ENGINE_KWARGS"):
        raise RuntimeError("smoke BASE_ENGINE_KWARGS differ from the frozen Experiment 5 worker")
    if rd._const(w, "SMOKE_OVERRIDES") != {"max_num_seqs": CONCURRENCY}:
        raise RuntimeError("smoke overrides must be exactly max_num_seqs = 4")
    for const, mine in (("PROMPT_TOKENS", PROMPT_TOKENS), ("OUTPUT_TOKENS", OUTPUT_TOKENS),
                        ("CONCURRENCY", CONCURRENCY), ("ALLOWED_KV_CACHE_DTYPES", DTYPES)):
        if rd._const(w, const) != mine and tuple(rd._const(w, const) or ()) != mine:
            raise RuntimeError(f"worker {const} differs")
    passages = rd._const(w, "TEST_PASSAGES"), rd._const(w, "WARMUP_PASSAGES")
    if len(passages[0]) != CONCURRENCY or len(set(passages[0]) | set(passages[1])) != 2 * CONCURRENCY:
        raise RuntimeError("need 4 distinct test passages and 4 distinct warmup passages")
    if "temperature=0.0" not in ast.unparse(_function(w, "main")) or "collective_rpc" in ast.unparse(w):
        raise RuntimeError("smoke must be greedy and must not use engine RPC")
    if ast.dump(_module_assign(m, "image")) != ast.dump(_module_assign(mr, "image")):
        raise RuntimeError("smoke Modal image differs from the canonical image")
    for fn in ("_emit", "_sha256_file", "_gpu_query", "_compute_apps", "_gpu_state", "_require_clean", "_run_guarded"):
        if ast.dump(_function(m, fn)) != ast.dump(_function(mr, fn)):
            raise RuntimeError(f"Modal helper {fn}() differs from the committed benchmark app")
    for const, mine in (("GATE_TIMEOUT_S", GATE_TIMEOUT_S), ("SMOKE_TIMEOUT_S", SMOKE_TIMEOUT_S),
                        ("GPU_CLEAN_TOLERANCE_MIB", 256), ("EXPECTED_RABIT_SHA256_LF", a2.r5.EXPECTED_RABIT_SHA256_LF)):
        if rd._const(m, const) != mine:
            raise RuntimeError(f"Modal {const} differs")
    backstop = next(ast.literal_eval(kw.value) for dec in _function(m, "smoke").decorator_list
                    for kw in getattr(dec, "keywords", []) if kw.arg == "timeout")
    if not (backstop == MODAL_FUNCTION_TIMEOUT_S > WATCHDOG_BUDGET_S):
        raise RuntimeError(f"Modal backstop {backstop} must exceed the watchdog budget {WATCHDOG_BUDGET_S}")
    ops = ROOT / "vllm-kvquant" / "vllm" / "v1" / "attention"
    if not (sha256(ops / "ops" / "rabit_kv2_stage3c_shared_decode.py") == cfg["accepted_shared_decode_module_sha256"]
            and sha256(ops / "backends" / "triton_attn.py") == cfg["accepted_triton_attn_sha256"]
            and sha256(ops / "ops" / "rabit_kv2.py") == cfg["accepted_rabit_kv2_sha256"]):
        raise RuntimeError("shared_decode / triton_attn / rabit_kv2 differ from the accepted implementation")
    return {"engine_kwargs_equal_frozen_exp5_except_max_num_seqs_4": True, "greedy": True,
            "image_and_helpers_equal_committed_benchmark_app": True, "accepted_implementation_sources": True,
            "watchdog_budget_s": WATCHDOG_BUDGET_S, "modal_backstop_s": backstop}


def preflight(dry_run: bool) -> dict:
    assert_protected_paths_clean("preflight")
    cfg = final_config()
    eq = verify_equivalence(cfg)
    uncommitted = run_git("status", "--short", "--", *[rel(p) for p in MUST_BE_COMMITTED])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: smoke code has uncommitted changes:\n" + uncommitted)
    leftovers = sorted(p.name for p in OUT_DIR.iterdir()) if OUT_DIR.is_dir() else []
    if leftovers and not dry_run:
        raise RuntimeError(f"Refusing to run: {rel(OUT_DIR)} is not empty ({leftovers})")
    return {"git_head": run_git("rev-parse", "HEAD"), "vllm_kvquant_tree": run_git("rev-parse", "HEAD:vllm-kvquant"),
            "final_config": cfg, "equivalence": eq, "runner_script_sha256": sha256(RUNNER_SCRIPT),
            "modal_app_sha256": sha256(MODAL_APP), "worker_sha256": sha256(WORKER),
            "protected_paths": [rel(p) for p in PROTECTED_PATHS], "prior_evidence_sha256_raw": prior_evidence_digest(),
            "uncommitted_files": uncommitted or None, "existing_output_files": leftovers or None}


def build_command(cfg: dict) -> list[str]:
    return [sys.executable, "-m", "modal", "run", str(MODAL_APP), "--stage3c-impl", cfg["impl"],
            "--query-block", str(cfg["query_block"])]


# ------------------------------------------------------------------ parsing
def demux(text: str):
    cells, gate, top = {d: [] for d in DTYPES}, [], []
    for line in text.splitlines():
        if line.startswith("[gate] "):
            gate.append(line[len("[gate] "):])
            continue
        for d, p in PREFIX.items():
            if line.startswith(p):
                cells[d].append(line[len(p):])
                break
        else:
            top.append(line)
    return cells, gate, top


def parse_worker(lines: list[str]) -> dict:
    out = {"tags": {}, "single": [], "concurrent": [], "markers": [], "jit": {}, "oom": 0, "malformed": []}
    phase = "init"
    for line in lines:
        s = line.strip()
        if s in MARKERS:
            out["markers"].append(s)
            phase = s.split("_")[1].lower() if s.endswith("_BEGIN") else "between"
            continue
        if s.startswith("EXP6S_") and "={" in s:
            m = TAG.match(s)
            if not m:
                out["malformed"].append(s[:200])
                continue
            tag, payload = m.group(1), json.loads(m.group(2))
            if tag == "EXP6S_SINGLE":
                out["single"].append(payload)
            elif tag == "EXP6S_CONCURRENT":
                out["concurrent"].append(payload)
            elif tag == "EXP6S_GPU_MEMORY":
                out["tags"].setdefault("gpu_memory", {})[payload["phase"]] = payload["memory_used_mib"]
            else:
                out["tags"][tag] = payload
            continue
        if JIT in s:
            out["jit"][phase] = out["jit"].get(phase, 0) + 1
        out["oom"] += any(o in s for o in OOM)
    return out


def parse_top(lines: list[str]) -> dict:
    out = {"pre": {}, "exit": {}, "start": {}, "proc": {}, "timeouts": [], "complete": False}
    for line in lines:
        s = line.strip()
        if s == "S3C_SMOKE_COMPLETE":
            out["complete"] = True
            continue
        m = TOP_TAG.match(s)
        if not m:
            continue
        tag, p = m.group(1), json.loads(m.group(2))
        if tag == "S3C_PRE_LEG_GPU_STATE":
            out["pre"][p["leg"]] = p
        elif tag == "S3C_SERIES_START":
            out["start"][p["series"]] = p
        elif tag == "S3C_SERIES_EXIT":
            out["exit"][p["series"]] = p
        elif tag == "S3C_PROCESS_EXIT":
            out["proc"][p["label"]] = p
        elif tag in ("S3C_WATCHDOG_TIMEOUT", "S3C_STOPPED"):
            out["timeouts"].append({tag: p})
        else:
            out[tag] = p
    return out


def overlap(rows: list[dict]) -> dict:
    ts = [(r.get("scheduled_ts"), r.get("first_token_ts"), r.get("last_token_ts")) for r in rows]
    if len(rows) != CONCURRENCY or any(not all(isinstance(x, (int, float)) and x > 0 for x in t) for t in ts):
        return {"evaluable": False}
    sched, first, last = zip(*ts)
    return {"evaluable": True, "all_active_simultaneously": max(sched) < min(last),
            "all_decoding_simultaneously": max(first) < min(last),
            "common_active_window_s": min(last) - max(sched), "common_decode_window_s": min(last) - max(first),
            "serial_in_order": all(last[i] < sched[i + 1] for i in range(len(rows) - 1))}


def first_divergence(a: list[int], b: list[int]):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(p: dict) -> dict:
    wl = p["tags"].get("EXP6S_WORKLOAD", {})
    tests = wl.get("test_prompt_sha256", [])
    single, conc = p["single"], p["concurrent"]
    per = []
    for i in range(CONCURRENCY):
        s = next((r for r in single if r["i"] == i), None)
        c = next((r for r in conc if r["i"] == i), None)
        ok = bool(s and c) and s["output_token_ids"] == c["output_token_ids"] \
            and s["output_token_ids_sha256"] == c["output_token_ids_sha256"]
        per.append({"i": i, "prompt_sha256": tests[i] if i < len(tests) else None,
                    "single_prompt_matches": bool(s) and s["prompt_token_ids_sha256"] == (tests[i] if i < len(tests) else None),
                    "concurrent_prompt_matches": bool(c) and c["prompt_token_ids_sha256"] == (tests[i] if i < len(tests) else None),
                    "single_output_sha256": s and s["output_token_ids_sha256"],
                    "concurrent_output_sha256": c and c["output_token_ids_sha256"], "equal": ok,
                    "first_divergent_position": None if ok or not (s and c) else
                    first_divergence(s["output_token_ids"], c["output_token_ids"])})
    return {"per_request": per, "equal_count": sum(x["equal"] for x in per),
            "single_outputs_pairwise_distinct": len({x["single_output_sha256"] for x in per}) == CONCURRENCY,
            "concurrent_overlap": overlap(sorted(conc, key=lambda r: r["i"])),
            "single_overlap": overlap(sorted(single, key=lambda r: r["i"]))}


def integrity(cells: dict, gate: dict, top: dict, cfg: dict) -> dict:
    checks = []

    def add(name, cat, state, observed=None):
        if isinstance(state, bool):
            state = PASSED if state else FAILED
        checks.append({"check": name, "category": cat, "state": state, "observed": observed})

    env = top.get("S3C_ENVIRONMENT", {})
    add("exactly one H100", "environment",
        (len(env.get("gpus", [])) == 1 and "H100" in env["gpus"][0].get("name", "")) if env else NOT_EVALUATED)
    add("rabit_kv2.py in image is frozen", "environment",
        env.get("rabit_kv2_sha256_lf") == a2.r5.EXPECTED_RABIT_SHA256_LF if env else NOT_EVALUATED)
    add("gate passed", "gate", (top.get("S3C_GATE_EXIT", {}).get("returncode") == 0
                                and (gate.get("result") or {}).get("passed") is True and gate.get("pytest_exit") == 0)
        if "S3C_GATE_START" in top else NOT_RUN)
    add("smoke completed (both engines), no watchdog / stop", "completion", top["complete"] and not top["timeouts"],
        top["timeouts"] or None)
    base = top.get("S3C_GPU_BASELINE", {})
    for label in [f"smoke_{d}" for d in DTYPES] + ["post_run"]:
        pre = top["pre"].get(label)
        add(f"GPU clean {'after the run' if label == 'post_run' else 'before ' + label}", "gpu_clean",
            a2.r5.gpu_leg_clean(pre, base) if pre else NOT_RUN)
    prompts = {}
    for d in DTYPES:
        p = cells[d]
        t = p["tags"]
        started = f"smoke_{d}" in top["start"]

        def chk(name, cat, ok, observed=None, _d=d, _s=started):
            add(f"{_d}: {name}", cat, ok if _s else NOT_RUN, observed if _s else None)

        proc = top["proc"].get(f"smoke_{d}") or {}
        chk("process exit 0, not timed out, group reaped, all markers", "completion",
            proc.get("returncode") == 0 and proc.get("timed_out") is False and not proc.get("group_processes_remaining")
            and p["markers"] == MARKERS and not p["malformed"], {"markers": p["markers"], "malformed": p["malformed"]})
        s3 = t.get("EXP6S_STAGE_IMPL") or {}
        if d == "rabit_kv2":
            ok = (s3.get("applicable") is True and s3.get("requested_impl") == s3.get("effective_impl") == cfg["impl"]
                  and s3.get("requested_query_block") == s3.get("effective_query_block") == cfg["query_block"]
                  and s3.get("env") == {"VLLM_RABIT2_STAGE3C_IMPL": cfg["impl"],
                                        "VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK": str(cfg["query_block"])}
                  and s3.get("shared_decode_module_sha256") == cfg["accepted_shared_decode_module_sha256"])
        else:
            ok = s3.get("applicable") is False and not any((s3.get("env") or {}).values())
        chk("Stage3C selection (rabit: shared_decode / QB requested == effective, accepted SHA; bf16: none)",
            "stage3c", ok, s3 or None)
        chk("profiling off", "stage3c", bool(s3) and all(v in (None, "0") for v in (s3.get("profiling_env") or {}).values()))
        req = t.get("EXP6S_REQUESTED_ENGINE_KWARGS", {})
        base_kw = rd._const(ast.parse(EXP5_WORKER.read_text(encoding="utf-8")), "BASE_ENGINE_KWARGS")
        chk("requested engine kwargs = frozen Exp5 kwargs with max_num_seqs 4", "config",
            {k: v for k, v in req.items() if k != "model"} == {**base_kw, "max_num_seqs": CONCURRENCY,
                                                                "kv_cache_dtype": d})
        eff = t.get("EXP6S_EFFECTIVE_ENGINE_CONFIG", {})
        chk("effective max_num_seqs 4, block 32, eager, Triton, no prefix cache", "config",
            eff.get("max_num_seqs") == CONCURRENCY and eff.get("block_size") == 32 and eff.get("enforce_eager") is True
            and str(eff.get("attention_backend", "")).endswith("TRITON_ATTN") and eff.get("enable_prefix_caching") is False
            and eff.get("log_stats") is True, eff or None)
        kv = t.get("EXP6S_KV_DTYPE", {})
        chk("resolved KV dtype", "kv_dtype", kv.get("requested_kv_cache_dtype") == d == kv.get("engine_cache_dtype"), kv)
        cap = t.get("EXP6S_CAPACITY", {})
        chk("allocator capacity recorded (num_gpu_blocks x block_size)", "capacity",
            bool(cap) and cap["capacity_tokens"] == cap["num_gpu_blocks"] * cap["block_size"], cap or None)
        wl = t.get("EXP6S_WORKLOAD", {})
        prompts[d] = wl.get("test_prompt_sha256")
        chk("workload: 2048-token prompts, 32 greedy outputs, 4 distinct test prompts", "workload",
            wl.get("prompt_tokens") == PROMPT_TOKENS and wl.get("output_tokens") == OUTPUT_TOKENS
            and wl.get("temperature") == 0.0 and wl.get("ignore_eos") is True
            and len(set(wl.get("test_prompt_sha256") or [])) == CONCURRENCY
            and not set(wl.get("test_prompt_sha256") or []) & set(wl.get("warmup_prompt_sha256") or []), wl or None)
        rows = p["single"] + p["concurrent"]
        chk("single phase: requests 0..3 in order; concurrent phase: requests 0..3", "workload",
            [r["i"] for r in p["single"]] == list(range(CONCURRENCY))
            and [r["i"] for r in p["concurrent"]] == list(range(CONCURRENCY)))
        chk("every request: exactly 2048 prompt / 32 output tokens", "workload",
            bool(rows) and all(r["prompt_tokens"] == PROMPT_TOKENS and r["output_tokens"] == OUTPUT_TOKENS
                               and len(r["output_token_ids"]) == OUTPUT_TOKENS for r in rows))
        cmp = compare(p)
        chk("request i uses test prompt i (single and concurrent)", "workload",
            all(x["single_prompt_matches"] and x["concurrent_prompt_matches"] for x in cmp["per_request"]))
        chk("the 4 single-stream outputs are pairwise distinct (a request swap would be detectable)", "design",
            cmp["single_outputs_pairwise_distinct"])
        co, so = cmp["concurrent_overlap"], cmp["single_overlap"]
        chk("TRUE concurrency: all 4 requests active AND decoding simultaneously (engine-core clock)", "concurrency",
            co.get("evaluable") is True and co["all_active_simultaneously"] and co["all_decoding_simultaneously"], co)
        chk("single phase was serial (control for the overlap metric)", "concurrency",
            so.get("evaluable") is True and so["serial_in_order"] and not so["all_active_simultaneously"], so)
        chk(f"per-request output equality concurrent vs single-stream: {cmp['equal_count']}/{CONCURRENCY}",
            "equality", cmp["equal_count"] == CONCURRENCY,
            [{k: x[k] for k in ("i", "equal", "first_divergent_position")} for x in cmp["per_request"]])
        chk("no Triton JIT in the single / concurrent phases", "jit",
            p["jit"].get("single", 0) == 0 and p["jit"].get("concurrent", 0) == 0, p["jit"] or None)
        chk("no OOM", "request", p["oom"] == 0)
    add("identical test prompts and order for both dtypes", "workload",
        prompts.get("bfloat16") == prompts.get("rabit_kv2") and bool(prompts.get("bfloat16")) if len(prompts) == 2
        else NOT_EVALUATED)
    counts = {s: sum(1 for c in checks if c["state"] == s) for s in (PASSED, FAILED, NOT_RUN, NOT_EVALUATED)}
    return {"checks": checks, "counts": counts, "all_ok": counts[PASSED] == len(checks),
            "failed_categories": sorted({c["category"] for c in checks if c["state"] == FAILED})}


def analysis(cells: dict, integ: dict, cfg: dict) -> dict:
    out = {"scope": SCOPE, "performance_claim": False, "all_integrity_passed": integ["all_ok"],
           "integrity_counts": integ["counts"], "final_rabit_configuration": cfg,
           "upstream_cli_audit": {
               "vllm_bench_throughput_accepts_rabit_kv2": True,
               "reason_not_used_for_correctness": ("run_vllm() samples with temperature=1.0 / top_p=1.0 and uses "
                                                   "outputs only for token counting; no deterministic per-request "
                                                   "output token IDs are exposed"),
               "block_size_requirement": "rabit_kv2 requires --block-size to be a multiple of 32"},
           "per_dtype": {}}
    for d in DTYPES:
        p = cells[d]
        cmp = compare(p)
        out["per_dtype"][d] = {"stage3c": p["tags"].get("EXP6S_STAGE_IMPL"), "capacity": p["tags"].get("EXP6S_CAPACITY"),
                               "gpu_memory_mib": p["tags"].get("gpu_memory"), "jit_lines_by_phase": p["jit"],
                               **cmp}
    return out


def analyze(text: str, cfg: dict, write: bool):
    cells_lines, gate_lines, top_lines = demux(text)
    cells = {d: parse_worker(cells_lines[d]) for d in DTYPES}
    gate = a2.r5.parse_gate(gate_lines)
    top = parse_top(top_lines)
    integ = integrity(cells, gate, top, cfg)
    an = analysis(cells, integ, cfg)
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        GATE_LOG.write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
        for d in DTYPES:
            (OUT_DIR / f"smoke_{d}.log").write_text("\n".join(cells_lines[d]) + "\n", encoding="utf-8")
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


def main(argv: list[str] | None = None) -> int:
    make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    print("RABIT-KV MLSys 2027 -- Experiment 6 concurrency CORRECTNESS SMOKE (no performance claim)")
    prov = preflight(a.dry_run)
    cfg = prov["final_config"]
    print("Preflight OK:", json.dumps({k: prov[k] for k in ("git_head", "vllm_kvquant_tree")}))
    print("  final RABIT:", json.dumps({k: cfg[k] for k in ("impl", "query_block", "query_block_source")}))
    print("  equivalence:", json.dumps(prov["equivalence"]))
    if prov["uncommitted_files"]:
        print("  WARNING (dry-run only): uncommitted:\n    " + prov["uncommitted_files"].replace("\n", "\n    "))
    print("Local command:\n  " + " ".join(build_command(cfg)))
    if a.dry_run:
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    m = {"experiment": "Experiment 6 concurrency correctness smoke", "scope": SCOPE, "started_utc": now(),
         "status": "running", "protected_paths_post_run_status": "pending", "provenance": prov}
    write_manifest(m)
    try:
        snap = Path(tempfile.mkdtemp(prefix="exp6s_vllm_snapshot_")) / "vllm_kvquant_snapshot.zip"
        run_git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(snap), "HEAD:vllm-kvquant")
        m["vllm_kvquant_snapshot"] = {"path": str(snap), "sha256": sha256_raw(snap), "bytes": snap.stat().st_size}
        m["command"] = build_command(cfg)
        write_manifest(m)
        code = stream_command(m["command"], SESSION_LOG, {"EXP6S_VLLM_SNAPSHOT": str(snap)})
        m["modal_returncode"] = code
        integ, _ = analyze(SESSION_LOG.read_text(encoding="utf-8", errors="replace"), cfg, write=True)
        m["integrity_counts"] = integ["counts"]
        if code != 0 or not integ["all_ok"]:
            finalize(m, "failed", {"stage": (integ["failed_categories"] or ["modal_nonzero_exit"])[0],
                                   "failed_categories": integ["failed_categories"]})
            raise SystemExit(f"\nEXP6 SMOKE STOPPED; evidence kept in {rel(OUT_DIR)}/ (integrity {integ['counts']}).")
        finalize(m, "completed", None)
        print(f"\nEXP6 SMOKE COMPLETED. Analysis: {ANALYSIS}")
        return 0
    except SystemExit:
        raise
    except (Exception, KeyboardInterrupt) as exc:
        finalize(m, "failed", {"stage": "local_runner_exception", "type": type(exc).__name__, "message": str(exc)})
        raise SystemExit(f"\nEXP6 SMOKE RUNNER FAILED: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
