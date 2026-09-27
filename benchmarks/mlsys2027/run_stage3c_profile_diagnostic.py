"""
RABIT-KV Stage3C COMPONENT PROFILE diagnostic runner (prep; review before any
run). Component attribution only -- NOT a latency benchmark, NOT paper
performance evidence, NOT Experiment 5 evidence.

One `modal run` of benchmarks/mlsys2027/stage3c_profile_modal.py:
  idle baseline -> frozen correctness gate -> tile32 correctness tests ->
  profiler tests (OFF = no-op; ON = bit-exact to OFF) -> two series, each one
  fresh engine with VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE=1, one unmeasured
  16416-token conditioning request and ONE profiled request per point:
      rabit_reference_profile = rabit_kv2, VLLM_RABIT2_STAGE3C_IMPL=reference
      rabit_tile32_profile    = rabit_kv2, VLLM_RABIT2_STAGE3C_IMPL=tile32
Points (prompt tokens -> second-chunk q_len), same prompt construction as the
tile32 benchmark: 16416->32, 16896->512, 18432->2048 (+ 24576->8192 only with
--include-8192). No BF16 series, no 16352 point, no Experiment 5.

The profiler synchronizes the device around every component; the profiled
requests' TTFT / wall are NEVER latency results. HOST (perf_counter issue time)
and GPU (CUDA-event device time) are reported separately; see
stage3c_profile_analysis.py for the categories and which boundaries remain
combined.

Usage:
    python benchmarks/mlsys2027/run_stage3c_profile_diagnostic.py --dry-run [--include-8192]
    python benchmarks/mlsys2027/run_stage3c_profile_diagnostic.py [--correctness-only] [--include-8192]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment3_deployment as r3  # noqa: E402  (frozen; helpers only)
import run_experiment5_context_scaling as r5  # noqa: E402  (frozen; helpers only)
import run_stage3c_cliff_diagnostic as rd  # noqa: E402  (committed diagnostic; helpers only)
import run_stage3c_tile32_benchmark as rt  # noqa: E402  (committed benchmark; helpers only)
import stage3c_profile_analysis as pa  # noqa: E402
from run_experiment3_deployment import (  # noqa: E402
    FAILED, NOT_EVALUATED, NOT_RUN, PASSED, _function, _module_assign, make_console_encoding_safe, now, rel,
    run_git, sha256, sha256_raw, stream_command,
)

ROOT = r3.ROOT
HERE = ROOT / "benchmarks" / "mlsys2027"
RUNNER_SCRIPT = Path(__file__).resolve()
MODAL_APP = HERE / "stage3c_profile_modal.py"
WORKER = HERE / "stage3c_profile_worker.py"
ANALYSIS_MODULE = HERE / "stage3c_profile_analysis.py"
OFFLINE_TESTS = HERE / "test_stage3c_profile_analysis.py"
BENCH_WORKER = rt.WORKER
BENCH_MODAL_APP = rt.MODAL_APP
GATE, WATCHDOG = rt.GATE, rt.WATCHDOG
VK = ROOT / "vllm-kvquant"
PROFILE_MODULE = VK / "vllm" / "v1" / "attention" / "ops" / "rabit_kv2_stage3c_profile.py"
PROFILE_TESTS = VK / "tests" / "quantization" / "test_rabit2_stage3c_profile.py"
TRITON_ATTN = rt.TRITON_ATTN
TILE32_MODULE, TILE32_TESTS = rt.TILE32_MODULE, rt.TILE32_TESTS
RABIT_KV2 = r3.RABIT_KV2
EXPECTED_RABIT_SHA256_LF = r3.EXPECTED_RABIT_SHA256_LF

BENCH_DIR = rt.OUT_DIR  # frozen tile32 benchmark evidence (external unprofiled reference)
BENCH_SUMMARY = BENCH_DIR / "summary.json"
BENCH_MANIFEST = BENCH_DIR / "manifest.json"
PROFILE_DIR = ROOT / "results" / "mlsys2027" / "diagnostics" / "stage3c_component_profile"
CORRECTNESS_DIR = ROOT / "results" / "mlsys2027" / "diagnostics" / "stage3c_component_profile_correctness"


def set_out_dir(d: Path) -> None:
    """Output location: PROFILE_DIR for the profiling run, CORRECTNESS_DIR for --correctness-only."""
    global OUT_DIR, SESSION_LOG, GATE_LOG, TILE32_TESTS_LOG, PROFILE_TESTS_LOG, MANIFEST, INTEGRITY, ANALYSIS
    OUT_DIR = d
    SESSION_LOG, GATE_LOG = d / "modal_session.log", d / "correctness_gate.log"
    TILE32_TESTS_LOG, PROFILE_TESTS_LOG = d / "tile32_correctness_tests.log", d / "profile_tests.log"
    MANIFEST, INTEGRITY, ANALYSIS = d / "manifest.json", d / "integrity_check.json", d / "profile_analysis.json"


set_out_dir(PROFILE_DIR)

EVIDENCE_DIRS = [*rt.EVIDENCE_DIRS, BENCH_DIR]
PROTECTED_PATHS = [*rt.PROTECTED_PATHS, BENCH_DIR, rt.RUNNER_SCRIPT, BENCH_MODAL_APP, BENCH_WORKER,
                   HERE / "build_stage3c_tile32_benchmark_summary.py",
                   HERE / "validate_stage3c_tile32_benchmark_summary.py"]
# Code under test (inside the protected vllm-kvquant tree or new here); a real run refuses unless committed.
MUST_BE_COMMITTED = [RUNNER_SCRIPT, MODAL_APP, WORKER, ANALYSIS_MODULE, OFFLINE_TESTS, GATE, WATCHDOG,
                     PROFILE_MODULE, PROFILE_TESTS, TRITON_ATTN]
BASELINE_REF_KEY = "git_head"  # triton_attn.py baseline = the commit the tile32 benchmark ran from

SERIES = [("rabit_reference_profile", "rabit_kv2", "reference"),
          ("rabit_tile32_profile", "rabit_kv2", "tile32")]
SERIES_LOG = {"rabit_reference_profile": "rabit_reference_profile_series.log",
              "rabit_tile32_profile": "rabit_tile32_profile_series.log"}
FIRST_CHUNK = 16384
CONDITIONING_PROMPT = 16416
Q_LENS = [32, 512, 2048]
OPTIONAL_Q_LEN = 8192
NUM_LAYERS = 32
REQUEST_CAP_S = 600
SERIES_TIMEOUT_S = 3600
TILE32_TEST_TIMEOUT_S = 1800
PROFILE_TEST_TIMEOUT_S = 1800
GATE_TIMEOUT_S = 600
WATCHDOG_BUDGET_S = GATE_TIMEOUT_S + TILE32_TEST_TIMEOUT_S + PROFILE_TEST_TIMEOUT_S + len(SERIES) * SERIES_TIMEOUT_S
MODAL_FUNCTION_TIMEOUT_S = 12600
REQUEST_GUARD_NOTE = rd.REQUEST_GUARD_NOTE
SCOPE = ("Stage3C COMPONENT PROFILE diagnostic only; NOT a latency benchmark, NOT paper performance evidence, NOT "
         "Experiment 5 evidence. The device is synchronized around every component; profiled TTFT/wall are not "
         "latency. One profiled request per implementation per point.")


def q_lens(include_8192: bool) -> list[int]:
    return Q_LENS + ([OPTIONAL_Q_LEN] if include_8192 else [])


def points(include_8192: bool) -> list[int]:
    return [FIRST_CHUNK + q for q in q_lens(include_8192)]


# ------------------------------------------------------------------ preflight
def assert_protected_paths_clean(context: str) -> None:
    under_test = {rel(p) for p in MUST_BE_COMMITTED}
    status = "\n".join(line for line in run_git("status", "--short", "--", *[rel(p) for p in PROTECTED_PATHS])
                       .splitlines() if line.split(maxsplit=1)[-1].strip() not in under_test)
    if status:
        raise RuntimeError(f"CRITICAL: protected paths changed ({context}):\n" + status)


def prior_evidence_digest() -> dict:
    out = {}
    for d in EVIDENCE_DIRS:
        if d.is_dir():
            for f in sorted(p for p in d.rglob("*") if p.is_file()):
                out[f"{rel(d)}/{f.relative_to(d).as_posix()}"] = sha256_raw(f)
    return out


def _git_show(ref: str, path: Path) -> str:
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:{rel(path)}"], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def triton_attn_unwrapped_equals(baseline_src: str, current_src: str) -> bool:
    """current == baseline after removing exactly the profiling import and the `with` wrapper."""
    cur, base = ast.parse(current_src), ast.parse(baseline_src)
    imports = [n for n in cur.body if isinstance(n, ast.ImportFrom)
               and n.module == "vllm.v1.attention.ops.rabit_kv2_stage3c_profile"]
    if len(imports) != 1 or [a.name for a in imports[0].names] != ["rabit2_stage3c_profile_scope"]:
        return False
    cur.body.remove(imports[0])
    withs = [w for w in ast.walk(cur) if isinstance(w, ast.With)
             and any(isinstance(i.context_expr, ast.Call) and getattr(i.context_expr.func, "id", None)
                     == "rabit2_stage3c_profile_scope" for i in w.items)]
    if len(withs) != 1:
        return False
    w = withs[0]
    call = w.items[0].context_expr
    if len(w.items) != 1 or w.items[0].optional_vars is not None or call.keywords \
            or [getattr(a, "id", None) for a in call.args] != ["q_len", "context_len"]:
        return False
    for parent in ast.walk(cur):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(parent, field, None)
            if isinstance(stmts, list) and w in stmts:
                i = stmts.index(w)
                stmts[i:i + 1] = w.body
    return ast.dump(cur) == ast.dump(base)


def verify_equivalence(bench_manifest: dict) -> dict:
    w, wb = (ast.parse(p.read_text(encoding="utf-8")) for p in (WORKER, BENCH_WORKER))
    m, mb = (ast.parse(p.read_text(encoding="utf-8")) for p in (MODAL_APP, BENCH_MODAL_APP))
    if rd._const(w, "BASE_ENGINE_KWARGS") != rd._const(wb, "BASE_ENGINE_KWARGS"):
        raise RuntimeError("BASE_ENGINE_KWARGS differ from the tile32 benchmark worker")
    if rd._const(w, "BASE_ENGINE_KWARGS")["max_num_batched_tokens"] != FIRST_CHUNK:
        raise RuntimeError("max_num_batched_tokens changed")
    if tuple(rd._const(w, "ALLOWED_KV_CACHE_DTYPES")) != ("bfloat16", "rabit_kv2"):
        raise RuntimeError("worker dtypes changed")
    main_w, main_b = _function(w, "main"), _function(wb, "main")
    if ast.dump(rd._nested(main_w, "one")) != ast.dump(rd._nested(main_b, "one")):
        raise RuntimeError("timed region one() differs from the tile32 benchmark worker")
    for target in ("tok", "bos", "filler", "sp", "prompt", "kwargs", "plan"):
        if rd._assigns(main_w, target) != rd._assigns(main_b, target):
            raise RuntimeError(f"'{target} = ...' differs from the tile32 benchmark worker")
    src = ast.unparse(main_w)
    if "collective_rpc" in src or "VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE" not in src \
            or "VLLM_RABIT2_STAGE3C_IMPL" not in src:
        raise RuntimeError("worker must export the impl and component-profile flags and must not use engine RPC")
    if ast.dump(_module_assign(m, "image")) != ast.dump(_module_assign(mb, "image")):
        raise RuntimeError("profiling Modal image differs from the benchmark (canonical) image")
    for fn in ("_emit", "_sha256_file", "_gpu_query", "_compute_apps", "_gpu_state", "_require_clean", "_run_guarded"):
        if ast.dump(_function(mb, fn)) != ast.dump(_function(m, fn)):
            raise RuntimeError(f"Modal helper {fn}() differs from the tile32 benchmark app")
    for const, mine in (("REQUEST_CAP_S", REQUEST_CAP_S), ("SERIES_TIMEOUT_S", SERIES_TIMEOUT_S),
                        ("TILE32_TEST_TIMEOUT_S", TILE32_TEST_TIMEOUT_S),
                        ("PROFILE_TEST_TIMEOUT_S", PROFILE_TEST_TIMEOUT_S), ("GATE_TIMEOUT_S", GATE_TIMEOUT_S),
                        ("GPU_CLEAN_TOLERANCE_MIB", 256), ("EXPECTED_RABIT_SHA256_LF", EXPECTED_RABIT_SHA256_LF)):
        if rd._const(m, const) != mine:
            raise RuntimeError(f"Modal {const} differs")
    if SERIES_TIMEOUT_S < (1 + len(points(True))) * REQUEST_CAP_S:
        raise RuntimeError("series timeout below (conditioning + points) x request cap")
    backstop = None
    for dec in _function(m, "benchmark").decorator_list:
        for kw in getattr(dec, "keywords", []):
            if kw.arg == "timeout":
                backstop = ast.literal_eval(kw.value)
    if not (backstop == MODAL_FUNCTION_TIMEOUT_S > WATCHDOG_BUDGET_S):
        raise RuntimeError(f"Modal backstop {backstop} must exceed the watchdog budget {WATCHDOG_BUDGET_S}")
    base_ref = bench_manifest["provenance"][BASELINE_REF_KEY]
    if not triton_attn_unwrapped_equals(_git_show(base_ref, TRITON_ATTN), TRITON_ATTN.read_text(encoding="utf-8")):
        raise RuntimeError("triton_attn.py differs from the benchmarked version beyond the profiling scope wrapper")
    if sha256(TILE32_MODULE) != bench_manifest["provenance"]["tile32_module_sha256"] \
            or sha256(TILE32_TESTS) != bench_manifest["provenance"]["tile32_tests_sha256"]:
        raise RuntimeError("tile32 module / tests differ from the benchmarked version (selector must be unchanged)")
    return {"engine_kwargs_equal_tile32_benchmark": True, "timed_region_equal_tile32_benchmark": True,
            "helpers_equal_tile32_benchmark": True, "image_equal_tile32_benchmark": True,
            "triton_attn_equal_benchmarked_modulo_profile_scope": True, "triton_attn_baseline_ref": base_ref,
            "tile32_module_and_tests_unchanged": True,
            "watchdog_budget_s": WATCHDOG_BUDGET_S, "modal_backstop_s": backstop}


def preflight(dry_run: bool) -> dict:
    assert_protected_paths_clean("preflight")
    if sha256(RABIT_KV2) != EXPECTED_RABIT_SHA256_LF:
        raise RuntimeError("rabit_kv2.py is not the frozen committed content")
    bench_manifest = json.loads(BENCH_MANIFEST.read_text(encoding="utf-8"))
    eq = verify_equivalence(bench_manifest)
    uncommitted = run_git("status", "--short", "--", *[rel(p) for p in MUST_BE_COMMITTED])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: profiling code has uncommitted changes:\n" + uncommitted)
    leftovers = sorted(p.name for p in OUT_DIR.iterdir()) if OUT_DIR.is_dir() else []
    if leftovers and not dry_run:
        raise RuntimeError(f"Refusing to run: {rel(OUT_DIR)} is not empty ({leftovers})")
    return {"git_head": run_git("rev-parse", "HEAD"), "vllm_kvquant_tree": run_git("rev-parse", "HEAD:vllm-kvquant"),
            "rabit_kv2_sha256": sha256(RABIT_KV2), "tile32_module_sha256": sha256(TILE32_MODULE),
            "profile_module_sha256": sha256(PROFILE_MODULE), "profile_tests_sha256": sha256(PROFILE_TESTS),
            "triton_attn_sha256": sha256(TRITON_ATTN), "runner_script_sha256": sha256(RUNNER_SCRIPT),
            "modal_app_sha256": sha256(MODAL_APP), "worker_sha256": sha256(WORKER),
            "analysis_module_sha256": sha256(ANALYSIS_MODULE), "equivalence": eq,
            "protected_paths": [rel(p) for p in PROTECTED_PATHS], "prior_evidence_sha256_raw": prior_evidence_digest(),
            "uncommitted_files": uncommitted or None, "existing_output_files": leftovers or None}


def build_snapshot() -> dict:
    out = Path(tempfile.mkdtemp(prefix="s3p_vllm_snapshot_")) / "vllm_kvquant_snapshot.zip"
    run_git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(out), "HEAD:vllm-kvquant")
    return {"path": str(out), "sha256": sha256_raw(out), "bytes": out.stat().st_size}


def build_command(correctness_only: bool, include_8192: bool) -> list[str]:
    cmd = [sys.executable, "-m", "modal", "run", str(MODAL_APP),
           "--series", ",".join(f"{l}={d}:{i}" for l, d, i in SERIES),
           "--points", ",".join(map(str, points(include_8192))), "--conditioning-prompt", str(CONDITIONING_PROMPT)]
    return cmd + (["--correctness-only"] if correctness_only else [])


def expected_runtime(include_8192: bool) -> dict:
    """Planning estimate only (from the tile32 benchmark's unprofiled TTFT plus a per-window overhead model)."""
    s = json.loads(BENCH_SUMMARY.read_text(encoding="utf-8"))
    ttft = {p["second_chunk_q_len"]: (p["reference_ttft_ms"], p["tile32_ttft_ms"]) for p in s["points"]}
    windows = {"reference": 12, "tile32": 11}  # instrumented windows per query-layer step (see analysis module)
    per_window_ms = 0.015  # wrapper bookkeeping + event records per window (assumed; measured values will differ)
    out = {}
    for k, impl in enumerate(("reference", "tile32")):
        per_q = {}
        for q in [32] + q_lens(include_8192):  # conditioning (q=32) + points
            base = ttft[q][k] / 1000.0
            per_q[q] = base + q * NUM_LAYERS * windows[impl] * per_window_ms / 1000.0
        out[impl] = {"requests_s": {str(q): round(v, 1) for q, v in per_q.items()},
                     "requests_total_s": round(sum(per_q.values()), 1)}
    fixed = {"image_rebuild_s": 600, "gate_s": 70, "tile32_tests_s": 50, "profile_tests_s": 120,
             "engine_init_per_series_s": 110}
    total = (sum(v for k, v in fixed.items() if k != "engine_init_per_series_s")
             + len(SERIES) * fixed["engine_init_per_series_s"] + sum(v["requests_total_s"] for v in out.values()))
    return {"per_impl": out, "fixed_s": fixed, "total_estimate_s": round(total), "assumed_per_window_ms": per_window_ms,
            "note": "planning estimate; the largest single request must stay far below the 600 s request cap"}


# ------------------------------------------------------------------ parsing
def demux(text: str) -> tuple[dict, list[str], list[str], list[str], list[str]]:
    series = {l: [] for l, _, _ in SERIES}
    gate, t32, prof, top = [], [], [], []
    prefixes = {f"[series{k}:{l}] ": l for k, (l, _, _) in enumerate(SERIES, start=1)}
    for line in text.splitlines():
        for p, bucket in (("[gate] ", gate), ("[tile32-tests] ", t32), ("[profile-tests] ", prof)):
            if line.startswith(p):
                bucket.append(line[len(p):])
                break
        else:
            for p, l in prefixes.items():
                if line.startswith(p):
                    series[l].append(line[len(p):])
                    break
            else:
                top.append(line)
    return series, gate, t32, prof, top


def profile_by_request(lines: list[str]) -> dict:
    """Profile payloads attributed to the open request (by S3C_POINT_BEGIN index); outside -> 'outside'."""
    out, cur = {"outside": []}, None
    for line in lines:
        s = line.strip()
        if s.startswith("S3C_POINT_BEGIN="):
            cur = json.loads(s.split("=", 1)[1])["i"]
            out[cur] = []
        elif s.startswith("S3C_POINT="):
            cur = None
        elif pa.TAG in s:
            out["outside" if cur is None else cur].extend(pa.find_records([s]))
    return out


def parse_top(lines: list[str]) -> dict:
    out = {"pre": {}, "exit": {}, "start": {}, "proc": {}, "timeouts": [], "complete": False,
           "correctness_only_complete": False}
    for line in lines:
        s = line.strip()
        if s == "S3C_PROFILE_COMPLETE":
            out["complete"] = True
            continue
        if s == "S3C_CORRECTNESS_ONLY_COMPLETE":
            out["correctness_only_complete"] = True
            continue
        tp = rd._strict_tag(s)
        if not tp:
            continue
        tag, p = tp
        if tag == "S3C_PRE_LEG_GPU_STATE":
            out["pre"][p["leg"]] = p
        elif tag == "S3C_SERIES_START":
            out["start"][p["series"]] = p
        elif tag == "S3C_SERIES_EXIT":
            out["exit"][p["series"]] = p
        elif tag == "S3C_PROCESS_EXIT":
            out["proc"][p["label"]] = p
        elif tag == "S3C_WATCHDOG_TIMEOUT":
            out["timeouts"].append(p)
        else:
            out[tag] = p
    return out


def parse_profiles(series_lines: dict) -> dict:
    """{label: {"requests": {i: aggregate|error}, "outside": n}} with strict record parsing."""
    out = {}
    for l, _, impl in SERIES:
        by = profile_by_request(series_lines[l])
        reqs = {}
        for i, payloads in by.items():
            if i == "outside":
                continue
            try:
                recs = [pa.parse_record(p) for p in payloads]
                agg = pa.aggregate(recs)
                reqs[i] = {"ok": True, "records": len(recs), "agg": agg}
            except pa.ProfileError as e:
                reqs[i] = {"ok": False, "records": len(payloads), "error": str(e)}
        out[l] = {"impl": impl, "requests": reqs, "outside": len(by["outside"])}
    return out


def integrity(series: dict, profiles: dict, gate: dict, t32: dict, ptests: dict, top: dict, correctness_only: bool,
              include_8192: bool) -> dict:
    checks = []
    pts = points(include_8192)

    def add(name, cat, state, observed=None):
        if isinstance(state, bool):
            state = PASSED if state else FAILED
        checks.append({"check": name, "category": cat, "state": state, "observed": observed})

    env = top.get("S3C_ENVIRONMENT", {})
    add("exactly one H100", "environment",
        (len(env.get("gpus", [])) == 1 and "H100" in env["gpus"][0].get("name", "")) if env else NOT_EVALUATED)
    add("rabit_kv2.py in image is frozen", "environment",
        env.get("rabit_kv2_sha256_lf") == EXPECTED_RABIT_SHA256_LF if env else NOT_EVALUATED)
    add("gate passed", "gate", (top.get("S3C_GATE_EXIT", {}).get("returncode") == 0
                                and (gate.get("result") or {}).get("passed") is True and gate.get("pytest_exit") == 0)
        if "S3C_GATE_START" in top else NOT_RUN)
    for label, tag, tests in (("tile32 correctness tests", "S3C_TILE_TESTS", t32),
                              ("profiler tests", "S3C_PROFILE_TESTS", ptests)):
        add(f"{label}: exit 0, all passed, none skipped, no failure/error", "tests",
            (top.get(f"{tag}_EXIT", {}).get("returncode") == 0 and tests["passed"] > 0 and tests["failed"] == 0
             and tests["errors"] == 0 and tests["skipped"] == 0) if f"{tag}_START" in top else NOT_RUN, tests)
    if correctness_only:
        add("correctness-only run completed", "completion", top["correctness_only_complete"])
    else:
        add("profiling run completed (all series)", "completion", top["complete"])
        add("no series watchdog timeout", "watchdog", not top["timeouts"], top["timeouts"] or None)
        base = top.get("S3C_GPU_BASELINE", {})
        local_prof_sha = sha256(PROFILE_MODULE)  # LF-normalized = snapshot content
        for label, dtype, impl in SERIES:
            s = series[label]
            t = s["tags"]
            started = label in top["start"]

            def chk(name, cat, ok, observed=None, _s=started, _l=label):
                add(f"{_l}: {name}", cat, ok if _s else NOT_RUN, observed if _s else None)

            pre = top["pre"].get(label)
            add(f"{label}: GPU clean before series", "gpu_clean", r5.gpu_leg_clean(pre, base) if pre else NOT_RUN)
            chk("exit 0, complete, no request timeout/failure", "series",
                (top["exit"].get(label) or {}).get("returncode") == 0 and s["complete"]
                and not s["timeouts"] and not s["failures"])
            si = t.get("S3C_STAGE_IMPL") or {}
            chk(f"Stage3C implementation = {impl} (requested and reported)", "impl",
                si.get("requested") == impl and si.get("selector_reports") == impl, si or None)
            pm = t.get("S3C_PROFILE_MODE") or {}
            chk("component profiling ON, legacy tile32 profile OFF, profiler module = committed", "profile_mode",
                pm.get("component_profiling") is True and pm.get("legacy_tile32_profile") is False
                and pm.get("schema") == pa.SCHEMA and pm.get("profile_module_sha256") == local_prof_sha, pm or None)
            eff = t.get("S3C_EFFECTIVE_ENGINE_CONFIG", {})
            chk("effective engine config frozen", "config",
                bool(eff) and all(eff.get(k, "<missing>") == v for k, v in rd.EXPECTED_EFFECTIVE.items()))
            kv = t.get("S3C_KV_DTYPE", {})
            chk("resolved KV dtype", "kv_dtype", kv.get("requested_kv_cache_dtype") == dtype
                and all(kv.get(k) == v for k, v in rd.EXPECTED_KV[dtype].items()))
            chk("model layers", "workload", (t.get("S3C_MODEL_LAYERS") or {}).get("num_hidden_layers") == NUM_LAYERS)
            meas = [p for p in s["points"] if p["begin"]["role"] == "measured"]
            chk("one profiled request per point, in order", "measurement",
                [p["row"]["planned_prompt_tokens"] for p in meas] == pts)
            chk("exact prompt tokens, 32 outputs, prompt hash == planned", "workload", bool(meas) and all(
                p["row"]["prompt_tokens"] == p["row"]["planned_prompt_tokens"] and p["row"]["output_tokens"] == 32
                and p["row"]["prompt_token_ids_sha256"] == p["row"]["planned_prompt_token_ids_sha256"]
                for p in s["points"]))
            chk("no Triton JIT during profiled requests", "jit", bool(meas) and all(p["jit"] == 0 for p in meas))
            chk("no OOM", "request", s["init"]["oom"] == 0 and all(p["oom"] == 0 for p in s["begins"]))
            pr = profiles[label]
            rows = {p["begin"]["i"]: p for p in s["points"]}
            ok_all = bool(rows) and pr["outside"] == 0
            obs = {}
            for i, p in rows.items():
                q = p["row"]["planned_prompt_tokens"] - FIRST_CHUNK
                r = pr["requests"].get(i)
                good = (r is not None and r["ok"] and r["records"] == NUM_LAYERS and r["agg"]["impl"] == impl
                        and r["agg"]["q_len"] == q and r["agg"]["context_len"] == FIRST_CHUNK)
                obs[str(i)] = {"q_len": q, "records": None if r is None else r["records"],
                               "error": None if (r is None or r["ok"]) else r["error"]}
                ok_all = ok_all and good
            chk(f"profile records: {NUM_LAYERS} valid records per request (accounting contract), impl/q_len/"
                "context match, none outside requests", "profile_records", ok_all, obs)
        ref = {p["row"]["planned_prompt_tokens"]: p["row"] for p in series[SERIES[0][0]]["points"]
               if p["begin"]["role"] == "measured"}
        til = {p["row"]["planned_prompt_tokens"]: p["row"] for p in series[SERIES[1][0]]["points"]
               if p["begin"]["role"] == "measured"}
        add("greedy output tokens identical, reference vs tile32 (profiled), at every point", "equivalence",
            all(ref[p]["output_token_ids_sha256"] == til[p]["output_token_ids_sha256"] for p in pts)
            if len(ref) == len(til) == len(pts) else NOT_EVALUATED)
        bench = {p["prompt_tokens"]: p for p in json.loads(BENCH_SUMMARY.read_text(encoding="utf-8"))["points"]}
        add("profiled output tokens identical to the unprofiled tile32 benchmark at every point", "equivalence",
            all(ref[p]["output_token_ids_sha256"] == bench[p]["reference_output_token_ids_sha256"]
                and til[p]["output_token_ids_sha256"] == bench[p]["tile32_output_token_ids_sha256"]
                and ref[p]["prompt_token_ids_sha256"] == bench[p]["prompt_token_ids_sha256"] for p in pts)
            if len(ref) == len(til) == len(pts) else NOT_EVALUATED)
        cfg = {l: rd.series_config(series[l]) for l, _, _ in SERIES}
        add("reference and tile32 engine configs identical", "config",
            cfg[SERIES[0][0]] == cfg[SERIES[1][0]] if all(cfg.values()) else NOT_EVALUATED)
    counts = {s: sum(1 for c in checks if c["state"] == s) for s in (PASSED, FAILED, NOT_RUN, NOT_EVALUATED)}
    return {"checks": checks, "counts": counts, "all_ok": counts[PASSED] == len(checks),
            "failed_categories": sorted({c["category"] for c in checks if c["state"] == FAILED})}


def analysis(series: dict, profiles: dict, integ: dict, include_8192: bool) -> dict:
    out = {"scope": SCOPE, "diagnostic_evidence": True, "latency_evidence": False, "experiment5_evidence": False,
           "component_profiling_enabled": True, "all_integrity_passed": integ["all_ok"],
           "integrity_counts": integ["counts"],
           "accounting_contract": pa.__doc__.split("Accounting contract", 1)[1].split("parse_record()", 1)[0],
           "percentages": "HOST shares are of the HOST wall; GPU shares are of the GPU span; never combined",
           "points": []}
    bench = {p["second_chunk_q_len"]: p for p in json.loads(BENCH_SUMMARY.read_text(encoding="utf-8"))["points"]}
    for q in q_lens(include_8192):
        row = {"second_chunk_q_len": q, "prompt_tokens": FIRST_CHUNK + q}
        for label, _, impl in SERIES:
            p = next((x for x in series[label]["points"] if x["begin"]["role"] == "measured"
                      and x["row"]["planned_prompt_tokens"] == FIRST_CHUNK + q), None)
            r = None if p is None else profiles[label]["requests"].get(p["begin"]["i"])
            if not (r and r["ok"]):
                row[impl] = None
                continue
            att = pa.attribute(r["agg"], NUM_LAYERS)
            row[impl] = {**att, "classification": pa.classify(att),
                         "profiled_request_ttft_ms_NOT_LATENCY": p["row"]["ttft_ms"],
                         "external_unprofiled_ttft_ms": bench[q][f"{impl}_ttft_ms"],
                         "external_unprofiled_bf16_control_ttft_ms": bench[q]["bf16_control_ttft_ms"]}
        out["points"].append(row)
    out["external_reference"] = {"source": rel(BENCH_SUMMARY),
                                 "note": "unprofiled tile32-benchmark TTFTs for the same prompts; context only, "
                                         "never pooled with profiled numbers"}
    out["decision"] = ("pre-registered rule outputs are per implementation / q_len under 'classification'; "
                       "the next optimization is chosen only after review")
    out["not_claimed"] = ["latency", "Experiment 5 results", "a complexity law",
                          "splits of the boundaries listed in combined_boundaries"]
    return out


def analyze(text: str, correctness_only: bool, include_8192: bool, write: bool) -> tuple[dict, dict]:
    ser_lines, gate_lines, t32_lines, prof_lines, top_lines = demux(text)
    series = {l: rd.parse_series(ser_lines[l]) for l, _, _ in SERIES}
    profiles = parse_profiles(ser_lines)
    gate = r5.parse_gate(gate_lines)
    t32, ptests = rt.parse_tests(t32_lines), rt.parse_tests(prof_lines)
    top = parse_top(top_lines)
    integ = integrity(series, profiles, gate, t32, ptests, top, correctness_only, include_8192)
    an = None if correctness_only else analysis(series, profiles, integ, include_8192)
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        GATE_LOG.write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
        TILE32_TESTS_LOG.write_text("\n".join(t32_lines) + "\n", encoding="utf-8")
        PROFILE_TESTS_LOG.write_text("\n".join(prof_lines) + "\n", encoding="utf-8")
        if not correctness_only:
            for l, _, _ in SERIES:
                (OUT_DIR / SERIES_LOG[l]).write_text("\n".join(ser_lines[l]) + "\n", encoding="utf-8")
            ANALYSIS.write_text(json.dumps(an, indent=2, default=str) + "\n", encoding="utf-8")
        INTEGRITY.write_text(json.dumps(integ, indent=2, default=str) + "\n", encoding="utf-8")
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


def run(m: dict, correctness_only: bool, include_8192: bool) -> int:
    snap = build_snapshot()
    m["vllm_kvquant_snapshot"] = snap
    m["command"] = build_command(correctness_only, include_8192)
    write_manifest(m)
    code = stream_command(m["command"], SESSION_LOG, {"S3P_VLLM_SNAPSHOT": snap["path"]})
    m["modal_returncode"] = code
    m["stage"] = "parse"
    integ, _ = analyze(SESSION_LOG.read_text(encoding="utf-8", errors="replace"), correctness_only, include_8192,
                       write=True)
    m.pop("stage")
    m["integrity_counts"] = integ["counts"]
    if code != 0 or not integ["all_ok"]:
        finalize(m, "failed", {"stage": (integ["failed_categories"] or ["modal_nonzero_exit"])[0],
                               "failed_categories": integ["failed_categories"]})
        raise SystemExit(f"\nSTAGE3C PROFILE STOPPED; partial evidence kept in {rel(OUT_DIR)}/.")
    finalize(m, "completed", None)
    print(f"\nSTAGE3C {'PROFILE CORRECTNESS' if correctness_only else 'COMPONENT PROFILE'} COMPLETED. "
          f"Analysis: {ANALYSIS}")
    return 0


def main(argv: list[str] | None = None) -> int:
    make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--correctness-only", action="store_true")
    ap.add_argument("--include-8192", action="store_true", help="optional q_len 8192 confirmation point")
    a = ap.parse_args(argv)
    if a.correctness_only:
        set_out_dir(CORRECTNESS_DIR)
    print("RABIT-KV Stage3C COMPONENT PROFILE diagnostic (not latency, not Experiment 5 evidence)")
    print(f"Series: {SERIES}; conditioning {CONDITIONING_PROMPT} (unmeasured)")
    print("Points (prompt -> q_len): " + ", ".join(f"{p}->{q}" for p, q in zip(points(a.include_8192),
                                                                              q_lens(a.include_8192))))
    print(f"Request guard: {REQUEST_GUARD_NOTE}")
    print(f"  (series watchdog {SERIES_TIMEOUT_S}s; watchdog budget {WATCHDOG_BUDGET_S}s; "
          f"Modal function timeout {MODAL_FUNCTION_TIMEOUT_S}s)")
    prov = preflight(a.dry_run)
    print("Preflight OK:", json.dumps({k: prov[k] for k in ("git_head", "rabit_kv2_sha256", "tile32_module_sha256",
                                                             "profile_module_sha256", "triton_attn_sha256")},
                                      indent=1))
    print("  equivalence:", json.dumps(prov["equivalence"]))
    print("  expected runtime:", json.dumps(expected_runtime(a.include_8192)))
    if prov["uncommitted_files"]:
        print("  WARNING (dry-run only): uncommitted files:\n    " + prov["uncommitted_files"].replace("\n", "\n    "))
    print("Local command:\n  " + " ".join(build_command(a.correctness_only, a.include_8192)))
    if a.dry_run:
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    m = {"diagnostic": "Stage3C component profile", "scope": SCOPE, "correctness_only": a.correctness_only,
         "series": SERIES, "points": points(a.include_8192), "q_lens": q_lens(a.include_8192),
         "include_8192": a.include_8192, "conditioning_prompt_tokens": CONDITIONING_PROMPT,
         "request_guard_note": REQUEST_GUARD_NOTE, "started_utc": now(), "status": "running",
         "protected_paths_post_run_status": "pending", "provenance": prov}
    write_manifest(m)
    try:
        return run(m, a.correctness_only, a.include_8192)
    except SystemExit:
        raise
    except (Exception, KeyboardInterrupt) as exc:
        stage = "parser_failure" if m.pop("stage", None) == "parse" else "local_runner_exception"
        finalize(m, "failed", {"stage": stage, "type": type(exc).__name__, "message": str(exc)})
        raise SystemExit(f"\nSTAGE3C PROFILE RUNNER FAILED: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
