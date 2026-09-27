"""
RABIT-KV Stage3C tile32 before/after BENCHMARK runner (prep; review before any
run). Benchmark evidence only -- NOT Experiment 5 evidence; no Experiment 5 claim.

Two modes (one `modal run` of benchmarks/mlsys2027/stage3c_tile32_bench_modal.py):
  --correctness-only : idle baseline, frozen correctness gate, then the tile32
                       correctness tests (tile32 vs reference Stage3C: exact
                       cache bytes / runtime state / attention output / next
                       decode). No timing.
  (default)          : the same two stages, then three series, each one fresh
                       engine with one unmeasured 16416-token conditioning
                       request and ONE measured request per point:
                         bf16_control   = bfloat16  (external control)
                         rabit_reference = rabit_kv2, VLLM_RABIT2_STAGE3C_IMPL=reference
                         rabit_tile32    = rabit_kv2, VLLM_RABIT2_STAGE3C_IMPL=tile32
Points (prompt tokens -> second-chunk q_len): 16386->2, 16415->31, 16416->32,
16417->33, 16896->512, 17408->1024, 18432->2048, 20480->4096, 24576->8192.
The 16352 (32K) point is deliberately NOT included; it is planned only after
tile32 shows practical runtime.

Frozen engine settings (verified by AST): max_num_batched_tokens 16384,
max_model_len 32768, block_size 32, gpu_memory_utilization 0.82, eager, Triton,
no torch.compile / CUDA graphs, 32 output tokens. Request guard: 600 s per-request
SIGALRM guard, with the process-group watchdog and the Modal function timeout as
hard process-level backstops. No retries; any failure stops the run.

Per-point outputs: reference and tile32 TTFT / TPOT / wall, speedup, TTFT saved,
greedy output-token hash equality (reference vs tile32), BF16 control TTFT.
Descriptive only; no complexity law. Component timing (VLLM_RABIT2_STAGE3C_PROFILE)
is NOT enabled here: it synchronizes the device and must never be used for latency.

Usage:
    python benchmarks/mlsys2027/run_stage3c_tile32_benchmark.py --dry-run [--correctness-only]
    python benchmarks/mlsys2027/run_stage3c_tile32_benchmark.py [--correctness-only]
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
import run_experiment3_deployment as r3  # noqa: E402  (frozen; helpers only)
import run_experiment5_context_scaling as r5  # noqa: E402  (frozen; helpers only)
import run_stage3c_cliff_diagnostic as rd  # noqa: E402  (committed diagnostic; helpers only)
from run_experiment3_deployment import (  # noqa: E402
    FAILED, NOT_EVALUATED, NOT_RUN, PASSED, _function, _module_assign, canonical_llm_kwargs,
    canonical_runner_source, flatten, make_console_encoding_safe, now, rel, run_git, sha256, sha256_raw,
    stream_command,
)

ROOT = r3.ROOT
HERE = ROOT / "benchmarks" / "mlsys2027"
RUNNER_SCRIPT = Path(__file__).resolve()
MODAL_APP = HERE / "stage3c_tile32_bench_modal.py"
WORKER = HERE / "stage3c_tile32_bench_worker.py"
DIAG_WORKER = HERE / "stage3c_diag_worker.py"
DIAG_MODAL_APP = HERE / "stage3c_diag_modal.py"
EXP3_WORKER = HERE / "exp3_engine_worker.py"
EXP3_MODAL_APP = HERE / "exp3_deployment_modal.py"
GATE = HERE / "exp3_correctness_gate.py"
WATCHDOG = HERE / "exp3_watchdog.py"
TILE32_MODULE = ROOT / "vllm-kvquant" / "vllm" / "v1" / "attention" / "ops" / "rabit_kv2_stage3c_tile32.py"
TILE32_TESTS = ROOT / "vllm-kvquant" / "tests" / "quantization" / "test_rabit2_stage3c_tile32.py"
TRITON_ATTN = ROOT / "vllm-kvquant" / "vllm" / "v1" / "attention" / "backends" / "triton_attn.py"
RABIT_KV2 = r3.RABIT_KV2
EXPECTED_RABIT_SHA256_LF = r3.EXPECTED_RABIT_SHA256_LF

OUT_DIR = ROOT / "results" / "mlsys2027" / "diagnostics" / "stage3c_tile32_benchmark"
SESSION_LOG = OUT_DIR / "modal_session.log"
GATE_LOG = OUT_DIR / "correctness_gate.log"
TESTS_LOG = OUT_DIR / "tile32_correctness_tests.log"
MANIFEST = OUT_DIR / "manifest.json"
INTEGRITY = OUT_DIR / "integrity_check.json"
ANALYSIS = OUT_DIR / "benchmark_analysis.json"

EVIDENCE_DIRS = [*rd.EVIDENCE_DIRS, ROOT / "results" / "mlsys2027" / "diagnostics" / "stage3c_cliff"]
PROTECTED_PATHS = [*rd.PROTECTED_PATHS, EVIDENCE_DIRS[-1], RABIT_KV2,
                   *sorted(HERE.glob("*stage3c_diag*")), HERE / "run_stage3c_cliff_diagnostic.py"]
MUST_BE_COMMITTED = [RUNNER_SCRIPT, MODAL_APP, WORKER, GATE, WATCHDOG, TILE32_MODULE, TILE32_TESTS, TRITON_ATTN]

SERIES = [("bf16_control", "bfloat16", "reference"),
          ("rabit_reference", "rabit_kv2", "reference"),
          ("rabit_tile32", "rabit_kv2", "tile32")]
SERIES_LOG = {"bf16_control": "bf16_control_series.log", "rabit_reference": "rabit_reference_series.log",
              "rabit_tile32": "rabit_tile32_series.log"}
FIRST_CHUNK = 16384
CONDITIONING_PROMPT = 16416
Q_LENS = [2, 31, 32, 33, 512, 1024, 2048, 4096, 8192]
POINTS = [FIRST_CHUNK + q for q in Q_LENS]
REQUEST_CAP_S = 600
SERIES_TIMEOUT_S = 6600
TILE32_TEST_TIMEOUT_S = 1800
GATE_TIMEOUT_S = 600
WATCHDOG_BUDGET_S = GATE_TIMEOUT_S + TILE32_TEST_TIMEOUT_S + len(SERIES) * SERIES_TIMEOUT_S  # 22200
MODAL_FUNCTION_TIMEOUT_S = 23400
REQUEST_GUARD_NOTE = rd.REQUEST_GUARD_NOTE
SCOPE = ("Stage3C tile32 before/after BENCHMARK evidence only; NOT Experiment 5 evidence; one measured request "
         "per point; descriptive only (no complexity law). The 32K (q_len 16352) point is not part of this run.")


def assert_protected_paths_clean(context: str) -> None:
    # The code under test (MUST_BE_COMMITTED) lives inside the protected vllm-kvquant
    # tree; it is policed by the separate uncommitted-code check (a real run refuses
    # unless it is committed). Everything else under the protected paths must be clean.
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


def verify_equivalence() -> dict:
    csrc = canonical_runner_source()
    ctree = ast.parse(csrc)
    w, wd, w3 = (ast.parse(p.read_text(encoding="utf-8")) for p in (WORKER, DIAG_WORKER, EXP3_WORKER))
    m, md, m3 = (ast.parse(p.read_text(encoding="utf-8")) for p in (MODAL_APP, DIAG_MODAL_APP, EXP3_MODAL_APP))
    canon = {k: v for k, v in canonical_llm_kwargs(csrc).items() if k not in ("model", "kv_cache_dtype")}
    base = rd._const(w, "BASE_ENGINE_KWARGS")
    if not (canon == rd._const(w3, "BASE_ENGINE_KWARGS") == rd._const(wd, "BASE_ENGINE_KWARGS") == base):
        raise RuntimeError("BASE_ENGINE_KWARGS differ from canonical / Experiment 3 / diagnostic")
    if base["max_num_batched_tokens"] != FIRST_CHUNK:
        raise RuntimeError("max_num_batched_tokens changed")
    if tuple(rd._const(w, "ALLOWED_KV_CACHE_DTYPES")) != ("bfloat16", "rabit_kv2"):
        raise RuntimeError("worker dtypes changed (FP8 must not be present)")
    main_d, main_w = _function(wd, "main"), _function(w, "main")
    one_d, one_w = rd._nested(main_d, "one"), rd._nested(main_w, "one")
    if ast.dump(one_d) != ast.dump(one_w):
        raise RuntimeError("timed region one() differs from the committed diagnostic worker")
    for target in ("tok", "bos", "filler", "sp", "prompt"):
        if rd._assigns(main_d, target) != rd._assigns(main_w, target):
            raise RuntimeError(f"'{target} = ...' differs from the committed diagnostic worker")
    mw = ast.dump(main_w)
    if "collective_rpc" in mw or "VLLM_RABIT2_STAGE3C_IMPL" not in ast.unparse(main_w):
        raise RuntimeError("worker must set VLLM_RABIT2_STAGE3C_IMPL and must not use engine RPC")
    if ast.dump(_module_assign(ctree, "image")) != ast.dump(_module_assign(m, "image")):
        raise RuntimeError("benchmark Modal image differs from the canonical image")
    for fn in ("_emit", "_sha256_file", "_gpu_query", "_compute_apps", "_gpu_state", "_require_clean", "_run_guarded"):
        if ast.dump(_function(m3, fn)).replace("EXP3_", "S3C_") != ast.dump(_function(m, fn)) \
                or ast.dump(_function(md, fn)) != ast.dump(_function(m, fn)):
            raise RuntimeError(f"Modal helper {fn}() differs from Experiment 3 / diagnostic")
    for const, mine in (("REQUEST_CAP_S", REQUEST_CAP_S), ("SERIES_TIMEOUT_S", SERIES_TIMEOUT_S),
                        ("TILE32_TEST_TIMEOUT_S", TILE32_TEST_TIMEOUT_S), ("GATE_TIMEOUT_S", GATE_TIMEOUT_S),
                        ("GPU_CLEAN_TOLERANCE_MIB", 256), ("EXPECTED_RABIT_SHA256_LF", EXPECTED_RABIT_SHA256_LF)):
        if rd._const(m, const) != mine:
            raise RuntimeError(f"Modal {const} differs")
    if SERIES_TIMEOUT_S < (1 + len(POINTS)) * REQUEST_CAP_S:
        raise RuntimeError("series timeout below (conditioning + points) x request cap")
    backstop = None
    for dec in _function(m, "benchmark").decorator_list:
        for kw in getattr(dec, "keywords", []):
            if kw.arg == "timeout":
                backstop = ast.literal_eval(kw.value)
    if not (backstop == MODAL_FUNCTION_TIMEOUT_S > WATCHDOG_BUDGET_S):
        raise RuntimeError(f"Modal backstop {backstop} must exceed the watchdog budget {WATCHDOG_BUDGET_S}")
    return {"engine_kwargs_equal_canonical_exp3_diag": True, "timed_region_equal_diag_worker": True,
            "helpers_equal_exp3_and_diag": True, "image_equal_canonical": True,
            "watchdog_budget_s": WATCHDOG_BUDGET_S, "modal_backstop_s": backstop}


def preflight(dry_run: bool) -> dict:
    assert_protected_paths_clean("preflight")
    if sha256(RABIT_KV2) != EXPECTED_RABIT_SHA256_LF:
        raise RuntimeError("rabit_kv2.py is not the frozen committed content")
    eq = verify_equivalence()
    uncommitted = run_git("status", "--short", "--", *[rel(p) for p in MUST_BE_COMMITTED])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: benchmark / tile32 code has uncommitted changes:\n" + uncommitted)
    leftovers = sorted(p.name for p in OUT_DIR.iterdir()) if OUT_DIR.is_dir() else []
    if leftovers and not dry_run:
        raise RuntimeError(f"Refusing to run: {rel(OUT_DIR)} is not empty ({leftovers})")
    return {"git_head": run_git("rev-parse", "HEAD"), "vllm_kvquant_tree": run_git("rev-parse", "HEAD:vllm-kvquant"),
            "rabit_kv2_sha256": sha256(RABIT_KV2), "tile32_module_sha256": sha256(TILE32_MODULE),
            "tile32_tests_sha256": sha256(TILE32_TESTS), "triton_attn_sha256": sha256(TRITON_ATTN),
            "runner_script_sha256": sha256(RUNNER_SCRIPT), "modal_app_sha256": sha256(MODAL_APP),
            "worker_sha256": sha256(WORKER), "equivalence": eq, "protected_paths": [rel(p) for p in PROTECTED_PATHS],
            "prior_evidence_sha256_raw": prior_evidence_digest(), "uncommitted_files": uncommitted or None,
            "existing_output_files": leftovers or None}


def build_snapshot() -> dict:
    out = Path(tempfile.mkdtemp(prefix="t32_vllm_snapshot_")) / "vllm_kvquant_snapshot.zip"
    run_git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(out), "HEAD:vllm-kvquant")
    return {"path": str(out), "sha256": sha256_raw(out), "bytes": out.stat().st_size}


def build_command(correctness_only: bool) -> list[str]:
    cmd = [sys.executable, "-m", "modal", "run", str(MODAL_APP),
           "--series", ",".join(f"{l}={d}:{i}" for l, d, i in SERIES),
           "--points", ",".join(map(str, POINTS)), "--conditioning-prompt", str(CONDITIONING_PROMPT)]
    return cmd + (["--correctness-only"] if correctness_only else [])


# ------------------------------------------------------------------ parsing
def demux(text: str) -> tuple[dict, list[str], list[str], list[str]]:
    series = {l: [] for l, _, _ in SERIES}
    gate, tests, top = [], [], []
    prefixes = {f"[series{k}:{l}] ": l for k, (l, _, _) in enumerate(SERIES, start=1)}
    for line in text.splitlines():
        if line.startswith("[gate] "):
            gate.append(line[7:])
        elif line.startswith("[tile32-tests] "):
            tests.append(line[len("[tile32-tests] "):])
        else:
            for p, l in prefixes.items():
                if line.startswith(p):
                    series[l].append(line[len(p):])
                    break
            else:
                top.append(line)
    return series, gate, tests, top


def parse_top(lines: list[str]) -> dict:
    out = {"pre": {}, "exit": {}, "start": {}, "proc": {}, "timeouts": [], "complete": False,
           "correctness_only_complete": False}
    for line in lines:
        s = line.strip()
        if s == "S3C_BENCHMARK_COMPLETE":
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


def parse_tests(lines: list[str]) -> dict:
    text = "\n".join(lines)
    m = re.search(r"(\d+) passed", text)
    f = re.search(r"(\d+) failed", text)
    e = re.search(r"(\d+) error", text)
    sk = re.search(r"(\d+) skipped", text)
    return {"passed": int(m.group(1)) if m else 0, "failed": int(f.group(1)) if f else 0,
            "errors": int(e.group(1)) if e else 0, "skipped": int(sk.group(1)) if sk else 0,
            "equality_failures": [ln.strip() for ln in lines if "differs" in ln][:20]}


def integrity(series: dict, gate: dict, tests: dict, top: dict, correctness_only: bool) -> dict:
    checks = []

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
    ran = "S3C_TILE_TESTS_START" in top
    add("tile32 correctness tests: exit 0, all passed, none skipped, no failure/error", "tile32_correctness",
        (top.get("S3C_TILE_TESTS_EXIT", {}).get("returncode") == 0 and tests["passed"] > 0
         and tests["failed"] == 0 and tests["errors"] == 0 and tests["skipped"] == 0) if ran else NOT_RUN, tests)
    if correctness_only:
        add("correctness-only run completed", "completion", top["correctness_only_complete"])
    else:
        add("benchmark completed (all series)", "completion", top["complete"])
        add("no series watchdog timeout", "watchdog", not top["timeouts"], top["timeouts"] or None)
        base = top.get("S3C_GPU_BASELINE", {})
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
            chk(f"Stage3C implementation = {impl} (requested and reported by the selector)", "impl",
                (t.get("S3C_STAGE_IMPL") or {}).get("requested") == impl
                and (t.get("S3C_STAGE_IMPL") or {}).get("selector_reports") == impl, t.get("S3C_STAGE_IMPL"))
            eff = t.get("S3C_EFFECTIVE_ENGINE_CONFIG", {})
            chk("effective engine config frozen", "config",
                bool(eff) and all(eff.get(k, "<missing>") == v for k, v in rd.EXPECTED_EFFECTIVE.items()))
            kv = t.get("S3C_KV_DTYPE", {})
            chk("resolved KV dtype", "kv_dtype", kv.get("requested_kv_cache_dtype") == dtype
                and all(kv.get(k) == v for k, v in rd.EXPECTED_KV[dtype].items()))
            meas = [p for p in s["points"] if p["begin"]["role"] == "measured"]
            chk("one measured request per point, in order", "measurement",
                [p["row"]["planned_prompt_tokens"] for p in meas] == POINTS)
            chk("exact prompt tokens, 32 outputs, prompt hash == planned", "workload", bool(meas) and all(
                p["row"]["prompt_tokens"] == p["row"]["planned_prompt_tokens"] and p["row"]["output_tokens"] == 32
                and p["row"]["prompt_token_ids_sha256"] == p["row"]["planned_prompt_token_ids_sha256"]
                for p in s["points"]))
            chk("no Triton JIT during measured requests", "jit", bool(meas) and all(p["jit"] == 0 for p in meas))
            chk("no OOM", "request", s["init"]["oom"] == 0 and all(p["oom"] == 0 for p in s["begins"]))
        ref = {p["row"]["planned_prompt_tokens"]: p["row"] for p in series["rabit_reference"]["points"]
               if p["begin"]["role"] == "measured"}
        til = {p["row"]["planned_prompt_tokens"]: p["row"] for p in series["rabit_tile32"]["points"]
               if p["begin"]["role"] == "measured"}
        add("greedy output tokens identical, reference vs tile32, at every point", "tile32_equivalence",
            all(ref[p]["output_token_ids_sha256"] == til[p]["output_token_ids_sha256"] for p in POINTS)
            if len(ref) == len(til) == len(POINTS) else NOT_EVALUATED)
        cfg = {l: rd.series_config(series[l]) for l, _, _ in SERIES}
        if all(cfg.values()):
            add("rabit_reference and rabit_tile32 engine configs identical", "config",
                cfg["rabit_reference"] == cfg["rabit_tile32"])
        else:
            add("rabit_reference and rabit_tile32 engine configs identical", "config", NOT_EVALUATED)
    counts = {s: sum(1 for c in checks if c["state"] == s) for s in (PASSED, FAILED, NOT_RUN, NOT_EVALUATED)}
    return {"checks": checks, "counts": counts, "all_ok": counts[PASSED] == len(checks),
            "failed_categories": sorted({c["category"] for c in checks if c["state"] == FAILED})}


def analysis(series: dict, tests: dict, integ: dict, correctness_only: bool) -> dict:
    out = {"scope": SCOPE, "experiment5_evidence": False, "all_integrity_passed": integ["all_ok"],
           "integrity_counts": integ["counts"], "tile32_correctness_tests": tests,
           "request_guard_note": REQUEST_GUARD_NOTE, "component_profiling_enabled": False}
    if correctness_only:
        return out
    rows = []
    meas = {l: {p["row"]["planned_prompt_tokens"]: p["row"] for p in series[l]["points"]
                if p["begin"]["role"] == "measured"} for l, _, _ in SERIES}
    for p, q in zip(POINTS, Q_LENS):
        r_, t_, b_ = meas["rabit_reference"].get(p), meas["rabit_tile32"].get(p), meas["bf16_control"].get(p)
        row = {"prompt_tokens": p, "second_chunk_q_len": q}
        for key, v in (("reference", r_), ("tile32", t_), ("bf16_control", b_)):
            row[key] = None if v is None else {k: v[k] for k in ("ttft_ms", "tpot_ms", "wall_ms",
                                                                   "output_token_ids_sha256")}
        if r_ and t_:
            row["ttft_speedup_reference_over_tile32"] = r_["ttft_ms"] / t_["ttft_ms"]
            row["ttft_saved_ms"] = r_["ttft_ms"] - t_["ttft_ms"]
            row["output_tokens_identical"] = r_["output_token_ids_sha256"] == t_["output_token_ids_sha256"]
        if t_ and b_:
            row["tile32_minus_bf16_ttft_ms"] = t_["ttft_ms"] - b_["ttft_ms"]
        rows.append(row)
    out["points"] = rows
    out["not_claimed"] = ["Experiment 5 results", "a complexity law", "tile32 correctness beyond the executed tests"]
    return out


def analyze(text: str, correctness_only: bool, write: bool) -> tuple[dict, dict]:
    ser_lines, gate_lines, test_lines, top_lines = demux(text)
    series = {l: rd.parse_series(ser_lines[l]) for l, _, _ in SERIES}
    gate = r5.parse_gate(gate_lines)
    tests = parse_tests(test_lines)
    top = parse_top(top_lines)
    integ = integrity(series, gate, tests, top, correctness_only)
    an = analysis(series, tests, integ, correctness_only)
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        GATE_LOG.write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
        TESTS_LOG.write_text("\n".join(test_lines) + "\n", encoding="utf-8")
        if not correctness_only:
            for l, _, _ in SERIES:
                (OUT_DIR / SERIES_LOG[l]).write_text("\n".join(ser_lines[l]) + "\n", encoding="utf-8")
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


def run(m: dict, correctness_only: bool) -> int:
    snap = build_snapshot()
    m["vllm_kvquant_snapshot"] = snap
    m["command"] = build_command(correctness_only)
    write_manifest(m)
    code = stream_command(m["command"], SESSION_LOG, {"T32_VLLM_SNAPSHOT": snap["path"]})
    m["modal_returncode"] = code
    m["stage"] = "parse"
    integ, _ = analyze(SESSION_LOG.read_text(encoding="utf-8", errors="replace"), correctness_only, write=True)
    m.pop("stage")
    m["integrity_counts"] = integ["counts"]
    if code != 0 or not integ["all_ok"]:
        finalize(m, "failed", {"stage": (integ["failed_categories"] or ["modal_nonzero_exit"])[0],
                               "failed_categories": integ["failed_categories"]})
        raise SystemExit(f"\nTILE32 BENCHMARK STOPPED; partial evidence kept in {rel(OUT_DIR)}/.")
    finalize(m, "completed", None)
    print(f"\nTILE32 {'CORRECTNESS' if correctness_only else 'BENCHMARK'} COMPLETED. Analysis: {ANALYSIS}")
    return 0


def main(argv: list[str] | None = None) -> int:
    make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--correctness-only", action="store_true")
    a = ap.parse_args(argv)
    print("RABIT-KV Stage3C tile32 before/after BENCHMARK (not Experiment 5 evidence)")
    print(f"Mode: {'correctness-only (gate + tile32 exact-equality tests, no timing)' if a.correctness_only else 'full'}")
    print(f"Series: {SERIES}; conditioning {CONDITIONING_PROMPT} (unmeasured)")
    print("Points (prompt -> q_len): " + ", ".join(f"{p}->{q}" for p, q in zip(POINTS, Q_LENS)) + "; no 16352 point")
    print(f"Request guard: {REQUEST_GUARD_NOTE}")
    print(f"  (tile32 tests {TILE32_TEST_TIMEOUT_S}s; series watchdog {SERIES_TIMEOUT_S}s; gate {GATE_TIMEOUT_S}s; "
          f"watchdog budget {WATCHDOG_BUDGET_S}s; Modal function timeout {MODAL_FUNCTION_TIMEOUT_S}s)")
    prov = preflight(a.dry_run)
    print("Preflight OK:", json.dumps({k: prov[k] for k in ("git_head", "rabit_kv2_sha256", "tile32_module_sha256",
                                                             "triton_attn_sha256", "runner_script_sha256")}, indent=1))
    print("  equivalence:", json.dumps(prov["equivalence"]))
    if prov["uncommitted_files"]:
        print("  WARNING (dry-run only): uncommitted files:\n    " + prov["uncommitted_files"].replace("\n", "\n    "))
    print("Local command:\n  " + " ".join(build_command(a.correctness_only)))
    if a.dry_run:
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    m = {"benchmark": "Stage3C tile32 before/after", "scope": SCOPE, "correctness_only": a.correctness_only,
         "series": SERIES, "points": POINTS, "q_lens": Q_LENS, "conditioning_prompt_tokens": CONDITIONING_PROMPT,
         "request_guard_note": REQUEST_GUARD_NOTE, "started_utc": now(), "status": "running",
         "protected_paths_post_run_status": "pending", "provenance": prov}
    write_manifest(m)
    try:
        return run(m, a.correctness_only)
    except SystemExit:
        raise
    except (Exception, KeyboardInterrupt) as exc:
        stage = "parser_failure" if m.pop("stage", None) == "parse" else "local_runner_exception"
        finalize(m, "failed", {"stage": stage, "type": type(exc).__name__, "message": str(exc)})
        raise SystemExit(f"\nTILE32 BENCHMARK RUNNER FAILED: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
