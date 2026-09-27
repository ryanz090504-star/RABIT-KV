"""
RABIT-KV Stage3C shared_decode kernel-tuning BENCHMARK runner (prep; review
before any run). Benchmark evidence only -- NOT Experiment 5 evidence.

Two modes (one `modal run` of benchmarks/mlsys2027/stage3c_shared_decode_bench_modal.py):
  --correctness-only : idle baseline, frozen correctness gate, tile32 exact suite,
                       shared_decode exact suite (reference oracle vs tile32 and
                       shared_decode at QUERY_BLOCK 4/8/16/32, incl. q_len 16352;
                       kernel-side decode counts). No timing.
                       -> results/mlsys2027/diagnostics/stage3c_shared_decode_correctness/
  (default)          : the same stages, then six series (rabit_kv2 only; no BF16),
                       each one fresh engine with one unmeasured 16416-token
                       conditioning request and ONE measured request per point:
                         rabit_reference, rabit_tile32,
                         rabit_shared_qb4 / qb8 / qb16 / qb32
                       -> results/mlsys2027/diagnostics/stage3c_shared_decode_benchmark/
Points (prompt tokens -> second-chunk q_len): 16416->32, 16896->512, 18432->2048,
20480->4096, 24576->8192. No 16352 point, no Experiment 5.

QUERY_BLOCK selection is kernel tuning (not an algorithmic ablation). Primary
comparison: reference vs shared_decode; secondary: tile32 vs shared_decode.
Frozen engine settings, prompt construction and timed region are AST-verified
identical to the tile32 benchmark. Request guard: 600 s per-request SIGALRM
guard, with the process-group watchdog and the Modal function timeout as hard
process-level backstops. No retries; any failure stops the run.

Usage:
    python benchmarks/mlsys2027/run_stage3c_shared_decode_benchmark.py --dry-run [--correctness-only]
    python benchmarks/mlsys2027/run_stage3c_shared_decode_benchmark.py [--correctness-only]
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment3_deployment as r3  # noqa: E402  (frozen; helpers only)
import run_experiment5_context_scaling as r5  # noqa: E402  (frozen; helpers only)
import run_stage3c_cliff_diagnostic as rd  # noqa: E402  (committed diagnostic; helpers only)
import run_stage3c_profile_diagnostic as rp  # noqa: E402  (committed diagnostic; helpers only)
import run_stage3c_tile32_benchmark as rt  # noqa: E402  (committed benchmark; helpers only)
from run_experiment3_deployment import (  # noqa: E402
    FAILED, NOT_EVALUATED, NOT_RUN, PASSED, _function, _module_assign, make_console_encoding_safe, now, rel,
    run_git, sha256, sha256_raw, stream_command,
)

ROOT = r3.ROOT
HERE = ROOT / "benchmarks" / "mlsys2027"
RUNNER_SCRIPT = Path(__file__).resolve()
MODAL_APP = HERE / "stage3c_shared_decode_bench_modal.py"
WORKER = HERE / "stage3c_shared_decode_bench_worker.py"
GATE, WATCHDOG = rt.GATE, rt.WATCHDOG
OPS = ROOT / "vllm-kvquant" / "vllm" / "v1" / "attention" / "ops"
SHARED_MODULE = OPS / "rabit_kv2_stage3c_shared_decode.py"
SHARED_TESTS = ROOT / "vllm-kvquant" / "tests" / "quantization" / "test_rabit2_stage3c_shared_decode.py"
TRITON_ATTN, TILE32_MODULE, TILE32_TESTS = rt.TRITON_ATTN, rt.TILE32_MODULE, rt.TILE32_TESTS
PROFILE_MODULE = rp.PROFILE_MODULE
RABIT_KV2 = r3.RABIT_KV2
EXPECTED_RABIT_SHA256_LF = r3.EXPECTED_RABIT_SHA256_LF
BENCH_MANIFEST = rt.OUT_DIR / "manifest.json"  # frozen tile32 benchmark (tile32 module / tests provenance)
PROFILED_COMMIT = rp.MEASUREMENT_CODE_COMMIT  # triton_attn.py baseline = the profiled code

BENCH_DIR = ROOT / "results" / "mlsys2027" / "diagnostics" / "stage3c_shared_decode_benchmark"
CORRECTNESS_DIR = ROOT / "results" / "mlsys2027" / "diagnostics" / "stage3c_shared_decode_correctness"
QB_TUNING_DIR = ROOT / "results" / "mlsys2027" / "diagnostics" / "stage3c_shared_decode_qb_tuning"
QB_TIEBREAK_DIR = ROOT / "results" / "mlsys2027" / "diagnostics" / "stage3c_shared_decode_qb_tiebreak"
FINAL_DIR = ROOT / "results" / "mlsys2027" / "diagnostics" / "stage3c_shared_decode_final_benchmark"


def set_out_dir(d: Path) -> None:
    global OUT_DIR, SESSION_LOG, GATE_LOG, TILE32_TESTS_LOG, SHARED_TESTS_LOG, MANIFEST, INTEGRITY, ANALYSIS
    OUT_DIR = d
    SESSION_LOG, GATE_LOG = d / "modal_session.log", d / "correctness_gate.log"
    TILE32_TESTS_LOG, SHARED_TESTS_LOG = d / "tile32_correctness_tests.log", d / "shared_decode_tests.log"
    MANIFEST, INTEGRITY, ANALYSIS = d / "manifest.json", d / "integrity_check.json", d / "benchmark_analysis.json"


set_out_dir(BENCH_DIR)

EVIDENCE_DIRS = [*rp.EVIDENCE_DIRS, rp.CORRECTNESS_DIR, rp.PROFILE_DIR]
PROTECTED_PATHS = [*rp.PROTECTED_PATHS, rp.CORRECTNESS_DIR, rp.PROFILE_DIR, rp.RUNNER_SCRIPT, rp.MODAL_APP, rp.WORKER,
                   rp.ANALYSIS_MODULE]
MUST_BE_COMMITTED = [RUNNER_SCRIPT, MODAL_APP, WORKER, GATE, WATCHDOG, SHARED_MODULE, SHARED_TESTS, TRITON_ATTN]

QUERY_BLOCKS = [4, 8, 16, 32]
DEFAULT_QB = 8  # passed to (and ignored by) the reference / tile32 series
SERIES = ([("rabit_reference", "rabit_kv2", "reference", DEFAULT_QB), ("rabit_tile32", "rabit_kv2", "tile32", DEFAULT_QB)]
          + [(f"rabit_shared_qb{qb}", "rabit_kv2", "shared_decode", qb) for qb in QUERY_BLOCKS])
FIRST_CHUNK = 16384
CONDITIONING_PROMPT = 16416
Q_LENS = [32, 512, 2048, 4096, 8192]
POINTS = [FIRST_CHUNK + q for q in Q_LENS]
REQUEST_CAP_S = 600
SERIES_TIMEOUT_S = 4200
TILE32_TEST_TIMEOUT_S = 1800
SHARED_TEST_TIMEOUT_S = 3600
GATE_TIMEOUT_S = 600
WATCHDOG_BUDGET_S = GATE_TIMEOUT_S + TILE32_TEST_TIMEOUT_S + SHARED_TEST_TIMEOUT_S + len(SERIES) * SERIES_TIMEOUT_S
MODAL_FUNCTION_TIMEOUT_S = 32400
# Pre-registered shared_decode test inventory (0 skips allowed).
EXPECTED_SHARED_TESTS = 7 * 4 * 4 + 3 * 5 * 4 + 4 + 3 * 4 + 1 + 2 + 1  # = 192
EXPECTED_32K_TESTS = [f"test_shared_decode_32k_model_limit_chunk[{qb}]" for qb in QUERY_BLOCKS]
REQUEST_GUARD_NOTE = rd.REQUEST_GUARD_NOTE
# --qb-tuning: shared_decode only, QUERY_BLOCK 4/8/16/32, q_len 512 and 2048 (no reference / tile32 series).
QB_TUNING_Q_LENS = [512, 2048]
QB_TUNING_SERIES = [s for s in SERIES if s[2] == "shared_decode"]
# Pre-registered selection rule (registered in code before the tuning run): the fixed QUERY_BLOCK is the one
# with the lowest geometric mean of measured TTFT over QB_TUNING_Q_LENS; an exact tie goes to the smaller block.
QB_SELECTION_RULE = ("argmin over QUERY_BLOCK of geomean(TTFT_ms at q_len " + " and ".join(map(str, QB_TUNING_Q_LENS))
                     + "); exact tie -> smaller QUERY_BLOCK; one measured request per point")


def select_query_block(ttft_ms: dict[int, dict[int, float]]) -> dict:
    """Apply QB_SELECTION_RULE to {query_block: {q_len: ttft_ms}} (every block must have every q_len)."""
    if sorted(ttft_ms) != QUERY_BLOCKS:
        raise ValueError(f"need every QUERY_BLOCK {QUERY_BLOCKS}, got {sorted(ttft_ms)}")
    score = {}
    for qb, row in ttft_ms.items():
        if sorted(row) != QB_TUNING_Q_LENS or any(not (v > 0) for v in row.values()):
            raise ValueError(f"QUERY_BLOCK {qb}: need positive TTFT at q_len {QB_TUNING_Q_LENS}, got {row}")
        score[qb] = math.exp(sum(math.log(row[q]) for q in QB_TUNING_Q_LENS) / len(QB_TUNING_Q_LENS))
    best = min(score.values())
    chosen = min(qb for qb, v in score.items() if v == best)
    return {"rule": QB_SELECTION_RULE, "geomean_ttft_ms": {str(k): score[k] for k in QUERY_BLOCKS},
            "selected_query_block": chosen,
            "geomean_relative_to_selected": {str(k): score[k] / best for k in QUERY_BLOCKS}}


# --qb-tiebreak: activated ONLY because the valid Stage-1 winner / runner-up gap was <= 1%.
# Fixed ABBA order of four fresh-engine series (fixed before the run); q_len 512 and 2048 only.
STAGE1_SELECTED_QUERY_BLOCK = 32
TIEBREAK_CANDIDATES = (16, 32)
TIEBREAK_GAP = 0.01
TIEBREAK_FALLBACK_QB = 16
TIEBREAK_SERIES = [("rabit_shared_qb16_A1", "rabit_kv2", "shared_decode", 16),
                   ("rabit_shared_qb32_B1", "rabit_kv2", "shared_decode", 32),
                   ("rabit_shared_qb32_B2", "rabit_kv2", "shared_decode", 32),
                   ("rabit_shared_qb16_A2", "rabit_kv2", "shared_decode", 16)]
TIEBREAK_RULE = ("per QB16/QB32 and q_len 512/2048: median TTFT over its two ABBA series; score = sqrt(median_512 * "
                 "median_2048); relative_gap = |score16 - score32| / min(score16, score32); if relative_gap > 0.01 "
                 "choose the lower score, else choose QB16")
TIEBREAK_NOTE_IF_16 = ("Stage 1's pre-registered single-sample rule selected QB32 by 0.07%; because the margin was "
                       "<=1%, the pre-registered replication stage was activated, and the final fixed configuration "
                       "was selected by the replication rule.")


def select_tiebreak(ttft_ms: dict[str, dict[int, float]]) -> dict:
    """Apply TIEBREAK_RULE to {series_label: {q_len: ttft_ms}} for the four TIEBREAK_SERIES."""
    if sorted(ttft_ms) != sorted(l for l, _, _, _ in TIEBREAK_SERIES):
        raise ValueError(f"need exactly the tie-break series, got {sorted(ttft_ms)}")
    qb_of = {l: qb for l, _, _, qb in TIEBREAK_SERIES}
    med, score = {}, {}
    for qb in TIEBREAK_CANDIDATES:
        rows = [ttft_ms[l] for l in ttft_ms if qb_of[l] == qb]
        for r in rows:
            if sorted(r) != QB_TUNING_Q_LENS or any(not (v > 0) for v in r.values()):
                raise ValueError(f"QB {qb}: need positive TTFT at q_len {QB_TUNING_Q_LENS}, got {r}")
        med[qb] = {q: statistics.median([r[q] for r in rows]) for q in QB_TUNING_Q_LENS}
        score[qb] = math.sqrt(med[qb][512] * med[qb][2048])
    gap = abs(score[16] - score[32]) / min(score[16], score[32])
    final = min(score, key=score.get) if gap > TIEBREAK_GAP else TIEBREAK_FALLBACK_QB
    return {"rule": TIEBREAK_RULE, "median_ttft_ms": {str(qb): {str(q): v for q, v in m.items()} for qb, m in med.items()},
            "score_ms": {str(k): v for k, v in score.items()}, "relative_gap": gap,
            "gap_exceeds_threshold": gap > TIEBREAK_GAP,
            "stage1_selected_query_block": STAGE1_SELECTED_QUERY_BLOCK,
            "final_tiebreak_selected_query_block": final,
            "note": TIEBREAK_NOTE_IF_16 if final == 16 else None}


def stage1_precondition() -> dict:
    """The tie-break is valid only for the committed Stage-1 evidence with a <= 1% winner / runner-up gap."""
    a = json.loads((QB_TUNING_DIR / "benchmark_analysis.json").read_text(encoding="utf-8"))
    sel = a["selection"]
    g = sorted(sel["geomean_ttft_ms"].values())
    gap = (g[1] - g[0]) / g[0]
    if not (a["all_integrity_passed"] and sel["selected_query_block"] == STAGE1_SELECTED_QUERY_BLOCK
            and gap <= TIEBREAK_GAP and run_git("status", "--short", "--", rel(QB_TUNING_DIR)) == ""
            and run_git("ls-files", rel(QB_TUNING_DIR / "benchmark_analysis.json"))):
        raise RuntimeError("tie-break precondition failed: committed valid Stage-1 evidence with a <=1% gap required")
    runner_up = sorted(sel["geomean_ttft_ms"], key=sel["geomean_ttft_ms"].get)[1]
    return {"stage1_selected_query_block": sel["selected_query_block"], "stage1_runner_up": int(runner_up),
            "stage1_relative_gap": gap, "stage1_evidence": rel(QB_TUNING_DIR)}


def set_mode_qb_tiebreak() -> None:
    global SERIES, Q_LENS, POINTS
    SERIES, Q_LENS = list(TIEBREAK_SERIES), list(QB_TUNING_Q_LENS)
    POINTS = [FIRST_CHUNK + q for q in Q_LENS]
    set_out_dir(QB_TIEBREAK_DIR)


# Reviewed lock (after the accepted tie-break). The final benchmark still reads its QUERY_BLOCK from the committed
# tie-break evidence; this constant only refuses a run if the evidence and the reviewed lock ever disagree.
LOCKED_FINAL_QUERY_BLOCK = 32


def final_query_block(require_committed: bool) -> dict:
    """The final fixed QUERY_BLOCK is read from the tie-break evidence (never chosen by hand, never retuned)."""
    a = json.loads((QB_TIEBREAK_DIR / "benchmark_analysis.json").read_text(encoding="utf-8"))
    sel = a.get("selection") or {}
    qb = sel.get("final_tiebreak_selected_query_block")
    if not (a["all_integrity_passed"] and qb in TIEBREAK_CANDIDATES
            and sel.get("stage1_selected_query_block") == STAGE1_SELECTED_QUERY_BLOCK):
        raise RuntimeError("final benchmark precondition failed: no valid tie-break selection")
    if qb != LOCKED_FINAL_QUERY_BLOCK:  # consistency guard only; the value used is the evidence's
        raise RuntimeError(f"tie-break evidence selects QB{qb}, but the reviewed lock is QB{LOCKED_FINAL_QUERY_BLOCK}")
    committed = (run_git("status", "--short", "--", rel(QB_TIEBREAK_DIR)) == ""
                 and bool(run_git("ls-files", rel(QB_TIEBREAK_DIR / "benchmark_analysis.json"))))
    if require_committed and not committed:
        raise RuntimeError("final benchmark precondition failed: tie-break evidence must be archived (committed)")
    return {"final_query_block": qb, "stage1_selected_query_block": sel["stage1_selected_query_block"],
            "tiebreak_relative_gap": sel["relative_gap"], "tiebreak_evidence": rel(QB_TIEBREAK_DIR),
            "tiebreak_evidence_committed": committed}


def set_mode_final(qb: int) -> None:
    global SERIES, Q_LENS, POINTS
    SERIES = [("rabit_reference", "rabit_kv2", "reference", qb), ("rabit_tile32", "rabit_kv2", "tile32", qb),
              (f"rabit_shared_qb{qb}", "rabit_kv2", "shared_decode", qb)]
    Q_LENS = [32, 512, 2048, 4096, 8192]
    POINTS = [FIRST_CHUNK + q for q in Q_LENS]
    set_out_dir(FINAL_DIR)


def set_mode_qb_tuning() -> None:
    global SERIES, Q_LENS, POINTS
    SERIES, Q_LENS = list(QB_TUNING_SERIES), list(QB_TUNING_Q_LENS)
    POINTS = [FIRST_CHUNK + q for q in Q_LENS]
    set_out_dir(QB_TUNING_DIR)


SCOPE = ("Stage3C shared_decode kernel-tuning BENCHMARK evidence only; NOT Experiment 5 evidence; one measured "
         "request per point; descriptive only. QUERY_BLOCK comparison is kernel tuning, not an ablation. The 32K "
         "(q_len 16352) point is not part of this run.")


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


def triton_attn_only_adds_shared_decode(baseline_src: str, current_src: str) -> bool:
    """current == baseline after removing exactly the shared_decode dispatch and moving the selector import back."""
    cur, base = ast.parse(current_src), ast.parse(baseline_src)
    sd_imp = [n for n in cur.body if isinstance(n, ast.ImportFrom)
              and n.module == "vllm.v1.attention.ops.rabit_kv2_stage3c_shared_decode"]
    t32_imp = [n for n in cur.body if isinstance(n, ast.ImportFrom)
               and n.module == "vllm.v1.attention.ops.rabit_kv2_stage3c_tile32"]
    if len(sd_imp) != 1 or len(t32_imp) != 1 or \
            [a.name for a in sd_imp[0].names] != ["rabit2_stage3c_forward_shared_decode", "rabit2_stage3c_impl"] or \
            [a.name for a in t32_imp[0].names] != ["rabit2_stage3c_forward_tile32"]:
        return False
    cur.body.remove(sd_imp[0])
    t32_imp[0].names.append(ast.alias(name="rabit2_stage3c_impl"))
    ifs = [n for n in ast.walk(cur) if isinstance(n, ast.If)
           and "rabit2_stage3c_forward_shared_decode" in ast.unparse(n.test)]
    if len(ifs) != 1 or [ast.unparse(s) for s in ifs[0].body] != ["continue"] or ifs[0].orelse:
        return False
    test = ast.unparse(ifs[0].test)
    if not test.startswith("q_len > 1 and rabit2_stage3c_impl() == 'shared_decode' and "
                           "rabit2_stage3c_forward_shared_decode(runtime, q_seq, k_seq, v_seq, kv_cache, "
                           "block_table_row, output[q0:q1], self.scale)"):
        return False
    for parent in ast.walk(cur):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(parent, field, None)
            if isinstance(stmts, list) and ifs[0] in stmts:
                stmts.remove(ifs[0])
    return ast.dump(cur) == ast.dump(base)


def verify_equivalence(bench_manifest: dict) -> dict:
    w, wb = (ast.parse(p.read_text(encoding="utf-8")) for p in (WORKER, rt.WORKER))
    m, mb = (ast.parse(p.read_text(encoding="utf-8")) for p in (MODAL_APP, rt.MODAL_APP))
    if rd._const(w, "BASE_ENGINE_KWARGS") != rd._const(wb, "BASE_ENGINE_KWARGS"):
        raise RuntimeError("BASE_ENGINE_KWARGS differ from the tile32 benchmark worker")
    if tuple(rd._const(w, "ALLOWED_KV_CACHE_DTYPES")) != ("bfloat16", "rabit_kv2"):
        raise RuntimeError("worker dtypes changed")
    main_w, main_b = _function(w, "main"), _function(wb, "main")
    if ast.dump(rd._nested(main_w, "one")) != ast.dump(rd._nested(main_b, "one")):
        raise RuntimeError("timed region one() differs from the tile32 benchmark worker")
    for target in ("tok", "bos", "filler", "sp", "prompt", "kwargs", "plan"):
        if rd._assigns(main_w, target) != rd._assigns(main_b, target):
            raise RuntimeError(f"'{target} = ...' differs from the tile32 benchmark worker")
    src = ast.unparse(main_w)
    if "collective_rpc" in src or "VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK" not in src:
        raise RuntimeError("worker must export the query block and must not use engine RPC")
    if ast.dump(_module_assign(m, "image")) != ast.dump(_module_assign(mb, "image")):
        raise RuntimeError("Modal image differs from the tile32 benchmark (canonical) image")
    for fn in ("_emit", "_sha256_file", "_gpu_query", "_compute_apps", "_gpu_state", "_require_clean", "_run_guarded"):
        if ast.dump(_function(mb, fn)) != ast.dump(_function(m, fn)):
            raise RuntimeError(f"Modal helper {fn}() differs from the tile32 benchmark app")
    for const, mine in (("REQUEST_CAP_S", REQUEST_CAP_S), ("SERIES_TIMEOUT_S", SERIES_TIMEOUT_S),
                        ("TILE32_TEST_TIMEOUT_S", TILE32_TEST_TIMEOUT_S),
                        ("SHARED_TEST_TIMEOUT_S", SHARED_TEST_TIMEOUT_S), ("GATE_TIMEOUT_S", GATE_TIMEOUT_S),
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
    if not triton_attn_only_adds_shared_decode(rp._git_show(PROFILED_COMMIT, TRITON_ATTN),
                                               TRITON_ATTN.read_text(encoding="utf-8")):
        raise RuntimeError("triton_attn.py differs from the profiled version beyond the shared_decode dispatch")
    prov = bench_manifest["provenance"]
    if sha256(TILE32_MODULE) != prov["tile32_module_sha256"] or sha256(TILE32_TESTS) != prov["tile32_tests_sha256"]:
        raise RuntimeError("tile32 module / tests differ from the benchmarked (frozen) version")
    if run_git("diff", "--name-only", PROFILED_COMMIT, "--", rel(PROFILE_MODULE)):
        raise RuntimeError("profiler module changed since the profiled commit")
    return {"engine_kwargs_equal_tile32_benchmark": True, "timed_region_equal_tile32_benchmark": True,
            "helpers_equal_tile32_benchmark": True, "image_equal_tile32_benchmark": True,
            "triton_attn_equal_profiled_modulo_shared_decode_dispatch": True, "triton_attn_baseline_ref":
                PROFILED_COMMIT, "tile32_module_and_tests_frozen": True, "profiler_module_unchanged": True,
            "watchdog_budget_s": WATCHDOG_BUDGET_S, "modal_backstop_s": backstop}


def preflight(dry_run: bool) -> dict:
    assert_protected_paths_clean("preflight")
    if sha256(RABIT_KV2) != EXPECTED_RABIT_SHA256_LF:
        raise RuntimeError("rabit_kv2.py is not the frozen committed content")
    eq = verify_equivalence(json.loads(BENCH_MANIFEST.read_text(encoding="utf-8")))
    uncommitted = run_git("status", "--short", "--", *[rel(p) for p in MUST_BE_COMMITTED])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: shared_decode code has uncommitted changes:\n" + uncommitted)
    leftovers = sorted(p.name for p in OUT_DIR.iterdir()) if OUT_DIR.is_dir() else []
    if leftovers and not dry_run:
        raise RuntimeError(f"Refusing to run: {rel(OUT_DIR)} is not empty ({leftovers})")
    return {"git_head": run_git("rev-parse", "HEAD"), "vllm_kvquant_tree": run_git("rev-parse", "HEAD:vllm-kvquant"),
            "rabit_kv2_sha256": sha256(RABIT_KV2), "tile32_module_sha256": sha256(TILE32_MODULE),
            "shared_decode_module_sha256": sha256(SHARED_MODULE), "shared_decode_tests_sha256": sha256(SHARED_TESTS),
            "triton_attn_sha256": sha256(TRITON_ATTN), "profile_module_sha256": sha256(PROFILE_MODULE),
            "runner_script_sha256": sha256(RUNNER_SCRIPT), "modal_app_sha256": sha256(MODAL_APP),
            "worker_sha256": sha256(WORKER), "equivalence": eq, "protected_paths": [rel(p) for p in PROTECTED_PATHS],
            "prior_evidence_sha256_raw": prior_evidence_digest(), "uncommitted_files": uncommitted or None,
            "existing_output_files": leftovers or None}


def build_snapshot() -> dict:
    out = Path(tempfile.mkdtemp(prefix="s3d_vllm_snapshot_")) / "vllm_kvquant_snapshot.zip"
    run_git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(out), "HEAD:vllm-kvquant")
    return {"path": str(out), "sha256": sha256_raw(out), "bytes": out.stat().st_size}


def build_command(correctness_only: bool) -> list[str]:
    cmd = [sys.executable, "-m", "modal", "run", str(MODAL_APP),
           "--series", ",".join(f"{l}={d}:{i}:{q}" for l, d, i, q in SERIES),
           "--points", ",".join(map(str, POINTS)), "--conditioning-prompt", str(CONDITIONING_PROMPT)]
    return cmd + (["--correctness-only"] if correctness_only else [])


def expected_runtime(correctness_only: bool) -> dict:
    """Planning estimate only: reference / tile32 from the frozen tile32 benchmark; shared_decode bounded by tile32."""
    s = json.loads((rt.OUT_DIR / "summary.json").read_text(encoding="utf-8"))
    t = {p["second_chunk_q_len"]: p for p in s["points"]}
    if OUT_DIR == FINAL_DIR:
        ref = sum(t[q]["reference_ttft_ms"] for q in [32] + Q_LENS) / 1000.0
        til = sum(t[q]["tile32_ttft_ms"] for q in [32] + Q_LENS) / 1000.0
        fixed = {"image_rebuild_s": 600, "gate_s": 70, "tile32_tests_s": 50, "shared_decode_tests_s": 120,
                 "engine_init_s": 110 * len(SERIES)}
        req = {"reference_requests_s": round(ref), "tile32_requests_s": round(til),
               "shared_decode_requests_s_upper_if_no_faster_than_tile32": round(til)}
        return {**fixed, **req, "total_estimate_s": round(sum(fixed.values()) + sum(req.values()))}
    if OUT_DIR in (QB_TUNING_DIR, QB_TIEBREAK_DIR):
        fixed = {"image_rebuild_s": 600, "gate_s": 70, "tile32_tests_s": 50, "shared_decode_tests_s": 120,
                 "engine_init_s": 110 * len(SERIES)}
        per = sum(t[q]["tile32_ttft_ms"] for q in [32] + Q_LENS) / 1000.0
        req = round(per * len(SERIES))
        return {**fixed, "shared_decode_requests_s_upper_if_no_faster_than_tile32": req,
                "total_estimate_s": round(sum(fixed.values()) + req)}
    ref = sum(t[q]["reference_ttft_ms"] for q in [32] + Q_LENS) / 1000.0
    til = sum(t[q]["tile32_ttft_ms"] for q in [32] + Q_LENS) / 1000.0
    fixed = {"image_rebuild_s": 600, "gate_s": 70, "tile32_tests_s": 50, "shared_decode_tests_s": 300}
    series = {} if correctness_only else {"engine_init_s": 110 * len(SERIES), "reference_requests_s": round(ref),
                                          "tile32_requests_s": round(til),
                                          "shared_decode_requests_s_upper_if_no_faster_than_tile32":
                                              round(til * len(QUERY_BLOCKS))}
    return {**fixed, **series, "total_estimate_s": round(sum(fixed.values()) + sum(series.values())),
            "note": "planning estimate; largest single request (reference q=8192) ~130 s vs the 600 s cap"}


# ------------------------------------------------------------------ parsing
def demux(text: str):
    series = {l: [] for l, _, _, _ in SERIES}
    gate, t32, shared, top = [], [], [], []
    prefixes = {f"[series{k}:{l}] ": l for k, (l, _, _, _) in enumerate(SERIES, start=1)}
    for line in text.splitlines():
        for p, bucket in (("[gate] ", gate), ("[tile32-tests] ", t32), ("[shared-tests] ", shared)):
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
    return series, gate, t32, shared, top


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


def parse_shared_tests(lines: list[str]) -> dict:
    base = rt.parse_tests(lines)
    passed = sorted({m.group(1) for ln in lines if (m := re.match(r"^PASSED \S+::(\S+)", ln.strip()))})
    counts = [json.loads(ln.split("SHARED_DECODE_COUNTS=", 1)[1]) for ln in lines
              if ln.startswith("SHARED_DECODE_COUNTS=")]
    uniq = {(c["prefix"], c["q_len"], c["query_block"]): c for c in counts}
    return {**base, "passed_test_ids": len(passed), "model_limit_16352_passed": [t for t in EXPECTED_32K_TESTS
                                                                                 if t in passed],
            "decode_counts": sorted(uniq.values(), key=lambda c: (c["prefix"], c["q_len"], c["query_block"]))}


PROFILE_MARKERS = ("VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE", "VLLM_RABIT2_STAGE3C_PROFILE",
                   "RABIT2_STAGE3C_COMPONENT_PROFILE=", "RABIT2_STAGE3C_TILE32_PROFILE")


def integrity(series: dict, gate: dict, t32: dict, shared: dict, top: dict, correctness_only: bool,
              series_lines: dict | None = None) -> dict:
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
    add("tile32 exact suite: exit 0, 43 passed, none skipped/failed", "tile32_regression",
        (top.get("S3C_TILE_TESTS_EXIT", {}).get("returncode") == 0 and t32["passed"] == 43 and t32["failed"] == 0
         and t32["errors"] == 0 and t32["skipped"] == 0) if "S3C_TILE_TESTS_START" in top else NOT_RUN, t32)
    ran = "S3C_SHARED_TESTS_START" in top
    add(f"shared_decode exact suite: exit 0, {EXPECTED_SHARED_TESTS} passed, none skipped/failed", "shared_decode",
        (top.get("S3C_SHARED_TESTS_EXIT", {}).get("returncode") == 0 and shared["passed"] == EXPECTED_SHARED_TESTS
         == shared["passed_test_ids"] and shared["failed"] == 0 and shared["errors"] == 0 and shared["skipped"] == 0)
        if ran else NOT_RUN, {k: shared[k] for k in ("passed", "failed", "errors", "skipped", "passed_test_ids")})
    add("q_len 16352 model-limit chunk executed and exact for every QUERY_BLOCK", "shared_decode",
        shared["model_limit_16352_passed"] == EXPECTED_32K_TESTS if ran else NOT_RUN,
        shared["model_limit_16352_passed"])
    dc = shared["decode_counts"]
    add("kernel-side decode count == closed x qgroups x ceil(n / QUERY_BLOCK) (12 cases)", "decode_reuse",
        (len(dc) == 12 and all(c["measured_shared_decode_page_decodes"] == c["expected_shared_decode"]
                               < c["per_query_page_decodes_tile32_reference"] for c in dc)) if ran else NOT_RUN,
        dc or None)
    if correctness_only:
        add("correctness-only run completed", "completion", top["correctness_only_complete"])
    else:
        add("benchmark completed (all series)", "completion", top["complete"])
        add("no series watchdog timeout", "watchdog", not top["timeouts"], top["timeouts"] or None)
        base = top.get("S3C_GPU_BASELINE", {})
        for label, dtype, impl, qb in SERIES:
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
            chk(f"Stage3C implementation = {impl}, query block = {qb} (requested and reported)", "impl",
                si.get("requested") == impl == si.get("selector_reports")
                and si.get("query_block_requested") == qb == si.get("query_block_reports"), si or None)
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
            if series_lines is not None:
                chk("profiling OFF (no profiling env var seen by the engine, no profile records)", "profiling",
                    not any(mk in ln for ln in series_lines[label] for mk in PROFILE_MARKERS))
        rows = {l: {p["row"]["planned_prompt_tokens"]: p["row"] for p in series[l]["points"]
                    if p["begin"]["role"] == "measured"} for l, _, _, _ in SERIES}
        complete = all(len(v) == len(POINTS) for v in rows.values())
        add(f"greedy output tokens identical across all {len(SERIES)} series at every point", "equivalence",
            all(len({rows[l][p]["output_token_ids_sha256"] for l, _, _, _ in SERIES}) == 1 for p in POINTS)
            if complete else NOT_EVALUATED)
        bench = {p["prompt_tokens"]: p for p in json.loads((rt.OUT_DIR / "summary.json").read_text(
            encoding="utf-8"))["points"]}
        add("output tokens identical to the frozen tile32-benchmark reference output at every point",
            "equivalence", all(rows[l][p]["output_token_ids_sha256"] == bench[p]["reference_output_token_ids_sha256"]
                               and rows[l][p]["prompt_token_ids_sha256"] == bench[p]["prompt_token_ids_sha256"]
                               for l, _, _, _ in SERIES for p in POINTS) if complete else NOT_EVALUATED)
        post = top.get("S3C_POST_RUN_GPU_STATE")
        add("GPU clean after run", "gpu_clean", NOT_RUN if not post else (
            not post["compute_apps"] and all(u <= b + 256 for u, b in zip(post["memory_used_mib"],
                                                                         base.get("memory_used_mib", [])))))
        cfg = {l: rd.series_config(series[l]) for l, _, _, _ in SERIES}
        add("engine configs identical across all series", "config",
            len({json.dumps(c, sort_keys=True, default=str) for c in cfg.values()}) == 1
            if all(cfg.values()) else NOT_EVALUATED)
    counts = {s: sum(1 for c in checks if c["state"] == s) for s in (PASSED, FAILED, NOT_RUN, NOT_EVALUATED)}
    return {"checks": checks, "counts": counts, "all_ok": counts[PASSED] == len(checks),
            "failed_categories": sorted({c["category"] for c in checks if c["state"] == FAILED})}


def final_analysis(series: dict, integ: dict) -> dict:
    rows = {impl: {p["row"]["planned_prompt_tokens"]: p["row"] for p in series[l]["points"]
                   if p["begin"]["role"] == "measured"} for l, _, impl, _ in SERIES}
    qb = SERIES[2][3]
    pts = []
    for p, q in zip(POINTS, Q_LENS):
        r, t, s_ = rows["reference"].get(p), rows["tile32"].get(p), rows["shared_decode"].get(p)
        row = {"prompt_tokens": p, "second_chunk_q_len": q}
        for k, v in (("reference", r), ("tile32", t), ("shared_decode", s_)):
            row[k] = None if v is None else {x: v[x] for x in ("ttft_ms", "tpot_ms", "wall_ms",
                                                               "output_token_ids_sha256")}
        if r and t and s_:
            row["shared_decode_ttft_over_reference"] = s_["ttft_ms"] / r["ttft_ms"]
            row["shared_decode_ttft_over_tile32"] = s_["ttft_ms"] / t["ttft_ms"]
            row["reference_ttft_over_shared_decode"] = r["ttft_ms"] / s_["ttft_ms"]
            row["shared_decode_wall_over_reference"] = s_["wall_ms"] / r["wall_ms"]
            row["shared_decode_tpot_delta_ms_vs_reference"] = s_["tpot_ms"] - r["tpot_ms"]
        pts.append(row)
    return {"scope": SCOPE + f" Final benchmark: reference, tile32, shared_decode(QB{qb} fixed by the tie-break).",
            "experiment5_evidence": False, "all_integrity_passed": integ["all_ok"],
            "integrity_counts": integ["counts"], "fixed_query_block": qb, "points": pts,
            "not_claimed": ["Experiment 5 results", "a complexity law", "q_len 16352 latency",
                            "QUERY_BLOCK retuning at 4096 / 8192"]}


def qb_tiebreak_analysis(series: dict, integ: dict) -> dict:
    rows = {l: {p["row"]["planned_prompt_tokens"] - FIRST_CHUNK: p["row"] for p in series[l]["points"]
                if p["begin"]["role"] == "measured"} for l, _, _, _ in SERIES}
    out = {"scope": SCOPE + " QUERY_BLOCK tie-break (QB16 vs QB32, ABBA, q_len 512 and 2048).",
           "experiment5_evidence": False, "all_integrity_passed": integ["all_ok"], "integrity_counts": integ["counts"],
           "series_order": [l for l, _, _, _ in SERIES],
           "per_series": [{"series": l, "query_block": qb, **{f"{k}_{q}": rows[l][q][k] for q in Q_LENS
                                                              for k in ("ttft_ms", "wall_ms", "tpot_ms")}}
                          for l, _, _, qb in SERIES if sorted(rows[l]) == Q_LENS]}
    out["selection"] = select_tiebreak({l: {q: r["ttft_ms"] for q, r in rows[l].items()} for l in rows}) \
        if integ["all_ok"] else None
    out["stage1_selected_query_block"] = STAGE1_SELECTED_QUERY_BLOCK
    out["not_claimed"] = ["speedup vs reference or tile32 (not run here)", "Experiment 5 results",
                          "q_len 4096 / 8192 / 16352 behaviour"]
    return out


def qb_tuning_analysis(series: dict, integ: dict) -> dict:
    rows = {qb: {p["row"]["planned_prompt_tokens"] - FIRST_CHUNK: p["row"] for p in series[l]["points"]
                 if p["begin"]["role"] == "measured"} for l, _, _, qb in SERIES}
    out = {"scope": SCOPE + " QUERY_BLOCK tuning subset: shared_decode only, q_len 512 and 2048.",
           "experiment5_evidence": False, "all_integrity_passed": integ["all_ok"], "integrity_counts": integ["counts"],
           "per_query_block": {str(qb): {str(q): {k: r[k] for k in ("ttft_ms", "tpot_ms", "wall_ms",
                                                                  "output_token_ids_sha256")}
                                         for q, r in sorted(rows[qb].items())} for qb in sorted(rows)}}
    out["selection"] = select_query_block({qb: {q: r["ttft_ms"] for q, r in rows[qb].items()} for qb in rows}) \
        if integ["all_ok"] else None
    out["not_claimed"] = ["speedup vs reference or tile32 (not run here)", "Experiment 5 results",
                          "q_len 4096 / 8192 / 16352 behaviour"]
    return out


def analysis(series: dict, shared: dict, integ: dict, correctness_only: bool) -> dict:
    out = {"scope": SCOPE, "experiment5_evidence": False, "all_integrity_passed": integ["all_ok"],
           "integrity_counts": integ["counts"], "request_guard_note": REQUEST_GUARD_NOTE,
           "component_profiling_enabled": False, "decode_counts": shared["decode_counts"]}
    if correctness_only:
        return out
    rows = {l: {p["row"]["planned_prompt_tokens"]: p["row"] for p in series[l]["points"]
                if p["begin"]["role"] == "measured"} for l, _, _, _ in SERIES}
    pts = []
    for p, q in zip(POINTS, Q_LENS):
        row = {"prompt_tokens": p, "second_chunk_q_len": q, "series": {}}
        for l, _, impl, qb in SERIES:
            v = rows[l].get(p)
            row["series"][l] = None if v is None else {
                "impl": impl, "query_block": qb if impl == "shared_decode" else None,
                **{k: v[k] for k in ("ttft_ms", "tpot_ms", "wall_ms", "output_token_ids_sha256")}}
        ref, til = row["series"]["rabit_reference"], row["series"]["rabit_tile32"]
        row["shared_decode_ratios"] = {
            f"qb{qb}": None if not (ref and til and row["series"][f"rabit_shared_qb{qb}"]) else {
                "ttft_over_reference": row["series"][f"rabit_shared_qb{qb}"]["ttft_ms"] / ref["ttft_ms"],
                "ttft_over_tile32": row["series"][f"rabit_shared_qb{qb}"]["ttft_ms"] / til["ttft_ms"],
                "wall_over_reference": row["series"][f"rabit_shared_qb{qb}"]["wall_ms"] / ref["wall_ms"],
                "tpot_delta_ms_vs_reference": row["series"][f"rabit_shared_qb{qb}"]["tpot_ms"] - ref["tpot_ms"]}
            for qb in QUERY_BLOCKS}
        ok = [(qb, r["ttft_over_reference"]) for qb, r in
              ((int(k[2:]), v) for k, v in row["shared_decode_ratios"].items()) if r]
        row["fastest_query_block_by_ttft"] = min(ok, key=lambda x: x[1])[0] if ok else None
        pts.append(row)
    out["points"] = pts
    out["note"] = ("single measured request per point; QUERY_BLOCK choice is kernel tuning and is made at review, "
                   "not by this script")
    out["not_claimed"] = ["Experiment 5 results", "a complexity law", "correctness beyond the executed tests",
                          "the 32K (q_len 16352) latency point"]
    return out


def analyze(text: str, correctness_only: bool, write: bool):
    ser_lines, gate_lines, t32_lines, shared_lines, top_lines = demux(text)
    series = {l: rd.parse_series(ser_lines[l]) for l, _, _, _ in SERIES}
    gate = r5.parse_gate(gate_lines)
    t32, shared = rt.parse_tests(t32_lines), parse_shared_tests(shared_lines)
    top = parse_top(top_lines)
    integ = integrity(series, gate, t32, shared, top, correctness_only, ser_lines)
    if not correctness_only and OUT_DIR == FINAL_DIR:
        an = final_analysis(series, integ)
    elif not correctness_only and OUT_DIR == QB_TIEBREAK_DIR:
        an = qb_tiebreak_analysis(series, integ)
    elif not correctness_only and OUT_DIR == QB_TUNING_DIR:
        an = qb_tuning_analysis(series, integ)
    else:
        an = analysis(series, shared, integ, correctness_only)
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        GATE_LOG.write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
        TILE32_TESTS_LOG.write_text("\n".join(t32_lines) + "\n", encoding="utf-8")
        SHARED_TESTS_LOG.write_text("\n".join(shared_lines) + "\n", encoding="utf-8")
        if not correctness_only:
            for l, _, _, _ in SERIES:
                (OUT_DIR / f"{l}_series.log").write_text("\n".join(ser_lines[l]) + "\n", encoding="utf-8")
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
    code = stream_command(m["command"], SESSION_LOG, {"S3D_VLLM_SNAPSHOT": snap["path"]})
    m["modal_returncode"] = code
    m["stage"] = "parse"
    integ, _ = analyze(SESSION_LOG.read_text(encoding="utf-8", errors="replace"), correctness_only, write=True)
    m.pop("stage")
    m["integrity_counts"] = integ["counts"]
    if code != 0 or not integ["all_ok"]:
        finalize(m, "failed", {"stage": (integ["failed_categories"] or ["modal_nonzero_exit"])[0],
                               "failed_categories": integ["failed_categories"]})
        raise SystemExit(f"\nSHARED_DECODE RUN STOPPED; partial evidence kept in {rel(OUT_DIR)}/.")
    finalize(m, "completed", None)
    print(f"\nSHARED_DECODE {'CORRECTNESS' if correctness_only else 'BENCHMARK'} COMPLETED. Analysis: {ANALYSIS}")
    return 0


def main(argv: list[str] | None = None) -> int:
    make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--correctness-only", action="store_true")
    ap.add_argument("--qb-tuning", action="store_true",
                    help="shared_decode QUERY_BLOCK 4/8/16/32 at q_len 512 and 2048 only")
    ap.add_argument("--qb-tiebreak", action="store_true",
                    help="QB16 vs QB32 ABBA tie-break at q_len 512 and 2048 only")
    ap.add_argument("--final", action="store_true",
                    help="final benchmark: reference, tile32, shared_decode(tie-break QB) at q_len 32..8192")
    a = ap.parse_args(argv)
    if sum((a.correctness_only, a.qb_tuning, a.qb_tiebreak, a.final)) > 1:
        raise SystemExit("--correctness-only, --qb-tuning, --qb-tiebreak and --final are exclusive")
    stage1 = final = None
    if a.final:
        final = final_query_block(require_committed=not a.dry_run)
        set_mode_final(final["final_query_block"])
        print(f"FINAL benchmark with the tie-break QUERY_BLOCK: {json.dumps(final)}")
        if not final["tiebreak_evidence_committed"]:
            print("  WARNING (dry-run only): tie-break evidence is not archived yet; a real run refuses.")
    if a.qb_tiebreak:
        stage1 = stage1_precondition()
        set_mode_qb_tiebreak()
        print(f"QB tie-break (ABBA {[l for l, _, _, _ in SERIES]}); stage 1: {json.dumps(stage1)}")
        print(f"Pre-registered tie-break rule: {TIEBREAK_RULE}")
    if a.correctness_only:
        set_out_dir(CORRECTNESS_DIR)
    if a.qb_tuning:
        set_mode_qb_tuning()
        print(f"QB tuning; pre-registered selection rule: {QB_SELECTION_RULE}")
    print("RABIT-KV Stage3C shared_decode kernel-tuning BENCHMARK (not Experiment 5 evidence)")
    print(f"Mode: {'correctness-only (gate + tile32 + shared_decode exact suites, no timing)' if a.correctness_only else 'full'}")
    print(f"Series: {SERIES}; conditioning {CONDITIONING_PROMPT} (unmeasured)")
    print("Points (prompt -> q_len): " + ", ".join(f"{p}->{q}" for p, q in zip(POINTS, Q_LENS)) + "; no 16352 point")
    print(f"Request guard: {REQUEST_GUARD_NOTE}")
    print(f"  (series watchdog {SERIES_TIMEOUT_S}s; watchdog budget {WATCHDOG_BUDGET_S}s; "
          f"Modal function timeout {MODAL_FUNCTION_TIMEOUT_S}s)")
    prov = preflight(a.dry_run)
    print("Preflight OK:", json.dumps({k: prov[k] for k in ("git_head", "rabit_kv2_sha256", "tile32_module_sha256",
                                                             "shared_decode_module_sha256", "triton_attn_sha256")},
                                      indent=1))
    print("  equivalence:", json.dumps(prov["equivalence"]))
    print("  expected runtime:", json.dumps(expected_runtime(a.correctness_only)))
    if prov["uncommitted_files"]:
        print("  WARNING (dry-run only): uncommitted files:\n    " + prov["uncommitted_files"].replace("\n", "\n    "))
    print("Local command:\n  " + " ".join(build_command(a.correctness_only)))
    if a.dry_run:
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    m = {"benchmark": "Stage3C shared_decode kernel tuning", "scope": SCOPE, "correctness_only": a.correctness_only,
         "qb_tuning": a.qb_tuning, "qb_selection_rule": QB_SELECTION_RULE if a.qb_tuning else None,
         "qb_tiebreak": a.qb_tiebreak, "tiebreak_rule": TIEBREAK_RULE if a.qb_tiebreak else None,
         "stage1": stage1, "final": final,
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
        raise SystemExit(f"\nSHARED_DECODE RUNNER FAILED: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
