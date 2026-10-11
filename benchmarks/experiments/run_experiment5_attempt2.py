"""
RABIT-KV MLSys 2027 -- Experiment 5 ATTEMPT 2: BF16 vs RABIT-KV context-length
scaling with the finalized RABIT Stage3C implementation.

This is the frozen Experiment 5 runner (run_experiment5_context_scaling.py,
imported UNCHANGED as `r5`) with ONLY these rebindings / additions:
  * outputs go to results/mlsys2027/context_scaling/attempt_2/ (must be absent /
    empty); results/mlsys2027/context_scaling/failed_attempt_1/ is protected and
    hashed before and after the run;
  * Modal app exp5_attempt2_modal.py and worker exp5_attempt2_engine_worker.py
    (frozen app / worker + explicit Stage3C selection, per-cell watchdog, and
    extra in-container checks -- proved by AST below);
  * every rabit_kv2 cell runs the finalized Stage3C: shared_decode with
    QUERY_BLOCK read from the committed tie-break evidence (locked 32), set
    explicitly before engine start and verified (requested == effective);
    bfloat16 cells carry no Stage3C selection;
  * watchdogs: gate 600 s and every cell 900 s, EXCEPT B32768 = 3600 s; Modal
    backstop 17100 s (> 600 + 13 x 900 + 3600 = 15900 s). The B32768 extension is
    the sole runtime-safety change (canonical q_len 16352 feasibility: 115.6687 s
    wall, 20-request projection 2313.4 s). The absence of a per-request guard
    is inherited unchanged from the frozen Experiment 5 protocol;
  * extra integrity checks: Stage3C selection per cell, per-cell watchdog
    values, the pre-established B32768 output-token hash for all 20 requests,
    and the expected allocator capacities (BF16 393024, RABIT 2074592 tokens).
Contexts, order, warmups (5), measured reps (15), prompts, output length,
engine settings, metrics and the failed-cell policy are the frozen protocol.

Usage:
    python benchmarks/mlsys2027/run_experiment5_attempt2.py --dry-run
    python benchmarks/mlsys2027/run_experiment5_attempt2.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment5_context_scaling as r5  # noqa: E402  (frozen Experiment 5 runner; never modified)
import run_stage3c_shared_decode_benchmark as rsd  # noqa: E402  (committed; tie-break evidence reader)
from run_experiment3_deployment import (  # noqa: E402
    FAILED, PASSED, _function, _module_assign, flatten, make_console_encoding_safe, now, rel, run_git, sha256,
)

HERE = r5.HERE
ROOT = r5.ROOT
FROZEN_RUNNER, FROZEN_MODAL_APP, FROZEN_WORKER = r5.RUNNER_SCRIPT, r5.MODAL_APP, r5.WORKER
RUNNER_SCRIPT = Path(__file__).resolve()
MODAL_APP = HERE / "exp5_attempt2_modal.py"
WORKER = HERE / "exp5_attempt2_engine_worker.py"
OFFLINE_TESTS = HERE / "test_experiment5_attempt2.py"
CONTEXT_SCALING = ROOT / "results" / "mlsys2027" / "context_scaling"
ATTEMPT_DIR = CONTEXT_SCALING / "attempt_2"
FAILED_ATTEMPT_1 = CONTEXT_SCALING / "failed_attempt_1"
DIAG = ROOT / "results" / "mlsys2027" / "diagnostics"
STAGE3C_EVIDENCE = [DIAG / n for n in ("stage3c_cliff", "stage3c_tile32_correctness", "stage3c_tile32_benchmark",
                                       "stage3c_component_profile_correctness", "stage3c_component_profile",
                                       "stage3c_shared_decode_correctness", "stage3c_shared_decode_qb_tuning",
                                       "stage3c_shared_decode_qb_tiebreak", "stage3c_shared_decode_final_benchmark",
                                       "stage3c_shared_decode_q16352_feasibility")]
FEASIBILITY_ANALYSIS = rsd.FEASIBILITY_DIR / "benchmark_analysis.json"

# Finalized, locked RABIT Stage3C configuration.
FINAL_STAGE3C_IMPL = "shared_decode"
B32768_LABEL = "B32768"
B32768_LEG_TIMEOUT_S = 3600
MODAL_FUNCTION_TIMEOUT_S = 17100
WATCHDOG_BUDGET_S = r5.GATE_TIMEOUT_S + (len(r5.ALL_CELLS) - 1) * r5.LEG_TIMEOUT_S + B32768_LEG_TIMEOUT_S  # 15900
FEASIBILITY_WALL_S = 115.6687
FEASIBILITY_PROJECTION_S = 2313.4
EXPECTED_CAPACITY = {r5.A: 393024, r5.B: 2074592}
WATCHDOG_CHANGE_REASON = (
    "Pre-run canonical q16352 feasibility measured 115.6687 s wall; unchanged 20-request B32768 protocol projects "
    "to 2313.4 s before engine startup, so the historical 900 s watchdog would terminate a valid cell.")
PER_REQUEST_GUARD_NOTE = (
    "No per-request guard: inherited unchanged from the frozen Experiment 5 protocol (gate 600 s and cell watchdog "
    "only). The B32768 cell-watchdog extension (900 s -> 3600 s) is the sole runtime-safety change, justified by the "
    "canonical 115.6687 s q_len 16352 feasibility measurement.")


def leg_timeout(label: str) -> int:
    return B32768_LEG_TIMEOUT_S if label == B32768_LABEL else r5.LEG_TIMEOUT_S


def final_stage3c() -> dict:
    """QUERY_BLOCK from the committed tie-break evidence; expected B32768 output hash from the committed feasibility
    evidence (which matched the frozen failed attempt). Never chosen by hand, never retuned."""
    fin = rsd.final_query_block(require_committed=True)
    committed = lambda d: run_git("status", "--short", "--", rel(d)) == "" and bool(run_git("ls-files", rel(d)))  # noqa: E731
    for d in (rsd.FINAL_DIR, rsd.FEASIBILITY_DIR):
        if not committed(d):
            raise RuntimeError(f"precondition: {rel(d)} must be committed")
    fa = json.loads(FEASIBILITY_ANALYSIS.read_text(encoding="utf-8"))
    req = fa["request"]
    exp5 = rsd.exp5_b32768_expectations()
    if not (fa["all_integrity_passed"] and req["prompt_tokens"] == 32736 and req["jit_lines"] == 0
            and req["prompt_hash_matches_frozen_exp5"] and req["output_hash_matches_frozen_exp5"]
            and fa["query_block"] == fin["final_query_block"]
            and req["prompt_token_ids_sha256"] == exp5["prompt_token_ids_sha256"]
            and req["output_token_ids_sha256"] == exp5["output_token_ids_sha256"]
            and abs(req["wall_ms"] / 1000.0 - FEASIBILITY_WALL_S) < 1e-3):
        raise RuntimeError("precondition: committed q16352 feasibility evidence is not the accepted run")
    return {"impl": FINAL_STAGE3C_IMPL, "query_block": fin["final_query_block"],
            "query_block_source": fin["tiebreak_evidence"], "stage1_selected_query_block": fin["stage1_selected_query_block"],
            "b32768_expected_prompt_sha256": req["prompt_token_ids_sha256"],
            "b32768_expected_output_sha256": req["output_token_ids_sha256"],
            "expected_output_source": rel(FEASIBILITY_ANALYSIS),
            "feasibility_wall_s": req["wall_ms"] / 1000.0}


# ------------------------------------------------------------------ rebind the frozen runner (paths / files)
r5.RUNNER_SCRIPT = RUNNER_SCRIPT
r5.MODAL_APP = MODAL_APP
r5.WORKER = WORKER
r5.OUT_DIR = ATTEMPT_DIR
r5.SESSION_LOG = ATTEMPT_DIR / "modal_session.log"
r5.GATE_LOG = ATTEMPT_DIR / "correctness_gate.log"
r5.MANIFEST = ATTEMPT_DIR / "manifest.json"
r5.CONFIG_DIFF = ATTEMPT_DIR / "matched_config_diff.json"
r5.INTEGRITY = ATTEMPT_DIR / "integrity_check.json"
r5.SUMMARY = ATTEMPT_DIR / "context_scaling_summary.json"
r5.MUST_BE_COMMITTED = [RUNNER_SCRIPT, FROZEN_RUNNER, MODAL_APP, WORKER, OFFLINE_TESTS, r5.GATE, r5.WATCHDOG]
r5.PROTECTED_PATHS = [*r5.PROTECTED_PATHS, FAILED_ATTEMPT_1, FROZEN_RUNNER, FROZEN_MODAL_APP, FROZEN_WORKER,
                      HERE / "build_exp5_failure_analysis.py", HERE / "validate_exp5_failure_analysis.py",
                      *STAGE3C_EVIDENCE]
r5.EVIDENCE_DIRS = [*r5.EVIDENCE_DIRS, *STAGE3C_EVIDENCE]
r5.MODAL_FUNCTION_TIMEOUT_S = MODAL_FUNCTION_TIMEOUT_S
r5.WATCHDOG_BUDGET_S = WATCHDOG_BUDGET_S


def archived_attempts_digest() -> dict:
    return {f"{FAILED_ATTEMPT_1.name}/{k}": v for k, v in r5._digest_dir(FAILED_ATTEMPT_1).items()}


r5.archived_attempts_digest = archived_attempts_digest

_frozen_verify_equivalence = r5.verify_equivalence
_frozen_integrity = r5.integrity
_frozen_build_summary = r5.build_summary
_frozen_write_manifest = r5.write_manifest
_STAGE3C: dict = {}


def _assign_dump(tree, name):
    return ast.dump(_module_assign(tree, name))


def verify_equivalence() -> dict:
    """The frozen Experiment 5 proof (unchanged; its backstop check uses the frozen uniform-900 s budget formula),
    plus: attempt-2 app / worker equal the frozen Experiment 5 app / worker except the reviewed additions."""
    saved = r5.WATCHDOG_BUDGET_S
    r5.WATCHDOG_BUDGET_S = r5.GATE_TIMEOUT_S + len(r5.ALL_CELLS) * r5.LEG_TIMEOUT_S
    try:
        eq = _frozen_verify_equivalence()
    finally:
        r5.WATCHDOG_BUDGET_S = saved
    m, mf = (ast.parse(p.read_text(encoding="utf-8")) for p in (MODAL_APP, FROZEN_MODAL_APP))
    w, wf = (ast.parse(p.read_text(encoding="utf-8")) for p in (WORKER, FROZEN_WORKER))
    for name in ("image", "MODEL", "BASE_COMMIT", "EXPECTED_RABIT_SHA256_LF", "GPU_CLEAN_TOLERANCE_MIB",
                 "GPU_CLEAN_MAX_WAIT_S", "GPU_CLEAN_POLL_S", "GATE_TIMEOUT_S", "LEG_TIMEOUT_S",
                 "WORKLOAD_FAILURE_EXIT", "CONTINUABLE_FAILURE_KINDS", "OUTPUT_TOKENS"):
        if _assign_dump(m, name) != _assign_dump(mf, name):
            raise RuntimeError(f"attempt-2 Modal {name} differs from the frozen Experiment 5 app")
    for fn in (*r5.MODAL_SHARED_FUNCTIONS, "_parse_cell", "_Tee"):
        node = lambda t: next(n for n in t.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == fn)  # noqa: E731
        if ast.dump(node(m)) != ast.dump(node(mf)):
            raise RuntimeError(f"attempt-2 Modal {fn} differs from the frozen Experiment 5 app")
    if r5._const(m, "B32768_LEG_TIMEOUT_S") != B32768_LEG_TIMEOUT_S or r5._const(m, "LEG_TIMEOUT_S") != 900:
        raise RuntimeError("attempt-2 Modal watchdog constants differ from the reviewed values")
    lt = ast.unparse(_function(m, "_leg_timeout").body[0])
    if lt != "return B32768_LEG_TIMEOUT_S if label == 'B32768' else LEG_TIMEOUT_S":
        raise RuntimeError(f"attempt-2 per-cell watchdog rule changed: {lt}")
    src = ast.unparse(_function(m, "_run_cells"))
    if src.count("_leg_timeout(label)") != 2 or "LEG_TIMEOUT_S," in src:
        raise RuntimeError("attempt-2 Modal must use _leg_timeout() for both the recorded and enforced watchdog")
    backstop = next(ast.literal_eval(kw.value) for dec in _function(m, "sweep").decorator_list
                    for kw in getattr(dec, "keywords", []) if kw.arg == "timeout")
    if not (backstop == MODAL_FUNCTION_TIMEOUT_S > WATCHDOG_BUDGET_S == 15900):
        raise RuntimeError(f"Modal backstop {backstop} must exceed the attempt-2 watchdog budget {WATCHDOG_BUDGET_S}")
    for name in ("BASE_ENGINE_KWARGS", "ALLOWED_KV_CACHE_DTYPES", "OUTPUT_TOKENS", "WORKLOAD_FAILURE_EXIT"):
        if _assign_dump(w, name) != _assign_dump(wf, name):
            raise RuntimeError(f"attempt-2 worker {name} differs from the frozen Experiment 5 worker")
    for fn in ("emit", "enum_name", "gpu_memory_used_mib", "is_oom", "failure_kind"):
        if ast.dump(_function(w, fn)) != ast.dump(_function(wf, fn)):
            raise RuntimeError(f"attempt-2 worker {fn}() differs from the frozen Experiment 5 worker")
    mw, mfw = _function(w, "main"), _function(wf, "main")
    if ast.dump(r5._nested_function(mw, "one")) != ast.dump(r5._nested_function(mfw, "one")):
        raise RuntimeError("attempt-2 timed region one() differs from the frozen Experiment 5 worker")
    for target in (*r5.SHARED_PROMPT_TARGETS, "prompt", "kwargs", "effective", "capacity", "role", "markers"):
        if ast.dump(r5._assign(mw, target)) != ast.dump(r5._assign(mfw, target)):
            raise RuntimeError(f"attempt-2 worker '{target} = ...' differs from the frozen Experiment 5 worker")
    # Everything the frozen worker's main() does after the engine is built is unchanged.
    tail = lambda f: [ast.dump(s) for s in f.body[next(i for i, s in enumerate(f.body)  # noqa: E731
                                                     if ast.unparse(s).startswith("llm = LLM(")):]]
    if tail(mw) != tail(mfw):
        raise RuntimeError("attempt-2 worker differs from the frozen worker after engine construction")
    return {**eq, "attempt2_modal_equal_frozen_exp5_except_reviewed_additions": True,
            "attempt2_worker_equal_frozen_exp5_except_stage3c_selection": True,
            "per_cell_watchdog_rule": lt, "modal_function_backstop_timeout_s": backstop,
            "total_watchdog_budget_s": WATCHDOG_BUDGET_S,
            "frozen_budget_check_note": ("the frozen proof's backstop check ran with the frozen uniform budget "
                                         f"{r5.GATE_TIMEOUT_S} + {len(r5.ALL_CELLS)} x {r5.LEG_TIMEOUT_S}; the "
                                         f"attempt-2 budget {WATCHDOG_BUDGET_S} s is checked here")}


r5.verify_equivalence = verify_equivalence


# ------------------------------------------------------------------ strict EXP5_STAGE3C parsing (attempt-2 only)
# The frozen Experiment-5 parser only captures tags matching EXP5_[A-Z_]+ (no digits), so the worker's
# EXP5_STAGE3C record is invisible to it. Attempt 2 therefore parses that record itself from each cell's own raw
# lines: exact marker with an identifier boundary, exactly one record per cell, strict JSON and schema.
STAGE3C_MARKER = re.compile(r"(?<![A-Za-z0-9_])EXP5_STAGE3C=")
STAGE3C_LINE = re.compile(r"(?<![A-Za-z0-9_])EXP5_STAGE3C=(\{.*\})\s*$")
STAGE3C_ENV_KEYS = ("VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK", "VLLM_RABIT2_STAGE3C_IMPL")
PROFILE_ENV_KEYS = ("VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE", "VLLM_RABIT2_STAGE3C_PROFILE")
RABIT_STAGE3C_KEYS = {"applicable", "requested_impl", "effective_impl", "requested_query_block",
                      "effective_query_block", "env", "profiling_env", "shared_decode_module_sha256"}
BF16_STAGE3C_KEYS = {"applicable", "env", "profiling_env"}
ACCEPTED_SHARED_DECODE_MANIFEST = rsd.FINAL_DIR / "manifest.json"  # the accepted final held-out benchmark


def accepted_shared_decode_sha256() -> str:
    return json.loads(ACCEPTED_SHARED_DECODE_MANIFEST.read_text(encoding="utf-8"))["provenance"][
        "shared_decode_module_sha256"]


def parse_stage3c_records(lines: list[str]) -> list[dict]:
    """Every exact-marker EXP5_STAGE3C record in one cell's lines; a malformed exact-marker line raises."""
    out = []
    for ln in lines:
        if STAGE3C_MARKER.search(ln):
            m = STAGE3C_LINE.search(ln.rstrip("\r\n"))
            if not m:
                raise ValueError(f"malformed EXP5_STAGE3C line: {ln[:200]!r}")
            try:
                rec = json.loads(m.group(1))
            except json.JSONDecodeError as e:
                raise ValueError(f"EXP5_STAGE3C payload is not JSON: {e}") from e
            if not isinstance(rec, dict):
                raise ValueError("EXP5_STAGE3C payload is not an object")
            out.append(rec)
    return out


def validate_stage3c(dtype: str, records: list[dict] | None, error: str | None, cfg: dict,
                     accepted_sha: str) -> tuple[bool, str]:
    """(ok, reason) for one cell's EXP5_STAGE3C records, by the cell's own dtype."""
    if error is not None:
        return False, error
    if len(records) != 1:
        return False, f"expected exactly one EXP5_STAGE3C record, found {len(records)}"
    s = records[0]
    off = lambda d: isinstance(d, dict) and set(d) == set(PROFILE_ENV_KEYS) and all(  # noqa: E731
        v in (None, "0") for v in d.values())
    if dtype == r5.B:
        if set(s) != RABIT_STAGE3C_KEYS:
            return False, f"unexpected RABIT record fields {sorted(set(s) ^ RABIT_STAGE3C_KEYS)}"
        if not (s["applicable"] is True and s["requested_impl"] == s["effective_impl"] == cfg["impl"]):
            return False, f"implementation {s['requested_impl']!r} -> {s['effective_impl']!r} != {cfg['impl']!r}"
        if not (s["requested_query_block"] == s["effective_query_block"] == cfg["query_block"]):
            return False, f"QUERY_BLOCK {s['requested_query_block']} -> {s['effective_query_block']}"
        if s["env"] != {"VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK": str(cfg["query_block"]),
                        "VLLM_RABIT2_STAGE3C_IMPL": cfg["impl"]}:
            return False, f"selector environment {s['env']}"
        if not off(s["profiling_env"]):
            return False, f"profiling environment {s['profiling_env']}"
        if s["shared_decode_module_sha256"] != accepted_sha:
            return False, "shared_decode module SHA differs from the accepted implementation"
        return True, "ok"
    if set(s) != BF16_STAGE3C_KEYS:
        return False, f"unexpected BF16 record fields {sorted(set(s) ^ BF16_STAGE3C_KEYS)}"
    if s["applicable"] is not False:
        return False, "bfloat16 record claims a Stage3C selection"
    if not (isinstance(s["env"], dict) and set(s["env"]) == set(STAGE3C_ENV_KEYS)
            and all(v is None for v in s["env"].values())):
        return False, f"bfloat16 cell has a Stage3C / QUERY_BLOCK selector active: {s['env']}"
    if not off(s["profiling_env"]):
        return False, f"profiling environment {s['profiling_env']}"
    return True, "ok"


_frozen_parse_worker = r5.parse_worker


def parse_worker(lines: list[str]) -> dict:
    """Frozen per-cell parser, plus this cell's own strictly parsed EXP5_STAGE3C records."""
    out = _frozen_parse_worker(lines)
    try:
        out["stage3c_records"], out["stage3c_parse_error"] = parse_stage3c_records(lines), None
    except ValueError as e:
        out["stage3c_records"], out["stage3c_parse_error"] = None, str(e)
    return out


r5.parse_worker = parse_worker


def integrity(parsed: dict, gate: dict, top: dict, diff: dict) -> dict:
    integ = _frozen_integrity(parsed, gate, top, diff)
    checks = integ["checks"]

    def add(name, category, state, observed=None):
        if isinstance(state, bool):
            state = PASSED if state else FAILED
        checks.append({"check": name, "category": category, "state": state, "observed": observed})

    s3cfg = _STAGE3C or final_stage3c()
    accepted_sha = accepted_shared_decode_sha256()
    env = top.get("EXP5_ENVIRONMENT", {})
    add("attempt 2: per-cell watchdogs recorded in-container = 900 s everywhere except B32768 = 3600 s",
        "watchdog", bool(env) and env.get("leg_timeouts_s") == {l: leg_timeout(l) for _, l, *_ in r5.ALL_CELLS},
        env.get("leg_timeouts_s"))
    for k, label, d, c, p in r5.ALL_CELLS:
        started = label in top["leg_start"]
        t = parsed[k]["tags"]
        ls = top["leg_start"].get(label) or {}
        state = lambda ok: ok if started else "not_run"  # noqa: E731
        add(f"{label}: enforced watchdog = {leg_timeout(label)} s", "watchdog",
            state(ls.get("timeout_s") == leg_timeout(label)), ls.get("timeout_s"))
        recs, err = parsed[k].get("stage3c_records"), parsed[k].get("stage3c_parse_error")
        if "stage3c_records" not in parsed[k] and err is None:
            recs, err = None, "cell was not parsed by the attempt-2 Stage3C parser"
        ok, why = validate_stage3c(d, recs, err, s3cfg, accepted_sha)
        what = (f"exactly one EXP5_STAGE3C record: {s3cfg['impl']} requested == effective, QUERY_BLOCK "
                f"{s3cfg['query_block']}, selector env set, profiling off, accepted shared_decode SHA"
                if d == r5.B else "exactly one EXP5_STAGE3C record: not applicable, no selector, profiling off")
        add(f"{label} ({d}): {what}", "stage3c", state(ok), {"reason": why, "records": recs})
        if label in {l for _, l, *_ in r5.LEGS}:
            cap = (t.get("EXP5_CAPACITY") or {}).get("capacity_tokens")
            add(f"{label}: allocator capacity = expected {EXPECTED_CAPACITY[d]} tokens", "capacity",
                state(cap == EXPECTED_CAPACITY[d]), cap)
    b = next(k for k, label, *_ in r5.LEGS if label == B32768_LABEL)
    rows = parsed[b]["warmups"] + parsed[b]["samples"]
    add(f"{B32768_LABEL}: all {r5.WARMUPS_PER_LEG + r5.REPS_PER_LEG} output-token hashes == pre-established "
        f"{s3cfg['b32768_expected_output_sha256'][:12]}...", "b32768_output",
        (len(rows) == r5.WARMUPS_PER_LEG + r5.REPS_PER_LEG
         and all(r.get("output_token_ids_sha256") == s3cfg["b32768_expected_output_sha256"] for r in rows))
        if B32768_LABEL in top["leg_start"] else "not_run",
        sorted({r.get("output_token_ids_sha256") for r in rows}))
    counts = {s: sum(1 for ch in checks if ch["state"] == s) for s in r5.PASSED_STATES}
    integ.update(counts=counts, all_ok=counts[PASSED] == len(checks),
                 failed_categories=sorted({ch["category"] for ch in checks if ch["state"] == FAILED}),
                 non_passed_categories=sorted({ch["category"] for ch in checks if ch["state"] != PASSED}))
    return integ


r5.integrity = integrity


def attempt2_block() -> dict:
    s3 = _STAGE3C or final_stage3c()
    return {"attempt": 2, "failed_attempt_1": rel(FAILED_ATTEMPT_1) + " (historical diagnostic evidence; never pooled)",
            "final_rabit_configuration": {"kv_representation": "K3 / V2 / G32 / R4 / META8g64", **s3},
            "original_cell_watchdog_s": r5.LEG_TIMEOUT_S, "b32768_cell_watchdog_s": B32768_LEG_TIMEOUT_S,
            "gate_timeout_s": r5.GATE_TIMEOUT_S,
            "leg_timeouts_s": {l: leg_timeout(l) for _, l, *_ in r5.ALL_CELLS},
            "watchdog_change_reason": WATCHDOG_CHANGE_REASON, "per_request_guard": PER_REQUEST_GUARD_NOTE,
            "total_watchdog_budget_s": WATCHDOG_BUDGET_S, "modal_function_backstop_timeout_s": MODAL_FUNCTION_TIMEOUT_S,
            "infrastructure_only_changes": ["B32768 cell watchdog 900 s -> 3600 s",
                                            "Modal function backstop 14400 s -> 17100 s"],
            "unchanged": ["workload", "context grid", "cell order", "warmups (5)", "measured reps (15)", "model",
                          "prompts", "output length", "engine settings", "metrics", "failed-cell policy",
                          "gate timeout (600 s)", "all other cell watchdogs (900 s)"],
            "frozen_runner_sha256": sha256(FROZEN_RUNNER), "frozen_modal_app_sha256": sha256(FROZEN_MODAL_APP),
            "frozen_worker_sha256": sha256(FROZEN_WORKER)}


def build_summary(parsed: dict, gate: dict, top: dict) -> dict:
    s = _frozen_build_summary(parsed, gate, top)
    s["experiment"] = "MLSys 2027 Experiment 5 ATTEMPT 2 -- BF16 vs RABIT-KV (finalized Stage3C) context scaling"
    s["design"]["cell_timeout_s"] = {l: leg_timeout(l) for _, l, *_ in r5.ALL_CELLS}
    s["attempt_2"] = attempt2_block()
    return s


r5.build_summary = build_summary


def write_manifest(manifest: dict) -> None:
    manifest["experiment"] = "Experiment 5 ATTEMPT 2 -- BF16 vs RABIT-KV context-length scaling (finalized Stage3C)"
    manifest["attempt_2"] = attempt2_block()
    manifest.setdefault("protocol", {})["cell_timeout_s"] = {l: leg_timeout(l) for _, l, *_ in r5.ALL_CELLS}
    manifest["protocol"]["total_watchdog_budget_s"] = WATCHDOG_BUDGET_S
    manifest["protocol"]["modal_function_backstop_timeout_s"] = MODAL_FUNCTION_TIMEOUT_S
    _frozen_write_manifest(manifest)


r5.write_manifest = write_manifest


def build_command() -> list[str]:
    s3 = _STAGE3C or final_stage3c()
    return [sys.executable, "-m", "modal", "run", str(MODAL_APP), "--legs", r5.legs_arg(),
            "--warmups", str(r5.WARMUPS_PER_LEG), "--reps-per-leg", str(r5.REPS_PER_LEG),
            "--stage3c-impl", s3["impl"], "--query-block", str(s3["query_block"]),
            "--b32768-expected-output-sha", s3["b32768_expected_output_sha256"]]


def worker_command(label, dtype, context, prompt, model_dir="<modelscope snapshot dir>"):
    cmd = r5._frozen_worker_command(label, dtype, context, prompt, model_dir)
    cmd[1] = "/opt/exp5/exp5_attempt2_engine_worker.py"
    if dtype == r5.B:
        s3 = _STAGE3C or final_stage3c()
        cmd += ["--stage3c-impl", s3["impl"], "--query-block", str(s3["query_block"])]
    return cmd


r5._frozen_worker_command = r5.worker_command
r5.build_command = build_command
r5.worker_command = worker_command


# ------------------------------------------------------------------ offline reparse of the immutable raw run
# The H100 sweep (raw run 1) completed; only the first local analysis failed, because the attempt-2 EXP5_STAGE3C
# tag was not captured by the reused frozen parser. Raw bytes are pinned here and must never change.
RAW_RUN_DIR = ATTEMPT_DIR / "raw_run_1"
RAW_PINNED_SHA256 = {
    "modal_session.log": "efa66e861bbc7a912d5e8b1708f86f10e4614295348d0d38a52647ff69487f0d",
    "manifest.json": "cbd44f2081c1f76ae39c7316da1cb087df24d89b7493e1bd0e38129c1e4729ae",
    "integrity_check.json": "f3033e9aa039359f59ed1843a794a822c260d28faeb5512d25381cfa0acba145",
}
MEASUREMENT_CODE_COMMIT = "6ac5567bd44529f0c175d31ba67dbb6949077ef7"
MEASUREMENT_MODAL_APP = "ap-d0K1rXwIzyty6fvusW4ByL"
REPARSE_MANIFEST = ATTEMPT_DIR / "reparse_manifest.json"
ORIGINAL_FAILURE_REASON = ("Stage3C records were emitted correctly but were not captured by the local Attempt-2 "
                           "parser/check.")
PROVENANCE_WORDING = ("The H100 sweep completed successfully. Initial local post-processing marked the run failed "
                      "because the Attempt-2 Stage3C validation tag was not captured by the reused parser. The "
                      "immutable raw session was then re-parsed after a parser-only fix; no H100 measurement was "
                      "repeated.")


def _raw_digest() -> dict:
    return {f.name: r5.sha256_raw(f) for f in sorted(RAW_RUN_DIR.iterdir()) if f.is_file()}


def reparse(dry_run: bool) -> int:
    """Re-derive integrity / summary of raw run 1 from its preserved modal_session.log. Parsing only: no Modal,
    no vLLM, no CUDA, no workers; raw_run_1/ is only read."""
    for name, want in RAW_PINNED_SHA256.items():
        got = r5.sha256_raw(RAW_RUN_DIR / name) if (RAW_RUN_DIR / name).is_file() else None
        if got != want:
            raise SystemExit(f"Refusing to reparse: raw_run_1/{name} SHA-256 {got} != pinned {want}")
    raw_before = _raw_digest()
    orig = json.loads((RAW_RUN_DIR / "manifest.json").read_text(encoding="utf-8"))
    if not (orig["status"] == "failed" and orig["failure"]["stage"] == "integrity"
            and orig["failure"]["failed_categories"] == ["stage3c"] and orig["failure"]["failed_cells"] == []
            and orig["failure"]["modal_returncode"] == 0
            and orig["provenance"]["git_head"] == MEASUREMENT_CODE_COMMIT):
        raise SystemExit("Refusing to reparse: raw run 1 is not the expected Stage3C-check-only failure")
    session = (RAW_RUN_DIR / "modal_session.log").read_text(encoding="utf-8", errors="replace")
    if MEASUREMENT_MODAL_APP not in session:
        raise SystemExit("Refusing to reparse: Modal app id not found in the raw session")
    derived = sorted(p.name for p in ATTEMPT_DIR.iterdir() if p.is_file()) if ATTEMPT_DIR.is_dir() else []
    if not dry_run:
        if derived:
            raise SystemExit(f"Refusing to reparse: derived outputs already exist in {rel(ATTEMPT_DIR)}: {derived}")
        if run_git("status", "--short", "--", rel(RUNNER_SCRIPT), rel(OFFLINE_TESTS)):
            raise SystemExit("Refusing to reparse: the attempt-2 runner / tests have uncommitted changes")
    print(f"Reparse of {rel(RAW_RUN_DIR)}/modal_session.log (sha256 {RAW_PINNED_SHA256['modal_session.log']}); "
          f"measurement commit {MEASUREMENT_CODE_COMMIT}; Modal {MEASUREMENT_MODAL_APP}; parsing only.")
    if dry_run:
        print("--reparse --dry-run: raw run verified; nothing written.")
        return 0
    _STAGE3C.update(final_stage3c())
    diff, integ, summary = r5.analyze(session, write=True)
    prov = orig["provenance"]
    gates = {
        "failed_attempt_1_unchanged": archived_attempts_digest() == prov["archived_attempts_sha256"],
        "prior_evidence_unchanged": r5.prior_evidence_digest() == prov["prior_evidence_sha256_raw"],
        "protected_paths_clean": run_git("status", "--short", "--", *[rel(p) for p in r5.PROTECTED_PATHS]) == "",
    }
    b32768 = (integ["processes"]["exits"].get(B32768_LABEL) or {})
    rm = {
        "reparsed_from_existing_raw": True, "h100_rerun": False,
        "measurement_code_commit": MEASUREMENT_CODE_COMMIT, "analysis_parser_commit": run_git("rev-parse", "HEAD"),
        "measurement_modal_app": MEASUREMENT_MODAL_APP, "raw_run_dir": rel(RAW_RUN_DIR),
        "raw_modal_session_sha256": RAW_PINNED_SHA256["modal_session.log"], "raw_files_sha256_before": raw_before,
        "original_analysis_status": orig["status"], "original_failure": orig["failure"],
        "original_failure_reason": ORIGINAL_FAILURE_REASON, "provenance_wording": PROVENANCE_WORDING,
        "integrity_counts": integ["counts"], "all_integrity_passed": integ["all_ok"],
        "non_passed_categories": integ["non_passed_categories"], "post_gates": gates,
        "b32768": {"cell_process_elapsed_s": b32768.get("elapsed_s"), "cell_watchdog_s": B32768_LEG_TIMEOUT_S,
                   "fraction_of_watchdog": (b32768.get("elapsed_s") or 0) / B32768_LEG_TIMEOUT_S,
                   "original_900_s_watchdog_would_terminate": (b32768.get("elapsed_s") or 0) > r5.LEG_TIMEOUT_S},
        "attempt_2": attempt2_block(), "reparsed_utc": now(), "summary_written": summary is not None,
    }
    if summary is not None:
        summary["reparse_provenance"] = {k: rm[k] for k in (
            "reparsed_from_existing_raw", "h100_rerun", "measurement_code_commit", "analysis_parser_commit",
            "measurement_modal_app", "raw_modal_session_sha256", "original_analysis_status",
            "original_failure_reason", "provenance_wording")}
        r5.SUMMARY.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    rm["raw_files_sha256_after"] = _raw_digest()
    rm["raw_unchanged"] = rm["raw_files_sha256_after"] == raw_before
    rm["status"] = "completed" if (integ["all_ok"] and all(gates.values()) and rm["raw_unchanged"]) else "failed"
    REPARSE_MANIFEST.write_text(json.dumps(rm, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"REPARSE {rm['status'].upper()}: integrity {integ['counts']}; gates {gates}; raw unchanged "
          f"{rm['raw_unchanged']}")
    return 0 if rm["status"] == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Preflight + proofs + planned commands. No Modal/GPU.")
    ap.add_argument("--reparse", action="store_true",
                    help="offline re-analysis of the immutable raw_run_1 session (no Modal / vLLM / CUDA)")
    args = ap.parse_args(argv)
    if args.reparse:
        return reparse(args.dry_run)
    _STAGE3C.update(final_stage3c())
    print("RABIT-KV MLSys 2027 -- Experiment 5 ATTEMPT 2 (finalized Stage3C); failed_attempt_1 is never pooled")
    print(f"Final RABIT configuration: {json.dumps({k: _STAGE3C[k] for k in ('impl', 'query_block', 'query_block_source')})}")
    print(f"Conditioning cells (UNMEASURED): {[l for _, l, *_ in r5.CONDITIONING]}; official order: "
          f"{[l for _, l, *_ in r5.LEGS]}; {r5.WARMUPS_PER_LEG} warmups + {r5.REPS_PER_LEG} measured reps per cell")
    print(f"Watchdogs: gate {r5.GATE_TIMEOUT_S}s; cells {json.dumps({l: leg_timeout(l) for _, l, *_ in r5.ALL_CELLS})}; "
          f"budget {WATCHDOG_BUDGET_S}s; Modal backstop {MODAL_FUNCTION_TIMEOUT_S}s")
    print(f"Watchdog change reason: {WATCHDOG_CHANGE_REASON}")
    print(f"Per-request guard: {PER_REQUEST_GUARD_NOTE}")
    print(f"B32768 expected output hash (pre-established): {_STAGE3C['b32768_expected_output_sha256']}")
    leftovers = sorted(p.name for p in ATTEMPT_DIR.iterdir()) if ATTEMPT_DIR.is_dir() else []
    if leftovers and not args.dry_run:
        raise SystemExit(f"Refusing to run: {rel(ATTEMPT_DIR)} is not empty ({leftovers})")
    prov = r5.preflight(args.dry_run)
    print("Preflight OK.")
    for k in ("git_branch", "git_head", "vllm_kvquant_tree", "rabit_kv2_sha256", "runner_script_sha256",
              "modal_app_sha256", "worker_sha256", "correctness_gate_sha256", "watchdog_sha256"):
        print(f"  {k:<32} {prov[k]}")
    print(f"  failed_attempt_1 files hashed: {len(prov['archived_attempts_sha256'])}; prior evidence files hashed: "
          f"{len(prov['prior_evidence_sha256_raw'])}; protected paths: {len(prov['protected_paths'])}")
    if prov["uncommitted_experiment_files"]:
        print("  WARNING (dry-run only): uncommitted:\n    " + prov["uncommitted_experiment_files"].replace("\n", "\n    "))
    plan = {label: {**flatten("requested", r5.requested_kwargs(d)),
                    **flatten("workload", {"context_point": c, "prompt_tokens": p})} for _, label, d, c, p in r5.LEGS}
    keys = sorted(set().union(*plan.values()))
    differing = [x for x in keys if r5._uniq(pp.get(x) for pp in plan.values()) > 1]
    if not set(differing) <= set(r5.DTYPE_INDUCED_ALLOWLIST) | set(r5.CONTEXT_INDUCED_ALLOWLIST):
        raise SystemExit("Planned configs differ in a non-allowlisted field -- refusing.")
    print(f"  planned config: only allowlisted fields differ ({differing})")
    print("\nLocal command (one Modal run):\n  " + " ".join(r5.build_command()))
    for k, label, d, c, p in r5.ALL_CELLS:
        print(f"  cell {k:2d} {label:18s} watchdog {leg_timeout(label):4d}s: " + " ".join(r5.worker_command(label, d, c, p)))
    if args.dry_run:
        print("\n--dry-run: no Modal/GPU commands executed, no snapshot built, no files written.")
        return 0
    manifest = {
        "model": "LLM-Research/Meta-Llama-3.1-8B-Instruct", "gpu": "NVIDIA H100 80GB (Modal)",
        "plan_reference": "docs/MLSYS_EXPERIMENT_PLAN.md", "scope": r5.SCOPE_NOTE,
        "cells": [{"index": k, "cell": label, "role": r5.ROLE[label], "kv_cache_dtype": d, "context_point": c,
                   "actual_prompt_tokens": p, "output_tokens": r5.OUTPUT_TOKENS} for k, label, d, c, p in r5.ALL_CELLS],
        "protocol": {"warmups_per_cell_excluded": r5.WARMUPS_PER_LEG, "measured_reps_per_cell": r5.REPS_PER_LEG,
                     "context_grid": r5.CONTEXT_GRID, "output_tokens": r5.OUTPUT_TOKENS,
                     "max_model_len": r5.MAX_MODEL_LEN, "same_container_same_gpu": True,
                     "fresh_process_per_cell": True, "dtype_induced_allowlist": r5.DTYPE_INDUCED_ALLOWLIST,
                     "context_induced_allowlist": r5.CONTEXT_INDUCED_ALLOWLIST,
                     "gpu_clean_tolerance_mib": r5.GPU_CLEAN_TOLERANCE_MIB,
                     "gpu_clean_max_wait_s": r5.GPU_CLEAN_MAX_WAIT_S, "gate_timeout_s": r5.GATE_TIMEOUT_S,
                     "conditioning_cells": [label for _, label, *_ in r5.CONDITIONING],
                     "retries": 0, "failed_cell_policy": r5.FAILED_POLICY},
        "labels": {"capacity": r5.CAPACITY_LABEL, "live_kv": r5.LIVE_KV_LABEL, "gpu_memory": r5.GPU_MEMORY_LABEL,
                   "latency": r5.LATENCY_LABEL},
        "started_utc": now(), "completed_utc": None, "status": "running",
        "protected_paths_post_run_status": "pending", "provenance": prov, "runs": [],
    }
    r5.write_manifest(manifest)
    return r5.execute(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
