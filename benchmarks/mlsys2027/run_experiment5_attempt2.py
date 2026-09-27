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


def integrity(parsed: dict, gate: dict, top: dict, diff: dict) -> dict:
    integ = _frozen_integrity(parsed, gate, top, diff)
    checks = integ["checks"]

    def add(name, category, state, observed=None):
        if isinstance(state, bool):
            state = PASSED if state else FAILED
        checks.append({"check": name, "category": category, "state": state, "observed": observed})

    s3cfg = _STAGE3C or final_stage3c()
    env = top.get("EXP5_ENVIRONMENT", {})
    add("attempt 2: per-cell watchdogs recorded in-container = 900 s everywhere except B32768 = 3600 s",
        "watchdog", bool(env) and env.get("leg_timeouts_s") == {l: leg_timeout(l) for _, l, *_ in r5.ALL_CELLS},
        env.get("leg_timeouts_s"))
    for k, label, d, c, p in r5.ALL_CELLS:
        started = label in top["leg_start"]
        t = parsed[k]["tags"]
        s3 = t.get("EXP5_STAGE3C") or {}
        ls = top["leg_start"].get(label) or {}
        state = lambda ok: ok if started else "not_run"  # noqa: E731
        add(f"{label}: enforced watchdog = {leg_timeout(label)} s", "watchdog",
            state(ls.get("timeout_s") == leg_timeout(label)), ls.get("timeout_s"))
        if d == r5.B:
            ok = (s3.get("applicable") is True and s3.get("requested_impl") == s3.get("effective_impl")
                  == s3cfg["impl"] and s3.get("requested_query_block") == s3.get("effective_query_block")
                  == s3cfg["query_block"] and not any(v not in (None, "0") for v in s3.get("profiling_env", {}).values()))
            add(f"{label}: Stage3C requested == effective == {s3cfg['impl']}, QUERY_BLOCK {s3cfg['query_block']}; "
                "profiling off", "stage3c", state(ok), s3 or None)
        else:
            ok = s3.get("applicable") is False and not any(s3.get("env", {}).values())
            add(f"{label}: bfloat16 cell carries no Stage3C selection", "stage3c", state(ok), s3 or None)
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


def main(argv: list[str] | None = None) -> int:
    make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Preflight + proofs + planned commands. No Modal/GPU.")
    args = ap.parse_args(argv)
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
