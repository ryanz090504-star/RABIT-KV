"""
Frozen summary of the Stage3C tile32 before/after BENCHMARK
(results/mlsys2027/diagnostics/stage3c_tile32_benchmark/). DIAGNOSTIC evidence
only -- never Experiment 5 final evidence, never component-attribution evidence.

Derived from the raw benchmark files (read-only): modal_session.log,
correctness_gate.log, tile32_correctness_tests.log, bf16_control_series.log,
rabit_reference_series.log, rabit_tile32_series.log, manifest.json,
integrity_check.json. benchmark_analysis.json is NOT a source (the validator
cross-checks it). Every number and every conclusion flag is computed here; the
only constants are the pre-registered grid and the pre-registered conclusion
thresholds below. No measured value is hard-coded.

Usage:
    python benchmarks/mlsys2027/build_stage3c_tile32_benchmark_summary.py \
        --bench-dir results/mlsys2027/diagnostics/stage3c_tile32_benchmark
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

RAW = ["modal_session.log", "correctness_gate.log", "tile32_correctness_tests.log", "bf16_control_series.log",
       "rabit_reference_series.log", "rabit_tile32_series.log", "manifest.json", "integrity_check.json"]
SERIES = {"bf16_control": "bf16_control_series.log", "rabit_reference": "rabit_reference_series.log",
          "rabit_tile32": "rabit_tile32_series.log"}
SERIES_IMPL = {"bf16_control": ("bfloat16", "reference"), "rabit_reference": ("rabit_kv2", "reference"),
               "rabit_tile32": ("rabit_kv2", "tile32")}
Q_LENS = [2, 31, 32, 33, 512, 1024, 2048, 4096, 8192]   # pre-registered grid (second-chunk q_len)
OPTIMIZATION = "stage3c_tile32_closed_page_batching"
# Pre-registered conclusion rules (fixed before looking at the numbers):
MODEST_MAX_TTFT_SPEEDUP = 1.25        # "modest": no point reaches a 1.25x TTFT speedup
MATERIAL_Q_LEN_MIN = 512              # "long-context bottleneck": q_len >= 512 (clear steady Stage3C regime)
BOTTLENECK_REMAINS_TILE32_OVER_BF16 = 2.0  # bottleneck remains if tile32 TTFT >= 2x the BF16 control there
JIT = "Triton kernel JIT compilation during inference"
PROFILE_LINE = "RABIT2_STAGE3C_TILE32_PROFILE"
PROFILE_ENV = "VLLM_RABIT2_STAGE3C_PROFILE"
CONCLUSION_MODEST = ("tile32 provides only a modest Stage3C improvement over the tested range and does not "
                     "materially eliminate the long-context chunked-prefill bottleneck.")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tag(line: str):
    m = re.match(r"^(S3C_[A-Z0-9_]+|EXP3_[A-Z_]+)=(\{.*\})$", line.strip())
    return (m.group(1), json.loads(m.group(2))) if m else None


def parse_series(text: str) -> dict:
    tags, reqs, cur, complete = {}, [], None, False
    cond_done, late_jit, oom, profile_lines, events = False, 0, 0, 0, []
    for line in text.splitlines():
        s = line.strip()
        if s == "S3C_SERIES_COMPLETE":
            complete = True
            continue
        oom += ("CUDA out of memory" in s) or ("OutOfMemoryError" in s)
        profile_lines += PROFILE_LINE in s
        t = tag(s)
        if t:
            name, p = t
            if name == "S3C_POINT_BEGIN":
                cur = {"begin": p, "jit": 0}
                reqs.append(cur)
            elif name == "S3C_POINT":
                cur["row"] = p
                cond_done = cond_done or p.get("role") == "conditioning"
                cur = None
            elif name in ("S3C_REQUEST_TIMEOUT", "S3C_REQUEST_FAILURE"):
                events.append({name: p})
            elif name == "S3C_GPU_MEMORY":
                tags.setdefault(name, []).append(p)
            else:
                tags[name] = p
            continue
        if cur is not None:
            cur["jit"] += JIT in s
        elif cond_done and JIT in s:
            late_jit += 1
    return {"tags": tags, "requests": reqs, "complete": complete, "jit_outside_requests_after_conditioning": late_jit,
            "oom_lines": oom, "profile_lines": profile_lines, "request_events": events,
            "profile_env_seen": PROFILE_ENV in text}


def parse_session(text: str) -> dict:
    out = {"proc": {}, "pre": {}, "exit": {}, "watchdog": [], "complete": False, "tags": {}}
    for line in text.splitlines():
        s = line.strip()
        if s == "S3C_BENCHMARK_COMPLETE":
            out["complete"] = True
        t = tag(s)
        if not t:
            continue
        name, p = t
        if name == "S3C_PROCESS_EXIT":
            out["proc"][p["label"]] = p
        elif name == "S3C_PRE_LEG_GPU_STATE":
            out["pre"][p["leg"]] = p
        elif name == "S3C_SERIES_EXIT":
            out["exit"][p["series"]] = p
        elif name in ("S3C_WATCHDOG_TIMEOUT", "S3C_STOPPED"):
            out["watchdog"].append({name: p})
        else:
            out["tags"][name] = p
    return out


def parse_tests(text: str) -> dict:
    passed_ids = re.findall(r"^PASSED (\S+)", text, re.M)
    summaries = re.findall(r"^(?:=+ )?(\d+ \w+.* in [\d.]+s)(?: =+)?$", text, re.M)
    summary = summaries[-1] if summaries else ""
    words = {w: int(n) for n, w in re.findall(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed)", summary)}
    return {"passed_test_ids": len(passed_ids), "unique_passed_test_ids": len(set(passed_ids)),
            "summary_line": summary or None, "summary_lines": len(summaries),
            "passed": words.get("passed", 0), "failed": words.get("failed", 0),
            "errors": words.get("error", 0) + words.get("errors", 0), "skipped": words.get("skipped", 0),
            "failed_or_error_lines": len(re.findall(r"^(FAILED|ERROR) ", text, re.M))}


def gpu_clean(pre: dict, baseline: dict) -> bool:
    if not pre or not pre.get("readings") or not baseline:
        return False
    last = pre["readings"][-1]
    tol = pre.get("tolerance_mib")
    return (pre.get("clean") is True and not last["compute_apps"] and tol == baseline.get("tolerance_mib")
            and all(u <= b + tol for u, b in zip(last["memory_used_mib"], baseline["memory_used_mib"])))


def build(bench: Path) -> dict:
    manifest = json.loads((bench / "manifest.json").read_text(encoding="utf-8"))
    integ = json.loads((bench / "integrity_check.json").read_text(encoding="utf-8"))
    gate = (bench / "correctness_gate.log").read_text(encoding="utf-8")
    tests = parse_tests((bench / "tile32_correctness_tests.log").read_text(encoding="utf-8"))
    session_text = (bench / "modal_session.log").read_text(encoding="utf-8")
    sess = parse_session(session_text)
    ser = {k: parse_series((bench / f).read_text(encoding="utf-8")) for k, f in SERIES.items()}

    def measured(k):
        return [r for r in ser[k]["requests"] if r["begin"].get("role") == "measured"]

    fc = {k: s["tags"]["S3C_EFFECTIVE_ENGINE_CONFIG"]["max_num_batched_tokens"] for k, s in ser.items()}
    first_chunk = fc["rabit_reference"]
    by = {k: {r["row"]["planned_prompt_tokens"]: r for r in measured(k)} for k in SERIES}
    prompts = [r["row"]["planned_prompt_tokens"] for r in measured("rabit_reference")]

    # --- gate (frozen correctness) ---
    gate_tags = [t for t in map(tag, gate.splitlines()) if t]
    g_begin = next((p for n, p in gate_tags if n == "EXP3_GATE_BEGIN"), None)
    g_res = [p for n, p in gate_tags if n == "EXP3_GATE_RESULT"]
    g_py = re.search(r"^(\d+) passed, (\d+) warnings? in", gate, re.M)
    g_prep = re.search(r"prep exactness: PASSED \((\d+)/(\d+)\)", gate)
    gate_block = {
        "pytest_passed": int(g_py.group(1)) if g_py else None,
        "pytest_exit_0": "pytest exit=0" in gate,
        "dispatch_preflight_passed": "Dispatch preflight: PASSED" in gate,
        "prep_exactness": f"{g_prep.group(1)}/{g_prep.group(2)}" if g_prep else None,
        "full_attention_exactness_passed": "full attention exactness: PASSED" in gate,
        "decode_append_exactness_passed": "decode append exactness: PASSED" in gate,
        "regression_passed_line": "RABIT-2 FINAL TARGETED REGRESSION PASSED" in gate,
        "gate_result_passed": len(g_res) == 1 and g_res[0].get("passed") is True,
        "gate_process_returncode": sess["proc"].get("gate", {}).get("returncode"),
        "gate_exit_returncode": sess["tags"].get("S3C_GATE_EXIT", {}).get("returncode"),
        "frozen_rabit_kv2_sha256_gate": (g_begin or {}).get("rabit_kv2_sha256_lf"),
        "frozen_rabit_kv2_sha256_environment": sess["tags"].get("S3C_ENVIRONMENT", {}).get("rabit_kv2_sha256_lf"),
    }
    gate_block["frozen_gate_passed"] = bool(
        gate_block["pytest_exit_0"] and gate_block["dispatch_preflight_passed"]
        and gate_block["full_attention_exactness_passed"] and gate_block["decode_append_exactness_passed"]
        and gate_block["regression_passed_line"] and gate_block["gate_result_passed"]
        and gate_block["gate_process_returncode"] == 0 and gate_block["gate_exit_returncode"] == 0
        and gate_block["frozen_rabit_kv2_sha256_gate"] == gate_block["frozen_rabit_kv2_sha256_environment"]
        == manifest["provenance"]["rabit_kv2_sha256"])

    # --- tile32 correctness tests ---
    tests["process_returncode"] = sess["proc"].get("tile32_tests", {}).get("returncode")
    tests["exit_returncode"] = sess["tags"].get("S3C_TILE_TESTS_EXIT", {}).get("returncode")
    tests["all_passed"] = bool(tests["passed"] > 0 and tests["passed"] == tests["passed_test_ids"]
                               == tests["unique_passed_test_ids"] and tests["failed"] == 0 and tests["errors"] == 0
                               and tests["skipped"] == 0 and tests["failed_or_error_lines"] == 0
                               and tests["process_returncode"] == 0 and tests["exit_returncode"] == 0)

    # --- request / process / GPU checks ---
    all_meas = [r for k in SERIES for r in measured(k)]
    procs = {k: {x: v.get(x) for x in ("returncode", "timed_out", "group_processes_remaining", "elapsed_s")}
             for k, v in sess["proc"].items()}
    baseline = sess["tags"].get("S3C_GPU_BASELINE", {})
    post = sess["tags"].get("S3C_POST_RUN_GPU_STATE", {})
    tol = baseline.get("tolerance_mib")
    impl_tags = {k: ser[k]["tags"].get("S3C_STAGE_IMPL", {}) for k in SERIES}
    eff = {k: ser[k]["tags"].get("S3C_EFFECTIVE_ENGINE_CONFIG") for k in SERIES}
    checks = {
        "series_complete": {k: ser[k]["complete"] for k in SERIES},
        "request_roles": {k: [r["begin"].get("role") for r in ser[k]["requests"]] for k in SERIES},
        "measured_requests": len(all_meas),
        "measured_requests_succeeded": sum(
            1 for r in all_meas if "row" in r and r["row"]["output_tokens"] == 32
            and r["row"]["prompt_tokens"] == r["row"]["planned_prompt_tokens"]
            and r["row"]["prompt_token_ids_sha256"] == r["row"]["planned_prompt_token_ids_sha256"]),
        "request_timeouts_or_failures": [e for k in SERIES for e in ser[k]["request_events"]],
        "watchdog_or_stop_events": sess["watchdog"],
        "benchmark_complete_marker": sess["complete"],
        "processes": procs,
        "all_processes_exit_0_not_timed_out": bool(procs) and all(
            v["returncode"] == 0 and v["timed_out"] is False and not v["group_processes_remaining"]
            for v in procs.values()),
        "series_exit_returncodes": {k: sess["exit"].get(k, {}).get("returncode") for k in SERIES},
        "jit_lines_in_measured_requests": sum(r["jit"] for r in all_meas),
        "jit_lines_outside_requests_after_conditioning": sum(
            ser[k]["jit_outside_requests_after_conditioning"] for k in SERIES),
        "oom_lines": sum(ser[k]["oom_lines"] for k in SERIES),
        "stage3c_impl": impl_tags,
        "stage3c_impl_as_requested": all(
            impl_tags[k].get("requested") == SERIES_IMPL[k][1] == impl_tags[k].get("selector_reports")
            for k in SERIES),
        "kv_dtype_as_requested": all(
            ser[k]["tags"].get("S3C_KV_DTYPE", {}).get("requested_kv_cache_dtype") == SERIES_IMPL[k][0]
            and ser[k]["tags"].get("S3C_KV_DTYPE", {}).get("engine_cache_dtype") == SERIES_IMPL[k][0]
            for k in SERIES),
        "effective_config_reference_equals_tile32": eff["rabit_reference"] == eff["rabit_tile32"],
        "first_chunk_tokens_from_engine_config": fc,
        "measured_grid_in_order": {k: [r["row"]["planned_prompt_tokens"] for r in measured(k)] for k in SERIES},
        "prompt_hash_identical_across_series_all_points": all(
            len({by[k][p]["row"]["prompt_token_ids_sha256"] for k in SERIES}) == 1 for p in prompts),
        "conditioning_prompt_tokens": {k: [r["row"]["planned_prompt_tokens"] for r in ser[k]["requests"]
                                           if r["begin"].get("role") == "conditioning"] for k in SERIES},
        "gpu_clean_before_each_series": {k: gpu_clean(sess["pre"].get(k), baseline) for k in SERIES},
        "gpu_clean_after_run": bool(post) and not post.get("compute_apps") and tol is not None and all(
            u <= b + tol for u, b in zip(post.get("memory_used_mib", []), baseline.get("memory_used_mib", []))),
    }

    # --- component profiling was OFF (derived: no profile env var seen, no profile lines) ---
    profiling_seen = (PROFILE_ENV in session_text or any(ser[k]["profile_env_seen"] or ser[k]["profile_lines"]
                                                         for k in SERIES))

    # --- per-point performance ---
    rows = []
    for p in prompts:
        r, t, b = (by[k][p]["row"] for k in ("rabit_reference", "rabit_tile32", "bf16_control"))
        rows.append({
            "prompt_tokens": p,
            "second_chunk_q_len": p - first_chunk,
            "reference_ttft_ms": r["ttft_ms"], "tile32_ttft_ms": t["ttft_ms"],
            "ttft_saved_ms": r["ttft_ms"] - t["ttft_ms"], "ttft_speedup": r["ttft_ms"] / t["ttft_ms"],
            "reference_wall_ms": r["wall_ms"], "tile32_wall_ms": t["wall_ms"],
            "wall_saved_ms": r["wall_ms"] - t["wall_ms"], "wall_speedup": r["wall_ms"] / t["wall_ms"],
            "reference_tpot_ms": r["tpot_ms"], "tile32_tpot_ms": t["tpot_ms"],
            "tpot_delta_ms_tile32_minus_reference": t["tpot_ms"] - r["tpot_ms"],
            "reference_output_token_ids_sha256": r["output_token_ids_sha256"],
            "tile32_output_token_ids_sha256": t["output_token_ids_sha256"],
            "output_hash_reference_equals_tile32": r["output_token_ids_sha256"] == t["output_token_ids_sha256"],
            "bf16_control_ttft_ms": b["ttft_ms"],
            "tile32_ttft_over_bf16_control": t["ttft_ms"] / b["ttft_ms"],
            "tile32_ttft_fraction_of_reference": t["ttft_ms"] / r["ttft_ms"],
            "prompt_token_ids_sha256": r["prompt_token_ids_sha256"],
            "jit_lines_measured": {k: by[k][p]["jit"] for k in SERIES},
        })

    speed = [x["ttft_speedup"] for x in rows]
    wall_speed = [x["wall_speedup"] for x in rows]
    large = [x for x in rows if x["second_chunk_q_len"] >= MATERIAL_Q_LEN_MIN]
    modest = max(speed) < MODEST_MAX_TTFT_SPEEDUP
    remains = bool(large) and all(x["tile32_ttft_over_bf16_control"] >= BOTTLENECK_REMAINS_TILE32_OVER_BF16
                                  for x in large)
    conclusion = CONCLUSION_MODEST if (modest and remains) else (
        "pre-registered 'modest / bottleneck remains' rule NOT satisfied; see conclusion_rules")
    frac = [x["tile32_ttft_fraction_of_reference"] for x in large]

    return {
        "schema": "stage3c_tile32_benchmark_summary/v1",
        "diagnostic_evidence": True,
        "experiment5_final_evidence": False,
        "optimization": OPTIMIZATION,
        "component_profiling_enabled": profiling_seen,
        "remaining_cost_attribution_is_inference": True,
        "generated_by": "benchmarks/mlsys2027/build_stage3c_tile32_benchmark_summary.py",
        "derived_from": RAW,
        "raw_files_sha256": {f: sha(bench / f) for f in RAW},
        "run_status": manifest["status"],
        "modal_returncode": manifest.get("modal_returncode"),
        "protected_paths_post_run_status": manifest["protected_paths_post_run_status"],
        "prior_evidence_unchanged": manifest["prior_evidence_unchanged"],
        "integrity_counts": integ["counts"],
        "integrity_all_ok": integ["all_ok"],
        "correctness_gate": gate_block,
        "tile32_correctness_tests": tests,
        "request_checks": checks,
        "grid": [{"prompt_tokens": x["prompt_tokens"], "second_chunk_q_len": x["second_chunk_q_len"]} for x in rows],
        "points": rows,
        "all_output_hashes_reference_equal_tile32": all(x["output_hash_reference_equals_tile32"] for x in rows),
        "ttft_speedup_range": {"min": min(speed), "min_at_q_len": rows[speed.index(min(speed))]["second_chunk_q_len"],
                               "max": max(speed), "max_at_q_len": rows[speed.index(max(speed))]["second_chunk_q_len"]},
        "wall_speedup_range": {"min": min(wall_speed), "max": max(wall_speed)},
        "conclusion_rules": {
            "modest_if_max_ttft_speedup_below": MODEST_MAX_TTFT_SPEEDUP,
            "modest": modest,
            "bottleneck_q_len_min": MATERIAL_Q_LEN_MIN,
            "bottleneck_remains_if_tile32_ttft_over_bf16_at_least": BOTTLENECK_REMAINS_TILE32_OVER_BF16,
            "min_tile32_ttft_over_bf16_for_q_len_ge_min": min(x["tile32_ttft_over_bf16_control"] for x in large),
            "bottleneck_remains": remains,
        },
        "conclusion": conclusion,
        "remaining_cost_inference": {
            "is_inference": True,
            "directly_measured_component_time": False,
            "tile32_ttft_fraction_of_reference_q_len_ge_min": {"min": min(frac), "max": max(frac)},
            "note": ("tile32 batches only the closed-page scan; the TTFT that remains after tile32 is attributed to "
                     "the other per-query Stage3C work (tail preparation, tail partial, reduce, per-query "
                     "Python/state exposure) by INFERENCE from end-to-end TTFT only. No component timing was "
                     "recorded in this run; the remaining fraction is NOT a measured component time."),
        },
        "not_claimed": ["Experiment 5 results", "a complexity law", "measured Stage3C component breakdown",
                        "tile32 correctness beyond the executed tests", "the 32K (q_len 16352) point"],
        "provenance": {k: manifest["provenance"][k] for k in (
            "git_head", "vllm_kvquant_tree", "rabit_kv2_sha256", "tile32_module_sha256", "tile32_tests_sha256",
            "triton_attn_sha256", "runner_script_sha256", "modal_app_sha256", "worker_sha256")},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    out = a.bench_dir / "summary.json"
    out.write_text(json.dumps(build(a.bench_dir), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
