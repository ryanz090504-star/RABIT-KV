"""
Validator for results/mlsys2027/diagnostics/stage3c_tile32_benchmark/summary.json.

Independent of build_stage3c_tile32_benchmark_summary.py (no shared code): every
value is recomputed by re-parsing the raw benchmark files; the runner's own
benchmark_analysis.json and integrity_check.json are used only as independent
cross-checks. The only constants are the pre-registered grid, the frozen
first-chunk size and the pre-registered conclusion rules; no measured value is
hard-coded.

Read-only. Exits non-zero on any failure.

Usage:
    python benchmarks/mlsys2027/validate_stage3c_tile32_benchmark_summary.py \
        --bench-dir results/mlsys2027/diagnostics/stage3c_tile32_benchmark
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EPS = 1e-9
FIRST_CHUNK = 16384
CONDITIONING = 16416
Q_LENS = [2, 31, 32, 33, 512, 1024, 2048, 4096, 8192]
GRID = [FIRST_CHUNK + q for q in Q_LENS]
RAW = ["modal_session.log", "correctness_gate.log", "tile32_correctness_tests.log", "bf16_control_series.log",
       "rabit_reference_series.log", "rabit_tile32_series.log", "manifest.json", "integrity_check.json"]
LOGS = {"bf16_control": "bf16_control_series.log", "rabit_reference": "rabit_reference_series.log",
        "rabit_tile32": "rabit_tile32_series.log"}
PREFIX = {"bf16_control": "[series1:bf16_control] ", "rabit_reference": "[series2:rabit_reference] ",
          "rabit_tile32": "[series3:rabit_tile32] "}
EXPECT_IMPL = {"bf16_control": "reference", "rabit_reference": "reference", "rabit_tile32": "tile32"}
EXPECT_DTYPE = {"bf16_control": "bfloat16", "rabit_reference": "rabit_kv2", "rabit_tile32": "rabit_kv2"}
MODEST_BELOW, BOTTLENECK_Q, BOTTLENECK_RATIO = 1.25, 512, 2.0
CONCLUSION = ("tile32 provides only a modest Stage3C improvement over the tested range and does not "
              "materially eliminate the long-context chunked-prefill bottleneck.")
JIT = "Triton kernel JIT compilation during inference"


def h(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def payload(line: str, name: str):
    s = line.strip()
    return json.loads(s[len(name) + 1:]) if s.startswith(name + "={") else None


def read_series(text: str) -> dict:
    """Requests in order; JIT lines attributed to the open request, plus JIT after conditioning."""
    reqs, cur, eff, impl, kv, done, events, late, cond_seen = [], None, None, None, None, False, 0, 0, False
    for line in text.splitlines():
        s = line.strip()
        if s == "S3C_SERIES_COMPLETE":
            done = True
        elif (p := payload(s, "S3C_POINT_BEGIN")) is not None:
            cur = {"begin": p, "jit": 0}
            reqs.append(cur)
        elif (p := payload(s, "S3C_POINT")) is not None:
            cur["row"] = p
            cond_seen = cond_seen or p["role"] == "conditioning"
            cur = None
        elif (p := payload(s, "S3C_EFFECTIVE_ENGINE_CONFIG")) is not None:
            eff = p
        elif (p := payload(s, "S3C_STAGE_IMPL")) is not None:
            impl = p
        elif (p := payload(s, "S3C_KV_DTYPE")) is not None:
            kv = p
        elif s.startswith(("S3C_REQUEST_TIMEOUT=", "S3C_REQUEST_FAILURE=")):
            events += 1
        elif JIT in s:
            if cur is not None:
                cur["jit"] += 1
            elif cond_seen:
                late += 1
    return {"reqs": reqs, "eff": eff, "impl": impl, "kv": kv, "done": done, "events": events, "late_jit": late,
            "oom": sum(("CUDA out of memory" in x) or ("OutOfMemoryError" in x) for x in text.splitlines())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-dir", type=Path, required=True)
    d = ap.parse_args(argv).bench_dir
    fails, n = [], [0]

    def need(ok, msg):
        n[0] += 1
        if not ok:
            fails.append(msg)

    def eq(a, b):
        return isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) \
            and abs(a - b) <= EPS * max(1.0, abs(b))

    def section(fn, name):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 (malformed input is a failure, never a crash)
            fails.append(f"{name}: could not evaluate ({type(e).__name__}: {e})")

    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    integ = json.loads((d / "integrity_check.json").read_text(encoding="utf-8"))
    runner = json.loads((d / "benchmark_analysis.json").read_text(encoding="utf-8"))
    session = (d / "modal_session.log").read_text(encoding="utf-8")
    sl = session.splitlines()
    raw = {k: read_series((d / f).read_text(encoding="utf-8")) for k, f in LOGS.items()}
    meas = {k: {r["row"]["planned_prompt_tokens"]: r for r in v["reqs"] if r["begin"]["role"] == "measured"}
            for k, v in raw.items()}
    top = {}
    for line in sl:
        for name in ("S3C_GPU_BASELINE", "S3C_POST_RUN_GPU_STATE", "S3C_GATE_EXIT", "S3C_TILE_TESTS_EXIT",
                     "S3C_ENVIRONMENT"):
            if (p := payload(line, name)) is not None:
                top[name] = p
    procs = [p for line in sl if (p := payload(line, "S3C_PROCESS_EXIT")) is not None]
    pre = {p["leg"]: p for line in sl if (p := payload(line, "S3C_PRE_LEG_GPU_STATE")) is not None}

    def flags():
        need(s.get("diagnostic_evidence") is True, "diagnostic_evidence must be true")
        need(s.get("experiment5_final_evidence") is False, "experiment5_final_evidence must be false")
        need(s.get("optimization") == "stage3c_tile32_closed_page_batching", "optimization label")
        profiling = ("VLLM_RABIT2_STAGE3C_PROFILE" in session
                     or any("RABIT2_STAGE3C_TILE32_PROFILE" in (d / f).read_text(encoding="utf-8")
                            for f in LOGS.values()))
        need(profiling is False and s.get("component_profiling_enabled") is False,
             "component_profiling_enabled must be false (and no profiling trace in the raw logs)")
        need(s.get("remaining_cost_attribution_is_inference") is True, "remaining-cost attribution must be inference")
        rci = s["remaining_cost_inference"]
        need(rci.get("is_inference") is True and rci.get("directly_measured_component_time") is False
             and "NOT a measured component time" in rci.get("note", ""), "remaining-cost inference labeling")
        need("measured Stage3C component breakdown" in s.get("not_claimed", [])
             and "Experiment 5 results" in s.get("not_claimed", []), "not_claimed list")
        need(s["run_status"] == manifest["status"] == "completed" and manifest.get("modal_returncode") == 0
             == s.get("modal_returncode"), "run status / modal returncode")
        need(s["protected_paths_post_run_status"] == manifest["protected_paths_post_run_status"] == "clean",
             "protected paths")
        need(s["prior_evidence_unchanged"] is manifest["prior_evidence_unchanged"] is True, "prior evidence")
        need(s["integrity_counts"] == integ["counts"] and integ["all_ok"] is True and s["integrity_all_ok"] is True
             and integ["counts"]["failed"] == 0, "integrity counts")

    def hashes():
        need(sorted(s["raw_files_sha256"]) == sorted(RAW) and s["derived_from"] == RAW, "raw file set")
        for f in RAW:
            need(s["raw_files_sha256"].get(f) == h(d / f), f"{f}: SHA-256 differs from file")
        for k, f in LOGS.items():  # series logs are exactly the demuxed session lines
            demux = "\n".join(x[len(PREFIX[k]):] for x in sl if x.startswith(PREFIX[k])) + "\n"
            need(demux == (d / f).read_text(encoding="utf-8"), f"{f}: not identical to the session demux")
        for pfx, f in (("[gate] ", "correctness_gate.log"), ("[tile32-tests] ", "tile32_correctness_tests.log")):
            demux = "\n".join(x[len(pfx):] for x in sl if x.startswith(pfx)) + "\n"
            need(demux == (d / f).read_text(encoding="utf-8"), f"{f}: not identical to the session demux")
        prov = manifest["provenance"]
        need(all(s["provenance"][k] == prov[k] for k in s["provenance"]) and len(s["provenance"]) >= 9, "provenance")

    def gate():
        g = (d / "correctness_gate.log").read_text(encoding="utf-8")
        gl = g.splitlines()
        res = [p for x in gl if (p := payload(x, "EXP3_GATE_RESULT")) is not None]
        beg = [p for x in gl if (p := payload(x, "EXP3_GATE_BEGIN")) is not None]
        gp = [p for p in procs if p["label"] == "gate"]
        ok = (len(res) == 1 and res[0].get("passed") is True and "pytest exit=0" in g
              and "Dispatch preflight: PASSED" in g and "full attention exactness: PASSED" in g
              and "decode append exactness: PASSED" in g and "RABIT-2 FINAL TARGETED REGRESSION PASSED" in g
              and len(gp) == 1 and gp[0]["returncode"] == 0 and top.get("S3C_GATE_EXIT", {}).get("returncode") == 0
              and len(beg) == 1 and beg[0]["rabit_kv2_sha256_lf"] == top["S3C_ENVIRONMENT"]["rabit_kv2_sha256_lf"]
              == manifest["provenance"]["rabit_kv2_sha256"])
        need(ok and s["correctness_gate"]["frozen_gate_passed"] is True, "frozen gate did not pass")
        passed = [ln.split()[0] for ln in gl if " passed" in ln and " in " in ln and ln.split()[0].isdigit()]
        need(len(passed) == 1 and int(passed[0]) == s["correctness_gate"]["pytest_passed"] > 0, "gate pytest count")

    def tests():
        t = (d / "tile32_correctness_tests.log").read_text(encoding="utf-8")
        ids = [ln.split()[1] for ln in t.splitlines() if ln.startswith("PASSED ")]
        last = [ln for ln in t.splitlines() if " in " in ln and ln.rstrip().endswith("s") and "passed" in ln][-1]
        words = last.replace(",", "").split()
        count = {words[i + 1]: int(words[i]) for i in range(len(words) - 1) if words[i].isdigit()}
        tp = [p for p in procs if p["label"] == "tile32_tests"]
        st = s["tile32_correctness_tests"]
        need(len(ids) == len(set(ids)) == count.get("passed") == st["passed"] == st["passed_test_ids"],
             f"tile32 correctness count ({len(ids)} ids, summary {count.get('passed')}, summary.json {st['passed']})")
        for w in ("skipped", "failed", "error", "errors"):
            need(count.get(w, 0) == 0, f"tile32 tests: {w} present in raw summary")
        need(st["skipped"] == 0 and st["failed"] == 0 and st["errors"] == 0, "tile32 skipped/failed/error in summary")
        need(not any(ln.startswith(("FAILED ", "ERROR ", "SKIPPED ")) for ln in t.splitlines()), "failure lines")
        need(len(tp) == 1 and tp[0]["returncode"] == 0 and top.get("S3C_TILE_TESTS_EXIT", {}).get("returncode") == 0
             and st["all_passed"] is True, "tile32 tests exit / all_passed")
        need(integ_obs("tile32 correctness tests") == {"passed": count.get("passed"), "failed": 0, "errors": 0,
                                                        "skipped": 0, "equality_failures": []},
             "tile32 tests disagree with integrity_check.json")

    def integ_obs(prefix):
        return next(c["observed"] for c in integ["checks"] if c["check"].startswith(prefix))

    def requests():
        rc = s["request_checks"]
        for k, v in raw.items():
            need(v["done"] and v["eff"]["max_num_batched_tokens"] == FIRST_CHUNK, f"{k}: incomplete / first chunk")
            need(v["events"] == 0, f"{k}: request timeout/failure present")
            roles = [r["begin"]["role"] for r in v["reqs"]]
            need(roles == ["conditioning"] + ["measured"] * len(GRID), f"{k}: request roles")
            need(v["reqs"][0]["row"]["planned_prompt_tokens"] == CONDITIONING, f"{k}: conditioning prompt")
            need(list(meas[k]) == GRID, f"{k}: measured grid/order")
            need(v["impl"]["requested"] == v["impl"]["selector_reports"] == EXPECT_IMPL[k], f"{k}: Stage3C impl")
            need(v["kv"]["requested_kv_cache_dtype"] == v["kv"]["engine_cache_dtype"] == EXPECT_DTYPE[k],
                 f"{k}: KV dtype")
            need(v["late_jit"] == 0 and v["oom"] == 0, f"{k}: JIT after conditioning / OOM")
            for p, r in meas[k].items():
                row = r["row"]
                need(row["prompt_tokens"] == p and row["output_tokens"] == 32
                     and row["prompt_token_ids_sha256"] == row["planned_prompt_token_ids_sha256"],
                     f"{k}@{p}: prompt tokens / outputs / prompt hash")
                need(r["jit"] == 0, f"{k}@{p}: JIT during measured request")
        need(raw["rabit_reference"]["eff"] == raw["rabit_tile32"]["eff"]
             and rc["effective_config_reference_equals_tile32"] is True, "reference/tile32 config identity")
        need(rc["measured_requests"] == rc["measured_requests_succeeded"] == 3 * len(GRID), "27 measured requests")
        need(rc["jit_lines_in_measured_requests"] == 0 == sum(r["jit"] for k in meas for r in meas[k].values()),
             "JIT in measured requests")
        need(rc["jit_lines_outside_requests_after_conditioning"] == 0 and rc["oom_lines"] == 0, "late JIT / OOM")
        need(rc["request_timeouts_or_failures"] == [] and rc["watchdog_or_stop_events"] == []
             and not any(x.startswith(("S3C_WATCHDOG_TIMEOUT=", "S3C_STOPPED=")) for x in sl), "timeout/watchdog")
        need("S3C_BENCHMARK_COMPLETE" in (x.strip() for x in sl) and rc["benchmark_complete_marker"] is True,
             "benchmark complete marker")
        need(len(procs) == 5 and all(p["returncode"] == 0 and p["timed_out"] is False
                                     and not p["group_processes_remaining"] for p in procs)
             and rc["all_processes_exit_0_not_timed_out"] is True, "process exits / timeouts")
        need(all(len({meas[k][p]["row"]["prompt_token_ids_sha256"] for k in LOGS}) == 1 for p in GRID)
             and rc["prompt_hash_identical_across_series_all_points"] is True, "prompt hash identity across series")
        base = top["S3C_GPU_BASELINE"]
        tol = base["tolerance_mib"]
        for k in LOGS:
            last = pre[k]["readings"][-1]
            clean = (pre[k]["clean"] is True and not last["compute_apps"]
                     and all(u <= b + tol for u, b in zip(last["memory_used_mib"], base["memory_used_mib"])))
            need(clean and rc["gpu_clean_before_each_series"][k] is True, f"{k}: GPU not clean before series")
        post = top["S3C_POST_RUN_GPU_STATE"]
        need(not post["compute_apps"] and all(u <= b + tol for u, b in zip(post["memory_used_mib"],
                                                                            base["memory_used_mib"]))
             and rc["gpu_clean_after_run"] is True, "GPU not clean after run")

    rec = {}

    def points():
        need([g["prompt_tokens"] for g in s["grid"]] == GRID
             and [g["second_chunk_q_len"] for g in s["grid"]] == Q_LENS, "grid / q_len mapping")
        need([p["prompt_tokens"] for p in s["points"]] == GRID
             and [p["second_chunk_q_len"] for p in s["points"]] == Q_LENS, "points order / q_len mapping")
        for p in s["points"]:
            pt = p["prompt_tokens"]
            r, t, b = (meas[k][pt]["row"] for k in ("rabit_reference", "rabit_tile32", "bf16_control"))
            for key, val in (("reference_ttft_ms", r["ttft_ms"]), ("tile32_ttft_ms", t["ttft_ms"]),
                             ("ttft_saved_ms", r["ttft_ms"] - t["ttft_ms"]),
                             ("ttft_speedup", r["ttft_ms"] / t["ttft_ms"]),
                             ("reference_wall_ms", r["wall_ms"]), ("tile32_wall_ms", t["wall_ms"]),
                             ("wall_saved_ms", r["wall_ms"] - t["wall_ms"]),
                             ("wall_speedup", r["wall_ms"] / t["wall_ms"]),
                             ("reference_tpot_ms", r["tpot_ms"]), ("tile32_tpot_ms", t["tpot_ms"]),
                             ("tpot_delta_ms_tile32_minus_reference", t["tpot_ms"] - r["tpot_ms"]),
                             ("bf16_control_ttft_ms", b["ttft_ms"]),
                             ("tile32_ttft_over_bf16_control", t["ttft_ms"] / b["ttft_ms"]),
                             ("tile32_ttft_fraction_of_reference", t["ttft_ms"] / r["ttft_ms"])):
                need(eq(p.get(key), val), f"q_len {p['second_chunk_q_len']}: {key} does not recompute")
            need(p["reference_output_token_ids_sha256"] == r["output_token_ids_sha256"]
                 and p["tile32_output_token_ids_sha256"] == t["output_token_ids_sha256"],
                 f"q_len {p['second_chunk_q_len']}: output hash differs from raw")
            same = r["output_token_ids_sha256"] == t["output_token_ids_sha256"]
            need(same and p["output_hash_reference_equals_tile32"] is True,
                 f"q_len {p['second_chunk_q_len']}: reference/tile32 output hash mismatch")
            need(p["prompt_token_ids_sha256"] == r["prompt_token_ids_sha256"], f"{pt}: prompt hash")
            need(p["jit_lines_measured"] == {k: 0 for k in LOGS}, f"{pt}: JIT counts")
            rec[pt - FIRST_CHUNK] = {"sp": r["ttft_ms"] / t["ttft_ms"], "wsp": r["wall_ms"] / t["wall_ms"],
                                     "over_bf16": t["ttft_ms"] / b["ttft_ms"], "frac": t["ttft_ms"] / r["ttft_ms"]}
        need(s["all_output_hashes_reference_equal_tile32"] is True, "all-hash flag")

    def conclusions():
        sp = [rec[q]["sp"] for q in Q_LENS]
        rng = s["ttft_speedup_range"]
        need(eq(rng["min"], min(sp)) and eq(rng["max"], max(sp)) and rng["min_at_q_len"] == Q_LENS[sp.index(min(sp))]
             and rng["max_at_q_len"] == Q_LENS[sp.index(max(sp))], "TTFT speedup range")
        wsp = [rec[q]["wsp"] for q in Q_LENS]
        need(eq(s["wall_speedup_range"]["min"], min(wsp)) and eq(s["wall_speedup_range"]["max"], max(wsp)),
             "wall speedup range")
        large = [q for q in Q_LENS if q >= BOTTLENECK_Q]
        modest = max(sp) < MODEST_BELOW
        remains = all(rec[q]["over_bf16"] >= BOTTLENECK_RATIO for q in large)
        cr = s["conclusion_rules"]
        need(cr["modest_if_max_ttft_speedup_below"] == MODEST_BELOW and cr["bottleneck_q_len_min"] == BOTTLENECK_Q
             and cr["bottleneck_remains_if_tile32_ttft_over_bf16_at_least"] == BOTTLENECK_RATIO,
             "pre-registered thresholds changed")
        need(cr["modest"] is modest and cr["bottleneck_remains"] is remains
             and eq(cr["min_tile32_ttft_over_bf16_for_q_len_ge_min"], min(rec[q]["over_bf16"] for q in large)),
             "conclusion rule values")
        need((s["conclusion"] == CONCLUSION) is (modest and remains), "conclusion text does not follow the rules")
        fr = s["remaining_cost_inference"]["tile32_ttft_fraction_of_reference_q_len_ge_min"]
        need(eq(fr["min"], min(rec[q]["frac"] for q in large)) and eq(fr["max"], max(rec[q]["frac"] for q in large)),
             "remaining fraction (inference) range")

    def cross_check_runner():
        rp = {p["second_chunk_q_len"]: p for p in runner["points"]}
        need(runner["experiment5_evidence"] is False and runner["component_profiling_enabled"] is False
             and runner["all_integrity_passed"] is True, "runner analysis flags")
        for p in s["points"]:
            q, x = p["second_chunk_q_len"], rp[p["second_chunk_q_len"]]
            need(x["prompt_tokens"] == p["prompt_tokens"], f"q_len {q}: runner q_len mapping disagrees")
            need(eq(x["reference"]["ttft_ms"], p["reference_ttft_ms"]) and eq(x["tile32"]["ttft_ms"],
                                                                             p["tile32_ttft_ms"])
                 and eq(x["reference"]["wall_ms"], p["reference_wall_ms"])
                 and eq(x["tile32"]["wall_ms"], p["tile32_wall_ms"])
                 and eq(x["ttft_speedup_reference_over_tile32"], p["ttft_speedup"])
                 and eq(x["ttft_saved_ms"], p["ttft_saved_ms"]), f"q_len {q}: runner analysis disagrees")
            need(x["output_tokens_identical"] is True
                 and x["tile32"]["output_token_ids_sha256"] == p["tile32_output_token_ids_sha256"],
                 f"q_len {q}: runner hash disagrees")
        need(all(c["state"] == "passed" for c in integ["checks"]), "integrity check not all passed")

    for fn, nm in ((flags, "flags"), (hashes, "hashes"), (gate, "gate"), (tests, "tests"), (requests, "requests"),
                   (points, "points"), (conclusions, "conclusions"), (cross_check_runner, "cross-check")):
        section(fn, nm)
    print(f"bench: {d.as_posix()} | checks: {n[0]}")
    print(f"FAILURES: {len(fails)}")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
