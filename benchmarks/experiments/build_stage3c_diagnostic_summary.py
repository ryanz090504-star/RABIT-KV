"""
Frozen summary of the pre-fix Stage3C chunked-prefill DIAGNOSTIC
(results/mlsys2027/diagnostics/stage3c_cliff/). DIAGNOSTIC evidence only --
never Experiment 5 final evidence.

Derived from the raw diagnostic files (read-only): modal_session.log,
correctness_gate.log, bf16_series.log, rabit_kv2_series.log, manifest.json,
integrity_check.json. diagnostic_analysis.json is NOT a source (the validator
may cross-check it). Every number and every conclusion flag is computed here;
nothing measured is hard-coded.

Usage:
    python benchmarks/mlsys2027/build_stage3c_diagnostic_summary.py \
        --diag-dir results/mlsys2027/diagnostics/stage3c_cliff
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

RAW = ["modal_session.log", "correctness_gate.log", "bf16_series.log", "rabit_kv2_series.log", "manifest.json",
       "integrity_check.json"]
SERIES = {"bfloat16": "bf16_series.log", "rabit_kv2": "rabit_kv2_series.log"}
FIRST_CHUNK = 16384          # max_num_batched_tokens (frozen engine setting, verified against the series log)
NOISE_REF_POINTS = (16384, 16385, 16386, 16415, 16416, 16417)  # BF16 points whose spread bounds single-sample noise
STAGE3C_MARK = re.compile(r"RABIT2_STAGE3C_CHUNKED_PREFILL_ACTIVE q_len=(\d+) context_len=(\d+)")
JIT = "Triton kernel JIT compilation during inference"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_series(text: str) -> dict:
    tags, reqs, cur, complete = {}, [], None, False
    after_conditioning, jit_unattributed = False, 0
    for line in text.splitlines():
        s = line.strip()
        if s == "S3C_SERIES_COMPLETE":
            complete = True
            continue
        m = re.match(r"^(S3C_[A-Z_]+)=(\{.*\})$", s)
        if m:
            tag, p = m.group(1), json.loads(m.group(2))
            if tag == "S3C_POINT_BEGIN":
                cur = {"begin": p, "jit": 0, "markers": [], "oom": 0}
                reqs.append(cur)
            elif tag == "S3C_POINT":
                cur["row"] = p
                after_conditioning = after_conditioning or p.get("role") == "conditioning"
                cur = None
            elif tag in ("S3C_REQUEST_TIMEOUT", "S3C_REQUEST_FAILURE"):
                tags.setdefault(tag, []).append(p)
            elif tag != "S3C_GPU_MEMORY":
                tags[tag] = p
            continue
        if cur is None and after_conditioning and not complete and JIT in s:
            jit_unattributed += 1  # e.g. a late EngineCore line between measured requests
        if cur is not None:
            cur["jit"] += JIT in s
            cur["oom"] += ("CUDA out of memory" in s) or ("OutOfMemoryError" in s)
            mk = STAGE3C_MARK.search(s)
            if mk:
                cur["markers"].append((int(mk.group(1)), int(mk.group(2))))
    return {"tags": tags, "requests": reqs, "complete": complete, "jit_between_measured_requests": jit_unattributed}


def build(diag: Path) -> dict:
    manifest = json.loads((diag / "manifest.json").read_text(encoding="utf-8"))
    integ = json.loads((diag / "integrity_check.json").read_text(encoding="utf-8"))
    gate = (diag / "correctness_gate.log").read_text(encoding="utf-8")
    session = (diag / "modal_session.log").read_text(encoding="utf-8")
    ser = {d: parse_series((diag / f).read_text(encoding="utf-8")) for d, f in SERIES.items()}

    procs = {}
    timeouts = []
    for line in session.splitlines():
        m = re.match(r"^(S3C_PROCESS_EXIT|S3C_WATCHDOG_TIMEOUT|S3C_STOPPED)=(\{.*\})$", line.strip())
        if m:
            p = json.loads(m.group(2))
            if m.group(1) == "S3C_PROCESS_EXIT":
                procs[p["label"]] = p
            else:
                timeouts.append({m.group(1): p})

    eff = {d: s["tags"]["S3C_EFFECTIVE_ENGINE_CONFIG"] for d, s in ser.items()}
    first_chunk = {d: e["max_num_batched_tokens"] for d, e in eff.items()}

    def measured(d):
        return [r for r in ser[d]["requests"] if r["begin"]["role"] == "measured"]

    def conditioning(d):
        return [r for r in ser[d]["requests"] if r["begin"]["role"] == "conditioning"]

    points = [r["row"]["planned_prompt_tokens"] for r in measured("rabit_kv2")]
    by = {d: {r["row"]["planned_prompt_tokens"]: r for r in measured(d)} for d in SERIES}
    fc = first_chunk["rabit_kv2"]

    def q_len(p):
        return p - fc if p > fc else None

    rows = []
    b0 = by["bfloat16"][fc]["row"]["ttft_ms"]
    r0 = by["rabit_kv2"][fc]["row"]["ttft_ms"]
    for p in points:
        b, r = by["bfloat16"][p]["row"], by["rabit_kv2"][p]["row"]
        q = q_len(p)
        exc = (r["ttft_ms"] - r0) - (b["ttft_ms"] - b0)
        rows.append({
            "prompt_tokens": p,
            "second_chunk_q_len": q,
            "context_len_derived": fc if q else None,
            "stage3c_active_by_source": bool(q is not None and q > 1),
            "path_by_source": ("single_chunk_dense_prefill" if q is None else
                               "decode_append_path" if q == 1 else "stage3c_per_token_loop"),
            "bf16_ttft_ms": b["ttft_ms"], "rabit_ttft_ms": r["ttft_ms"],
            "bf16_wall_ms": b["wall_ms"], "rabit_wall_ms": r["wall_ms"],
            "bf16_tpot_ms": b["tpot_ms"], "rabit_tpot_ms": r["tpot_ms"],
            "raw_rabit_minus_bf16_ttft_ms": r["ttft_ms"] - b["ttft_ms"],
            "bf16_increment_over_bf16_16384_ms": b["ttft_ms"] - b0,
            "rabit_increment_over_rabit_16384_ms": r["ttft_ms"] - r0,
            "baseline_adjusted_excess_ms": exc,
            "excess_per_second_chunk_token_ms": exc / q if q and q > 1 else None,
            "prompt_token_ids_sha256": r["prompt_token_ids_sha256"],
            "prompt_hash_equal_across_dtypes": r["prompt_token_ids_sha256"] == b["prompt_token_ids_sha256"],
            "jit_lines_measured": {"bfloat16": by["bfloat16"][p]["jit"], "rabit_kv2": by["rabit_kv2"][p]["jit"]},
        })
    R = {r["prompt_tokens"]: r for r in rows}
    noise = [R[p]["bf16_ttft_ms"] for p in NOISE_REF_POINTS]
    noise_span = max(noise) - min(noise)
    large = [r for r in rows if (r["second_chunk_q_len"] or 0) >= 512]
    per_tok = [r["excess_per_second_chunk_token_ms"] for r in large]
    spread_ratio = max(per_tok) / min(per_tok)
    ex31, ex32, ex33 = (R[fc + q]["baseline_adjusted_excess_ms"] for q in (31, 32, 33))
    step_31_32, step_32_33 = ex32 - ex31, ex33 - ex32
    all_reqs = [r for d in SERIES for r in ser[d]["requests"]]
    meas_reqs = [r for d in SERIES for r in measured(d)]
    cond_rabit = conditioning("rabit_kv2")
    cond_markers = cond_rabit[0]["markers"] if cond_rabit else []

    conclusions = {
        "q_len_1_is_decode_path_by_source": R[fc + 1]["path_by_source"] == "decode_append_path",
        "q_len_1_excess_within_noise": abs(R[fc + 1]["baseline_adjusted_excess_ms"]) <= noise_span,
        "stage3c_begins_at_q_len_gt_1_by_source": (not R[fc + 1]["stage3c_active_by_source"]
                                                   and R[fc + 2]["stage3c_active_by_source"]),
        "q_len_2_per_token_excess_in_large_q_range": min(per_tok) * 0.8 <= R[fc + 2]["excess_per_second_chunk_token_ms"]
        <= max(per_tok) * 1.2,
        "no_visible_discontinuity_at_32": (abs(step_31_32) <= noise_span and abs(step_32_33) <= noise_span
                                           and abs(step_31_32 - step_32_33) <= noise_span),
        "excess_per_token_approximately_stable_large_q": spread_ratio <= 1.25,
        "formal_complexity_law_claimed": False,
        "measured_point_layer_markers_available": False,
        "failed_attempt_1_layer_timing": "external frozen reference only (never pooled)",
    }
    fa = diag.parent.parent / "context_scaling" / "failed_attempt_1" / "failure_analysis.json"
    fref = json.loads(fa.read_text(encoding="utf-8"))["b32768"] if fa.is_file() else None
    g_py = re.search(r"^(\d+) passed, (\d+) warnings?", gate, re.M)
    g_prep = re.search(r"prep exactness: PASSED \((\d+)/(\d+)\)", gate)
    g_dec = re.search(r"decode append exactness: PASSED \((\d+) state", gate)
    return {
        "schema": "stage3c_diagnostic_summary/v1",
        "diagnostic_evidence": True,
        "experiment5_final_evidence": False,
        "generated_by": "benchmarks/mlsys2027/build_stage3c_diagnostic_summary.py",
        "derived_from": RAW,
        "raw_files_sha256": {f: sha(diag / f) for f in RAW},
        "run_status": manifest["status"],
        "protected_paths_post_run_status": manifest["protected_paths_post_run_status"],
        "prior_evidence_unchanged": manifest["prior_evidence_unchanged"],
        "integrity_counts": integ["counts"],
        "correctness_gate": {"pytest_passed": int(g_py.group(1)), "pytest_warnings": int(g_py.group(2)),
                             "dispatch_preflight_passed": "Dispatch preflight: PASSED" in gate,
                             "prep_exactness": f"{g_prep.group(1)}/{g_prep.group(2)}",
                             "full_attention_exactness_passed": "full attention exactness: PASSED" in gate,
                             "decode_append_steps": int(g_dec.group(1)),
                             "regression_passed": "RABIT-2 FINAL TARGETED REGRESSION PASSED" in gate},
        "first_chunk_tokens_from_engine_config": first_chunk,
        "grid": [{"prompt_tokens": p, "second_chunk_q_len": q_len(p)} for p in points],
        "request_checks": {
            "measured_requests": len(meas_reqs),
            "measured_requests_succeeded": sum(1 for r in meas_reqs if "row" in r and r["row"]["output_tokens"] == 32
                                               and r["row"]["prompt_tokens"] == r["row"]["planned_prompt_tokens"]),
            "request_guard_fired": any(ser[d]["tags"].get("S3C_REQUEST_TIMEOUT") for d in SERIES),
            "request_failures": any(ser[d]["tags"].get("S3C_REQUEST_FAILURE") for d in SERIES),
            "watchdog_or_stop_events": timeouts,
            "series_processes": {k: {x: v[x] for x in ("returncode", "timed_out", "elapsed_s",
                                                         "group_processes_remaining")} for k, v in procs.items()},
            "jit_lines_in_measured_requests": sum(r["jit"] for r in meas_reqs),
            "jit_lines_between_measured_requests": sum(ser[d]["jit_between_measured_requests"] for d in SERIES),
            "oom_lines": sum(r["oom"] for r in all_reqs),
            "prompt_hash_equal_across_dtypes_all_points": all(r["prompt_hash_equal_across_dtypes"] for r in rows),
            "conditioning_requests": {d: [r["row"]["planned_prompt_tokens"] for r in conditioning(d)] for d in SERIES},
            "conditioning_excluded_from_measured_statistics": all(r["begin"]["role"] == "measured" for r in meas_reqs)
            and set(points) == {r["row"]["planned_prompt_tokens"] for r in measured("bfloat16")},
            "series_complete": {d: ser[d]["complete"] for d in SERIES},
        },
        "conditioning_stage3c_verification": {
            "markers": len(cond_markers), "q_len_context_len": sorted(set(cond_markers)),
            "note": "verification only (path entered, all layers); not a latency distribution",
        },
        "points": rows,
        "noise_reference": {"bf16_points": list(NOISE_REF_POINTS), "bf16_ttft_span_ms": noise_span,
                            "note": "single-sample BF16 TTFT spread over the near-16384 points; used as the "
                                    "resolution limit for small differences"},
        "boundary_31_32_33": {"excess_ms": {"31": ex31, "32": ex32, "33": ex33},
                              "step_31_to_32_ms": step_31_32, "step_32_to_33_ms": step_32_33},
        "large_q_excess_per_token": {"q_lens": [r["second_chunk_q_len"] for r in large], "ms_per_token": per_tok,
                                     "max_over_min": spread_ratio},
        "conclusions": conclusions,
        "external_frozen_reference_failed_attempt_1": None if fref is None else {
            "source": "results/mlsys2027/context_scaling/failed_attempt_1/failure_analysis.json",
            "q_len": fref["chunked_prefill_markers"][0]["q_len"],
            "warmup_ttft_ms": [w["ttft_ms"] for w in fref["warmups"]],
            "median_warmup_ttft_ms": statistics.median(w["ttft_ms"] for w in fref["warmups"]),
            "layer_markers": fref["chunk_marker_count"], "median_layer_interval_s": fref["chunk_marker_interval_s_median"],
            "note": "external frozen reference; never pooled with the diagnostic series",
        },
        "provenance": {k: manifest["provenance"][k] for k in ("git_head", "vllm_kvquant_tree", "rabit_kv2_sha256",
                                                              "runner_script_sha256", "modal_app_sha256",
                                                              "worker_sha256")},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    out = a.diag_dir / "summary.json"
    out.write_text(json.dumps(build(a.diag_dir), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
