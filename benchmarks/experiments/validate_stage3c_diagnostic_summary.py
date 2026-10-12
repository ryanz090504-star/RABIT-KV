"""
Validator for results/mlsys2027/diagnostics/stage3c_cliff/summary.json.

Independent of build_stage3c_diagnostic_summary.py (no shared code): every value
is recomputed by re-parsing the raw diagnostic files, and the runner's own
diagnostic_analysis.json is used only as an independent cross-check. The only
constants are the pre-registered grid, the frozen first-chunk size and the
conclusion rules; no measured value is hard-coded.

Read-only. Exits non-zero on any failure.

Usage:
    python benchmarks/mlsys2027/validate_stage3c_diagnostic_summary.py \
        --diag-dir results/mlsys2027/diagnostics/stage3c_cliff
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EPS = 1e-9
GRID = [16384, 16385, 16386, 16415, 16416, 16417, 16896, 17408, 18432, 20480, 24576]
FIRST_CHUNK = 16384
CONDITIONING = 16416
RAW = ["modal_session.log", "correctness_gate.log", "bf16_series.log", "rabit_kv2_series.log", "manifest.json",
       "integrity_check.json"]


def h(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows_of(path: Path) -> tuple[list[dict], dict, bool]:
    """(requests in order, effective config, complete) -- JIT/marker lines attributed to the open request."""
    reqs, eff, done, open_req = [], None, False, None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s == "S3C_SERIES_COMPLETE":
            done = True
        elif s.startswith("S3C_POINT_BEGIN="):
            open_req = {"begin": json.loads(s.split("=", 1)[1]), "jit": 0, "mark": 0}
            reqs.append(open_req)
        elif s.startswith("S3C_POINT="):
            open_req["row"] = json.loads(s.split("=", 1)[1])
            open_req = None
        elif s.startswith("S3C_EFFECTIVE_ENGINE_CONFIG="):
            eff = json.loads(s.split("=", 1)[1])
        elif s.startswith(("S3C_REQUEST_TIMEOUT=", "S3C_REQUEST_FAILURE=")):
            reqs.append({"begin": {"role": "failed"}, "failed": s})
        elif open_req is not None:
            open_req["jit"] += "Triton kernel JIT compilation during inference" in s
            open_req["mark"] += "RABIT2_STAGE3C_CHUNKED_PREFILL_ACTIVE" in s
    return reqs, eff, done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag-dir", type=Path, required=True)
    d = ap.parse_args(argv).diag_dir
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    fails, n = [], [0]

    def need(ok, msg):
        n[0] += 1
        if not ok:
            fails.append(msg)

    def eq(a, b):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= EPS

    def section(fn, name):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 (malformed input is a failure, never a crash)
            fails.append(f"{name}: could not evaluate ({type(e).__name__}: {e})")

    raw = {k: rows_of(d / f) for k, f in (("bf16", "bf16_series.log"), ("rabit", "rabit_kv2_series.log"))}
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    integ = json.loads((d / "integrity_check.json").read_text(encoding="utf-8"))
    runner = json.loads((d / "diagnostic_analysis.json").read_text(encoding="utf-8"))
    meas = {k: {r["row"]["planned_prompt_tokens"]: r for r in v[0] if r["begin"].get("role") == "measured"}
            for k, v in raw.items()}
    rec = {}

    def flags():
        need(s["diagnostic_evidence"] is True and s["experiment5_final_evidence"] is False,
             "diagnostic / final-evidence flags wrong")
        need(s["run_status"] == manifest["status"] == "completed", "run status")
        need(s["protected_paths_post_run_status"] == manifest["protected_paths_post_run_status"] == "clean",
             "protected paths")
        need(s["integrity_counts"] == integ["counts"] and integ["all_ok"] is True, "integrity counts")

    def hashes():
        need(sorted(s["raw_files_sha256"]) == sorted(RAW), "raw file set")
        for f in RAW:
            need(s["raw_files_sha256"][f] == h(d / f), f"{f}: SHA-256 differs from file")

    def requests():
        for k, (reqs, eff, done) in raw.items():
            need(done and eff["max_num_batched_tokens"] == FIRST_CHUNK, f"{k}: series incomplete / first chunk")
            need(not any("failed" in r for r in reqs), f"{k}: request timeout/failure present")
            roles = [r["begin"]["role"] for r in reqs]
            need(roles == ["conditioning"] + ["measured"] * len(GRID), f"{k}: request roles {roles}")
            need(reqs[0]["row"]["planned_prompt_tokens"] == CONDITIONING, f"{k}: conditioning prompt")
            need(sorted(meas[k]) == GRID and list(meas[k]) == GRID, f"{k}: measured grid/order")
            for p, r in meas[k].items():
                row = r["row"]
                need(row["prompt_tokens"] == p and row["output_tokens"] == 32
                     and row["prompt_token_ids_sha256"] == row["planned_prompt_token_ids_sha256"],
                     f"{k}@{p}: token counts / prompt hash")
                need(r["jit"] == 0, f"{k}@{p}: JIT during measured request")
        rc = s["request_checks"]
        need(rc["measured_requests"] == rc["measured_requests_succeeded"] == 2 * len(GRID), "22 measured requests")
        need(rc["request_guard_fired"] is False and rc["request_failures"] is False and rc["watchdog_or_stop_events"] == [],
             "request guard / failure / watchdog flags")
        need(rc["jit_lines_in_measured_requests"] == sum(r["jit"] for k in meas for r in meas[k].values()) == 0,
             "measured JIT count")
        late = 0
        for f in ("bf16_series.log", "rabit_kv2_series.log"):  # any JIT line after the conditioning request finished
            text = (d / f).read_text(encoding="utf-8")
            cond_end = next(i for i, ln in enumerate(text.splitlines())
                            if ln.startswith("S3C_POINT=") and '"role": "conditioning"' in ln)
            late += sum("Triton kernel JIT compilation during inference" in ln
                        for ln in text.splitlines()[cond_end + 1:])
        need(late == 0 and rc.get("jit_lines_between_measured_requests") == 0,
             f"JIT line(s) after conditioning ({late}) / between-request JIT flag")
        need(rc["prompt_hash_equal_across_dtypes_all_points"] is all(
            meas["bf16"][p]["row"]["prompt_token_ids_sha256"] == meas["rabit"][p]["row"]["prompt_token_ids_sha256"]
            for p in GRID) is True, "prompt hash equality across dtypes")
        need(rc["conditioning_excluded_from_measured_statistics"] is True
             and rc["conditioning_requests"] == {"bfloat16": [CONDITIONING], "rabit_kv2": [CONDITIONING]},
             "conditioning exclusion")
        cm = s["conditioning_stage3c_verification"]
        need(cm["markers"] == raw["rabit"][0][0]["mark"] == 32 and cm["q_len_context_len"] == [[32, 16384]],
             "conditioning Stage3C verification")

    def points():
        b0, r0 = meas["bf16"][FIRST_CHUNK]["row"]["ttft_ms"], meas["rabit"][FIRST_CHUNK]["row"]["ttft_ms"]
        need([g["prompt_tokens"] for g in s["grid"]] == GRID and [g["second_chunk_q_len"] for g in s["grid"]]
             == [p - FIRST_CHUNK if p > FIRST_CHUNK else None for p in GRID], "grid / q_len mapping")
        need([p["prompt_tokens"] for p in s["points"]] == GRID, "points order")
        for p in s["points"]:
            pt = p["prompt_tokens"]
            q = pt - FIRST_CHUNK if pt > FIRST_CHUNK else None
            b, r = meas["bf16"][pt]["row"], meas["rabit"][pt]["row"]
            exc = (r["ttft_ms"] - r0) - (b["ttft_ms"] - b0)
            path = "single_chunk_dense_prefill" if q is None else "decode_append_path" if q == 1 else \
                "stage3c_per_token_loop"
            need(p["second_chunk_q_len"] == q and p["context_len_derived"] == (FIRST_CHUNK if q else None),
                 f"{pt}: q_len/context_len")
            need(p["stage3c_active_by_source"] is bool(q is not None and q > 1) and p["path_by_source"] == path,
                 f"{pt}: Stage3C classification")
            for key, val in (("bf16_ttft_ms", b["ttft_ms"]), ("rabit_ttft_ms", r["ttft_ms"]),
                             ("bf16_wall_ms", b["wall_ms"]), ("rabit_wall_ms", r["wall_ms"]),
                             ("raw_rabit_minus_bf16_ttft_ms", r["ttft_ms"] - b["ttft_ms"]),
                             ("bf16_increment_over_bf16_16384_ms", b["ttft_ms"] - b0),
                             ("rabit_increment_over_rabit_16384_ms", r["ttft_ms"] - r0),
                             ("baseline_adjusted_excess_ms", exc)):
                need(eq(p[key], val), f"{pt}: {key} does not recompute")
            need((p["excess_per_second_chunk_token_ms"] is None) if not (q and q > 1)
                 else eq(p["excess_per_second_chunk_token_ms"], exc / q), f"{pt}: excess per token")
            need(p["prompt_token_ids_sha256"] == r["prompt_token_ids_sha256"], f"{pt}: prompt hash")
            rec[pt] = {"exc": exc, "q": q, "per": exc / q if q and q > 1 else None, "b": b["ttft_ms"]}

    def conclusions():
        c = s["conclusions"]
        noise = max(rec[p]["b"] for p in (16384, 16385, 16386, 16415, 16416, 16417)) - \
            min(rec[p]["b"] for p in (16384, 16385, 16386, 16415, 16416, 16417))
        need(eq(s["noise_reference"]["bf16_ttft_span_ms"], noise), "noise span")
        per = [rec[p]["per"] for p in GRID if (rec[p]["q"] or 0) >= 512]
        need(eq(s["large_q_excess_per_token"]["max_over_min"], max(per) / min(per)), "large-q spread")
        s31, s32 = rec[16416]["exc"] - rec[16415]["exc"], rec[16417]["exc"] - rec[16416]["exc"]
        bd = s["boundary_31_32_33"]
        need(eq(bd["step_31_to_32_ms"], s31) and eq(bd["step_32_to_33_ms"], s32)
             and eq(bd["excess_ms"]["32"], rec[16416]["exc"]), "31/32/33 values")
        expect = {
            "q_len_1_is_decode_path_by_source": True,
            "q_len_1_excess_within_noise": abs(rec[16385]["exc"]) <= noise,
            "stage3c_begins_at_q_len_gt_1_by_source": True,
            "q_len_2_per_token_excess_in_large_q_range": min(per) * 0.8 <= rec[16386]["per"] <= max(per) * 1.2,
            "no_visible_discontinuity_at_32": abs(s31) <= noise and abs(s32) <= noise and abs(s31 - s32) <= noise,
            "excess_per_token_approximately_stable_large_q": max(per) / min(per) <= 1.25,
            "formal_complexity_law_claimed": False,
            "measured_point_layer_markers_available": False,
        }
        for k, v in expect.items():
            need(c.get(k) is v, f"conclusion {k} = {c.get(k)!r}, recomputed {v!r}")
        need(c.get("failed_attempt_1_layer_timing") == "external frozen reference only (never pooled)",
             "failed_attempt_1 labeling")
        ext = s["external_frozen_reference_failed_attempt_1"]
        need(ext is not None and "never pooled" in ext["note"] and ext["q_len"] == 16352 and ext["layer_markers"] == 32,
             "external frozen reference block")

    def cross_check_runner_analysis():
        rp = {p["prompt_tokens"]: p for p in runner["points"]}
        need(runner["measured_point_layer_markers_available"] is False, "runner analysis marker flag")
        for pt in GRID:
            need(eq(rp[pt]["bf16"]["ttft_ms"], rec[pt]["b"]), f"{pt}: runner analysis BF16 TTFT disagrees")
            need(eq(rp[pt]["rabit_kv2"]["ttft_ms"], meas["rabit"][pt]["row"]["ttft_ms"]),
                 f"{pt}: runner analysis RABIT TTFT disagrees")

    for fn, nm in ((flags, "flags"), (hashes, "hashes"), (requests, "requests"), (points, "points"),
                   (conclusions, "conclusions"), (cross_check_runner_analysis, "cross-check")):
        section(fn, nm)
    print(f"diag: {d.as_posix()} | checks: {n[0]}")
    print(f"FAILURES: {len(fails)}")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
