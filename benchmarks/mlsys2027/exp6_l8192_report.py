"""
MLSys 2027 Experiment 6 -- L8192 three-trial REPORT (read-only post-processing; no GPU, no analyzer change).

Input: the pre-registered cross-trial combine output (combined_summary.json, produced by
`run_experiment6_concurrency.py --prompt-tokens 8192 --combine` from the three accepted trial-level analyses) and
the three trials' ACCEPTED_RUN.json. Every value used here is a TRIAL-LEVEL statistic; no per-request sample is read,
so nothing is pooled across trials.

Adds, descriptively:
  * per dtype x offered C and metric: T1, T2, T3, cross-trial median, min, max, (max - min) / median
    (a descriptive relative spread; n = 3, NOT a confidence interval);
  * RABIT/BF16 ratios computed per trial, then the median of the three trial-level ratios (the combine's
    median(RABIT) / median(BF16) is kept alongside for reference);
  * the capacity result (allocator capacity, derived full-length ceiling, realized max in-flight, all-C overlap per
    trial), reported separately from throughput;
  * an observed run-to-run stability section (no causal attribution).

Usage: python benchmarks/mlsys2027/exp6_l8192_report.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp6_workload as wl  # noqa: E402
import run_experiment6_concurrency as r6  # noqa: E402

LENGTH = 8192
TRIALS = ("1", "2", "3")
PRIMARY = ("requests_per_s", "output_tokens_per_s", "total_tokens_per_s", "latency_median_s", "latency_p90_s",
           "latency_p99_s", "ttft_median_s", "ttft_p90_s", "tpot_median_s", "tpot_p90_s")
ACCEPTED_EVIDENCE_COMMITS = {"1": "6042a1edafc983107c5800f104dd4ab2222c95df",
                             "2": "a812f2f918b314fb95c9f4dd15c54c9d4b170fb3",
                             "3": "81028406846131a3647a3802ee01bd1c6baefced"}
SPREAD_NOTE = "(max - min) / median over n = 3 trial-level values; descriptive only, NOT a confidence interval"
INTERPRETATION = (
    "Across three independent H100 trials, RABIT-KV sustained the offered 64-request concurrency while BF16 was "
    "limited to 47 overlapping in-flight requests, matching its allocator-derived full-length KV ceiling. However, "
    "RABIT-KV had lower request throughput than BF16 at every tested offered concurrency, with the gap widening at "
    "high concurrency. Thus the physical capacity advantage did not translate into a throughput advantage in the "
    "current implementation.")


def spread(values: dict) -> dict:
    v = [values[t] for t in TRIALS]
    if any(x is None for x in v):
        return {"per_trial": values, "median": None, "min": None, "max": None, "relative_spread": None}
    med = statistics.median(v)
    return {"per_trial": values, "median": med, "min": min(v), "max": max(v),
            "relative_spread": (max(v) - min(v)) / med if med else None}


def build_report(combined: dict, accepted: dict) -> dict:
    if not (combined.get("complete_three_trial_result") and combined.get("all_trials_integrity_passed")):
        raise ValueError("the combine output is not a complete, fully valid three-trial result")
    if sorted(combined["per_trial_validity"]) != list(TRIALS) or sorted(accepted) != list(TRIALS):
        raise ValueError("exactly trials 1, 2, 3 are required")
    ct = combined["cross_trial"]
    metrics, ratios = {}, {}
    for dtype in ("bfloat16", "rabit_kv2"):
        for c in wl.CONCURRENCY_GRID:
            row = ct[f"{dtype}|{c}"]
            metrics[f"{dtype}|{c}"] = {m: spread(row[m]["per_trial"]) for m in PRIMARY}
    for c in wl.CONCURRENCY_GRID:
        b, r = ct[f"bfloat16|{c}"], ct[f"rabit_kv2|{c}"]
        ratios[str(c)] = {}
        for m in PRIMARY:
            per = {t: (r[m]["per_trial"][t] / b[m]["per_trial"][t]) if b[m]["per_trial"][t] else None for t in TRIALS}
            s = spread(per)
            s["ratio_of_cross_trial_medians_from_combine"] = combined["rabit_over_bf16_cross_trial_median_ratio"][str(c)][m]
            s["identical_to_ratio_of_medians"] = (s["median"] is not None and
                                                  abs(s["median"] - s["ratio_of_cross_trial_medians_from_combine"]) < 1e-12)
            ratios[str(c)][m] = s
    capacity = {}
    for dtype, tokens in r6.EXPECTED_CAPACITY.items():
        blocks = tokens // 32
        ceil = r6.allocator_full_length_ceiling({"num_gpu_blocks": blocks, "block_size": 32}, LENGTH)
        capacity[dtype] = {"allocator_capacity_kv_tokens": tokens, **ceil,
                           "realized_max_inflight_per_trial": {
                               str(c): ct[f"{dtype}|{c}"]["observed_max_inflight_concurrency"]["per_trial"]
                               for c in wl.CONCURRENCY_GRID},
                           "all_c_inflight_overlap_s_per_trial": {
                               str(c): ct[f"{dtype}|{c}"]["all_c_inflight_overlap_total_s"]["per_trial"]
                               for c in wl.CONCURRENCY_GRID},
                           "outcome_class_per_trial": {str(c): ct[f"{dtype}|{c}"]["outcome_class_per_trial"]
                                                       for c in wl.CONCURRENCY_GRID},
                           "preemptions_per_trial": {str(c): ct[f"{dtype}|{c}"]["preemptions"]["per_trial"]
                                                     for c in wl.CONCURRENCY_GRID}}
    mono = {f"{d}|{c}": all(ct[f"{d}|{c}"]["requests_per_s"]["per_trial"][a]
                            < ct[f"{d}|{c}"]["requests_per_s"]["per_trial"][b] for a, b in (("1", "2"), ("2", "3")))
            for d in ("bfloat16", "rabit_kv2") for c in wl.CONCURRENCY_GRID}
    rel = {d: {str(c): metrics[f"{d}|{c}"]["requests_per_s"]["relative_spread"] for c in wl.CONCURRENCY_GRID}
           for d in ("bfloat16", "rabit_kv2")}
    stability = {
        "requests_per_s_strictly_increasing_T1_T2_T3": mono,
        "all_points_strictly_increasing": all(mono.values()),
        "requests_per_s_relative_spread": rel,
        "rabit_spread_exceeds_bf16_at_C": [c for c in wl.CONCURRENCY_GRID if rel["rabit_kv2"][str(c)] > rel["bfloat16"][str(c)]],
        "distinct_gpu_per_trial": {t: accepted[t]["gpu"]["uuid"] for t in TRIALS},
        "trial_dtype_order": {str(k): list(v) for k, v in wl.TRIAL_DTYPE_ORDER.items()},
        "attribution": "none -- the drift is reported as observed; no cause is claimed"}
    return {
        "experiment": "Experiment 6 L8192 three-trial report (descriptive)",
        "combine_input": "combined_summary.json (pre-registered --combine over the three accepted trial analyses)",
        "cross_trial_statistic": "median of the three trial-level statistics; min / max / relative spread alongside",
        "ratio_statistic": "RABIT/BF16 computed within each trial, then the median of the three trial-level ratios",
        "spread_note": SPREAD_NOTE,
        "sample_pooling": "none -- only trial-level statistics are used; raw per-request samples are never concatenated",
        "trials": {t: {"accepted_evidence_commit": ACCEPTED_EVIDENCE_COMMITS[t], "run_id": accepted[t]["run_id"],
                       "modal_app": accepted[t]["modal_app"], "function_call_id": accepted[t]["function_call_id"],
                       "measurement_commit": accepted[t]["measurement_commit"], "gpu": accepted[t]["gpu"],
                       "remote_session_log_sha256": accepted[t]["canonical_raw_evidence"]["sha256"],
                       "remote_session_log_bytes": accepted[t]["canonical_raw_evidence"]["bytes"],
                       "integrity_counts": combined["per_trial_validity"][t]["integrity_counts"],
                       "dtype_order": list(wl.TRIAL_DTYPE_ORDER[int(t)])} for t in TRIALS},
        "metrics": metrics, "rabit_over_bf16": ratios, "capacity": capacity,
        "c64_comparison_note": ("same offered C = 64 for both dtypes; different realized concurrency (BF16 max 47, "
                                "capacity-bound; RABIT max 64). C64 is NOT a matched-realized-concurrency comparison."),
        "outcome_class_counts": combined["outcome_class_counts"],
        "highest_successfully_tested_concurrency": combined["highest_successfully_tested_concurrency"],
        "rabit_only_extension_rule_evaluation": combined["rabit_only_extension"],
        "stability": stability, "interpretation": INTERPRETATION,
    }


def main() -> int:
    d = r6.out_dir(LENGTH)
    combined = json.loads((d / "combined_summary.json").read_text(encoding="utf-8"))
    accepted = {t: json.loads((d / f"trial_{t}" / "ACCEPTED_RUN.json").read_text(encoding="utf-8")) for t in TRIALS}
    for t in TRIALS:  # the canonical raw logs must still be the accepted ones
        raw = d / f"trial_{t}" / "remote_session.log"
        if r6.sha256_raw(raw) != accepted[t]["canonical_raw_evidence"]["sha256"]:
            raise SystemExit(f"trial {t} raw log differs from its accepted evidence")
    report = build_report(combined, accepted)
    report["provenance"] = {"combined_summary_sha256": r6.sha256_raw(d / "combined_summary.json"),
                            "report_code_commit": r6.run_git("rev-parse", "HEAD"),
                            "report_script_sha256": r6.sha256(Path(__file__)), "generated_utc": r6.now()}
    (d / "combined_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {r6.rel(d / 'combined_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
