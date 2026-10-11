"""
Side-by-side comparison of independent Experiment 3 sessions.

Each session is given as LABEL=SUMMARY_JSON=RESULTS_DIR. Both the frozen run #1
summary (build_exp3_summary.py) and generic session summaries
(build_exp3_session_summary.py) are accepted: they share the key names used
here. Before comparing, each summary's recorded LF-normalized result-file
hashes are verified against its results directory, and the GPU UUID is read
from that directory's matched_capacity_latency_summary.json environment.

Sessions are treated as INDEPENDENT matched runs: raw samples are never pooled
across sessions and no pooled cross-session headline or statistical confidence
is computed. Cross-session conclusions are descriptive only (exact capacity
agreement, direction agreement, min-max ranges of the per-session overheads,
absolute-latency differences).

Usage:
    python benchmarks/mlsys2027/compare_exp3_sessions.py \
        --session run_1=results/mlsys2027/deployment/summary.json=results/mlsys2027/deployment \
        --session independent_replication_1=results/mlsys2027/deployment/replication_1/summary.json=results/mlsys2027/deployment/replication_1 \
        --output results/mlsys2027/deployment/replication_comparison.json
    (add --check to recompute and compare against an existing --output file)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MATERIAL_ABS_LATENCY_REL_DIFF = 0.10  # descriptive threshold for "materially different", stated in output


def lf_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def session_row(label: str, summary_path: Path, results_dir: Path) -> dict:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    bad = [f for f, h in s["result_files_sha256"].items() if lf_sha(results_dir / f) != h["lf_normalized"]]
    if bad:
        raise SystemExit(f"{label}: summary hashes do not match raw files: {bad}")
    runner = json.loads((results_dir / "matched_capacity_latency_summary.json").read_text(encoding="utf-8"))
    cap, pooled, dl, drift = s["capacity"], s["pooled_per_dtype"], s["signed_deltas_rabit_minus_bf16_pooled"], s["drift"]
    a_key = next(k for k in drift if k.startswith("bf16_"))
    b_key = next(k for k in drift if k.startswith("rabit_kv2_"))

    def dr(key: str) -> dict:
        return {"legs": drift[key]["legs"],
                **{m: {"median_diff_pct": drift[key][m]["median_diff_pct"],
                       "ranges_overlap": drift[key][m]["ranges_overlap"]} for m in ("tpot_ms", "ttft_ms", "wall_ms")}}

    return {
        "label": label,
        "summary": summary_path.as_posix(),
        "results_dir": results_dir.as_posix(),
        "summary_sha256_lf": lf_sha(summary_path),
        "git_head": s["provenance"]["git_head"],
        "gpu_uuid": [g.get("uuid") for g in runner["environment"]["gpus"]],
        "capacity": {"bf16_tokens": cap["bf16_capacity_tokens"], "rabit_kv2_tokens": cap["rabit_kv2_capacity_tokens"],
                     "ratio": cap["ratio_rabit_over_bf16"]},
        "bf16_pooled": {m: pooled["bfloat16"][f"{m}_ms"]["median"] for m in ("tpot", "ttft", "wall")}
        | {"tpot_p90": pooled["bfloat16"]["tpot_ms"]["p90"]},
        "rabit_kv2_pooled": {m: pooled["rabit_kv2"][f"{m}_ms"]["median"] for m in ("tpot", "ttft", "wall")}
        | {"tpot_p90": pooled["rabit_kv2"]["tpot_ms"]["p90"]},
        "deltas": {k: {"ms": v["signed_delta_ms"], "pct": v["signed_delta_pct"]} for k, v in dl.items()},
        "bf16_drift": dr(a_key),
        "rabit_kv2_drift": dr(b_key),
        "integrity": s["integrity"]["counts"],
        "integrity_all_ok": s["integrity"]["all_ok"],
        "config_status": s["matched_config"]["status"],
    }


def compare(rows: list[dict]) -> dict:
    def rng(values: list[float]) -> dict:
        return {"min": min(values), "max": max(values), "spread": max(values) - min(values)}

    caps = {json.dumps(r["capacity"], sort_keys=True) for r in rows}
    signs = {k: {r["label"]: (r["deltas"][k]["ms"] > 0) - (r["deltas"][k]["ms"] < 0) for r in rows}
             for k in rows[0]["deltas"]}
    abs_diff = {}
    for dtype in ("bf16_pooled", "rabit_kv2_pooled"):
        abs_diff[dtype] = {}
        for m in ("tpot", "ttft", "wall"):
            vals = [r[dtype][m] for r in rows]
            rel = (max(vals) - min(vals)) / min(vals)
            abs_diff[dtype][m] = {"per_session": {r["label"]: r[dtype][m] for r in rows},
                                  "relative_spread": rel,
                                  "materially_different": rel > MATERIAL_ABS_LATENCY_REL_DIFF}
    return {
        "capacity_matches_exactly_across_sessions": len(caps) == 1,
        "latency_direction_agrees_across_sessions": {k: len(set(v.values())) == 1 for k, v in signs.items()},
        "latency_direction_per_session": {k: {lab: ("rabit_kv2 slower" if sg > 0 else "rabit_kv2 faster" if sg < 0 else "equal")
                                              for lab, sg in v.items()} for k, v in signs.items()},
        "tpot_median_overhead_pct_range": rng([r["deltas"]["tpot_median"]["pct"] for r in rows]),
        "tpot_p90_overhead_pct_range": rng([r["deltas"]["tpot_p90"]["pct"] for r in rows]),
        "ttft_median_overhead_pct_range": rng([r["deltas"]["ttft_median"]["pct"] for r in rows]),
        "wall_median_overhead_pct_range": rng([r["deltas"]["wall_median"]["pct"] for r in rows]),
        "absolute_latency_across_sessions": abs_diff,
        "materiality_threshold_relative": MATERIAL_ABS_LATENCY_REL_DIFF,
        "relative_spread_definition": "(max - min) / min across sessions; materially_different if > threshold",
        "all_sessions_integrity_fully_passed": all(r["integrity_all_ok"] and r["integrity"]["failed"] == 0 for r in rows),
        "all_sessions_config_passed": all(r["config_status"] == "passed" for r in rows),
        "same_gpu_across_sessions": len({json.dumps(r["gpu_uuid"]) for r in rows}) == 1,
    }


def build(sessions: list[str]) -> dict:
    rows = []
    for spec in sessions:
        label, summary, results = spec.split("=", 2)
        rows.append(session_row(label, Path(summary), Path(results)))
    return {
        "comparison": "MLSys 2027 Experiment 3 -- independent matched sessions, side by side",
        "generated_by": "benchmarks/mlsys2027/compare_exp3_sessions.py",
        "method": ("Each session is an independent matched ABBA run with its own pooled statistics. Raw samples are "
                   "NOT pooled across sessions; no cross-session pooled headline and no statistical confidence is "
                   "computed from this small number of sessions. Conclusions are descriptive only."),
        "sessions": rows,
        "descriptive_conclusions": compare(rows),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", action="append", required=True, help="LABEL=SUMMARY_JSON=RESULTS_DIR")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="recompute and compare with the existing --output")
    args = ap.parse_args(argv)
    if len(args.session) < 2:
        raise SystemExit("need at least two sessions")
    result = build(args.session)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != json.loads(text):
            print("CHECK FAILED: existing comparison differs from recomputation")
            return 1
        print("CHECK OK: comparison reproduces exactly")
        return 0
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
