"""
RABIT-KV MLSys 2027 -- read-only reproducibility audit of the IDENTICAL-config generated-QA controls.

Written after Experiment 10 Attempt 1 failed its HotpotQA R4 control-reproduction gate (the second such failure after
Experiment 9 Attempt 1). No GPU, no model, no new generation: this parses the committed raw logs of every run of the
UNCHANGED canonical rabit2 config ("2b META8g64 K3V2 G32 R4") and of the bf16 baseline on HotpotQA and Qasper, and
compares every run, per example, with the frozen canonical reference run (results/quality/*.log, the source of the
canonical targets pinned by the Experiment 1 runner).

Only the `bf16` and `rabit2` rows are read. Treatment rows (K2/K4, V1/V3, G16/G64, R0/R2/R8, rabit8/4/3) are never
parsed, so they cannot influence the audit. Scores are the official LongBench qa_f1_score values printed by the
canonical scripts; answers are compared as the exact strings the scripts logged (no normalisation).

Usage:
    python benchmarks/mlsys2027/audit_qa_control_reproducibility.py --write   (writes the audit JSON)
    python benchmarks/mlsys2027/audit_qa_control_reproducibility.py --check   (regenerates and compares)
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "mlsys2027" / "control_reproducibility_audit" / "qa_control_audit.json"
CONTROL_METHODS = ("bf16", "rabit2")
CONTROL_CONFIG_LINES = {"bf16": "bf16: BF16 baseline", "rabit2": "rabit2: 2b META8g64 K3V2 G32 R4"}
REFERENCE = "canonical"
# Every committed run of the identical canonical control, in chronological order. status: whether the run's
# evidence was accepted; the audit uses ALL of them (identical config) but labels each.
RUNS = {
    "hotpotqa": [
        ("canonical", "results/quality/hotpotqa.log", "canonical reference (source of the pinned targets)"),
        ("exp1", "results/mlsys2027/quality_frontier/hotpotqa.log", "accepted"),
        ("exp7", "results/mlsys2027/ablations/k_bit/hotpotqa.log", "accepted"),
        ("exp8", "results/mlsys2027/ablations/v_bit/hotpotqa.log", "accepted"),
        ("exp9_attempt1", "results/mlsys2027/ablations/group_size/failed_attempt_1/hotpotqa.log",
         "invalid_control_reproduction (excluded)"),
        ("exp9_attempt2", "results/mlsys2027/ablations/group_size/hotpotqa.log", "accepted"),
        ("exp10_attempt1", "results/mlsys2027/ablations/residual_window/failed_attempt_1/hotpotqa.log",
         "invalid_control_reproduction (excluded)"),
    ],
    "qasper": [
        ("canonical", "results/quality/qasper.log", "canonical reference (source of the pinned targets)"),
        ("exp1", "results/mlsys2027/quality_frontier/qasper.log", "accepted"),
        ("exp7", "results/mlsys2027/ablations/k_bit/qasper.log", "accepted"),
        ("exp8", "results/mlsys2027/ablations/v_bit/qasper.log", "accepted"),
        ("exp9_attempt2", "results/mlsys2027/ablations/group_size/qasper.log", "accepted"),
        # Exp9 attempt 1 and Exp10 attempt 1 stopped at HotpotQA: Qasper was never run.
    ],
}
SAMPLE_RE = re.compile(r"^Sample (\d+)/(\d+) \(dataset index (\d+)\)")
ROW_RE = re.compile(r"^\s+(\S+)\s+score=([0-9.]+) KV=([0-9.]+) MB answer=(.*)$")
SUMMARY_COL = 1  # f1_pct column of the printed summary row (same parser as the accepted Exp1 runner)


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def parse_run(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    gpu = next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("GPU:")), None)
    config_ok = {m: any(ln.strip() == CONTROL_CONFIG_LINES[m] for ln in text.splitlines()) for m in CONTROL_METHODS}
    per = {m: [] for m in CONTROL_METHODS}
    current = None
    for ln in text.splitlines():
        s = SAMPLE_RE.match(ln)
        if s:
            current = {"sample": int(s.group(1)), "dataset_index": int(s.group(3))}
            continue
        r = ROW_RE.match(ln)
        if r and current is not None and r.group(1) in CONTROL_METHODS:
            per[r.group(1)].append({**current, "score": float(r.group(2)), "kv_mb": float(r.group(3)),
                                    "answer": ast.literal_eval(r.group(4))})
    summary = {}
    for m in CONTROL_METHODS:
        rows = [ln for ln in text.splitlines() if re.match(rf"^{m}\s", ln)]
        summary[m] = float(rows[-1].split()[SUMMARY_COL]) if rows else None
    return {"gpu": gpu, "config_line_present": config_ok, "per_example": per, "summary_f1": summary}


def audit() -> dict:
    out = {"kind": "read-only reproducibility audit of identical-config generated-QA controls",
           "written_after": "Experiment 10 Attempt 1 failed the HotpotQA R4 control-reproduction gate (second such "
                            "failure after Experiment 9 Attempt 1)",
           "gpu_used": False, "treatment_rows_read": False,
           "reference": "canonical = results/quality/<benchmark>.log (source of the pinned canonical targets)",
           "scorer": "official LongBench qa_f1_score as printed by the canonical scripts; answers compared as the "
                     "exact logged strings (no normalisation)",
           "benchmarks": {}}
    for bench, runs in RUNS.items():
        parsed = {rid: {**parse_run(ROOT / p), "path": p, "status": st, "log_sha256_lf": sha256_lf(ROOT / p)}
                  for rid, p, st in runs}
        ref = parsed[REFERENCE]
        bench_out = {"runs": {}, "controls": {}}
        for rid, pr in parsed.items():
            bench_out["runs"][rid] = {k: pr[k] for k in ("path", "status", "gpu", "config_line_present",
                                                         "log_sha256_lf")}
        for m in CONTROL_METHODS:
            ref_rows = ref["per_example"][m]
            n = len(ref_rows)
            ctrl = {"n_examples": n, "reference_run": REFERENCE, "runs": {}}
            unstable: dict[int, dict] = {}
            for rid, pr in parsed.items():
                rows = pr["per_example"][m]
                assert [r["dataset_index"] for r in rows] == [r["dataset_index"] for r in ref_rows], (bench, rid, m)
                scores = [r["score"] for r in rows]
                mean_f1 = round(100 * sum(scores) / len(scores), 1)
                score_diff = [i for i, (a, b) in enumerate(zip(rows, ref_rows)) if a["score"] != b["score"]]
                answer_diff = [i for i, (a, b) in enumerate(zip(rows, ref_rows)) if a["answer"] != b["answer"]]
                kv_equal = all(a["kv_mb"] == b["kv_mb"] for a, b in zip(rows, ref_rows))
                ctrl["runs"][rid] = {
                    "status": pr["status"], "gpu": pr["gpu"],
                    "aggregate_f1_printed": pr["summary_f1"][m], "aggregate_f1_from_scores": mean_f1,
                    "score_vector": scores,
                    "n_score_mismatches_vs_reference": len(score_diff), "score_mismatch_indices": score_diff,
                    "n_answer_text_mismatches_vs_reference": len(answer_diff), "answer_mismatch_indices": answer_diff,
                    "answer_mismatch_same_score_indices": [i for i in answer_diff if i not in score_diff],
                    "per_example_kv_mb_identical_to_reference": kv_equal,
                    "aggregate_delta_vs_reference": round(pr["summary_f1"][m] - ref["summary_f1"][m], 1)}
                for i in answer_diff:
                    u = unstable.setdefault(i, {"dataset_index": ref_rows[i]["dataset_index"],
                                                "reference_score": ref_rows[i]["score"],
                                                "reference_answer": ref_rows[i]["answer"], "variants": {}})
                    u["variants"][rid] = {"score": rows[i]["score"], "answer": rows[i]["answer"],
                                          "gpu": pr["gpu"]}
            others = {k: v for k, v in ctrl["runs"].items() if k != REFERENCE}
            aggs = [v["aggregate_f1_printed"] for v in ctrl["runs"].values()]
            for i, u in unstable.items():
                u["n_runs_differing_answer"] = len(u["variants"])
                u["n_runs_differing_score"] = sum(v["score"] != u["reference_score"] for v in u["variants"].values())
                u["score_contribution_range_points"] = round(
                    100 * (max([u["reference_score"], *[v["score"] for v in u["variants"].values()]])
                           - min([u["reference_score"], *[v["score"] for v in u["variants"].values()]])) / n, 2)
            ctrl["unstable_examples"] = {str(i): unstable[i] for i in sorted(unstable)}
            ctrl["summary"] = {
                "n_runs_including_reference": len(ctrl["runs"]),
                "max_score_mismatches_in_any_run": max(v["n_score_mismatches_vs_reference"] for v in others.values()),
                "max_answer_text_mismatches_in_any_run": max(v["n_answer_text_mismatches_vs_reference"]
                                                             for v in others.values()),
                "score_mismatch_count_by_run": {k: v["n_score_mismatches_vs_reference"] for k, v in others.items()},
                "examples_ever_score_mismatched": sorted({i for v in others.values()
                                                          for i in v["score_mismatch_indices"]}),
                "examples_ever_answer_mismatched": sorted(unstable),
                "aggregate_f1_min": min(aggs), "aggregate_f1_max": max(aggs),
                "aggregate_f1_range": round(max(aggs) - min(aggs), 1),
                "aggregate_f1_mean": round(statistics.mean(aggs), 2),
                "one_example_max_aggregate_effect_points": round(100 / n, 2)}
            bench_out["controls"][m] = ctrl
        out["benchmarks"][bench] = bench_out
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    data = json.loads(json.dumps(audit()))
    if a.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
        return 0
    same = json.loads(OUT.read_text(encoding="utf-8")) == data
    print("audit regenerates identically" if same else "AUDIT DIFFERS FROM REGENERATION")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
