"""
MLSys 2027 Experiment 14 -- OFFLINE posterior evaluation of the FINAL candidate fixed dtype-aware numerical-conformance
criterion against the accepted Attempt-4 capture (commit 4947050). No Modal, no GPU, no recomputation.

r = UNROUNDED FP32 kernel-semantics reference output (R_sem, primary: reference_state = independent
    Rabit2OnlineStateRef state); y = runtime BF16 output (as FP32); u = 2^-8; floor = 0.01 * max|r|.
  C1: max|y - r| <= u * max|r|                         (if max|r| == 0: y must be exactly 0)
  C2: max over {i : |r_i| >= floor} of |y_i - r_i| / |r_i| <= u
Both are computed EXACTLY from recorded fields of the frozen diagnostic (exp14_shape_gate_numdiag.compare):
  max|r| = ref_max_abs; max|y - r| = max_abs_err_vs_fp32_ref (error vs the UNROUNDED reference);
  C2 max = max_rel_err, whose mask is exactly |r| >= 1e-2 * max|r| (REL_FLOOR = 1e-2).
The BF16-rounded reference is reported as a diagnostic only. runtime_state results are a secondary consistency check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/mlsys2027/second_model/shape_gate_numdiag/attempt_4/numdiag_result.json"
OUT = ROOT / "results/mlsys2027/second_model/shape_gate_numdiag/unrounded_ref_criterion_evaluation.json"
EXPECTED_SHA256 = "10ce1ac0aa7073ce70e7fff747bf792bd81188a7cb6c3019080c70bbc0337f95"
U = 2.0 ** -8


def check(v: dict) -> dict:
    m, e, rel = v["ref_max_abs"], v["max_abs_err_vs_fp32_ref"], v["max_rel_err"]
    if m == 0:
        return {"C1_ratio": 0.0 if e == 0 else float("inf"), "C1_pass": e == 0, "C2_ratio": 0.0, "C2_pass": e == 0}
    c1 = e / (U * m)
    c2 = (rel or 0.0) / U
    return {"C1_ratio": c1, "C1_pass": c1 <= 1.0, "C2_ratio": c2, "C2_pass": c2 <= 1.0,
            "max_abs_r": m, "max_abs_err": e, "max_rel_err_above_floor": rel,
            "diag_bf16_rounded_ref_max_abs_err": v["max_abs_err_vs_dtype_rounded_ref"], "diag_max_ulp": v["max_ulp"]}


def summarize(rows: list[dict], side: str) -> dict:
    w1 = max(rows, key=lambda x: x[side]["C1_ratio"])
    w2 = max(rows, key=lambda x: x[side]["C2_ratio"])
    return {"checkpoints": len(rows),
            "C1_pass": sum(x[side]["C1_pass"] for x in rows), "C1_fail": sum(not x[side]["C1_pass"] for x in rows),
            "C2_pass": sum(x[side]["C2_pass"] for x in rows), "C2_fail": sum(not x[side]["C2_pass"] for x in rows),
            "C1_worst_ratio": w1[side]["C1_ratio"], "C1_worst_at": {"replay": w1["replay"], "T": w1["T"]},
            "C2_worst_ratio": w2[side]["C2_ratio"], "C2_worst_at": {"replay": w2["replay"], "T": w2["T"]}}


def main() -> int:
    data = RESULT.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("captured result differs from the accepted Attempt-4 capture")
    s = json.loads(data)
    out = {"source_sha256": EXPECTED_SHA256, "u": U, "floor": "0.01 * max|r| (unrounded)", "geometries": {}}
    for g, G in s["geometries"].items():
        rows = []
        for rn, rep in G["replays"].items():
            for c in rep["checkpoints"]:
                rows.append({"replay": rn, "T": c["T"], "old_gate_failed": c["gate_attempt1_would_fail"],
                             "bytes_identical": c["bytes_identical"],
                             "reference_state": check(c["R_sem"]["reference_state"]),
                             "runtime_state": check(c["R_sem"]["runtime_state"])})
        old = [x for x in rows if x["old_gate_failed"]]
        main_ = [x for x in rows if x["replay"].startswith("main_")]
        out["geometries"][g] = {
            "primary_reference_state": summarize(rows, "reference_state"),
            "secondary_runtime_state": summarize(rows, "runtime_state"),
            "old_gate_failure_subset": summarize(old, "reference_state"),
            "main_2048_replays": summarize(main_, "reference_state"),
            "bytes_identical_all": all(x["bytes_identical"] for x in rows),
            "rows": rows}
    out["all_pass"] = all(G["primary_reference_state"]["C1_fail"] == 0 and G["primary_reference_state"]["C2_fail"] == 0
                          for G in out["geometries"].values())
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": out["all_pass"], **{g: {k: v for k, v in G.items() if k != "rows"}
                                                       for g, G in out["geometries"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
