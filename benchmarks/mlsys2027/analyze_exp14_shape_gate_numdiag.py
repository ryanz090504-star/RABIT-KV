"""
MLSys 2027 Experiment 14 -- OFFLINE analysis of the VALID non-evidence shape-gate numerical diagnosis (Attempt 4).
Reads only results/mlsys2027/second_model/shape_gate_numdiag/attempt_4/numdiag_result.json (verifies its SHA-256 and
completeness first); no Modal, no GPU, no new computation of attention. Writes attempt_4_analysis.json next to the
attempt directory (derived; the captured result is never modified).

The frozen diagnostic records, per checkpoint and per reference (R_fp32 / R_sem / R_bf16all) and per state
(runtime / independent reference): max |reference output|, max abs error vs the unrounded FP32-accumulated reference
output, max relative error (|ref| >= 1e-2 max|ref|), max error in BF16 ULPs at the reference value, max abs error vs the
BF16-rounded reference output, exact-equal element count and the BF16 ULP histogram {0, 1, 2, 3+} with the max ULP.
It does NOT record element indices or individual runtime / reference values, so outliers are located to the
checkpoint; their magnitude is BOUNDED arithmetically: two same-sign BF16 values d >= 1 ULPs apart differ by at least
d x ulp(smaller), and |v| < 256 ulp(v) for a normal BF16 v (8 significand bits), so the smaller magnitude of an
outlier pair is < 256 * E / d, where E is the checkpoint's max abs error vs the rounded reference; if the pair straddles
zero both magnitudes are <= E. Hence outlier magnitude < max(256 * E / d, E).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_exp14_shape_gate_numdiag as nd  # noqa: E402  (completeness gate; read-only)

ROOT = HERE.parents[1]
RESULT = ROOT / "results/mlsys2027/second_model/shape_gate_numdiag/attempt_4/numdiag_result.json"
OUT = ROOT / "results/mlsys2027/second_model/shape_gate_numdiag/attempt_4_analysis.json"
EXPECTED_SHA256 = "10ce1ac0aa7073ce70e7fff747bf792bd81188a7cb6c3019080c70bbc0337f95"
EXPECTED_BYTES = 705659
OLD_TOL = 5e-3
REFS = ("R_fp32", "R_sem", "R_bf16all")
SIDES = ("runtime_state", "reference_state")
ULP_KEYS = ("0", "1", "2", "3+")


def ulp_bf16(v: float) -> float:
    """BF16 ULP at |v| (normal range)."""
    v = max(abs(v), 2.0 ** -126)
    return 2.0 ** (math.floor(math.log2(v)) - 7)


def outlier_magnitude_bound(err: float, d: int) -> float:
    return max(256.0 * err / d, err) if d >= 1 else float("nan")


def family(name: str) -> str:
    return "main_P2048" if name.startswith("main_") else name


def agg(rows: list[dict], ref: str, side: str) -> dict:
    vs = [r[ref][side] for r in rows]
    hist = {k: sum(v["ulp_hist"][k] for v in vs) for k in ULP_KEYS}
    n = sum(v["elements"] for v in vs)
    worst = max(rows, key=lambda r: r[ref][side]["max_ulp"])
    return {"checkpoints": len(rows), "elements": n,
            "max_abs_err_vs_unrounded_ref": max(v["max_abs_err_vs_fp32_ref"] for v in vs),
            "max_abs_err_vs_bf16_rounded_ref": max(v["max_abs_err_vs_dtype_rounded_ref"] for v in vs),
            "max_ref_output_magnitude": max(v["ref_max_abs"] for v in vs),
            "max_rel_err": max((v["max_rel_err"] or 0.0) for v in vs),
            "max_err_in_bf16_ulp_at_ref": max(v["max_err_in_bf16_ulp_at_ref"] for v in vs),
            "exact_equal_pct": 100.0 * sum(v["exact_equal_elements"] for v in vs) / n,
            "ulp_hist": hist, "ulp_hist_pct": {k: 100.0 * c / n for k, c in hist.items()},
            "max_ulp": worst[ref][side]["max_ulp"], "max_ulp_at_T": worst["T"]}


def main() -> int:
    data = RESULT.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != EXPECTED_SHA256 or len(data) != EXPECTED_BYTES:
        raise SystemExit(f"captured result differs from the validated Attempt-4 capture: {sha} {len(data)}")
    s = json.loads(data)
    comp = nd.completeness_gate(s)
    if not comp["passed"]:
        raise SystemExit(f"completeness gate failed: {comp}")
    out = {"source": {"path": RESULT.relative_to(ROOT).as_posix(), "sha256": sha, "bytes": len(data),
                      "completeness": {k: comp[k] for k in ("rows", "expected_rows", "missing_count", "passed")}},
           "limitations": ["no element indices or individual runtime / reference values are recorded by the frozen "
                           "diagnostic: outliers are located to the checkpoint; magnitudes are bounded arithmetically",
                           "runtime-output magnitude is not recorded; reported only as the derived bound "
                           "ref_max_abs +/- max_abs_err_vs_fp32_ref (R_fp32, runtime state)"],
           "geometries": {}}
    for g, G in s["geometries"].items():
        allrows = [c for rep in G["replays"].values() for c in rep["checkpoints"]]
        # 1. runtime-state and independent-reference-state results must agree exactly
        agree = all(c[ref]["runtime_state"] == c[ref]["reference_state"] and c[ref]["states_identical"]
                    for c in allrows for ref in REFS)
        fams = {}
        for name, rep in G["replays"].items():
            fams.setdefault(family(name), []).extend(rep["checkpoints"])
        per_family = {}
        for fam, rows in fams.items():
            per_family[fam] = {
                "checkpoints": len(rows), "prefix_T": sorted({r["T"] for r in rows}),
                "bytes_identical_all": all(r["bytes_identical"] for r in rows),
                "old_gate_max_abs": max(max(r["gate_attempt1_max_abs_runtime_state"],
                                            r["gate_attempt1_max_abs_reference_state"]) for r in rows),
                "old_gate_failures": sum(r["gate_attempt1_would_fail"] for r in rows),
                **{ref: agg(rows, ref, "runtime_state") for ref in REFS}}
        failing = []
        for name, rep in G["replays"].items():
            for c in rep["checkpoints"]:
                if not c["gate_attempt1_would_fail"]:
                    continue
                f32, sem = c["R_fp32"]["runtime_state"], c["R_sem"]["runtime_state"]
                mag = f32["ref_max_abs"]
                failing.append({
                    "replay": name, "T": c["T"], "closed_pages": c["closed_pages"], "open_tokens": c["open_tokens"],
                    "bytes_identical": c["bytes_identical"], "ref_output_magnitude": mag,
                    "half_ulp_bf16_at_magnitude": 0.5 * ulp_bf16(mag),
                    "old_gate_err_fp32_ref": max(c["gate_attempt1_max_abs_runtime_state"],
                                                 c["gate_attempt1_max_abs_reference_state"]),
                    "R_fp32_err_vs_rounded": f32["max_abs_err_vs_dtype_rounded_ref"], "R_fp32_max_ulp": f32["max_ulp"],
                    "R_sem_err_vs_unrounded": sem["max_abs_err_vs_fp32_ref"],
                    "R_sem_err_vs_rounded": sem["max_abs_err_vs_dtype_rounded_ref"],
                    "R_sem_exact_pct": 100.0 * sem["exact_equal_elements"] / sem["elements"],
                    "R_sem_ulp_hist": sem["ulp_hist"], "R_sem_max_ulp": sem["max_ulp"],
                    "R_sem_unrounded_err_in_ulp_at_magnitude": sem["max_abs_err_vs_fp32_ref"] / ulp_bf16(mag)})
        outliers = []
        for name, rep in G["replays"].items():
            for c in rep["checkpoints"]:
                v = c["R_sem"]["runtime_state"]
                if v["ulp_hist"]["3+"] > 0:
                    e = v["max_abs_err_vs_dtype_rounded_ref"]
                    outliers.append({"replay": name, "T": c["T"], "count_ge3": v["ulp_hist"]["3+"], "max_ulp": v["max_ulp"],
                                     "checkpoint_max_abs_err_vs_rounded_ref": e,
                                     "checkpoint_ref_output_magnitude": v["ref_max_abs"],
                                     "outlier_element_magnitude_upper_bound": outlier_magnitude_bound(e, v["max_ulp"]),
                                     "max_rel_err": v["max_rel_err"]})
        outliers.sort(key=lambda x: -x["max_ulp"])
        out["geometries"][g] = {
            "runtime_state_equals_reference_state_all": agree,
            "bytes_identical_all": all(c["bytes_identical"] for c in allrows),
            "overall": {ref: agg(allrows, ref, "runtime_state") for ref in REFS},
            "per_family": per_family, "old_gate_failing_checkpoints": failing, "R_sem_ge3_ulp_checkpoints": outliers,
            "derived_runtime_output_magnitude_bound": {
                "max": max(c["R_fp32"]["runtime_state"]["ref_max_abs"] + c["R_fp32"]["runtime_state"]["max_abs_err_vs_fp32_ref"]
                           for c in allrows)}}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"written": OUT.relative_to(ROOT).as_posix(), "source_sha256": sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
