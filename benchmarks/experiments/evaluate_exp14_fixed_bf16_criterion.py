"""
MLSys 2027 Experiment 14 -- OFFLINE posterior evaluation of the proposed FIXED BF16 numerical-conformance criterion
against the accepted Attempt-4 capture (commit 4947050). No Modal, no GPU, no recomputation of attention.

Criterion under evaluation (per checkpoint; reference = independent Rabit2OnlineStateRef state, kernel semantics,
output rounded to BF16 = ref_bf16; u = 2^-8):
  C1 (normwise):   max_i |rt_i - ref_bf16_i| <= u * max_i |ref_bf16_i|
  C2 (relative):   for |ref_bf16_i| >= 0.01 * max|ref_bf16|:  |rt_i - ref_bf16_i| / |ref_bf16_i| <= u

What the capture records (R_sem, reference_state): max|ref| of the UNROUNDED reference, the max abs error vs the
BF16-rounded reference, the exact-equal count and the ULP histogram. Hence:
  * C1 is computed EXACTLY: rounding is monotone, so max|ref_bf16| = bf16(max|ref_fp32|);
  * C2 per-element relative errors vs ref_bf16 are NOT recorded; each checkpoint is classified by BF16 spacing:
      - pass_all_exact: every element equals ref_bf16;
      - pass_all_disagreements_below_floor: any disagreeing pair (d >= 1 ULP apart, error <= Er) has smaller magnitude
        < 256 * Er, so |ref| < 257 * Er < floor for every disagreement;
      - FAIL_1ulp_flip_above_floor: all disagreements are 1 ULP (max_ulp == 1) and the pair achieving Er has smaller
        magnitude >= 128 * Er >= floor; a 1-ULP gap relative to its reference is in (2^-8, 2^-7] -- i.e. > u --
        except the single edge case where the reference is exactly a power of two and the runtime value is the next
        lower BF16 number (relative error exactly 2^-8); the capture cannot exclude that edge case;
      - undetermined: neither provable from the recorded maxima.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/mlsys2027/second_model/shape_gate_numdiag/attempt_4/numdiag_result.json"
OUT = ROOT / "results/mlsys2027/second_model/shape_gate_numdiag/fixed_bf16_criterion_evaluation.json"
EXPECTED_SHA256 = "10ce1ac0aa7073ce70e7fff747bf792bd81188a7cb6c3019080c70bbc0337f95"
U = 2.0 ** -8
FLOOR = 0.01


def bf16(x: float) -> float:
    """Round-to-nearest-even FP32 -> BF16 (x is an FP32 value)."""
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", b))[0]


def classify(v: dict) -> dict:
    m = bf16(v["ref_max_abs"])
    er = v["max_abs_err_vs_dtype_rounded_ref"]
    floor = FLOOR * m
    nonexact = v["elements"] - v["exact_equal_elements"]
    c1 = er / (U * m) if m else (0.0 if er == 0 else float("inf"))
    if nonexact == 0:
        c2 = "pass_all_exact"
    elif 257 * er < floor:
        c2 = "pass_all_disagreements_below_floor"
    elif v["max_ulp"] == 1 and 128 * er >= floor:
        c2 = "FAIL_1ulp_flip_above_floor"
    else:
        c2 = "undetermined"
    return {"max_abs_ref_bf16": m, "max_abs_err_vs_ref_bf16": er, "C1_ratio": c1, "C1_pass": c1 <= 1.0,
            "C2_class": c2, "nonexact_elements": nonexact, "max_ulp": v["max_ulp"]}


def main() -> int:
    data = RESULT.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("captured result differs from the accepted Attempt-4 capture")
    s = json.loads(data)
    out = {"source_sha256": EXPECTED_SHA256, "criterion": {"u_bf16": U, "relative_floor": "0.01 * max|ref_bf16|"},
           "reference": "R_sem reference_state (independent Rabit2OnlineStateRef state)", "geometries": {}}
    for g, G in s["geometries"].items():
        rows = []
        for rn, r in G["replays"].items():
            for c in r["checkpoints"]:
                rows.append({"replay": rn, "T": c["T"], "old_gate_failed": c["gate_attempt1_would_fail"],
                             **classify(c["R_sem"]["reference_state"])})
        worst = max(rows, key=lambda x: x["C1_ratio"])
        out["geometries"][g] = {
            "checkpoints": len(rows), "C1_pass": sum(x["C1_pass"] for x in rows),
            "C1_worst_ratio": worst["C1_ratio"], "C1_worst_at": {"replay": worst["replay"], "T": worst["T"]},
            "C2_classes": dict(Counter(x["C2_class"] for x in rows)),
            "old_gate_failures": {"count": sum(x["old_gate_failed"] for x in rows),
                                  "C1_pass": sum(x["C1_pass"] for x in rows if x["old_gate_failed"]),
                                  "C2_classes": dict(Counter(x["C2_class"] for x in rows if x["old_gate_failed"]))},
            "rows": rows}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({g: {k: v for k, v in G.items() if k != "rows"} for g, G in out["geometries"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
