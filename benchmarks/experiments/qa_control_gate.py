"""
RABIT-KV MLSys 2027 -- post-failure per-example generated-QA CONTROL gate (transparent post-failure methodological
amendment; NOT pre-registered). Written after Experiment 10 Attempt 1 failed the HotpotQA R4 control-reproduction gate.

Derives, ONLY from the committed identical-config control audit
(results/mlsys2027/control_reproducibility_audit/qa_control_audit.json -- bf16 and canonical rabit2 rows only; no
treatment row is ever read), separately for each benchmark (hotpotqa, qasper) and control (bf16, rabit2):

    historical_max_score_mismatch_count  = max over identical-config runs of #examples whose logged official
                                           LongBench F1 differs from the canonical reference run's
    historical_max_l1_score_distance     = max over those runs of sum_i |score_i - reference_score_i|

and evaluates a new run's control rows against the canonical reference with exactly those maxima (no slack).
Scores are compared as the evaluator's printed 3-decimal values, held as integer thousandths, so the L1 sums are exact.
Answer-text differences with an identical score are recorded but are not mismatches. The official aggregate F1 is
reported unchanged; its position relative to the historical envelope is REPORT-ONLY.

Usage:
    python benchmarks/mlsys2027/qa_control_gate.py --write   (writes the amendment once; never overwrites)
    python benchmarks/mlsys2027/qa_control_gate.py --check   (regenerates and compares)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_qa_control_reproducibility as audit  # noqa: E402  (read-only reuse: run list and control-row parser)

ROOT = audit.ROOT
AUDIT_JSON = audit.OUT
AMENDMENT = ROOT / "results" / "mlsys2027" / "control_reproducibility_audit" / "qa_control_gate_amendment.json"
BENCHMARKS = ("hotpotqa", "qasper")
CONTROLS = audit.CONTROL_METHODS  # ("bf16", "rabit2")
IDENTITY_HEADER_RE = re.compile(r"^(Dataset|Length bucket|Samples|Maximum input length|Metric):")
IDENTITY_SAMPLE_RE = re.compile(r"^(Sample \d+/\d+ \(dataset index \d+\)|Ground truth:)")


def parse_text(text: str) -> dict:
    """Control rows (bf16, rabit2 ONLY) of one log, with the audit's own regexes; any other row is skipped unread."""
    per = {m: [] for m in CONTROLS}
    current = None
    for ln in text.splitlines():
        s = audit.SAMPLE_RE.match(ln)
        if s:
            current = {"sample": int(s.group(1)), "dataset_index": int(s.group(3))}
            continue
        r = audit.ROW_RE.match(ln)
        if r and current is not None and r.group(1) in CONTROLS:
            per[r.group(1)].append({**current, "score": float(r.group(2)), "kv_mb": float(r.group(3)),
                                    "answer": audit.ast.literal_eval(r.group(4))})
    summary = {}
    for m in CONTROLS:
        rows = [ln for ln in text.splitlines() if re.match(rf"^{m}\s", ln)]
        summary[m] = float(rows[-1].split()[audit.SUMMARY_COL]) if rows else None
    return {"per_example": per, "summary_f1": summary}


def milli(score: float) -> int:
    """A printed 3-decimal F1 score as exact integer thousandths."""
    return int(round(score * 1000))


def derive_thresholds(audit_data: dict) -> dict:
    """Per benchmark / control maxima over the identical-config runs of the committed audit (controls only)."""
    out = {}
    for b in BENCHMARKS:
        out[b] = {}
        for m in CONTROLS:
            ctrl = audit_data["benchmarks"][b]["controls"][m]
            ref = [milli(s) for s in ctrl["runs"][ctrl["reference_run"]]["score_vector"]]
            per_run = {}
            for rid, run in ctrl["runs"].items():
                if rid == ctrl["reference_run"]:
                    continue
                vec = [milli(s) for s in run["score_vector"]]
                per_run[rid] = {"score_mismatch_count": sum(a != r for a, r in zip(vec, ref)),
                                "l1_score_distance_milli": sum(abs(a - r) for a, r in zip(vec, ref))}
            aggs = [r["aggregate_f1_printed"] for r in ctrl["runs"].values()]
            out[b][m] = {
                "historical_max_score_mismatch_count": max(v["score_mismatch_count"] for v in per_run.values()),
                "historical_max_l1_score_distance": max(v["l1_score_distance_milli"] for v in per_run.values()) / 1000,
                "historical_max_l1_score_distance_milli": max(v["l1_score_distance_milli"] for v in per_run.values()),
                "per_run": per_run,
                "n_historical_runs_excluding_reference": len(per_run),
                "aggregate_f1_envelope_report_only": [min(aggs), max(aggs)]}
    return out


def build_amendment() -> dict:
    audit_data = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    return {
        "label": "transparent post-failure methodological amendment",
        "pre_registered": False,
        "written_after": "Experiment 10 Attempt 1 failed the HotpotQA R4 control-reproduction gate "
                         "(results/mlsys2027/ablations/residual_window/failed_attempt_1/, archived at 9889e33)",
        "motivation": "repeated identical-config generated-QA control instability (HotpotQA rabit2 55.2 / 57.7 from "
                      "one example's greedy answer; Qasper rabit2 34.8-36.2 from three examples)",
        "derived_from": {"audit": "results/mlsys2027/control_reproducibility_audit/qa_control_audit.json",
                         "audit_sha256_lf": audit.sha256_lf(AUDIT_JSON),
                         "rows_used": "bf16 and canonical rabit2 control rows only",
                         "treatment_quality_used": False},
        "status_of_prior_attempts": {"exp10_attempt1": "remains invalid_control_reproduction (excluded)",
                                     "exp9_attempt1": "remains invalid_control_reproduction (excluded)",
                                     "prior_accepted_experiments": "unchanged; none relabeled"},
        "unchanged": ["LongBench scorer (qa_f1_score)", "prompts", "generation settings", "datasets",
                      "sample selection and counts", "configs", "logical-storage accounting",
                      "official aggregate F1 metric and its reporting",
                      "continuation_ppl / NIAH / passage-retrieval control-reproduction rules",
                      "original frozen Exp10 protocol file (benchmarks/mlsys2027/exp10_residual_protocol.json, "
                      "sha256 171cdbcab5c57137edda12ad044e038a52c722f32b740b26cc084ff0539c53d4) -- not modified; "
                      "this amendment overlays it"],
        "changes": "ONLY the QA control-validity rule, prospectively: for hotpotqa and qasper the aggregate "
                   "+/-1.0-point F1 control-reproduction pass/fail gate (bf16 and canonical control) is superseded "
                   "by the per-example gate below; the bf16 / control logical KV MB reproduction checks keep their "
                   "original tolerance",
        "applies_to": ["Experiment 10 Attempt 2", "Experiment 11"],
        "reference": "the frozen canonical run results/quality/<benchmark>.log (per-example scores, dataset indices, "
                     "ordering and ground truths)",
        "gate": {
            "mandatory": ["exact canonical config equality (existing protocol gates)",
                          "exact dataset identity: the Dataset / Length bucket / Samples / Maximum input length / "
                          "Metric header lines equal the reference's",
                          "exact sample indices, ordering, token counts and ground truths: every 'Sample i/N "
                          "(dataset index d): ...' and 'Ground truth: ...' line equals the reference's, in order",
                          "exact sample count: one parsed row per sample for bf16 and for the control",
                          "exact logical-storage accounting (existing protocol gates)",
                          "no missing / malformed per-example rows"],
            "per_example_bound": ("independently for bf16 and for the canonical control: score_mismatch_count <= "
                                  "historical_max_score_mismatch_count AND l1_score_distance <= "
                                  "historical_max_l1_score_distance"),
            "definitions": {
                "score_mismatch": "an example whose logged official F1 (3 decimals) differs from the reference's",
                "l1_score_distance": "sum over examples of |run F1 - reference F1| using the logged per-example "
                                     "scores (0-1 scale), computed exactly in integer thousandths",
                "text_only_differences": "answer-text differences with an identical logged F1 are recorded, not counted"},
            "numerical_tolerance": "none needed: scores are compared as exact integer thousandths of the printed values",
            "slack": "none: thresholds equal the historical maxima"},
        "thresholds": derive_thresholds(audit_data),
        "aggregate_f1": "reported exactly as before (official LongBench aggregate); whether it lies inside "
                        "aggregate_f1_envelope_report_only is REPORT-ONLY and never a pass/fail gate",
    }


def load_amendment() -> dict:
    committed = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if committed != json.loads(json.dumps(build_amendment())):
        raise RuntimeError("qa_control_gate_amendment.json differs from its regeneration")
    return committed


def _identity_lines(text: str) -> tuple[list[str], list[str]]:
    lines = text.splitlines()
    return ([ln for ln in lines if IDENTITY_HEADER_RE.match(ln)],
            [ln for ln in lines if IDENTITY_SAMPLE_RE.match(ln)])


def evaluate(benchmark: str, log_text: str, amendment: dict, reference_text: str | None = None) -> dict:
    """Apply the frozen per-example QA control gate to one run's log (bf16 and canonical rabit2 rows only)."""
    ref_path = ROOT / next(p for rid, p, _ in audit.RUNS[benchmark] if rid == audit.REFERENCE)
    ref_text = reference_text if reference_text is not None else ref_path.read_text(encoding="utf-8", errors="replace")
    ref, run = parse_text(ref_text), parse_text(log_text)
    ref_header, ref_samples = _identity_lines(ref_text)
    header, samples = _identity_lines(log_text)
    checks = {"dataset_header_identical": bool(ref_header) and header == ref_header,
              "sample_indices_order_ground_truth_identical": bool(ref_samples) and samples == ref_samples}
    controls = {}
    for m in CONTROLS:
        thr = amendment["thresholds"][benchmark][m]
        rrows, orows = ref["per_example"][m], run["per_example"][m]
        rows_ok = (len(orows) == len(rrows) and
                   [r["dataset_index"] for r in orows] == [r["dataset_index"] for r in rrows])
        c = {"rows_complete_and_ordered": rows_ok}
        if rows_ok:
            d = [(i, milli(o["score"]), milli(r["score"])) for i, (o, r) in enumerate(zip(orows, rrows))]
            mism = [i for i, a, b in d if a != b]
            l1 = sum(abs(a - b) for _, a, b in d)
            text_only = [i for i, (o, r) in enumerate(zip(orows, rrows)) if o["answer"] != r["answer"] and i not in mism]
            agg = run["summary_f1"][m]
            lo, hi = thr["aggregate_f1_envelope_report_only"]
            c.update({"score_mismatch_count": len(mism), "score_mismatch_indices": mism,
                      "l1_score_distance": l1 / 1000,
                      "max_allowed_mismatch_count": thr["historical_max_score_mismatch_count"],
                      "max_allowed_l1_score_distance": thr["historical_max_l1_score_distance"],
                      "mismatch_count_within_bound": len(mism) <= thr["historical_max_score_mismatch_count"],
                      "l1_within_bound": l1 <= thr["historical_max_l1_score_distance_milli"],
                      "text_only_answer_differences": text_only,
                      "mismatch_details": [{"index": i, "dataset_index": orows[i]["dataset_index"],
                                            "reference_score": rrows[i]["score"], "run_score": orows[i]["score"],
                                            "reference_answer": rrows[i]["answer"], "run_answer": orows[i]["answer"]}
                                           for i in mism],
                      "aggregate_f1": agg, "aggregate_f1_envelope_report_only": [lo, hi],
                      "aggregate_inside_envelope_report_only": agg is not None and lo <= agg <= hi})
        c["passed"] = bool(rows_ok and c.get("mismatch_count_within_bound") and c.get("l1_within_bound"))
        controls[m] = c
    passed = all(checks.values()) and all(c["passed"] for c in controls.values())
    return {"benchmark": benchmark, "checks": checks, "controls": controls, "passed": passed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.write:
        if AMENDMENT.exists():
            raise SystemExit(f"{AMENDMENT.name} already exists; the frozen amendment is never overwritten")
        AMENDMENT.write_text(json.dumps(build_amendment(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {AMENDMENT.relative_to(ROOT).as_posix()}")
        return 0
    load_amendment()
    print("amendment regenerates identically")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
