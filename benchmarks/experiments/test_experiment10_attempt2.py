"""Offline tests for the post-failure per-example QA control gate and the Experiment 10 Attempt 2 runner (no GPU)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_qa_control_reproducibility as audit  # noqa: E402
import qa_control_gate as qg  # noqa: E402
import run_experiment1_quality_frontier as e1  # noqa: E402
import run_experiment10_attempt2 as a2  # noqa: E402

ROOT = qg.ROOT
TREATMENT_ROWS = ("rabit2_k2", "rabit2_k4", "rabit2_v1", "rabit2_v3", "rabit2_g16", "rabit2_g64", "rabit2_r0",
                  "rabit2_r2", "rabit2_r8", "rabit8", "rabit4", "rabit3")


def _ref_text(b: str) -> str:
    return (ROOT / "results" / "quality" / f"{b}.log").read_text(encoding="utf-8", errors="replace")


def _mutate(text: str, method: str, changes: dict) -> str:
    """Replace the score and/or answer of `method`'s per-example rows at the given example indices."""
    out, k = [], -1
    for ln in text.splitlines():
        m = audit.ROW_RE.match(ln)
        if m and m.group(1) == method:
            k += 1
            if k in changes:
                score, answer = changes[k]
                s = f"{score:.3f}" if score is not None else m.group(2)
                ans = repr(answer) if answer is not None else m.group(4)
                ln = f"  {method:<8} score={s} KV={m.group(3)} MB answer={ans}"
        out.append(ln)
    return "\n".join(out)


def test_thresholds_are_the_historical_maxima_of_control_rows():
    th = qg.load_amendment()["thresholds"]
    exp = {("hotpotqa", "bf16"): (1, 0.065), ("hotpotqa", "rabit2"): (1, 0.5),
           ("qasper", "bf16"): (1, 0.096), ("qasper", "rabit2"): (2, 0.185)}
    for (b, m), (count, l1) in exp.items():
        t = th[b][m]
        assert (t["historical_max_score_mismatch_count"], t["historical_max_l1_score_distance"]) == (count, l1)
        assert t["historical_max_score_mismatch_count"] == max(v["score_mismatch_count"] for v in t["per_run"].values())
        assert t["historical_max_l1_score_distance_milli"] == max(v["l1_score_distance_milli"]
                                                                 for v in t["per_run"].values())
    assert th["hotpotqa"]["rabit2"]["aggregate_f1_envelope_report_only"] == [55.2, 57.7]
    assert th["qasper"]["rabit2"]["aggregate_f1_envelope_report_only"] == [34.8, 36.2]


def test_1_every_historical_identical_config_control_run_passes():
    am = qg.load_amendment()
    for b in qg.BENCHMARKS:
        for rid, path, _ in audit.RUNS[b]:
            r = qg.evaluate(b, (ROOT / path).read_text(encoding="utf-8", errors="replace"), am)
            assert r["passed"], (b, rid, r)


def test_2_hotpotqa_index1_dash_flip_runs_pass():
    am = qg.load_amendment()
    for rid in ("exp9_attempt1", "exp10_attempt1"):
        path = next(p for r, p, _ in audit.RUNS["hotpotqa"] if r == rid)
        r = qg.evaluate("hotpotqa", (ROOT / path).read_text(encoding="utf-8", errors="replace"), am)
        c = r["controls"]["rabit2"]
        assert r["passed"] and c["score_mismatch_indices"] == [1] and c["l1_score_distance"] == 0.5
        assert c["mismatch_details"][0]["run_answer"] == "2016–17"
        assert c["mismatch_details"][0]["reference_answer"] == "2016-17"


def test_3_qasper_historical_control_variants_pass():
    am = qg.load_amendment()
    seen = set()
    for rid, path, _ in audit.RUNS["qasper"]:
        r = qg.evaluate("qasper", (ROOT / path).read_text(encoding="utf-8", errors="replace"), am)
        assert r["passed"], rid
        seen |= set(r["controls"]["rabit2"]["score_mismatch_indices"])
    assert seen == {18, 20, 21}


def test_4_exceeding_mismatch_count_fails():
    am = qg.load_amendment()
    ref = _ref_text("hotpotqa")
    rows = qg.parse_text(ref)["per_example"]["rabit2"]
    # two tiny changes (L1 0.002 << 0.5) but count 2 > 1
    text = _mutate(ref, "rabit2", {i: (rows[i]["score"] + (0.001 if rows[i]["score"] < 1 else -0.001), None)
                                   for i in (5, 9)})
    c = qg.evaluate("hotpotqa", text, am, reference_text=ref)["controls"]["rabit2"]
    assert c["score_mismatch_count"] == 2 and c["l1_within_bound"] and not c["mismatch_count_within_bound"]
    assert not c["passed"]


def test_5_within_count_but_large_l1_fails():
    am = qg.load_amendment()
    ref = _ref_text("qasper")
    rows = qg.parse_text(ref)["per_example"]["rabit2"]
    i = next(k for k, r in enumerate(rows) if r["score"] == 0.0)
    text = _mutate(ref, "rabit2", {i: (0.2, None)})  # 1 mismatch (<= 2) but L1 0.200 > 0.185
    c = qg.evaluate("qasper", text, am, reference_text=ref)["controls"]["rabit2"]
    assert c["mismatch_count_within_bound"] and not c["l1_within_bound"] and not c["passed"]
    text = _mutate(ref, "rabit2", {i: (0.185, None)})  # exactly at the bound: passes (no slack, no rounding error)
    assert qg.evaluate("qasper", text, am, reference_text=ref)["controls"]["rabit2"]["passed"]


def test_6_text_only_answer_changes_do_not_count():
    am = qg.load_amendment()
    ref = _ref_text("hotpotqa")
    text = _mutate(ref, "rabit2", {0: (None, "a different answer string"), 3: (None, "another")})
    c = qg.evaluate("hotpotqa", text, am, reference_text=ref)["controls"]["rabit2"]
    assert c["passed"] and c["score_mismatch_count"] == 0 and c["text_only_answer_differences"] == [0, 3]


def test_7_treatment_rows_are_never_read():
    a = json.loads(audit.OUT.read_text(encoding="utf-8"))
    for b in a["benchmarks"].values():
        assert set(b["controls"]) == {"bf16", "rabit2"}
    # both parsers keep a row only if its method is one of exactly these two control rows
    assert audit.CONTROL_METHODS == qg.CONTROLS == ("bf16", "rabit2")
    for src in ((HERE / "audit_qa_control_reproducibility.py").read_text(encoding="utf-8"),
                (HERE / "qa_control_gate.py").read_text(encoding="utf-8")):
        assert re.search(r"r\.group\(1\) in (CONTROL_METHODS|CONTROLS)", src)
    # injecting junk treatment rows changes nothing that either parser (and hence the derivation) reads
    ref = _ref_text("hotpotqa")
    junk = "\n".join(ln + "".join(f"\n  {t} score=9.999 KV=1.000 MB answer='junk'" for t in TREATMENT_ROWS)
                     if ln.startswith("Ground truth:") else ln for ln in ref.splitlines())
    assert qg.parse_text(junk) == qg.parse_text(ref)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p_ref, p_junk = Path(d) / "ref.log", Path(d) / "junk.log"
        p_ref.write_text(ref, encoding="utf-8")
        p_junk.write_text(junk, encoding="utf-8")
        assert audit.parse_run(p_junk)["per_example"] == audit.parse_run(p_ref)["per_example"]
    assert qg.derive_thresholds(a) == qg.load_amendment()["thresholds"]


def test_8_audit_and_amendment_regenerate_identically():
    assert json.loads(audit.OUT.read_text(encoding="utf-8")) == json.loads(json.dumps(audit.audit()))
    qg.load_amendment()
    assert qg.audit.sha256_lf(qg.AMENDMENT) == a2.AMENDMENT_SHA256_LF
    try:
        qg.main(["--write"])
    except SystemExit as e:
        assert "never overwritten" in str(e)
    else:
        raise AssertionError("amendment overwritten")


def test_9_exp10_attempt1_stays_archived_and_invalid():
    assert e1.run_git("diff", "--name-only", a2.ATTEMPT1_ARCHIVE_COMMIT, "--",
                      a2.ATTEMPT1_DIR.relative_to(ROOT).as_posix()) == ""
    rec = json.loads((a2.ATTEMPT1_DIR / "attempt_record.json").read_text(encoding="utf-8"))
    assert rec["status"] == "invalid_control_reproduction" and rec["excluded_from_accepted_results"]
    am = qg.load_amendment()
    assert "remains invalid" in am["status_of_prior_attempts"]["exp10_attempt1"]
    assert "remains invalid" in am["status_of_prior_attempts"]["exp9_attempt1"]


def test_10_prior_accepted_evidence_unchanged():
    frozen = {"0f5f6efa7cfdf0b7add27521270e91246ec4c191": ["results/mlsys2027/concurrency_scaling"],
              "599d059cc3cad96f8cdf3c4f813f5460e5b35654": [
                  "results/quality", "results/mlsys2027/quality_frontier", "results/mlsys2027/multilingual_frontier",
                  "results/mlsys2027/ablations/k_bit", "results/mlsys2027/ablations/v_bit",
                  "results/mlsys2027/ablations/group_size", "benchmarks/quality", "benchmarks/mlsys2027/exp9_group"]}
    for commit, paths in frozen.items():
        assert e1.run_git("diff", "--name-only", commit, "--", *paths) == "", commit
    assert e1.run_git("diff", "--name-only", "ad5f262face49af73d0a37efd7899edf5162f57a", "--",
                      "benchmarks/mlsys2027/exp10_residual_protocol.json",
                      "benchmarks/mlsys2027/run_experiment10_residual_ablation.py",
                      "benchmarks/mlsys2027/exp10_residual") == ""  # original Exp10 harness untouched


def test_runner_overlays_only_the_qa_aggregate_f1_check(monkeypatch=None):
    am = qg.load_amendment()
    ref = _ref_text("qasper")
    fake = {"benchmark": "qasper", "rows": {}, "passed": False,
            "checks": {"exit_code_zero": True, "no_traceback": True, "all_five_rows_present": True,
                       "counts_exact": True, "kv_mb_matches_expected": True, "kv_mb_ordering_matches_formula": True,
                       "bf16_and_r4_control_reproduce_canonical": False},
            "regression": {"checks": [{"metric": "bf16.f1_pct", "within_tolerance": True},
                                      {"metric": "bf16.avg_kv_mb", "within_tolerance": True},
                                      {"metric": "rabit2.f1_pct", "within_tolerance": False},
                                      {"metric": "rabit2.avg_kv_mb", "within_tolerance": True}]}}
    orig = a2.r10.integrity
    try:
        a2.r10.integrity = lambda *a, **k: json.loads(json.dumps(fake))
        r = a2.integrity("qasper", 0, ref, {}, am)
        assert r["passed"] and "bf16_and_r4_control_reproduce_canonical" not in r["checks"]
        assert r["checks"]["qa_control_gate_passed"] and r["checks"]["bf16_and_r4_logical_kv_mb_reproduce_canonical"]
        fake["regression"]["checks"][3]["within_tolerance"] = False  # KV MB reproduction still enforced
        assert not a2.integrity("qasper", 0, ref, {}, am)["passed"]
        fake["regression"]["checks"][3]["within_tolerance"] = True
        fake["checks"]["kv_mb_matches_expected"] = False  # original storage gate still enforced
        assert not a2.integrity("qasper", 0, ref, {}, am)["passed"]
        # non-QA benchmarks keep the original rule untouched
        r = a2.integrity("niah", 0, "", {}, am)
        assert r["checks"] == fake["checks"] and r["control_rule"] == "original Exp10 control-reproduction rule"
    finally:
        a2.r10.integrity = orig


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
