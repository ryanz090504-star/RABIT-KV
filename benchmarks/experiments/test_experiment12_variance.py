"""Offline tests for the Experiment 12 larger-N / paired-bootstrap harness (no GPU, no torch)."""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paired_bootstrap_ci as pb  # noqa: E402
import qa_control_gate as qg  # noqa: E402
import run_experiment1_quality_frontier as e1  # noqa: E402
import run_experiment12_variance as r12  # noqa: E402

ROOT = e1.ROOT
REF = ROOT / "results" / "quality"


def _ref(b: str) -> str:
    return (REF / f"{b}.log").read_text(encoding="utf-8", errors="replace")


def test_conditions_are_bf16_and_canonical_rabit2_on_unchanged_scripts():
    p = r12.load_protocol()
    fn = next(n for n in ast.walk(ast.parse((ROOT / "benchmarks/quality/qasper.py").read_text(encoding="utf-8")))
              if isinstance(n, ast.FunctionDef) and n.name == "config_for_method")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<c>", "exec"), ns)  # noqa: S102
    cfg = p["conditions"]["rabit2"]["config"]
    assert cfg == ns["config_for_method"]("rabit2")
    assert (cfg["k_bits"], cfg["v_bits"], cfg["k_group"], cfg["v_group"], cfg["residual"], cfg["metadata_mode"],
            cfg["metadata_group_size"]) == (3, 2, 32, 32, 4, "int8", 64)
    assert set(p["conditions"]) == {"bf16", "rabit2"} and r12.METHODS == "bf16,rabit2"
    for r in r12.runs():
        assert r["script"] == ROOT / "benchmarks" / "quality" / f"{r['name']}.py"  # canonical, unchanged scripts
        assert r["args"][-2:] == ["--methods", "bf16,rabit2"]


def test_sample_selection_is_a_deterministic_superset_of_canonical():
    canon = {s["name"]: s["args"] for s in e1.RUNS}
    for r in r12.runs():
        a, c = r["args"], canon[r["name"]]
        drop = lambda args, keys: [x for i, x in enumerate(args) if not (x in keys or (i and args[i - 1] in keys))]  # noqa: E731
        keys = {"--samples", "--needle-depths", "--methods"}
        assert drop(a, keys) == drop(c, keys), r["name"]  # every other argument identical to canonical
    sel = r12.SELECTION
    assert [f"{d:.2f}" for d in (0.1, 0.25, 0.5, 0.75, 0.9)] == [x for x in r12.NIAH_DEPTHS if x in
                                                                  ("0.10", "0.25", "0.50", "0.75", "0.90")]
    assert len(r12.NIAH_DEPTHS) == 19 and len(set(r12.NIAH_DEPTHS)) == 19
    assert sel["niah"]["exp12_units"] == 19 * 3
    assert sel["hotpotqa"]["exp12_units"] == r12.POOL_FACTS["hotpotqa"]["bucket_8k_plus"] == 100
    assert sel["qasper"]["exp12_units"] == sel["qasper"]["canonical_units"] == r12.POOL_FACTS["qasper"]["bucket_8k_plus"]
    assert sel["passage_retrieval"]["exp12_units"] == r12.POOL_FACTS["passage_retrieval"]["rows"] == 200
    assert sel["continuation_ppl"]["exp12_units"] == 4 * sel["continuation_ppl"]["canonical_units"]
    for b in pb.BENCHMARKS:
        assert sel[b]["exp12_units"] >= sel[b]["canonical_units"]
    assert r12.runs() == r12.runs()  # pure / deterministic


def test_extraction_reproduces_canonical_printed_aggregates():
    for b in pb.BENCHMARKS:
        t = _ref(b)
        keys, bf, rb = pb.paired_values(b, pb.extract(b, t))
        _, qi, _ = r12.r7.ROW_COLUMNS[b]
        for m, vals in (("bf16", bf), ("rabit2", rb)):
            printed = float(e1._last_row_tokens(t, m)[qi])
            got = pb.aggregate(b, vals)
            tol = 1e-4 * printed if b == "continuation_ppl" else 0.05 + 1e-9
            assert abs(got - printed) <= tol, (b, m, got, printed)
        assert len(keys) == r12.SELECTION[b]["canonical_units"]


FROZEN_SEEDS = {"continuation_ppl": 20270929, "passage_retrieval": 20270931, "hotpotqa": 20270932, "qasper": 20270933}


def test_exact_per_benchmark_seeds_are_frozen_and_niah_has_none():
    assert pb.BOOTSTRAP["seed_by_benchmark"] == FROZEN_SEEDS
    sp = r12.load_protocol()["statistical_procedure"]
    assert sp["seed_by_benchmark"] == FROZEN_SEEDS and "niah" not in sp["seed_by_benchmark"]
    assert sp["resamples"] == 10000 and sp["confidence"] == 0.95
    assert sp["inferential_benchmarks"] == ["continuation_ppl", "passage_retrieval", "hotpotqa", "qasper"]
    assert sp["niah"]["inferential_statistics"] is None
    assert "not independently sampled documents" in sp["ci_description"]["continuation_ppl"]
    assert "does NOT increase Qasper sample coverage" in sp["ci_description"]["qasper"]
    assert "base_seed" not in json.dumps(sp)  # concrete integers only


def test_bootstrap_is_reproducible_for_each_inferential_benchmark():
    bf = [0.1 * (i % 7) + 0.05 for i in range(50)]
    rb = [0.1 * ((i * 3) % 7) + 0.05 for i in range(50)]
    for b, seed in FROZEN_SEEDS.items():
        x, y = ([math.log(1 + v) for v in bf], [math.log(1 + v) for v in rb]) if b == "continuation_ppl" else (bf, rb)
        a = pb.paired_bootstrap(b, x, y, 2000, seed)
        assert a == pb.paired_bootstrap(b, x, y, 2000, seed)
        assert a != pb.paired_bootstrap(b, x, y, 2000, seed + 1000)
    assert pb.bootstrap_indices(10, 3, 5) == pb.bootstrap_indices(10, 3, 5)
    assert all(0 <= i < 10 for row in pb.bootstrap_indices(10, 50, 1) for i in row)


def _niah_rows(contexts, depths, fail=()):
    rows = {}
    for m in ("bf16", "rabit2"):
        rows[m] = [{"key": [c, d], "value": 0.0 if (m, c, d) in fail else 1.0, "kv_mb": 0.0, "prefix_tokens": c - 1}
                   for c in contexts for d in depths]
    return rows


def test_niah_is_excluded_from_inferential_bootstrap():
    try:
        pb.analyze("niah", _ref("niah"))
    except ValueError as e:
        assert "no inferential bootstrap" in str(e)
    else:
        raise AssertionError("NIAH bootstrap computed")
    assert "niah" not in pb.INFERENTIAL and "niah" not in pb.BOOTSTRAP["seed_by_benchmark"]
    src = (HERE / "paired_bootstrap_ci.py").read_text(encoding="utf-8")
    assert '"benchmarks": {b: analyze(b, read(b)) for b in INFERENTIAL}' in src


def test_niah_deterministic_grid_is_complete_and_descriptive():
    depths = [round(k * 0.05, 2) for k in range(1, 20)]
    assert [f"{d:.2f}" for d in depths] == r12.NIAH_DEPTHS
    g = pb.niah_grid(_niah_rows(r12.NIAH_CONTEXTS, depths), depths=depths)
    assert g["grid_complete"] and g["methods_identical_outcomes"]
    for m in ("bf16", "rabit2"):
        assert (g[m]["passed"], g[m]["cases"], g[m]["accuracy_pct"]) == (57, 57, 100.0)
        assert all(v == {"passed": 19, "cases": 19} for v in g[m]["per_context"].values())
        assert g[m]["failed_coordinates"] == []
    assert "ci_low" not in json.dumps(g) and "bootstrap or pseudo-CI" in g["statistics"]
    g = pb.niah_grid(_niah_rows(r12.NIAH_CONTEXTS, depths, fail={("rabit2", 16384, 0.35)}), depths=depths)
    assert g["rabit2"]["passed"] == 56 and g["rabit2"]["per_context"]["16384"] == {"passed": 18, "cases": 19}
    assert g["rabit2"]["failed_coordinates"] == [[16384, 0.35]] and not g["methods_identical_outcomes"]
    missing = _niah_rows(r12.NIAH_CONTEXTS, depths)
    missing["bf16"].pop()
    missing["rabit2"].pop()
    assert not pb.niah_grid(missing, depths=depths)["grid_complete"]
    canon = pb.niah_grid(pb.extract("niah", _ref("niah")), depths=[0.1, 0.25, 0.5, 0.75, 0.9])
    assert canon["grid_complete"] and canon["bf16"]["cases"] == 15  # parser handles the real log format


def test_worst_k_contributions_are_reported():
    bf = [0.0, 0.5, 1.0, 0.2, 0.3, 0.9]
    rb = [0.0, 0.0, 1.0, 0.4, 0.3, 0.6]
    r = pb.robustness("hotpotqa", list(range(6)), bf, rb)
    assert [u["key"] for u in r["worst1"]["units"]] == [1]
    assert [u["key"] for u in r["worst3"]["units"]] == [1, 5, 0]
    assert math.isclose(r["worst3"]["sum_contribution"], 100 * (-0.5 - 0.3) / 6)


def test_paired_resampling_preserves_example_pairing():
    bf = [((i * 37) % 101) / 101 for i in range(80)]
    rb = [x + 0.05 for x in bf]  # a constant per-example shift: paired replicates must all equal +5 points exactly
    boot = pb.paired_bootstrap("hotpotqa", bf, rb, 500, 7)
    assert math.isclose(boot["ci_low"], 5.0, abs_tol=1e-9) and math.isclose(boot["ci_high"], 5.0, abs_tol=1e-9)
    lbf = [math.log(2 + (i % 9)) for i in range(32)]
    lrb = [x + math.log(1.02) for x in lbf]  # constant 2% per-window PPL ratio
    boot = pb.paired_bootstrap("continuation_ppl", lbf, lrb, 500, 7)
    assert math.isclose(boot["ci_low"], 2.0, abs_tol=1e-9) and math.isclose(boot["ci_high"], 2.0, abs_tol=1e-9)
    rows = {"bf16": [{"key": 1, "value": 1.0}, {"key": 2, "value": 0.0}],
            "rabit2": [{"key": 2, "value": 1.0}, {"key": 1, "value": 0.0}]}
    try:
        pb.paired_values("niah", rows)
    except ValueError:
        pass
    else:
        raise AssertionError("mis-paired units accepted")


def test_robustness_is_descriptive_and_removes_nothing():
    bf = [0.0, 0.5, 1.0, 0.2, 0.3]
    rb = [0.0, 0.0, 1.0, 0.4, 0.3]
    r = pb.robustness("hotpotqa", list(range(5)), bf, rb)
    assert r["n"] == 5 and (r["n_rabit2_better"], r["n_rabit2_worse"], r["n_equal"]) == (1, 1, 3)
    assert r["largest_abs_contribution"]["key"] == 1 and math.isclose(r["largest_abs_contribution"]["contribution"], -10.0)
    agg = pb.delta("hotpotqa", bf, rb)
    assert math.isclose(sum(c["contribution"] for c in r["top3_by_abs"]["units"]), agg)  # only 2 non-zero units here
    assert math.isclose(r["top1_by_abs"]["share_of_aggregate_delta"], -10.0 / agg)
    assert math.isclose(r["aggregate_delta_on_unit_scale"], agg)


def test_qa_subset_gate_matches_frozen_gate_on_historical_runs():
    am = qg.load_amendment()
    for b in ("hotpotqa", "qasper"):
        for rid, path, _ in qg.audit.RUNS[b]:
            text = (ROOT / path).read_text(encoding="utf-8", errors="replace")
            sub = r12._qa_subset_gate(b, text, am)
            assert sub["passed"] == qg.evaluate(b, text, am)["passed"] is True, (b, rid)
    ref = _ref("hotpotqa")
    bad = ref.replace("  rabit2   score=0.000 KV=352.046 MB", "  rabit2   score=0.900 KV=352.046 MB", 1)
    assert not r12._qa_subset_gate("hotpotqa", bad, am)["passed"]  # L1 0.9 > 0.5


def test_treatment_rows_cannot_modify_validity_thresholds():
    p, am = r12.load_protocol(), qg.load_amendment()
    th = p["validity"]["hotpotqa_qasper"]
    assert th["new_thresholds_derived"] is False and th["exp12_data_can_modify_thresholds"] is False
    for b in ("hotpotqa", "qasper"):
        for m in ("bf16", "rabit2"):
            for k in ("historical_max_score_mismatch_count", "historical_max_l1_score_distance"):
                assert th["thresholds"][b][m][k] == am["thresholds"][b][m][k]
    assert qg.audit.sha256_lf(qg.AMENDMENT) == r12.r11.AMENDMENT_SHA256_LF
    ref = _ref("hotpotqa")
    junk = "\n".join(ln + "\n  rabit2_g16 score=0.123 KV=1.000 MB answer='x'" if ln.startswith("Ground truth:") else ln
                     for ln in ref.splitlines())
    assert r12._qa_subset_gate("hotpotqa", junk, am) == r12._qa_subset_gate("hotpotqa", ref, am)
    assert "Experiment 12" not in json.dumps(am)  # the amendment is not re-derived for / by Exp12


def test_integrity_end_to_end_on_canonical_logs_at_canonical_sizes():
    p, am = r12.load_protocol(), qg.load_amendment()
    saved = (json.loads(json.dumps(r12.SELECTION)), dict(r12.EXPECTED_COUNT))
    try:
        for b in pb.BENCHMARKS:
            r12.SELECTION[b]["exp12_units"] = r12.SELECTION[b]["canonical_units"]
            r12.EXPECTED_COUNT[b] = {"continuation_ppl": 8 * 128}.get(b, r12.SELECTION[b]["canonical_units"])
        for b in pb.BENCHMARKS:
            res = r12.integrity(b, 0, _ref(b), p, am)
            assert res["passed"], (b, res["checks"], res.get("storage_mismatches"))
        corrupt = _ref("passage_retrieval").replace("KV=1401.125 MB", "KV=1401.200 MB", 1)
        assert not r12.integrity("passage_retrieval", 0, corrupt, p, am)["checks"]["per_unit_kv_mb_matches_accounting"]
    finally:
        r12.SELECTION.clear()
        r12.SELECTION.update(saved[0])
        r12.EXPECTED_COUNT.clear()
        r12.EXPECTED_COUNT.update(saved[1])
    assert r12.SELECTION["hotpotqa"]["exp12_units"] == 100


def test_protocol_frozen_complete_and_not_overwritable():
    p = r12.load_protocol()
    assert p["experiment"] == 12 and "NOT a physical serving benchmark" in p["type"]
    assert "does_not_address" in p["purpose_separation"] and p["validity"]["separate_from_statistics"]
    assert p["statistical_procedure"]["no_outlier_removal"] is True
    assert p["expected_count_per_method"] == r12.EXPECTED_COUNT
    try:
        r12.main(["--write-protocol"])
    except SystemExit as e:
        assert "never overwritten" in str(e)
    else:
        raise AssertionError("protocol overwritten")


def test_prior_accepted_evidence_unchanged():
    frozen = {r12.EXP11_EVIDENCE_COMMIT: r12.EXP11_FILES, r12.r11.EXP10_EVIDENCE_COMMIT: r12.r11.EXP10_FILES,
              r12.r11.AMENDMENT_COMMIT: r12.r11.AMENDMENT_FILES,
              "599d059cc3cad96f8cdf3c4f813f5460e5b35654": [ROOT / "benchmarks" / "quality", REF,
                                                           ROOT / "results/mlsys2027/quality_frontier",
                                                           ROOT / "results/mlsys2027/ablations/k_bit",
                                                           ROOT / "results/mlsys2027/ablations/v_bit",
                                                           ROOT / "results/mlsys2027/ablations/group_size"],
              "0f5f6efa7cfdf0b7add27521270e91246ec4c191": [ROOT / "results/mlsys2027/concurrency_scaling"]}
    for commit, paths in frozen.items():
        assert e1.run_git("diff", "--name-only", commit, "--", *[str(x.relative_to(ROOT)) for x in paths]) == "", commit


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
