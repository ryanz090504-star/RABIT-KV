"""Offline tests for the Experiment 9 group-size ablation harness (no GPU, no torch). Run directly or with pytest."""

from __future__ import annotations

import ast
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp7_kbit_scripts as g7  # noqa: E402
import exp8_vbit_scripts as g8  # noqa: E402
import exp9_group_scripts as gen  # noqa: E402
import run_experiment1_quality_frontier as e1  # noqa: E402
import run_experiment9_group_ablation as r9  # noqa: E402

QUANT_FUNCTIONS = ("q_group_sym", "q_group_affine", "q_seq_affine", "q_tensor", "q_with_residual", "dequantize_state",
                   "stored_state_logical_bytes", "encode_metadata", "decode_metadata", "metadata_bytes",
                   "tensor_bytes", "bf16_cache_bytes", "quantize_then_dequantize_cache")
GROUP_FIELD_RE = re.compile(r"\b[kv]_group\b|\{side\}_group")


def _functions(text: str) -> dict:
    return {n.name: n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef)}


def test_derived_scripts_are_exact_reversible_derivations():
    assert all(gen.check().values())
    for b in gen.BENCHMARKS:
        canon = gen.canonical_path(b).read_text(encoding="utf-8")
        derived = gen.derived_path(b).read_text(encoding="utf-8")
        assert derived == gen.derive(b, canon)
        body = derived.split("\n", 1)[1]
        for new, old in ((gen.G_BRANCH, ""), (gen.NEW_ALLOWED, gen.OLD_ALLOWED), (gen.NEW_USE, gen.OLD_USE)):
            assert body.count(new) == 1
            body = body.replace(new, old)
        body = g7.APP_RE.sub(f'app = modal.App("{g7.APP_RE.search(canon).group(1)}")', body)
        assert body == canon, b


def test_conditions_differ_only_in_group_size_and_everything_else_frozen():
    for b in gen.BENCHMARKS:
        cfg = r9.derived_configs(b)
        control = cfg["rabit2"]
        assert control == r9.canonical_configs(b, ["rabit2"])["rabit2"]  # control is exactly the canonical rabit2
        for method, g in r9.CONDITIONS.items():
            c = cfg[method]
            assert c["k_group"] == c["v_group"] == g
            assert c["k_bits"] == 3 and c["v_bits"] == 2 and c["residual"] == 4
            assert c["metadata_mode"] == "int8" and c["metadata_group_size"] == 64
            assert {k: c[k] for k in r9.FROZEN} == r9.FROZEN
            diff = {k for k in set(c) | set(control) if c.get(k) != control.get(k)}
            assert diff == (set() if method == "rabit2" else {"k_group", "v_group", "name"}), (b, method, diff)


def test_canonical_evaluation_code_preserved():
    for b in gen.BENCHMARKS:
        canon = _functions(gen.canonical_path(b).read_text(encoding="utf-8"))
        derived = _functions(gen.derived_path(b).read_text(encoding="utf-8"))
        assert set(derived) == set(canon)
        for fn in QUANT_FUNCTIONS:
            if fn in canon:
                assert ast.dump(canon[fn]) == ast.dump(derived[fn]), (b, fn)
        enclosing = {n for n, node in canon.items()
                     if any(isinstance(x, ast.FunctionDef) and x.name == "config_for_method" for x in ast.walk(node))
                     or gen.OLD_ALLOWED in ast.unparse(node)}
        changed = {n for n in canon if ast.dump(canon[n]) != ast.dump(derived[n])}
        assert changed <= enclosing | {"config_for_method"}, (b, changed - enclosing)
        # the group-size fields are read only by q_tensor
        readers = {n for n, node in canon.items() if n not in enclosing and n != "config_for_method"
                   and GROUP_FIELD_RE.search(ast.unparse(node))}
        assert readers == {"q_tensor"}, (b, readers)


def _meta_reference(n: int, g: int = 64) -> int:
    return math.ceil(n / g) * g + 4 * math.ceil(n / g)


def test_storage_model_hand_computed_and_matches_exp7_formula():
    ctrl = r9.derived_configs("hotpotqa")
    for method, g in r9.CONDITIONS.items():
        cfg = ctrl[method]
        for T in (1, 4, 5, 36, 1024, 4095, 8191, 11209, 16191, 16383):
            got = r9.traced_logical_bytes(T, cfg)
            assert got["total"] == r9.r7.logical_kv_bytes(T, 3, 2, 4, g)["total"], (method, T)  # Exp7 frozen formula
            if T <= 4:
                assert got["total"] == 32 * 2 * T * 8 * 128 * 2
                continue
            lq = T - 4
            lk = math.ceil(lq / g) * g
            hand = {"k_payload": 32 * (8 * lk * 128 * 3 // 8),
                    "k_meta": 32 * 2 * _meta_reference(8 * (lk // g) * 128),
                    "v_payload": 32 * (8 * lq * 128 * 2 // 8),
                    "v_meta": 32 * 2 * _meta_reference(8 * lq * (128 // g)),
                    "residual": 32 * 2 * 4 * 8 * 128 * 2}
            assert {k: got[k] for k in hand} == hand, (method, T)


def test_metadata_overhead_accounted_and_only_group_dependent_terms_move():
    cfgs = r9.derived_configs("qasper")
    for T in (1024, 4095, 11209, 16191):
        parts = {m: r9.traced_logical_bytes(T, cfgs[m]) for m in r9.CONDITIONS}
        assert len({p["v_payload"] for p in parts.values()}) == 1  # V payload independent of G
        assert len({p["residual"] for p in parts.values()}) == 1
        # metadata strictly decreases with G (~1/G); halving G roughly doubles it
        assert parts["rabit2_g16"]["k_meta"] > parts["rabit2"]["k_meta"] > parts["rabit2_g64"]["k_meta"] > 0
        assert parts["rabit2_g16"]["v_meta"] > parts["rabit2"]["v_meta"] > parts["rabit2_g64"]["v_meta"] > 0
        # K payload moves only through sequence padding: ceil(Lq/G)*G
        for m, g in r9.CONDITIONS.items():
            assert parts[m]["k_payload"] == 32 * 8 * math.ceil((T - 4) / g) * g * 128 * 3 // 8
    # continuation_ppl: Lq = 1020 pads to 1024 for every G -> identical K payload
    p = r9.load_protocol()["logical_storage_expectations"]["expected_breakdown_avg_mb"]["continuation_ppl"]
    assert len({v["k_payload"] for v in p.values()}) == 1


def test_storage_model_reproduces_observed_exp1_presets_and_canonical_control():
    p = r9.load_protocol()
    L = p["logical_storage_expectations"]
    for b, per in L["model_anchors"]["values"].items():
        for m, v in per.items():
            assert v["model_mb"] == v["exp1_observed_mb"], (b, m)  # G128 / G32, R0 / R2 / R4, META g64 / g256
    tgt = p["control_reproduction"]["canonical_targets"]
    for b in gen.BENCHMARKS:
        exp = L["expected_avg_logical_kv_mb"][b]
        assert exp["rabit2"] == tgt[b]["rabit2_G32"]["avg_logical_kv_mb"]
        assert exp["bf16_reference"] == tgt[b]["bf16"]["avg_logical_kv_mb"]
        assert L["gates"]["ordering"]["expected_order_largest_first"][b] == sorted(r9.CONDITIONS, key=lambda m: -exp[m])


def test_protocol_frozen_complete_and_not_overwritable():
    p = r9.load_protocol()
    assert p["experiment"] == 9 and "NOT a physical serving benchmark" in p["type"]
    assert "G32 is not presupposed best" in p["question"]
    assert p["methods_argument"] == "bf16,rabit2_g16,rabit2,rabit2_g64"
    proof = p["only_group_size_differs_proof"]
    assert proof["holds"] and proof["fields_differing_from_control_excluding_display_name"] == {
        "rabit2_g16": ["k_group", "v_group"], "rabit2": [], "rabit2_g64": ["k_group", "v_group"]}
    tol = p["control_reproduction"]["tolerances"]
    assert tol["continuation_ppl.ppl"] == {"relative": 0.005} and tol["avg_logical_kv_mb"] == {"relative": 0.001}
    assert all(tol[k] == {"absolute_points": 1.0} for k in ("niah.accuracy_pct", "passage_retrieval.accuracy_pct",
                                                            "hotpotqa.f1_pct", "qasper.f1_pct"))
    for b in gen.BENCHMARKS:
        assert p["benchmarks"][b]["expected_count_per_method"] == r9.COUNT_COLUMNS[b][1]
    assert "NOT physical allocator capacity" in p["logical_storage_expectations"]["label"]
    assert "NOT linear" in p["logical_storage_expectations"]["group_size_dependence"]
    try:
        r9.main(["--write-protocol"])
    except SystemExit as e:
        assert "never overwritten" in str(e)
    else:
        raise AssertionError("protocol overwritten")


def test_runs_identical_to_exp1_except_methods_and_script():
    for mine, theirs in zip(r9.runs(), e1.RUNS):
        a, b = list(mine["args"]), list(theirs["args"])
        i = a.index("--methods")
        assert a[i + 1] == "bf16,rabit2_g16,rabit2,rabit2_g64"
        assert a[:i + 1] + a[i + 2:] == b[:i + 1] + b[i + 2:]
        assert mine["script"] == gen.DERIVED_DIR / theirs["script"]


def _log(g16_kv="28.952", g64_kv="22.590", count=1024):
    return "\n".join([
        f"bf16        8.5020      0.00        9.3659              128.000       1.000         {count}",
        f"rabit2_g16  8.6000      1.15        9.5000              {g16_kv}        4.421         {count}",
        f"rabit2      8.6317      1.53        9.5462              24.710        5.180         {count}",
        f"rabit2_g64  8.7000      2.33        9.6000              {g64_kv}        5.666         {count}"])


def test_integrity_gates():
    good = r9.integrity("continuation_ppl", 0, _log())
    assert good["passed"], good["checks"]
    assert not r9.integrity("continuation_ppl", 0, _log(g16_kv="29.100"))["checks"]["kv_mb_matches_expected"]
    swapped = r9.integrity("continuation_ppl", 0, _log(g16_kv="22.590", g64_kv="28.952"))["checks"]
    assert not swapped["kv_mb_ordering_matches_formula"]
    assert not r9.integrity("continuation_ppl", 0, _log(count=1000))["checks"]["counts_exact"]
    drift = r9.integrity("continuation_ppl", 0, _log().replace("rabit2      8.6317", "rabit2      8.9000"))
    assert not drift["checks"]["bf16_and_g32_control_reproduce_canonical"]
    assert not r9.integrity("continuation_ppl", 1, _log())["passed"]


def test_prior_evidence_unchanged_and_exp7_exp8_reproducible():
    assert e1.run_git("diff", "--name-only", r9.EXP6_FROZEN_COMMIT, "--", "results/mlsys2027/concurrency_scaling") == ""
    for commit, files in ((r9.EXP7_EVIDENCE_COMMIT, r9.r8.EXP7_FILES), (r9.EXP8_EVIDENCE_COMMIT, r9.EXP8_FILES)):
        assert e1.run_git("diff", "--name-only", commit, "--", *[str(p.relative_to(e1.ROOT)) for p in files]) == ""
    for b in gen.BENCHMARKS:
        path = gen.canonical_path(b)
        assert e1.run_git("show", f"HEAD:{path.relative_to(e1.ROOT).as_posix()}") == \
            path.read_text(encoding="utf-8").rstrip("\n")
    assert all(g7.check().values()) and all(g8.check().values())  # Exp7 / Exp8 derived scripts still regenerate


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
