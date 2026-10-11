"""Offline tests for the Experiment 10 residual-window ablation harness (no GPU, no torch). Run directly or with pytest."""

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
import exp9_group_scripts as g9  # noqa: E402
import exp10_residual_scripts as gen  # noqa: E402
import run_experiment1_quality_frontier as e1  # noqa: E402
import run_experiment10_residual_ablation as r10  # noqa: E402

QUANT_FUNCTIONS = ("q_group_sym", "q_group_affine", "q_seq_affine", "q_tensor", "q_with_residual", "dequantize_state",
                   "stored_state_logical_bytes", "encode_metadata", "decode_metadata", "metadata_bytes",
                   "tensor_bytes", "bf16_cache_bytes", "quantize_then_dequantize_cache")
RESIDUAL_RE = re.compile(r"""['"]residual['"]""")


def _functions(text: str) -> dict:
    return {n.name: n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef)}


def _canonical_rabit2(b: str) -> dict:
    text = gen.canonical_path(b).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "config_for_method")
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<c>", "exec"), ns)  # noqa: S102
    return ns["config_for_method"]("rabit2")


def test_derived_scripts_are_exact_reversible_derivations():
    assert all(gen.check().values())
    for b in gen.BENCHMARKS:
        canon = gen.canonical_path(b).read_text(encoding="utf-8")
        derived = gen.derived_path(b).read_text(encoding="utf-8")
        assert derived == gen.derive(b, canon)
        body = derived.split("\n", 1)[1]
        for new, old in ((gen.R_BRANCH, ""), (gen.NEW_ALLOWED, gen.OLD_ALLOWED), (gen.NEW_USE, gen.OLD_USE)):
            assert body.count(new) == 1
            body = body.replace(new, old)
        body = g7.APP_RE.sub(f'app = modal.App("{g7.APP_RE.search(canon).group(1)}")', body)
        assert body == canon, b


def test_conditions_differ_only_in_residual_and_everything_else_frozen():
    for b in gen.BENCHMARKS:
        cfg = r10.derived_configs(b)
        control = cfg["rabit2"]
        assert control == _canonical_rabit2(b)  # control is exactly the canonical rabit2
        for method, r in r10.CONDITIONS.items():
            c = cfg[method]
            assert c["residual"] == r
            assert c["k_bits"] == 3 and c["v_bits"] == 2 and c["k_group"] == c["v_group"] == 32
            assert c["k_style"] == "seq_affine" and c["v_style"] == "group_affine"
            assert c["metadata_mode"] == "int8" and c["metadata_group_size"] == 64
            assert {k: c[k] for k in r10.FROZEN} == r10.FROZEN
            diff = {k for k in set(c) | set(control) if c.get(k) != control.get(k)}
            assert diff == (set() if method == "rabit2" else {"residual", "name"}), (b, method, diff)


def test_canonical_evaluation_code_preserved_and_residual_read_only_by_q_with_residual():
    ref = None
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
        readers = {n for n, node in canon.items() if n not in enclosing and n != "config_for_method"
                   and RESIDUAL_RE.search(ast.unparse(node))}
        assert readers == {"q_with_residual"}, (b, readers)
        # the residual split is the same code in every benchmark: newest min(R, T) positions BF16, R <= 0 quantizes all
        dump = ast.dump(canon["q_with_residual"])
        assert ref is None or dump == ref, b
        ref = dump
        src = ast.unparse(canon["q_with_residual"])
        assert "tensor[..., :-residual, :]" in src and "tensor[..., -residual:, :]" in src
        assert "if residual <= 0:" in src and "if tensor.shape[-2] <= residual:" in src
        qdq = ast.unparse(canon["quantize_then_dequantize_cache"])
        assert qdq.count("q_with_residual(") == 2  # K and V, separately, same config


def _meta(n: int, g: int = 64) -> int:
    return math.ceil(n / g) * g + 4 * math.ceil(n / g)


def test_storage_model_hand_computed_for_every_residual():
    cfgs = r10.derived_configs("hotpotqa")
    for method, r in r10.CONDITIONS.items():
        for T in (1, 2, 4, 8, 9, 36, 1024, 4095, 8191, 11209, 16383):
            got = r10.traced_logical_bytes(T, cfgs[method])
            if 0 < T <= r:
                assert got["total"] == 32 * 2 * T * 8 * 128 * 2
                continue
            lq = T - r
            lk = math.ceil(lq / 32) * 32
            hand = {"k_payload": 32 * (8 * lk * 128 * 3 // 8), "k_meta": 32 * 2 * _meta(8 * (lk // 32) * 128),
                    "v_payload": 32 * (8 * lq * 128 * 2 // 8), "v_meta": 32 * 2 * _meta(8 * lq * 4),
                    "residual": 32 * 2 * r * 8 * 128 * 2}
            assert {k: got[k] for k in hand} == hand, (method, T)
            assert got["total"] == r10.r7.logical_kv_bytes(T, 3, 2, r, 32)["total"]  # Exp7 frozen formula


def test_accounting_changes_only_through_residual_dependent_terms():
    cfgs = r10.derived_configs("qasper")
    for T in (1024, 4095, 10546, 11209, 16191):
        parts = {m: r10.traced_logical_bytes(T, cfgs[m]) for m in r10.CONDITIONS}
        for m, r in r10.CONDITIONS.items():
            assert parts[m]["residual"] == r * 131072  # BF16 residual: exactly 0.125 MiB per token (K+V, 32 layers)
            lq = T - r
            assert parts[m]["v_payload"] == 32 * 256 * lq  # V payload linear in the quantized length
            assert parts[m]["k_payload"] == 32 * 384 * math.ceil(lq / 32) * 32  # K moves only across 32-token groups
    p = r10.load_protocol()["logical_storage_expectations"]
    assert p["bf16_residual_contribution"]["mib_per_residual_token_all_layers"] == 0.125
    # not linear in R: the net per-token step differs between benchmarks because of K sequence padding
    d = p["expected_delta_vs_r4_mb"]
    assert d["hotpotqa"]["rabit2_r8"] != d["niah"]["rabit2_r8"]


def test_expected_mb_match_storage_model_anchors_and_canonical_control():
    p = r10.load_protocol()
    L = p["logical_storage_expectations"]
    for b, per in L["model_anchors"]["values"].items():
        for m, v in per.items():
            assert v["model_mb"] == v["exp1_observed_mb"], (b, m)  # includes R0 (rabit8, rabit4) and R2 (rabit3)
    cfgs = r10.derived_configs("niah")
    tgt = p["control_reproduction"]["canonical_targets"]
    for b in gen.BENCHMARKS:
        exp = L["expected_avg_logical_kv_mb"][b]
        prefixes = r10.r7.quantized_prefixes(b)
        for m in r10.CONDITIONS:
            model = round(sum(r10.traced_logical_bytes(t, cfgs[m])["total"] for t in prefixes) / len(prefixes) / 2**20, 3)
            assert exp[m] == model, (b, m)
        assert exp["rabit2"] == tgt[b]["rabit2_R4"]["avg_logical_kv_mb"]
        assert exp["bf16_reference"] == tgt[b]["bf16"]["avg_logical_kv_mb"]
        assert L["gates"]["ordering"]["expected_order_smallest_first"][b] == \
            ["rabit2_r0", "rabit2_r2", "rabit2", "rabit2_r8"]
        assert p["benchmarks"][b]["min_quantized_prefix_tokens"] > 8  # T <= R never occurs


def test_storage_gate_uses_full_precision_from_integer_bytes():
    p = r10.load_protocol()
    L = p["logical_storage_expectations"]
    cfgs = r10.derived_configs("hotpotqa")
    for b in gen.BENCHMARKS:
        prefixes = r10.r7.quantized_prefixes(b)
        for m in r10.CONDITIONS:
            per = [r10.traced_logical_bytes(t, cfgs[m])["total"] for t in prefixes]
            assert all(isinstance(x, int) for x in per)  # exact integer byte counts
            assert L["expected_total_logical_bytes"][b][m] == sum(per)
            full = L["expected_avg_logical_kv_mb_full_precision"][b][m]
            assert full == sum(per) / len(per) / 2**20  # unrounded
            assert round(full, 3) == L["expected_avg_logical_kv_mb"][b][m]  # display values unchanged
    # the gate reads the full-precision value, not the 3-decimal display value
    exp = dict(L["expected_avg_logical_kv_mb_full_precision"]["continuation_ppl"])
    shifted = {**p, "logical_storage_expectations": {**L, "expected_avg_logical_kv_mb_full_precision": {
        **L["expected_avg_logical_kv_mb_full_precision"], "continuation_ppl": {**exp, "rabit2_r2": exp["rabit2_r2"] + 0.0015}}}}
    assert r10.integrity("continuation_ppl", 0, _log(), p)["checks"]["kv_mb_matches_expected"]
    assert not r10.integrity("continuation_ppl", 0, _log(), shifted)["checks"]["kv_mb_matches_expected"]
    assert "ACCOUNTING-INTEGRITY" in L["gates"]["matches_expected"]["rule"]


def test_protocol_frozen_complete_and_not_overwritable():
    p = r10.load_protocol()
    assert p["experiment"] == 10 and "NOT a physical serving benchmark" in p["type"]
    assert "R4 is not presupposed best" in p["question"]
    assert p["methods_argument"] == "bf16,rabit2_r0,rabit2_r2,rabit2,rabit2_r8"
    proof = p["only_residual_differs_proof"]
    assert proof["holds"] and proof["fields_differing_from_control_excluding_display_name"] == {
        "rabit2_r0": ["residual"], "rabit2_r2": ["residual"], "rabit2": [], "rabit2_r8": ["residual"]}
    tol = p["control_reproduction"]["tolerances"]
    assert tol["continuation_ppl.ppl"] == {"relative": 0.005} and tol["avg_logical_kv_mb"] == {"relative": 0.001}
    assert all(tol[k] == {"absolute_points": 1.0} for k in ("niah.accuracy_pct", "passage_retrieval.accuracy_pct",
                                                            "hotpotqa.f1_pct", "qasper.f1_pct"))
    assert p["logical_storage_expectations"]["gates"]["matches_expected"]["absolute_mb"] == 0.001
    for b in gen.BENCHMARKS:
        assert p["benchmarks"][b]["expected_count_per_method"] == r10.COUNT_COLUMNS[b][1]
    assert "NOT physical allocator capacity" in p["logical_storage_expectations"]["label"]
    assert "NOT linear" in p["logical_storage_expectations"]["r_dependence"]
    try:
        r10.main(["--write-protocol"])
    except SystemExit as e:
        assert "never overwritten" in str(e)
    else:
        raise AssertionError("protocol overwritten")


def test_runs_identical_to_exp1_except_methods_and_script():
    for mine, theirs in zip(r10.runs(), e1.RUNS):
        a, b = list(mine["args"]), list(theirs["args"])
        i = a.index("--methods")
        assert a[i + 1] == "bf16,rabit2_r0,rabit2_r2,rabit2,rabit2_r8"
        assert a[:i + 1] + a[i + 2:] == b[:i + 1] + b[i + 2:]
        assert mine["script"] == gen.DERIVED_DIR / theirs["script"]


def _log(r2_kv="24.480", r8_kv="25.171", count=1024):
    return "\n".join([
        f"bf16        8.5020      0.00        9.3659              128.000       1.000         {count}",
        f"rabit2_r0   8.7000      2.33        9.6000              24.250        5.278         {count}",
        f"rabit2_r2   8.6500      1.74        9.5600              {r2_kv}        5.229         {count}",
        f"rabit2      8.6317      1.53        9.5462              24.710        5.180         {count}",
        f"rabit2_r8   8.6200      1.39        9.5300              {r8_kv}        5.085         {count}"])


def test_integrity_gates():
    good = r10.integrity("continuation_ppl", 0, _log())
    assert good["passed"], good["checks"]
    assert not r10.integrity("continuation_ppl", 0, _log(r2_kv="24.483"))["checks"]["kv_mb_matches_expected"]
    swapped = r10.integrity("continuation_ppl", 0, _log(r2_kv="25.171", r8_kv="24.480"))["checks"]
    assert not swapped["kv_mb_ordering_matches_formula"] and not swapped["kv_mb_matches_expected"]
    assert not r10.integrity("continuation_ppl", 0, _log(count=1000))["checks"]["counts_exact"]
    drift = r10.integrity("continuation_ppl", 0, _log().replace("rabit2      8.6317", "rabit2      8.9000"))
    assert not drift["checks"]["bf16_and_r4_control_reproduce_canonical"]
    assert not r10.integrity("continuation_ppl", 1, _log())["passed"]
    assert not r10.integrity("continuation_ppl", 0, "\n".join(_log().splitlines()[:-1]))["passed"]  # missing R8 row


def test_prior_evidence_unchanged_and_exp7_to_exp9_reproducible():
    assert e1.run_git("diff", "--name-only", r10.EXP6_FROZEN_COMMIT, "--", "results/mlsys2027/concurrency_scaling") == ""
    for commit, files in ((r10.EXP7_EVIDENCE_COMMIT, r10.r8.EXP7_FILES), (r10.EXP8_EVIDENCE_COMMIT, r10.r9.EXP8_FILES),
                          (r10.EXP9_EVIDENCE_COMMIT, r10.EXP9_FILES)):
        assert e1.run_git("diff", "--name-only", commit, "--", *[str(p.relative_to(e1.ROOT)) for p in files]) == ""
    for b in gen.BENCHMARKS:
        path = gen.canonical_path(b)
        assert e1.run_git("show", f"HEAD:{path.relative_to(e1.ROOT).as_posix()}") == \
            path.read_text(encoding="utf-8").rstrip("\n")
    assert all(g7.check().values()) and all(g8.check().values()) and all(g9.check().values())


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
