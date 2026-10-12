"""Offline tests for the Experiment 8 V-bit ablation harness (no GPU, no torch). Run directly or with pytest."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp7_kbit_scripts as g7  # noqa: E402
import exp8_vbit_scripts as gen  # noqa: E402
import run_experiment1_quality_frontier as e1  # noqa: E402
import run_experiment8_vbit_ablation as r8  # noqa: E402

EXP8_HELPERS = ("exp8_state_codes", "exp8_check_codes", "exp8_check_finite")


def _functions(text: str) -> dict:
    return {n.name: n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef)}


def _strip_exp8_calls(fn: ast.FunctionDef) -> str:
    """AST of a function with the two inserted Experiment 8 check calls removed."""
    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "").startswith("exp8_check_"):
                return None
            return node
    return ast.dump(Strip().visit(ast.parse(ast.unparse(fn))))


def test_derived_scripts_are_exact_reversible_derivations():
    assert all(gen.check().values())
    for b in gen.BENCHMARKS:
        canon = gen.canonical_path(b).read_text(encoding="utf-8")
        derived = gen.derived_path(b).read_text(encoding="utf-8")
        assert derived == gen.derive(b, canon)
        body = derived.split("\n", 1)[1]
        for new, old in ((gen.RETURN_CALL + gen.RETURN_ANCHOR, gen.RETURN_ANCHOR),
                         (gen.LOOP_ANCHOR + gen.LOOP_CALL, gen.LOOP_ANCHOR),
                         (gen.CHECK_HELPERS + gen.QDQ_DEF, gen.QDQ_DEF), (gen.V_BRANCH, ""),
                         (gen.NEW_ALLOWED, gen.OLD_ALLOWED), (gen.NEW_USE, gen.OLD_USE)):
            assert body.count(new) == 1
            body = body.replace(new, old)
        body = g7.APP_RE.sub(f'app = modal.App("{g7.APP_RE.search(canon).group(1)}")', body)
        assert body == canon, b


def test_conditions_differ_only_in_v_bits_and_k_stays_3():
    for b in gen.BENCHMARKS:
        cfg = r8.derived_configs(b)
        canon = r7_canonical_config(b)
        control = cfg["rabit2"]
        assert control == canon  # control is exactly the canonical rabit2
        assert {k: control[k] for k in r8.FROZEN} == r8.FROZEN and control["v_bits"] == 2
        for method, bits in {**r8.CONDITIONS, **r8.SUBSTITUTE}.items():
            c = cfg[method]
            assert c["v_bits"] == bits and c["k_bits"] == 3
            assert {k: c[k] for k in r8.FROZEN} == r8.FROZEN  # every grouping / style / residual / metadata field
            diff = {k for k in set(c) | set(control) if c.get(k) != control.get(k)}
            assert diff <= {"v_bits", "name"} and (method == "rabit2" or diff == {"v_bits", "name"}), (b, method, diff)


def r7_canonical_config(b: str) -> dict:
    text = gen.canonical_path(b).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "config_for_method")
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<c>", "exec"), ns)  # noqa: S102
    return ns["config_for_method"]("rabit2")


def test_canonical_evaluation_code_preserved():
    for b in gen.BENCHMARKS:
        canon = _functions(gen.canonical_path(b).read_text(encoding="utf-8"))
        derived = _functions(gen.derived_path(b).read_text(encoding="utf-8"))
        assert set(derived) == set(canon) | set(EXP8_HELPERS)
        for fn in ("q_group_sym", "q_group_affine", "q_seq_affine", "q_tensor", "q_with_residual", "dequantize_state",
                   "stored_state_logical_bytes", "encode_metadata", "decode_metadata", "metadata_bytes"):
            assert ast.dump(canon[fn]) == ast.dump(derived[fn]), (b, fn)
        # quantize_then_dequantize_cache differs ONLY by the two observation-only check calls
        assert _strip_exp8_calls(derived["quantize_then_dequantize_cache"]) == \
            _strip_exp8_calls(canon["quantize_then_dequantize_cache"])
        enclosing = {n for n, node in canon.items()
                     if any(isinstance(x, ast.FunctionDef) and x.name in ("config_for_method", "quantize_then_dequantize_cache")
                            for x in ast.walk(node)) or gen.OLD_ALLOWED in ast.unparse(node)}
        changed = {n for n in canon if ast.dump(canon[n]) != ast.dump(derived[n])}
        assert changed <= enclosing | {"config_for_method", "quantize_then_dequantize_cache"}, (b, changed - enclosing)


class _T:
    """Minimal stand-in for a torch tensor (amin / amax / isfinite)."""

    def __init__(self, lo, hi, finite=True):
        self.lo, self.hi, self.finite = lo, hi, finite

    def amin(self):
        return _S(self.lo)

    def amax(self):
        return _S(self.hi)


class _S:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v

    def all(self):
        return self


class _Torch:
    @staticmethod
    def isfinite(t):
        return _S(t.finite)


def _helpers():
    text = gen.derived_path("hotpotqa").read_text(encoding="utf-8")
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name in EXP8_HELPERS]
    ns = {"torch": _Torch(), "exp8_layer_stats": []}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<exp8>", "exec"), ns)  # noqa: S102
    return ns


def _state(lo, hi, bits):
    return {"type": "split", "old": {"type": "int", "codes": _T(lo, hi), "bits": bits}, "recent": {"type": "bf16"}}


def test_correctness_check_logic(capsys=None):
    ns = _helpers()
    cfg = {"name": "K3V1 META8g64 G32 R4 (V-bit ablation of rabit2)", "k_bits": 3, "v_bits": 1}
    for _ in range(32):
        ns["exp8_check_codes"](cfg, _state(0, 7, 3), _state(0, 1, 1))
    ns["exp8_check_finite"](cfg, [(_T(0, 0), _T(0, 0))] * 32)  # prints the OK line, clears the stats
    assert ns["exp8_layer_stats"] == []
    for bad, kind in ((_state(1, 1, 1), "degenerate_all_codes_identical"), (_state(0, 2, 1), "code_out_of_range")):
        try:
            ns["exp8_check_codes"](cfg, _state(0, 7, 3), bad)
        except RuntimeError as e:
            assert "EXP8_CORRECTNESS_FAILURE" in str(e) and kind in str(e) and "side=v" in str(e)
        else:
            raise AssertionError(kind)
    try:
        ns["exp8_check_finite"](cfg, [(_T(0, 0), _T(0, 0, finite=False))])
    except RuntimeError as e:
        assert "nonfinite_dequantized_kv" in str(e)
    else:
        raise AssertionError("NaN / Inf not detected")
    assert ns["exp8_state_codes"]({"type": "bf16"}) == []  # BF16 residual / short sequences are never checked


def test_storage_moves_only_through_v_payload_and_is_linear():
    for T in (1024, 4095, 11209, 16191):
        parts = {v: r8.r7.logical_kv_bytes(T, 3, v) for v in (1, 2, 3, 4)}
        for key in ("k_payload", "k_meta", "v_meta", "residual"):
            assert len({parts[v][key] for v in parts}) == 1, (T, key)
        steps = [parts[v + 1]["v_payload"] - parts[v]["v_payload"] for v in (1, 2, 3)]
        assert steps[0] == steps[1] == steps[2] > 0
    p = r8.load_protocol()
    exp = p["logical_storage_expectations"]["expected_avg_logical_kv_mb"]
    tgt = p["control_reproduction"]["canonical_targets"]
    for b in gen.BENCHMARKS:
        assert exp[b]["rabit2"] == tgt[b]["rabit2_V2"]["avg_logical_kv_mb"]  # formula reproduces canonical V2
        assert exp[b]["bf16_reference"] == tgt[b]["bf16"]["avg_logical_kv_mb"]
        assert exp[b]["rabit2_v1"] < exp[b]["rabit2"] < exp[b]["rabit2_v3"] < exp[b]["rabit2_v4"]
        assert abs((exp[b]["rabit2_v3"] - exp[b]["rabit2"]) - (exp[b]["rabit2"] - exp[b]["rabit2_v1"])) <= 0.002


def test_protocol_frozen_complete_and_not_overwritable():
    p = r8.load_protocol()
    assert p["experiment"] == 8 and p["axis"] == "V bits" and "NOT a physical serving benchmark" in p["type"]
    assert "V2 is not presupposed best" in p["question"]
    assert p["methods_argument"] == "bf16,rabit2_v1,rabit2,rabit2_v3"
    assert p["only_v_bits_differs_proof"]["holds"]
    assert "quality" in p["pre_registered_substitute"]["rule"] and "not automatic" in p["pre_registered_substitute"]["rule"]
    assert p["correctness_gate"]["expected_ok_lines_per_method"] == r8.QUANTIZE_CALLS_PER_METHOD
    tol = p["control_reproduction"]["tolerances"]
    assert tol["continuation_ppl.ppl"] == {"relative": 0.005} and tol["avg_logical_kv_mb"] == {"relative": 0.001}
    assert all(tol[k] == {"absolute_points": 1.0} for k in ("niah.accuracy_pct", "passage_retrieval.accuracy_pct",
                                                            "hotpotqa.f1_pct", "qasper.f1_pct"))
    for b in gen.BENCHMARKS:
        args = p["benchmarks"][b]["args"]
        if "--samples" in args:
            assert int(args[args.index("--samples") + 1]) == r8.QUANTIZE_CALLS_PER_METHOD[b]
    assert "NOT physical allocator capacity" in p["logical_storage_expectations"]["label"]
    try:
        r8.main(["--write-protocol"])
    except SystemExit as e:
        assert "never overwritten" in str(e)
    else:
        raise AssertionError("protocol overwritten")


def test_runs_identical_to_exp1_except_methods_and_script():
    for mine, theirs in zip(r8.runs(), e1.RUNS):
        a, b = list(mine["args"]), list(theirs["args"])
        i = a.index("--methods")
        assert a[i + 1] == "bf16,rabit2_v1,rabit2,rabit2_v3"
        assert a[:i + 1] + a[i + 2:] == b[:i + 1] + b[i + 2:]
        assert mine["script"] == gen.DERIVED_DIR / theirs["script"]


def _log(v1_kv="20.726", fail=False, ok_lines=8):
    p = r8.load_protocol()
    names = {m: p["conditions"][m]["config"]["name"] for m in r8.CONDITIONS}
    ok = [f"EXP8_CORRECTNESS_OK name={names[m]} layers=32 k_bits=3 v_bits={v} nonfinite=0 min_code_span_k=7 "
          f"min_code_span_v={2 ** v - 1}" for m, v in r8.CONDITIONS.items() for _ in range(ok_lines)]
    rows = ["bf16        8.5020      0.00        9.3659              128.000       1.000         1024",
            f"rabit2_v1   9.4000      10.56       9.9000              {v1_kv}        6.176         1024",
            "rabit2      8.6317      1.53        9.5462              24.710        5.180         1024",
            "rabit2_v3   8.5600      0.68        9.4100              28.695        4.461         1024"]
    extra = [f"RuntimeError: EXP8_CORRECTNESS_FAILURE kind=degenerate_all_codes_identical name={names['rabit2_v1']} "
             "side=v bits=1 code=0"] if fail else []
    return "\n".join(ok + rows + extra)


def test_integrity_gates_and_v1_substitution_classification():
    good = r8.integrity("continuation_ppl", 0, _log())
    assert good["passed"] and good["classification"] == "passed", good["checks"]
    assert good["correctness_ok_lines"] == {"rabit2_v1": 8, "rabit2": 8, "rabit2_v3": 8}
    assert not r8.integrity("continuation_ppl", 0, _log(ok_lines=7))["checks"]["correctness_ok_lines_complete"]
    assert not r8.integrity("continuation_ppl", 0, _log(v1_kv="20.900"))["checks"]["kv_mb_matches_expected"]
    v1 = r8.integrity("continuation_ppl", 1, _log(fail=True))
    assert v1["classification"] == "v1_correctness_failure_substitution_requires_review" and not v1["passed"]
    drift = r8.integrity("continuation_ppl", 0, _log().replace("rabit2      8.6317", "rabit2      8.9000"))
    assert not drift["checks"]["bf16_and_v2_control_reproduce_canonical"]


def test_prior_evidence_unchanged_and_exp7_untouched():
    assert e1.run_git("diff", "--name-only", r8.EXP6_FROZEN_COMMIT, "--", "results/mlsys2027/concurrency_scaling") == ""
    assert e1.run_git("diff", "--name-only", r8.EXP7_EVIDENCE_COMMIT, "--",
                      *[str(p.relative_to(e1.ROOT)) for p in r8.EXP7_FILES]) == ""
    for b in gen.BENCHMARKS:
        path = gen.canonical_path(b)
        assert e1.run_git("show", f"HEAD:{path.relative_to(e1.ROOT).as_posix()}") == \
            path.read_text(encoding="utf-8").rstrip("\n")
    assert all(g7.check().values())  # Experiment 7's derived scripts still match their generator


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
