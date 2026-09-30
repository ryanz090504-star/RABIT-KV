"""Offline tests for the Exp14 shape gate's amended attention criterion (fixed dtype-aware numerical conformance
checks C1 / C2). Pure Python: no torch, no GPU, no Modal. Run directly or with pytest."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp14_shape_gate as sg  # noqa: E402

ROOT = HERE.parents[1]
U = 2.0 ** -8
CAPTURE = ROOT / "results/mlsys2027/second_model/shape_gate_numdiag/attempt_4/numdiag_result.json"
CAPTURE_SHA256 = "10ce1ac0aa7073ce70e7fff747bf792bd81188a7cb6c3019080c70bbc0337f95"
R = [3.0, -1.5, 0.75, 0.02, -0.001, 0.5]  # max|r| = 3.0 -> floor 0.03; 0.02 / -0.001 are below the floor


def test_constants_fixed():
    assert sg.U_BF16 == 2.0 ** -8 and sg.RELATIVE_FLOOR == 0.01
    assert sg.MAX_ABS_TOL == 5e-3  # retained ONLY as a non-gating diagnostic (frozen numerical diagnostic imports it)


def test_1_exact_equality_passes():
    c = sg.conformance(R, R)
    assert c["passed"] and c["C1_ratio"] == 0.0 and c["C2_ratio"] == 0.0


def test_2_valid_bf16_scale_perturbation_passes():
    y = [v * (1 + 0.5 * U) for v in R]  # half a unit roundoff, relative
    c = sg.conformance(y, R)
    assert c["passed"] and c["C1_pass"] and c["C2_pass"]


def test_3_normwise_error_just_above_u_max_fails():
    y = list(R)
    y[4] = R[4] + U * 3.0 * 1.001  # below-floor element; absolute error just above u * max|r|
    c = sg.conformance(y, R)
    assert not c["C1_pass"] and not c["passed"] and c["C2_pass"]


def test_4_above_floor_relative_error_just_above_u_fails():
    y = list(R)
    y[2] = R[2] * (1 + U * 1.001)  # 0.75: above floor; relative error just above u (absolute error well within C1)
    c = sg.conformance(y, R)
    assert c["C1_pass"] and not c["C2_pass"] and not c["passed"]


def test_5_near_zero_large_relative_error_below_floor_does_not_fail_c2():
    y = list(R)
    y[4] = R[4] + 0.002  # -0.001 -> 0.001: 200 % relative error, below the 1 % floor, absolute error < u * max|r|
    c = sg.conformance(y, R)
    assert abs(y[4] - R[4]) / abs(R[4]) > 1.0 and abs(y[4] - R[4]) < U * 3.0
    assert c["C1_pass"] and c["C2_pass"] and c["passed"]


def test_6_all_zero_reference_and_zero_runtime_passes():
    assert sg.conformance([0.0, 0.0, -0.0], [0.0, 0.0, 0.0])["passed"]


def test_7_all_zero_reference_and_nonzero_runtime_fails():
    c = sg.conformance([0.0, 1e-30, 0.0], [0.0, 0.0, 0.0])
    assert not c["passed"] and not c["C1_pass"]


def test_non_finite_or_mismatched_fails():
    assert not sg.conformance([float("nan"), 1.0], [1.0, 1.0])["passed"]
    assert not sg.conformance([1.0], [1.0, 2.0])["passed"]
    assert not sg.conformance([], [])["passed"]


def test_8_same_criterion_for_both_geometries():
    assert set(sg.GEOMETRIES) == {"model_b_qwen2_5_7b", "control_llama3_1_8b"}
    # the criterion takes no geometry argument; replay() calls it identically for every geometry
    assert list(inspect.signature(sg.conformance).parameters) == ["y", "r"]
    src = inspect.getsource(sg.replay)
    assert src.count("conformance(got.flatten().tolist(), ref_out.flatten().tolist())") == 1
    run_src = inspect.getsource(sg.main)
    assert "for name, geom in GEOMETRIES.items():" in run_src
    # no geometry-specific constant anywhere in the decision path
    for fn in (sg.conformance, sg.conformance_decision, sg.checkpoint_failures):
        body = inspect.getsource(fn)
        assert "geom" not in body and "q_heads" not in body and "kv_heads" not in body


def test_9_exact_invariant_failure_fails_even_if_conformance_passes():
    conf = sg.conformance(R, R)
    assert conf["passed"]
    fails = sg.checkpoint_failures(40, ["closed page 1 bytes differ from Rabit2OnlineStateRef"], conf)
    assert fails and "bytes differ" in fails[0]
    fails = sg.checkpoint_failures(40, ["total_tokens runtime=40 ref=39"], conf)
    assert fails
    assert sg.checkpoint_failures(40, [], conf) == []
    # replay() routes exact failures and conformance through checkpoint_failures
    src = inspect.getsource(sg.replay)
    assert 'out["failures"].extend(checkpoint_failures(t, exact, conf))' in src


def test_gating_path_does_not_use_old_threshold_or_rounded_reference():
    for fn in (sg.conformance, sg.conformance_decision, sg.checkpoint_failures):
        body = inspect.getsource(fn)
        assert "MAX_ABS_TOL" not in body and "bfloat16" not in body and "ulp" not in body.lower()
    src = inspect.getsource(sg.replay)
    # the old threshold only feeds the diagnostics counter
    assert 'dg["old_5e-3_would_fail"] += int(not (e_rt < MAX_ABS_TOL and e_ref < MAX_ABS_TOL))' in src
    # primary reference: the INDEPENDENT reference state (not runtime state / runtime output / kernel)
    assert ("k_s, v_s = kernel_semantics_state(r, ref.pages, ref.layout, ref._decode_open(dtype=torch.bfloat16),"
            in src)


def test_reference_helpers_still_verbatim_stage3c():
    stage3c = ast.parse((ROOT / "vllm-kvquant/tests/quantization/test_rabit_kv2_stage3c.py").read_text(encoding="utf-8"))
    gate = ast.parse((HERE / "exp14_shape_gate.py").read_text(encoding="utf-8"))
    fn = lambda t, n: next(x for x in ast.walk(t) if isinstance(x, ast.FunctionDef) and x.name == n)  # noqa: E731
    body = lambda f: [ast.dump(s) for s in f.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]  # noqa: E731
    assert body(fn(gate, "_gqa_ref")) == body(fn(stage3c, "_gqa_ref"))
    assert body(fn(gate, "_materialize")) == body(fn(stage3c, "_materialize"))


def test_posterior_attempt4_capture_regression():
    """The amended decision applied to the accepted Attempt-4 capture (primary: independent reference state, R_sem =
    the same kernel-semantics reference). Checkpoint COUNTS only -- no observed maxima are used as thresholds."""
    data = CAPTURE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == CAPTURE_SHA256
    s = json.loads(data)
    got = {}
    for g, G in s["geometries"].items():
        c1 = c2 = n = 0
        for rep in G["replays"].values():
            for c in rep["checkpoints"]:
                v = c["R_sem"]["reference_state"]
                d = sg.conformance_decision(v["ref_max_abs"], v["max_abs_err_vs_fp32_ref"], v["max_rel_err"] or 0.0,
                                            y_all_zero=False)
                n += 1
                c1 += d["C1_pass"]
                c2 += d["C2_pass"]
        got[g] = (n, c1, c2)
    assert got["control_llama3_1_8b"] == (127, 127, 127)
    assert got["model_b_qwen2_5_7b"] == (127, 127, 127)
    assert sum(v[0] for v in got.values()) == 254  # 254 TOTAL checkpoints (127 Llama + 127 Qwen)


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
