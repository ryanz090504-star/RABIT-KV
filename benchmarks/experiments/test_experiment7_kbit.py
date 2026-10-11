"""Offline tests for the Experiment 7 K-bit ablation harness (no GPU, no torch). Run directly or with pytest."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp7_kbit_scripts as gen  # noqa: E402
import run_experiment1_quality_frontier as e1  # noqa: E402
import run_experiment7_kbit_ablation as r7  # noqa: E402


def _config_fn(text: str):
    """Compile the (nested) config_for_method of a quality script on its own and return it."""
    node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "config_for_method")
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<config_for_method>", "exec"), ns)  # noqa: S102
    return ns["config_for_method"]


def _functions(text: str) -> dict:
    return {n.name: n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef)}


def test_derived_scripts_are_exact_derivations():
    assert all(gen.check().values())
    for b in gen.BENCHMARKS:
        canon = gen.canonical_path(b).read_text(encoding="utf-8")
        derived = gen.derived_path(b).read_text(encoding="utf-8")
        assert derived == gen.derive(b, canon)
        body = derived.split("\n", 1)[1]  # undo the four edits -> exactly the canonical script
        body = body.replace(gen.K_BRANCH, "").replace(gen.NEW_ALLOWED, gen.OLD_ALLOWED).replace(gen.NEW_USE, gen.OLD_USE)
        body = gen.APP_RE.sub(f'app = modal.App("{gen.APP_RE.search(canon).group(1)}")', body)
        assert body == canon, b


def test_conditions_differ_only_in_k_bits():
    for b in gen.BENCHMARKS:
        cfg = _config_fn(gen.derived_path(b).read_text(encoding="utf-8"))
        canon_cfg = _config_fn(gen.canonical_path(b).read_text(encoding="utf-8"))
        control = cfg("rabit2")
        assert control == canon_cfg("rabit2")  # the control is the unchanged canonical rabit2
        assert {k: control[k] for k in r7.FROZEN} == r7.FROZEN and control["k_bits"] == 3
        for method, bits in r7.CONDITIONS.items():
            c = cfg(method)
            assert c["k_bits"] == bits
            diff = {k for k in set(c) | set(control) if c.get(k) != control.get(k)}
            assert diff <= {"k_bits", "name"} and (method == "rabit2" or diff == {"k_bits", "name"}), (b, method, diff)
        for other in ("rabit8", "rabit4", "rabit3"):  # pre-existing presets untouched
            assert cfg(other) == canon_cfg(other)


def test_all_quantization_accounting_and_evaluation_code_identical():
    for b in gen.BENCHMARKS:
        canon = _functions(gen.canonical_path(b).read_text(encoding="utf-8"))
        derived = _functions(gen.derived_path(b).read_text(encoding="utf-8"))
        assert set(canon) == set(derived)
        changed = {n for n in canon if ast.dump(canon[n]) != ast.dump(derived[n])}
        # only config_for_method and the enclosing function(s) holding it / the allowed set may differ
        enclosing = {n for n, node in canon.items()
                     if any(isinstance(x, ast.FunctionDef) and x.name == "config_for_method" for x in ast.walk(node))
                     or gen.OLD_ALLOWED in ast.unparse(node)}
        assert changed <= enclosing | {"config_for_method"}, (b, changed - enclosing)
        for fn in ("q_group_sym", "q_group_affine", "q_seq_affine", "q_tensor", "q_with_residual", "dequantize_state",
                   "stored_state_logical_bytes", "encode_metadata", "decode_metadata", "metadata_bytes"):
            assert fn in canon and ast.dump(canon[fn]) == ast.dump(derived[fn]), (b, fn)


def test_canonical_scripts_and_exp1_pins_untouched():
    for b in gen.BENCHMARKS:
        path = gen.canonical_path(b)
        text = path.read_text(encoding="utf-8")
        assert e1.REQUIRED_ALLOWED_LINE in text and e1.REQUIRED_RABIT2_MARKER in text
        assert e1.run_git("show", f"HEAD:{path.relative_to(e1.ROOT).as_posix()}") == text.rstrip("\n")


def test_runs_identical_to_exp1_except_methods_and_script():
    for mine, theirs in zip(r7.runs(), e1.RUNS):
        a, b = list(mine["args"]), list(theirs["args"])
        i = a.index("--methods")
        assert a[i + 1] == "bf16,rabit2_k2,rabit2,rabit2_k4" and b[i + 1] == e1.METHODS
        assert a[:i + 1] + a[i + 2:] == b[:i + 1] + b[i + 2:], mine["name"]
        assert mine["script"] == gen.DERIVED_DIR / theirs["script"]
    assert [r["name"] for r in r7.runs()] == list(gen.BENCHMARKS)


def test_logical_memory_formula():
    bf16_1024 = r7.LAYERS * 2 * 1024 * r7.KV_HEADS * r7.HEAD_DIM * 2
    assert bf16_1024 / 2**20 == 128.0  # canonical continuation_ppl bf16 avg KV MB
    mb = {k: r7.logical_kv_bytes(1024, k)["total"] / 2**20 for k in (2, 3, 4)}
    assert round(mb[3], 3) == 24.710  # canonical continuation_ppl rabit2 avg KV MB
    assert round(mb[2], 3) == 20.710 and round(mb[4], 3) == 28.710
    parts = {k: r7.logical_kv_bytes(8192, k) for k in (2, 3, 4)}
    for key in ("k_meta", "v_payload", "v_meta", "residual"):  # only the K payload depends on k_bits
        assert parts[2][key] == parts[3][key] == parts[4][key]
    assert parts[4]["k_payload"] - parts[3]["k_payload"] == parts[3]["k_payload"] - parts[2]["k_payload"] > 0


def test_row_parser_distinguishes_control_from_ablation_rows():
    log = "\n".join(["bf16 8.5020 0.0 1.00x 128.000", "rabit2_k2 9.1000 7.0 6.18x 20.710",
                     "rabit2 8.6317 1.5 5.18x 24.710", "rabit2_k4 8.5500 0.6 4.46x 28.710"])
    rows = r7.parse_rows("continuation_ppl", log)
    assert rows["rabit2"] == {"ppl": 8.6317, "avg_logical_kv_mb": 24.71}
    assert rows["rabit2_k2"]["avg_logical_kv_mb"] == 20.71 and rows["rabit2_k4"]["ppl"] == 8.55
    res = r7.integrity("continuation_ppl", 0, log)
    assert res["checks"]["kv_mb_linear_in_k_bits"] and res["checks"]["kv_mb_strictly_increasing_in_k_bits"]
    assert res["passed"]
    bad = r7.integrity("continuation_ppl", 0, log.replace("rabit2 8.6317", "rabit2 8.9000"))
    assert not bad["checks"]["bf16_and_k3_control_reproduce_canonical"] and not bad["passed"]


def test_protocol_frozen_and_complete():
    import json
    p = r7.load_protocol()  # committed == regenerated, and the only-k_bits proof holds
    assert p["experiment"] == 7 and p["axis"] == "K bits" and "NOT a physical serving benchmark" in p["type"]
    proof = p["only_k_bits_differs_proof"]
    assert proof["holds"] and proof["fields_differing_from_control_excluding_display_name"] == {
        "rabit2_k2": ["k_bits"], "rabit2": [], "rabit2_k4": ["k_bits"]}
    cfg = {m: p["conditions"][m]["config"] for m in r7.CONDITIONS}
    assert [cfg[m]["k_bits"] for m in ("rabit2_k2", "rabit2", "rabit2_k4")] == [2, 3, 4]
    for c in cfg.values():
        assert {k: c[k] for k in r7.FROZEN} == r7.FROZEN
        assert set(c) == {"name", "k_bits", *r7.FROZEN}  # every field recorded explicitly
    tol = p["control_reproduction"]["tolerances"]
    assert tol["continuation_ppl.ppl"] == {"relative": 0.005} and tol["avg_logical_kv_mb"] == {"relative": 0.001}
    for k in ("niah.accuracy_pct", "passage_retrieval.accuracy_pct", "hotpotqa.f1_pct", "qasper.f1_pct"):
        assert tol[k] == {"absolute_points": 1.0}
    assert (e1.PPL_RELATIVE_TOLERANCE, e1.PERCENTAGE_ABSOLUTE_TOLERANCE, e1.KV_MB_RELATIVE_TOLERANCE) == (0.005, 1.0, 0.001)
    counts = {b: p["benchmarks"][b].get("samples", p["benchmarks"][b].get("cases")) for b in gen.BENCHMARKS}
    assert counts == {"continuation_ppl": 8, "niah": 15, "passage_retrieval": 10, "hotpotqa": 20, "qasper": 24}
    for b in gen.BENCHMARKS:  # counts agree with the executed arguments
        args = p["benchmarks"][b]["args"]
        if "--samples" in args:
            assert int(args[args.index("--samples") + 1]) == counts[b]
        assert len(r7.quantized_prefixes(b)) == counts[b]
    assert p["benchmarks"]["continuation_ppl"]["scored_tokens_per_method"] == 8 * 128
    exp = p["logical_storage_expectations"]["expected_avg_logical_kv_mb"]
    targets = p["control_reproduction"]["canonical_targets"]
    for b in gen.BENCHMARKS:  # the frozen formula reproduces every canonical K3 and bf16 logical MB exactly
        assert exp[b]["rabit2"] == targets[b]["rabit2_K3"]["avg_logical_kv_mb"], b
        assert exp[b]["bf16_reference"] == targets[b]["bf16"]["avg_logical_kv_mb"], b
        assert exp[b]["rabit2_k2"] < exp[b]["rabit2"] < exp[b]["rabit2_k4"]
        assert abs((exp[b]["rabit2_k4"] - exp[b]["rabit2"]) - (exp[b]["rabit2"] - exp[b]["rabit2_k2"])) <= 0.002
    assert "NOT physical allocator capacity" in p["logical_storage_expectations"]["label"]
    try:
        r7.main(["--write-protocol"])
    except SystemExit as e:
        assert "never overwritten" in str(e)
    else:
        raise AssertionError("protocol overwritten")
    json.dumps(p)


def test_expected_storage_gate_fails_on_mismatch():
    good = "\n".join(["bf16 8.5020 0.0 1.00x 128.000", "rabit2_k2 9.1000 7.0 6.18x 20.710",
                      "rabit2 8.6317 1.5 5.18x 24.710", "rabit2_k4 8.5500 0.6 4.46x 28.710"])
    assert r7.integrity("continuation_ppl", 0, good)["checks"]["kv_mb_matches_expected"]
    bad = good.replace("rabit2_k4 8.5500 0.6 4.46x 28.710", "rabit2_k4 8.5500 0.6 4.46x 28.900")
    res = r7.integrity("continuation_ppl", 0, bad)
    assert not res["checks"]["kv_mb_matches_expected"] and not res["passed"]


def test_exp6_evidence_frozen():
    assert e1.run_git("diff", "--name-only", r7.EXP6_FROZEN_COMMIT, "--", "results/mlsys2027/concurrency_scaling") == ""


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
