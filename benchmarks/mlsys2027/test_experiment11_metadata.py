"""Offline tests for the Experiment 11 metadata-ablation harness (no GPU, no torch). Run directly or with pytest."""

from __future__ import annotations

import ast
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp7_kbit_scripts as g7  # noqa: E402
import exp8_vbit_scripts as g8  # noqa: E402
import exp9_group_scripts as g9  # noqa: E402
import exp10_residual_scripts as g10  # noqa: E402
import exp11_metadata_scripts as gen  # noqa: E402
import qa_control_gate as qg  # noqa: E402
import run_experiment1_quality_frontier as e1  # noqa: E402
import run_experiment9_group_ablation as r9  # noqa: E402
import run_experiment11_metadata_ablation as r11  # noqa: E402

QUANT_FUNCTIONS = ("q_group_sym", "q_group_affine", "q_seq_affine", "q_tensor", "q_with_residual", "dequantize_state",
                   "stored_state_logical_bytes", "encode_metadata", "decode_metadata", "metadata_bytes",
                   "tensor_bytes", "bf16_cache_bytes", "quantize_then_dequantize_cache")


def _functions(text: str) -> dict:
    return {n.name: n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef)}


def _canonical_rabit2(b: str) -> dict:
    fn = _functions(gen.canonical_path(b).read_text(encoding="utf-8"))["config_for_method"]
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<c>", "exec"), ns)  # noqa: S102
    return ns["config_for_method"]("rabit2")


def test_derived_scripts_are_exact_reversible_derivations():
    assert all(gen.check().values())
    for b in gen.BENCHMARKS:
        canon = gen.canonical_path(b).read_text(encoding="utf-8")
        derived = gen.derived_path(b).read_text(encoding="utf-8")
        assert derived == gen.derive(b, canon)
        body = derived.split("\n", 1)[1]
        for new, old in ((gen.M_BRANCH, ""), (gen.NEW_ALLOWED, gen.OLD_ALLOWED), (gen.NEW_USE, gen.OLD_USE)):
            assert body.count(new) == 1
            body = body.replace(new, old)
        body = g7.APP_RE.sub(f'app = modal.App("{g7.APP_RE.search(canon).group(1)}")', body)
        assert body == canon, b


def test_control_is_canonical_and_each_treatment_changes_exactly_one_metadata_field():
    for b in gen.BENCHMARKS:
        cfg = r11.derived_configs(b)
        control = cfg["rabit2"]
        assert control == _canonical_rabit2(b)
        assert control["metadata_mode"] == "int8" and control["metadata_group_size"] == 64  # META8g64
        for method, meta in r11.CONDITIONS.items():
            c = cfg[method]
            assert c["k_bits"] == 3 and c["v_bits"] == 2 and c["k_group"] == c["v_group"] == 32 and c["residual"] == 4
            assert c["k_style"] == "seq_affine" and c["v_style"] == "group_affine"
            assert {k: c[k] for k in r11.FROZEN} == r11.FROZEN
            assert {k: c[k] for k in ("metadata_mode", "metadata_group_size")} == meta
            diff = {k for k in set(c) | set(control) if c.get(k) != control.get(k)} - {"name"}
            assert sorted(diff) == r11.CHANGED_FIELD[method], (b, method, diff)
            assert len(diff) == (0 if method == "rabit2" else 1)


def test_method_names_keep_summary_row_delimiters():
    for m in ("bf16", *r11.CONDITIONS):
        assert len(m) <= 11  # every summary table uses a >= 12-wide method column


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


def test_metadata_codec_semantics_match_the_model():
    for b in gen.BENCHMARKS:
        src = ast.unparse(_functions(gen.canonical_path(b).read_text(encoding="utf-8"))["encode_metadata"])
        # bf16 mode returns before the metadata group size is read (group size inert in bf16 mode)
        assert src.index("if mode == 'bf16':") < src.index("metadata_group_size")
        assert "max(8, int(config.get('metadata_group_size', 64)))" in src.replace("\n", "").replace("    ", "")
        assert "/ 255.0" in src and ".clamp(0, 255).to(torch.uint8)" in src
        assert "'min': meta_min.to(dtype)" in src and "'scale': meta_scale.to(dtype)" in src  # BF16 secondary params
    phys = (e1.ROOT / "vllm-kvquant" / "vllm" / "v1" / "attention" / "ops" / "kvquant_k3.py").read_text(encoding="utf-8")
    assert "RABIT2_METADATA_GROUP_SIZE = 64" in phys and "/ 255.0" in phys
    assert 'meta_min.to(torch.bfloat16)' in phys and 'meta_scale.to(torch.bfloat16)' in phys


def _meta(n: int, m: int) -> tuple[int, int]:
    groups = math.ceil(n / m)
    return groups * m, 4 * groups


def test_exact_byte_accounting_with_secondary_metadata():
    cfgs = r11.derived_configs("qasper")
    for method, c in cfgs.items():
        for T in (5, 36, 1024, 4095, 10679, 16383):
            got = r11.logical_bytes(T, c)
            lq, lk = T - 4, math.ceil((T - 4) / 32) * 32
            nk, nv = 8 * (lk // 32) * 128, 8 * lq * 4
            if c["metadata_mode"] == "bf16":
                (kp, ks), (vp, vs) = (2 * nk, 0), (2 * nv, 0)
            else:
                (kp, ks), (vp, vs) = _meta(nk, c["metadata_group_size"]), _meta(nv, c["metadata_group_size"])
            hand = {"k_payload": 32 * (8 * lk * 128 * 3 // 8), "v_payload": 32 * (8 * lq * 128 * 2 // 8),
                    "k_meta_primary": 32 * 2 * kp, "k_meta_secondary": 32 * 2 * ks,
                    "v_meta_primary": 32 * 2 * vp, "v_meta_secondary": 32 * 2 * vs, "residual": 32 * 2 * 4 * 8 * 128 * 2}
            assert {k: got[k] for k in hand} == hand, (method, T)
            assert got["total"] == sum(hand.values()) == r9.traced_logical_bytes(T, c)["total"]  # accepted model
            if c["metadata_mode"] == "int8":
                assert got["k_meta_secondary"] > 0 and got["v_meta_secondary"] > 0  # secondary params not omitted
            else:
                assert got["k_meta_secondary"] == got["v_meta_secondary"] == 0


def test_expected_storage_in_protocol_matches_model_and_canonical():
    p = r11.load_protocol()
    L = p["logical_storage_expectations"]
    cfgs = r11.derived_configs("niah")
    tgt = p["control_reproduction"]["canonical_targets"]
    for b in gen.BENCHMARKS:
        exp = L["expected"][b]
        prefixes = r11.r7.quantized_prefixes(b)
        for m in r11.CONDITIONS:
            total = sum(r11.logical_bytes(t, cfgs[m])["total"] for t in prefixes)
            assert exp[m]["total_logical_bytes_all_samples"] == total
            assert exp[m]["avg_logical_kv_mb_full_precision"] == total / len(prefixes) / 2**20
            comp = exp[m]["component_bytes_all_samples"]
            assert sum(comp[k] for k in r11.COMPONENTS) == total
            assert comp["metadata_total"] == sum(comp[k] for k in r11.COMPONENTS if "meta" in k)
        assert exp["rabit2"]["avg_logical_kv_mb"] == tgt[b]["rabit2_META8g64"]["avg_logical_kv_mb"]
        assert exp["bf16_reference_avg_mb"] == tgt[b]["bf16"]["avg_logical_kv_mb"]
        assert L["gates"]["ordering"]["expected_order_smallest_first"][b] == \
            ["rabit2_m128", "rabit2", "rabit2_m32", "rabit2_mbf"]
        # payload / residual identical across conditions: only metadata moves
        for k in ("k_payload", "v_payload", "residual"):
            assert len({exp[m]["component_bytes_all_samples"][k] for m in r11.CONDITIONS}) == 1


def test_quality_is_not_invariant_by_construction():
    cfgs = r11.derived_configs("hotpotqa")
    rng = random.Random(0)
    for trial in range(5):
        # a realistic-looking primary scale / min tensor (positive scales spanning a few orders of magnitude)
        values = [rng.lognormvariate(-3.0, 1.0) * (1 if trial % 2 else -1) for _ in range(640)]
        dec = {m: r11.emulate_decoded_metadata(values, c) for m, c in cfgs.items()}
        assert r11.emulate_decoded_metadata(values, cfgs["rabit2"]) == dec["rabit2"]  # deterministic
        keys = list(dec)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                assert dec[a] != dec[b], (trial, a, b)  # different reconstructed parameters -> different K / V
    assert r11.build_protocol()["quality_semantics"]["reconstructed_kv_changes"] is True


def test_qa_rule_is_the_frozen_amendment_unchanged():
    p = r11.load_protocol()
    am = qg.load_amendment()
    assert qg.audit.sha256_lf(qg.AMENDMENT) == r11.AMENDMENT_SHA256_LF
    th = p["control_reproduction"]["hotpotqa_qasper"]
    assert th["new_thresholds_derived"] is False and th["amendment_commit"] == r11.AMENDMENT_COMMIT
    for b in r11.QA_BENCHMARKS:
        for m in ("bf16", "rabit2"):
            for k in ("historical_max_score_mismatch_count", "historical_max_l1_score_distance"):
                assert th["thresholds"][b][m][k] == am["thresholds"][b][m][k]
    assert "Experiment 11" in am["applies_to"]
    assert e1.run_git("diff", "--name-only", r11.AMENDMENT_COMMIT, "--",
                      *[str(x.relative_to(e1.ROOT)) for x in r11.AMENDMENT_FILES]) == ""


def _log(mbf="28.453", m32="24.960", m128="24.586", count=1024):
    return "\n".join([
        f"bf16        8.5020      0.00        9.3659              128.000       1.000         {count}",
        f"rabit2_mbf  8.6200      1.39        9.5300              {mbf}        4.499         {count}",
        f"rabit2_m32  8.6250      1.45        9.5400              {m32}        5.128         {count}",
        f"rabit2      8.6317      1.53        9.5462              24.710        5.180         {count}",
        f"rabit2_m128 8.6400      1.62        9.5500              {m128}        5.206         {count}"])


def test_integrity_gates_non_qa():
    p, am = r11.load_protocol(), qg.load_amendment()
    good = r11.integrity("continuation_ppl", 0, _log(), p, am)
    assert good["passed"], good["checks"]
    assert not r11.integrity("continuation_ppl", 0, _log(m32="24.963"), p, am)["checks"]["kv_mb_matches_expected"]
    assert not r11.integrity("continuation_ppl", 0, _log(count=1000), p, am)["checks"]["counts_exact"]
    drift = r11.integrity("continuation_ppl", 0, _log().replace("rabit2      8.6317", "rabit2      8.9000"), p, am)
    assert not drift["checks"]["bf16_and_control_reproduce_canonical"]
    assert not r11.integrity("continuation_ppl", 0, "\n".join(_log().splitlines()[:-1]), p, am)["passed"]


def test_integrity_uses_frozen_qa_gate_for_hotpotqa_qasper():
    p, am = r11.load_protocol(), qg.load_amendment()
    r = r11.integrity("hotpotqa", 0, (e1.ROOT / "results" / "quality" / "hotpotqa.log").read_text(encoding="utf-8"), p, am)
    assert "qa_control_gate_passed" in r["checks"] and "bf16_and_control_reproduce_canonical" not in r["checks"]
    assert r["checks"]["qa_control_gate_passed"] and r["control_rule"] == "frozen post-failure per-example QA control gate"
    assert not r["checks"]["all_five_rows_present"]  # the canonical log has no treatment rows -> run would fail


def test_protocol_frozen_complete_and_not_overwritable():
    p = r11.load_protocol()
    assert p["experiment"] == 11 and "NOT a physical serving benchmark" in p["type"]
    assert "META8g64 is not presupposed best" in p["question"]
    assert p["methods_argument"] == "bf16,rabit2_mbf,rabit2_m32,rabit2,rabit2_m128"
    assert p["only_metadata_field_differs_proof"]["holds"]
    kinds = {c["kind"] for c in p["meta8g64_definition"]["stored_components_meta8g64"]}
    assert kinds == {"primary", "secondary"}
    for b in gen.BENCHMARKS:
        assert p["benchmarks"][b]["expected_count_per_method"] == r11.COUNT_COLUMNS[b][1]
    try:
        r11.main(["--write-protocol"])
    except SystemExit as e:
        assert "never overwritten" in str(e)
    else:
        raise AssertionError("protocol overwritten")


def test_runs_identical_to_exp1_except_methods_and_script():
    for mine, theirs in zip(r11.runs(), e1.RUNS):
        a, b = list(mine["args"]), list(theirs["args"])
        i = a.index("--methods")
        assert a[i + 1] == "bf16,rabit2_mbf,rabit2_m32,rabit2,rabit2_m128"
        assert a[:i + 1] + a[i + 2:] == b[:i + 1] + b[i + 2:]
        assert mine["script"] == gen.DERIVED_DIR / theirs["script"]


def test_prior_evidence_exp10_and_amendment_unchanged():
    assert e1.run_git("diff", "--name-only", r11.EXP6_FROZEN_COMMIT, "--", "results/mlsys2027/concurrency_scaling") == ""
    for commit, files in ((r11.EXP7_EVIDENCE_COMMIT, r11.r8.EXP7_FILES), (r11.EXP8_EVIDENCE_COMMIT, r11.r9.EXP8_FILES),
                          (r11.EXP9_EVIDENCE_COMMIT, r11.r10.EXP9_FILES), (r11.EXP10_EVIDENCE_COMMIT, r11.EXP10_FILES),
                          (r11.AMENDMENT_COMMIT, r11.AMENDMENT_FILES)):
        assert e1.run_git("diff", "--name-only", commit, "--", *[str(x.relative_to(e1.ROOT)) for x in files]) == ""
    for b in gen.BENCHMARKS:
        path = gen.canonical_path(b)
        assert e1.run_git("show", f"HEAD:{path.relative_to(e1.ROOT).as_posix()}") == \
            path.read_text(encoding="utf-8").rstrip("\n")
    assert all(g.check()[b] for g in (g7, g8, g9, g10) for b in gen.BENCHMARKS)


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
