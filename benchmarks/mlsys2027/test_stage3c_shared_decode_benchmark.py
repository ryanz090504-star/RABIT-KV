"""
Offline tests for the Stage3C shared_decode benchmark harness (no GPU, no torch,
no pytest required: `python benchmarks/mlsys2027/test_stage3c_shared_decode_benchmark.py`;
also collected by pytest).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_stage3c_shared_decode_benchmark as rs  # noqa: E402


def _base_and_current():
    return rs.rp._git_show(rs.PROFILED_COMMIT, rs.TRITON_ATTN), rs.TRITON_ATTN.read_text(encoding="utf-8")


def test_triton_attn_only_adds_shared_decode_dispatch():
    base, cur = _base_and_current()
    assert rs.triton_attn_only_adds_shared_decode(base, cur)
    assert not rs.triton_attn_only_adds_shared_decode(base, base)  # the dispatch must be present
    for old, new in (('rabit2_stage3c_impl() == "shared_decode"', 'rabit2_stage3c_impl() == "tile32"'),
                     ("                    continue\r\n\r\n                # Stage4B4",
                      "                    pass\r\n\r\n                # Stage4B4"),
                     ("                    continue\n\n                # Stage4B4",
                      "                    pass\n\n                # Stage4B4"),
                     ("chunk_plan.apply_step(local_idx)", "chunk_plan.apply_step(local_idx + 0)"),
                     ("q_len > 1 and not self._rabit2_logged_chunked", "q_len > 2 and not self._rabit2_logged_chunked")):
        mutated = cur.replace(old, new, 1)
        if mutated != cur:
            assert not rs.triton_attn_only_adds_shared_decode(base, mutated), old


def test_equivalence_and_frozen_sources():
    eq = rs.verify_equivalence(json.loads(rs.BENCH_MANIFEST.read_text(encoding="utf-8")))
    assert eq["triton_attn_equal_profiled_modulo_shared_decode_dispatch"] and eq["tile32_module_and_tests_frozen"]
    assert eq["profiler_module_unchanged"] and eq["modal_backstop_s"] > eq["watchdog_budget_s"]
    assert rs.sha256(rs.RABIT_KV2) == rs.EXPECTED_RABIT_SHA256_LF


def test_series_grid_and_inventory():
    assert [s[2] for s in rs.SERIES] == ["reference", "tile32"] + ["shared_decode"] * 4
    assert [s[3] for s in rs.SERIES[2:]] == [4, 8, 16, 32] and all(s[1] == "rabit_kv2" for s in rs.SERIES)
    assert rs.Q_LENS == [32, 512, 2048, 4096, 8192] and 16384 + 16352 not in rs.POINTS
    assert rs.EXPECTED_SHARED_TESTS == 192
    cmd = rs.build_command(False)
    assert "--correctness-only" not in cmd and "rabit_shared_qb32=rabit_kv2:shared_decode:32" in " ".join(cmd)
    assert "--correctness-only" in rs.build_command(True)


def _count(prefix, q, qb, measured=None, expected=None, per_query=None):
    return "SHARED_DECODE_COUNTS=" + json.dumps({
        "prefix": prefix, "q_len": q, "query_block": qb, "tiles": 2,
        "measured_shared_decode_page_decodes": 10 if measured is None else measured,
        "expected_shared_decode": 10 if expected is None else expected,
        "per_query_page_decodes_tile32_reference": 40 if per_query is None else per_query, "reuse_factor": 4.0})


def _shared_lines(n_pass=rs.EXPECTED_SHARED_TESTS, skipped=0, counts=None, drop_32k=False):
    ids = [f"test_x[{i}]" for i in range(n_pass - 4)] + ([] if drop_32k else rs.EXPECTED_32K_TESTS)
    if drop_32k:
        ids += [f"test_y[{i}]" for i in range(4)]
    lines = [f"PASSED vllm-kvquant/tests/quantization/test_rabit2_stage3c_shared_decode.py::{t}" for t in ids]
    lines += counts if counts is not None else [_count(p, q, qb) for p, q in ((79, 33), (95, 1024), (16384, 512))
                                                 for qb in (4, 8, 16, 32)]
    lines.append(f"{n_pass} passed" + (f", {skipped} skipped" if skipped else "") + " in 123.45s")
    return lines


def _top(ok=True):
    bsess = (rs.rt.OUT_DIR / "modal_session.log").read_text(encoding="utf-8").splitlines()
    env = next(ln for ln in bsess if ln.startswith("S3C_ENVIRONMENT="))
    lines = [env, 'S3C_GATE_START={"cmd": []}', 'S3C_GATE_EXIT={"returncode": 0}', 'S3C_TILE_TESTS_START={"cmd": []}',
             'S3C_TILE_TESTS_EXIT={"returncode": 0}', 'S3C_SHARED_TESTS_START={"cmd": []}',
             f'S3C_SHARED_TESTS_EXIT={{"returncode": {0 if ok else 1}}}', "S3C_CORRECTNESS_ONLY_COMPLETE"]
    return rs.parse_top(lines)


def _gate():
    g = (rs.rt.OUT_DIR / "correctness_gate.log").read_text(encoding="utf-8").splitlines()
    return rs.r5.parse_gate(g)


def _t32():
    return rs.rt.parse_tests((rs.rt.OUT_DIR / "tile32_correctness_tests.log").read_text(encoding="utf-8").splitlines())


def test_correctness_only_integrity_passes_and_fails_correctly():
    integ = rs.integrity({}, _gate(), _t32(), rs.parse_shared_tests(_shared_lines()), _top(), True)
    assert integ["all_ok"], [c["check"] for c in integ["checks"] if c["state"] != "passed"]
    bad_cases = {
        "skipped": rs.parse_shared_tests(_shared_lines(skipped=1)),
        "too_few": rs.parse_shared_tests(_shared_lines(n_pass=191)),
        "no_16352": rs.parse_shared_tests(_shared_lines(drop_32k=True)),
        "count_mismatch": rs.parse_shared_tests(_shared_lines(counts=[_count(79, 33, 4, measured=11)] + [
            _count(p, q, qb) for p, q in ((79, 33), (95, 1024), (16384, 512)) for qb in (4, 8, 16, 32)][1:])),
        "no_reuse": rs.parse_shared_tests(_shared_lines(counts=[_count(79, 33, 4, per_query=10)] + [
            _count(p, q, qb) for p, q in ((79, 33), (95, 1024), (16384, 512)) for qb in (4, 8, 16, 32)][1:])),
        "missing_counts": rs.parse_shared_tests(_shared_lines(counts=[])),
    }
    for name, shared in bad_cases.items():
        assert not rs.integrity({}, _gate(), _t32(), shared, _top(), True)["all_ok"], name
    assert not rs.integrity({}, _gate(), _t32(), rs.parse_shared_tests(_shared_lines()), _top(ok=False), True)["all_ok"]


def test_analysis_ratios_and_fastest_block():
    def series(ttft):
        pts = [{"begin": {"role": "measured"}, "row": {"planned_prompt_tokens": p, "ttft_ms": ttft * (i + 1),
                                                       "tpot_ms": 30.0, "wall_ms": ttft * (i + 1) + 900,
                                                       "output_token_ids_sha256": "h"}}
               for i, p in enumerate(rs.POINTS)]
        return {"points": pts}
    ttfts = {"rabit_reference": 100.0, "rabit_tile32": 90.0, "rabit_shared_qb4": 60.0, "rabit_shared_qb8": 40.0,
             "rabit_shared_qb16": 45.0, "rabit_shared_qb32": 70.0}
    an = rs.analysis({k: series(v) for k, v in ttfts.items()}, {"decode_counts": []},
                     {"all_ok": True, "counts": {}}, False)
    p = an["points"][2]
    assert p["fastest_query_block_by_ttft"] == 8
    r = p["shared_decode_ratios"]["qb8"]
    assert abs(r["ttft_over_reference"] - 0.4) < 1e-12 and abs(r["ttft_over_tile32"] - 40 / 90) < 1e-12
    assert p["series"]["rabit_reference"]["query_block"] is None and p["series"]["rabit_shared_qb16"]["query_block"] == 16


def test_qb_selection_rule_geomean():
    import math

    t = {4: {512: 100.0, 2048: 400.0}, 8: {512: 90.0, 2048: 420.0}, 16: {512: 120.0, 2048: 300.0},
         32: {512: 200.0, 2048: 290.0}}
    sel = rs.select_query_block(t)
    gm = {qb: math.sqrt(r[512] * r[2048]) for qb, r in t.items()}
    assert sel["selected_query_block"] == min(gm, key=gm.get) == 16
    assert all(abs(sel["geomean_ttft_ms"][str(k)] - gm[k]) < 1e-9 for k in gm)
    assert abs(sel["geomean_relative_to_selected"]["16"] - 1.0) < 1e-12
    # geometric mean, not arithmetic: qb 8 wins arithmetically? (100+400=500 vs 90+420=510) -> no; check a case
    t2 = {4: {512: 10.0, 2048: 1000.0}, 8: {512: 200.0, 2048: 200.0}, 16: {512: 300.0, 2048: 300.0},
          32: {512: 400.0, 2048: 400.0}}
    assert rs.select_query_block(t2)["selected_query_block"] == 4  # geomean 100 < 200 (arithmetic would pick 8)
    # exact tie -> smaller block
    t3 = {qb: {512: 100.0, 2048: 400.0} for qb in (4, 8, 16, 32)}
    assert rs.select_query_block(t3)["selected_query_block"] == 4
    for bad in ({4: t[4], 8: t[8], 16: t[16]}, {**t, 32: {512: 1.0}}, {**t, 32: {512: 0.0, 2048: 1.0}}):
        try:
            rs.select_query_block(bad)
        except ValueError:
            continue
        raise AssertionError("invalid tuning input accepted")


def test_qb_tuning_mode_grid():
    import importlib

    m = importlib.reload(rs)
    try:
        m.set_mode_qb_tuning()
        assert [s[2] for s in m.SERIES] == ["shared_decode"] * 4 and [s[3] for s in m.SERIES] == [4, 8, 16, 32]
        assert m.Q_LENS == [512, 2048] and m.POINTS == [16896, 18432] and m.OUT_DIR == m.QB_TUNING_DIR
        cmd = " ".join(m.build_command(False))
        assert "rabit_reference" not in cmd and "rabit_tile32" not in cmd and "--points 16896,18432 " in cmd
        assert "--correctness-only" not in cmd
        eq = m.verify_equivalence(json.loads(m.BENCH_MANIFEST.read_text(encoding="utf-8")))
        assert eq["modal_backstop_s"] > eq["watchdog_budget_s"]
    finally:
        importlib.reload(rs)


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
