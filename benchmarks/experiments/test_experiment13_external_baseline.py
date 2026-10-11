"""Offline tests for the Experiment 13 matched external-baseline harness (no GPU). Run directly or with pytest."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiment1_quality_frontier as e1  # noqa: E402
import run_experiment13_external_baseline as r13  # noqa: E402

ROOT = e1.ROOT
RABIT2 = {"k_bits": 3, "v_bits": 2, "k_style": "seq_affine", "v_style": "group_affine", "k_group": 32, "v_group": 32,
          "residual": 4, "metadata_mode": "int8", "metadata_group_size": 64}


def test_rabit_frozen_config_and_source_unchanged():
    text = (ROOT / "benchmarks" / "quality" / "hotpotqa.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "config_for_method")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<c>", "exec"), ns)  # noqa: S102
    cfg = ns["config_for_method"]("rabit2")
    assert {k: cfg[k] for k in RABIT2} == RABIT2  # K3 / V2 / G32 / R4 / META8g64
    src = (ROOT / "vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(src).hexdigest() == r13.EXPECTED_RABIT_SHA256_LF


def test_worker_and_image_equivalent_to_exp4_except_documented_differences():
    eq = r13.verify_equivalence()
    assert all(eq.values()), eq
    w = (HERE / "exp13_engine_worker.py").read_text(encoding="utf-8")
    assert w.count('kwargs.pop("attention_config")') == 1  # the ONLY engine-kwarg difference, TurboQuant only


def test_turboquant_physical_mode_enabled_and_backport_present():
    p = r13.load_protocol()
    assert p["conditions"]["D"]["kv_cache_dtype"] == "turboquant_k3v4_nc"
    assert p["conditions"]["D"]["boundary_layers"] == ["0", "1", "30", "31"]
    assert e1.run_git("rev-parse", f"HEAD:{r13.ATTN_UTILS}") == r13.UPSTREAM_ATTN_UTILS_BLOB
    e1.run_git("merge-base", "--is-ancestor", r13.BACKPORT_COMMIT, "HEAD")
    assert e1.run_git("diff", "--name-only", r13.BACKPORT_COMMIT, "HEAD", "--", "vllm-kvquant") == ""
    changed = e1.run_git("show", "--name-only", "--format=", r13.BACKPORT_COMMIT).split()
    assert changed == [r13.ATTN_UTILS]  # the backport commit touches exactly one file
    # integrity rejects a D leg whose live engine is not in the TurboQuant cache dtype, or lacks the boundary policy
    text = _session()
    wrong_dtype = text.replace('[leg4:turboquant_k3v4_nc] EXP13_KV_DTYPE={"engine_cache_dtype": "turboquant_k3v4_nc"',
                               '[leg4:turboquant_k3v4_nc] EXP13_KV_DTYPE={"engine_cache_dtype": "bfloat16"', 1)
    assert wrong_dtype != text
    res = r13.analyze(wrong_dtype)
    assert not res["integrity"]["checks"]["D1_engine_cache_dtype"] and not res["integrity"]["passed"]
    no_boundary = text.replace('[leg5:turboquant_k3v4_nc] EXP13_EFFECTIVE_ENGINE_CONFIG={"block_size": 32, '
                               '"kv_cache_dtype_skip_layers": ["0", "1", "30", "31"]',
                               '[leg5:turboquant_k3v4_nc] EXP13_EFFECTIVE_ENGINE_CONFIG={"block_size": 32, '
                               '"kv_cache_dtype_skip_layers": []', 1)
    assert no_boundary != text
    assert not r13.analyze(no_boundary)["integrity"]["checks"]["D2_tq_boundary_layers"]


def _cap(blocks, bs=32, cap=None):
    return {"tags": {"EXP13_CAPACITY": {"num_gpu_blocks": blocks, "block_size": bs,
                                        "capacity_tokens": blocks * bs if cap is None else cap}}}


def test_capacity_parser_reads_allocator_evidence_not_nominal_bits():
    assert r13.capacity(_cap(37600)) == 1203200
    assert r13.capacity(_cap(37600, cap=999)) is None  # inconsistent record rejected
    assert r13.capacity({"tags": {}}) is None  # no allocator record -> no capacity (never inferred from bits)
    src = Path(r13.__file__).read_text(encoding="utf-8")
    cap_fn = ast.unparse(next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "capacity"))
    assert "THEORETICAL" not in cap_fn and "bits" not in cap_fn.replace("nominal bit", "")


def test_workload_settings_match_and_order_trials_frozen():
    p = r13.load_protocol()
    assert [l["label"] for l in p["session"]["legs"]] == ["A1", "B1", "C1", "D1", "D2", "C2", "B2", "A2"]
    assert [l["kv_cache_dtype"] for l in p["session"]["legs"]] == [r13.A, r13.B, r13.C, r13.D, r13.D, r13.C, r13.B, r13.A]
    assert (p["session"]["warmups_per_leg"], p["session"]["reps_per_leg"], p["session"]["samples_per_condition"]) == (5, 30, 60)
    assert p["workload"]["context_tokens"] == 2048 and p["workload"]["output_tokens"] == 32
    kw = p["engine"]["worker_base_engine_kwargs"]
    assert (kw["block_size"], kw["gpu_memory_utilization"], kw["max_model_len"], kw["max_num_seqs"],
            kw["max_num_batched_tokens"], kw["enforce_eager"], kw["enable_chunked_prefill"], kw["enable_prefix_caching"]) == \
        (32, 0.82, 32768, 32, 16384, True, True, False)
    assert r13.legs_arg() == "A1=bfloat16,B1=fp8_e4m3,C1=rabit_kv2,D1=turboquant_k3v4_nc,D2=turboquant_k3v4_nc," \
                             "C2=rabit_kv2,B2=fp8_e4m3,A2=bfloat16"
    modal = (HERE / "exp13_deployment_modal.py").read_text(encoding="utf-8")
    assert "tests/quantization/test_turboquant.py" in modal and "VLLM_USE_V2_MODEL_RUNNER" not in modal


BACKEND_LOG = {  # verbatim formats observed in the non-evidence V2 probe log
    "tq": ["INFO [cuda.py:476] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN'].",
           "INFO [flash_attn.py:718] Using FlashAttention version 2",
           "INFO [cuda.py:476] Using TURBOQUANT attention backend out of potential backends: ['TURBOQUANT']."],
    "triton": ["INFO [cuda.py:416] Using AttentionBackendEnum.TRITON_ATTN backend."]}
KV_GIB = {"bfloat16": "47.98", "fp8_e4m3": "47.95", "rabit_kv2": "47.98", "turboquant_k3v4_nc": "47.98"}


def _leg_lines(d, label, blocks, tpot, skip=None, backend="AttentionBackendEnum.TRITON_ATTN", gib=None,
               backend_log=None):
    eff = {"block_size": 32, "kv_cache_dtype_skip_layers": skip or [], "attention_backend": backend,
           "flash_attn_version": 2 if d == r13.D else None, "gpu_memory_utilization": 0.82}
    req = {"block_size": 32, "kv_cache_dtype": d, **({} if d == r13.D else {"attention_config": {"backend": "TRITON_ATTN"}})}
    lines = [f"EXP13_REQUESTED_ENGINE_KWARGS={json.dumps(req)}", f"EXP13_EFFECTIVE_ENGINE_CONFIG={json.dumps(eff)}",
             f"EXP13_WORKLOAD={json.dumps({'context_tokens': 2048, 'output_tokens': 32})}",
             f"EXP13_KV_DTYPE={json.dumps({'engine_cache_dtype': d, 'requested_kv_cache_dtype': d})}",
             f"EXP13_CAPACITY={json.dumps({'num_gpu_blocks': blocks, 'block_size': 32, 'capacity_tokens': blocks * 32})}",
             "INFO Using V2 Model Runner", f"INFO Available KV cache memory: {gib or KV_GIB[d]} GiB",
             *(backend_log if backend_log is not None else BACKEND_LOG["tq" if d == r13.D else "triton"])]
    if d == r13.C:
        lines.append('EXP13_RABIT_MARKERS={"a": true, "b": true, "c": true}')
    lines += [f"EXP13_WARMUP {json.dumps({'rep': i, 'prompt_tokens': 2048, 'output_tokens': 32, 'tpot_ms': tpot, 'ttft_ms': 50.0, 'wall_ms': 500.0})}" for i in range(5)]
    lines += [f"EXP13_SAMPLE {json.dumps({'rep': i, 'prompt_tokens': 2048, 'output_tokens': 32, 'tpot_ms': tpot + i * 0.01, 'ttft_ms': 50.0, 'wall_ms': 500.0})}" for i in range(30)]
    return [f"[leg{k}:{d}] {ln}" for k, lb, dd in r13.LEGS if lb == label for ln in lines]


def _session(tq_blocks_leg2=37600):
    blocks = {r13.A: 12282, r13.B: 24549, r13.C: 64831, r13.D: 37600}
    tp = {r13.A: 8.0, r13.B: 8.5, r13.C: 9.0, r13.D: 10.0}
    lines = ['EXP13_ENVIRONMENT={"gpus": [{"name": "NVIDIA H100 80GB HBM3"}]}', '[gate] EXP3_GATE_RESULT={"passed": true}',
             "EXP13_GATE_EXIT={\"returncode\": 0}",
             '[tqgate] EXP13_TQ_GATE_SUMMARY={"stage": "execution", "valid": true, "counts": {"passed": 121, "skipped": 2}}',
             'EXP13_TQ_GATE_EXIT={"returncode": 0}']
    for k, label, d in r13.LEGS:
        lines.append(f'EXP13_PRE_LEG_GPU_STATE={{"leg": "{label}", "clean": true}}')
        b = tq_blocks_leg2 if label == "D2" else blocks[d]
        lines += _leg_lines(d, label, b, tp[d], skip=["0", "1", "30", "31"] if d == r13.D else None,
                            backend="None" if d == r13.D else "AttentionBackendEnum.TRITON_ATTN")
        lines.append(f'EXP13_LEG_EXIT={{"leg": "{label}", "returncode": 0}}')
    lines.append("EXP13_MIRRORED_COMPLETE")
    return "\n".join(lines)


def test_measurement_parser_is_condition_agnostic_and_integrity_works():
    res = r13.analyze(_session())
    assert res["integrity"]["passed"], {k: v for k, v in res["integrity"]["checks"].items() if not v}
    s = res["summary"]
    assert s["observed_capacity"]["turboquant"]["observed_capacity_tokens"] == 1203200
    assert s["observed_capacity"]["rabit"]["observed_capacity_tokens"] == 2074592
    assert all(s["latency_secondary_pooled_within_session_descriptive"][c]["n"] == 60
               for c in ("bf16", "fp8", "rabit", "turboquant"))
    assert abs(s["capacity_ratios"]["rabit_over_turboquant"] - 2074592 / 1203200) < 1e-12
    assert set(s["capacity_ratios"]) == {"rabit_over_turboquant", "rabit_over_bf16", "rabit_over_fp8",
                                         "turboquant_over_bf16", "turboquant_over_fp8", "fp8_over_bf16"}
    assert set(s["rabit_comparisons"]) == {"rabit_vs_bf16", "rabit_vs_fp8", "rabit_vs_turboquant"}
    assert "METHOD-NATIVE SYSTEM" in s["rabit_comparisons"]["rabit_vs_turboquant"]["framing"]


def test_frozen_aggregation_per_leg_cross_leg_pooled_and_drift():
    res = r13.analyze(_session())
    s = res["summary"]
    rab = s["latency_primary_per_leg"]["rabit"]
    assert set(rab) == {"C1", "C2"} and all(rab[l]["n"] == 30 for l in rab)
    assert set(rab["C1"]) == {"n", *r13.LEG_STATS}
    cross = s["latency_primary_cross_leg"]["rabit"]
    for m in r13.LEG_STATS:
        assert cross[f"median_of_leg_{m}"] == (rab["C1"][m] + rab["C2"][m]) / 2  # median of two values
    dr = s["leg_to_leg_drift"]["rabit"]
    assert dr["legs"] == ["C1", "C2"] and dr["tpot_abs_diff_ms"] == rab["C2"]["median_tpot_ms"] - rab["C1"]["median_tpot_ms"]
    assert {"tpot_pct_diff", "ttft_abs_diff_ms", "wall_abs_diff_ms"} <= set(dr)
    comp = s["rabit_comparisons"]["rabit_vs_fp8"]
    fp8 = s["latency_primary_cross_leg"]["fp8"]
    assert comp["tpot_abs_diff_ms"] == cross["median_of_leg_median_tpot_ms"] - fp8["median_of_leg_median_tpot_ms"]
    assert "no significance tests" in s["statistics_note"]
    assert not any(k in json.dumps(s) for k in ("ci_low", "ci_high", "p_value", "confidence"))
    p = r13.load_protocol()["metrics"]["latency"]
    assert "not inferential" in p["primary_cross_leg"] and "NOT treated as 60 independent" in p["secondary"]


def test_physical_mode_consistency_and_backend_evidence_are_enforced():
    bf16_sized = _session().replace("[leg4:turboquant_k3v4_nc] INFO Available KV cache memory: 47.98 GiB",
                                    "[leg4:turboquant_k3v4_nc] INFO Available KV cache memory: 146.88 GiB", 1)
    res = r13.analyze(bf16_sized)  # a BF16-sized footprint at TurboQuant capacity is not the packed layout
    assert not res["integrity"]["checks"]["D1_physical_layout_consistent"] and not res["integrity"]["passed"]
    no_tq_backend = "\n".join(ln for ln in _session().splitlines()
                              if not (ln.startswith("[leg5:turboquant_k3v4_nc]") and "TURBOQUANT attention backend" in ln))
    assert not r13.analyze(no_tq_backend)["integrity"]["checks"]["D2_backend_evidence"]
    triton_on_tq = _session().replace("[leg4:turboquant_k3v4_nc] INFO [cuda.py:476] Using TURBOQUANT attention backend",
                                      "[leg4:turboquant_k3v4_nc] INFO [cuda.py:416] Using AttentionBackendEnum.TRITON_ATTN "
                                      "backend. Using TURBOQUANT attention backend", 1)
    assert not r13.analyze(triton_on_tq)["integrity"]["checks"]["D1_backend_evidence"]
    # the same parser code path handles every condition (no per-condition branch in parse_worker)
    src = ast.unparse(next(n for n in ast.walk(ast.parse(Path(r13.__file__).read_text(encoding="utf-8")))
                           if isinstance(n, ast.FunctionDef) and n.name == "parse_worker"))
    assert all(x not in src for x in ("bfloat16", "fp8_e4m3", "rabit_kv2", "turboquant"))
    bad = r13.analyze(_session(tq_blocks_leg2=37000))  # the two TurboQuant legs disagree on capacity
    assert not bad["integrity"]["passed"] and bad["summary"] is None


def test_no_fake_quant_path_can_be_labelled_physical():
    p = r13.load_protocol()
    # every condition is a real vLLM engine KV-cache dtype; no HF fake-quant / quality script is involved
    assert {c["kv_cache_dtype"] for c in p["conditions"].values()} == {r13.A, r13.B, r13.C, r13.D}
    for f in (HERE / "exp13_engine_worker.py", HERE / "exp13_deployment_modal.py", Path(r13.__file__)):
        t = f.read_text(encoding="utf-8")
        assert "benchmarks/quality" not in t and "quantize_then_dequantize" not in t
    assert "OBSERVED PHYSICAL" in p["metrics"]["capacity"]["primary"]
    assert "theoretical" in json.dumps(p["metrics"]["capacity"]).lower()


def test_protocol_frozen_fairness_claims_and_not_overwritable():
    p = r13.load_protocol()
    assert "UNAVOIDABLY_DIFFERENT" in p["fairness"] and "TURBOQUANT" in p["fairness"]["UNAVOIDABLY_DIFFERENT"][0]
    assert "method-native system comparison" in p["claim_boundary"] and "No throughput claim" in p["claim_boundary"]
    assert "fa4321de3d894c50c5ca0766dffa352d3fb07423" in p["disclosure"] and "PR #47609" in p["disclosure"]
    try:
        r13.main(["--write-protocol"])
    except SystemExit as e:
        assert "never overwritten" in str(e)
    else:
        raise AssertionError("protocol overwritten")


def test_prior_accepted_evidence_unchanged():
    frozen = {"4f767ab03d83e043b2871dd0cd4cf2f8dc862e6b": ["results/mlsys2027/variance", "results/mlsys2027/ablations",
                                                           "results/mlsys2027/control_reproducibility_audit",
                                                           "results/quality", "results/mlsys2027/quality_frontier",
                                                           "results/mlsys2027/deployment", "results/mlsys2027/fp8_baseline",
                                                           "benchmarks/quality"],
              "0f5f6efa7cfdf0b7add27521270e91246ec4c191": ["results/mlsys2027/concurrency_scaling"]}
    for commit, paths in frozen.items():
        assert e1.run_git("diff", "--name-only", commit, "--", *paths) == "", commit
    for probe in ("feasibility_probe", "legacy_runner_probe", "v2_runner_probe"):
        assert e1.run_git("status", "--short", "--", f"results/mlsys2027/external_baseline/{probe}") == ""



# ---------------------------------------------------------------- frozen TurboQuant gate (post-failure harness amendment)
import exp13_tq_gate as tqg  # noqa: E402


def _junit(outcomes: dict) -> str:
    rows = []
    for nid, (outcome, msg) in outcomes.items():
        cls, name = nid.split("::")[1], nid.split("::")[2]
        inner = {"passed": "", "skipped": f'<skipped type="pytest.skip" message="{msg}"/>',
                 "xfailed": f'<skipped type="pytest.xfail" message="{msg}"/>',
                 "failed": f'<failure message="{msg}"/>', "error": f'<error message="{msg}"/>'}[outcome]
        rows.append(f'<testcase classname="tests.quantization.test_turboquant.{cls}" name="{name}">{inner}</testcase>')
    return "<testsuites><testsuite>" + "".join(rows) + "</testsuite></testsuites>"


def _outcomes(skip_scipy=True, **override):
    out = {n: ("passed", "") for n in tqg.EXPECTED_NODE_IDS}
    if skip_scipy:
        for n in tqg.SCIPY_NODE_IDS:
            out[n] = ("skipped", "could not import 'scipy': No module named 'scipy'")
    out.update(override)
    return out


def _counts(outcomes):
    c = {k: 0 for k in tqg.COUNT_KEYS}
    for o, _ in outcomes.values():
        c[{"error": "errors"}.get(o, o)] += 1
    return c


def _eval(outcomes, scipy_importable=False, gpgpu=True, collected=None, run_rc=0, counts=None):
    collected = list(tqg.EXPECTED_NODE_IDS) if collected is None else collected
    return tqg.evaluate(collected, 0, scipy_importable, gpgpu, run_rc, tqg.parse_junit(_junit(outcomes)),
                        counts if counts is not None else _counts(outcomes))


def test_tq_gate_frozen_values_match_the_collection_probe():
    rec = json.loads((ROOT / "results/mlsys2027/external_baseline/tq_collect_probe/collect_record.json").read_text(encoding="utf-8"))
    assert tqg.EXPECTED_COUNT == 123 == len(tqg.EXPECTED_NODE_IDS) == rec["count"]
    assert sorted(tqg.EXPECTED_NODE_IDS) == sorted(rec["node_ids"])
    assert tqg.EXPECTED_NODE_IDS_SHA256 == rec["sha256_sorted"] == tqg.node_ids_sha256(rec["node_ids"])
    assert len(tqg.EXPECTED_NODE_IDS_SHA256) == 64  # FULL hash, not abbreviated
    assert tqg.SCIPY_NODE_IDS == sorted(rec["scipy_reference"]) and len(tqg.SCIPY_NODE_IDS) == 2
    assert all("test_centroids_match_scipy_reference[" in n for n in tqg.SCIPY_NODE_IDS)
    assert tqg.GPU_ONLY_NODE_IDS == sorted(rec["gpu_only"]) and len(tqg.GPU_ONLY_NODE_IDS) == 15
    assert rec["environment"]["scipy_importable"] is False and rec["environment"]["gpgpu_available_expr"] == "True"
    assert tqg.CONFCUTDIR == "--confcutdir=/root/vllm-kvquant/tests/quantization"
    p = r13.load_protocol()["tq_gate"]
    assert p["expected_node_ids_sha256"] == tqg.EXPECTED_NODE_IDS_SHA256 and p["expected_collected_items"] == 123
    assert "does NOT include an end-to-end store / decode round-trip item for turboquant_k3v4_nc" in p["correctness_claim_boundary"]


def test_tq_gate_accepts_only_the_frozen_outcomes():
    assert _eval(_outcomes())["valid"]  # scipy absent: 121 passed + 2 frozen SciPy skips
    assert _eval(_outcomes(skip_scipy=False), scipy_importable=True)["valid"]  # scipy present: 123 passed
    assert not _eval(_outcomes(), scipy_importable=True)["valid"]  # scipy present but skipped
    gpu = tqg.GPU_ONLY_NODE_IDS[0]
    bad_gpu = _eval(_outcomes(**{gpu: ("skipped", "GPGPU not available")}))
    assert not bad_gpu["valid"] and not bad_gpu["checks"]["gpu_only_all_executed_and_passed"]
    other = next(n for n in tqg.EXPECTED_NODE_IDS if n not in tqg.SCIPY_NODE_IDS and n not in tqg.GPU_ONLY_NODE_IDS)
    assert not _eval(_outcomes(**{other: ("skipped", "No module named 'scipy'")}))["checks"]["skipped_exactly_allowed_set"]
    wrong_reason = _outcomes(**{tqg.SCIPY_NODE_IDS[0]: ("skipped", "some other reason")})
    assert not _eval(wrong_reason)["checks"]["skip_reasons_missing_scipy"]
    assert not _eval(_outcomes(**{other: ("failed", "assert")}), run_rc=1)["valid"]
    assert not _eval(_outcomes(**{other: ("xfailed", "x")}))["checks"]["xfailed_0"]
    xp = _counts(_outcomes())
    xp["xpassed"] = 1
    assert not _eval(_outcomes(), counts=xp)["checks"]["xpassed_0"]
    assert not _eval(_outcomes(), gpgpu=False)["valid"]


def test_tq_gate_collection_stage_stops_before_execution():
    short = list(tqg.EXPECTED_NODE_IDS)[:-1]
    res = tqg.evaluate(short, 0, False, True, None, {}, {})
    assert res["stage"] == "collection" and res["valid"] is False and not res["checks"]["collected_count_123"]
    swapped = list(tqg.EXPECTED_NODE_IDS)[:-1] + [tqg.EXPECTED_NODE_IDS[-1] + "_renamed"]
    res = tqg.evaluate(swapped, 0, False, True, None, {}, {})
    assert res["checks"]["collected_count_123"] and not res["checks"]["collected_hash_matches"] and res["valid"] is False
    ok = tqg.evaluate(list(tqg.EXPECTED_NODE_IDS), 0, False, True, None, {}, {})
    assert ok["stage"] == "collection" and ok["valid"] is None  # passes collection; execution not yet evaluated
    src = (HERE / "exp13_tq_gate.py").read_text(encoding="utf-8")
    assert src.index('return 3') < src.index('run_cmd = [*BASE, "-rA"')  # stop happens before the test run


def test_tq_gate_parsers():
    assert tqg.parse_summary("== 121 passed, 2 skipped, 14 warnings in 12.34s ==")["passed"] == 121
    c = tqg.parse_summary("== 1 failed, 120 passed, 2 skipped, 1 error in 9.1s ==")
    assert (c["failed"], c["passed"], c["skipped"], c["errors"]) == (1, 120, 2, 1)
    items = tqg.parse_junit(_junit(_outcomes()))
    assert sorted(items) == sorted(tqg.EXPECTED_NODE_IDS) and sum(v["outcome"] == "skipped" for v in items.values()) == 2


def test_modal_diff_is_harness_only_and_attempt1_frozen():
    modal = (HERE / "exp13_deployment_modal.py").read_text(encoding="utf-8")
    assert "tq_cmd = [sys.executable, TQ_GATE_REMOTE]" in modal
    installs = [ln for ln in modal.splitlines() if "pip_install" in ln or "pip install" in ln or "apt_install" in ln]
    assert not any(("tblib" in ln or "scipy" in ln) for ln in installs)  # nothing added to the environment
    assert "tblib" not in modal
    assert r13.verify_equivalence()["image_expression_identical_to_exp4"]
    assert e1.run_git("diff", "--name-only", r13.ATTEMPT1_ARCHIVE_COMMIT, "--",
                      r13.ATTEMPT1_DIR.relative_to(ROOT).as_posix()) == ""
    rec = json.loads((r13.ATTEMPT1_DIR / "attempt_record.json").read_text(encoding="utf-8"))
    assert rec["status"] == "invalid_gate_harness_failure" and rec["excluded_from_accepted_results"] is True
    h = r13.load_protocol()["attempt_history"]
    assert h["no_cross_attempt_pooling"] and "zero measured benchmark legs executed" in h["attempt_1"]["facts"]
    tq_test = "vllm-kvquant/tests/quantization/test_turboquant.py"
    assert e1.run_git("show", f"HEAD:{tq_test}") == (ROOT / tq_test).read_text(encoding="utf-8").rstrip("\n")
    assert e1.run_git("diff", "--name-only", r13.BACKPORT_COMMIT, "HEAD", "--", tq_test) == ""  # tests unchanged
    assert e1.run_git("diff", "--name-only", r13.BACKPORT_COMMIT, "HEAD", "--", "vllm-kvquant") == ""

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
