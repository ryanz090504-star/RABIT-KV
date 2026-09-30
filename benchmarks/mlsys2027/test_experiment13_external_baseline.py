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


def _leg_lines(d, label, blocks, tpot, skip=None, backend="AttentionBackendEnum.TRITON_ATTN"):
    eff = {"block_size": 32, "kv_cache_dtype_skip_layers": skip or [], "attention_backend": backend,
           "flash_attn_version": 2 if d == r13.D else None, "gpu_memory_utilization": 0.82}
    req = {"block_size": 32, "kv_cache_dtype": d, **({} if d == r13.D else {"attention_config": {"backend": "TRITON_ATTN"}})}
    lines = [f"EXP13_REQUESTED_ENGINE_KWARGS={json.dumps(req)}", f"EXP13_EFFECTIVE_ENGINE_CONFIG={json.dumps(eff)}",
             f"EXP13_WORKLOAD={json.dumps({'context_tokens': 2048, 'output_tokens': 32})}",
             f"EXP13_KV_DTYPE={json.dumps({'engine_cache_dtype': d, 'requested_kv_cache_dtype': d})}",
             f"EXP13_CAPACITY={json.dumps({'num_gpu_blocks': blocks, 'block_size': 32, 'capacity_tokens': blocks * 32})}",
             "INFO Using V2 Model Runner", "INFO Available KV cache memory: 47.98 GiB"]
    if d == r13.C:
        lines.append('EXP13_RABIT_MARKERS={"a": true, "b": true, "c": true}')
    lines += [f"EXP13_WARMUP {json.dumps({'rep': i, 'prompt_tokens': 2048, 'output_tokens': 32, 'tpot_ms': tpot, 'ttft_ms': 50.0, 'wall_ms': 500.0})}" for i in range(5)]
    lines += [f"EXP13_SAMPLE {json.dumps({'rep': i, 'prompt_tokens': 2048, 'output_tokens': 32, 'tpot_ms': tpot + i * 0.01, 'ttft_ms': 50.0, 'wall_ms': 500.0})}" for i in range(30)]
    return [f"[leg{k}:{d}] {ln}" for k, lb, dd in r13.LEGS if lb == label for ln in lines]


def _session(tq_blocks_leg2=37600):
    blocks = {r13.A: 12282, r13.B: 24549, r13.C: 64831, r13.D: 37600}
    tp = {r13.A: 8.0, r13.B: 8.5, r13.C: 9.0, r13.D: 10.0}
    lines = ['EXP13_ENVIRONMENT={"gpus": [{"name": "NVIDIA H100 80GB HBM3"}]}', '[gate] EXP3_GATE_RESULT={"passed": true}',
             "EXP13_GATE_EXIT={\"returncode\": 0}", "[tqgate] 45 passed, 3 warnings in 12.00s",
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
    assert s["capacity"]["turboquant"]["observed_capacity_tokens"] == 1203200
    assert s["capacity"]["rabit"]["observed_capacity_tokens"] == 2074592
    assert all(s["latency"][c]["n"] == 60 for c in ("bf16", "fp8", "rabit", "turboquant"))
    assert abs(s["pairs"]["rabit_vs_turboquant"]["capacity_ratio"] - 2074592 / 1203200) < 1e-12
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
