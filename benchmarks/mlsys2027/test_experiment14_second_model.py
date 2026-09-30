"""Offline tests for the Experiment 14 second-model harness (no GPU). Run directly or with pytest."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp14_engine_worker as w14  # noqa: E402
import exp14_shape_gate as sg  # noqa: E402
import run_experiment1_quality_frontier as e1  # noqa: E402
import run_experiment9_group_ablation as r9  # noqa: E402
import run_experiment12_variance as r12  # noqa: E402
import run_experiment14_second_model as r14  # noqa: E402

ROOT = e1.ROOT
RABIT2 = {"k_bits": 3, "v_bits": 2, "k_style": "seq_affine", "v_style": "group_affine", "k_group": 32, "v_group": 32,
          "residual": 4, "metadata_mode": "int8", "metadata_group_size": 64}


def page_bytes(h: int, d: int) -> int:
    """rabit2_page_layout(block_size=32) re-derived from kv_cache_interface.py (K3 / V2 / G32 / META8g64)."""
    k_payload = 32 * h * math.ceil(d * 3 / 8)
    v_payload = 32 * h * math.ceil(math.ceil(d / 32) * 32 * 2 / 8)
    meta = lambda n: n + math.ceil(n / 64) * 4  # noqa: E731
    return k_payload + v_payload + 2 * meta(h * d) + 2 * meta(32 * h * math.ceil(d / 32))


def test_frozen_policy_source_and_protocol():
    assert {k: r12.canonical_rabit2()[k] for k in RABIT2} == RABIT2
    src = (ROOT / "vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(src).hexdigest() == r14.r13.EXPECTED_RABIT_SHA256_LF
    assert r14.load_protocol()["policy"]["retuning"].startswith("none")
    assert r14.load_protocol()["compatibility_audit"]["source_change_required"] is False


def test_harness_equivalent_to_exp13_and_exp4():
    eq = r14.verify_equivalence()
    assert all(eq.values()), eq


def test_model_b_geometry_and_theoretical_bytes():
    g = r14.GEOMETRY
    assert (g["num_hidden_layers"], g["num_attention_heads"], g["num_kv_heads"], g["head_dim"]) == (28, 28, 4, 128)
    assert g["rabit_gqa4_decode_engaged"] is False and g["rabit_fast_append_engaged"] is False
    assert (28 // 4) % 4 != 0  # ratio 7 -> GQA4 fallback
    assert page_bytes(4, 128) == r14.RABIT_PAGE_BYTES == 12416
    assert page_bytes(8, 128) == 24832  # Llama page (Exp13 theoretical 32 x 776)
    assert r14.THEORETICAL_BYTES_PER_TOKEN[r14.A]["bytes"] == 57344
    assert r14.THEORETICAL_BYTES_PER_TOKEN[r14.B]["bytes"] == 10864
    # per-head-per-token cost is geometry-independent at head_dim 128 -> same theoretical ratio as Llama
    assert abs(57344 / 10864 - 131072 / 24832) < 1e-12


def test_shape_gate_constants():
    qb, lc = sg.GEOMETRIES["model_b_qwen2_5_7b"], sg.GEOMETRIES["control_llama3_1_8b"]
    assert (qb["q_heads"], qb["kv_heads"], qb["head_dim"]) == (28, 4, 128)
    assert (lc["q_heads"], lc["kv_heads"], lc["head_dim"]) == (32, 8, 128)
    for g in (qb, lc):
        assert g["expected_page_bytes"] == page_bytes(g["kv_heads"], g["head_dim"])
        assert g["expect_gqa4"] == ((g["q_heads"] // g["kv_heads"]) % 4 == 0)
        assert g["expect_fast_append"] == (g["kv_heads"] == 8 and g["head_dim"] == 128)
    assert sg.MAIN_PREFILL == w14.CONTEXT_TOKENS and sg.MAIN_DECODE_STEPS == w14.OUTPUT_TOKENS - 1
    assert sg.MAIN_PREFILL < w14.BASE_ENGINE_KWARGS["max_num_batched_tokens"]  # single initial prefill chunk
    assert sg.MAX_ABS_TOL == 5e-3
    stage3c = (ROOT / "vllm-kvquant/tests/quantization/test_rabit_kv2_stage3c.py").read_text(encoding="utf-8")
    assert "assert max_abs < 5e-3" in stage3c  # tolerance is the established one


def test_shape_gate_reference_helpers_match_stage3c_test():
    stage3c = ast.parse((ROOT / "vllm-kvquant/tests/quantization/test_rabit_kv2_stage3c.py").read_text(encoding="utf-8"))
    gate = ast.parse((HERE / "exp14_shape_gate.py").read_text(encoding="utf-8"))
    fn = lambda t, n: next(x for x in ast.walk(t) if isinstance(x, ast.FunctionDef) and x.name == n)  # noqa: E731
    body = lambda f: [ast.dump(s) for s in f.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]  # noqa: E731
    # _gqa_ref: identical statements except the extra `torch` parameter
    assert body(fn(gate, "_gqa_ref")) == body(fn(stage3c, "_gqa_ref"))
    assert body(fn(gate, "_materialize")) == body(fn(stage3c, "_materialize"))


def test_worker_prompt_rule():
    assert w14.build_prompt_ids(None, 279) == [279] * 2048
    assert w14.build_prompt_ids(128000, 279) == [128000] + [279] * 2047  # Exp13 rule for tokenizers with BOS
    assert w14.ALLOWED_KV_CACHE_DTYPES == ("bfloat16", "rabit_kv2")


def test_modal_app_structure():
    src = (HERE / "exp14_deployment_modal.py").read_text(encoding="utf-8")
    assert 'MODEL = "Qwen/Qwen2.5-7B-Instruct"' in src
    assert src.index("EXP14_GATE_START") < src.index("EXP14_SHAPE_GATE_START") < src.index("EXP14_LEG_START")
    assert "turboquant" not in src.lower() and "PROBE_WORKER_REMOTE" in src
    assert r14.legs_arg() == "A1=bfloat16,B1=rabit_kv2,B2=rabit_kv2,A2=bfloat16"
    probe = (HERE / "exp14_probe_worker.py").read_text(encoding="utf-8")
    code = ast.dump(ast.parse(probe))  # no timing code (docstring text aside)
    assert "perf_counter" not in code and "'time'" not in code and "metrics" not in code


def test_storage_model_equals_r9_at_llama_geometry():
    for t in (1, 3, 4, 5, 31, 36, 1023, 1024, 4095, 8191, 16191):
        for cfg in (RABIT2, r12.canonical_rabit2()):
            assert r14.traced_logical_bytes_geom(t, cfg, 32, 8, 128) == r9.traced_logical_bytes(t, cfg)


def test_storage_model_reproduces_accepted_exp12_logs_at_llama_geometry():
    cfg = r12.canonical_rabit2()
    for b in ("hotpotqa", "niah", "continuation_ppl"):
        rows = r14.pb.extract(b, (r12.OUT_DIR / f"{b}.log").read_text(encoding="utf-8", errors="replace"))
        for m in ("bf16", "rabit2"):
            for x in rows[m][:10]:
                t = 1024 if b == "continuation_ppl" else x["prefix_tokens"]
                exp = (32 * 2 * t * 8 * 128 * 2 if m == "bf16" else r14.traced_logical_bytes_geom(t, cfg, 32, 8, 128)["total"])
                assert abs(x["kv_mb"] - exp / 2**20) <= r14.STORAGE_TOL_MB + 1e-9, (b, m, x)


def test_quality_runs_and_guards():
    runs = r14.quality_runs()
    assert [r["name"] for r in runs] == list(r14.pb.BENCHMARKS)
    for r in runs:
        assert r["args"][-4:] == ["--methods", "bf16,rabit2", "--model-name", "Qwen/Qwen2.5-7B-Instruct"]
        assert r["args"][:-4] == r12.SELECTION[r["name"]]["args"]
    # the accepted Llama Exp12 log passes the identity checks but must FAIL the Model-B guards
    text = (r12.OUT_DIR / "hotpotqa.log").read_text(encoding="utf-8", errors="replace")
    res = r14.quality_integrity("hotpotqa", 0, text)
    c = res["checks"]
    assert c["unit_keys_identical_to_exp12"] and c["dataset_identity_identical_to_exp12"] and c["counts_exact"]
    assert not c["log_names_model_b"] and not c["per_unit_kv_mb_matches_model_b_accounting"] and not res["passed"]


def _synthetic_session(mutate=None, probe=False) -> str:
    legs = r14.PROBE_LEGS if probe else r14.LEGS
    tag = "EXP14P" if probe else "EXP14"
    top = ['EXP14_ENVIRONMENT=' + json.dumps({"gpus": [{"name": "NVIDIA H100 80GB HBM3"}]}),
           '[gate] EXP3_GATE_RESULT={"passed": true}', "EXP14_GATE_EXIT={\"returncode\": 0}",
           '[shapegate] EXP14_SHAPE_GATE_SUMMARY=' + json.dumps(
               {"passed": True, "geometries": {"model_b_qwen2_5_7b": {}, "control_llama3_1_8b": {}}}),
           'EXP14_SHAPE_GATE_EXIT={"returncode": 0}', 'EXP14_MODEL=' + json.dumps({"model": r14.MODEL_B})]
    for k, label, d in legs:
        pre = f"[leg{k}:{d}] "
        gib = 48.0
        cap = int(gib * 2**30 / r14.THEORETICAL_BYTES_PER_TOKEN[d]["bytes"]) // 32 * 32
        geom = dict(r14.GEOMETRY)
        lines = ["INFO Using V2 Model Runner", "INFO Using AttentionBackendEnum.TRITON_ATTN backend.",
                 f"INFO Available KV cache memory: {gib} GiB", f"INFO GPU KV cache size: {cap:,} tokens"]
        if probe:
            lines += [f"{tag}_MODEL_GEOMETRY=" + json.dumps(geom),
                      f"{tag}_KV=" + json.dumps({"engine_cache_dtype": d, "block_size": 32, "num_gpu_blocks": cap // 32,
                                                 "capacity_tokens": cap}),
                      f"{tag}_WORKLOAD_GENERATION=" + json.dumps({"prompt_tokens": 2048, "output_tokens": 32,
                                                                  "bos_token_id": None}),
                      f"{tag}_SANITY_GENERATION=" + json.dumps({"text": "Paris", "output_tokens": 2})]
        else:
            lines += [f"{tag}_RABIT_MARKERS=" + json.dumps({"a": True}),
                      f"{tag}_REQUESTED_ENGINE_KWARGS=" + json.dumps({"kv_cache_dtype": d}),
                      f"{tag}_EFFECTIVE_ENGINE_CONFIG=" + json.dumps({"block_size": 32, "kv_cache_dtype_skip_layers": []}),
                      f"{tag}_MODEL_GEOMETRY=" + json.dumps(geom),
                      f"{tag}_KV_DTYPE=" + json.dumps({"requested_kv_cache_dtype": d, "engine_cache_dtype": d}),
                      f"{tag}_CAPACITY=" + json.dumps({"num_gpu_blocks": cap // 32, "block_size": 32, "capacity_tokens": cap}),
                      f"{tag}_WORKLOAD=" + json.dumps({"prompt_rule": r14.PROMPT_RULE, "bos_token_id": None})]
            lines += [f"EXP14_WARMUP " + json.dumps({"rep": i, "prompt_tokens": 2048, "output_tokens": 32, "tpot_ms": 30.0,
                                                     "ttft_ms": 60.0, "wall_ms": 1000.0}) for i in range(5)]
            lines += ["EXP14_MEASUREMENT_BEGIN"] + [
                "EXP14_SAMPLE " + json.dumps({"rep": i, "prompt_tokens": 2048, "output_tokens": 32,
                                             "tpot_ms": 25.0 + i * 0.01 + (3 if d == r14.B else 0), "ttft_ms": 60.0,
                                             "wall_ms": 900.0}) for i in range(30)] + ["EXP14_MEASUREMENT_END"]
        if mutate:
            lines = mutate(label, lines)
        top.append(f'EXP14_PRE_LEG_GPU_STATE=' + json.dumps({"leg": label, "clean": True}))
        top += [pre + ln for ln in lines]
        top.append('EXP14_LEG_EXIT=' + json.dumps({"leg": label, "returncode": 0}))
    top.append("EXP14_MIRRORED_COMPLETE")
    return "\n".join(top) + "\n"


def test_serving_analysis_synthetic_pass_and_failures():
    res = r14.analyze_serving(_synthetic_session())
    assert res["integrity"]["passed"], {k: v for k, v in res["integrity"]["checks"].items() if not v}
    s = res["summary"]
    assert abs(s["capacity_ratio_rabit_over_bf16"] - 57344 / 10864) < 0.01
    assert s["rabit_vs_bf16"]["tpot_abs_diff_ms"] > 2.9 and s["latency_secondary_pooled_within_session_descriptive"]["bf16"]["n"] == 60

    def bad_geometry(label, lines):
        return [ln.replace('"num_kv_heads": 4', '"num_kv_heads": 8') for ln in lines]

    def bad_backend(label, lines):
        return [ln.replace("TRITON_ATTN", "FLASH_ATTN") for ln in lines]

    def bf16_fallback_capacity(label, lines):  # RABIT legs silently storing BF16-sized pages
        if label.startswith("B"):
            cap = int(48.0 * 2**30 / 57344) // 32 * 32
            out = []
            for ln in lines:
                if ln.startswith("EXP14_CAPACITY="):
                    ln = "EXP14_CAPACITY=" + json.dumps({"num_gpu_blocks": cap // 32, "block_size": 32, "capacity_tokens": cap})
                out.append(ln)
            return out
        return lines

    def bos_prompt(label, lines):
        return [ln.replace('"bos_token_id": null', '"bos_token_id": 1') for ln in lines]

    for mut, key in ((bad_geometry, "A1_geometry"), (bad_backend, "A1_backend_evidence"),
                     (bf16_fallback_capacity, "B1_physical_layout_consistent"), (bos_prompt, "A1_prompt_rule")):
        r = r14.analyze_serving(_synthetic_session(mut))
        assert not r["integrity"]["passed"] and not r["integrity"]["checks"][key], key


def test_probe_analysis_synthetic():
    ok = r14.analyze_probe(_synthetic_session(probe=True))
    assert ok["passed"], {k: v for k, v in ok["checks"].items() if not v}
    bad = r14.analyze_probe(_synthetic_session(probe=True, mutate=lambda l, ls: [x for x in ls if "V2 Model Runner" not in x]))
    assert not bad["passed"] and not bad["checks"]["P1_v2_runner"]


def test_shape_gate_failure_blocks_serving():
    text = _synthetic_session().replace('"passed": true, "geometries"', '"passed": false, "geometries"')
    assert not r14.analyze_serving(text)["integrity"]["checks"]["shape_gate_passed"]


def test_protected_evidence_unchanged():
    assert e1.run_git("diff", "--name-only", r14.EXP13_EVIDENCE_COMMIT, "--",
                      r14.EXP13_OUT.relative_to(ROOT).as_posix()) == ""
    assert e1.run_git("diff", "--name-only", r14.r13.EXP12_EVIDENCE_COMMIT, "--",
                      r12.OUT_DIR.relative_to(ROOT).as_posix()) == ""
    assert e1.run_git("diff", "--name-only", r14.r13.BACKPORT_COMMIT, "HEAD", "--", "vllm-kvquant") == ""


def test_exp13_wording_preserved():
    w = r14.load_protocol()["exp13_correctness_wording_for_paper"]
    assert w.startswith("The upstream TurboQuant suite passed under the frozen item-level gate.")
    assert "physical feasibility probe" in w and "round-trip" not in w


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
