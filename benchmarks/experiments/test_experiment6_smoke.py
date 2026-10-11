"""
Offline tests for the Experiment 6 concurrency correctness smoke harness (no GPU,
no torch, no pytest required: `python benchmarks/mlsys2027/test_experiment6_smoke.py`;
also collected by pytest).
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiment6_smoke as rs6  # noqa: E402

CFG = None


def cfg():
    global CFG
    if CFG is None:
        CFG = rs6.final_config()
    return CFG


def _rows(phase, outs, serial):
    rows = []
    for i in range(4):
        if serial:
            sched, first, last = 10.0 * i + 1, 10.0 * i + 2, 10.0 * i + 3
        else:
            sched, first, last = 1.0 + 0.01 * i, 2.0 + 0.01 * i, 5.0 + 0.01 * i
        ids = outs[i]
        rows.append({"i": i, "prompt_tokens": 2048, "prompt_token_ids_sha256": f"p{i}", "output_tokens": 32,
                     "output_token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                     "output_token_ids": ids, "finish_reason": "length",
                     "scheduled_ts": sched, "first_token_ts": first, "last_token_ts": last})
    return rows


def _worker_lines(dtype, single_outs=None, conc_outs=None, conc_serial=False, s3=None, jit_in=None, prompts=None):
    outs = single_outs or [[100 + i] * 32 for i in range(4)]
    conc = conc_outs or outs
    c = cfg()
    if s3 is None:
        s3 = ({"applicable": True, "requested_impl": c["impl"], "effective_impl": c["impl"],
               "requested_query_block": c["query_block"], "effective_query_block": c["query_block"],
               "env": {"VLLM_RABIT2_STAGE3C_IMPL": c["impl"], "VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK": str(c["query_block"])},
               "profiling_env": {"VLLM_RABIT2_STAGE3C_PROFILE": None, "VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE": None},
               "shared_decode_module_sha256": c["accepted_shared_decode_module_sha256"]}
              if dtype == "rabit_kv2" else
              {"applicable": False, "env": {"VLLM_RABIT2_STAGE3C_IMPL": None, "VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK": None},
               "profiling_env": {"VLLM_RABIT2_STAGE3C_PROFILE": None, "VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE": None}})
    base = rs6.rd._const(rs6.ast.parse(rs6.EXP5_WORKER.read_text(encoding="utf-8")), "BASE_ENGINE_KWARGS")
    tags = {
        "EXP6S_STAGE_IMPL": s3,
        "EXP6S_REQUESTED_ENGINE_KWARGS": {"model": "/m", **base, "max_num_seqs": 4, "kv_cache_dtype": dtype},
        "EXP6S_EFFECTIVE_ENGINE_CONFIG": {"max_num_seqs": 4, "block_size": 32, "enforce_eager": True,
                                          "attention_backend": "AttentionBackendEnum.TRITON_ATTN",
                                          "enable_prefix_caching": False, "log_stats": True},
        "EXP6S_KV_DTYPE": {"requested_kv_cache_dtype": dtype, "engine_cache_dtype": dtype},
        "EXP6S_CAPACITY": {"num_gpu_blocks": 10, "block_size": 32, "capacity_tokens": 320},
        "EXP6S_WORKLOAD": {"prompt_tokens": 2048, "output_tokens": 32, "concurrency": 4, "temperature": 0.0,
                           "ignore_eos": True, "test_prompt_sha256": prompts or [f"p{i}" for i in range(4)],
                           "warmup_prompt_sha256": [f"w{i}" for i in range(4)]},
    }
    lines = [f"{k}={json.dumps(v)}" for k, v in tags.items()]
    lines += ["EXP6S_WARMUP_BEGIN", "WARNING Triton kernel JIT compilation during inference: k", "EXP6S_WARMUP_END"]
    lines += ["EXP6S_SINGLE_BEGIN"] + [f"EXP6S_SINGLE={json.dumps(r)}" for r in _rows("single", outs, True)]
    if jit_in == "single":
        lines.append("WARNING Triton kernel JIT compilation during inference: k")
    lines += ["EXP6S_SINGLE_END", "EXP6S_CONCURRENT_BEGIN"]
    lines += [f"EXP6S_CONCURRENT={json.dumps(r)}" for r in _rows("concurrent", conc, conc_serial)]
    if jit_in == "concurrent":
        lines.append("WARNING Triton kernel JIT compilation during inference: k")
    return lines + ["EXP6S_CONCURRENT_END", "EXP6S_WORKER_COMPLETE"]


def _session(**kw):
    bsess = (rs6.rsd.rt.OUT_DIR / "modal_session.log").read_text(encoding="utf-8").splitlines()
    env = next(ln for ln in bsess if ln.startswith("S3C_ENVIRONMENT="))
    base = next(ln for ln in bsess if ln.startswith("S3C_GPU_BASELINE="))
    gate = [ln for ln in bsess if ln.startswith("[gate] ")]
    b = json.loads(base.split("=", 1)[1])
    lines = [base, env, 'S3C_GATE_START={"cmd": []}', *gate, 'S3C_GATE_EXIT={"returncode": 0}']
    for k, d in enumerate(rs6.DTYPES, start=1):
        pre = {"leg": f"smoke_{d}", "clean": True, "tolerance_mib": 256, "baseline_memory_used_mib": b["memory_used_mib"],
               "readings": [{"compute_apps": [], "memory_used_mib": b["memory_used_mib"]}]}
        lines += [f"S3C_PRE_LEG_GPU_STATE={json.dumps(pre)}", f'S3C_SERIES_START={json.dumps({"series": f"smoke_{d}"})}']
        dkw = kw.get(d, {})
        lines += [f"[smoke{k}:{d}] {ln}" for ln in _worker_lines(d, **dkw)]
        lines += [f'S3C_PROCESS_EXIT={json.dumps({"label": f"smoke_{d}", "returncode": 0, "timed_out": False, "group_processes_remaining": []})}',
                  f'S3C_SERIES_EXIT={json.dumps({"series": f"smoke_{d}", "returncode": 0})}']
    post = {"leg": "post_run", "clean": True, "tolerance_mib": 256, "baseline_memory_used_mib": b["memory_used_mib"],
            "readings": [{"compute_apps": [], "memory_used_mib": b["memory_used_mib"]}]}
    lines += [f"S3C_PRE_LEG_GPU_STATE={json.dumps(post)}", "S3C_SMOKE_COMPLETE"]
    return "\n".join(lines) + "\n"


def _failed(text):
    integ, _ = rs6.analyze(text, cfg(), write=False)
    return integ, [c["check"] for c in integ["checks"] if c["state"] != "passed"]


def test_valid_session_passes():
    integ, bad = _failed(_session())
    assert integ["all_ok"], bad


def test_equality_mismatch_detected_with_first_divergence():
    outs = [[100 + i] * 32 for i in range(4)]
    conc = copy.deepcopy(outs)
    conc[2][7] = 999
    integ, bad = _failed(_session(rabit_kv2={"single_outs": outs, "conc_outs": conc}))
    assert any("rabit_kv2: per-request output equality" in b and "3/4" in b for b in bad), bad
    obs = next(c["observed"] for c in integ["checks"] if c["check"].startswith("rabit_kv2: per-request output equality"))
    assert obs[2] == {"i": 2, "equal": False, "first_divergent_position": 7}


def test_swapped_requests_detected():
    outs = [[100 + i] * 32 for i in range(4)]
    swapped = [outs[1], outs[0], outs[2], outs[3]]
    integ, bad = _failed(_session(bfloat16={"single_outs": outs, "conc_outs": swapped}))
    assert any("bfloat16: per-request output equality" in b and "2/4" in b for b in bad), bad


def test_identical_outputs_fail_distinctness():
    same = [[5] * 32 for _ in range(4)]
    integ, bad = _failed(_session(bfloat16={"single_outs": same}))
    assert any("pairwise distinct" in b for b in bad), bad


def test_serialized_concurrent_phase_is_not_concurrency():
    integ, bad = _failed(_session(rabit_kv2={"conc_serial": True}))
    assert any("rabit_kv2: TRUE concurrency" in b for b in bad), bad


def test_selector_profiling_and_jit_violations():
    c = cfg()
    wrong = {"applicable": True, "requested_impl": c["impl"], "effective_impl": "reference",
             "requested_query_block": 32, "effective_query_block": 32,
             "env": {"VLLM_RABIT2_STAGE3C_IMPL": c["impl"], "VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK": "32"},
             "profiling_env": {"VLLM_RABIT2_STAGE3C_PROFILE": "1", "VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE": None},
             "shared_decode_module_sha256": c["accepted_shared_decode_module_sha256"]}
    _, bad = _failed(_session(rabit_kv2={"s3": wrong}))
    assert any("rabit_kv2: Stage3C selection" in b for b in bad) and any("rabit_kv2: profiling off" in b for b in bad)
    leaked = {"applicable": False, "env": {"VLLM_RABIT2_STAGE3C_IMPL": "shared_decode"}, "profiling_env": {}}
    _, bad = _failed(_session(bfloat16={"s3": leaked}))
    assert any("bfloat16: Stage3C selection" in b for b in bad)
    for phase in ("single", "concurrent"):
        _, bad = _failed(_session(bfloat16={"jit_in": phase}))
        assert any("bfloat16: no Triton JIT" in b for b in bad), phase
    # JIT during the unmeasured warmup is allowed (present in every synthetic session)


def test_prompt_mismatch_across_dtypes_detected():
    _, bad = _failed(_session(rabit_kv2={"prompts": ["p0", "p1", "p2", "q3"]}))
    assert any("identical test prompts and order for both dtypes" in b for b in bad)
    assert any("rabit_kv2: request i uses test prompt i" in b for b in bad)


def test_preflight_equivalence_and_grid():
    eq = rs6.verify_equivalence(cfg())
    assert eq["engine_kwargs_equal_frozen_exp5_except_max_num_seqs_4"] and eq["accepted_implementation_sources"]
    assert eq["modal_backstop_s"] > eq["watchdog_budget_s"] == 600 + 2 * 1800
    assert cfg()["impl"] == "shared_decode" and cfg()["query_block"] == 32
    cmd = " ".join(rs6.build_command(cfg()))
    assert "--stage3c-impl shared_decode --query-block 32" in cmd


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
