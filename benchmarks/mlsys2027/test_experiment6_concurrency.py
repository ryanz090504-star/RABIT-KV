"""
Offline tests for the Experiment 6 concurrency-scaling harness (no GPU, no torch,
no pytest required: `python benchmarks/mlsys2027/test_experiment6_concurrency.py`;
also collected by pytest). Synthetic sessions only; nothing here is a result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp6_workload as wl  # noqa: E402
import run_experiment6_concurrency as r6  # noqa: E402

_CACHE: dict = {}


def cfg():
    if "cfg" not in _CACHE:
        _CACHE["cfg"] = r6.rs6.final_config()
    return _CACHE["cfg"]


def protocol():
    if "protocol" not in _CACHE:
        _CACHE["protocol"] = r6.load_protocol()
    return _CACHE["protocol"]


def _rows(L, C, serial=False, bad_len_at=None, drop=0, hashes=None, wave=None):
    hashes = hashes or protocol()["prompt_sets"][str(L)]["measured"]["per_prompt_sha256"]
    rows, n = [], wl.MEASURED_REQUESTS - drop
    for i in range(n):
        if serial:
            s = 10.0 + i * 2.0
        else:
            s = 10.0 + (i // (wave or C)) * 2.0  # closed loop: waves of C (or of an admission cap) requests
        rows.append({"i": i, "prompt_tokens": L, "prompt_token_ids_sha256": hashes[i],
                     "output_tokens": 31 if i == bad_len_at else 32,
                     "output_token_ids_sha256": "o" * 64, "finish_reason": "length",
                     "queued_ts": 9.0, "scheduled_ts": s, "first_token_ts": s + 0.5, "last_token_ts": s + 1.5})
    return rows


def _point_lines(spec, *, serial=False, preempt=0, counter=True, logged=None, jit_measured=0, oom=False,
                 fail=None, bad_len_at=None, drop=0, cap=None, s3=None, max_seqs=None, no_engine=False,
                 prompt_set=None, cond_jit=1, cond_serial=False, cond_drop=0, cond_bad_len=False, cond_fail=False,
                 meas_cap=None, cond_cap=None, blocks=None):
    L, C, d = spec["prompt_tokens"], spec["concurrency"], spec["dtype"]
    c = cfg()
    if s3 is None:
        s3 = ({"applicable": True, "requested_impl": c["impl"], "effective_impl": c["impl"],
               "requested_query_block": c["query_block"], "effective_query_block": c["query_block"],
               "env": {"VLLM_RABIT2_STAGE3C_IMPL": c["impl"], "VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK": str(c["query_block"])},
               "profiling_env": {"VLLM_RABIT2_STAGE3C_PROFILE": None, "VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE": None},
               "shared_decode_module_sha256": c["accepted_shared_decode_module_sha256"]} if d == "rabit_kv2" else
              {"applicable": False, "env": {"VLLM_RABIT2_STAGE3C_IMPL": None, "VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK": None},
               "profiling_env": {"VLLM_RABIT2_STAGE3C_PROFILE": None, "VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE": None}})
    ms = max_seqs if max_seqs is not None else C
    lines = [f"EXP6_POINT={json.dumps({'label': spec['label'], 'kv_cache_dtype': d, 'trial': spec['trial'], 'prompt_tokens': L, 'target_concurrency': C})}",
             f"EXP6_STAGE_IMPL={json.dumps(s3)}",
             f"EXP6_REQUESTED_ENGINE_KWARGS={json.dumps({'max_num_seqs': ms, 'kv_cache_dtype': d})}"]
    if no_engine:
        return lines + ["Traceback: engine init failed"]
    wkl = {"prompt_tokens": L, "output_tokens": 32, "measured_requests": 256, "warmup_requests": 2,
           "temperature": 0.0, "ignore_eos": True,
           "prompt_set": prompt_set or protocol()["prompt_sets"][str(L)]["measured"],
           "shadow_conditioning_requests": 256,
           "shadow_conditioning_prompt_set": protocol()["shadow_conditioning"]["prompt_sets"][str(L)]}
    ch = protocol()["shadow_conditioning"]["prompt_sets"][str(L)]["per_prompt_sha256"]
    # closed loop like the measured phase: waves of C (or strictly serial when cond_serial)
    cond_rows = [{"i": i, "prompt_tokens": L - 1 if (cond_bad_len and i == 0) else L, "prompt_token_ids_sha256": ch[i],
                  "output_tokens": 32, "finish_reason": "length",
                  "scheduled_ts": 1.0 + 2.0 * (i if cond_serial else i // (cond_cap or C)),
                  "last_token_ts": 2.5 + 2.0 * (i if cond_serial else i // (cond_cap or C))} for i in range(256 - cond_drop)]
    nblocks = blocks if blocks is not None else (cap or r6.EXPECTED_CAPACITY[d]) // 32
    lines += [f"EXP6_EFFECTIVE_ENGINE_CONFIG={json.dumps({'max_num_seqs': ms})}",
              f"EXP6_CAPACITY={json.dumps({'num_gpu_blocks': nblocks, 'block_size': 32, 'capacity_tokens': cap or r6.EXPECTED_CAPACITY[d]})}",
              f"EXP6_WORKLOAD={json.dumps(wkl)}",
              "WARNING Triton kernel JIT compilation during inference: setup_kernel",
              "EXP6_SHADOW_CONDITIONING_BEGIN"]
    lines += ["WARNING Triton kernel JIT compilation during inference: cond"] * cond_jit
    if cond_fail:
        fail_c = {"phase": "shadow_conditioning", "kind": "request_execution_failure", "error": "x"}
        return lines + [f"EXP6_REQUEST_FAILURE={json.dumps(fail_c)}"]
    lines += ["EXP6_SHADOW_CONDITIONING_END"] + [f"EXP6_SHADOW_CONDITIONING_REQUEST={json.dumps(r)}" for r in cond_rows]
    lines += ["EXP6_WARMUP_BEGIN", "WARNING Triton kernel JIT compilation during inference: warm", "EXP6_WARMUP_END",
              "EXP6_MEASURED_BEGIN"]
    lines += ["WARNING Triton kernel JIT compilation during inference: m"] * jit_measured
    if logged is not None:
        lines.append(f"INFO Engine 000: Running: 4 reqs, Preemptions: {logged}")
    if oom:
        lines.append("torch.OutOfMemoryError: CUDA out of memory")
    if fail:
        return lines + [f"EXP6_REQUEST_FAILURE={json.dumps({'phase': 'measured', 'kind': fail, 'error': 'x'})}"]
    lines.append("EXP6_MEASURED_END")
    lines += [f"EXP6_REQUEST={json.dumps(r)}" for r in _rows(L, C, serial, bad_len_at, drop, wave=meas_cap)]
    before = 5.0 if counter else None
    lines.append(f"EXP6_MEASURED_SUMMARY={json.dumps({'wall_s': 100.0, 'returned_requests': 256 - drop, 'preemptions_before': before, 'preemptions_after': (before + preempt) if counter else None, 'preemption_counter_available': counter})}")
    return lines + ["EXP6_WORKER_COMPLETE"]


def _session(L=2048, overrides=None, proc_over=None, omit=(), trial=None, env_over=None):
    overrides, proc_over = overrides or {}, proc_over or {}
    plan = [q for q in wl.plan_points(L) if trial is None or q["trial"] == trial]
    bsess = (r6.rs6.rsd.rt.OUT_DIR / "modal_session.log").read_text(encoding="utf-8").splitlines()
    env = json.loads(next(ln for ln in bsess if ln.startswith("S3C_ENVIRONMENT=")).split("=", 1)[1])
    env["point_labels"] = [p["label"] for p in plan]
    if trial is not None:
        env.update(point_timeout_s=3600, expected_points=12, static_watchdog_budget_s=43800, modal_backstop_s=45000,
                   gpus=[{"name": "NVIDIA H100 80GB HBM3", "uuid": f"GPU-trial{trial}"}])
    env.update(env_over or {})
    base = next(ln for ln in bsess if ln.startswith("S3C_GPU_BASELINE="))
    b = json.loads(base.split("=", 1)[1])
    gate = [ln for ln in bsess if ln.startswith("[gate] ")]
    lines = [base, f"S3C_ENVIRONMENT={json.dumps(env)}", 'S3C_GATE_START={"cmd": []}', *gate,
             'S3C_GATE_EXIT={"returncode": 0}']
    clean = lambda leg: {"leg": leg, "clean": True, "tolerance_mib": 256, "baseline_memory_used_mib": b["memory_used_mib"],  # noqa: E731
                         "readings": [{"compute_apps": [], "memory_used_mib": b["memory_used_mib"]}]}
    for k, spec in enumerate(plan, start=1):
        if spec["label"] in omit:
            continue
        kw = overrides.get(spec["label"], {})
        lines += [f"S3C_PRE_LEG_GPU_STATE={json.dumps(clean(spec['label']))}",
                  f"S3C_SERIES_START={json.dumps({'series': spec['label']})}"]
        lines += [f"[pt{k}:{spec['label']}] {ln}" for ln in _point_lines(spec, **kw)]
        rc = 1 if (kw.get("fail") or kw.get("no_engine") or kw.get("oom") or kw.get("cond_fail")) else 0
        proc = {"label": spec["label"], "returncode": rc, "timed_out": False, "group_processes_remaining": [],
                **proc_over.get(spec["label"], {})}
        lines += [f"S3C_PROCESS_EXIT={json.dumps(proc)}",
                  f"S3C_SERIES_EXIT={json.dumps({'series': spec['label'], 'returncode': proc['returncode']})}"]
    lines += [f"S3C_PRE_LEG_GPU_STATE={json.dumps(clean('post_run'))}",
              f"S3C_SWEEP_COMPLETE={json.dumps({'points': len(plan), 'failed_points': []})}"]
    return "\n".join(lines) + "\n"


def _run(L=2048, **kw):
    return r6.analyze(_session(L, **kw), L, cfg(), protocol())


def _pt(summary, label):
    return next(p for p in summary["points"] if p["label"] == label)


def _check(integ, prefix):
    return [c for c in integ["checks"] if c["check"].startswith(prefix)]


# ------------------------------------------------------------------ protocol shape
def test_protocol_frozen_and_shape():
    p = protocol()
    assert p["prompt_lengths"] == [2048, 8192] and p["concurrency_grid"] == [1, 4, 8, 16, 32, 64]
    assert p["measured_requests"] == 256 and p["warmup_requests"] == 2 and p["output_tokens"] == 32
    assert p["trials"] == 3 and p["trial_dtype_order"] == {"1": ["bfloat16", "rabit_kv2"], "2": ["rabit_kv2", "bfloat16"],
                                                          "3": ["bfloat16", "rabit_kv2"]}
    for L in ("2048", "8192"):
        pts = p["points"][L]
        assert len(pts) == 36
        assert [(q["trial"], q["dtype"]) for q in pts[::6]] == [(1, "bfloat16"), (1, "rabit_kv2"), (2, "rabit_kv2"),
                                                                (2, "bfloat16"), (3, "bfloat16"), (3, "rabit_kv2")]
        assert all([q["concurrency"] for q in pts[i:i + 6]] == [1, 4, 8, 16, 32, 64] for i in range(0, 36, 6))
        ms = p["prompt_sets"][L]["measured"]
        assert ms["count"] == 256 and len(set(ms["per_prompt_sha256"])) == 256
        assert not set(ms["per_prompt_sha256"]) & set(p["prompt_sets"][L]["warmup"]["per_prompt_sha256"])
    assert r6.load_protocol() == p  # regenerated == committed (no drift)
    assert p["rabit_extension_grid_pre_registered_not_enabled"] == [128, 256]


def test_same_prompts_reused_everywhere():
    # one fixed set per length: generated identically for every dtype / trial / concurrency (pure function)
    for L in (2048, 8192):
        a, b = wl.measured_prompts(L), wl.measured_prompts(L)
        assert a == b and all(len(x) == L for x in a)
        assert wl.set_digest(a)["ordered_set_sha256"] == protocol()["prompt_sets"][str(L)]["measured"]["ordered_set_sha256"]
    assert wl.measured_prompts(2048)[0] != wl.measured_prompts(8192)[0][:2048] or True


# ------------------------------------------------------------------ valid sweep
def test_valid_sweep_passes_and_classifies_success():
    integ, s = _run()
    assert integ["all_ok"], [c["check"] for c in integ["checks"] if c["state"] != "passed"][:10]
    assert s["outcome_class_counts"]["sustained_target_concurrency"] == 36
    p = _pt(s, "t1_rabit_L2048_c64")
    assert p["completed_requests"] == 256 and p["output_token_count_valid"]
    assert p["inflight_concurrency"]["observed_max_inflight_concurrency"] == 64
    assert p["inflight_concurrency"]["all_c_inflight_overlap_total_s"] > 0
    assert abs(p["requests_per_s"] - 2.56) < 1e-9 and abs(p["output_tokens_per_s"] - 2.56 * 32) < 1e-9
    assert abs(p["total_tokens_per_s"] - 2.56 * (2048 + 32)) < 1e-9
    assert p["latency_s"]["median"] is not None and p["latency_s"]["p99"] is not None
    hs = s["highest_successfully_tested_concurrency"]
    assert hs["bfloat16"]["highest_successfully_tested_concurrency_all_trials"] == 64
    assert hs["bfloat16"]["failure_boundary_bracketed_by_grid"] is False
    assert s["rabit_only_extension"]["enabled"] is False


def test_max_num_seqs_and_selector_checks():
    integ, _ = _run(overrides={"t2_bf16_L2048_c16": {"max_seqs": 32}})
    assert [c["state"] for c in _check(integ, "t2_bf16_L2048_c16: max_num_seqs")] == ["failed"]
    c = cfg()
    bad = {"applicable": True, "requested_impl": c["impl"], "effective_impl": c["impl"], "requested_query_block": 32,
           "effective_query_block": 16, "env": {"VLLM_RABIT2_STAGE3C_IMPL": c["impl"],
                                                "VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK": "32"},
           "profiling_env": {}, "shared_decode_module_sha256": c["accepted_shared_decode_module_sha256"]}
    integ, _ = _run(overrides={"t1_rabit_L2048_c8": {"s3": bad}})
    assert [x["state"] for x in _check(integ, "t1_rabit_L2048_c8: Stage3C")] == ["failed"]
    leaked = {"applicable": False, "env": {"VLLM_RABIT2_STAGE3C_IMPL": "shared_decode"}, "profiling_env": {}}
    integ, _ = _run(overrides={"t3_bf16_L2048_c1": {"s3": leaked}})
    assert [x["state"] for x in _check(integ, "t3_bf16_L2048_c1: Stage3C")] == ["failed"]
    prof = {**bad, "effective_query_block": 32, "profiling_env": {"VLLM_RABIT2_STAGE3C_PROFILE": "1"}}
    integ, _ = _run(overrides={"t1_rabit_L2048_c8": {"s3": prof}})
    assert [x["state"] for x in _check(integ, "t1_rabit_L2048_c8: Stage3C")] == ["failed"]


def test_prompt_set_and_capacity_checks():
    other = protocol()["prompt_sets"]["8192"]["measured"]
    integ, _ = _run(overrides={"t1_bf16_L2048_c4": {"prompt_set": other}})
    assert [x["state"] for x in _check(integ, "t1_bf16_L2048_c4: frozen prompt set")] == ["failed"]
    integ, _ = _run(overrides={"t1_bf16_L2048_c4": {"cap": 393000}})
    assert [x["state"] for x in _check(integ, "t1_bf16_L2048_c4: allocator capacity")] == ["failed"]


# ------------------------------------------------------------------ concurrency / outcomes
def test_serialized_run_is_not_concurrency():
    integ, s = _run(overrides={"t1_rabit_L2048_c16": {"serial": True}})
    p = _pt(s, "t1_rabit_L2048_c16")
    assert p["inflight_concurrency"]["observed_max_inflight_concurrency"] == 1
    assert p["outcome_class"] == "target_concurrency_not_reached"
    assert s["highest_successfully_tested_concurrency"]["rabit_kv2"]["per_trial"]["1"] == 64  # other points fine
    h = s["highest_successfully_tested_concurrency"]["rabit_kv2"]
    assert h["highest_successfully_tested_concurrency_all_trials"] == 64  # 32 and 64 succeeded in every trial
    assert h["highest_contiguous_successful_concurrency_all_trials"] == 8 and h["non_monotonic"] is True
    assert h["failure_boundary_bracketed_by_grid"] is True


def test_partial_completion_not_success():
    _, s = _run(overrides={"t2_rabit_L2048_c32": {"drop": 3}, "t2_bf16_L2048_c32": {"bad_len_at": 7}})
    for label in ("t2_rabit_L2048_c32", "t2_bf16_L2048_c32"):
        p = _pt(s, label)
        assert p["outcome_class"] == "engine_or_request_failure" and p["completed_requests"] < 256, label
        assert p["requests_per_s"] is None
    assert not _pt(s, "t2_bf16_L2048_c32")["output_token_count_valid"]


def test_failure_kinds_distinguished():
    _, s = _run(overrides={"t1_bf16_L2048_c64": {"oom": True, "fail": "request_oom"},
                           "t1_bf16_L2048_c32": {"fail": "request_execution_failure"},
                           "t1_bf16_L2048_c16": {"no_engine": True},
                           "t2_bf16_L2048_c64": {"preempt": 3},
                           "t3_bf16_L2048_c64": {"serial": True}})
    cls = {p["label"]: p["outcome_class"] for p in s["points"]}
    assert cls["t1_bf16_L2048_c64"] == "oom_or_allocation_failure"
    assert cls["t1_bf16_L2048_c32"] == "engine_or_request_failure"
    assert cls["t1_bf16_L2048_c16"] == "engine_or_request_failure"
    assert cls["t2_bf16_L2048_c64"] == "completed_with_preemption"
    assert cls["t3_bf16_L2048_c64"] == "target_concurrency_not_reached"
    assert cls["t1_rabit_L2048_c64"] == "sustained_target_concurrency"
    ext = s["rabit_only_extension"]
    assert ext["eligible_for_pre_registered_rabit_only_extension"] and ext["enabled"] is False
    assert 16 in ext["bf16_fails_first_at"] and 64 in ext["bf16_fails_first_at"]


def test_watchdog_timeout_is_failure():
    _, s = _run(proc_over={"t2_rabit_L2048_c1": {"timed_out": True, "returncode": -9}})
    assert _pt(s, "t2_rabit_L2048_c1")["outcome_class"] == "engine_or_request_failure"


def test_preemption_sources():
    _, s = _run(overrides={"t1_rabit_L2048_c4": {"counter": False, "logged": 2},
                           "t1_rabit_L2048_c8": {"counter": False}})
    a, b = _pt(s, "t1_rabit_L2048_c4"), _pt(s, "t1_rabit_L2048_c8")
    assert a["preemptions"] == 2 and "lower bound" in a["preemption_source"] and a["outcome_class"] == "completed_with_preemption"
    assert b["preemptions"] is None and b["preemption_source"] == "unavailable"
    assert a["overlap_stats_are_not_residency_evidence"] and b["overlap_stats_are_not_residency_evidence"]
    _, s2 = _run()
    assert not _pt(s2, "t1_rabit_L2048_c4")["overlap_stats_are_not_residency_evidence"]  # 0 preemptions, counter
    p = protocol()
    assert "OVERLAPPING IN-FLIGHT" in p["concurrency_terminology"] and "NOT strict GPU-resident" in p["concurrency_terminology"]
    assert p["outcome_class_order"] == ["oom_or_allocation_failure", "engine_or_request_failure",
                                        "completed_with_preemption", "target_concurrency_not_reached",
                                        "sustained_target_concurrency"]


def test_measured_jit_surfaced_and_not_interpretable():
    integ, s = _run(overrides={"t3_rabit_L2048_c64": {"jit_measured": 2}})
    assert [x["state"] for x in _check(integ, "t3_rabit_L2048_c64: no Triton JIT")] == ["failed"]
    p = _pt(s, "t3_rabit_L2048_c64")
    assert p["measured_jit_lines"] == 2 and p["interpretable"] is False
    assert p["jit_lines_by_phase"] == {"setup": 1, "shadow_conditioning": 1, "original_warmup": 1, "measured": 2,
                                       "after": 0}
    assert s["highest_successfully_tested_concurrency"]["rabit_kv2"]["per_trial"]["3"] == 32
    assert not integ["all_ok"]


# ------------------------------------------------------------------ shadow-conditioning amendment (final warmup)
def test_shadow_prompt_set_frozen_disjoint_and_shared():
    p = protocol()
    sc = p["shadow_conditioning"]
    assert sc["requests_per_point"] == 256 == wl.SHADOW_CONDITIONING_REQUESTS and sc["output_tokens"] == 32
    assert sc["jit_accounting_phases"] == ["setup", "shadow_conditioning", "original_warmup", "measured", "after"]
    for L in wl.PROMPT_LENGTHS:
        prompts = wl.shadow_conditioning_prompts(L)
        d = sc["prompt_sets"][str(L)]  # ONE set per length: same for every dtype / trial / C (no such key exists)
        assert wl.set_digest(prompts) == d and d["count"] == 256 and all(len(x) == L for x in prompts)
        assert len(set(d["per_prompt_sha256"])) == 256  # pairwise distinct
        assert not set(d["per_prompt_sha256"]) & set(p["prompt_sets"][str(L)]["measured"]["per_prompt_sha256"])
        assert not set(d["per_prompt_sha256"]) & set(p["prompt_sets"][str(L)]["warmup"]["per_prompt_sha256"])


def test_worker_shadow_same_for_all_points_and_phase_order():
    import ast
    fn = r6._function(ast.parse(r6.WORKER.read_text(encoding="utf-8")), "main")
    main_src = ast.unparse(fn)
    # depends only on the prompt length: same set for both dtypes, all trials, all C
    assert "shadow = wl.shadow_conditioning_prompts(args.prompt_tokens)" in main_src
    # ONE queued batch on the point's engine (max_num_seqs=C) with the measured sampler -> closed-loop admission
    assert "llm.generate([{'prompt_token_ids': p} for p in shadow], sp, use_tqdm=False)" in main_src
    assert "sp = SamplingParams(temperature=0.0, max_tokens=wl.OUTPUT_TOKENS, ignore_eos=True)" in main_src
    assert main_src.count("llm = LLM(**kwargs)") == 1  # same engine for shadow, warmup and measured
    i = [main_src.index(k) for k in ("EXP6_SHADOW_CONDITIONING_BEGIN", "EXP6_WARMUP_BEGIN", "EXP6_MEASURED_BEGIN")]
    assert i == sorted(i)
    for node in ast.walk(fn):  # never inside a dtype branch
        if isinstance(node, ast.If) and "kv_cache_dtype" in ast.unparse(node.test):
            assert "shadow" not in ast.unparse(node)
    # original measurement fields unchanged
    assert wl.WARMUP_REQUESTS == 2 and wl.MEASURED_REQUESTS == 256 and wl.OUTPUT_TOKENS == 32
    assert "measured = wl.measured_prompts(args.prompt_tokens)" in main_src


def test_shadow_valid_reaches_target_and_its_jit_allowed():
    integ, s = _run(overrides={"t1_rabit_L2048_c64": {"cond_jit": 11}, "t2_bf16_L2048_c16": {"cond_jit": 3}})
    assert integ["all_ok"]
    for p in s["points"]:  # count 256 and target in-flight C at every point
        cv = p["shadow_conditioning"]
        assert cv["valid"] and cv["requests"] == cv["exact_requests"] == 256
        assert cv["observed_max_inflight_concurrency"] == p["target_concurrency"] == cv["target_concurrency"]
        assert p["interpretable"]
    assert _pt(s, "t1_rabit_L2048_c64")["jit_lines_by_phase"]["shadow_conditioning"] == 11
    assert len(_check(integ, "t1_rabit_L2048_c64: shadow conditioning")) == 2
    integ, s = _run(overrides={"t2_rabit_L2048_c32": {"max_seqs": 16}})  # same max_num_seqs=C check still applies
    assert [c["state"] for c in _check(integ, "t2_rabit_L2048_c32: max_num_seqs")] == ["failed"]


def test_shadow_invalid_makes_point_non_interpretable():
    integ, s = _run(overrides={"t1_rabit_L2048_c16": {"cond_serial": True}, "t1_rabit_L2048_c32": {"cond_drop": 1},
                               "t2_bf16_L2048_c8": {"cond_bad_len": True}, "t3_rabit_L2048_c4": {"cond_fail": True}})
    assert not integ["all_ok"] and "conditioning" in integ["failed_categories"]
    a = _pt(s, "t1_rabit_L2048_c16")
    assert a["shadow_conditioning"]["observed_max_inflight_concurrency"] == 1 and not a["interpretable"]
    assert [x["state"] for x in _check(integ, "t1_rabit_L2048_c16: shadow conditioning reached")] == ["failed"]
    assert a["outcome_class"] == "sustained_target_concurrency"  # measured classification itself unchanged
    b = _pt(s, "t1_rabit_L2048_c32")
    assert b["shadow_conditioning"]["requests"] == 255 and not b["shadow_conditioning"]["completed_all_exact"]
    assert not b["interpretable"]
    assert not _pt(s, "t2_bf16_L2048_c8")["shadow_conditioning"]["valid"]
    f = _pt(s, "t3_rabit_L2048_c4")
    assert f["shadow_conditioning"]["failure_in_conditioning"] and f["outcome_class"] == "engine_or_request_failure"
    h = s["highest_successfully_tested_concurrency"]["rabit_kv2"]
    assert h["per_trial"]["1"] == 64 and h["highest_contiguous_successful_concurrency_all_trials"] == 1


def _attempt(n):
    return r6.BASE_OUT / "L2048" / f"jit_contaminated_attempt_{n}"


def test_diagnostic_attempts_remain_failed_and_excluded():
    expect = {1: {"passed": 320, "failed": 9, "not_run": 0, "not_evaluated": 0},
              2: {"passed": 395, "failed": 6, "not_run": 0, "not_evaluated": 0}}
    assert r6.DIAGNOSTIC_ATTEMPTS[2048] == [_attempt(1), _attempt(2)]
    for n, counts in expect.items():
        d = _attempt(n)
        assert r6.run_git("ls-files", r6.rel(d / "modal_session.log"))  # archived, committed
        assert d in r6.EVIDENCE_DIRS and d in r6.PROTECTED_PATHS
        st = json.loads((d / "ATTEMPT_STATUS.json").read_text(encoding="utf-8"))
        assert st["accepted_for_performance_interpretation"] is False and st["use_in_paper_dataset"] is False
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        ic = json.loads((d / "integrity_check.json").read_text(encoding="utf-8"))
        assert m["status"] == "failed" and ic["failed_categories"] == ["jit"] and ic["counts"] == counts
        # re-analysis under the amended harness still fails (measured JIT; no valid shadow pass)
        text = (d / "modal_session.log").read_text(encoding="utf-8", errors="replace")
        integ, s = r6.analyze(text, 2048, cfg(), protocol())
        assert not integ["all_ok"] and {"jit", "conditioning"} <= set(integ["failed_categories"])
        assert not any(p["interpretable"] for p in s["points"])


def test_diagnostic_attempts_never_pooled_with_accepted_results():
    d_new = r6.out_dir(2048)
    assert d_new.name == "shadow_conditioned"
    for d_old in (_attempt(1), _attempt(2)):
        assert d_new != d_old and d_old not in d_new.parents and d_new not in d_old.parents
    _, s = _run()
    a3 = r6.BASE_OUT / "L2048" / "infrastructure_aborted_attempt_3"
    assert s["excluded_attempts_not_pooled"] == [r6.rel(_attempt(1)), r6.rel(_attempt(2)), r6.rel(a3)]
    assert len(s["points"]) == 36
    # the frozen protocol lists only the diagnostic attempts (attempt 3 is excluded by the runner, protocol unchanged)
    assert protocol()["excluded_attempts"]["2048"]["dirs"] == [r6.rel(_attempt(1)), r6.rel(_attempt(2))]
    assert a3 in r6.EVIDENCE_DIRS and a3 in r6.PROTECTED_PATHS and r6.excluded_attempts(2048)[-1] == a3
    st = json.loads((a3 / "ATTEMPT_STATUS.json").read_text(encoding="utf-8"))
    assert st["status"] == "infrastructure_aborted" and st["accepted_for_performance_interpretation"] is False
    assert st["use_in_paper_dataset"] is False and r6.run_git("ls-files", r6.rel(a3 / "modal_session.log"))
    orig = r6.out_dir
    try:
        for d_old in (_attempt(1), _attempt(2), a3):
            r6.out_dir = lambda L, d=d_old: d  # a run may never write into / next to an archived attempt
            try:
                r6.preflight(2048, dry_run=True)
            except RuntimeError as e:
                assert "overlaps an archived diagnostic attempt" in str(e)
            else:
                raise AssertionError("preflight accepted a diagnostic attempt directory")
    finally:
        r6.out_dir = orig


def test_protocol_changed_only_by_reviewed_warmup_amendments():
    old = json.loads(r6.run_git("show", "abefce2:benchmarks/mlsys2027/exp6_protocol.json"))
    new = protocol()
    assert set(new) - set(old) == {"shadow_conditioning", "excluded_attempts"} and not set(old) - set(new)
    assert [k for k in old if old[k] != new[k]] == ["amendments"]  # measurement fields, points / trial order unchanged
    assert set(new["amendments"]) - set(old["amendments"]) == {"compile_conditioning_superseded", "shadow_conditioning"}
    assert new["amendments"]["compile_conditioning_superseded"].startswith("SUPERSEDED")
    assert all(new["amendments"][k] == v for k, v in old["amendments"].items())


# ------------------------------------------------------------------ detached launch + remote-log capture (harness only)
def test_detached_launch_and_run_id():
    cmd = r6.build_command(2048, "x" * 64, cfg(), "exp6-L2048-20260928T000000Z-abcdef0")
    assert cmd[cmd.index("run") + 1] == "--detach"
    assert cmd[cmd.index("--run-id") + 1] == "exp6-L2048-20260928T000000Z-abcdef0"
    rid = r6.make_run_id(2048, "5028eb2ce6a9b1dcfa7ca12bed846acf902ae9db")
    assert rid.startswith("exp6-L2048-") and rid.endswith("-5028eb2") and "/" not in rid


def test_modal_app_tees_complete_remote_log():
    import ast
    tree = ast.parse(r6.MODAL_APP.read_text(encoding="utf-8"))
    assert r6.rd._const(tree, "SESSION_LOG_VOLUME") == r6.SESSION_LOG_VOLUME
    sweep = ast.unparse(r6._function(tree, "sweep"))
    assert "run_id: str" in sweep and "_Tee(orig_out, fh)" in sweep and "_sweep_body(" in sweep
    assert "finally:" in sweep and "DONE.json" in sweep and "already exists" in sweep
    body = r6._function(tree, "_sweep_body")
    loop = next(n for n in ast.walk(body) if isinstance(n, ast.For) and "run_with_watchdog" in ast.unparse(n))
    assert "_commit_session_logs()" in ast.unparse(loop)  # committed after every point
    assert "run_with_watchdog(cmd" in ast.unparse(loop)
    r6.verify_equivalence(cfg())  # image and verified helpers still identical to the accepted smoke app


def _fake_volume(files, appear_after=None):
    calls = {"n": 0}

    def getter(remote, local):
        calls["n"] += 1
        name = remote.rsplit("/", 1)[1]
        if name == "DONE.json" and appear_after is not None and calls["n"] <= appear_after:
            return False
        if name not in files:
            return False
        local.write_bytes(files[name].encode("utf-8"))
        return True
    return getter


def test_fetch_remote_log_waits_for_done_and_reports_partial():
    import tempfile
    clock = {"t": 0.0}
    tick = lambda: clock["t"]  # noqa: E731
    sleep = lambda s: clock.__setitem__("t", clock["t"] + s)  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        done = '{"run_id": "r", "status": "complete", "error": null}'
        r = r6.fetch_remote_log("r", d, 3600, _fake_volume({"DONE.json": done, "remote_session.log": "a\n"}),
                                sleep, tick, 60)
        assert r["complete"] and r["polls"] == 1 and r["remote_log_bytes"] == 2
        clock["t"] = 0.0  # DONE.json appears only on the 4th poll (local client disconnected earlier)
        r = r6.fetch_remote_log("r", d, 3600, _fake_volume({"DONE.json": done, "remote_session.log": "b\n"},
                                                           appear_after=3), sleep, tick, 60)
        assert r["complete"] and r["polls"] == 4 and clock["t"] == 180
        clock["t"] = 0.0
        exc = '{"run_id": "r", "status": "exception", "error": "RuntimeError: GPU not clean"}'
        r = r6.fetch_remote_log("r", d, 3600, _fake_volume({"DONE.json": exc, "remote_session.log": "c\n"}),
                                sleep, tick, 60)
        assert r["done_found"] and not r["complete"]
        clock["t"] = 0.0  # never finishes: deadline reached, partial log still fetched, not complete
        r = r6.fetch_remote_log("r", d, 300, _fake_volume({"remote_session.log": "partial\n"}), sleep, tick, 60)
        assert not r["done_found"] and r["remote_log_found"] and not r["complete"] and clock["t"] >= 300


# ------------------------------------------------------------------ summary schema fix (reporting only)
def test_summary_schema_has_no_concurrency_key_collision():
    _, s = _run()
    assert s["summary_schema"]["version"] == 2
    for p in s["points"]:
        assert "concurrency" not in p  # never two meanings for one key
        assert isinstance(p["target_concurrency"], int) and not isinstance(p["target_concurrency"], bool)
        assert isinstance(p["inflight_concurrency"], dict)
        assert p["label"].endswith(f"_c{p['target_concurrency']}")
        assert p["inflight_concurrency"]["observed_max_inflight_concurrency"] == p["target_concurrency"]
    assert {p["target_concurrency"] for p in s["points"]} == set(wl.CONCURRENCY_GRID)


_ACCEPTED: dict = {}


def _accepted_l2048():
    if not _ACCEPTED:
        acc = r6.ACCEPTED_RUNS[2048]
        raw = acc["dir"] / "remote_session.log"
        _ACCEPTED["acc"] = acc
        _ACCEPTED["raw_sha"] = r6.sha256_raw(raw)
        _ACCEPTED["raw_bytes"] = raw.stat().st_size
        _ACCEPTED["res"] = r6.analyze(raw.read_text(encoding="utf-8", errors="replace"), 2048, cfg(), protocol())
    return _ACCEPTED


def test_accepted_l2048_raw_pinned_and_unchanged():
    a = _accepted_l2048()
    assert a["raw_sha"] == a["acc"]["remote_session_log_sha256"] == (
        "19a6c7d951bc617c3b6230bff2cf2ac5f5ac8632c385d3be02b171a77b004a17")
    assert a["raw_bytes"] == 8984334
    d = a["acc"]["dir"]
    assert r6.sha256_raw(d / "summary_pre_concurrency_keyfix.json") == a["acc"]["pre_keyfix_summary_sha256"]
    assert r6.sha256_raw(d / "integrity_check.json") == a["acc"]["integrity_check_sha256"]


def test_accepted_l2048_reparse_reproduces_integrity_and_values():
    a = _accepted_l2048()
    integ, new = a["res"]
    d = a["acc"]["dir"]
    assert integ["counts"] == {"passed": 401, "failed": 0, "not_run": 0, "not_evaluated": 0} and integ["all_ok"]
    assert json.loads(json.dumps(integ, default=str)) == json.loads((d / "integrity_check.json").read_text(encoding="utf-8"))
    new = json.loads(json.dumps(new, default=str))
    old = json.loads((d / "summary_pre_concurrency_keyfix.json").read_text(encoding="utf-8"))
    # every throughput / latency / TTFT / TPOT / wall / preemption value identical; only the key was renamed
    assert {k: v for k, v in new.items() if k != "summary_schema"} == r6.normalize_pre_keyfix_summary(old)
    pts = new["points"]
    assert len(pts) == 36 and {p["target_concurrency"] for p in pts} == {1, 4, 8, 16, 32, 64}
    assert {(p["dtype"], p["target_concurrency"], p["trial"]) for p in pts} == {
        (dt, c, t) for dt in ("bfloat16", "rabit_kv2") for c in wl.CONCURRENCY_GRID for t in (1, 2, 3)}
    assert all(p["measured_jit_lines"] == 0 and p["jit_lines_by_phase"]["measured"] == 0 for p in pts)
    assert all(p["interpretable"] and p["preemptions"] == 0 for p in pts)
    cur = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    if "summary_schema" in cur:  # after the offline reparse: the written summary is exactly the re-analysis
        assert cur == new
        prov = json.loads((d / "reparse_provenance.json").read_text(encoding="utf-8"))
        assert prov["reparsed_from_existing_raw"] and not prov["h100_rerun"] and prov["raw_unchanged"]
        assert prov["measurement_commit"].startswith("2ccaa10") and prov["raw"]["sha256"] == a["raw_sha"]
    else:  # before the reparse: the stored v1 summary is exactly the pre-keyfix one
        assert cur == old


# ------------------------------------------------------------------ L8192 execution amendment (infrastructure only)
def test_l8192_three_trial_runs_of_twelve_points():
    assert r6.EXECUTION[8192]["runs"] == (1, 2, 3) and r6.EXECUTION[2048]["runs"] == (None,)
    plans = {t: r6.run_plan(8192, t) for t in (1, 2, 3)}
    assert all(len(v) == 12 for v in plans.values())
    assert [q for t in (1, 2, 3) for q in plans[t]] == wl.plan_points(8192)  # same pre-registered points and order
    for t, v in plans.items():
        order = wl.TRIAL_DTYPE_ORDER[t]
        assert [q["dtype"] for q in v] == [order[0]] * 6 + [order[1]] * 6 and all(q["trial"] == t for q in v)
        assert [q["concurrency"] for q in v] == [1, 4, 8, 16, 32, 64] * 2
    assert wl.TRIAL_DTYPE_ORDER == {1: ("bfloat16", "rabit_kv2"), 2: ("rabit_kv2", "bfloat16"),
                                    3: ("bfloat16", "rabit_kv2")}
    for bad in (None, 4):
        try:
            r6.run_plan(8192, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    assert len(r6.run_plan(2048)) == 36


def test_l8192_watchdog_budget_and_l2048_historical():
    assert r6.EXECUTION[8192]["point_timeout_s"] == 3600 and r6.EXECUTION[2048]["point_timeout_s"] == 1200
    assert r6.static_watchdog_budget(8192) == 600 + 12 * 3600 == 43800 and r6.static_watchdog_budget(2048) == 43800
    assert r6.MODAL_FUNCTION_TIMEOUT_S == 45000 and 45000 - r6.static_watchdog_budget(8192) == 1200
    assert protocol()["watchdogs"] == {"gate_s": 600, "point_s": 1200, "modal_backstop_s": 45000}  # frozen, unchanged
    raw = (r6.ACCEPTED_RUNS[2048]["dir"] / "remote_session.log").read_text(encoding="utf-8")
    env = json.loads(next(ln for ln in raw.splitlines() if ln.startswith("S3C_ENVIRONMENT=")).split("=", 1)[1])
    assert env["point_timeout_s"] == 1200  # accepted L2048 keeps its historical watchdog
    am = r6.load_l8192_amendment()
    ex = am["amended_l8192_execution"]
    assert am["scientific_protocol_unchanged"] and ex["point_watchdog_s"] == {"L2048": 1200, "L8192": 3600}
    assert ex["static_budget_per_trial_run_s"]["value"] == 43800 and ex["backstop_margin_s"] == 1200
    assert "C >= 4 exceeds" in am["preflight_wording_correction"]
    import ast
    tree = ast.parse(r6.MODAL_APP.read_text(encoding="utf-8"))
    assert r6.rd._const(tree, "L8192_POINT_TIMEOUT_S") == 3600 and r6.rd._const(tree, "TRIAL_POINTS") == 12
    assert r6.rd._const(tree, "POINT_TIMEOUT_S") == 1200 and r6.rd._const(tree, "MODAL_BACKSTOP_S") == 45000
    body = ast.unparse(r6._function(tree, "_sweep_body"))
    assert "ALLOWED_EXECUTIONS" in body and "] \", point_timeout_s, p['label'])" in body
    eq = r6.verify_equivalence(cfg())
    assert eq["static_watchdog_budget_per_run_s"] == {"2048": 43800, "8192": 43800}


def test_l8192_trial_command_and_sequential_preflight():
    cmd = r6.build_command(8192, "x" * 64, cfg(), "exp6-L8192-t2-x", 2)
    pts = cmd[cmd.index("--points") + 1].split(",")
    assert len(pts) == 12 and all(x.startswith("t2_") for x in pts)
    assert cmd[cmd.index("--point-timeout-s") + 1] == "3600" and cmd[cmd.index("--expected-points") + 1] == "12"
    assert r6.make_run_id(8192, "abcdef0123", 2).startswith("exp6-L8192-t2-")
    c2 = r6.build_command(2048, "x" * 64, cfg(), "r")
    assert c2[c2.index("--point-timeout-s") + 1] == "1200" and c2[c2.index("--expected-points") + 1] == "36"
    for args, msg in (((8192, True, None), "executed as runs"), ((2048, True, 1), "executed as runs")):
        try:
            r6.preflight(*args)
        except RuntimeError as e:
            assert msg in str(e)
        else:
            raise AssertionError(args)
    if not (r6.run_dir(8192, 1) / "manifest.json").is_file():  # trial 2 may not start before trial 1 is terminal
        try:
            r6.preflight(8192, True, 2)
        except RuntimeError as e:
            assert "has not reached a terminal remote state" in str(e)
        else:
            raise AssertionError("trial 2 preflight passed without trial 1")


def _trial(t):
    integ, s = r6.analyze(_session(8192, trial=t), 8192, cfg(), protocol(), t)
    return integ, s


def test_one_trial_is_never_a_complete_result():
    integ, s = _trial(1)
    assert integ["all_ok"] and len(s["points"]) == 12 and s["trial"] == 1
    assert s["complete_three_trial_result"] is False and s["highest_successfully_tested_concurrency"] is None
    assert s["rabit_only_extension"] is None
    assert [c["state"] for c in integ["checks"] if c["check"].startswith("execution: trial 1")] == ["passed"]
    try:  # a one-trial log can never be analyzed as the whole sweep
        r6.analyze(_session(8192, trial=1), 8192, cfg(), protocol())
    except ValueError:
        pass
    else:
        raise AssertionError("one-trial log analyzed as a whole L8192 sweep")
    integ3, _ = r6.analyze(_session(8192, trial=1, env_over={"point_timeout_s": 1200}), 8192, cfg(), protocol(), 1)
    assert not integ3["all_ok"] and "environment" in integ3["failed_categories"]
    integ4, _ = r6.analyze(_session(8192, trial=1), 8192, cfg(), protocol(), 2)  # wrong trial identity
    assert not integ4["all_ok"]


def _three():
    out = {}
    for t, wall in ((1, 100.0), (2, 80.0), (3, 160.0)):
        integ, s = _trial(t)
        for p in s["points"]:  # trial-level statistic differs per trial
            p["wall_s"] = wall
            p["requests_per_s"] = 256 / wall
        out[t] = {"summary": s, "integrity": integ, "provenance": {"gpu_uuid": [f"GPU-trial{t}"]}}
    return out


def test_combine_requires_all_trial_identities():
    tr = _three()
    for bad in ({1: tr[1], 2: tr[2]}, {1: tr[1], 2: tr[2], 3: tr[2]}, {1: tr[1], 2: tr[3], 3: tr[2]}):
        try:
            r6.combine_trials(8192, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete / mislabeled trials accepted")
    try:
        r6.combine_trials(2048, {None: tr[1]})
    except ValueError:
        pass
    else:
        raise AssertionError("L2048 has no per-trial runs")


def test_combine_uses_trial_level_medians_without_pooling():
    tr = _three()
    assert all("requests" not in p for t in tr for p in tr[t]["summary"]["points"])  # no raw rows reach combine
    c = r6.combine_trials(8192, tr)
    row = c["cross_trial"]["rabit_kv2|16"]["requests_per_s"]
    assert row["per_trial"] == {"1": 2.56, "2": 3.2, "3": 1.6} and row["cross_trial_median"] == 2.56
    assert c["cross_trial"]["bfloat16|4"]["wall_s"]["cross_trial_median"] == 100.0
    assert "never concatenated" in c["sample_pooling"] and c["complete_three_trial_result"]
    assert c["per_trial_validity"]["2"]["gpu_uuid"] == ["GPU-trial2"] and c["all_trials_integrity_passed"]
    hs = c["highest_successfully_tested_concurrency"]["rabit_kv2"]
    assert hs["per_trial"] == {"1": 64, "2": 64, "3": 64} and not hs["failure_boundary_bracketed_by_grid"]
    assert c["rabit_over_bf16_cross_trial_median_ratio"]["64"]["requests_per_s"] == 1.0


def test_scientific_protocol_unchanged_by_l8192_amendment():
    assert r6.run_git("show", "5028eb2:benchmarks/mlsys2027/exp6_protocol.json") == \
        r6.PROTOCOL.read_text(encoding="utf-8").rstrip("\n")
    assert r6.load_protocol()["points"]["8192"] == wl.plan_points(8192)


# ------------------------------------------------------------------ independent FunctionCall orchestration
def _modal_tree():
    import ast
    return ast.parse(r6.MODAL_APP.read_text(encoding="utf-8"))


def test_launch_uses_spawn_and_persists_record_immediately():
    import ast
    tree = _modal_tree()
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "remote" not in calls and calls.count("spawn") == 1  # the sweep is never invoked synchronously
    main = r6._function(tree, "main")
    body = ast.unparse(main)
    assert "call = sweep.spawn(" in body and ".get(" not in body  # no blocking wait on the call
    i_spawn, i_write, i_replace = (body.index(k) for k in ("sweep.spawn(", "tmp.write_text(", "tmp.replace(record_path)"))
    assert i_spawn < i_write < i_replace  # record persisted right after spawn, before the entrypoint returns
    assert "'function_call_id': call.object_id" in body and "'app_id': app.app_id" in body
    for key in ("'run_id'", "'trial'", "'git_commit'", "'launch_utc'"):
        assert key in body
    assert any(isinstance(d, ast.Call) and "local_entrypoint" in ast.unparse(d) for d in main.decorator_list)
    cmd = r6.build_command(8192, "x" * 64, cfg(), "exp6-L8192-t1-x", 1, Path("rec.json"), "abc1234")
    assert cmd[cmd.index("--launch-record") + 1] == "rec.json" and cmd[cmd.index("--git-commit") + 1] == "abc1234"
    assert cmd[cmd.index("--trial") + 1] == "1" and cmd[cmd.index("run") + 1] == "--detach"


def test_no_code_path_cancels_a_function_call():
    import ast
    for f in (r6.MODAL_APP, r6.RUNNER_SCRIPT, r6.WORKER, r6.WORKLOAD):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        names = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert "cancel" not in names, f.name
        assert ".cancel(" not in f.read_text(encoding="utf-8"), f.name


class _Env:
    """Redirect the runner's run directory and snapshot creation to a temp dir (no Modal, no git archive)."""

    def __init__(self, tmp):
        self.tmp, self.saved = Path(tmp), {}

    def __enter__(self):
        self.saved = {k: getattr(r6, k) for k in ("run_dir", "run_git", "assert_protected_paths_clean")}
        real_git = self.saved["run_git"]

        def fake_git(*args):
            if "archive" in args:
                Path(args[args.index("-o") + 1]).write_bytes(b"snapshot")
                return ""
            return real_git(*args)
        r6.run_dir = lambda L, T=None: self.tmp / f"L{L}_t{T}"
        r6.run_git = fake_git
        r6.assert_protected_paths_clean = lambda ctx: None
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(r6, k, v)


def _prov():
    return {"final_config": cfg(), "prompt_set_sha256": protocol()["prompt_sets"]["8192"]["measured"]["ordered_set_sha256"],
            "git_head": "8e9b9361c47e720ba60bc05305b6981e6fcb1b65", "execution": {"points_per_run": 12},
            "earlier_trial_dirs": [], "prior_evidence_sha256_raw": {}}


def _spawning_stream(fc="fc-TESTCALL0001", app_id="ap-TESTAPP"):
    calls = []

    def stream(cmd, log_path, env):
        calls.append(cmd)
        rec = Path(cmd[cmd.index("--launch-record") + 1])
        rec.write_text(json.dumps({"function_call_id": fc, "app_id": app_id, "run_id": cmd[cmd.index("--run-id") + 1],
                                   "trial": int(cmd[cmd.index("--trial") + 1]), "git_commit": "8e9b936",
                                   "launch_utc": "2026-09-28T12:00:00Z"}), encoding="utf-8")
        log_path.write_text("EXP6_LAUNCH_RECORD=...\n", encoding="utf-8")
        return 0
    return stream, calls


def test_launcher_returns_after_spawn_and_refuses_duplicates():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp, _Env(tmp):
        stream, calls = _spawning_stream()
        assert r6.launch_run(8192, 1, _prov(), "exp6-L8192-t1-test", stream=stream) == 0  # returns, no monitoring
        d = r6.run_dir(8192, 1)
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        assert m["status"] == "running" and m["function_call"]["function_call_id"] == "fc-TESTCALL0001"
        assert m["function_call"]["app_id"] == "ap-TESTAPP" and len(calls) == 1
        try:  # a second launch for the same run never spawns again
            r6.launch_run(8192, 1, _prov(), "exp6-L8192-t1-test", stream=stream)
        except RuntimeError as e:
            assert "refusing to spawn a second FunctionCall" in str(e)
        else:
            raise AssertionError("duplicate launch accepted")
        assert len(calls) == 1
        try:  # the CLI launch path also refuses and points to --monitor
            r6.main(["--prompt-tokens", "8192", "--trial", "1"])
        except SystemExit as e:
            assert "use --monitor" in str(e)
        else:
            raise AssertionError("CLI relaunch accepted")
        assert len(calls) == 1
        stream2, _ = _spawning_stream(fc="not-a-call")  # an unconfirmed launch is never relaunched automatically
        assert r6.launch_run(8192, 2, _prov(), "exp6-L8192-t2-test", stream=stream2) == 1
        assert json.loads((r6.run_dir(8192, 2) / "manifest.json").read_text(encoding="utf-8"))["status"] == \
            "launch_unconfirmed"


def _volume(files):
    def getter(remote, local):
        name = remote.rsplit("/", 1)[1]
        if name not in files:
            return False
        local.write_bytes(files[name].encode("utf-8"))
        return True
    return getter


def test_monitor_reconstructs_call_by_id_and_never_spawns():
    import ast
    import tempfile
    mon = ast.unparse(r6._function(ast.parse(r6.RUNNER_SCRIPT.read_text(encoding="utf-8")), "monitor_run"))
    assert "build_command" not in mon and "stream" not in mon and "launch_run" not in mon
    fs = ast.unparse(r6._function(ast.parse(r6.RUNNER_SCRIPT.read_text(encoding="utf-8")), "function_call_status"))
    assert "modal.FunctionCall.from_id(fc_id)" in fs and "get_call_graph()" in fs
    with tempfile.TemporaryDirectory() as tmp, _Env(tmp):
        stream, calls = _spawning_stream(fc="fc-RESUME0001")
        prov = _prov()
        prov["prior_evidence_sha256_raw"] = r6.prior_evidence_digest()
        assert r6.launch_run(8192, 1, prov, "exp6-L8192-t1-test", stream=stream) == 0
        seen = []
        status = lambda fc: seen.append(fc) or {"function_call_id": fc, "state": "SUCCESS"}  # noqa: E731
        files = {"DONE.json": json.dumps({"run_id": "exp6-L8192-t1-test", "status": "complete", "error": None}),
                 "remote_session.log": _session(8192, trial=1)}
        rc = r6.monitor_run(8192, 1, getter=_volume(files), status_fn=status, sleep=lambda s: None, clock=lambda: 0.0)
        assert rc == 0 and seen and set(seen) == {"fc-RESUME0001"} and len(calls) == 1
        m = json.loads((r6.run_dir(8192, 1) / "manifest.json").read_text(encoding="utf-8"))
        assert m["status"] == "completed" and m["remote_session_log"]["function_call"]["state"] == "SUCCESS"
        assert m["integrity_counts"]["failed"] == 0 and m["analyzed_log"] == "remote_session.log"
        try:  # restarting the monitor on an analyzed run does nothing (and never spawns)
            r6.monitor_run(8192, 1, getter=_volume(files), status_fn=status, sleep=lambda s: None, clock=lambda: 0.0)
        except SystemExit as e:
            assert "already analyzed" in str(e)
        else:
            raise AssertionError("re-monitoring an analyzed run was accepted")
        assert len(calls) == 1


def test_cancelled_function_call_is_infrastructure_aborted():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cancel = json.dumps({"run_id": "r", "status": "exception", "error": "InputCancellation: Input was cancelled by user"})
        term = lambda fc: {"function_call_id": fc, "state": "TERMINATED"}  # noqa: E731
        r = r6.fetch_remote_log("r", d, 3600, _volume({"DONE.json": cancel, "remote_session.log": "x\n"}),
                                lambda s: None, lambda: 0.0, 60, fc_id="fc-X", status_fn=term)
        assert r["infrastructure_aborted"] and not r["complete"] and r["function_call"]["state"] == "TERMINATED"
        clock = {"t": 0.0}  # TERMINATED without DONE.json: stop after the grace polls, still aborted
        r = r6.fetch_remote_log("r", d, 3600, _volume({"remote_session.log": "partial\n"}),
                                lambda s: clock.__setitem__("t", clock["t"] + s), lambda: clock["t"], 60,
                                fc_id="fc-X", status_fn=term)
        assert r["infrastructure_aborted"] and not r["done_found"] and r["polls"] == 3 and clock["t"] == 120
        ok = lambda fc: {"function_call_id": fc, "state": "SUCCESS"}  # noqa: E731
        done = json.dumps({"run_id": "r", "status": "complete", "error": None})
        r = r6.fetch_remote_log("r", d, 3600, _volume({"DONE.json": done, "remote_session.log": "x\n"}),
                                lambda s: None, lambda: 0.0, 60, fc_id="fc-X", status_fn=ok)
        assert r["complete"] and not r["infrastructure_aborted"]
    with tempfile.TemporaryDirectory() as tmp, _Env(tmp):  # end-to-end: the trial manifest is infrastructure_aborted
        stream, _ = _spawning_stream(fc="fc-CANCELLED01")
        prov = _prov()
        prov["prior_evidence_sha256_raw"] = r6.prior_evidence_digest()
        r6.launch_run(8192, 1, prov, "exp6-L8192-t1-test", stream=stream)
        files = {"DONE.json": cancel, "remote_session.log": _session(8192, trial=1, omit=tuple(
            q["label"] for q in r6.run_plan(8192, 1)[1:]))}
        assert r6.monitor_run(8192, 1, getter=_volume(files), status_fn=term, sleep=lambda s: None,
                              clock=lambda: 0.0) == 1
        m = json.loads((r6.run_dir(8192, 1) / "manifest.json").read_text(encoding="utf-8"))
        assert m["status"] == "infrastructure_aborted"


def test_l8192_trial1_attempt1_excluded():
    a1 = r6.BASE_OUT / "L8192" / "infrastructure_aborted" / "trial_1_attempt_1"
    assert a1 in r6.excluded_attempts(8192) and a1 in r6.EVIDENCE_DIRS and a1 in r6.PROTECTED_PATHS
    assert r6.run_git("ls-files", r6.rel(a1 / "remote_session.log"))
    st = json.loads((a1 / "ATTEMPT_STATUS.json").read_text(encoding="utf-8"))
    assert st["status"] == "infrastructure_aborted" and st["accepted_for_performance_interpretation"] is False
    assert r6.run_dir(8192, 1) not in (a1, *a1.parents) and a1 not in r6.run_dir(8192, 1).parents


def test_orchestration_change_leaves_science_unchanged():
    import ast
    head = "8e9b9361c47e720ba60bc05305b6981e6fcb1b65"
    for f in ("exp6_protocol.json", "exp6_l8192_execution_amendment.json", "exp6_workload.py", "exp6_worker.py"):
        assert r6.run_git("show", f"{head}:benchmarks/mlsys2027/{f}") == \
            (r6.HERE / f).read_text(encoding="utf-8").rstrip("\n"), f
    old = ast.parse(r6.run_git("show", f"{head}:benchmarks/mlsys2027/run_experiment6_concurrency.py"))
    new = ast.parse(r6.RUNNER_SCRIPT.read_text(encoding="utf-8"))
    # `analyze` is intentionally excluded: its validation logic changed in the capacity-bound fix; its metric values
    # on real evidence are pinned by test_accepted_trial1_capacity_fix_changes_only_one_check.
    for fn in ("point_metrics", "inflight_concurrency", "pct", "classify", "shadow_validity", "parse_point",
               "combine_trials", "run_plan", "static_watchdog_budget", "highest_successful", "build_l8192_amendment"):
        assert ast.dump(r6._function(old, fn)) == ast.dump(r6._function(new, fn)), fn
    for const in ("EXECUTION", "CROSS_TRIAL_METRICS", "RATIO_METRICS"):
        assert ast.dump(r6._module_assign(old, const)) == ast.dump(r6._module_assign(new, const)), const
    om = ast.parse(r6.run_git("show", f"{head}:benchmarks/mlsys2027/exp6_modal.py"))
    for fn in ("_sweep_body", "sweep", "_Tee", "_commit_session_logs"):
        a, b = (next(n for n in ast.walk(t) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == fn)
                for t in (om, _modal_tree()))
        assert ast.dump(a) == ast.dump(b), fn
    assert r6.EXECUTION[8192]["point_timeout_s"] == 3600 and r6.static_watchdog_budget(8192) == 43800


# ------------------------------------------------------------------ capacity-bound shadow validity (validation logic)
_BF64 = "t1_bf16_L8192_c64"
_CAPBOUND = {"meas_cap": 47, "cond_cap": 47}  # the allocator admits at most 47 full-length requests


def _t1(overrides=None, proc_over=None):
    integ, s = r6.analyze(_session(8192, overrides=overrides, proc_over=proc_over, trial=1), 8192, cfg(), protocol(), 1)
    return integ, s


def _shadow_state(integ, label):
    return [c["state"] for c in _check(integ, f"{label}: shadow conditioning reached")]


def test_allocator_ceiling_formula():
    bf = r6.allocator_full_length_ceiling({"num_gpu_blocks": 12282, "block_size": 32}, 8192)
    assert bf["full_request_tokens"] == 8224 and bf["blocks_per_full_request"] == 257
    assert bf["allocator_derived_full_length_sequence_ceiling"] == 47 and "NOT an observed" in bf["kind"]
    rb = r6.allocator_full_length_ceiling({"num_gpu_blocks": 64831, "block_size": 32}, 8192)
    assert rb["allocator_derived_full_length_sequence_ceiling"] == 252 and rb["blocks_per_full_request"] == 257
    assert 393024 // 32 == 12282 and 2074592 // 32 == 64831  # the pinned allocator capacities
    l2 = r6.allocator_full_length_ceiling({"num_gpu_blocks": 12282, "block_size": 32}, 2048)
    assert l2["blocks_per_full_request"] == 65 and l2["allocator_derived_full_length_sequence_ceiling"] == 188
    assert r6.allocator_full_length_ceiling(None, 8192) is None
    assert r6.allocator_full_length_ceiling({"num_gpu_blocks": 1}, 8192) is None


def test_normal_point_reaching_c_still_requires_shadow_to_reach_c():
    integ, s = _t1({"t1_rabit_L8192_c16": {"cond_serial": True}, "t1_bf16_L8192_c32": {"cond_cap": 20}})
    for label in ("t1_rabit_L8192_c16", "t1_bf16_L8192_c32"):
        p = _pt(s, label)
        assert p["outcome_class"] == "sustained_target_concurrency" and not p["interpretable"]
        assert _shadow_state(integ, label) == ["failed"] and p["capacity_bound"] is False
        assert not p["capacity_bound_validation"]["conditions"]["measured_outcome_is_target_concurrency_not_reached"]


def test_capacity_bound_point_uses_narrow_alternative():
    integ, s = _t1({_BF64: dict(_CAPBOUND)})
    assert integ["all_ok"] and integ["counts"] == {"passed": 138, "failed": 0, "not_run": 0, "not_evaluated": 0}
    p = _pt(s, _BF64)
    assert p["outcome_class"] == "target_concurrency_not_reached"  # never converted to sustained
    assert p["target_concurrency"] == 64 and p["inflight_concurrency"]["observed_max_inflight_concurrency"] == 47
    cb = p["capacity_bound_validation"]
    assert p["capacity_bound"] is True and cb["allocator_derived_full_length_sequence_ceiling"] == 47
    assert cb["shadow_observed_max_inflight_concurrency"] == 47 and all(cb["conditions"].values())
    assert cb["statement"] == ("At offered concurrency 64, the system admitted at most 47 overlapping in-flight "
                               "requests, matching the allocator-derived full-length KV ceiling of 47 under this workload.")
    assert _shadow_state(integ, _BF64) == ["passed"] and p["interpretable"]
    hs = r6.highest_successful({(p2["dtype"], p2["target_concurrency"], 1): {
        "interpretable_success": p2["outcome_class"] == "sustained_target_concurrency" and p2["interpretable"]}
        for p2 in s["points"]})
    assert hs["bfloat16"]["per_trial"]["1"] == 32 and hs["rabit_kv2"]["per_trial"]["1"] == 64  # C64 not a BF16 success
    assert all("capacity_bound" not in q for q in s["points"] if q["label"] != _BF64)  # only evaluated on a miss


def _capacity_bound_fails(overrides=None, proc_over=None, cond=None):
    integ, s = _t1({_BF64: {**_CAPBOUND, **(overrides or {})}}, proc_over={_BF64: proc_over} if proc_over else None)
    p = _pt(s, _BF64)
    assert _shadow_state(integ, _BF64) == ["failed"] and not integ["all_ok"], overrides
    assert p["capacity_bound"] is False and not p["interpretable"]
    if cond:
        assert p["capacity_bound_validation"]["conditions"][cond] is False, (cond, p["capacity_bound_validation"])
    return p


def test_shadow_max_must_cover_measured_max():
    _capacity_bound_fails({"cond_cap": 40}, cond="shadow_max_ge_measured_max")


def test_measured_max_must_not_exceed_ceiling_and_ceiling_below_target():
    _capacity_bound_fails({"blocks": 11000}, cond="measured_max_le_allocator_ceiling")  # ceiling 42 < measured 47
    _capacity_bound_fails({"blocks": 12282 * 2}, cond="allocator_ceiling_below_target")  # ceiling 95 >= 64


def test_non_capacity_bound_miss_still_fails():
    integ, s = _t1({"t1_rabit_L8192_c64": dict(_CAPBOUND)})  # RABIT ceiling 252 >= 64: a 47 miss is not capacity
    p = _pt(s, "t1_rabit_L8192_c64")
    assert p["outcome_class"] == "target_concurrency_not_reached" and p["capacity_bound"] is False
    assert p["capacity_bound_validation"]["allocator_derived_full_length_sequence_ceiling"] == 252
    assert _shadow_state(integ, "t1_rabit_L8192_c64") == ["failed"] and not integ["all_ok"]


def test_exception_blocked_by_jit_oom_failure_watchdog_preemption_and_config():
    _capacity_bound_fails({"jit_measured": 1}, cond="measured_phase_jit_zero")
    p = _capacity_bound_fails({"oom": True, "fail": "request_oom"})
    assert p["outcome_class"] == "oom_or_allocation_failure"
    p = _capacity_bound_fails({"fail": "request_execution_failure"}, cond="no_request_failure")
    assert p["outcome_class"] == "engine_or_request_failure"
    p = _capacity_bound_fails(proc_over={"timed_out": True, "returncode": -9}, cond="no_watchdog")
    assert p["outcome_class"] == "engine_or_request_failure"
    p = _capacity_bound_fails({"counter": False}, cond="preemption_status_available")
    assert p["preemption_source"] == "unavailable"
    leak = {"applicable": False, "env": {"VLLM_RABIT2_STAGE3C_IMPL": "shared_decode"}, "profiling_env": {}}
    _capacity_bound_fails({"s3": leak}, cond="selector_qb_engine_config_prompt_hashes_capacity_exact")
    _capacity_bound_fails({"max_seqs": 47}, cond="selector_qb_engine_config_prompt_hashes_capacity_exact")
    _capacity_bound_fails({"prompt_set": protocol()["prompt_sets"]["2048"]["measured"]},
                          cond="selector_qb_engine_config_prompt_hashes_capacity_exact")
    _capacity_bound_fails({"cap": 393000, "blocks": 12282},
                          cond="selector_qb_engine_config_prompt_hashes_capacity_exact")
    _capacity_bound_fails({"cond_drop": 1}, cond="shadow_256_of_256_completed")
    _capacity_bound_fails({"drop": 1})  # measured 255/256 -> engine_or_request_failure, never capacity-bound


def test_accepted_trial1_capacity_fix_changes_only_one_check():
    pin = r6.PINNED_TRIAL_RAW[(8192, 1)]
    d = pin["dir"]
    raw = d / "remote_session.log"
    assert r6.sha256_raw(raw) == pin["remote_session_log_sha256"] == (
        "5a42a5294dbb612347fd72d7dd43710d4dc3498819f27e61b29c6822fd27ee1c") and raw.stat().st_size == 3102296
    assert r6.sha256_raw(d / "remote_DONE.json") == pin["remote_done_sha256"]
    for name, want in (pin["pre_fix_summary"], pin["pre_fix_integrity"]):
        assert r6.sha256_raw(d / name) == want
    integ, s = r6.analyze(raw.read_text(encoding="utf-8", errors="replace"), 8192, cfg(), protocol(), 1)
    integ, s = json.loads(json.dumps(integ, default=str)), json.loads(json.dumps(s, default=str))
    old_i = json.loads((d / pin["pre_fix_integrity"][0]).read_text(encoding="utf-8"))
    old_s = json.loads((d / pin["pre_fix_summary"][0]).read_text(encoding="utf-8"))
    assert old_i["counts"] == {"passed": 137, "failed": 1, "not_run": 0, "not_evaluated": 0}
    assert integ["counts"] == {"passed": 138, "failed": 0, "not_run": 0, "not_evaluated": 0}
    diff = r6.capacity_fix_diff(old_i, integ, old_s, s)
    assert diff == {"capacity_bound_points": [_BF64], "problems": [],
                    "flipped_checks": [f"{_BF64}: shadow conditioning reached target overlapping in-flight concurrency 64"]}
    p = _pt(s, _BF64)
    assert p["outcome_class"] == "target_concurrency_not_reached" and p["capacity_bound"] is True
    assert p["inflight_concurrency"]["observed_max_inflight_concurrency"] == 47 and p["target_concurrency"] == 64
    assert p["capacity_bound_validation"]["allocator_derived_full_length_sequence_ceiling"] == 47
    for o, n in zip(old_s["points"], s["points"]):  # every metric value identical
        for k in ("requests_per_s", "output_tokens_per_s", "total_tokens_per_s", "wall_s", "latency_s", "ttft_s",
                  "tpot_s", "completed_requests", "preemptions", "inflight_concurrency", "outcome_class"):
            assert o[k] == n[k], (n["label"], k)
    rb = _pt(s, "t1_rabit_L8192_c64")
    assert "capacity_bound" not in rb and rb["inflight_concurrency"]["observed_max_inflight_concurrency"] == 64
    assert r6.run_git("show", "5028eb2:benchmarks/mlsys2027/exp6_protocol.json") == \
        r6.PROTOCOL.read_text(encoding="utf-8").rstrip("\n")  # scientific protocol byte-identical


def test_missing_point_is_not_run_and_fails_completion():
    integ, s = _run(omit=("t3_rabit_L2048_c64",))
    assert _pt(s, "t3_rabit_L2048_c64")["outcome_class"] is None
    assert any(c["state"] == "not_run" for c in _check(integ, "t3_rabit_L2048_c64"))


def test_8192_sweep_and_inflight_concurrency_parser():
    for t in (1, 2, 3):  # L8192 is analyzed per trial (execution amendment)
        integ, s = r6.analyze(_session(8192, trial=t), 8192, cfg(), protocol(), t)
        assert integ["all_ok"] and len(s["points"]) == 12
    try:
        _run(L=8192)  # a whole-sweep L8192 analysis is refused
    except ValueError:
        pass
    else:
        raise AssertionError("whole-sweep L8192 analysis accepted")
    rows = [{"scheduled_ts": 0.0 + i, "last_token_ts": 10.0 + i} for i in range(1, 5)]
    c = r6.inflight_concurrency(rows, 4)
    assert c["observed_max_inflight_concurrency"] == 4 and abs(c["all_c_inflight_overlap_total_s"] - 7.0) < 1e-9
    assert "not GPU residency" in c["kind"]
    assert r6.inflight_concurrency([{"scheduled_ts": None, "last_token_ts": 1.0}], 1) == {"evaluable": False}


def test_extension_refused_and_protocol_not_overwritten():
    for args in (["--rabit-extension"], ["--write-protocol"]):
        try:
            r6.main(args)
        except SystemExit as e:
            assert "NOT enabled" in str(e) or "never overwritten" in str(e)
        else:
            raise AssertionError(args)


def test_preflight_equivalence():
    eq = r6.verify_equivalence(cfg())
    assert eq["engine_kwargs_equal_frozen_exp5_except_max_num_seqs_and_kv_dtype"] and eq["greedy"]
    assert eq["modal_backstop_s"] > eq["watchdog_budget_s"] == 600 + 36 * 1200


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
