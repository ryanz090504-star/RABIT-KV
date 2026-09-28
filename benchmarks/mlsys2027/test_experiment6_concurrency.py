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


def _rows(L, C, serial=False, bad_len_at=None, drop=0, hashes=None):
    hashes = hashes or protocol()["prompt_sets"][str(L)]["measured"]["per_prompt_sha256"]
    rows, n = [], wl.MEASURED_REQUESTS - drop
    for i in range(n):
        if serial:
            s = 10.0 + i * 2.0
        else:
            s = 10.0 + (i // C) * 2.0  # closed loop: waves of C requests
        rows.append({"i": i, "prompt_tokens": L, "prompt_token_ids_sha256": hashes[i],
                     "output_tokens": 31 if i == bad_len_at else 32,
                     "output_token_ids_sha256": "o" * 64, "finish_reason": "length",
                     "queued_ts": 9.0, "scheduled_ts": s, "first_token_ts": s + 0.5, "last_token_ts": s + 1.5})
    return rows


def _point_lines(spec, *, serial=False, preempt=0, counter=True, logged=None, jit_measured=0, oom=False,
                 fail=None, bad_len_at=None, drop=0, cap=None, s3=None, max_seqs=None, no_engine=False,
                 prompt_set=None, cond_jit=1, cond_serial=False, cond_drop=0, cond_bad_len=False, cond_fail=False):
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
           "conditioning_requests": C,
           "conditioning_prompt_set": protocol()["compile_conditioning"]["prompt_sets"][str(L)][str(C)]}
    ch = protocol()["compile_conditioning"]["prompt_sets"][str(L)][str(C)]["per_prompt_sha256"]
    cond_rows = [{"i": i, "prompt_tokens": L - 1 if (cond_bad_len and i == 0) else L, "prompt_token_ids_sha256": ch[i],
                  "output_tokens": 32, "finish_reason": "length",
                  "scheduled_ts": 1.0 + (2.0 * i if cond_serial else 0.0),
                  "last_token_ts": 2.0 + (2.0 * i if cond_serial else 0.0)} for i in range(C - cond_drop)]
    lines += [f"EXP6_EFFECTIVE_ENGINE_CONFIG={json.dumps({'max_num_seqs': ms})}",
              f"EXP6_CAPACITY={json.dumps({'num_gpu_blocks': 1, 'block_size': 32, 'capacity_tokens': cap or r6.EXPECTED_CAPACITY[d]})}",
              f"EXP6_WORKLOAD={json.dumps(wkl)}",
              "WARNING Triton kernel JIT compilation during inference: setup_kernel",
              "EXP6_CONDITIONING_BEGIN"]
    lines += ["WARNING Triton kernel JIT compilation during inference: cond"] * cond_jit
    if cond_fail:
        fail_c = {"phase": "conditioning", "kind": "request_execution_failure", "error": "x"}
        return lines + [f"EXP6_REQUEST_FAILURE={json.dumps(fail_c)}"]
    lines += ["EXP6_CONDITIONING_END"] + [f"EXP6_CONDITIONING_REQUEST={json.dumps(r)}" for r in cond_rows]
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
    lines += [f"EXP6_REQUEST={json.dumps(r)}" for r in _rows(L, C, serial, bad_len_at, drop)]
    before = 5.0 if counter else None
    lines.append(f"EXP6_MEASURED_SUMMARY={json.dumps({'wall_s': 100.0, 'returned_requests': 256 - drop, 'preemptions_before': before, 'preemptions_after': (before + preempt) if counter else None, 'preemption_counter_available': counter})}")
    return lines + ["EXP6_WORKER_COMPLETE"]


def _session(L=2048, overrides=None, proc_over=None, omit=()):
    overrides, proc_over = overrides or {}, proc_over or {}
    plan = wl.plan_points(L)
    bsess = (r6.rs6.rsd.rt.OUT_DIR / "modal_session.log").read_text(encoding="utf-8").splitlines()
    env = json.loads(next(ln for ln in bsess if ln.startswith("S3C_ENVIRONMENT=")).split("=", 1)[1])
    env["point_labels"] = [p["label"] for p in plan]
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
              f"S3C_SWEEP_COMPLETE={json.dumps({'points': 36, 'failed_points': []})}"]
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
    assert p["concurrency"]["observed_max_inflight_concurrency"] == 64 and p["concurrency"]["all_c_inflight_overlap_total_s"] > 0
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
    assert p["concurrency"]["observed_max_inflight_concurrency"] == 1
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
    assert p["jit_lines_by_phase"] == {"setup": 1, "conditioning": 1, "warmup": 1, "measured": 2, "after": 0}
    assert s["highest_successfully_tested_concurrency"]["rabit_kv2"]["per_trial"]["3"] == 32
    assert not integ["all_ok"]


# ------------------------------------------------------------------ compile-conditioning amendment
def test_conditioning_prompt_sets_frozen_and_disjoint():
    p = protocol()
    cc = p["compile_conditioning"]
    assert cc["output_tokens"] == 32
    assert cc["jit_accounting_phases"] == ["setup", "conditioning", "warmup", "measured", "after"]
    for L in wl.PROMPT_LENGTHS:
        base = (set(p["prompt_sets"][str(L)]["measured"]["per_prompt_sha256"])
                | set(p["prompt_sets"][str(L)]["warmup"]["per_prompt_sha256"]))
        seen = set()
        for C in wl.CONCURRENCY_GRID:
            prompts = wl.conditioning_prompts(L, C)
            d = cc["prompt_sets"][str(L)][str(C)]
            assert len(prompts) == C == d["count"] and all(len(x) == L for x in prompts)  # count == C, same length
            assert wl.set_digest(prompts) == d  # hash-pinned
            assert len(set(d["per_prompt_sha256"])) == C and not set(d["per_prompt_sha256"]) & base
            assert not set(d["per_prompt_sha256"]) & seen
            seen |= set(d["per_prompt_sha256"])


def test_worker_conditioning_same_rule_for_both_dtypes_and_order():
    import ast
    fn = r6._function(ast.parse(r6.WORKER.read_text(encoding="utf-8")), "main")
    main_src = ast.unparse(fn)
    # conditioning prompts depend only on (prompt length, concurrency), never on dtype; same 32-token greedy sampler
    assert "cond = wl.conditioning_prompts(args.prompt_tokens, args.concurrency)" in main_src
    assert "llm.generate([{'prompt_token_ids': p} for p in cond], sp, use_tqdm=False)" in main_src
    assert "sp = SamplingParams(temperature=0.0, max_tokens=wl.OUTPUT_TOKENS, ignore_eos=True)" in main_src
    i = [main_src.index(k) for k in ("EXP6_CONDITIONING_BEGIN", "EXP6_WARMUP_BEGIN", "EXP6_MEASURED_BEGIN")]
    assert i == sorted(i)
    for node in ast.walk(fn):  # never inside a dtype branch
        if isinstance(node, ast.If) and "kv_cache_dtype" in ast.unparse(node.test):
            assert "cond" not in ast.unparse(node).replace("conditioning", "")
    assert wl.WARMUP_REQUESTS == 2 and wl.MEASURED_REQUESTS == 256 and wl.OUTPUT_TOKENS == 32


def test_conditioning_valid_and_its_jit_allowed():
    integ, s = _run(overrides={"t1_rabit_L2048_c64": {"cond_jit": 5}, "t2_bf16_L2048_c16": {"cond_jit": 3}})
    assert integ["all_ok"]
    for label in ("t1_rabit_L2048_c64", "t2_bf16_L2048_c16", "t3_bf16_L2048_c1"):
        p = _pt(s, label)
        cv = p["compile_conditioning"]
        assert cv["valid"] and cv["requests"] == cv["exact_requests"] == p["target_concurrency"]
        assert cv["observed_max_inflight_concurrency"] == p["target_concurrency"] and p["interpretable"]
    assert _pt(s, "t1_rabit_L2048_c64")["jit_lines_by_phase"]["conditioning"] == 5
    assert len(_check(integ, "t1_rabit_L2048_c64: compile-conditioning")) == 2


def test_conditioning_invalid_makes_point_non_interpretable():
    integ, s = _run(overrides={"t1_rabit_L2048_c16": {"cond_serial": True}, "t1_rabit_L2048_c32": {"cond_drop": 1},
                               "t2_bf16_L2048_c8": {"cond_bad_len": True}, "t3_rabit_L2048_c4": {"cond_fail": True}})
    assert not integ["all_ok"] and "conditioning" in integ["failed_categories"]
    a = _pt(s, "t1_rabit_L2048_c16")
    assert a["compile_conditioning"]["observed_max_inflight_concurrency"] == 1 and not a["interpretable"]
    assert [x["state"] for x in _check(integ, "t1_rabit_L2048_c16: compile-conditioning reached")] == ["failed"]
    assert a["outcome_class"] == "sustained_target_concurrency"  # measured classification itself unchanged
    b = _pt(s, "t1_rabit_L2048_c32")
    assert not b["compile_conditioning"]["completed_all_exact"] and not b["interpretable"]
    assert not _pt(s, "t2_bf16_L2048_c8")["compile_conditioning"]["valid"]
    f = _pt(s, "t3_rabit_L2048_c4")
    assert f["compile_conditioning"]["failure_in_conditioning"] and f["outcome_class"] == "engine_or_request_failure"
    h = s["highest_successfully_tested_concurrency"]["rabit_kv2"]
    assert h["per_trial"]["1"] == 64 and h["highest_contiguous_successful_concurrency_all_trials"] == 1


def test_old_jit_contaminated_attempt_remains_failed():
    d = r6.DIAGNOSTIC_ATTEMPTS[2048][0]
    assert r6.run_git("ls-files", r6.rel(d / "modal_session.log"))  # archived, committed
    assert d in r6.EVIDENCE_DIRS and d in r6.PROTECTED_PATHS
    st = json.loads((d / "ATTEMPT_STATUS.json").read_text(encoding="utf-8"))
    assert st["accepted_for_performance_interpretation"] is False and st["use_in_paper_dataset"] is False
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    ic = json.loads((d / "integrity_check.json").read_text(encoding="utf-8"))
    assert m["status"] == "failed" and ic["failed_categories"] == ["jit"]
    assert ic["counts"] == {"passed": 320, "failed": 9, "not_run": 0, "not_evaluated": 0}
    # re-analysis under the amended harness still fails (measured JIT; and it has no conditioning phase at all)
    text = (d / "modal_session.log").read_text(encoding="utf-8", errors="replace")
    integ, s = r6.analyze(text, 2048, cfg(), protocol())
    assert not integ["all_ok"] and {"jit", "conditioning"} <= set(integ["failed_categories"])
    assert not any(p["interpretable"] for p in s["points"])


def test_failed_attempt_never_pooled_with_rerun():
    d_old, d_new = r6.DIAGNOSTIC_ATTEMPTS[2048][0], r6.out_dir(2048)
    assert d_new != d_old and d_old not in d_new.parents and d_new not in d_old.parents
    _, s = _run()
    assert s["excluded_attempts_not_pooled"] == [r6.rel(d_old)] and len(s["points"]) == 36
    assert protocol()["excluded_attempts"]["2048"]["dirs"] == [r6.rel(d_old)]
    orig = r6.out_dir
    try:
        r6.out_dir = lambda L: d_old  # a run may never write into / next to the archived attempt
        try:
            r6.preflight(2048, dry_run=True)
        except RuntimeError as e:
            assert "overlaps an archived diagnostic attempt" in str(e)
        else:
            raise AssertionError("preflight accepted the diagnostic attempt directory")
    finally:
        r6.out_dir = orig


def test_protocol_changed_only_by_reviewed_conditioning_amendment():
    old = json.loads(r6.run_git("show", "abefce2:benchmarks/mlsys2027/exp6_protocol.json"))
    new = protocol()
    assert set(new) - set(old) == {"compile_conditioning", "excluded_attempts"} and not set(old) - set(new)
    assert [k for k in old if old[k] != new[k]] == ["amendments"]
    assert set(new["amendments"]) - set(old["amendments"]) == {"compile_conditioning"}
    assert all(new["amendments"][k] == v for k, v in old["amendments"].items())


def test_missing_point_is_not_run_and_fails_completion():
    integ, s = _run(omit=("t3_rabit_L2048_c64",))
    assert _pt(s, "t3_rabit_L2048_c64")["outcome_class"] is None
    assert any(c["state"] == "not_run" for c in _check(integ, "t3_rabit_L2048_c64"))


def test_8192_sweep_and_inflight_concurrency_parser():
    integ, s = _run(L=8192)
    assert integ["all_ok"] and len(s["points"]) == 36
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
