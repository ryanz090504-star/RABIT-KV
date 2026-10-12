"""
Offline tests for Experiment 5 attempt 2 (no GPU, no torch, no pytest required:
`python benchmarks/mlsys2027/test_experiment5_attempt2.py`; also collected by pytest).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiment5_attempt2 as a2  # noqa: E402

r5 = a2.r5
FROZEN_ORDER = ["A512", "B512", "B2048", "A2048", "A4096", "B4096", "B8192", "A8192", "A16384", "B16384", "B32768",
                "A32768"]


def test_protocol_unchanged():
    assert [l for _, l, *_ in r5.LEGS] == FROZEN_ORDER
    assert [l for _, l, *_ in r5.CONDITIONING] == ["conditioning_A512", "conditioning_B512"]
    assert r5.WARMUPS_PER_LEG == 5 and r5.REPS_PER_LEG == 15 and r5.OUTPUT_TOKENS == 32
    assert r5.CONTEXT_GRID == [512, 2048, 4096, 8192, 16384, 32768]
    assert r5.prompt_tokens_for(32768) == 32736 and r5.MAX_MODEL_LEN == 32768
    assert r5.GATE_TIMEOUT_S == 600 and r5.LEG_TIMEOUT_S == 900


def test_watchdogs_and_backstop():
    assert {l: a2.leg_timeout(l) for _, l, *_ in r5.ALL_CELLS} == {
        **{l: 900 for _, l, *_ in r5.ALL_CELLS}, "B32768": 3600}
    assert a2.WATCHDOG_BUDGET_S == 600 + 13 * 900 + 3600 == 15900
    assert a2.MODAL_FUNCTION_TIMEOUT_S == 17100 > a2.WATCHDOG_BUDGET_S
    assert a2.MODAL_FUNCTION_TIMEOUT_S - a2.WATCHDOG_BUDGET_S == 14400 - 13200  # original 1200 s margin preserved


def test_equivalence_proofs_pass():
    eq = r5.verify_equivalence()
    assert eq["attempt2_modal_equal_frozen_exp5_except_reviewed_additions"]
    assert eq["attempt2_worker_equal_frozen_exp5_except_stage3c_selection"]
    assert eq["per_cell_watchdog_rule"] == "return B32768_LEG_TIMEOUT_S if label == 'B32768' else LEG_TIMEOUT_S"


def test_final_stage3c_from_committed_evidence():
    s3 = a2.final_stage3c()
    assert s3["impl"] == "shared_decode" and s3["query_block"] == 32
    assert s3["b32768_expected_prompt_sha256"] == "dbc5bd884143e25fc9b417b9a9af888aaf92092e6556e4b8fdd1d6ca7d63b969"
    assert s3["b32768_expected_output_sha256"] == "aef936f8e6b2523e82f1b3144793012fe50cf21526fd24aea3309bea02851d26"
    cmd = " ".join(r5.build_command())
    assert "--stage3c-impl shared_decode --query-block 32" in cmd and s3["b32768_expected_output_sha256"] in cmd
    for _, label, d, c, p in r5.ALL_CELLS:
        w = r5.worker_command(label, d, c, p)
        assert w[1] == "/opt/exp5/exp5_attempt2_engine_worker.py"
        assert ("--stage3c-impl" in w) is (d == r5.B)


def test_prompt_identity_with_failed_attempt():
    sha = lambda n: hashlib.sha256(json.dumps([128000] + [279] * (n - 1)).encode("utf-8")).hexdigest()  # noqa: E731
    seen = 0
    for _, label, d, c, p in r5.ALL_CELLS:
        log = a2.FAILED_ATTEMPT_1 / r5.LOG_NAME[label]
        if log.is_file():
            wl = next((json.loads(ln.split("=", 1)[1]) for ln in log.read_text(encoding="utf-8").splitlines()
                       if ln.startswith("EXP5_WORKLOAD=")), None)
            if wl:
                assert wl["prompt_tokens"] == p and wl["prompt_token_ids_sha256"] == sha(p), label
                seen += 1
    assert seen == 13 and sha(32736) == "dbc5bd884143e25fc9b417b9a9af888aaf92092e6556e4b8fdd1d6ca7d63b969"


def test_frozen_files_and_failed_attempt_untouched():
    for f in (a2.FROZEN_RUNNER, a2.FROZEN_MODAL_APP, a2.FROZEN_WORKER):
        assert a2.run_git("status", "--short", "--", a2.rel(f)) == "", f
    assert a2.run_git("status", "--short", "--", a2.rel(a2.FAILED_ATTEMPT_1)) == ""
    assert len(a2.archived_attempts_digest()) == 20
    assert r5.OUT_DIR == a2.ATTEMPT_DIR and r5.SUMMARY.parent == a2.ATTEMPT_DIR
    assert a2.rel(a2.FAILED_ATTEMPT_1) in [a2.rel(p) for p in r5.PROTECTED_PATHS]


def _modal():
    os.environ.setdefault("EXP5_VLLM_SNAPSHOT", str(Path(__file__).resolve()))
    return importlib.import_module("exp5_attempt2_modal")


def _cell_lines(prefix, dtype, label, s3, out_sha="h" * 64, reps=15):
    wl = {"role": "measured", "context_point": 32768, "prompt_tokens": 32736, "output_tokens": 32, "max_tokens": 32,
          "temperature": 0.0, "ignore_eos": True, "warmups": 5, "reps": reps, "prompt_token_ids_sha256": "p"}
    eff = {"compilation_mode": "NONE", "cudagraph_mode": "NONE", "enforce_eager": True,
           "attention_backend": "AttentionBackendEnum.TRITON_ATTN", "calculate_kv_scales": False,
           "hf_quantization_config": None, "quantization": None}
    tags = {"EXP5_LEG": {"leg": label, "kv_cache_dtype": dtype, "role": "measured", "context_point": 32768,
                         "prompt_tokens": 32736},
            "EXP5_REQUESTED_ENGINE_KWARGS": {"kv_cache_dtype": dtype}, "EXP5_EFFECTIVE_ENGINE_CONFIG": eff,
            "EXP5_KV_DTYPE": {"requested_kv_cache_dtype": dtype, "engine_cache_dtype": dtype},
            "EXP5_CAPACITY": {"capacity_tokens": 1}, "EXP5_WORKLOAD": wl, "EXP5_STAGE3C": s3}
    row = {"prompt_tokens": 32736, "output_tokens": 32, "prompt_token_ids_sha256": "p",
           "output_token_ids_sha256": out_sha}
    lines = [f"{prefix}{k}={json.dumps(v)}" for k, v in tags.items()]
    lines += [f"{prefix}EXP5_WARMUP_BEGIN"] + [f"{prefix}EXP5_WARMUP {json.dumps(row)}"] * 5 + [f"{prefix}EXP5_WARMUP_END"]
    lines += [f"{prefix}EXP5_MEASUREMENT_BEGIN"] + [f"{prefix}EXP5_SAMPLE {json.dumps(row)}"] * reps
    return lines + [f"{prefix}EXP5_MEASUREMENT_END", f"{prefix}EXP5_WORKER_COMPLETE"]


def test_in_container_verdict_enforces_selection_and_b32768_hash():
    m = _modal()
    good_s3 = {"applicable": True, "requested_impl": "shared_decode", "effective_impl": "shared_decode",
               "requested_query_block": 32, "effective_query_block": 32, "env": {}}
    exp = "a" * 64
    cell = {"label": "B32768", "dtype": "rabit_kv2", "role": "measured", "context": 32768, "prompt": 32736,
            "warmups": 5, "reps": 15, "stage3c": {"impl": "shared_decode", "query_block": 32},
            "expected_output_sha": exp}
    meta = {"returncode": 0, "timed_out": False, "group_processes_after_leader_exit": [],
            "group_processes_remaining": []}
    pfx = "[leg13:rabit_kv2] "
    assert m._verify_cell(_cell_lines(pfx, "rabit_kv2", "B32768", good_s3, exp), pfx, cell, meta, {})["verdict"] == "ok"
    v = m._verify_cell(_cell_lines(pfx, "rabit_kv2", "B32768", good_s3, "b" * 64), pfx, cell, meta, {})
    assert v["verdict"] == "stop" and any("pre-established B32768" in r for r in v["reasons"])
    bad_s3 = {**good_s3, "effective_impl": "reference"}
    v = m._verify_cell(_cell_lines(pfx, "rabit_kv2", "B32768", bad_s3, exp), pfx, cell, meta, {})
    assert v["verdict"] == "stop" and any("Stage3C selection mismatch" in r for r in v["reasons"])
    bf = {**cell, "label": "A32768", "dtype": "bfloat16", "expected_output_sha": None}
    pfa = "[leg14:bfloat16] "
    none = {"applicable": False, "env": {"VLLM_RABIT2_STAGE3C_IMPL": None}}
    assert m._verify_cell(_cell_lines(pfa, "bfloat16", "A32768", none), pfa, bf, meta, {})["verdict"] == "ok"
    leaked = {"applicable": False, "env": {"VLLM_RABIT2_STAGE3C_IMPL": "shared_decode"}}
    assert m._verify_cell(_cell_lines(pfa, "bfloat16", "A32768", leaked), pfa, bf, meta, {})["verdict"] == "stop"
    assert m._leg_timeout("B32768") == 3600 and m._leg_timeout("A32768") == 900 and m._leg_timeout("B16384") == 900


def _stage3c_states(integ):
    return {c["check"].split(" ")[0]: (c["state"], c["observed"]["reason"]) for c in integ["checks"]
            if c["category"] == "stage3c"}


def test_integrity_additions_fail_on_failed_attempt_1():
    """The frozen failed attempt has no Stage3C record and only 3 B32768 warmups: the added checks must fail."""
    text = (a2.FAILED_ATTEMPT_1 / "modal_session.log").read_text(encoding="utf-8")
    _, integ, summary = r5.analyze(text, write=False)
    assert summary is None and not integ["all_ok"]
    names = {c["check"]: c["state"] for c in integ["checks"]}
    st = _stage3c_states(integ)
    assert st["B512"] == ("failed", "expected exactly one EXP5_STAGE3C record, found 0")
    assert st["A512"] == ("failed", "expected exactly one EXP5_STAGE3C record, found 0")
    assert [s for n, s in names.items() if n.startswith("B32768: all 20 output-token hashes")] == ["failed"]
    assert names["A512: allocator capacity = expected 393024 tokens"] == "passed"
    assert names["B512: allocator capacity = expected 2074592 tokens"] == "passed"


# ------------------------------------------------------------------ strict EXP5_STAGE3C regression tests
RAW_SESSION = a2.RAW_RUN_DIR / "modal_session.log"


def _raw_lines():
    return RAW_SESSION.read_text(encoding="utf-8").splitlines()


def _real(dtype):
    """A real EXP5_STAGE3C line (with its leg prefix) emitted by the H100 run."""
    tag = "[leg4:rabit_kv2] " if dtype == r5.B else "[leg3:bfloat16] "
    return next(ln for ln in _raw_lines() if ln.startswith(tag + "EXP5_STAGE3C="))[len(tag):]


def _cfg():
    return {"impl": "shared_decode", "query_block": 32}


def _check(dtype, lines):
    try:
        recs, err = a2.parse_stage3c_records(lines), None
    except ValueError as e:
        recs, err = None, str(e)
    return a2.validate_stage3c(dtype, recs, err, _cfg(), a2.accepted_shared_decode_sha256())


def _mut(line, fn):
    head, payload = line.split("EXP5_STAGE3C=", 1)
    rec = json.loads(payload)
    fn(rec)
    return f"{head}EXP5_STAGE3C={json.dumps(rec, sort_keys=True)}"


def test_01_02_real_records_pass():
    assert _check(r5.B, ["INFO something", _real(r5.B)]) == (True, "ok")
    assert _check(r5.A, [_real(r5.A), "EXP5_CAPACITY={}"]) == (True, "ok")


def test_03_real_raw_session_all_14_cells_pass():
    if not RAW_SESSION.is_file():
        return
    _, integ, _ = r5.analyze(RAW_SESSION.read_text(encoding="utf-8"), write=False)
    st = _stage3c_states(integ)
    assert len(st) == 14 and all(v == ("passed", "ok") for v in st.values()), st
    dtypes = {label: d for _, label, d, *_ in r5.ALL_CELLS}
    assert sum(dtypes[l] == r5.A for l in st) == 7 and sum(dtypes[l] == r5.B for l in st) == 7


def _session_without(pred, extra=None):
    out = []
    for ln in _raw_lines():
        if pred(ln):
            if extra is not None:
                out.append(extra(ln))
            continue
        out.append(ln)
    return "\n".join(out) + "\n"


def test_04_05_12_session_level_missing_duplicate_wrong_cell():
    if not RAW_SESSION.is_file():
        return
    miss = _session_without(lambda ln: ln.startswith("[leg9:rabit_kv2] EXP5_STAGE3C="))
    st = _stage3c_states(r5.analyze(miss, write=False)[1])
    assert st["B8192"] == ("failed", "expected exactly one EXP5_STAGE3C record, found 0")
    assert sum(v[0] == "passed" for v in st.values()) == 13
    lines = _raw_lines()
    i = next(n for n, ln in enumerate(lines) if ln.startswith("[leg6:bfloat16] EXP5_STAGE3C="))
    dup = "\n".join(lines[:i + 1] + [lines[i]] + lines[i + 1:]) + "\n"
    assert _stage3c_states(r5.analyze(dup, write=False)[1])["A2048"] == (
        "failed", "expected exactly one EXP5_STAGE3C record, found 2")
    rabit_payload = _real(r5.B).split("EXP5_STAGE3C=", 1)[1]
    wrong = _session_without(lambda ln: ln.startswith("[leg6:bfloat16] EXP5_STAGE3C="),
                             lambda ln: "[leg6:bfloat16] EXP5_STAGE3C=" + rabit_payload)
    state, why = _stage3c_states(r5.analyze(wrong, write=False)[1])["A2048"]
    assert state == "failed" and "unexpected BF16 record fields" in why


def test_06_malformed_json_and_payloads_fail():
    for bad in ("EXP5_STAGE3C={bad json", "EXP5_STAGE3C=", "EXP5_STAGE3C=garbage", "EXP5_STAGE3C=[1, 2]",
                "EXP5_STAGE3C={} trailing"):
        ok, why = _check(r5.B, [bad])
        assert not ok, bad


def test_07_08_09_rabit_field_violations_fail():
    real = _real(r5.B)
    for fn, frag in ((lambda r: r.__setitem__("effective_impl", "tile32"), "implementation"),
                     (lambda r: r.__setitem__("effective_impl", "reference"), "implementation"),
                     (lambda r: r.__setitem__("effective_query_block", 16), "QUERY_BLOCK"),
                     (lambda r: r.__setitem__("requested_query_block", 8), "QUERY_BLOCK"),
                     (lambda r: r["env"].__setitem__("VLLM_RABIT2_STAGE3C_IMPL", None), "selector environment"),
                     (lambda r: r["env"].__setitem__("VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK", None),
                      "selector environment"),
                     (lambda r: r["env"].pop("VLLM_RABIT2_STAGE3C_IMPL"), "selector environment"),
                     (lambda r: r.__setitem__("applicable", False), "implementation"),
                     (lambda r: r.__setitem__("extra_field", 1), "unexpected RABIT record fields"),
                     (lambda r: r.pop("profiling_env"), "unexpected RABIT record fields")):
        ok, why = _check(r5.B, [_mut(real, fn)])
        assert not ok and frag in why, (frag, why)


def test_10_bf16_with_rabit_selector_fails():
    real = _real(r5.A)
    for fn in (lambda r: r["env"].__setitem__("VLLM_RABIT2_STAGE3C_IMPL", "shared_decode"),
               lambda r: r["env"].__setitem__("VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK", "32"),
               lambda r: r.__setitem__("applicable", True)):
        ok, why = _check(r5.A, [_mut(real, fn)])
        assert not ok, why


def test_11_profiling_enabled_fails():
    for dtype in (r5.A, r5.B):
        for flag in ("VLLM_RABIT2_STAGE3C_PROFILE", "VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE"):
            ok, why = _check(dtype, [_mut(_real(dtype), lambda r, f=flag: r["profiling_env"].__setitem__(f, "1"))])
            assert not ok and "profiling" in why, why
        ok, _ = _check(dtype, [_mut(_real(dtype), lambda r: r["profiling_env"].__setitem__(
            "VLLM_RABIT2_STAGE3C_PROFILE", "0"))])
        assert ok  # explicit "0" is off


def test_12_record_validated_against_its_own_cell_type():
    ok, why = _check(r5.A, [_real(r5.B)])  # a RABIT record inside a BF16 cell
    assert not ok and "unexpected BF16 record fields" in why
    ok, why = _check(r5.B, [_real(r5.A)])  # a BF16 record inside a RABIT cell
    assert not ok and "unexpected RABIT record fields" in why


def test_13_shared_decode_sha_mismatch_fails():
    ok, why = _check(r5.B, [_mut(_real(r5.B), lambda r: r.__setitem__("shared_decode_module_sha256", "0" * 64))])
    assert not ok and "shared_decode module SHA" in why
    assert a2.accepted_shared_decode_sha256().startswith("ace859940728")


def test_14_logger_prefix_parses():
    line = "(EngineCore pid=7) INFO 09-27 12:00:00 [worker.py:1] " + _real(r5.B)
    assert _check(r5.B, [line]) == (True, "ok")
    assert _check(r5.B, [_real(r5.B) + "   \r\n"]) == (True, "ok")


def test_15_identifier_prefixed_lookalike_is_not_the_tag():
    payload = _real(r5.B).split("EXP5_STAGE3C=", 1)[1]
    for fake in (f"XEXP5_STAGE3C={payload}", f"_EXP5_STAGE3C={payload}", f"MY_EXP5_STAGE3C={payload}",
                 f"EXP5_STAGE3C_X={payload}", f"EXP5_STAGE3CX={payload}"):
        assert a2.parse_stage3c_records([fake]) == [], fake
    assert _check(r5.B, [f"XEXP5_STAGE3C={payload}"])[1] == "expected exactly one EXP5_STAGE3C record, found 0"
    assert _check(r5.B, [f"XEXP5_STAGE3C={payload}", _real(r5.B)]) == (True, "ok")


def test_reparse_is_offline_and_pinned():
    import ast as _ast
    src = Path(a2.__file__).read_text(encoding="utf-8")
    fn = next(n for n in _ast.parse(src).body if isinstance(n, _ast.FunctionDef) and n.name == "reparse")
    called = {_ast.unparse(n.func) for n in _ast.walk(fn) if isinstance(n, _ast.Call)}
    assert not called & {"r5.execute", "r5.run", "r5.build_snapshot", "r5.stream_command", "stream_command",
                         "r5.preflight", "r5.finalize", "r5.write_manifest", "subprocess.run"}
    assert "r5.analyze" in called and not any(isinstance(n, (_ast.Import, _ast.ImportFrom)) for n in _ast.walk(fn))
    assert a2.RAW_PINNED_SHA256["modal_session.log"] == "efa66e861bbc7a912d5e8b1708f86f10e4614295348d0d38a52647ff69487f0d"
    assert a2.MEASUREMENT_CODE_COMMIT.startswith("6ac5567") and a2.MEASUREMENT_MODAL_APP == "ap-d0K1rXwIzyty6fvusW4ByL"
    if RAW_SESSION.is_file():
        for name, want in a2.RAW_PINNED_SHA256.items():
            assert r5.sha256_raw(a2.RAW_RUN_DIR / name) == want, name


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
