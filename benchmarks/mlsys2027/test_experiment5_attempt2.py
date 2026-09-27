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


def test_integrity_additions_fail_on_failed_attempt_1():
    """The frozen failed attempt has no Stage3C record and only 3 B32768 warmups: the added checks must fail."""
    text = (a2.FAILED_ATTEMPT_1 / "modal_session.log").read_text(encoding="utf-8")
    _, integ, summary = r5.analyze(text, write=False)
    assert summary is None and not integ["all_ok"]
    names = {c["check"]: c["state"] for c in integ["checks"]}
    assert names["B512: Stage3C requested == effective == shared_decode, QUERY_BLOCK 32; profiling off"] == "failed"
    assert [s for n, s in names.items() if n.startswith("B32768: all 20 output-token hashes")] == ["failed"]
    assert names["A512: bfloat16 cell carries no Stage3C selection"] == "failed"  # no record -> not proven
    assert names["A512: allocator capacity = expected 393024 tokens"] == "passed"
    assert names["B512: allocator capacity = expected 2074592 tokens"] == "passed"


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
