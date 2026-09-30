"""Offline tests for the Exp14 shape-gate numerical-diagnosis Attempt 2 harness (no GPU, no Modal call).
Run directly or with pytest."""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_exp13_turboquant_probe as p1  # noqa: E402
import run_exp14_shape_gate_numdiag as nd  # noqa: E402
import run_experiment1_quality_frontier as e1  # noqa: E402

ROOT = e1.ROOT
ATTEMPT1_WRAPPER_COMMIT = "e6361ebe3e3fb62ce79719084db9f52630a55b35"


def _attempt1_wrapper() -> bytes:
    return subprocess.run(["git", "show", f"{ATTEMPT1_WRAPPER_COMMIT}:benchmarks/mlsys2027/exp14_shape_gate_numdiag_modal.py"],
                          cwd=ROOT, capture_output=True, check=True).stdout


def test_image_expression_equals_frozen_exp14_image():
    assert p1.image_expr(nd.MODAL_APP) == p1.image_expr(nd.DEPLOY_MODAL)
    full = p1.image_expr(nd.DEPLOY_MODAL) + nd.IMAGE_TAIL
    assert full in nd.MODAL_APP.read_text(encoding="utf-8")
    assert nd.image_expression_equal()
    # the constants the frozen expression references are identical
    for const in ("BASE_COMMIT", "SNAP"):
        def val(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            return next(ast.dump(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and getattr(n.targets[0], "id", None) == const)
        assert val(nd.MODAL_APP) == val(nd.DEPLOY_MODAL), const


def test_wrapper_is_self_contained():
    r = nd.self_contained_report()
    assert r["wrapper_imports_only_stdlib_and_modal"] and r["wrapper_imports_no_sibling_module"], r
    assert r["payload_files"] == nd.EXPECTED_PAYLOAD and r["payload_exactly_expected"]
    assert r["payload_sibling_imports"]["exp14_shape_gate_numdiag.py"] == ["exp14_shape_gate"]
    assert r["payload_sibling_imports"]["exp14_shape_gate.py"] == []
    assert r["payload_sibling_imports_all_shipped"] and r["diagnostic_is_the_only_diagnostic_payload"]
    src = nd.MODAL_APP.read_text(encoding="utf-8")
    assert "exp14_deployment_modal" not in "".join(
        ast.unparse(n) for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom)))


def test_isolated_remote_import_simulation_passes():
    r = nd.isolated_import_simulation()
    assert r["passed"], r
    assert r["deployment_module_importable"] is False and r["loaded_sibling_modules"] == []


def test_isolated_import_catches_the_attempt1_failure():
    """Negative control: the exact Attempt-1 wrapper must FAIL the same simulation with the same error."""
    old = _attempt1_wrapper()
    r = nd.isolated_import_simulation(old)
    assert not r["passed"] and r["returncode"] != 0
    assert "No module named 'exp14_deployment_modal'" in r["stderr_tail"]
    assert "exp14_deployment_modal" in nd._imports(ast.parse(old.decode("utf-8"))) & nd.sibling_modules()


def test_diagnostic_unchanged_since_e6361eb():
    cur = nd.DIAG.read_bytes().replace(b"\r\n", b"\n")
    old = nd._blob(nd.DIAG_COMMIT, nd.DIAG).replace(b"\r\n", b"\n")
    assert hashlib.sha256(cur).hexdigest() == hashlib.sha256(old).hexdigest()


def test_wall_clock_limits():
    assert nd.WALL_CLOCK_LIMIT_S == 45 * 60
    tree = ast.parse(nd.MODAL_APP.read_text(encoding="utf-8"))
    consts = {n.targets[0].id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
              and isinstance(n.value, ast.Constant) and isinstance(n.value.value, int)}
    assert consts["DIAG_SUBPROCESS_TIMEOUT_S"] < consts["FUNCTION_TIMEOUT_S"] <= nd.WALL_CLOCK_LIMIT_S


class FakeModal:
    """In-memory stand-in for `modal app list --json` / `modal app stop` (no Modal call). The app of this attempt
    appears from the SECOND listing on (i.e. after launch), as a real `modal run` creates it."""

    def __init__(self, pre: dict, created: dict | None = None, stop_effective: bool = True):
        self.apps = {k: dict(v) for k, v in pre.items()}
        self.created = created or {}
        self.stop_effective = stop_effective
        self.calls, self.stops = 0, []

    def list(self):
        self.calls += 1
        if self.calls == 2:
            self.apps.update({k: dict(v) for k, v in self.created.items()})
        return {k: dict(v) for k, v in self.apps.items()}

    def stop(self, app_id):
        self.stops.append(app_id)
        if self.stop_effective:
            self.apps[app_id] = {"App ID": app_id, "Description": nd.APP_NAME, "State": "stopped", "Tasks": "0"}
        return {"app_id": app_id, "returncode": 0, "stderr_tail": ""}


def _row(app_id, state, tasks):
    return {"App ID": app_id, "Description": nd.APP_NAME, "State": state, "Tasks": str(tasks)}


OLD_RUNNING = "ap-OldPreExistingAppAAAAAAA1"
NEW_APP = "ap-NewAttemptAppBBBBBBBBBBB2"


def _with_fake(fake, fn):
    orig = nd.list_matching_apps, nd.stop_app
    nd.list_matching_apps, nd.stop_app = fake.list, fake.stop
    try:
        return fn()
    finally:
        nd.list_matching_apps, nd.stop_app = orig


def test_child_env_is_utf8_and_matches_accepted_stream_command():
    env = nd.child_env()
    assert env["PYTHONUTF8"] == "1" and env["PYTHONIOENCODING"] == "utf-8"
    src = (HERE / "run_experiment1_quality_frontier.py").read_text(encoding="utf-8")
    assert 'env["PYTHONUTF8"] = "1"' in src and 'env["PYTHONIOENCODING"] = "utf-8"' in src
    runner = nd.RUNNER.read_text(encoding="utf-8")
    assert 'env=child_env()' in runner and 'encoding="utf-8"' in runner and 'errors="replace"' in runner
    assert "e1.console_write(line)" in runner and "e1.make_console_encoding_safe()" in runner


def test_unicode_child_output_streams_and_child_sees_utf8_env():
    import tempfile
    fake = FakeModal({})
    child = [sys.executable, "-c", "import os, sys; print('\\u2713 Initialized \\u2713'); "
                                   "print('ENV', os.environ.get('PYTHONUTF8'), os.environ.get('PYTHONIOENCODING'), "
                                   "sys.stdout.encoding)"]
    with tempfile.TemporaryDirectory() as t:
        run = _with_fake(fake, lambda: nd.run_with_wall_clock(child, Path(t) / "log.txt", limit_s=60))
        text = (Path(t) / "log.txt").read_text(encoding="utf-8")
    assert run["returncode"] == 0 and not run["timed_out"]
    assert "\u2713 Initialized \u2713" in text and "ENV 1 utf-8 utf-8" in text


def test_parent_with_strict_gbk_stdout_does_not_raise():
    """Reproduces Attempt 2's host condition in a child test process: the PARENT's stdout is strict GBK and the
    streamed client output contains U+2713. The runner must complete without UnicodeEncodeError."""
    code = (
        "import io, sys, tempfile\n"
        "from pathlib import Path\n"
        "sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gbk', errors='strict', line_buffering=True)\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "import run_exp14_shape_gate_numdiag as nd\n"
        "nd.list_matching_apps = lambda: {}\n"
        "nd.stop_app = lambda a: {}\n"
        "d = tempfile.mkdtemp()\n"
        "run = nd.run_with_wall_clock([sys.executable, '-c', \"print('\\\\u2713 Initialized')\"], Path(d) / 'l.txt', 60)\n"
        "assert '\\u2713' in (Path(d) / 'l.txt').read_text(encoding='utf-8')\n"
        "print('PARENT_OK', run['returncode'])\n"
    )
    env = {k: v for k, v in __import__("os").environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONIOENCODING"] = "gbk"
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, timeout=120)
    out = p.stdout.decode("gbk", errors="replace")
    assert p.returncode == 0 and "PARENT_OK 0" in out, (p.returncode, out[-500:], p.stderr.decode("utf-8", "replace")[-1500:])
    assert b"UnicodeEncodeError" not in p.stderr


def test_gbk_negative_control_attempt2_runner_fails():
    """The exact Attempt-2 runner (d67ffce) under the same GBK host condition must reproduce Attempt 2's failure
    (the U+2713 line is lost to a UnicodeEncodeError in the client), proving the test above is not vacuous."""
    import tempfile
    old = subprocess.run(["git", "show", "d67ffcec7ce128010c723e2c59a9a348555e46ca:"
                          "benchmarks/mlsys2027/run_exp14_shape_gate_numdiag.py"],
                         cwd=ROOT, capture_output=True, check=True).stdout
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "old_numdiag_runner.py").write_bytes(old)
        code = (
            "import io, sys, tempfile\n"
            "from pathlib import Path\n"
            "sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gbk', errors='strict', line_buffering=True)\n"
            f"sys.path[:0] = [{t!r}, {str(HERE)!r}]\n"
            "import old_numdiag_runner as nd\n"
            "nd.stop_app = lambda a: {}\n"
            "nd.modal_app_state = lambda a: None\n"
            "d = tempfile.mkdtemp()\n"
            "run = nd.run_with_wall_clock([sys.executable, '-c', \"print('\\\\u2713 Initialized')\"], Path(d) / 'l.txt', 60)\n"
            "log = (Path(d) / 'l.txt').read_text(encoding='utf-8', errors='replace')\n"
            "print('OLD_RC', run['returncode'], 'CHECKMARK_LOGGED', '\\u2713' in log, 'ENCODE_ERROR', 'UnicodeEncodeError' in log)\n"
        )
        env = {k: v for k, v in __import__("os").environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
        env["PYTHONIOENCODING"] = "gbk"
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, timeout=120)
    out = p.stdout.decode("gbk", errors="replace")
    assert "OLD_RC 1 CHECKMARK_LOGGED False ENCODE_ERROR True" in out, (out[-800:], p.stderr[-800:])


def test_client_dies_before_printing_app_id_new_app_still_found_and_stopped():
    import tempfile
    fake = FakeModal({OLD_RUNNING: _row(OLD_RUNNING, "ephemeral", 1)}, {NEW_APP: _row(NEW_APP, "ephemeral", 1)})
    child = [sys.executable, "-c", "raise SystemExit(1)"]  # no output at all
    with tempfile.TemporaryDirectory() as t:
        run = _with_fake(fake, lambda: nd.run_with_wall_clock(child, Path(t) / "log.txt", limit_s=60))
    a = run["apps"]
    assert a["parsed_stdout_app_ids"] == [] and a["pre_launch_app_ids"] == [OLD_RUNNING]
    assert a["new_app_ids"] == [NEW_APP] and fake.stops == [NEW_APP]  # only the new app is stopped
    assert OLD_RUNNING not in fake.stops and fake.apps[OLD_RUNNING]["State"] == "ephemeral"  # pre-existing untouched
    assert a["final_states"][NEW_APP]["State"] == "stopped" and a["cleanup_verified"]


def test_timeout_kills_client_and_stops_discovered_app():
    import tempfile
    fake = FakeModal({OLD_RUNNING: _row(OLD_RUNNING, "ephemeral", 1)}, {NEW_APP: _row(NEW_APP, "ephemeral", 1)})
    hang = [sys.executable, "-c", "import time; time.sleep(120)"]  # never prints an app id
    with tempfile.TemporaryDirectory() as t:
        run = _with_fake(fake, lambda: nd.run_with_wall_clock(hang, Path(t) / "log.txt", limit_s=3))
    assert run["timed_out"] is True and run["elapsed_s"] < 60 and run["wall_clock_limit_s"] == 3
    assert fake.stops == [NEW_APP] and run["apps"]["cleanup_verified"]
    assert run["apps"]["final_states"][NEW_APP]["Tasks"] == "0"


def test_stdout_parsed_id_is_unioned_but_preexisting_never_stopped():
    import tempfile
    fake = FakeModal({OLD_RUNNING: _row(OLD_RUNNING, "ephemeral", 1)}, {NEW_APP: _row(NEW_APP, "stopped", 0)})
    child = [sys.executable, "-c", f"print('View run at {NEW_APP} and old {OLD_RUNNING}')"]
    with tempfile.TemporaryDirectory() as t:
        run = _with_fake(fake, lambda: nd.run_with_wall_clock(child, Path(t) / "log.txt", limit_s=60))
    a = run["apps"]
    assert set(a["parsed_stdout_app_ids"]) == {NEW_APP, OLD_RUNNING} and a["new_app_ids"] == [NEW_APP]
    assert fake.stops == []  # the new app was already stopped with 0 tasks; the old one is never touched


def test_final_task_count_is_checked():
    fake = FakeModal({}, {NEW_APP: _row(NEW_APP, "ephemeral", 1)}, stop_effective=False)
    fake.calls = 1  # the created app is listed from the next call on (post-launch)
    res = _with_fake(fake, lambda: nd.cleanup_new_apps({}, set(), wait_s=0, poll_s=0))
    assert fake.stops == [NEW_APP] and res["cleanup_verified"] is False  # still 1 task -> not verified
    runner = nd.RUNNER.read_text(encoding="utf-8")
    assert 'and run["apps"]["cleanup_verified"]' in runner  # 'completed' requires verified cleanup


def _wrapper_module():
    """Import the Attempt-4 wrapper in-process (dummy snapshot path; importing only builds lazy Modal objects)."""
    import importlib.util
    import os
    import tempfile
    dummy = Path(tempfile.mkdtemp()) / "dummy.zip"
    dummy.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    os.environ["EXP14_VLLM_SNAPSHOT"] = str(dummy)
    spec = importlib.util.spec_from_file_location("exp14_numdiag_wrapper_under_test", nd.MODAL_APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compare(x):
    return {"ref_max_abs": 1.0 + x, "max_abs_err_vs_fp32_ref": 0.001, "max_rel_err": None if x < 0 else 0.01,
            "max_err_in_bf16_ulp_at_ref": 0.5, "max_abs_err_vs_dtype_rounded_ref": 0.0, "exact_equal_elements": 3584,
            "elements": 3584, "max_ulp": 0, "ulp_hist": {"0": 3584, "1": 0, "2": 0, "3+": 0}}


def _synthetic_summary(pad_chars: int = 0) -> dict:
    """A COMPLETE synthetic result in the frozen diagnostic's schema (all geometries / replays / checkpoints)."""
    geos = {}
    for g in nd.EXPECTED_GEOMETRIES:
        reps = {}
        for name, ts in nd.expected_replays().items():
            rows = []
            for t in ts:
                row = {"T": t, "closed_pages": max(0, (t - 4) // 32), "open_tokens": (t - 4) % 32 if t > 4 else 0,
                       "runtime_output_dtype": "torch.bfloat16", "bytes_identical": True,
                       "gate_attempt1_max_abs_runtime_state": 0.001, "gate_attempt1_max_abs_reference_state": 0.001,
                       "gate_attempt1_would_fail": False}
                for ref in nd.REFERENCES:
                    row[ref] = {"reference_output_dtype": "torch.float32", "states_identical": True,
                                "runtime_state": _compare(t % 3), "reference_state": _compare(t % 3)}
                rows.append(row)
            reps[name] = {"summary": {"checkpoints": len(rows)}, "checkpoints": rows}
        geos[g] = {"replays": reps, "overall": {"checkpoints": sum(len(r["checkpoints"]) for r in reps.values())}}
    out = {"completed": True, "non_evidence": True, "rel_floor": 0.01, "geometries": geos}
    if pad_chars:
        out["unicode_padding_\u2713"] = ("\u2713\u4e2d\u00e9" * (pad_chars // 3 + 1))[:pad_chars]
    return out


def test_large_result_capture_roundtrip():
    """>1 MB structured result through the SAME container-side split and local write path: no truncation, valid JSON,
    exact content equality, matching SHA-256, Unicode-safe, no console output involved."""
    import json
    import tempfile
    w = _wrapper_module()
    src = _synthetic_summary(pad_chars=700_000)
    text = json.dumps(src, sort_keys=True)  # the diagnostic's own serialisation (json.dumps(..., sort_keys=True))
    assert len(text.encode("utf-8")) > 1_000_000
    stdout = "model_b_qwen2_5_7b: {}\ncontrol_llama3_1_8b: {}\n" + w.SUMMARY_PREFIX + text + "\n"
    got, rest = w.split_summary(stdout)
    assert got == text and w.SUMMARY_PREFIX not in rest  # the summary is removed from what gets logged
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "sub" / "numdiag_result.json"
        meta = w.write_capture(got, str(path))
        data = path.read_bytes()
        assert data.decode("utf-8") == text and json.loads(data.decode("utf-8")) == src
        assert meta["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest() == hashlib.sha256(data).hexdigest()
        assert meta["bytes"] == len(data) > 1_000_000 and meta["roundtrip_identical"] and meta["completed"]
        assert meta["rows"] == nd.completeness_gate(src)["expected_rows"] == 254
        v = nd.validate_result_file(path)
        assert v["sha256"] == meta["sha256"] and v["completeness"]["passed"], v["completeness"]["missing"][:5]
    wrapper_src = nd.MODAL_APP.read_text(encoding="utf-8")
    assert "print(rest[-20000:]" in wrapper_src and "print(p.stdout" not in wrapper_src  # full summary never logged
    assert 'print("EXP14_NUMDIAG_CAPTURE=" + json.dumps(meta, sort_keys=True)' in wrapper_src
    assert '"summary_text": summary_text' in wrapper_src  # returned through the Modal result channel


def test_completeness_gate_accepts_complete_and_rejects_gaps():
    import copy
    ok = _synthetic_summary()
    r = nd.completeness_gate(ok)
    assert r["passed"] and r["rows"] == r["expected_rows"] == 254
    g0, rep0 = nd.EXPECTED_GEOMETRIES[1], "boundary_P4"
    bad = copy.deepcopy(ok); bad["geometries"][g0]["replays"][rep0]["checkpoints"].pop(3)  # noqa: E702
    assert not nd.completeness_gate(bad)["passed"]
    bad = copy.deepcopy(ok); del bad["geometries"][g0]["replays"][rep0]["checkpoints"][2]["R_sem"]["runtime_state"]["max_ulp"]  # noqa: E702,E501
    r = nd.completeness_gate(bad)
    assert not r["passed"] and any("R_sem/runtime_state: max_ulp" in m for m in r["missing"])
    bad = copy.deepcopy(ok); del bad["geometries"]["control_llama3_1_8b"]  # noqa: E702
    assert not nd.completeness_gate(bad)["passed"]
    bad = copy.deepcopy(ok); bad["geometries"][g0]["replays"]["main_P2048_seed140001"]["checkpoints"][0]["T"] = 2047  # noqa: E702,E501
    assert not nd.completeness_gate(bad)["passed"]
    bad = copy.deepcopy(ok); del bad["geometries"][g0]["replays"][rep0]["checkpoints"][0]["runtime_output_dtype"]  # noqa: E702,E501
    assert not nd.completeness_gate(bad)["passed"]
    bad = copy.deepcopy(ok); bad["completed"] = False  # noqa: E702
    assert not nd.completeness_gate(bad)["passed"]


def test_preflight_dry_run_validation():
    """The diagnostic is COMPLETE (valid Attempt 4, commit 4947050) and runs once: its preflight must now refuse --
    Attempt 4 exists (run-once guard) and the shape gate is no longer the pre-amendment version the diagnostic was
    pinned to (the post-failure correctness-criterion amendment changed only the gate's attention criterion)."""
    try:
        nd.preflight(dry_run=True)  # read-only; no Modal call
    except RuntimeError as e:
        msg = str(e)
        assert "attempt4_not_already_run" in msg and "shape_gate_unchanged" in msg, msg[:400]
        for k in ("wrapper_self_contained", "image_expression_equal_to_frozen_exp14", "isolated_remote_import_simulation",
                  "diagnostic_unchanged_since_e6361eb", "vllm_kvquant_tree_unchanged", "rabit_source_unchanged",
                  "qwen_revision_unchanged", "exp13_evidence_unchanged", "attempt1_archive_unchanged",
                  "attempt2_archive_unchanged", "attempt3_archive_unchanged"):
            assert f"'{k}'" not in msg.split("\n")[0], k  # every other pre-run check still holds
    else:
        raise AssertionError("the completed diagnostic's preflight accepted a re-run")


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
