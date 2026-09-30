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


def test_timeout_kills_client_and_stops_app(tmp_path=None):
    """A fake client that prints an app id and hangs: the runner must time out, kill it, stop THAT app and record the
    final state (stop / state calls are faked -- no Modal call)."""
    import tempfile
    calls = {}
    orig_stop, orig_state = nd.stop_app, nd.modal_app_state
    try:
        nd.stop_app = lambda app_id: calls.setdefault("stop", app_id) and {"attempted": True, "returncode": 0}
        nd.modal_app_state = lambda app_id: {"App ID": app_id, "State": "stopped", "Tasks": "0"}
        fake = [sys.executable, "-c", "import time; print('View run at ap-AbCdEfGhIjKlMnOpQrStUv12', flush=True); "
                                      "time.sleep(120)"]
        with tempfile.TemporaryDirectory() as t:
            run = nd.run_with_wall_clock(fake, Path(t) / "log.txt", limit_s=3)
            assert "ap-AbCdEfGhIjKlMnOpQrStUv12" in (Path(t) / "log.txt").read_text(encoding="utf-8")
        assert run["timed_out"] is True and run["elapsed_s"] < 60
        assert run["app_id"] == "ap-AbCdEfGhIjKlMnOpQrStUv12" and calls["stop"] == run["app_id"]
        assert run["final_app_state"]["State"] == "stopped"
        # a non-zero exit also stops the app defensively
        calls.clear()
        fail = [sys.executable, "-c", "print('ap-ZyXwVuTsRqPoNmLkJiHgFe98'); raise SystemExit(1)"]
        with tempfile.TemporaryDirectory() as t:
            run = nd.run_with_wall_clock(fail, Path(t) / "log.txt", limit_s=60)
        assert run["timed_out"] is False and run["returncode"] == 1 and calls["stop"] == "ap-ZyXwVuTsRqPoNmLkJiHgFe98"
    finally:
        nd.stop_app, nd.modal_app_state = orig_stop, orig_state


def test_preflight_dry_run_validation():
    prov = nd.preflight(dry_run=True)  # read-only; no Modal call
    c = prov["checks"]
    for k in ("wrapper_self_contained", "image_expression_equal_to_frozen_exp14", "isolated_remote_import_simulation",
              "diagnostic_unchanged_since_e6361eb", "shape_gate_unchanged", "vllm_kvquant_tree_unchanged",
              "rabit_source_unchanged", "qwen_revision_unchanged", "exp13_evidence_unchanged",
              "exp1_12_evidence_unchanged", "attempt1_archive_unchanged"):
        assert c[k], k


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
