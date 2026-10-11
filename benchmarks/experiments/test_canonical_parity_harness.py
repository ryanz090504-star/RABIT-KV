"""Offline pre-launch tests for the canonical-quality-v2 CPU parity harness (no Modal call, no GPU, no model).
Run directly or with pytest."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WRAPPER = HERE / "canonical_quality_parity_modal.py"
ATTEMPT1_COMMIT = "c36069781b259e2e9b4d8adf60b7c58ea5df5cac"
REQUIRED_FILES = {"benchmarks/mlsys2027/canonical_rabit_quality.py",
                  "benchmarks/mlsys2027/canonical_quality_parity_tests.py",
                  "vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py", "benchmarks/quality/hotpotqa.py"}

# Simulates the REMOTE import: only the wrapper file in an otherwise empty directory, repository directories removed
# from sys.path, modal.is_local() forced False (as inside a Modal container), and __file__ = /root/<wrapper>.
REMOTE_IMPORT = r'''
import json, sys, types
tmp, src_path, repo = sys.argv[1], sys.argv[2], sys.argv[3]
norm = lambda p: p.replace("\\", "/").lower()
sys.path[:] = [tmp] + [p for p in sys.path[1:] if p and norm(repo) not in norm(p)]
import modal
modal.is_local = lambda: False
src = open(src_path, encoding="utf-8").read()
g = {"__name__": "canonical_quality_parity_modal", "__file__": "/root/canonical_quality_parity_modal.py"}
out = {}
try:
    exec(compile(src, "/root/canonical_quality_parity_modal.py", "exec"), g)
    out["import_ok"] = True
except Exception as e:
    out["import_ok"] = False
    out["error"] = f"{type(e).__name__}: {e}"
out["has_app"] = "app" in g
out["has_parity"] = "parity" in g
leaked = []
for k, v in g.items():
    if isinstance(v, (str,)) or type(v).__name__ in ("WindowsPath", "PosixPath"):
        if norm(repo) in norm(str(v)):
            leaked.append(k)
out["local_repo_path_in_module_globals"] = leaked
out["sibling_modules_loaded"] = sorted(m for m in sys.modules if m.startswith(("canonical_", "run_", "exp14_")))
print("REMOTE_IMPORT=" + json.dumps(out))
'''


def remote_import(source: bytes) -> dict:
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "canonical_quality_parity_modal.py"
        p.write_bytes(source)
        r = subprocess.run([sys.executable, "-c", REMOTE_IMPORT, t, str(p), str(ROOT)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=t, timeout=300)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("REMOTE_IMPORT=")), None)
    return json.loads(line.split("=", 1)[1]) if line else {"import_ok": False, "error": r.stderr[-800:]}


def attempt1_wrapper() -> bytes:
    return subprocess.run(["git", "show", f"{ATTEMPT1_COMMIT}:benchmarks/mlsys2027/canonical_quality_parity_modal.py"],
                          cwd=ROOT, capture_output=True, check=True).stdout


def test_isolated_remote_import_passes():
    r = remote_import(WRAPPER.read_bytes())
    assert r["import_ok"] and r["has_app"] and r["has_parity"], r
    assert r["local_repo_path_in_module_globals"] == [] and r["sibling_modules_loaded"] == [], r


def test_attempt1_wrapper_fails_the_same_simulation():
    r = remote_import(attempt1_wrapper())
    assert not r["import_ok"] and r["error"].startswith("IndexError"), r


def test_file_shipping_complete():
    tree = ast.parse(WRAPPER.read_text(encoding="utf-8"))
    files = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], "id", None) == "FILES")
    assert set(files) == REQUIRED_FILES and all((ROOT / f).is_file() for f in files)
    src = WRAPPER.read_text(encoding="utf-8")
    assert 'f"{REMOTE_REPO}/{_rel}"' in src and 'REMOTE_REPO = "/repo"' in src
    # the parity suite reads exactly these repo-relative paths under the root it is given (/repo remotely)
    tests = (HERE / "canonical_quality_parity_tests.py").read_text(encoding="utf-8")
    assert 'repo_root / "vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py"' in tests
    assert 'repo_root / "benchmarks/quality/hotpotqa.py"' in tests
    # runtime imports of the shipped Python files: stdlib, torch, transformers (installed) or shipped modules only
    allowed = {"__future__", "ast", "hashlib", "json", "sys", "pathlib", "typing", "torch", "transformers",
               "canonical_rabit_quality"}
    for rel in ("benchmarks/mlsys2027/canonical_rabit_quality.py", "benchmarks/mlsys2027/canonical_quality_parity_tests.py"):
        mods = set()
        for n in ast.walk(ast.parse((ROOT / rel).read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module.split(".")[0])
        assert mods <= allowed, (rel, mods - allowed)


def test_wrapper_compiles_and_has_no_module_level_ancestry():
    src = WRAPPER.read_text(encoding="utf-8")
    compile(src, str(WRAPPER), "exec")
    tree = ast.parse(src)
    for node in tree.body:  # only the is_local() block may reference the __file__ name
        uses = any(isinstance(x, ast.Name) and x.id == "__file__" for x in ast.walk(node))
        if uses:
            assert isinstance(node, ast.If) and ast.unparse(node.test) == "modal.is_local()", ast.unparse(node)[:80]
    assert any(isinstance(n, ast.If) and ast.unparse(n.test) == "modal.is_local()" for n in tree.body)



# ------------------------------------------------------------------ Attempt-3 transport / cleanup tests (zero torch)
def _wrapper_local():
    import importlib.util
    spec = importlib.util.spec_from_file_location("parity_wrapper_under_test", WRAPPER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("parity_runner_under_test", HERE / "run_canonical_quality_parity.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _synthetic_result():
    geo = {"T1_full_state": {"normal": {"1": [], "75": ["\u2713 decoded V"]}},
           "T2_sequential_aging": {"normal": {"0": []}}, "T3_hf_cache": {"normal": {"1": []}},
           "T4_old_harness": {"normal": [{"n": 36, "old_k_equals_canonical": True, "old_v_equals_canonical": False}]},
           "summary": {"parity_failures": 0, "T4_old_v_mismatch_min_n": None, "ratio": 0.5}}
    return {"oracle": {"file_sha256_lf": "ab", "extracted_sha256": "cd", "functions": ["f"]}, "lengths": [1, 75],
            "prefills": [0], "geometries": {"qwen2_5_7b": geo, "llama3_1_8b": json.loads(json.dumps(geo))},
            "passed": True, "negative_control_passed": True, "torch_version": "2.11.0+cpu", "python_version": "3.11.9"}


def test_strict_json_transport_roundtrip():
    import hashlib
    w = _wrapper_local()
    res = _synthetic_result()
    payload = w.serialize_result(res)
    assert type(payload) is str
    back = w.validate_payload(payload)
    assert back == res and json.loads(payload) == res
    h1 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert h1 == hashlib.sha256(w.serialize_result(_synthetic_result()).encode("utf-8")).hexdigest()  # stable
    assert "\u2713" in payload  # Unicode carried as-is


def test_strict_transport_rejects_non_json_native():
    w = _wrapper_local()

    class FakeTorchVersion(str):  # str subclass, like torch.torch_version.TorchVersion
        pass

    class Custom:
        pass

    for bad in (FakeTorchVersion("2.11.0"), {1, 2}, (1, 2), Custom(), float("nan")):
        res = _synthetic_result()
        res["geometries"]["qwen2_5_7b"]["summary"]["x"] = bad
        try:
            w.serialize_result(res)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"strict serializer accepted {type(bad).__qualname__}")
    try:  # plain strict json.dumps also refuses a set (no default= fallback anywhere)
        json.dumps({"x": {1}})
    except TypeError:
        pass
    else:
        raise AssertionError("json.dumps accepted a set")
    try:
        w.validate_payload(b"{}")
    except TypeError:
        pass
    else:
        raise AssertionError("non-str payload accepted")
    bad_schema = _synthetic_result()
    del bad_schema["geometries"]["llama3_1_8b"]
    try:
        w.validate_payload(json.dumps(bad_schema))
    except ValueError:
        pass
    else:
        raise AssertionError("schema gap accepted")
    src = WRAPPER.read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "serialize_result")
    dumps = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and ast.unparse(c.func) == "json.dumps"]
    assert len(dumps) == 1 and "default" not in {k.arg for k in dumps[0].keywords}
    assert 'res["torch_version"] = str(torch.__version__)' in src and "return payload" in src


def test_cleanup_polling():
    r = _runner()
    t = {"now": 0.0}
    clock = lambda: t["now"]  # noqa: E731

    def sleep(dt):
        t["now"] += dt

    seq = iter([{"ap-A": {"State": "ephemeral", "Tasks": "1"}, "ap-OLD": {"State": "ephemeral", "Tasks": "1"}},
                {"ap-A": {"State": "stopped", "Tasks": "1"}},
                {"ap-A": {"State": "stopped", "Tasks": "0"}}])
    out = r.poll_cleanup(["ap-A"], lambda: next(seq), lambda a: (_ for _ in ()).throw(AssertionError("no stop")),
                         timeout_s=60, poll_s=3, sleep=sleep, clock=clock)
    assert out["verified"] and out["polls"] == 3 and out["first_observed_states"]["ap-A"]["Tasks"] == "1"
    assert out["final_states"]["ap-A"] == {"State": "stopped", "Tasks": "0"} and out["elapsed_s"] == 6.0
    stops = []
    t["now"] = 0.0
    stuck = r.poll_cleanup(["ap-B"], lambda: {"ap-B": {"State": "ephemeral", "Tasks": "1"}}, stops.append,
                           timeout_s=60, poll_s=3, sleep=sleep, clock=clock)
    assert stops == ["ap-B"] and not stuck["verified"] and stuck["stopped_again_after_timeout"] == ["ap-B"]


def test_planned_cases_match_frozen_suite():
    r = _runner()
    tree = ast.parse((HERE / "canonical_quality_parity_tests.py").read_text(encoding="utf-8"))
    const = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             and getattr(n.targets[0], "id", None) in ("LENGTHS", "DISTRIBUTIONS", "PREFILLS")}
    nl, nd, npre = len(const["LENGTHS"]), len(const["DISTRIBUTIONS"]), len(const["PREFILLS"])
    assert r.PLANNED_PER_GEOMETRY == {"T1_full_state": nd * nl, "T2_sequential_aging": nd * npre,
                                      "T3_hf_cache": 2 * 3, "T4_old_harness": nd * nl}


def test_no_ppl_files_in_parity_workflow():
    assert not (HERE / "canonical_quality_v2").exists() and not (HERE / "gen_canonical_ppl_v2.py").exists()


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
