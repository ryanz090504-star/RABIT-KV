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
