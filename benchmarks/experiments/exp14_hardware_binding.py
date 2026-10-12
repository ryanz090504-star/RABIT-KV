"""
MLSys 2027 Experiment 14 -- SCHEDULER-ONLY hardware binding amendment: machine checks (stdlib only).

After serving Attempt 1 was invalid (Modal placed the permissive gpu="H100" request on an NVIDIA H200 while the frozen
protocol requires one H100 80GB), the accepted non-evidence probe (0303a5e; ran on an H100 80GB) stays valid for a
deployment wrapper that differs from the probe-bound one by EXACTLY two documented changes:
  1. the Modal GPU selector  gpu="H100"  ->  gpu="H100!:1"  (strict: no automatic H200 upgrade; one GPU);
  2. one fail-closed hardware guard block at the top of mirrored() (before any gate, engine or measurement).
The runner changes ONLY in its probe-prerequisite binding (to accept exactly this recorded amendment). No protocol,
image, package, snapshot, model, gate, worker, engine, prompt, seed, warmup, repetition, order or accounting change.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ACCEPTED_PROBE_COMMIT = "0303a5ed5a6b4a39171b3f6b2c12536b33abe116"
DEPLOYMENT_PATH = "benchmarks/mlsys2027/exp14_deployment_modal.py"
RUNNER_PATH = "benchmarks/mlsys2027/run_experiment14_second_model.py"
SELECTOR_OLD, SELECTOR_NEW = "H100", "H100!:1"
H100_80GB_MIB_RANGE = (79 * 1024, 82 * 1024)  # nvidia-smi memory.total of an H100 80GB is ~81,559 MiB

# The exact guard block inserted at the top of mirrored() (after its docstring), verified by AST equality.
GUARD_SRC = '''
_gpus = _gpu_query()
_hw_ok = (len(_gpus) == 1 and "H100" in _gpus[0].get("name", "")
          and 79 * 1024 <= int(float(_gpus[0].get("memory.total", 0))) <= 82 * 1024)
_emit("EXP14_HARDWARE_CHECK", {"gpus": _gpus, "passed": _hw_ok, "required": "exactly 1 x NVIDIA H100 80GB"})
if not _hw_ok:
    _emit("EXP14_HARDWARE_MISMATCH", {"gpus": _gpus})
    raise RuntimeError(f"hardware mismatch: {_gpus}; no gate, engine or measurement run")
'''
# Top-level names of the runner that the amendment may add or change (prerequisite binding only).
RUNNER_ALLOWED_NAMES = {"check_probe_prerequisite", "HW_BINDING_AMENDMENT", "hardware_binding_amendment_ok",
                        "exp14_hardware_binding"}


def guard_statements() -> list:
    return ast.parse(GUARD_SRC).body


def hardware_ok(gpus: list[dict]) -> bool:
    """The guard's predicate, evaluated from GUARD_SRC itself (so the test exercises the shipped expression)."""
    assign = next(n for n in guard_statements() if isinstance(n, ast.Assign) and n.targets[0].id == "_hw_ok")
    return bool(eval(compile(ast.Expression(assign.value), "<guard>", "eval"), {"_gpus": gpus, "int": int,  # noqa: S307
                                                                                    "float": float, "len": len}))


def _mirrored(tree: ast.Module) -> ast.FunctionDef:
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "mirrored")


def deployment_diff(old_src: str, new_src: str) -> dict:
    """Require new == old except (1) the selector and (2) the guard block right after mirrored()'s docstring."""
    old, new = ast.parse(old_src), ast.parse(new_src)
    fn = _mirrored(new)
    guard = [ast.dump(s) for s in guard_statements()]
    body = fn.body
    has_doc = isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
    start = 1 if has_doc else 0
    guard_found = [ast.dump(s) for s in body[start:start + len(guard)]] == guard
    selector = None
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Call):
            for kw in dec.keywords:
                if kw.arg == "gpu" and isinstance(kw.value, ast.Constant):
                    selector = kw.value.value
                    kw.value.value = SELECTOR_OLD  # normalise for the comparison below
    if guard_found:
        del body[start:start + len(guard)]
    old_sel = next((kw.value.value for dec in _mirrored(old).decorator_list if isinstance(dec, ast.Call)
                    for kw in dec.keywords if kw.arg == "gpu"), None)
    rest_identical = ast.dump(new) == ast.dump(old)
    return {"old_selector": old_sel, "new_selector": selector, "guard_block_found": guard_found,
            "everything_else_identical": rest_identical,
            "passed": old_sel == SELECTOR_OLD and selector == SELECTOR_NEW and guard_found and rest_identical}


def _top_names(node: ast.stmt) -> set:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    if isinstance(node, ast.Import):
        return {a.name.split(".")[0] for a in node.names}  # the imported MODULE (not its alias)
    if isinstance(node, ast.ImportFrom):
        return {node.module.split(".")[0]} if node.module else set()
    return set()


def runner_diff(old_src: str, new_src: str) -> dict:
    """Every top-level statement identical except statements that only define RUNNER_ALLOWED_NAMES."""
    def keep(tree):
        return [ast.dump(n) for n in tree.body if not (_top_names(n) and _top_names(n) <= RUNNER_ALLOWED_NAMES)]
    old, new = ast.parse(old_src), ast.parse(new_src)
    changed = sorted({nm for n in new.body for nm in _top_names(n) if nm in RUNNER_ALLOWED_NAMES})
    same = keep(old) == keep(new)
    return {"changed_or_added_names": changed, "everything_else_identical": same, "passed": same}


def load_amendment(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
