"""
RABIT-KV MLSys 2027 -- Experiment 14 NON-EVIDENCE shape-gate numerical diagnosis runner, ATTEMPT 2 (Attempt 1 is
permanently invalid: its wrapper imported a sibling module absent in the container; 39 container starts failed).

One Modal H100 session running the UNCHANGED exp14_shape_gate_numdiag.py (e6361eb) through the SELF-CONTAINED wrapper
exp14_shape_gate_numdiag_modal.py; no model, no engine, no download, no serving, no quality.

Protections added after Attempt 1:
  * pre-run validation (clean tree; wrapper self-contained; image expression == frozen Exp14 image; isolated
    remote-import simulation; diagnostic / shape-gate / RABIT / vllm-kvquant / model-revision / evidence unchanged);
  * a HARD wall-clock limit (WALL_CLOCK_LIMIT_S): on expiry the local client is killed, the Modal app is stopped
    (`modal app stop -y <app id>`), the timeout is recorded as a harness failure, and the run STOPS; the app's final
    state / task count is recorded in every outcome.
Writes results/mlsys2027/second_model/shape_gate_numdiag/attempt_2/. Runs once; never overwrites.

Usage:
    python benchmarks/mlsys2027/run_exp14_shape_gate_numdiag.py --dry-run
    python benchmarks/mlsys2027/run_exp14_shape_gate_numdiag.py
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp14_model_snapshot as ms  # noqa: E402  (frozen Model-B identity; read-only)
import run_exp13_turboquant_probe as p1  # noqa: E402  (image-expression extractor; read-only)
import run_experiment1_quality_frontier as e1  # noqa: E402  (read-only helpers)
import run_experiment14_second_model as r14  # noqa: E402  (read-only: snapshot builder, protected paths)

ROOT = e1.ROOT
RUNNER = Path(__file__).resolve()
MODAL_APP = HERE / "exp14_shape_gate_numdiag_modal.py"
DIAG = HERE / "exp14_shape_gate_numdiag.py"
DEPLOY_MODAL = HERE / "exp14_deployment_modal.py"
TEST_FILE = HERE / "test_exp14_shape_gate_numdiag.py"
BASE = ROOT / "results" / "mlsys2027" / "second_model" / "shape_gate_numdiag"
OUT_DIR = BASE / "attempt_2"
ATTEMPT1_DIR = BASE / "failed_attempt_1"
APP_NAME = "rabit-kv-mlsys2027-exp14-shape-gate-numdiag"
WALL_CLOCK_LIMIT_S = 45 * 60
DIAG_COMMIT = "e6361ebe3e3fb62ce79719084db9f52630a55b35"  # the diagnostic script must be byte-identical to this commit
SHAPE_GATE_COMMIT = "35344e0ac8ea5fc3bf4a81d3bc62dfcec2872fc4"  # shape gate as used by probe Attempt 1
MODEL_SNAPSHOT_COMMIT = "35344e0ac8ea5fc3bf4a81d3bc62dfcec2872fc4"
ATTEMPT1_ARCHIVE_COMMIT = "f54df78fa5684cda6b7c03295bbf798729fb8d4f"
EXP13_EVIDENCE_COMMIT = "42c2799f4e7393c6270193a1c852af90eaf7d402"
EXP12_EVIDENCE_COMMIT = "4f767ab03d83e043b2871dd0cd4cf2f8dc862e6b"
EXPECTED_RABIT_SHA256_LF = "7e628c94eebb9fe689bf416ea61f748c0f909a82d0f229c463edd1a0df92e6ae"
EXPECTED_VLLM_TREE = "390fc793d47fb85321d80df03e50da96433e4a95"
IMAGE_TAIL = '.pip_install("pytest", "modelscope")\n)'
# Modules the wrapper may import (stdlib + modal); the payload files appended to the image after the frozen image.
WRAPPER_ALLOWED_IMPORTS = {"__future__", "os", "subprocess", "sys", "pathlib", "modal"}
EXPECTED_PAYLOAD = {"exp14_shape_gate_numdiag.py": "/opt/exp14/exp14_shape_gate_numdiag.py",
                    "exp14_shape_gate.py": "/opt/exp14/exp14_shape_gate.py"}
SUMMARY_RE = re.compile(r"^EXP14_NUMDIAG_SUMMARY=(\{.*\})\s*$")
APP_ID_RE = re.compile(r"ap-[A-Za-z0-9]{20,}")


# ------------------------------------------------------------------------------ self-containment checks (offline)
def _imports(tree: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            out.add(n.module.split(".")[0])
    return out


def sibling_modules() -> set[str]:
    return {p.stem for p in HERE.glob("*.py")}


def image_expression_equal() -> bool:
    return (p1.image_expr(MODAL_APP) == p1.image_expr(DEPLOY_MODAL) and
            (p1.image_expr(MODAL_APP) + IMAGE_TAIL) in MODAL_APP.read_text(encoding="utf-8") and
            (p1.image_expr(DEPLOY_MODAL) + IMAGE_TAIL) in DEPLOY_MODAL.read_text(encoding="utf-8"))


def payload_files() -> dict:
    """{local file name: remote path} appended to the frozen image in the wrapper (resolved from its constants)."""
    tree = ast.parse(MODAL_APP.read_text(encoding="utf-8"))
    consts = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            t = n.targets[0].id
            if isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                consts[t] = n.value.value
            elif (isinstance(n.value, ast.BinOp) and isinstance(n.value.right, ast.Constant)
                  and isinstance(n.value.right.value, str)):
                consts[t] = Path(n.value.right.value).name  # Path(__file__).resolve().parent / "<file>"
    out = {}
    for c in ast.walk(tree):
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "add_local_file":
            a0, a1 = c.args[0], c.args[1]
            if isinstance(a0, ast.Call) and a0.args and isinstance(a0.args[0], ast.Name) and a0.args[0].id != "SNAP":
                out[consts[a0.args[0].id]] = consts[a1.id] if isinstance(a1, ast.Name) else a1.value
    return out


def self_contained_report() -> dict:
    sib = sibling_modules()
    wrap_imports = _imports(ast.parse(MODAL_APP.read_text(encoding="utf-8")))
    payload = payload_files()
    shipped = {Path(k).stem for k in payload}
    payload_sibling_imports = {name: sorted(_imports(ast.parse((HERE / name).read_text(encoding="utf-8"))) & sib)
                               for name in payload}
    return {
        "wrapper_imports": sorted(wrap_imports),
        "wrapper_imports_only_stdlib_and_modal": wrap_imports <= WRAPPER_ALLOWED_IMPORTS,
        "wrapper_imports_no_sibling_module": not (wrap_imports & sib),
        "payload_files": payload,
        "payload_exactly_expected": payload == EXPECTED_PAYLOAD,
        "payload_sibling_imports": payload_sibling_imports,
        "payload_sibling_imports_all_shipped": all(set(v) <= shipped for v in payload_sibling_imports.values()),
        "diagnostic_is_the_only_diagnostic_payload": [k for k in payload if "numdiag" in k] == ["exp14_shape_gate_numdiag.py"],
    }


ISOLATED_IMPORT_CODE = r'''
import importlib, importlib.util, json, sys
tmp = sys.argv[1]
sys.path[:] = [tmp] + [p for p in sys.path[1:] if p and "mlsys2027" not in p.replace("\\", "/")]
out = {"deployment_module_importable": importlib.util.find_spec("exp14_deployment_modal") is not None,
       "any_exp14_sibling_importable": any(importlib.util.find_spec(n) is not None for n in
                                           ("exp14_shape_gate", "exp14_model_snapshot", "exp14_engine_worker",
                                            "run_experiment14_second_model"))}
m = importlib.import_module("exp14_shape_gate_numdiag_modal")
out["import_ok"] = True
out["has_app_function_entrypoint"] = all(hasattr(m, a) for a in ("app", "numdiag", "main", "image"))
out["loaded_sibling_modules"] = sorted(k for k in sys.modules if k.startswith(("exp14_", "run_experiment", "run_exp"))
                                     and k != "exp14_shape_gate_numdiag_modal")
print("ISOLATED=" + json.dumps(out))
'''


def isolated_import_simulation(source: bytes | None = None) -> dict:
    """Import the wrapper from a directory that contains ONLY the wrapper (as in the remote /root module context),
    with benchmarks/mlsys2027 removed from sys.path, so exp14_deployment_modal and every other sibling is unimportable.
    No Modal call: importing the module only builds lazy image / app definitions."""
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / MODAL_APP.name).write_bytes(MODAL_APP.read_bytes() if source is None else source)
        dummy = Path(t) / "dummy_snapshot.zip"
        dummy.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
        env = {**os.environ, "EXP14_VLLM_SNAPSHOT": str(dummy)}
        env.pop("PYTHONPATH", None)
        p = subprocess.run([sys.executable, "-c", ISOLATED_IMPORT_CODE, t], cwd=t, env=env, capture_output=True,
                           text=True, timeout=300)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("ISOLATED=")), None)
        res = json.loads(line.split("=", 1)[1]) if line else {"import_ok": False}
        res.update(returncode=p.returncode, stderr_tail=p.stderr[-1500:] if p.returncode else "")
        res["passed"] = (p.returncode == 0 and res.get("import_ok") is True and
                         res.get("deployment_module_importable") is False and
                         res.get("any_exp14_sibling_importable") is False and
                         res.get("has_app_function_entrypoint") is True and res.get("loaded_sibling_modules") == [])
        return res


# ------------------------------------------------------------------------------------------------- preflight
def _blob(commit: str, path: Path) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT,
                          capture_output=True, check=True).stdout


def _lf(b: bytes) -> str:
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def preflight(dry_run: bool) -> dict:
    status = e1.run_git("status", "--short")
    sc = self_contained_report()
    iso = isolated_import_simulation()
    rabit = _lf((ROOT / "vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py").read_bytes())
    checks = {
        "clean_tree": status == "",
        "wrapper_self_contained": all(v for k, v in sc.items() if isinstance(v, bool)),
        "image_expression_equal_to_frozen_exp14": image_expression_equal(),
        "isolated_remote_import_simulation": iso["passed"],
        "diagnostic_unchanged_since_e6361eb": _lf(DIAG.read_bytes()) == _lf(_blob(DIAG_COMMIT, DIAG)),
        "shape_gate_unchanged": _lf(r14.SHAPE_GATE.read_bytes()) == _lf(_blob(SHAPE_GATE_COMMIT, r14.SHAPE_GATE)),
        "vllm_kvquant_tree_unchanged": e1.run_git("rev-parse", "HEAD:vllm-kvquant") == EXPECTED_VLLM_TREE,
        "vllm_kvquant_unchanged_since_backport": e1.run_git("diff", "--name-only", r14.r13.BACKPORT_COMMIT, "HEAD",
                                                            "--", "vllm-kvquant") == "",
        "rabit_source_unchanged": rabit == EXPECTED_RABIT_SHA256_LF,
        "qwen_revision_unchanged": ms.MODEL_REVISION == "16c174980d8a1492910551634b4969e69cdc2444" and
                                   _lf((HERE / "exp14_model_snapshot.py").read_bytes()) ==
                                   _lf(_blob(MODEL_SNAPSHOT_COMMIT, HERE / "exp14_model_snapshot.py")),
        "exp13_evidence_unchanged": e1.run_git("diff", "--name-only", EXP13_EVIDENCE_COMMIT, "--",
                                               "results/mlsys2027/external_baseline") == "",
        "exp1_12_evidence_unchanged": e1.run_git("diff", "--name-only", EXP12_EVIDENCE_COMMIT, "--", "results",
                                                 ":!results/mlsys2027/external_baseline",
                                                 ":!results/mlsys2027/second_model") == "",
        "attempt1_archive_unchanged": e1.run_git("diff", "--name-only", ATTEMPT1_ARCHIVE_COMMIT, "--",
                                                 *[(ATTEMPT1_DIR / f).relative_to(ROOT).as_posix()
                                                   for f in ("numdiag_session.log", "record.json",
                                                             "attempt_record.json")]) == "",
        "protected_paths_clean": r14.protected_status() == "",
        "attempt2_not_already_run": not (OUT_DIR / "record.json").exists(),
    }
    if dry_run:
        checks["clean_tree"] = True  # reported below; a dry run may be made before committing
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"pre-run validation failed: {failed}\nself_contained={sc}\nisolated_import={iso}\n"
                           f"git status:\n{status}")
    return {"git_head": e1.run_git("rev-parse", "HEAD"), "vllm_kvquant_tree": EXPECTED_VLLM_TREE,
            "rabit_kv2_sha256_lf": rabit, "model_revision": ms.MODEL_REVISION, "checks": checks,
            "self_contained": sc, "isolated_import": iso, "wall_clock_limit_s": WALL_CLOCK_LIMIT_S,
            "harness_sha256": {p.name: e1.sha256(p) for p in (RUNNER, MODAL_APP, DIAG, r14.SHAPE_GATE, TEST_FILE)},
            "tree_clean_at_launch": status == ""}


# ---------------------------------------------------------------------------------------------- execution
def modal_app_state(app_id: str | None) -> dict | None:
    if not app_id:
        return None
    p = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"], capture_output=True, text=True,
                       timeout=180)
    try:
        rows = json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"error": "app list unavailable", "stderr": p.stderr[-500:]}
    return next((r for r in rows if r.get("App ID") == app_id), {"App ID": app_id, "State": "not listed"})


def stop_app(app_id: str | None) -> dict:
    if not app_id:
        return {"attempted": False, "reason": "no app id captured"}
    p = subprocess.run([sys.executable, "-m", "modal", "app", "stop", "-y", app_id], capture_output=True, text=True,
                       timeout=180)
    return {"attempted": True, "returncode": p.returncode, "stderr_tail": p.stderr[-500:]}


def run_with_wall_clock(cmd: list[str], log: Path, limit_s: int) -> dict:
    """Stream the Modal client to `log`; kill it and stop the app when the hard limit expires."""
    t0 = time.time()
    state: dict = {"app_id": None}
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", cwd=ROOT)

        def pump():
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                sys.stdout.write(line)
                if state["app_id"] is None and (m := APP_ID_RE.search(line)):
                    state["app_id"] = m.group(0)

        th = threading.Thread(target=pump, daemon=True)
        th.start()
        timed_out = False
        try:
            rc = proc.wait(timeout=limit_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            rc = proc.wait()
        th.join(timeout=30)
    out = {"returncode": rc, "timed_out": timed_out, "elapsed_s": round(time.time() - t0, 1), "app_id": state["app_id"]}
    if timed_out or rc != 0:
        out["app_stop"] = stop_app(state["app_id"])  # never leave a remote task running
    out["final_app_state"] = modal_app_state(state["app_id"])
    return out


def main(argv=None) -> int:
    e1.make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    prov = preflight(a.dry_run)
    print("Exp14 shape-gate numerical diagnosis ATTEMPT 2 (NON-EVIDENCE) -- pre-run validation passed:",
          json.dumps(prov["checks"]))
    if a.dry_run:
        print(f"tree clean: {prov['tree_clean_at_launch']}; --dry-run: nothing executed.")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    log = OUT_DIR / "numdiag_session.log"
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    rec = {"experiment": 14, "kind": "shape_gate_numerical_diagnosis", "attempt": 2, "non_evidence": True,
           "status": "running", "started_utc": started, **prov}
    (OUT_DIR / "record.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    os.environ["EXP14_VLLM_SNAPSHOT"] = str(r14.build_snapshot())
    run = run_with_wall_clock([sys.executable, "-m", "modal", "run", str(MODAL_APP)], log, WALL_CLOCK_LIMIT_S)
    text = log.read_text(encoding="utf-8", errors="replace")
    summary = next((json.loads(m.group(1)) for ln in text.splitlines() if (m := SUMMARY_RE.match(ln.strip()))), None)
    if summary is not None:
        (OUT_DIR / "numdiag_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    if run["timed_out"]:
        status = "timeout_harness_failure"
    elif run["returncode"] == 0 and summary and summary.get("completed"):
        status = "completed"
    else:
        status = "failed"
    rec.update(status=status, completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), run=run,
               modal_app_ids=sorted(set(APP_ID_RE.findall(text))), session_log_sha256=e1.sha256(log),
               summary_sha256=e1.sha256(OUT_DIR / "numdiag_summary.json") if summary is not None else None)
    (OUT_DIR / "record.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"\nEXP14 SHAPE-GATE NUMERICAL DIAGNOSIS ATTEMPT 2 {status.upper()} "
          f"(app {run['app_id']}, final state {run['final_app_state']})")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
