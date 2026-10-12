"""
Runner for the canonical-quality-v2 CPU parity tests (CPU-only Modal container; no GPU, no model).
Requires a clean tree; UTF-8 child process; hard wall clock with before/after app discovery and cleanup; writes
results/mlsys2027/quality_semantic_audit/parity/{parity_session.log, parity_result.json, record.json}.
"""

from __future__ import annotations

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
ROOT = HERE.parents[1]
MODAL_APP = HERE / "canonical_quality_parity_modal.py"
APP_NAME = "rabit-kv-canonical-quality-v2-parity"
OUT = ROOT / "results/mlsys2027/quality_semantic_audit/parity_attempt_3"
ATTEMPT2_DIR = "results/mlsys2027/quality_semantic_audit/parity_failed_attempt_2"
ATTEMPT2_ARCHIVE_COMMIT = "8abbde395e61531178cea5586655175efab6cc98"
CLEANUP_TIMEOUT_S, CLEANUP_POLL_S = 60, 3
# planned cases per geometry: T1 4 dists x 13 lengths, T2 4 x 7 prefills, T3 2 x 3, T4 4 x 13
PLANNED_PER_GEOMETRY = {"T1_full_state": 52, "T2_sequential_aging": 28, "T3_hf_cache": 6, "T4_old_harness": 52}
WALL_CLOCK_S = 20 * 60
ATTEMPT1_DIR = "results/mlsys2027/quality_semantic_audit/parity_failed_attempt_1"
ATTEMPT1_ARCHIVE_COMMIT = "9d6b819db91157f3fb3bda47c640ccb1cd814d14"
AUDIT_COMMIT, AUDIT_PATH = "e88f46014acebba8458a59fdc714adc26f7ac9fe", "results/mlsys2027/quality_semantic_audit/semantic_audit_record.json"
IMPL_COMMIT = "c36069781b259e2e9b4d8adf60b7c58ea5df5cac"
IMPL_FILES = ["benchmarks/mlsys2027/canonical_rabit_quality.py", "benchmarks/mlsys2027/canonical_quality_parity_tests.py"]
FILES = [HERE / "canonical_rabit_quality.py", HERE / "canonical_quality_parity_tests.py", MODAL_APP, Path(__file__).resolve(),
         ROOT / "vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py", ROOT / "benchmarks/quality/hotpotqa.py"]
APP_RE = re.compile(r"ap-[A-Za-z0-9]{20,}")


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def env() -> dict:
    e = os.environ.copy()
    e["PYTHONUTF8"], e["PYTHONIOENCODING"] = "1", "utf-8"
    return e


def apps() -> dict:
    p = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env(), timeout=180)
    return {r["App ID"]: r for r in json.loads(p.stdout) if r.get("Description") == APP_NAME}


def poll_cleanup(new: list, list_fn, stop_fn, timeout_s: float = CLEANUP_TIMEOUT_S, poll_s: float = CLEANUP_POLL_S,
                 sleep=time.sleep, clock=time.time) -> dict:
    """Poll until every NEW app is stopped with 0 tasks (<= timeout_s); stop any still-active app again on timeout."""
    def done(row):
        return bool(row) and row.get("State") == "stopped" and str(row.get("Tasks")) == "0"
    t0, polls = clock(), 0
    first = list_fn()
    polls += 1
    states = {a: first.get(a) for a in new}
    first_states = dict(states)
    while not all(done(v) for v in states.values()) and clock() - t0 < timeout_s:
        sleep(poll_s)
        cur = list_fn()
        polls += 1
        states = {a: cur.get(a) for a in new}
    stopped_again = []
    if not all(done(v) for v in states.values()):
        for a in new:
            if not done(states[a]):
                stop_fn(a)
                stopped_again.append(a)
        cur = list_fn()
        polls += 1
        states = {a: cur.get(a) for a in new}
    return {"new_app_ids": list(new), "first_observed_states": first_states, "final_states": states, "polls": polls,
            "elapsed_s": round(clock() - t0, 1), "stopped_again_after_timeout": stopped_again,
            "verified": all(done(v) for v in states.values()) and not stopped_again}


def cases_executed(res: dict) -> dict:
    out = {}
    for g, G in res["geometries"].items():
        out[g] = {"T1_full_state": sum(len(v) for v in G["T1_full_state"].values()),
                  "T2_sequential_aging": sum(len(v) for v in G["T2_sequential_aging"].values()),
                  "T3_hf_cache": sum(len(v) for v in G["T3_hf_cache"].values()),
                  "T4_old_harness": sum(len(v) for v in G["T4_old_harness"].values())}
    return out


def preflight() -> dict:
    """Pre-run requirements (no Modal / GPU / model): clean tree, archives / audit / implementation unchanged, offline
    harness tests (isolated remote import, Attempt-1 negative control, file shipping, compile)."""
    h = subprocess.run([sys.executable, str(HERE / "test_canonical_parity_harness.py")], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env(), timeout=900)
    checks = {
        "clean_tree": git("status", "--short") == "",
        "attempt1_archive_unchanged": git("diff", "--name-only", ATTEMPT1_ARCHIVE_COMMIT, "--",
                                          f"{ATTEMPT1_DIR}/parity_session.log", f"{ATTEMPT1_DIR}/record.json",
                                          f"{ATTEMPT1_DIR}/attempt_record.json") == "",
        "semantic_audit_unchanged": git("diff", "--name-only", AUDIT_COMMIT, "--", AUDIT_PATH) == "",
        "implementation_unchanged_since_c360697": git("diff", "--name-only", IMPL_COMMIT, "--", *IMPL_FILES) == "",
        "offline_harness_tests_pass": h.returncode == 0 and bool(re.search(r"^(\d+)/\1 passed$", h.stdout, re.M)),
        "attempt2_archive_unchanged": git("diff", "--name-only", ATTEMPT2_ARCHIVE_COMMIT, "--",
                                          f"{ATTEMPT2_DIR}/parity_session.log", f"{ATTEMPT2_DIR}/record.json",
                                          f"{ATTEMPT2_DIR}/attempt_record.json") == "",
        "attempt3_not_already_run": not OUT.exists()}
    if not all(checks.values()):
        raise SystemExit(f"pre-run validation failed: {checks} | harness tests: {h.stdout[-1500:]}")
    return checks


def main() -> int:
    checks = preflight()
    OUT.mkdir(parents=True, exist_ok=False)
    result_path = Path(tempfile.mkdtemp()) / "parity_result.json"
    e = env()
    e["CANONICAL_PARITY_RESULT_PATH"] = str(result_path)
    rec = {"kind": "canonical-quality-v2 CPU parity tests (correctness testing, not a quality experiment)",
           "source_commit": git("rev-parse", "HEAD"), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "attempt": 2, "preflight": checks,
           "file_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in FILES}, "wall_clock_s": WALL_CLOCK_S}
    pre = apps()
    t0, parsed = time.time(), set()
    log = OUT / "parity_session.log"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen([sys.executable, "-m", "modal", "run", str(MODAL_APP)], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=e, cwd=ROOT)

        def pump():
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                parsed.update(APP_RE.findall(line))

        th = threading.Thread(target=pump, daemon=True)
        th.start()
        try:
            rc, timed_out = proc.wait(timeout=WALL_CLOCK_S), False
        except subprocess.TimeoutExpired:
            proc.kill()
            rc, timed_out = proc.wait(), True
        th.join(timeout=30)
    post = apps()
    new = sorted((set(post) - set(pre)) | (parsed - set(pre)))
    cleanup = poll_cleanup(new, apps, lambda a: subprocess.run([sys.executable, "-m", "modal", "app", "stop", "-y", a],
                                                               env=env(), capture_output=True, timeout=180))
    res = None
    if result_path.is_file():
        raw = result_path.read_bytes()  # exact UTF-8 JSON text written by the local entrypoint
        (OUT / "parity_result.json").write_bytes(raw)
        res = json.loads((OUT / "parity_result.json").read_bytes().decode("utf-8"))
    executed = cases_executed(res) if res else None
    all_cases = bool(executed) and all(executed[g] == PLANNED_PER_GEOMETRY for g in executed) and len(executed) == 2
    rec.update(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_s=round(time.time() - t0, 1),
               modal_returncode=rc, timed_out=timed_out, app_ids_new=new, cleanup=cleanup,
               cleanup_verified=cleanup["verified"], cases_planned_per_geometry=PLANNED_PER_GEOMETRY,
               cases_executed=executed, all_planned_cases_ran=all_cases,
               result_bytes=(OUT / "parity_result.json").stat().st_size if res else None,
               result_sha256=hashlib.sha256((OUT / "parity_result.json").read_bytes()).hexdigest() if res else None,
               passed=bool(res and res["passed"] and res["negative_control_passed"] and rc == 0 and not timed_out
                           and all_cases and cleanup["verified"]))
    (OUT / "record.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("passed", "modal_returncode", "timed_out", "app_ids_new", "cleanup_verified")}))
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
