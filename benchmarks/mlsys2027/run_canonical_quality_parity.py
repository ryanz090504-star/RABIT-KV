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
OUT = ROOT / "results/mlsys2027/quality_semantic_audit/parity"
WALL_CLOCK_S = 45 * 60
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


def main() -> int:
    if git("status", "--short"):
        raise SystemExit("tree not clean")
    if (OUT / "record.json").exists():
        raise SystemExit("parity already run; runs once")
    OUT.mkdir(parents=True, exist_ok=False)
    result_path = Path(tempfile.mkdtemp()) / "parity_result.json"
    e = env()
    e["CANONICAL_PARITY_RESULT_PATH"] = str(result_path)
    rec = {"kind": "canonical-quality-v2 CPU parity tests (correctness testing, not a quality experiment)",
           "source_commit": git("rev-parse", "HEAD"), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
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
    stops = [a for a in new if not (post.get(a, {}).get("State") == "stopped" and post.get(a, {}).get("Tasks") == "0")]
    for a in stops:
        subprocess.run([sys.executable, "-m", "modal", "app", "stop", "-y", a], env=env(), capture_output=True, timeout=180)
    final = apps()
    res = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    if res is not None:
        (OUT / "parity_result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    rec.update(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_s=round(time.time() - t0, 1),
               modal_returncode=rc, timed_out=timed_out, app_ids_new=new, apps_stopped=stops,
               final_app_states={a: final.get(a) for a in new},
               cleanup_verified=all(final.get(a, {}).get("State") == "stopped" and final.get(a, {}).get("Tasks") == "0"
                                    for a in new),
               result_sha256=hashlib.sha256((OUT / "parity_result.json").read_bytes()).hexdigest() if res else None,
               passed=bool(res and res["passed"] and res["negative_control_passed"] and rc == 0 and not timed_out))
    (OUT / "record.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("passed", "modal_returncode", "timed_out", "app_ids_new", "cleanup_verified")}))
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
