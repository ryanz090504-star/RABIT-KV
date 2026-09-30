"""
RABIT-KV MLSys 2027 -- Experiment 14 NON-EVIDENCE shape-gate numerical diagnosis runner (after probe Attempt 1).
One Modal H100 session: exp14_shape_gate_numdiag.py on the frozen Exp14 image; no model, no engine, no download.
Writes results/mlsys2027/second_model/shape_gate_numdiag/{numdiag_session.log, numdiag_summary.json, record.json}.
Runs once; refuses to overwrite a completed record. Changes no gate, protocol or evidence.

Usage:
    python benchmarks/mlsys2027/run_exp14_shape_gate_numdiag.py --dry-run
    python benchmarks/mlsys2027/run_exp14_shape_gate_numdiag.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiment1_quality_frontier as e1  # noqa: E402  (read-only helpers)
import run_experiment14_second_model as r14  # noqa: E402  (read-only: snapshot builder, protected paths)

ROOT = e1.ROOT
MODAL_APP = HERE / "exp14_shape_gate_numdiag_modal.py"
DIAG = HERE / "exp14_shape_gate_numdiag.py"
OUT_DIR = ROOT / "results" / "mlsys2027" / "second_model" / "shape_gate_numdiag"
RUNNER = Path(__file__).resolve()
HARNESS = [RUNNER, MODAL_APP, DIAG, r14.SHAPE_GATE, r14.MODAL_APP]
SUMMARY_RE = re.compile(r"^EXP14_NUMDIAG_SUMMARY=(\{.*\})\s*$")


def preflight(dry_run: bool) -> dict:
    if r14.protected_status():
        raise RuntimeError("protected paths are not clean")
    uncommitted = e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in HARNESS])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: diagnosis harness has uncommitted changes:\n" + uncommitted)
    rec = OUT_DIR / "record.json"
    if rec.exists() and json.loads(rec.read_text(encoding="utf-8")).get("status") == "completed":
        raise RuntimeError("the numerical diagnosis already completed; it runs once")
    return {"git_head": e1.run_git("rev-parse", "HEAD"), "vllm_kvquant_tree": e1.run_git("rev-parse", "HEAD:vllm-kvquant"),
            "harness_sha256": {p.name: e1.sha256(p) for p in HARNESS}, "uncommitted_files": uncommitted or None}


def main(argv=None) -> int:
    e1.make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    prov = preflight(a.dry_run)
    print("Exp14 shape-gate numerical diagnosis (NON-EVIDENCE):", json.dumps({k: prov[k] for k in ("git_head",
                                                                                                   "vllm_kvquant_tree")}))
    if a.dry_run:
        print("--dry-run: nothing executed.")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / "numdiag_session.log"
    rec = {"experiment": 14, "kind": "shape_gate_numerical_diagnosis", "non_evidence": True, "status": "running",
           "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **prov}
    (OUT_DIR / "record.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    os.environ["EXP14_VLLM_SNAPSHOT"] = str(r14.build_snapshot())
    rc = e1.stream_command([sys.executable, "-m", "modal", "run", str(MODAL_APP)], log)
    text = log.read_text(encoding="utf-8", errors="replace")
    summary = next((json.loads(m.group(1)) for ln in text.splitlines() if (m := SUMMARY_RE.match(ln.strip()))), None)
    if summary is not None:
        (OUT_DIR / "numdiag_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    ok = rc == 0 and bool(summary and summary.get("completed"))
    rec.update(status="completed" if ok else "failed", modal_returncode=rc,
               completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               modal_app_ids=sorted(set(re.findall(r"ap-[A-Za-z0-9]{20,}", text))), session_log_sha256=e1.sha256(log),
               summary_sha256=e1.sha256(OUT_DIR / "numdiag_summary.json") if summary is not None else None)
    (OUT_DIR / "record.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"\nEXP14 SHAPE-GATE NUMERICAL DIAGNOSIS {rec['status'].upper()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
