"""
RABIT-KV MLSys 2027 -- Experiment 10 ATTEMPT 2 runner (one full, fresh re-execution; logical fake-quant quality only).

Runs the UNCHANGED Experiment 10 sweep (run_experiment10_residual_ablation.py, frozen protocol
exp10_residual_protocol.json -- neither is modified; both are imported / verified read-only) under the transparent
post-failure methodological amendment results/mlsys2027/control_reproducibility_audit/qa_control_gate_amendment.json
(pinned by hash below). The amendment overlays exactly one rule, prospectively: for hotpotqa and qasper the aggregate
+/-1.0-point F1 control-reproduction pass/fail check of bf16 and the canonical R4 control is superseded by the frozen
per-example QA control gate (qa_control_gate.evaluate). Everything else -- configs, datasets, counts, prompts,
generation, scorer, storage gates, the continuation_ppl / NIAH / passage-retrieval control rules, the bf16 / R4 logical
KV MB reproduction checks -- is the original Exp10 gate set.

Attempt 1 (results/mlsys2027/ablations/residual_window/failed_attempt_1/) stays invalid and excluded; nothing is pooled.
Stops at the first failing benchmark; no retry, no further gate change.

Usage:
    python benchmarks/mlsys2027/run_experiment10_attempt2.py --dry-run
    python benchmarks/mlsys2027/run_experiment10_attempt2.py            (ONE execution, only when authorized)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_qa_control_reproducibility as audit  # noqa: E402
import qa_control_gate as qg  # noqa: E402
import run_experiment1_quality_frontier as e1  # noqa: E402  (accepted; read-only)
import run_experiment10_residual_ablation as r10  # noqa: E402  (frozen Exp10 harness; read-only)

ROOT = r10.ROOT
RUNNER_SCRIPT = Path(__file__).resolve()
ATTEMPT = 2
QA_BENCHMARKS = ("hotpotqa", "qasper")
AMENDMENT_SHA256_LF = "eaf28d66f38eedc13f447eb9776cd6affd2abb06dc2c64dec8d7e7d2a1a2e10c"
EXP10_PROTOCOL_SHA256 = "171cdbcab5c57137edda12ad044e038a52c722f32b740b26cc084ff0539c53d4"
ATTEMPT1_DIR = r10.OUT_DIR / "failed_attempt_1"
ATTEMPT1_ARCHIVE_COMMIT = "9889e334437fda567f82e67d3ffb7e6528b86422"
AUDIT_COMMIT = "9edc016ba0e24e0f1a7b47fbcf2be964863df10e"
AUDIT_FILES = [HERE / "audit_qa_control_reproducibility.py", audit.OUT]
EXTRA_PROTECTED = [ATTEMPT1_DIR, qg.AMENDMENT, HERE / "qa_control_gate.py", *AUDIT_FILES]
OWN_FILES = [RUNNER_SCRIPT, HERE / "qa_control_gate.py", HERE / "test_experiment10_attempt2.py", qg.AMENDMENT]


def protected_status() -> str:
    return "\n".join(s for s in (r10.protected_status(), e1.run_git(
        "status", "--short", "--", *[str(p.relative_to(ROOT)) for p in EXTRA_PROTECTED])) if s)


def preflight(dry_run: bool) -> dict:
    prov = r10.preflight(dry_run)  # original Exp10 preflight: protocol regenerates, Exp6-9 frozen, scripts derived
    if prov["protocol_sha256"] != EXP10_PROTOCOL_SHA256:
        raise RuntimeError("original Exp10 protocol hash changed")
    status = protected_status()
    if status:
        raise RuntimeError("protected paths are not clean:\n" + status)
    if qg.audit.sha256_lf(qg.AMENDMENT) != AMENDMENT_SHA256_LF:
        raise RuntimeError("methodological amendment hash does not match the pinned hash")
    amendment = qg.load_amendment()  # regenerates identically from the committed audit
    if json.loads(audit.OUT.read_text(encoding="utf-8")) != json.loads(json.dumps(audit.audit())):
        raise RuntimeError("control reproducibility audit does not regenerate identically")
    for commit, paths in ((ATTEMPT1_ARCHIVE_COMMIT, [ATTEMPT1_DIR]), (AUDIT_COMMIT, AUDIT_FILES)):
        if e1.run_git("diff", "--name-only", commit, "--", *[str(p.relative_to(ROOT)) for p in paths]):
            raise RuntimeError(f"archived evidence differs from its commit {commit[:7]}")
    record = json.loads((ATTEMPT1_DIR / "attempt_record.json").read_text(encoding="utf-8"))
    if record["status"] != "invalid_control_reproduction" or not record["excluded_from_accepted_results"]:
        raise RuntimeError("Exp10 attempt 1 is not recorded as invalid and excluded")
    uncommitted = e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in OWN_FILES])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: attempt-2 harness has uncommitted changes:\n" + uncommitted)
    return {**prov, "attempt": ATTEMPT, "attempt2_runner_sha256": e1.sha256(RUNNER_SCRIPT),
            "qa_control_gate_sha256": e1.sha256(HERE / "qa_control_gate.py"),
            "amendment_path": qg.AMENDMENT.relative_to(ROOT).as_posix(), "amendment_sha256_lf": AMENDMENT_SHA256_LF,
            "amendment": amendment, "attempt1_archive_commit": ATTEMPT1_ARCHIVE_COMMIT, "audit_commit": AUDIT_COMMIT,
            "uncommitted_files": "\n".join(s for s in (prov["uncommitted_files"], uncommitted) if s) or None}


def integrity(name: str, rc: int, log_text: str, protocol: dict, amendment: dict) -> dict:
    res = r10.integrity(name, rc, log_text, protocol)  # the original Exp10 gates
    if name not in QA_BENCHMARKS:
        return {**res, "control_rule": "original Exp10 control-reproduction rule"}
    reg = res["regression"]
    checks = {k: v for k, v in res["checks"].items() if k != "bf16_and_r4_control_reproduce_canonical"}
    checks["bf16_and_r4_logical_kv_mb_reproduce_canonical"] = all(
        c["within_tolerance"] for c in reg["checks"] if c["metric"].endswith(".avg_kv_mb"))
    gate = qg.evaluate(name, log_text, amendment)
    checks["qa_control_gate_passed"] = gate["passed"]
    return {**res, "checks": checks, "qa_control_gate": gate,
            "superseded_aggregate_f1_control_check_report_only": [c for c in reg["checks"]
                                                                  if c["metric"].endswith(".f1_pct")],
            "control_rule": "post-failure empirical per-example reproducibility envelope (qa_control_gate_amendment)",
            "passed": all(checks.values())}


def main(argv: list[str] | None = None) -> int:
    e1.make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    prov = preflight(a.dry_run)
    print("RABIT-KV MLSys 2027 -- Experiment 10 ATTEMPT 2 (residual-window ablation; logical fake-quant quality)")
    print("Preflight OK:", json.dumps({k: prov[k] for k in ("git_head", "protocol_sha256", "amendment_sha256_lf",
                                                             "attempt1_archive_commit", "audit_commit")}))
    for c in r10.build_commands():
        print(f"  {c['name']}: {' '.join(c['command'][3:])}")
    if prov["uncommitted_files"]:
        print("  WARNING (dry-run only): uncommitted:\n    " + prov["uncommitted_files"].replace("\n", "\n    "))
    if a.dry_run:
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    r10.OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": "Experiment 10 -- residual-window ablation (logical fake-quant quality)",
                "attempt": ATTEMPT, "methods": r10.METHODS, "conditions_residual": r10.CONDITIONS,
                "frozen": r10.FROZEN, "failed_attempt_1_excluded": True, "no_cross_attempt_pooling": True,
                "qa_control_gate": "post-failure empirical per-example reproducibility envelope",
                "status": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "provenance": prov, "runs": []}
    r10.MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    results = []
    for c in r10.build_commands():
        log = ROOT / c["log"]
        rc = e1.stream_command(c["command"], log)
        res = integrity(c["name"], rc, log.read_text(encoding="utf-8", errors="replace"), prov["protocol"],
                        prov["amendment"])
        results.append(res)
        manifest["runs"].append({"name": c["name"], "returncode": rc, "passed": res["passed"],
                                 "control_rule": res["control_rule"], "log_sha256": e1.sha256(log)})
        r10.MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not res["passed"]:  # no retry; stop at the first failing benchmark
            break
    r10.REGRESSION_CHECK.write_text(json.dumps([r["regression"] for r in results], indent=2) + "\n", encoding="utf-8")
    r10.RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = len(results) == len(e1.RUNS) and all(r["passed"] for r in results) and not protected_status()
    manifest.update(status="passed" if ok else "failed", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    r10.MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nEXPERIMENT 10 ATTEMPT 2 {manifest['status'].upper()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
