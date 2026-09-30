"""
MLSys 2027 Experiment 13 -- local launcher for the TurboQuant test COLLECTION probe (NON-EVIDENCE).

Runs exp13_tq_collect_probe_modal.py exactly ONCE (unchanged Exp4-verbatim image, same committed patched snapshot,
one H100, `pytest --collect-only` only) and writes results/mlsys2027/external_baseline/tq_collect_probe/ with the raw
log and collect_record.json: the authoritative collected node IDs, their count, the SciPy-reference node-ID set
(S_SCIPY), the GPU-only node-ID set, scipy availability and the GPGPU_AVAILABLE value.

Usage:
    python benchmarks/mlsys2027/run_exp13_tq_collect_probe.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_exp13_turboquant_probe as p1  # noqa: E402  (image-expression extractor; read-only)
import run_experiment1_quality_frontier as e1  # noqa: E402

ROOT = e1.ROOT
OUT_DIR = ROOT / "results" / "mlsys2027" / "external_baseline" / "tq_collect_probe"
LOG = OUT_DIR / "collect.log"
RECORD = OUT_DIR / "collect_record.json"
MODAL_APP = HERE / "exp13_tq_collect_probe_modal.py"
TARGET = "tests/quantization/test_turboquant.py"
SCIPY_TEST = "::TestLloydMax::test_centroids_match_scipy_reference["
GPU_ONLY_CLASSES = ("TestRotationMatrix", "TestHadamardRotation", "TestStoreDecodeRoundTrip")
LABEL = "NON-EVIDENCE FEASIBILITY PROBE (test collection only)"


def node_ids(stdout: str) -> list[str]:
    return [ln.strip() for ln in stdout.splitlines() if "::" in ln and not ln.startswith(" ")]


def classify(ids: list[str]) -> dict:
    return {"count": len(ids), "all_in_target": all(i.startswith(TARGET + "::") for i in ids),
            "scipy_reference": sorted(i for i in ids if SCIPY_TEST in i),
            "gpu_only": sorted(i for i in ids if i.split("::")[1] in GPU_ONLY_CLASSES),
            "sha256_sorted": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()}


def main() -> int:
    e1.make_console_encoding_safe()
    if LOG.exists():
        raise SystemExit(f"{LOG} already exists; the probe is run exactly once")
    if p1.image_expr(MODAL_APP) != p1.image_expr(HERE / "exp4_deployment_modal.py"):
        raise SystemExit("probe image expression differs from the accepted Experiment 4 image")
    dirty = e1.run_git("status", "--short", "--", str(MODAL_APP.relative_to(ROOT)),
                       str(Path(__file__).resolve().relative_to(ROOT)), "vllm-kvquant")
    if dirty:
        raise SystemExit("probe harness / vllm-kvquant not committed:\n" + dirty)
    snap = Path(tempfile.mkdtemp(prefix="exp13_vllm_snapshot_")) / "vllm_kvquant_snapshot.zip"
    e1.run_git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(snap), "HEAD:vllm-kvquant")
    os.environ["EXP13_VLLM_SNAPSHOT"] = str(snap)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    rc = e1.stream_command([sys.executable, "-m", "modal", "run", str(MODAL_APP)], LOG)
    text = LOG.read_text(encoding="utf-8", errors="replace")
    env = json.loads(re.search(r"EXP13_TQCOLLECT_ENV=(\{.*\})", text).group(1)) if "EXP13_TQCOLLECT_ENV=" in text else None
    cmd = json.loads(re.search(r"EXP13_TQCOLLECT_CMD=(\{.*\})", text).group(1)) if "EXP13_TQCOLLECT_CMD=" in text else None
    out = re.search(r"EXP13_TQCOLLECT_STDOUT_BEGIN\n(.*?)EXP13_TQCOLLECT_STDOUT_END", text, re.S)
    ids = node_ids(out.group(1)) if out else []
    summary = re.findall(r"^(\d+) tests? collected.*$", out.group(1), re.M) if out else []
    raw = LOG.read_bytes()
    record = {"label": LABEL, "not_exp13_evidence": True, "tests_executed": False,
              "git_head": e1.run_git("rev-parse", "HEAD"), "vllm_kvquant_tree": e1.run_git("rev-parse", "HEAD:vllm-kvquant"),
              "started_utc": started, "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "modal_returncode": rc, "modal_app_ids": sorted(set(re.findall(r"ap-[A-Za-z0-9]{20,}", text))),
              "environment": env, "collect_command": cmd, "pytest_summary_line_count": summary,
              "node_ids": ids, **classify(ids),
              "log_sha256": hashlib.sha256(raw).hexdigest(),
              "log_sha256_lf_normalized": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}
    RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\n{LABEL}: modal rc={rc}; {len(ids)} node ids; record {RECORD.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
