"""
MLSys 2027 Experiment 13 -- local launcher for the TurboQuant FEASIBILITY PROBE (NON-EVIDENCE).

Runs exp13_turboquant_probe_modal.py exactly ONCE and preserves its raw log under
results/mlsys2027/external_baseline/feasibility_probe/ with a probe_record.json labelled
"NON-EVIDENCE FEASIBILITY PROBE" and the log hashes. No latency, no BF16 / FP8 / RABIT legs, no retry, no alternate
settings. vllm-kvquant is snapshotted from the committed tree exactly as in Experiment 4 and is never modified.

Usage:
    python benchmarks/mlsys2027/run_exp13_turboquant_probe.py
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
import run_experiment1_quality_frontier as e1  # noqa: E402  (accepted; read-only helpers)

ROOT = e1.ROOT
OUT_DIR = ROOT / "results" / "mlsys2027" / "external_baseline" / "feasibility_probe"
LOG = OUT_DIR / "probe.log"
RECORD = OUT_DIR / "probe_record.json"
MODAL_APP = HERE / "exp13_turboquant_probe_modal.py"
FILES = [MODAL_APP, HERE / "exp13_turboquant_probe_worker.py", Path(__file__).resolve(), HERE / "exp3_watchdog.py"]
LABEL = "NON-EVIDENCE FEASIBILITY PROBE"


def image_expr(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    start = src.index("image = (\n    modal.Image.from_registry(")
    end = src.index('.pip_install("pytest", "modelscope")\n)', start)
    return src[start:end]


def parse(log_text: str) -> dict:
    out = {}
    for m in re.finditer(r"EXP13_PROBE_([A-Z_]+)=(.*)$", log_text, re.M):
        try:
            out.setdefault(m.group(1), json.loads(m.group(2)))
        except json.JSONDecodeError:
            out.setdefault(m.group(1), m.group(2))
    tb = re.search(r"EXP13_PROBE_TRACEBACK_BEGIN\n(.*?)EXP13_PROBE_TRACEBACK_END", log_text, re.S)
    out["traceback"] = tb.group(1) if tb else None
    out["warning_lines"] = [ln for ln in log_text.splitlines()
                            if re.search(r"\bWARNING\b|fallback|not supported|unsupported|Setting attention block size",
                                         ln, re.I)][:200]
    out["backend_lines"] = [ln for ln in log_text.splitlines()
                            if re.search(r"Using .*backend|TURBOQUANT|FLASH_ATTN|flash_attn_version|block size",
                                         ln, re.I)][:200]
    out["kv_memory_lines"] = [ln for ln in log_text.splitlines()
                              if re.search(r"KV cache memory|GPU KV cache size|Maximum concurrency", ln)][:50]
    return out


def main() -> int:
    e1.make_console_encoding_safe()
    if LOG.exists():
        raise SystemExit(f"{LOG} already exists; the probe is run exactly once")
    if image_expr(MODAL_APP) != image_expr(HERE / "exp4_deployment_modal.py"):
        raise SystemExit("probe image expression differs from the accepted Experiment 4 image")
    dirty = e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in FILES], "vllm-kvquant")
    if dirty:
        raise SystemExit("probe harness / vllm-kvquant not committed:\n" + dirty)
    snap = Path(tempfile.mkdtemp(prefix="exp13_vllm_snapshot_")) / "vllm_kvquant_snapshot.zip"
    e1.run_git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(snap), "HEAD:vllm-kvquant")
    os.environ["EXP13_VLLM_SNAPSHOT"] = str(snap)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    rc = e1.stream_command([sys.executable, "-m", "modal", "run", str(MODAL_APP)], LOG)
    text = LOG.read_text(encoding="utf-8", errors="replace")
    raw = LOG.read_bytes()
    record = {
        "label": LABEL,
        "not_exp13_evidence": True, "latency_measured": False, "other_legs_run": False, "retried": False,
        "vllm_modified": False,
        "purpose": "TurboQuant turboquant_k3v4_nc engine start-up / allocator feasibility under the intended "
                   "frozen Exp13 settings (requested block_size 32)",
        "git_head": e1.run_git("rev-parse", "HEAD"),
        "vllm_kvquant_snapshot_sha256": hashlib.sha256(snap.read_bytes()).hexdigest(),
        "harness_sha256": {p.name: e1.sha256(p) for p in FILES},
        "started_utc": started, "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "modal_returncode": rc,
        "modal_app_ids": sorted(set(re.findall(r"ap-[A-Za-z0-9]{20,}", text))),
        "log": LOG.relative_to(ROOT).as_posix(),
        "log_sha256": hashlib.sha256(raw).hexdigest(),
        "log_sha256_lf_normalized": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        "parsed": parse(text),
    }
    RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{LABEL}: modal rc={rc}; record {RECORD.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
