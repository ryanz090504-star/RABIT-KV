"""
MLSys 2027 Experiment 13 -- local launcher for the LEGACY-RUNNER feasibility probe (NON-EVIDENCE).

Runs exp13_legacy_probe_modal.py exactly ONCE and preserves its raw log under
results/mlsys2027/external_baseline/legacy_runner_probe/ with probe_record.json labelled
"NON-EVIDENCE FEASIBILITY PROBE" (parsed per phase) and hashes. No latency, no retry, no setting variation,
no source modification. vllm-kvquant is snapshotted from the committed tree exactly as in Experiment 4.

Usage:
    python benchmarks/mlsys2027/run_exp13_legacy_probe.py
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
OUT_DIR = ROOT / "results" / "mlsys2027" / "external_baseline" / "legacy_runner_probe"
LOG = OUT_DIR / "probe.log"
RECORD = OUT_DIR / "probe_record.json"
MODAL_APP = HERE / "exp13_legacy_probe_modal.py"
FILES = [MODAL_APP, HERE / "exp13_legacy_probe_worker.py", Path(__file__).resolve(), HERE / "exp3_watchdog.py",
         HERE / "exp3_correctness_gate.py"]
LABEL = "NON-EVIDENCE FEASIBILITY PROBE"
LINE_RE = re.compile(r"^(?:\[(\S+)\] )?EXP13_LPROBE_([A-Z_]+)=(.*)$", re.M)


def parse(text: str) -> dict:
    phases: dict = {}
    top: dict = {}
    for m in LINE_RE.finditer(text):
        label, tag, raw = m.group(1), m.group(2), m.group(3)
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        (phases.setdefault(label, {}) if label else top).setdefault(tag, val)
    for label in list(phases):
        lines = [ln for ln in text.splitlines() if ln.startswith(f"[{label}] ")]
        tb = re.search(r"EXP13_LPROBE_TRACEBACK_BEGIN\n(.*?)EXP13_LPROBE_TRACEBACK_END", "\n".join(lines), re.S)
        phases[label]["traceback"] = tb.group(1) if tb else None
        phases[label]["warning_lines"] = [ln for ln in lines if re.search(
            r"\bWARNING\b|fallback|not supported|unsupported|Setting attention block size", ln, re.I)][:100]
        phases[label]["backend_lines"] = [ln for ln in lines if re.search(
            r"Using .*backend|FlashAttention version|Model Runner", ln)][:50]
        phases[label]["kv_memory_lines"] = [ln for ln in lines if re.search(
            r"KV cache memory|GPU KV cache size|Maximum concurrency", ln)][:20]
        phases[label]["gate_result_lines"] = [ln for ln in lines if "EXP3_GATE" in ln][:20]
    return {"top": top, "phases": phases}


def main() -> int:
    e1.make_console_encoding_safe()
    if LOG.exists():
        raise SystemExit(f"{LOG} already exists; the probe is run exactly once")
    if p1.image_expr(MODAL_APP) != p1.image_expr(HERE / "exp4_deployment_modal.py"):
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
        "label": LABEL, "not_exp13_evidence": True, "latency_measured": False, "retried": False,
        "vllm_modified": False, "runner_setting": "VLLM_USE_V2_MODEL_RUNNER=0 (legacy model runner) for every process",
        "purpose": "whether BF16 / native FP8 / RABIT / TurboQuant all run under one common legacy model runner with "
                   "block_size 32 and the intended Exp13 serving settings, with physical caches",
        "git_head": e1.run_git("rev-parse", "HEAD"),
        "vllm_kvquant_snapshot_sha256": hashlib.sha256(snap.read_bytes()).hexdigest(),
        "harness_sha256": {p.name: e1.sha256(p) for p in FILES},
        "started_utc": started, "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "modal_returncode": rc, "modal_app_ids": sorted(set(re.findall(r"ap-[A-Za-z0-9]{20,}", text))),
        "log": LOG.relative_to(ROOT).as_posix(), "log_sha256": hashlib.sha256(raw).hexdigest(),
        "log_sha256_lf_normalized": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        "parsed": parse(text),
    }
    RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{LABEL}: modal rc={rc}; record {RECORD.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
