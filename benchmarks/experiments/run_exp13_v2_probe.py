"""
MLSys 2027 Experiment 13 -- local launcher for the V2-RUNNER feasibility probe on the PATCHED snapshot (NON-EVIDENCE).

Runs exp13_v2_probe_modal.py exactly ONCE and preserves its raw log under
results/mlsys2027/external_baseline/v2_runner_probe/ with probe_record.json labelled
"NON-EVIDENCE FEASIBILITY PROBE" (parsed per phase) and hashes. Requires the committed snapshot to contain exactly the
backported upstream attn_utils.py (vLLM fa4321de3; local commit 611a4ff). No latency, no retry, no setting variation,
no further source change; VLLM_USE_V2_MODEL_RUNNER must be unset (default V2 runner).

Usage:
    python benchmarks/mlsys2027/run_exp13_v2_probe.py
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
OUT_DIR = ROOT / "results" / "mlsys2027" / "external_baseline" / "v2_runner_probe"
LOG = OUT_DIR / "probe.log"
RECORD = OUT_DIR / "probe_record.json"
MODAL_APP = HERE / "exp13_v2_probe_modal.py"
FILES = [MODAL_APP, HERE / "exp13_v2_probe_worker.py", Path(__file__).resolve(), HERE / "exp3_watchdog.py",
         HERE / "exp3_correctness_gate.py"]
LABEL = "NON-EVIDENCE FEASIBILITY PROBE"
BACKPORT_COMMIT = "611a4ffc96c70c81529978dc01a290e87ccf76e9"
UPSTREAM_FIX = "fa4321de3d894c50c5ca0766dffa352d3fb07423"
ATTN_UTILS = "vllm-kvquant/vllm/v1/worker/gpu/attn_utils.py"
UPSTREAM_ATTN_UTILS_BLOB = "5fcc9053bf4c65b5f0f110f62edae66f4c363c4c"  # git blob of attn_utils.py at fa4321de3
LINE_RE = re.compile(r"^(?:\[(\S+)\] )?EXP13_V2PROBE_([A-Z_]+)=(.*)$", re.M)
TB_RE = re.compile(r"EXP13_V2PROBE_TRACEBACK_BEGIN\n(.*?)EXP13_V2PROBE_TRACEBACK_END", re.S)


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
        tb = TB_RE.search("\n".join(lines))
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
    if e1.run_git("rev-parse", f"HEAD:{ATTN_UTILS}") != UPSTREAM_ATTN_UTILS_BLOB:
        raise SystemExit("snapshot does not contain the backported upstream attn_utils.py")
    if e1.run_git("merge-base", "--is-ancestor", BACKPORT_COMMIT, "HEAD") != "":
        raise SystemExit("backport commit is not an ancestor of HEAD")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") is not None:
        raise SystemExit("VLLM_USE_V2_MODEL_RUNNER must be unset")
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
        "source_modifications_beyond_backport": False,
        "backport": {"local_commit": BACKPORT_COMMIT, "upstream_commit": UPSTREAM_FIX, "upstream_pr": 47609,
                     "upstream_first_release": "v0.25.0", "file": ATTN_UTILS, "blob": UPSTREAM_ATTN_UTILS_BLOB},
        "runner_setting": "default model runner (V2); VLLM_USE_V2_MODEL_RUNNER unset for every process",
        "topologies": {"inprocess": "VLLM_ENABLE_V1_MULTIPROCESSING=0, introspection only",
                       "multiprocess": "vLLM default engine-core process (accepted Exp4 topology; intended Exp13)"},
        "purpose": "whether BF16 / native FP8 / RABIT / TurboQuant all run on the SAME patched snapshot under the "
                   "default V2 model runner with block_size 32, the intended Exp13 settings and topology, with "
                   "physical caches",
        "git_head": e1.run_git("rev-parse", "HEAD"),
        "vllm_kvquant_tree": e1.run_git("rev-parse", "HEAD:vllm-kvquant"),
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
