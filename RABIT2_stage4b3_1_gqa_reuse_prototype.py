"""RABIT-2 Stage 4B3.1 GQA KV-decode reuse prototype.

No source files are modified.

Current closed-page attention launches one program per (page, query-head).
For Llama-3.1-8B, four query heads share each KV head, so the same packed
K3/V2 page data and metadata are decoded four times.

This prototype tests fusing 2 or 4 query heads per program so they reuse one
packed K/V decode. Scratch layout remains unchanged; this experiment targets
latency rather than the Stage4B3 grouped-page scratch tradeoff.
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"
SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4b3_1_gqa_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4b3_1_gqa.py"
LOG = ROOT / "rabit2_stage4b3_1_gqa_reuse.log"
RUNNER_Z = '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'

IGNORE = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", ".venv", "venv", "node_modules"
}

def dec(s):
    return zlib.decompress(base64.b64decode(s)).decode("utf-8")

def preflight():
    if not RABIT.is_file():
        raise FileNotFoundError(RABIT)
    text = RABIT.read_text(encoding="utf-8")
    for needle in (
        "RABIT2_STAGE4B2_EXACT_V2_FIXED_BEGIN",
        "def _rabit2_closed_page_partial_kernel",
        "def _rabit2_reduce_partials_kernel",
        "def _rabit2_stage4b1_exactmeta_emit_tail_partial",
    ):
        if needle not in text:
            raise RuntimeError(f"Stage4B3.1 preflight missing {needle}")
    compile(text, str(RABIT), "exec")
    print("Stage 4B3.1 local source preflight: PASSED")

def include(p):
    rel = p.relative_to(VLLM)
    return (
        p.is_file()
        and not any(x in IGNORE for x in rel.parts)
        and p.suffix.lower() not in {".pyc", ".pyo", ".log"}
    )

def snapshot():
    tmp = SNAP.with_suffix(".zip.tmp")
    for p in (tmp, SNAP):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(VLLM.rglob("*")):
            if include(p):
                z.write(p, p.relative_to(VLLM).as_posix())
                n += 1
    os.replace(tmp, SNAP)
    print(f"Created frozen snapshot: {SNAP} ({n} files, {SNAP.stat().st_size/1024**2:.1f} MB)")

def run():
    RUNNER.write_text(dec(RUNNER_Z), encoding="utf-8")
    time.sleep(2)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-m", "modal", "run", str(RUNNER)]
    print("Running:", " ".join(cmd))
    with LOG.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return p.wait()

def main():
    print("RABIT-2 Stage 4B3.1 GQA KV-decode reuse prototype")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    code = run()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__ == "__main__":
    main()
