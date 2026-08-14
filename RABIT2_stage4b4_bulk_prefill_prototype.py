"""RABIT-2 Stage 4B4 bulk-prefill prototype.

No source files are modified.

The real-engine smoke showed the final Stage4B serving lifecycle is correct,
but chunked prefill still feeds the runtime through the token-at-a-time append
path. This prototype tests an exact bulk sidecar update that:
- moves a whole incoming chunk through R4 in one operation,
- quantizes newly-aged V tokens in one batched exact call,
- closes physical 32-token pages in page-sized units,
- performs one batched index_copy_ for all newly closed pages,
- preserves the exact physical bytes and bounded open/recent state.
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

SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4b4_bulk_prefill_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4b4_bulk_prefill.py"
LOG = ROOT / "rabit2_stage4b4_bulk_prefill.log"
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
        "RABIT2_STAGE4B3_GQA4_BEGIN",
        "class Rabit2SingleSequenceRuntime",
        "def _closed_page_exact",
    ):
        if needle not in text:
            raise RuntimeError(f"Stage4B4 preflight missing {needle}")
    compile(text, str(RABIT), "exec")
    print("Stage 4B4 local source preflight: PASSED")

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
    with zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for p in sorted(VLLM.rglob("*")):
            if include(p):
                z.write(p, p.relative_to(VLLM).as_posix())
                n += 1
    os.replace(tmp, SNAP)
    print(
        f"Created frozen snapshot: {SNAP} "
        f"({n} files, {SNAP.stat().st_size/1024**2:.1f} MB)"
    )

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
    print("RABIT-2 Stage 4B4 bulk-prefill prototype")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    code = run()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__ == "__main__":
    main()
