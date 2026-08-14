"""RABIT-2 Stage 4B3 grouped closed-page prototype.

No source code is modified.

The current deployment emits one (m,l,acc) softmax partial per 32-token
physical page. This prototype evaluates 2/4/8 pages per Triton program and
emits one partial per group. It measures:
- closed-page + final-reduction latency at 512/2K/8K/16K context,
- scratch memory,
- output difference vs the current Stage4B2 baseline,
- one full-online correctness case including the Stage4B1 exactmeta tail.
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

SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4b3_grouped_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4b3_grouped.py"
LOG = ROOT / "rabit2_stage4b3_grouped_pages.log"
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
        "RABIT2_STAGE4B1_EXACTMETA_BEGIN",
        "RABIT2_STAGE4B2_EXACT_V2_FIXED_BEGIN",
        "def _rabit2_closed_page_partial_kernel",
        "def _rabit2_reduce_partials_kernel",
    ):
        if needle not in text:
            raise RuntimeError(f"Stage4B3 preflight missing {needle}")
    compile(text, str(RABIT), "exec")
    print("Stage 4B3 local source preflight: PASSED")


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
    print(
        f"Created frozen snapshot: {SNAP} "
        f"({n} files, {SNAP.stat().st_size / 1024**2:.1f} MB)"
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
    print("RABIT-2 Stage 4B3 grouped closed-page prototype")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    code = run()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")


if __name__ == "__main__":
    main()
