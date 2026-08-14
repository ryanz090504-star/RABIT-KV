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
TRITON = VLLM / "vllm/v1/attention/backends/triton_attn.py"

SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4d1_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4d1_profiler.py"
LOG = ROOT / "rabit2_stage4d1_bottleneck_profiler.log"
RUNNER_Z = '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'

IGNORE = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", ".venv", "venv", "node_modules",
}

def dec(s):
    return zlib.decompress(base64.b64decode(s)).decode("utf-8")

def preflight():
    if not RABIT.is_file():
        raise FileNotFoundError(RABIT)
    if not TRITON.is_file():
        raise FileNotFoundError(TRITON)
    rt = RABIT.read_text(encoding="utf-8")
    tt = TRITON.read_text(encoding="utf-8")
    for needle in (
        "RABIT2_STAGE4B2_EXACT_V2_FIXED_BEGIN",
        "RABIT2_STAGE4B3_GQA4_BEGIN",
        "RABIT2_STAGE4B4_CAUSAL_PREFILL_BEGIN",
    ):
        if needle not in rt:
            raise RuntimeError(f"Stage4D1 preflight missing {needle}")
    if "RABIT2_STAGE4B4_CAUSAL_PREFILL_TRITON_BEGIN" not in tt:
        raise RuntimeError("Stage4D1 preflight missing Stage4B4 Triton integration")
    print("Final source preflight: PASSED")

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
        f"Frozen snapshot: {SNAP} "
        f"({n} files, {SNAP.stat().st_size/1024**2:.1f} MB)"
    )

def run():
    RUNNER.write_text(dec(RUNNER_Z), encoding="utf-8")
    compile(RUNNER.read_text(encoding="utf-8"), str(RUNNER), "exec")
    time.sleep(2)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-m", "modal", "run", str(RUNNER)]
    print("Running:", " ".join(cmd))
    with LOG.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=env, bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return p.wait()

def main():
    print("RABIT-2 Stage 4D1 bottleneck profiler")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    code = run()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__ == "__main__":
    main()
