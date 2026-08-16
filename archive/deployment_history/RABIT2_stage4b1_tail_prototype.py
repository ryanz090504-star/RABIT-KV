"""Run RABIT-2 Stage 4B1 fused-sidecar-tail prototype on H100.

This script does NOT modify the vLLM fork. It snapshots the current Stage-3C
source, implements a prototype Triton tail path in the Modal runner, compares
it against the current exact tail path, then measures diagnostic latency.
"""
from __future__ import annotations
import base64, os, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"
SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4b1_vllm_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4b1_tail_prototype.py"
LOG = ROOT / "rabit2_stage4b1_tail_prototype.log"
RUNNER_Z = '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'

IGNORE_DIRS = {".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
               "build", "dist", ".venv", "venv", "node_modules"}
IGNORE_SUFFIX = {".pyc", ".pyo", ".log"}

def preflight():
    if not RABIT.is_file():
        raise FileNotFoundError(RABIT)
    text = RABIT.read_text(encoding="utf-8")
    for needle in (
        "RABIT2_STAGE3C_LIFECYCLE_BEGIN",
        "class Rabit2SingleSequenceRuntime",
        "_rabit2_tail_partial_kernel",
    ):
        if needle not in text:
            raise RuntimeError(f"Stage4B1 preflight missing {needle}")
    compile(text, str(RABIT), "exec")
    print("Stage 4B1 local source preflight: PASSED")

def include(p: Path):
    rel = p.relative_to(VLLM)
    return p.is_file() and not any(x in IGNORE_DIRS for x in rel.parts) and p.suffix.lower() not in IGNORE_SUFFIX

def snapshot():
    tmp = SNAP.with_suffix(".zip.tmp")
    for p in (tmp, SNAP):
        try: p.unlink()
        except FileNotFoundError: pass
    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(VLLM.rglob("*")):
            if include(p):
                z.write(p, p.relative_to(VLLM).as_posix())
                n += 1
    os.replace(tmp, SNAP)
    print(f"Created frozen vLLM snapshot: {SNAP} ({n} files, {SNAP.stat().st_size/1024**2:.1f} MB)")

def run():
    RUNNER.write_text(zlib.decompress(base64.b64decode(RUNNER_Z)).decode("utf-8"), encoding="utf-8")
    time.sleep(2)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = [sys.executable, "-m", "modal", "run", str(RUNNER)]
    print("Running:", " ".join(cmd))
    with LOG.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", env=env, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line); log.flush()
        return p.wait()

def main():
    print("RABIT-2 Stage 4B1 fused-sidecar-tail prototype")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    code = run()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__ == "__main__":
    main()
