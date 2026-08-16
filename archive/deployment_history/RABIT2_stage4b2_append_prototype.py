"""Run RABIT-2 Stage 4B2 append-path prototype on H100.

No source files are modified. The prototype measures:
1) exact eager V2 quantization,
2) torch.compile V2 quantization with an exactness gate,
3) current runtime.append using torch.cat,
4) fixed-size ring/open buffers with in-place updates.

It deliberately stays below the 32-token page-close boundary; Stage4A already
measured page-close separately (~once per 32 aged tokens).
"""
from __future__ import annotations

import base64, os, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"
SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4b2_append_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4b2_append.py"
LOG = ROOT / "rabit2_stage4b2_append_prototype.log"
RUNNER_Z = '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'

IGNORE_DIRS = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", ".venv", "venv", "node_modules"
}
IGNORE_SUFFIX = {".pyc", ".pyo", ".log"}

def dec(x):
    return zlib.decompress(base64.b64decode(x.encode("ascii"))).decode("utf-8")

def preflight():
    if not RABIT.is_file():
        raise FileNotFoundError(RABIT)
    text = RABIT.read_text(encoding="utf-8")
    for needle in (
        "RABIT2_STAGE4B1_EXACTMETA_BEGIN",
        "class Rabit2SingleSequenceRuntime",
        "def _append_aged",
        "def append",
    ):
        if needle not in text:
            raise RuntimeError(f"Stage4B2 preflight missing {needle}")
    compile(text, str(RABIT), "exec")
    print("Stage 4B2 local source preflight: PASSED")

def include(p):
    rel = p.relative_to(VLLM)
    return (
        p.is_file()
        and not any(x in IGNORE_DIRS for x in rel.parts)
        and p.suffix.lower() not in IGNORE_SUFFIX
    )

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
    RUNNER.write_text(dec(RUNNER_Z), encoding="utf-8")
    time.sleep(2)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = [sys.executable, "-m", "modal", "run", str(RUNNER)]
    print("Running:", " ".join(cmd))
    with LOG.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env, bufsize=1
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line); log.flush()
        return p.wait()

def main():
    print("RABIT-2 Stage 4B2 append-path prototype")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    code = run()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__ == "__main__":
    main()
