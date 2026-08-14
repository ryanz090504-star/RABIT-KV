"""RABIT-2 Stage 4B2.1 exact-V2 quantization prototype.

No source files are modified. This tests hybrid compilation strategies that
keep torch.round in eager PyTorch (for byte-exact tie handling) while compiling
other V2 steps. Every candidate must match the reference packed bytes, minima,
and scales over a deterministic exactness gate before timing is accepted.
"""
from __future__ import annotations

import base64, os, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"
SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4b2_1_exactv2_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4b2_1_exactv2.py"
LOG = ROOT / "rabit2_stage4b2_1_exactv2.log"
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
        "def _quantize_v2_primary_ref",
    ):
        if needle not in text:
            raise RuntimeError(f"Stage4B2.1 preflight missing {needle}")
    compile(text, str(RABIT), "exec")
    print("Stage 4B2.1 local source preflight: PASSED")

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
            log.write(line)
            log.flush()
        return p.wait()

def main():
    print("RABIT-2 Stage 4B2.1 exact-V2 prototype")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    code = run()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__ == "__main__":
    main()
