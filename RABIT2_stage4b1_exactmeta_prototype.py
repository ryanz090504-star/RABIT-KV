
"""RABIT-2 Stage 4B1.1 exact-metadata fused-tail prototype.

The first all-Triton Stage4B1 integration was fast but failed regression on
some seeds because Triton re-quantized META8g64/K3 with slightly different
rounding from the byte-exact PyTorch quality oracle.

This script does NOT modify the fork. It keeps exact PyTorch quantization and
metadata encoding, but feeds the exact packed codes/metadata directly into a
fused Triton tail-attention kernel, avoiding BF16 tail materialization,
metadata decode, V/K dequant materialization, and torch.cat.
"""
from __future__ import annotations

import base64, os, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"
SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4b1_exactmeta_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4b1_exactmeta.py"
LOG = ROOT / "rabit2_stage4b1_exactmeta.log"
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
        "class Rabit2SingleSequenceRuntime",
        "_rabit2_tail_partial_kernel",
        "_rabit2_closed_page_partial_kernel",
        "_rabit2_reduce_partials_kernel",
    ):
        if needle not in text:
            raise RuntimeError(f"Stage4B1.1 preflight missing {needle}")
    compile(text, str(RABIT), "exec")
    print("Stage 4B1.1 local source preflight: PASSED")
    if "RABIT2_STAGE4B1_FUSED_TAIL_BEGIN" in text:
        print("Detected failed Stage4B1 integration block; prototype will NOT use its public dispatch.")

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
    print("RABIT-2 Stage 4B1.1 exact-metadata fused-tail prototype")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    code = run()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__ == "__main__":
    main()
