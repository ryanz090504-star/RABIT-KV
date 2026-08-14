"""Final real-engine smoke before Stage 4C.

This does NOT modify source. It snapshots the user's current vllm-kvquant tree
and starts an actual vLLM V1 engine with:
- TinyLlama/TinyLlama_v1.1
- kv_cache_dtype="rabit_kv2"
- TRITON_ATTN
- block_size=32
- max_num_seqs=3
- max_num_batched_tokens=64
- enable_chunked_prefill=True
- enable_prefix_caching=False
- enforce_eager=True

It runs two waves of three long prompts so the latest Stage4B3 source is
exercised in a real serving lifecycle, including request cleanup/reuse.
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

SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4b_final_engine_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4b_final_engine_smoke.py"
LOG = ROOT / "rabit2_stage4b_final_engine_smoke.log"
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
        "RABIT2_STAGE3C_LIFECYCLE_BEGIN",
        "RABIT2_STAGE4B1_EXACTMETA_BEGIN",
        "RABIT2_STAGE4B2_EXACT_V2_FIXED_BEGIN",
        "RABIT2_STAGE4B3_GQA4_BEGIN",
        "rabit2_online_decode_attention_triton_stage4b3_gqa4",
    ):
        if needle not in text:
            raise RuntimeError(f"Final engine smoke preflight missing {needle}")
    compile(text, str(RABIT), "exec")
    print("Stage 4B final local source preflight: PASSED")

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
    print("RABIT-2 Stage 4B final real-engine smoke")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    code = run()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__ == "__main__":
    main()
