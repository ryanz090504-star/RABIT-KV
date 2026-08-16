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
