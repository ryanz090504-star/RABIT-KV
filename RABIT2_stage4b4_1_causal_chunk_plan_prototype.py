"""RABIT-2 Stage 4B4.1 causal chunk-plan prototype.

No source files are modified.

Stage4B4 proved that final sidecar state can be bulk-updated ~10x faster, but a
chunked-prefill integration must preserve causal semantics at EVERY token, not
just the final state. This prototype precomputes the future exact sidecar
representation once, then exposes only the state prefix that is legal after
each token.

It verifies:
- exact physical page bytes at every page-close boundary,
- exact open/recent state after every token,
- bit-identical per-token Stage4B3 attention outputs,
- state-update and full 64-token causal-loop diagnostic speed.
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

SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4b4_1_causal_plan_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4b4_1_causal_plan.py"
LOG = ROOT / "rabit2_stage4b4_1_causal_chunk_plan.log"
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
        "rabit2_online_decode_attention_triton_stage4b3_gqa4",
    ):
        if needle not in text:
            raise RuntimeError(f"Stage4B4.1 preflight missing {needle}")
    compile(text, str(RABIT), "exec")
    print("Stage 4B4.1 local source preflight: PASSED")

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
    print("RABIT-2 Stage 4B4.1 causal chunk-plan prototype")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    code = run()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__ == "__main__":
    main()
