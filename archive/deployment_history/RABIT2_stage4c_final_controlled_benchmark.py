from __future__ import annotations

import base64
import hashlib
import json
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

MODEL = os.environ.get(
    "RABIT_STAGE4C_MODEL",
    "meta-llama/Llama-3.1-8B",
)

SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4c_final_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4c_final.py"
LOG = ROOT / "rabit2_stage4c_final_controlled.log"
JSON_OUT = ROOT / "rabit2_stage4c_final_results.json"

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
        "RABIT2_STAGE4B1_EXACTMETA_BEGIN",
        "RABIT2_STAGE4B2_EXACT_V2_FIXED_BEGIN",
        "RABIT2_STAGE4B3_GQA4_BEGIN",
        "RABIT2_STAGE4B4_CAUSAL_PREFILL_BEGIN",
    ):
        if needle not in rt:
            raise RuntimeError(f"Stage4C preflight missing {needle}")
    if "RABIT2_STAGE4B4_CAUSAL_PREFILL_TRITON_BEGIN" not in tt:
        raise RuntimeError("Stage4C preflight missing Stage4B4 Triton integration")

    compile(rt, str(RABIT), "exec")
    compile(tt, str(TRITON), "exec")
    print("Final Stage4B source preflight: PASSED")
    print(f"Benchmark model: {MODEL}")
    if "meta-llama/" in MODEL and not (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ):
        print(
            "NOTE: no local HF_TOKEN detected. If this gated model is not already "
            "accessible, set HF_TOKEN in this PowerShell before running."
        )

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

    h = hashlib.sha256()
    with SNAP.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    sha = h.hexdigest()
    print(
        f"Frozen source snapshot: {SNAP} "
        f"({n} files, {SNAP.stat().st_size/1024**2:.1f} MB)"
    )
    print(f"Snapshot SHA256: {sha}")
    return sha

def run():
    text = dec(RUNNER_Z)
    text = text.replace("__RABIT_STAGE4C_MODEL__", MODEL)
    text = text.replace("__RABIT_STAGE4C_SNAPSHOT__", SNAP.as_posix())
    RUNNER.write_text(text, encoding="utf-8")
    compile(text, str(RUNNER), "exec")

    time.sleep(2)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-m", "modal", "run", str(RUNNER)]
    print("Running:", " ".join(cmd))

    final_json = None
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
            if line.startswith("STAGE4C_FINAL_JSON="):
                final_json = line[len("STAGE4C_FINAL_JSON="):].strip()
        code = p.wait()

    if code:
        raise SystemExit(code)
    if final_json is None:
        raise RuntimeError("Stage4C completed without final JSON marker")

    parsed = json.loads(final_json)
    JSON_OUT.write_text(
        json.dumps(parsed, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Structured results saved to: {JSON_OUT}")

def main():
    print("RABIT-2 Stage 4C final controlled benchmark")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    run()
    print(f"Full log saved to: {LOG}")

if __name__ == "__main__":
    main()
