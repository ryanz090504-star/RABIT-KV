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
    "LLM-Research/Meta-Llama-3.1-8B-Instruct",
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
    print(
        "Model source: ModelScope persistent cache "
        "(modelscope-llama31-cache -> /model_cache)"
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
