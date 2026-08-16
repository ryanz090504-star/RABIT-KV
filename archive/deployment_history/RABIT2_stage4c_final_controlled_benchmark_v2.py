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
LOG = ROOT / "rabit2_stage4c_v2_controlled.log"
JSON_OUT = ROOT / "rabit2_stage4c_v2_results.json"

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
            if line.startswith("STAGE4C_V2_FINAL_JSON="):
                final_json = line[len("STAGE4C_V2_FINAL_JSON="):].strip()
        code = p.wait()

    if code:
        raise SystemExit(code)
    if final_json is None:
        raise RuntimeError("Stage4C v2 completed without final JSON marker")

    parsed = json.loads(final_json)
    JSON_OUT.write_text(
        json.dumps(parsed, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Structured results saved to: {JSON_OUT}")

def main():
    print("RABIT-2 Stage 4C v2 corrected benchmark")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    run()
    print(f"Full log saved to: {LOG}")

if __name__ == "__main__":
    main()
