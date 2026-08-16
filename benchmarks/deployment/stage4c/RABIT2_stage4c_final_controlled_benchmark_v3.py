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
LOG = ROOT / "rabit2_stage4c_v3_controlled.log"
JSON_OUT = ROOT / "rabit2_stage4c_v3_results.json"

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
            if line.startswith("STAGE4C_V3_FINAL_JSON="):
                final_json = line[len("STAGE4C_V3_FINAL_JSON="):].strip()
        code = p.wait()

    if code:
        raise SystemExit(code)
    if final_json is None:
        raise RuntimeError("Stage4C v3 completed without final JSON marker")

    parsed = json.loads(final_json)
    JSON_OUT.write_text(
        json.dumps(parsed, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Structured results saved to: {JSON_OUT}")

def main():
    print("RABIT-2 Stage 4C v3 strict backend-controlled benchmark")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    run()
    print(f"Full log saved to: {LOG}")

if __name__ == "__main__":
    main()
