"""RABIT-2 Stage 4A H100 diagnostic profiler.

No source code is modified. This snapshots the current Stage-3C vLLM fork and
profiles the real RABIT-2 packed read/tail/reduction/page-close paths on an H100.
The output is diagnostic only and must not be used as final TTFT/TPOT claims.
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

PROJECT_ROOT = Path.cwd().resolve()
VLLM_ROOT = PROJECT_ROOT / "vllm-kvquant"
RABIT_PATH = VLLM_ROOT / "vllm/v1/attention/ops/rabit_kv2.py"
ATTN_PATH = VLLM_ROOT / "vllm/v1/attention/backends/triton_attn.py"

SNAPSHOT_PATH = Path(tempfile.gettempdir()) / "rabit2_stage4a_vllm_snapshot.zip"
RUNNER_PATH = Path(tempfile.gettempdir()) / "modal_rabit2_stage4a_profile.py"
LOG_PATH = PROJECT_ROOT / "rabit2_stage4a_profile.log"
RUNNER_Z = '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'

IGNORED_DIR_NAMES = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", ".venv", "venv", "node_modules",
}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".log"}


def _decode(value: str) -> str:
    return zlib.decompress(base64.b64decode(value.encode("ascii"))).decode("utf-8")


def _preflight() -> None:
    required = [RABIT_PATH, ATTN_PATH]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing current Stage-3C source files: {missing}")

    rabit = RABIT_PATH.read_text(encoding="utf-8")
    attn = ATTN_PATH.read_text(encoding="utf-8")
    needles_r = [
        "RABIT2_STAGE3C_LIFECYCLE_BEGIN",
        "rabit2_online_decode_attention_triton",
        "_rabit2_closed_page_partial_kernel",
        "_rabit2_tail_partial_kernel",
        "_rabit2_reduce_partials_kernel",
    ]
    needles_a = [
        "Stage-3C multi-request + chunked-prefill",
        "rabit2_online_decode_attention_triton",
    ]
    miss_r = [x for x in needles_r if x not in rabit]
    miss_a = [x for x in needles_a if x not in attn]
    if miss_r or miss_a:
        raise RuntimeError(f"Stage4A preflight failed: rabit={miss_r}, attn={miss_a}")
    compile(rabit, str(RABIT_PATH), "exec")
    compile(attn, str(ATTN_PATH), "exec")
    print("Stage 4A local source preflight: PASSED")


def _include(path: Path) -> bool:
    rel = path.relative_to(VLLM_ROOT)
    return (
        path.is_file()
        and not any(part in IGNORED_DIR_NAMES for part in rel.parts)
        and path.suffix.lower() not in IGNORED_SUFFIXES
        and not path.name.endswith(".egg-info")
    )


def _make_snapshot() -> None:
    tmp = SNAPSHOT_PATH.with_suffix(".zip.tmp")
    for path in (tmp, SNAPSHOT_PATH):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    count = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(VLLM_ROOT.rglob("*")):
            if _include(path):
                zf.write(path, path.relative_to(VLLM_ROOT).as_posix())
                count += 1
    os.replace(tmp, SNAPSHOT_PATH)
    print(
        f"Created frozen vLLM snapshot: {SNAPSHOT_PATH} "
        f"({count} files, {SNAPSHOT_PATH.stat().st_size / 1024**2:.1f} MB)"
    )


def _run_modal() -> int:
    RUNNER_PATH.write_text(_decode(RUNNER_Z), encoding="utf-8")
    time.sleep(2)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = [sys.executable, "-m", "modal", "run", str(RUNNER_PATH)]
    print("Running:", " ".join(cmd))
    with LOG_PATH.open("w", encoding="utf-8") as log:
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


def main() -> None:
    print("RABIT-2 Stage 4A diagnostic profiler")
    print(f"Project root: {PROJECT_ROOT}")
    _preflight()
    _make_snapshot()
    code = _run_modal()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
