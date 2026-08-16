"""Integrate RABIT-2 Stage 4B1 fused sidecar-tail path and verify on H100.

This appends a guarded Stage4B1 block to rabit_kv2.py, keeps the previous
Stage3C online function as an explicit fallback/reference, installs correctness
tests, snapshots the fork, and runs regression + microbenchmark + real engine smoke.
"""
from __future__ import annotations

import base64, os, shutil, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"
TEST = VLLM / "tests/quantization/test_rabit_kv2_stage4b1.py"
BACKUP = ROOT / "rabit2_stage4b1_backup"
SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4b1_integrated_vllm_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4b1_integrated.py"
LOG = ROOT / "rabit2_stage4b1_integrated.log"

PATCH_Z = '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'
TEST_Z = 'eNqdVMGOmzAQvfMVVk5GpV5C2t1oVaRW7RfsqqeqsgwMiQXYxDZos1/fMQ4JVG2lLQJbtmfevHkzOKqN7gjn9eAGA5wT2fXaOCKU0k44qZWNostef3Zg3bxy2pTHKIo+h23WCdMw28he1hR9wzkrh0owabkYhWxF0QKNE2JAWK3yzdfv377g4jRIA9UmXkP1wogOnJGvQDcOybTc6QaU3STkx8eE3Cdku0tIluL3kJAdbj2kP+Oogpp4FG6dOMCHYss74coj2LCxK7lWrVTAhXOgfIZ0if5IpHLxY0TwmbQZ27Zj45ZdzZnuLTOikI43YzYLRicP/zz5o+xZqkMLz5gcqBKeBnTtILka8Qkgm7lUUOpqQYlj3g6nmTK8iNLdvN/gjAIEvziaplCWTqgBU7YAFd2maUrekaUI8WRawShLIPnFJyzpxpd0EyxOx4Q0Iw4VWu2yhOyxKNl+OkNN8n9pQYMjvgFLDR0vWl02Fv068UIRa8mJ3N1hCCS6D/alwKJeyb2C0fZWA3pDQ0bzaxxrxVkPDpvrALzw3RbfVK3cuYc84A3YBfvF0ZR7HqZZTz8W7koB+1UdYBV5iYiAXqEVUsBorhCIUCm66sdZ4HiNVtStFm57/0fAcQ3IW9mg3PGlLEz0PaiKNgkZkyBjgnlcKvo7F5Tt9Jb4IQj+g/l/9Tg9LSj5igVaB+11flvb/wVqcS/ZsyqPRit/xVyYgzEYiGICbMqQxuS9jz6vYiYKSwOSsBbmixDvuFoq6YCiMRq17doouCM6860dM7TsEI18yknK0iz6BX2xylk='
RUNNER_Z = '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'

MARK = "# === RABIT2_STAGE4B1_FUSED_TAIL_BEGIN ==="
IGNORE_DIRS = {".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
               "build", "dist", ".venv", "venv", "node_modules"}
IGNORE_SUFFIX = {".pyc", ".pyo", ".log"}

def dec(x):
    return zlib.decompress(base64.b64decode(x.encode("ascii"))).decode("utf-8")

def backup(path):
    if not path.exists():
        return
    dst = BACKUP / path.relative_to(ROOT)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)

def install():
    if not RABIT.is_file():
        raise FileNotFoundError(RABIT)
    text = RABIT.read_text(encoding="utf-8")
    for need in (
        "RABIT2_STAGE3C_LIFECYCLE_BEGIN",
        "def rabit2_online_decode_attention_triton(",
        "_rabit2_closed_page_partial_kernel",
        "_rabit2_reduce_partials_kernel",
    ):
        if need not in text:
            raise RuntimeError(f"Stage4B1 source preflight missing {need}")
    backup(RABIT)
    backup(TEST)
    if MARK not in text:
        text = text.rstrip() + "\n\n" + dec(PATCH_Z).strip() + "\n"
        compile(text, str(RABIT), "exec")
        RABIT.write_text(text, encoding="utf-8")
        print(f"Installed Stage4B1 fused-tail path: {RABIT}")
    else:
        print("Stage4B1 fused-tail patch already present; not duplicating.")
    TEST.parent.mkdir(parents=True, exist_ok=True)
    TEST.write_text(dec(TEST_Z), encoding="utf-8")
    compile(TEST.read_text(encoding="utf-8"), str(TEST), "exec")
    print(f"Installed Stage4B1 tests: {TEST}")

def preflight():
    text = RABIT.read_text(encoding="utf-8")
    for need in (
        "RABIT2_STAGE4B1_FUSED_TAIL_BEGIN",
        "rabit2_online_decode_attention_triton_stage4b1",
        "_rabit2_online_decode_attention_triton_stage3c_exact",
    ):
        if need not in text:
            raise RuntimeError(f"Stage4B1 post-install preflight missing {need}")
    compile(text, str(RABIT), "exec")
    print("Stage 4B1 local source preflight: PASSED")

def include(p):
    rel = p.relative_to(VLLM)
    return p.is_file() and not any(x in IGNORE_DIRS for x in rel.parts) and p.suffix.lower() not in IGNORE_SUFFIX

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

def run_modal():
    RUNNER.write_text(dec(RUNNER_Z), encoding="utf-8")
    time.sleep(2)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = [sys.executable, "-m", "modal", "run", str(RUNNER)]
    print("Running:", " ".join(cmd))
    with LOG.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", env=env, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line); log.flush()
        return p.wait()

def main():
    print("RABIT-2 Stage 4B1 integration + H100 verification")
    print(f"Project root: {ROOT}")
    install()
    preflight()
    snapshot()
    code = run_modal()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__ == "__main__":
    main()
