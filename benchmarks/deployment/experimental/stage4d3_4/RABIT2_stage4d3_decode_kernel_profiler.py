from __future__ import annotations
import base64, os, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"

SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4d3_decode_profiler_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4d3_decode_profiler.py"
LOG = ROOT / "rabit2_stage4d3_decode_kernel_profiler.log"
RUNNER_Z = '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'

IGNORE = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", ".venv", "venv", "node_modules",
}

def dec(x):
    return zlib.decompress(base64.b64decode(x)).decode("utf-8")

def include(path):
    rel = path.relative_to(VLLM)
    return (
        path.is_file()
        and not any(x in IGNORE for x in rel.parts)
        and path.suffix.lower() not in {".pyc", ".pyo", ".log"}
    )

def preflight():
    if not RABIT.is_file():
        raise FileNotFoundError(RABIT)
    text = RABIT.read_text(encoding="utf-8")
    for needle in (
        "RABIT2_STAGE4D2_VECTORIZED_WRITER_BEGIN",
        "RABIT2_STAGE4D2_2_WRITER_ONLY_LOCKED_BEGIN",
        "RABIT2_STAGE4B3_GQA4_BEGIN",
    ):
        if needle not in text:
            raise RuntimeError(f"Stage4D3 prerequisite missing: {needle}")
    print("Stage4D3 locked-source preflight: PASSED")

def snapshot():
    tmp = SNAP.with_suffix(".zip.tmp")
    for p in (tmp, SNAP):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    count = 0
    with zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for p in sorted(VLLM.rglob("*")):
            if include(p):
                z.write(p, p.relative_to(VLLM).as_posix())
                count += 1

    os.replace(tmp, SNAP)
    print(
        f"Frozen Stage4D3 snapshot: {SNAP} "
        f"({count} files, {SNAP.stat().st_size/1024**2:.1f} MB)"
    )

def run():
    RUNNER.write_text(dec(RUNNER_Z), encoding="utf-8")
    compile(RUNNER.read_text(encoding="utf-8"), str(RUNNER), "exec")
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
    print("RABIT-2 Stage4D3 decode kernel profiler")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    code = run()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")
    print("Source tree was NOT modified.")

if __name__ == "__main__":
    main()
