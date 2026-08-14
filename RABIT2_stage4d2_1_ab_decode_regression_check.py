from __future__ import annotations
import base64, os, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"
BACKUP = ROOT / "rabit2_stage4d2_backup/rabit_kv2.py"

SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4d2_ab_single_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4d2_1_ab_fixed.py"
LOG = ROOT / "rabit2_stage4d2_1_ab_decode_regression.log"
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
    if not BACKUP.is_file():
        raise FileNotFoundError(BACKUP)

    cur = RABIT.read_text(encoding="utf-8")
    pre = BACKUP.read_text(encoding="utf-8")

    if "RABIT2_STAGE4D2_VECTORIZED_WRITER_BEGIN" not in cur:
        raise RuntimeError("Current source is not Stage4D2 integrated")
    if "RABIT2_STAGE4D2_VECTORIZED_WRITER_BEGIN" in pre:
        raise RuntimeError("Backup unexpectedly contains Stage4D2")

    print("A/B source preflight: PASSED")

def make_single_snapshot():
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
            if not include(p):
                continue
            z.write(p, p.relative_to(VLLM).as_posix())
            count += 1

        z.write(
            BACKUP,
            "_stage4d2_ab/pre_rabit_kv2.py",
        )

    os.replace(tmp, SNAP)
    print(
        f"Frozen single A/B snapshot: {SNAP} "
        f"({count} repo files, {SNAP.stat().st_size/1024**2:.1f} MB)"
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
    print("RABIT-2 Stage4D2.1 controlled A/B — FIXED SINGLE-SNAPSHOT")
    print(f"Project root: {ROOT}")
    preflight()
    make_single_snapshot()
    code = run()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__ == "__main__":
    main()
