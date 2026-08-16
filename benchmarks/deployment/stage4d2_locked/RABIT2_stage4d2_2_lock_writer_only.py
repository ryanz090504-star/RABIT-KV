from __future__ import annotations
import base64, os, shutil, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"
BACKUP = ROOT / "rabit2_stage4d2_2_backup/rabit_kv2.py"

SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4d2_2_locked_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4d2_2_lock.py"
LOG = ROOT / "rabit2_stage4d2_2_lock_writer_only.log"
RUNNER_Z = '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'

MARK = "# === RABIT2_STAGE4D2_2_WRITER_ONLY_LOCKED_BEGIN ==="

IGNORE = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", ".venv", "venv", "node_modules",
}

def dec(x):
    return zlib.decompress(base64.b64decode(x)).decode("utf-8")

def install():
    if not RABIT.is_file():
        raise FileNotFoundError(RABIT)

    text = RABIT.read_text(encoding="utf-8")
    for needle in (
        "RABIT2_STAGE4D2_VECTORIZED_WRITER_BEGIN",
        "_rabit2_stage4d2_old_chunkplan_init",
        "Rabit2CausalChunkPlan.__init__ = _rabit2_stage4d2_chunkplan_init",
    ):
        if needle not in text:
            raise RuntimeError(f"Stage4D2.2 prerequisite missing: {needle}")

    if MARK in text:
        print("Stage4D2.2 writer-only lock already installed")
        return

    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2(RABIT, BACKUP)

    patch = """
# === RABIT2_STAGE4D2_2_WRITER_ONLY_LOCKED_BEGIN ===
# Keep the Stage4D2 vectorized physical writer, but restore the locked
# Stage4B4 causal chunk-plan constructor. Controlled H100 A/B showed that
# the Stage4D2 chunk-plan override caused a major decode regression.
Rabit2CausalChunkPlan.__init__ = _rabit2_stage4d2_old_chunkplan_init
_RABIT2_STAGE4D2_2_WRITER_ONLY_LOCKED = True
# === RABIT2_STAGE4D2_2_WRITER_ONLY_LOCKED_END ===
"""
    text = text.rstrip() + "\n\n" + patch.strip() + "\n"
    compile(text, str(RABIT), "exec")
    RABIT.write_text(text, encoding="utf-8")
    print("Installed Stage4D2.2 writer-only lock")

def restore():
    if BACKUP.is_file():
        shutil.copy2(BACKUP, RABIT)
        print("Validation failed; restored full Stage4D2 source")

def include(path):
    rel = path.relative_to(VLLM)
    return (
        path.is_file()
        and not any(x in IGNORE for x in rel.parts)
        and path.suffix.lower() not in {".pyc", ".pyo", ".log"}
    )

def snapshot():
    tmp = SNAP.with_suffix(".zip.tmp")
    for p in (tmp, SNAP):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    count = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(VLLM.rglob("*")):
            if include(p):
                z.write(p, p.relative_to(VLLM).as_posix())
                count += 1

    os.replace(tmp, SNAP)
    print(
        f"Frozen locked snapshot: {SNAP} "
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
    print("RABIT-2 Stage4D2.2 — lock writer-only candidate")
    print(f"Project root: {ROOT}")
    install()
    snapshot()
    code = run()
    if code:
        restore()
        raise SystemExit(code)
    print("Stage4D2.2 writer-only lock is now active locally.")
    print(f"Backup of full Stage4D2: {BACKUP}")
    print(f"Log: {LOG}")

if __name__ == "__main__":
    main()
