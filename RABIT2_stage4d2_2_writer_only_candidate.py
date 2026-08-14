from __future__ import annotations
import base64, os, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"

SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4d2_2_writer_only_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4d2_2_writer_only.py"
LOG = ROOT / "rabit2_stage4d2_2_writer_only_candidate.log"
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
        "_rabit2_stage4d2_old_chunkplan_init",
        "Rabit2CausalChunkPlan.__init__ = _rabit2_stage4d2_chunkplan_init",
    ):
        if needle not in text:
            raise RuntimeError(f"Stage4D2.2 preflight missing: {needle}")
    print("Stage4D2.2 local source preflight: PASSED")

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
        f"Frozen Stage4D2 snapshot: {SNAP} "
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
    print("RABIT-2 Stage4D2.2 writer-only candidate")
    print(f"Project root: {ROOT}")
    preflight()
    snapshot()
    code = run()
    if code:
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")
    print("Local production source was NOT modified.")

if __name__ == "__main__":
    main()
