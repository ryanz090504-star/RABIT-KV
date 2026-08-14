from __future__ import annotations
import base64, os, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"

SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4d3_3_fast_meta_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4d3_3_fast_meta.py"
LOG = ROOT / "rabit2_stage4d3_3_fast_meta_prototype.log"
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
        "def _rabit2_stage4b1_exactmeta_emit_tail_partial",
    ):
        if needle not in text:
            raise RuntimeError(f"Stage4D3.3 prerequisite missing: {needle}")
    print("Stage4D3.3 locked-source preflight: PASSED")

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
        f"Frozen Stage4D3.3 snapshot: {SNAP} "
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
    print("RABIT-2 Stage4D3.3 fast META8g64 prototype")
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
