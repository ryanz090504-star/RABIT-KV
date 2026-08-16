from __future__ import annotations
import base64, os, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"

SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4d3_1_exactmeta_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4d3_1_exactmeta.py"
LOG = ROOT / "rabit2_stage4d3_1_exactmeta_compile_prototype.log"
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
            raise RuntimeError(f"Stage4D3.1 prerequisite missing: {needle}")
    print("Stage4D3.1 locked-source preflight: PASSED")

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
        f"Frozen Stage4D3.1 snapshot: {SNAP} "
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
    print("RABIT-2 Stage4D3.1 exactmeta compile prototype")
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
