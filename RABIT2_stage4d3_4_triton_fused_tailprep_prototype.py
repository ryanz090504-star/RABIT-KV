from __future__ import annotations
import base64, os, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"

SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4d3_4_triton_tailprep_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4d3_4_triton_tailprep.py"
LOG = ROOT / "rabit2_stage4d3_4_triton_fused_tailprep_prototype.log"
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
            raise RuntimeError(f"Stage4D3.4 prerequisite missing: {needle}")
    print("Stage4D3.4 locked-source preflight: PASSED")

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
        f"Frozen Stage4D3.4 snapshot: {SNAP} "
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
    print("RABIT-2 Stage4D3.4 Triton fused exact tail-prep prototype")
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
