from __future__ import annotations
import base64, os, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"

SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4d3_2_hybrid_k3_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4d3_2_hybrid_k3.py"
LOG = ROOT / "rabit2_stage4d3_2_hybrid_exact_k3_prototype.log"
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
            raise RuntimeError(f"Stage4D3.2 prerequisite missing: {needle}")
    print("Stage4D3.2 locked-source preflight: PASSED")

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
        f"Frozen Stage4D3.2 snapshot: {SNAP} "
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
    print("RABIT-2 Stage4D3.2 hybrid exact K3 prototype")
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
