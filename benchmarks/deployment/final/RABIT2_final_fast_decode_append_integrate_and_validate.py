from __future__ import annotations
import base64, hashlib, os, shutil, subprocess, sys, tempfile, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"
BACKUP = ROOT / "rabit2_final_perf_backup/rabit_kv2.py"
SNAP = Path(tempfile.gettempdir()) / "rabit2_final_fast_decode_append_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_final_fast_decode_append.py"
LOG = ROOT / "rabit2_final_fast_decode_append.log"

MARK = "# === RABIT2_FINAL_FAST_DECODE_APPEND_BEGIN ==="
PATCH_Z = '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'
RUNNER_Z = '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'

IGNORE = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", ".venv", "venv", "node_modules",
}

def dec(x):
    return zlib.decompress(base64.b64decode(x)).decode("utf-8")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def install():
    if not RABIT.is_file():
        raise FileNotFoundError(RABIT)

    text = RABIT.read_text(encoding="utf-8")
    for needle in (
        "RABIT2_STAGE4D2_VECTORIZED_WRITER_BEGIN",
        "RABIT2_STAGE4D2_2_WRITER_ONLY_LOCKED_BEGIN",
        "RABIT2_STAGE4D3_4_TRITON_TAILPREP_BEGIN",
        "_rabit2_stage4d3_4_pack_k3_kernel",
        "_rabit2_stage4d3_4_emit_tail_partial",
    ):
        if needle not in text:
            raise RuntimeError(f"Final targeted fix prerequisite missing: {needle}")

    print(f"source_sha256_before={sha(RABIT)}")
    if MARK in text:
        print("Final fast decode-append already installed; validating existing source")
        print(f"source_sha256_after={sha(RABIT)}")
        return

    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2(RABIT, BACKUP)
        print(f"backup_created={BACKUP}")
        print(f"backup_sha256={sha(BACKUP)}")
    else:
        print(f"backup_exists={BACKUP}")
        print(f"backup_sha256={sha(BACKUP)}")

    patch = dec(PATCH_Z)
    text = text.rstrip() + "\n\n" + patch.strip() + "\n"
    compile(text, str(RABIT), "exec")
    RABIT.write_text(text, encoding="utf-8")
    print("Installed FINAL fast decode-append patch")
    print(f"source_sha256_after={sha(RABIT)}")

def restore():
    if BACKUP.is_file():
        shutil.copy2(BACKUP, RABIT)
        print("VALIDATION FAILED: restored validated D3.4-v5 source")
        print(f"restored_sha256={sha(RABIT)}")

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
        f"Frozen final targeted snapshot: {SNAP} "
        f"({count} files, {SNAP.stat().st_size/1024**2:.1f} MB)"
    )

def run():
    RUNNER.write_text(dec(RUNNER_Z), encoding="utf-8")
    compile(RUNNER.read_text(encoding="utf-8"), str(RUNNER), "exec")

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
    print("RABIT-2 FINAL targeted performance cycle — fast exact decode append")
    print(f"Project root: {ROOT}")
    install()
    snapshot()
    code = run()
    if code:
        restore()
        raise SystemExit(code)

    print("FINAL targeted validation succeeded; candidate remains active locally.")
    print(f"active_source={RABIT}")
    print(f"active_source_sha256={sha(RABIT)}")
    print(f"d3.4_v5_backup={BACKUP}")
    print(f"log={LOG}")
    print("PERFORMANCE FREEZE GATE REACHED: no further redesign after evaluating this result.")

if __name__ == "__main__":
    main()
