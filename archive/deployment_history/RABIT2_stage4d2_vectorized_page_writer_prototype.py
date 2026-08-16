from __future__ import annotations
import base64, os, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"
TRITON = VLLM / "vllm/v1/attention/backends/triton_attn.py"
SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4d2_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4d2_writer.py"
LOG = ROOT / "rabit2_stage4d2_vectorized_writer.log"
RUNNER_Z = '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'
IGNORE = {".git",".github","__pycache__",".pytest_cache",".mypy_cache",".ruff_cache","build","dist",".venv","venv","node_modules"}

def dec(s):
    return zlib.decompress(base64.b64decode(s)).decode("utf-8")

def preflight():
    if not RABIT.is_file(): raise FileNotFoundError(RABIT)
    if not TRITON.is_file(): raise FileNotFoundError(TRITON)
    rt=RABIT.read_text(encoding="utf-8")
    for needle in (
        "RABIT2_STAGE4B4_CAUSAL_PREFILL_BEGIN",
        "def _rabit2_stage4b4_encode_page_from_primary",
        "def _rabit2_stage4b4_exact_v2_batch",
    ):
        if needle not in rt: raise RuntimeError(f"Stage4D2 preflight missing {needle}")
    compile(rt,str(RABIT),"exec")
    print("Final source preflight: PASSED")

def include(p):
    rel=p.relative_to(VLLM)
    return p.is_file() and not any(x in IGNORE for x in rel.parts) and p.suffix.lower() not in {".pyc",".pyo",".log"}

def snapshot():
    tmp=SNAP.with_suffix(".zip.tmp")
    for p in (tmp,SNAP):
        try:p.unlink()
        except FileNotFoundError:pass
    n=0
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(VLLM.rglob("*")):
            if include(p):
                z.write(p,p.relative_to(VLLM).as_posix()); n+=1
    os.replace(tmp,SNAP)
    print(f"Frozen snapshot: {SNAP} ({n} files, {SNAP.stat().st_size/1024**2:.1f} MB)")

def run():
    RUNNER.write_text(dec(RUNNER_Z),encoding="utf-8")
    compile(RUNNER.read_text(encoding="utf-8"),str(RUNNER),"exec")
    time.sleep(2)
    env=os.environ.copy(); env["PYTHONUTF8"]="1"; env["PYTHONIOENCODING"]="utf-8"
    cmd=[sys.executable,"-m","modal","run",str(RUNNER)]
    print("Running:"," ".join(cmd))
    with LOG.open("w",encoding="utf-8") as log:
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",env=env,bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line,end=""); log.write(line); log.flush()
        return p.wait()

def main():
    print("RABIT-2 Stage 4D2 vectorized exact page-writer prototype")
    print(f"Project root: {ROOT}")
    preflight(); snapshot()
    code=run()
    if code: raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__=="__main__":
    main()
