from __future__ import annotations
import base64, os, shutil, subprocess, sys, tempfile, time, zipfile, zlib
from pathlib import Path

ROOT=Path.cwd().resolve()
VLLM=ROOT/"vllm-kvquant"
RABIT=VLLM/"vllm/v1/attention/ops/rabit_kv2.py"
FAILED_BACKUP=ROOT/"rabit2_stage4b1_backup"/"vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py"
TEST=VLLM/"tests/quantization/test_rabit_kv2_stage4b1_exactmeta.py"
SNAP=Path(tempfile.gettempdir())/"rabit2_stage4b1_exactmeta_integrated_snapshot.zip"
RUNNER=Path(tempfile.gettempdir())/"modal_rabit2_stage4b1_exactmeta_integrated.py"
LOG=ROOT/"rabit2_stage4b1_exactmeta_integrated.log"
PATCH_Z='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'
TEST_Z='eNqdVF1v2yAUffevQHnCGqP+aJp2UqRN2y9otadpQti+SZBtcABbTX/9LibJkqqb1DqRfczlXM49XJNsrOmJEJvRjxaEIKofjPVEam289MpolxyHhoMH509v3th6lyRf4yjvpW25a9WgNhSZMczrsZFcOSEnqTpZdUBTRixIZ/R68f3nj2/4sh+VhWaRXqcapJU9eKtegCYEr4VHPZ3wpgXtmAOksDnwiy5ZnmXLlNG7AO4Q5CWivERUZIiKLKBVQCtEZc7KrMwDCtQyUJcP7DZbPiBaZYhWWfqbJWnSwIYEUcJ5uYXbKhfwLGuPyqTopa934GKorIXRndJAr3SSIDT9MgudnZ66rudTzqX3oIO93AyOW1kpL9qpONkfaw7XYwgVT0pvO3hCs0DX8DgitQd2niTmBMVRgmigNg2I8xoCffT4OCmda/jLfgf5yoGYIZ3vcb97qUcsPhRN58rn2H7HSDvhrSFrUhaM3DOSF/dzDGtd/69GGon4j7mqztStQ04vnynmuXSb3NxgevKJ3Me5tcT9walR2wtY4yiNCVg+/6znnTyY0WO/bUFUoQGxQxt/GGAdaaPSHtdpYFI1rBehoxdHKf6cG5tVb+GU+4qP9FDyG/z2TEd2o+l154S6m1daqk1npM/v3kw3XacTnWrRvfToMpfDALqhLSMTi84wrOC4Qa+V5Gz/ztUtfijrD/Uh3V/owdPhqGlrgr0fbc1/JL04ldxB1ztrdDhgYlA6B6eDjWMXyo5iVSwoQSdefdezFzRM4BGmn3HeCadcVo6mPDRpypWHHsfS5A+7u8J3'
RUNNER_Z='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'
FAILED_MARK="# === RABIT2_STAGE4B1_FUSED_TAIL_BEGIN ==="
FINAL_MARK="# === RABIT2_STAGE4B1_EXACTMETA_BEGIN ==="
IGNORE={".git",".github","__pycache__",".pytest_cache",".mypy_cache",".ruff_cache","build","dist",".venv","venv","node_modules"}

def dec(s): return zlib.decompress(base64.b64decode(s)).decode()

def restore():
    text=RABIT.read_text(encoding="utf-8")
    if FAILED_MARK in text:
        if not FAILED_BACKUP.is_file():
            raise FileNotFoundError(f"Missing automatic Stage3C backup: {FAILED_BACKUP}")
        good=FAILED_BACKUP.read_text(encoding="utf-8")
        if FAILED_MARK in good or FINAL_MARK in good:
            raise RuntimeError("Stage3C backup is not clean")
        compile(good,str(RABIT),"exec")
        RABIT.write_text(good,encoding="utf-8")
        print(f"Restored clean Stage3C source from: {FAILED_BACKUP}")

def install():
    if not RABIT.is_file(): raise FileNotFoundError(RABIT)
    restore()
    text=RABIT.read_text(encoding="utf-8")
    if FINAL_MARK not in text:
        text=text.rstrip()+"\n\n"+dec(PATCH_Z).strip()+"\n"
        compile(text,str(RABIT),"exec")
        RABIT.write_text(text,encoding="utf-8")
        print(f"Installed Stage4B1 exactmeta path: {RABIT}")
    TEST.write_text(dec(TEST_Z),encoding="utf-8")
    compile(TEST.read_text(encoding="utf-8"),str(TEST),"exec")
    print(f"Installed tests: {TEST}")

def preflight():
    text=RABIT.read_text(encoding="utf-8")
    if FAILED_MARK in text: raise RuntimeError("Failed Stage4B1 block still present")
    for n in ("RABIT2_STAGE4B1_EXACTMETA_BEGIN","_rabit2_online_decode_attention_triton_stage3c_exact",
              "rabit2_online_decode_attention_triton_stage4b1_exactmeta"):
        if n not in text: raise RuntimeError(f"Missing {n}")
    print("Stage 4B1 exactmeta local source preflight: PASSED")

def include(p):
    rel=p.relative_to(VLLM)
    return p.is_file() and not any(x in IGNORE for x in rel.parts) and p.suffix.lower() not in {".pyc",".pyo",".log"}

def snapshot():
    tmp=SNAP.with_suffix(".zip.tmp")
    for p in (tmp,SNAP):
        try: p.unlink()
        except FileNotFoundError: pass
    n=0
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(VLLM.rglob("*")):
            if include(p):
                z.write(p,p.relative_to(VLLM).as_posix()); n+=1
    os.replace(tmp,SNAP)
    print(f"Created frozen snapshot: {SNAP} ({n} files, {SNAP.stat().st_size/1024**2:.1f} MB)")

def run():
    RUNNER.write_text(dec(RUNNER_Z),encoding="utf-8"); time.sleep(2)
    env=os.environ.copy(); env.setdefault("PYTHONUTF8","1"); env.setdefault("PYTHONIOENCODING","utf-8")
    cmd=[sys.executable,"-m","modal","run",str(RUNNER)]
    print("Running:"," ".join(cmd))
    with LOG.open("w",encoding="utf-8") as log:
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",
                           errors="replace",env=env,bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line,end=""); log.write(line); log.flush()
        return p.wait()

def main():
    print("RABIT-2 Stage 4B1 exactmeta repair + integration")
    print(f"Project root: {ROOT}")
    install(); preflight(); snapshot()
    code=run()
    if code: raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__=="__main__":
    main()
