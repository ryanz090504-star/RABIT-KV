from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from pathlib import Path

ROOT = Path.cwd().resolve()
VLLM = ROOT / "vllm-kvquant"
RABIT = VLLM / "vllm/v1/attention/ops/rabit_kv2.py"
TEST = VLLM / "tests/quantization/test_rabit_kv2_stage4b3_gqa4.py"
BACK = ROOT / "rabit2_stage4b3_backup" / "rabit_kv2.py"
SNAP = Path(tempfile.gettempdir()) / "rabit2_stage4b3_gqa4_integrated_snapshot.zip"
RUNNER = Path(tempfile.gettempdir()) / "modal_rabit2_stage4b3_gqa4_integrated.py"
LOG = ROOT / "rabit2_stage4b3_gqa4_integrated.log"

P = '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'
T = 'eNrtlFFv2zYQx9/1KQ7ui4QqXOymnTHAw9Zue+nTEnQvRUCcpLNNWCJlktLqfPodRVu2k2xo0vRtgi1bJO/4v/+Jv2RpTQNSLjvfWZISVNMa6wG1Nh69MtolyX6s3Xly/vDkjS3XSZL8EodFg3Yj3Ea1aplybJwXZVehUE5ij6rGoqY0y8ESOqMXkw+ffvuVH7adslRNsvNULVpsyFt1R+mkrI2jSra4IjfJ4fM0h1kO07c5vLu6zZKKlhAipfO84qp4I1dbvJIN+nJN7jA6k/QFS5+eJst+SoCvfU19XTeinwr0nnQoXpjWCYuF8nLTzwAd2GQIiNU1qDuspSOq0h8vLy/hNZwlH5Zu1zlser5VsIA3LHvOymfzYY73XIAV12GH2Y3Sq5pu2A/SJV13rKChNIbyJ2YrkUvioCiAmtbv0mEiXGelsZiw0+FjvahxZzovwqwsgtVZPoZWftfSImbtlPbzkynqVUmLSejlJA5n0YVX8Ieqa0D+tuudUyXWewMgSvhb+TUMrg8DMOwqhtilsdCC0mBRr+ixpoRrM5bKyyqdpsHAaAm/SKeii2Vt0E/f5ed6szFVf55K1mrD7h7nB4GhG+y+qUgObZ8NiuKLI0PCdJNDn0O0cjGaekwzNOhzeyssuTW2lF5MM1GadifTkGrvXOFHNfjQgNi60+q4I6Hye6Xtm/D7YHBhOl2x805VVKIFzycuWu0fuDi7fI6L/lEL/d5D9gLblnTFIzmvzaMVOdc6LjgrcnF2WGIx2/tK+c3dPlWnZRyERh5aaHStNEm2JfR1PNyS4eL55x4exk5uTyoI52f/5of7ysSD+6QNIpW+Kv8JPN1Ol2trdOBgnETn6ABgwbTAOuWK8yCKXTrmP3X3eJoH49IQIeLfDC5C6OEpE1i4NGO2feG78tTw2HjqXxL3/0LtJeOkwHIjmRBSs32V6pVTnIz7ybamz2X2K5jDn/AD3z/+BYufYcgW0Nh0zkPnaE+qm0HO+xkclIiHHJ+/KMbvyBp3gvHvyO3H4fNVtHmUx1fPAcl/ofiIkUjaexD5HxDfBojkHxOXHGE='
R = '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'
MARK = "# === RABIT2_STAGE4B3_GQA4_BEGIN ==="
IGNORE = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", ".venv", "venv", "node_modules"
}

def dec(s):
    return zlib.decompress(base64.b64decode(s)).decode("utf-8")

def install():
    if not RABIT.is_file():
        raise FileNotFoundError(RABIT)
    text = RABIT.read_text(encoding="utf-8")
    for need in (
        "RABIT2_STAGE4B1_EXACTMETA_BEGIN",
        "RABIT2_STAGE4B2_EXACT_V2_FIXED_BEGIN",
    ):
        if need not in text:
            raise RuntimeError(f"Missing required stage: {need}")

    BACK.parent.mkdir(parents=True, exist_ok=True)
    if not BACK.exists():
        shutil.copy2(RABIT, BACK)

    if MARK not in text:
        text = text.rstrip() + "\n\n" + dec(P).strip() + "\n"
        compile(text, str(RABIT), "exec")
        RABIT.write_text(text, encoding="utf-8")
        print("Installed Stage4B3 GQA4 path")
    else:
        print("Stage4B3 GQA4 patch already present.")

    TEST.parent.mkdir(parents=True, exist_ok=True)
    TEST.write_text(dec(T), encoding="utf-8")
    compile(TEST.read_text(encoding="utf-8"), str(TEST), "exec")
    print("Stage 4B3 GQA4 local source preflight: PASSED")

def include(p):
    rel = p.relative_to(VLLM)
    return (
        p.is_file()
        and not any(x in IGNORE for x in rel.parts)
        and p.suffix.lower() not in {".pyc", ".pyo", ".log"}
    )

def snapshot():
    tmp = SNAP.with_suffix(".zip.tmp")
    for p in (tmp, SNAP):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    n = 0
    with zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for p in sorted(VLLM.rglob("*")):
            if include(p):
                z.write(p, p.relative_to(VLLM).as_posix())
                n += 1
    os.replace(tmp, SNAP)
    print(
        f"Created frozen snapshot: {SNAP} "
        f"({n} files, {SNAP.stat().st_size/1024**2:.1f} MB)"
    )

def restore():
    if BACK.is_file():
        shutil.copy2(BACK, RABIT)
        print("Verification failed; restored pre-Stage4B3 rabit_kv2.py")

def run():
    RUNNER.write_text(dec(R), encoding="utf-8")
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
    print("RABIT-2 Stage 4B3 GQA4 integration")
    print(f"Project root: {ROOT}")
    install()
    snapshot()
    code = run()
    if code:
        restore()
        raise SystemExit(code)
    print(f"Log saved to: {LOG}")

if __name__ == "__main__":
    main()
