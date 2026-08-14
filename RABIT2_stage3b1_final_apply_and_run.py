"""Apply and run RABIT-2 Stage 3B1 Triton closed-page kernels.

Run from:
  C:\\Users\\ryanz\\Documents\\GitHub\\benchquant\\kvquant_full\\kvquant_pkg

What this stage DOES:
- Keeps an experimental Triton K3/V2 writer for diagnostics.
- Uses the byte-exact CUDA/PyTorch closed-page writer for deployed correctness.
- Adds Triton direct page decode.
- Adds a fused packed-read/dequant attention kernel for one immutable 32-token page.
- Runs Stage 1/2/3A regression tests plus Stage 3B1 CUDA tests on H100.

What this stage intentionally DOES NOT do:
- It does not remove the rabit_kv2 NotImplemented guard in triton_attn.py.
- It does not claim end-to-end vLLM serving or latency yet.
  Exact R4/open-K online staging must be integrated first (Stage 3B2).
"""
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

PROJECT_ROOT = Path.cwd().resolve()
VLLM_ROOT = PROJECT_ROOT / "vllm-kvquant"
RABIT_PATH = VLLM_ROOT / "vllm/v1/attention/ops/rabit_kv2.py"
TRITON_ATTN_PATH = VLLM_ROOT / "vllm/v1/attention/backends/triton_attn.py"
TEST_PATH = VLLM_ROOT / "tests/quantization/test_rabit_kv2_stage3b1.py"
BACKUP_ROOT = PROJECT_ROOT / "rabit2_stage3b1_backup"
SNAPSHOT_PATH = Path(tempfile.gettempdir()) / "rabit2_stage3b1_vllm_snapshot.zip"
RUNNER_PATH = Path(tempfile.gettempdir()) / "modal_rabit2_stage3b1_frozen_test.py"
LOG_PATH = PROJECT_ROOT / "rabit2_stage3b1_final.log"

BEGIN = "# === RABIT2_STAGE3B1_TRITON_BEGIN ==="
END = "# === RABIT2_STAGE3B1_TRITON_END ==="

FRAGMENT_Z = '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'
TEST_Z = 'eNrtV91v2zYQf/dfwbkvUiErtrKmmTENKxasGLCHAsWAAcMg0CIVE6JImaSUuH/9jh+y4kRZki710/wQSby7330fL2/Q509Xfy5+ZyUVmi5+I1QYVjGq1uhDi8stXWTpcsaaViqDGmy2s+Gj3RuqzfBlpCqBVinZoJ7zJu1XKTbGokmRylanCm+YKeo+Q0EkmiH4EVpKQgtHzYoWX8M7rZLHaEYxI4UnU/GATG9xaYqyI/hRlgP6BM2jA0pLFWvAeMw9b2AqudSUeN6Ddweb4mP3674obQQLJgxVFS7p4PldlRzvZWdmsxmhFQpfUbz2WqnplJhg96Gzvw2XZV1o9oXm51lyOBZdA6EuthQTnV+O5/bAcRd1vsomCf1IiAezNG5aTiNCe6iTfG7jOw82usSnDRYd5oWmlETZMrtYXi7fx45eoxxFnklhQUQEZqLLBFklKAD6R4zeomX6Pk6NDAKbiktsVhceqX8h0g+PIoW41gnqwcOffSWDD6pOdc1aVkVChpJOra8p0wXuMeN4A1GIEwDAWop8/ssfVx/gY9cxRck8dsGyWIU2kKzzzWqoKF/JqoAOgorQtgipggKkLqtDvn1ywc1DHfgQJs73IQv+0MrB4XSBR1YkCXi5fwQsBRQFgtN9F9mXe4IQXLNvae7j4QJ5ngU0IPZmGs17/kJAf2RDyMR1irWmyviui6wua70ykucruniXIDy8PincW0MfFf7GNTAOpeIGggJlAFgbUOgpX5N9SNVXJN+pm5YbbXxc3Ec0xAQcxjxyYjYW1QmiOA7loa3uhFN2gsBoJgxfCwkVUA5hnc/nV4dDJAXfrxGRyNrVwiCggiAGoi28ADb69ROqqRKUa4QVRTZNC+ZuxRLzFMBOlCvwcFpq4o56OmWKpXqLW5gYubXHf9zncF1pOXy+Ori3Lh1Pw7QbXGAQnEWRte07BxSnumuiOIU0wMPrfIM+dlgRhK8xE9ogjLRRXQkjF/OzEDhFrxXVGm5PdLNlnCLMubyBtgVu6N59AHKzAU4XrQTFSNkkWybCqjA/wSBkthTdDQfyZZHe9e/gwo8oe3dxoqnvsxf2BWc8EFpbrpryqiil0EzbLeJ1L4AXVcj/F8LpL4Sqs1vk9DL5YEN43dp4zpyvCVDIa5TEv2+Il17dDg0TZ1zr7u50ft081nC8z2kYzdbfVbpEZ+6/lFTvlIkAxTOAhYUfqM9Y5MflepegSV+1rEyDbwunN3d//b7sPY+C47CAM3J78A6De7DrWfeOF2l0doa+9xI2FjX5a50E4QSt//bLr6X0UxRdShilBy0UZi6M5PmWJGZLFj9tDYRuNxgG0ju7HY82t0puRungWeRBwVDW5IvVGER/l93TZIImApocHJTP7skODDlJBtyxF7OxF7Nv3otCFptqdWH7DqYl5uwLdrUwCBWVVKFhAZScvhv/e3t4Q59T+dMFf7RKwPe4SuweLhKWflgkdv714fbIdMUEXNK2CobajFNYAiBk/wCKYYI5'
RUNNER_Z = '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'

IGNORED_DIR_NAMES = {
    ".git", ".github", ".buildkite", ".venv", "__pycache__", "build",
    "dist", "docs", "examples", "benchmarks", ".deps", ".pytest_cache",
    ".mypy_cache", ".ruff_cache",
}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".so", ".dll", ".dylib"}


def _decode(value: str) -> str:
    return zlib.decompress(base64.b64decode(value.encode("ascii"))).decode("utf-8")


def _backup(path: Path) -> None:
    if not path.exists():
        return
    rel = path.relative_to(PROJECT_ROOT)
    dst = BACKUP_ROOT / rel
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def _strip_old_fragment(text: str) -> str:
    if BEGIN not in text:
        return text.rstrip() + "\n"
    before, rest = text.split(BEGIN, 1)
    if END not in rest:
        raise RuntimeError("Found Stage3B1 begin marker without end marker")
    _, after = rest.split(END, 1)
    return (before.rstrip() + "\n" + after.lstrip()).rstrip() + "\n"


def _install() -> None:
    if not VLLM_ROOT.is_dir():
        raise FileNotFoundError(
            f"Expected vLLM fork at {VLLM_ROOT}. Run this file from kvquant_pkg."
        )
    if not RABIT_PATH.is_file() or not TRITON_ATTN_PATH.is_file():
        raise FileNotFoundError("Missing Stage-3A rabit_kv2.py or triton_attn.py")

    rabit = RABIT_PATH.read_text(encoding="utf-8")
    required_rabit = [
        "encode_rabit2_page_ref",
        "decode_rabit2_page_ref",
        "Rabit2OnlineStateRef",
        "rabit2_page_layout",
    ]
    missing = [x for x in required_rabit if x not in rabit]
    if missing:
        raise RuntimeError(f"Stage3B1 preflight: rabit_kv2.py missing {missing}")

    attn = TRITON_ATTN_PATH.read_text(encoding="utf-8")
    required_attn = [
        'self._is_rabit_kv2',
        'rabit_kv2 Stage 2 has exact physical page allocation',
        'rabit_kv2 Stage 2 allocator is active',
    ]
    missing_attn = [x for x in required_attn if x not in attn]
    if missing_attn:
        raise RuntimeError(
            "Stage3B1 preflight: triton_attn.py does not match the guarded Stage-3A version: "
            + repr(missing_attn)
        )

    _backup(RABIT_PATH)
    _backup(TRITON_ATTN_PATH)
    _backup(TEST_PATH)

    rabit = _strip_old_fragment(rabit)
    rabit = rabit.rstrip() + "\n" + _decode(FRAGMENT_Z).lstrip()
    compile(rabit, str(RABIT_PATH), "exec")
    RABIT_PATH.write_text(rabit, encoding="utf-8")

    test_text = _decode(TEST_Z)
    compile(test_text, str(TEST_PATH), "exec")
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.write_text(test_text, encoding="utf-8")

    print(f"Installed Stage3B1 kernels: {RABIT_PATH}")
    print(f"Installed Stage3B1 tests:   {TEST_PATH}")
    print("triton_attn.py engine guard: RETAINED (correct until Stage 3B2 sidecar integration)")


def _preflight() -> None:
    rabit = RABIT_PATH.read_text(encoding="utf-8")
    required = [
        BEGIN,
        "encode_rabit2_page_exact_cuda",
        "encode_rabit2_page_triton_experimental",
        "decode_rabit2_page_triton",
        "rabit2_closed_page_attention_triton",
        "_rabit2_closed_page_attention_kernel",
    ]
    missing = [x for x in required if x not in rabit]
    if missing:
        raise RuntimeError(f"Stage3B1 source preflight missing: {missing}")
    compile(rabit, str(RABIT_PATH), "exec")
    compile(TEST_PATH.read_text(encoding="utf-8"), str(TEST_PATH), "exec")
    print("Stage 3B1 local source preflight: PASSED")


def _include(path: Path) -> bool:
    rel = path.relative_to(VLLM_ROOT)
    return (
        path.is_file()
        and not any(part in IGNORED_DIR_NAMES for part in rel.parts)
        and path.suffix.lower() not in IGNORED_SUFFIXES
        and not path.name.endswith(".egg-info")
    )


def _make_snapshot() -> None:
    tmp = SNAPSHOT_PATH.with_suffix(".zip.tmp")
    for path in (tmp, SNAPSHOT_PATH):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    count = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(VLLM_ROOT.rglob("*")):
            if _include(path):
                zf.write(path, path.relative_to(VLLM_ROOT).as_posix())
                count += 1
    os.replace(tmp, SNAPSHOT_PATH)
    print(
        f"Created frozen vLLM snapshot: {SNAPSHOT_PATH} "
        f"({count} files, {SNAPSHOT_PATH.stat().st_size / 1024**2:.1f} MB)"
    )


def _run_modal() -> int:
    RUNNER_PATH.write_text(_decode(RUNNER_Z), encoding="utf-8")
    time.sleep(2)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    command = [sys.executable, "-m", "modal", "run", str(RUNNER_PATH)]
    print("Running:", " ".join(command))
    with LOG_PATH.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def main() -> None:
    print("RABIT-2 Stage 3B1 FINAL installer + H100 runner (exact writer + Triton fused read)")
    print(f"Project root: {PROJECT_ROOT}")
    _install()
    _preflight()
    _make_snapshot()
    code = _run_modal()
    if code != 0:
        raise SystemExit(code)
    print(f"Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
