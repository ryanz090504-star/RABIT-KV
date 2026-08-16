"""Apply and run RABIT-2 Stage 3B1 Triton closed-page kernels.

Run from:
  C:\\Users\\ryanz\\Documents\\GitHub\\benchquant\\kvquant_full\\kvquant_pkg

What this stage DOES:
- Adds Triton K3/V2 closed-page quantize+pack kernels.
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
LOG_PATH = PROJECT_ROOT / "rabit2_stage3b1_triton_fix2.log"

BEGIN = "# === RABIT2_STAGE3B1_TRITON_BEGIN ==="
END = "# === RABIT2_STAGE3B1_TRITON_END ==="

FRAGMENT_Z = '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'
TEST_Z = 'eNrtV1tv2zYUfvev4NwXaZMVW+lSz4CGZSs6DOtDi10wIAgIWqJiwpIok7SQC/Lfe3iRZDlS0mxtniYgscjD85374dEr9MeHt//M3rOElpLOfktpqVjGqFih84okGzqLwvmEFRUXChVEbSbNorpRVKpmpbhIgJYJXqA6z4uwXoREKY3Gy5BXMhRkzRTe1hFyLN4EwZPShKcUG2qEK3IF7zQLxmhKMMVLS6blOOsA7ZDV7Sc5lzS15Fbb9qDfN2db40R7BLNSUZGRhDaWHErJyQ3fq8lkktIMuZXnr6xUqvaiHDhuXaGfdc6TLZbslsanUdBul/sCXIc3lKQyXnb7esOcxtt4EQ0S6o7gN2pJUlQ59VJaQ9zjabJPydTpaAIZFqTckxxLSlMvmkdn8+X8jW/oWxQjzx4SpExLD9REywBpIcgB2h8ffYvm4Rs/VNwxrLOcE7U4s0j1M5F+GEVyft0GqAYLf7KZCTaIbSi3rGKZV3KXoqG2NWQSk5qwnKzBC34AAETyMp7+8tfbc1js9kzQdOobZ2ksLBUE63S9cLmBbWYKDBUBGSF13lEBOUdNVJt42+CCmW0eWBcGxvYmCnZT88HmcE57miVweLH9cVgCKAIYh+vI0y9HjOBcdVPR2PrDOPI0cmhArNUwmrX8mYB2S7uQlVchkZIKZavO07K09kLxPF7Q2fcBIs3rk8y1VnSU+WVywIZKYABagzRMr0miMBAkzWmioK/0q/852QCh+xfJAIoNc7nYjTKyDHXeAdNJ7gGL9knmlDftmGVaKy3lm1jTWkrBpKkEoEJz9PTBUO4Lzw+ZovDjdyfJNRRProg7quV0ZQ07UNNopsGPd/2QrCUgAsIArqBX0LoloN61e/qZbsEDN5CS6XSFPGtz2O5hnmWSqsYnBwQdUekHfax6AKsew6ofx9rigpU9nWD9UB+9OcIvE5LTHoLZeYhht0cs6mtRD2lRj2tRP9CiHtaiHtHi/iiCOOH7Uj2MY0kKumqT6wLQV/CHvkP6jrscSTb9ZFwY5gB5wBCY8z4ANRljmKTnD2iUMSF1vdosLHl5SwU38qHbSKz2UK/xO5JL6kPXM/OD51+sovkl5G7OpDpAdf0og4bj9fWb/mmK014AGpwKiczg0V4qq7bArP/iu2Z9H6DpEZyuMNOPTJnB0abi7oPG5viu52wgGFNdzIBslvcd8su2VDeQCVAuBUL18Jb9shfsU93x/zv25e/YbK8H8+H5/Cunw+PzVgqU9EvkwuPT9tKK27XtpxuRD+djO7r3JfRnY9N2AWURztGJ+YIL5U4oD1DsAdAQ28HhMz6Kut61C9CgrZJnSrccIzc2/+23h7Xcc4a/Qu9zUpDZabiYLX9Gv348B92qCpJwhU4j9NF8wEjQeIl+/9stYvS6JVRUNITQBqbGLL1u3UXAXzCIa3/1v3LQyQl6bTm0c7fpxSpwzAFaXdovE02phygy4YLKVgplpb55pps0UJt09uNGQSx2jaXAvdOfLp0TKsHXHbdzlWdBQVFWxLNFFxU7BB5JUk5SCpIMHOTj7sladkEOGtyuqqOuqqOvXtUlx+tscaYrmApGcnZLTHI1TBhua1f6AJq+YF3/90KzGn5ODQ2XjkWxUdNgodyQCkyJIaHM6zHdqGLp5vWQ7gZnmbES5hsd/iYpYYzOc/DVJ35f0zY='
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
        "encode_rabit2_page_triton",
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
    print("RABIT-2 Stage 3B1 direct-Python installer + H100 runner (precise-div fix2)")
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
