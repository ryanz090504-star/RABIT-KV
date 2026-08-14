"""RABIT-KV Stage 2.1 allocator/staging verification using a stable ZIP snapshot.

Why this variant exists:
Modal rejects a local directory mount when any file in that directory changes
while the image is being assembled. This script first creates one immutable ZIP
snapshot of ``vllm-kvquant`` in the local temp directory, then uploads that single
archive. VS Code, antivirus, or file-indexing activity can therefore no longer
change the files Modal is hashing during the build.
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import modal

BASE_COMMIT = "f329ce405b12623fb8b1cf1830f12e5a712523be"
APP_NAME = "rabit2-stage2-1-staging-snapshot-test"

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "vllm-kvquant"
SNAPSHOT_PATH = Path(tempfile.gettempdir()) / "rabit2_stage2_1_vllm_snapshot.zip"

# Keep the snapshot small and deterministic enough for Modal upload.
IGNORED_DIR_NAMES = {
    ".git",
    ".github",
    ".buildkite",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "docs",
    "examples",
    "benchmarks",
    ".deps",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".so", ".dll", ".dylib"}


def _should_include(path: Path) -> bool:
    relative = path.relative_to(SOURCE_DIR)
    if any(part in IGNORED_DIR_NAMES for part in relative.parts):
        return False
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return False
    if path.name.endswith(".egg-info"):
        return False
    return path.is_file()


def _create_stable_snapshot() -> Path:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected vLLM fork beside this script: {SOURCE_DIR}"
        )

    temporary_path = SNAPSHOT_PATH.with_suffix(".zip.tmp")
    for candidate in (temporary_path, SNAPSHOT_PATH):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass

    file_count = 0
    with zipfile.ZipFile(
        temporary_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(SOURCE_DIR.rglob("*")):
            if not _should_include(path):
                continue
            archive.write(path, arcname=path.relative_to(SOURCE_DIR).as_posix())
            file_count += 1

    os.replace(temporary_path, SNAPSHOT_PATH)
    size_mb = SNAPSHOT_PATH.stat().st_size / (1024**2)
    print(
        f"Created stable vLLM source snapshot: {SNAPSHOT_PATH} "
        f"({file_count} files, {size_mb:.1f} MB)"
    )
    return SNAPSHOT_PATH


STABLE_SNAPSHOT = _create_stable_snapshot()

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "curl", "libnuma1", "libnuma-dev")
    .pip_install(
        "pip>=25",
        "cmake>=3.26.1",
        "ninja",
        "packaging>=24.2",
        "setuptools>=77.0.3,<81.0.0",
        "setuptools-scm>=8.0",
        "setuptools-rust>=1.9.0",
        "wheel",
        "jinja2",
        "pytest>=7",
        "tblib>=3.0",
    )
    .run_commands(
        "python -m pip install "
        "torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 "
        "--index-url https://download.pytorch.org/whl/cu130"
    )
    .add_local_file(
        str(STABLE_SNAPSHOT),
        "/tmp/rabit2_stage2_1_vllm_snapshot.zip",
        copy=True,
    )
    .run_commands(
        "rm -rf /root/vllm-kvquant && mkdir -p /root/vllm-kvquant && "
        "python -m zipfile -e /tmp/rabit2_stage2_1_vllm_snapshot.zip /root/vllm-kvquant"
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "VLLM_TARGET_DEVICE": "cuda",
            "VLLM_USE_PRECOMPILED": "1",
            "VLLM_PRECOMPILED_WHEEL_COMMIT": BASE_COMMIT,
            "VLLM_PRECOMPILED_WHEEL_VARIANT": "cu130",
            "VLLM_MAIN_CUDA_VERSION": "13.0",
            "VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX": "1",
            "SETUPTOOLS_SCM_PRETEND_VERSION": "0.10.0+kvquant",
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM": "0.10.0+kvquant",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    .run_commands(
        "cd /root/vllm-kvquant && python use_existing_torch.py --prefix",
        (
            "cd /root/vllm-kvquant && "
            "rm -rf build dist *.egg-info /tmp/vllm_precompiled_install.log && "
            "(python -m pip install -e . --no-build-isolation "
            "> /tmp/vllm_precompiled_install.log 2>&1 "
            "|| (echo '===== INSTALL FAILED: FINAL 250 LINES ====='; "
            "tail -n 250 /tmp/vllm_precompiled_install.log; exit 1)) && "
            "echo '===== PRECOMPILED INSTALL COMPLETED =====' && "
            "tail -n 30 /tmp/vllm_precompiled_install.log"
        ),
    )
)


@app.function(image=image, gpu="H100", timeout=1800)
def verify_stage2_1_allocator() -> None:
    import subprocess
    import sys

    import torch
    import vllm

    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
    from vllm.v1.kv_cache_interface import (
        rabit2_effective_compression_with_staging,
        rabit2_exact_online_staging_bytes_per_sequence,
        rabit2_page_layout,
        rabit2_residual_bytes_per_sequence,
    )

    print("=" * 86)
    print("RABIT-2 Stage 2.1 exact allocator + online staging check")
    print("=" * 86)
    print(f"Base wheel commit: {BASE_COMMIT}")
    print(f"Python:            {sys.version.split()[0]}")
    print(f"PyTorch:           {torch.__version__}")
    print(f"CUDA runtime:      {torch.version.cuda}")
    print(f"GPU:               {torch.cuda.get_device_name(0)}")
    print(f"vLLM version:      {getattr(vllm, '__version__', 'unknown')}")

    layout = rabit2_page_layout(
        block_size=32,
        num_kv_heads=8,
        head_size_k=128,
        head_size_v=128,
    )
    shape = TritonAttentionBackend.get_kv_cache_shape(
        num_blocks=1,
        block_size=32,
        num_kv_heads=8,
        head_size=128,
        cache_dtype_str="rabit_kv2",
    )
    residual = rabit2_residual_bytes_per_sequence(
        num_kv_heads=8,
        head_size_k=128,
        head_size_v=128,
    )
    staging = rabit2_exact_online_staging_bytes_per_sequence(
        num_kv_heads=8,
        head_size_k=128,
        head_size_v=128,
    )
    bf16_page = 2 * 32 * 8 * 128 * 2

    print("\nLlama-3.1-8B physical layout:")
    print(f"  K3 payload/page:        {layout.k_payload_bytes:8d} bytes")
    print(f"  V2 payload/page:        {layout.v_payload_bytes:8d} bytes")
    print(f"  compressed metadata:    {layout.metadata_bytes:8d} bytes")
    print(f"  quantized page total:   {layout.page_bytes:8d} bytes")
    print(f"  BF16 page total:        {bf16_page:8d} bytes")
    print(f"  page-only compression:  {bf16_page / layout.page_bytes:8.3f}x")
    print(f"  R4-only lower bound:    {residual:8d} bytes")
    print(f"  exact online staging:   {staging:8d} bytes")
    print(f"  effective comp @1K:     {rabit2_effective_compression_with_staging(context_tokens=1024, block_size=32, num_kv_heads=8, head_size_k=128, head_size_v=128):8.3f}x")
    print(f"  effective comp @4K:     {rabit2_effective_compression_with_staging(context_tokens=4096, block_size=32, num_kv_heads=8, head_size_k=128, head_size_v=128):8.3f}x")
    print(f"  effective comp @16K:    {rabit2_effective_compression_with_staging(context_tokens=16384, block_size=32, num_kv_heads=8, head_size_k=128, head_size_v=128):8.3f}x")
    print(f"  backend raw shape:      {shape}")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "/root/vllm-kvquant/tests/quantization/test_kvquant_k3.py",
            "/root/vllm-kvquant/tests/quantization/test_rabit_kv2_layout.py",
            "--confcutdir=/root/vllm-kvquant/tests/quantization",
            "-q",
            "--tb=short",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    print("\n" + result.stdout)
    if result.stderr:
        print(result.stderr)
    print(f"pytest exit code: {result.returncode}")
    if result.returncode != 0:
        raise RuntimeError("RABIT-2 Stage 2.1 allocator/staging tests failed")

    print("RABIT-2 STAGE 2.1 ALLOCATOR/STAGING CHECK PASSED")
    print("Stage 3 will implement the open-K-group staging and physical write/read path.")


@app.local_entrypoint()
def main() -> None:
    verify_stage2_1_allocator.remote()
