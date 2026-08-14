"""RABIT-KV Stage 2 exact physical allocator/layout verification.

Place this file beside ``vllm-kvquant`` after applying the Stage-2 overlay.
This uses the precompiled vLLM extension path and tests Python allocator/layout
changes without rebuilding all CUDA targets.
"""

from __future__ import annotations

import modal

BASE_COMMIT = "f329ce405b12623fb8b1cf1830f12e5a712523be"

app = modal.App("rabit2-stage2-allocator-test")

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
    .add_local_dir(
        "vllm-kvquant",
        "/root/vllm-kvquant",
        copy=True,
        ignore=[
            ".git/**",
            "build/**",
            "dist/**",
            ".venv/**",
            "__pycache__/**",
            "*.pyc",
            "docs/**",
            "examples/**",
            "benchmarks/**",
            ".github/**",
            ".buildkite/**",
        ],
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
def verify_stage2_allocator() -> None:
    import subprocess
    import sys

    import torch
    import vllm

    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
    from vllm.v1.kv_cache_interface import (
        rabit2_page_layout,
        rabit2_residual_bytes_per_sequence,
    )

    print("=" * 86)
    print("RABIT-2 Stage 2 exact allocator/layout check")
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
    bf16_page = 2 * 32 * 8 * 128 * 2

    print("\nLlama-3.1-8B physical layout:")
    print(f"  K3 payload/page:        {layout.k_payload_bytes:8d} bytes")
    print(f"  V2 payload/page:        {layout.v_payload_bytes:8d} bytes")
    print(f"  compressed metadata:    {layout.metadata_bytes:8d} bytes")
    print(f"  quantized page total:   {layout.page_bytes:8d} bytes")
    print(f"  BF16 page total:        {bf16_page:8d} bytes")
    print(f"  page-only compression:  {bf16_page / layout.page_bytes:8.3f}x")
    print(f"  separate R4/sequence:   {residual:8d} bytes")
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
        raise RuntimeError("RABIT-2 Stage 2 allocator/layout tests failed")

    print("RABIT-2 STAGE 2 ALLOCATOR/LAYOUT CHECK PASSED")
    print("Stage 3 fused write/read kernels are intentionally not claimed here.")


@app.local_entrypoint()
def main() -> None:
    verify_stage2_allocator.remote()
