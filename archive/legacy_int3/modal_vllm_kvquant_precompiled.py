"""
Install the custom vLLM KVQuant fork using the official precompiled-extension path.

Place this file in kvquant_pkg beside:
- vllm-kvquant
- Quantization_v2
"""

import modal

BASE_COMMIT = "f329ce405b12623fb8b1cf1830f12e5a712523be"

app = modal.App("vllm-kvquant-precompiled-check-v3")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "curl",
        "libnuma1",
        "libnuma-dev",
    )
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
        (
            "cd /root/vllm-kvquant && "
            "python use_existing_torch.py --prefix"
        ),
        (
            "cd /root/vllm-kvquant && "
            "rm -rf build dist *.egg-info /tmp/vllm_precompiled_install.log && "
            "(python -m pip install -e . --no-build-isolation "
            "> /tmp/vllm_precompiled_install.log 2>&1 "
            "|| (echo '===== INSTALL FAILED: FINAL 250 LINES ====='; "
            "tail -n 250 /tmp/vllm_precompiled_install.log; exit 1)) && "
            "echo '===== PRECOMPILED INSTALL COMPLETED =====' && "
            "tail -n 50 /tmp/vllm_precompiled_install.log"
        ),
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
)
def verify_precompiled_install() -> None:
    import subprocess
    import sys
    from pathlib import Path

    import torch
    import vllm

    print("=" * 78)
    print("Custom vLLM KVQuant precompiled installation check")
    print("=" * 78)
    print(f"Base wheel commit: {BASE_COMMIT}")
    print(f"Python:            {sys.version.split()[0]}")
    print(f"PyTorch:           {torch.__version__}")
    print(f"CUDA runtime:      {torch.version.cuda}")
    print(f"CUDA available:    {torch.cuda.is_available()}")
    print(f"GPU:               {torch.cuda.get_device_name(0)}")
    print(f"vLLM version:      {getattr(vllm, '__version__', 'unknown')}")
    print(f"vLLM source path:  {vllm.__file__}")

    source_file = Path(
        "/root/vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py"
    )
    if not source_file.exists():
        raise RuntimeError(f"Custom KVQuant source file missing: {source_file}")

    from vllm.v1.attention.ops import kvquant_k3  # noqa: F401

    print("Custom kvquant_k3 import: PASSED")
    print()
    print("Running custom kvquant_k3 tests...")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "/root/vllm-kvquant/tests/quantization/test_kvquant_k3.py",
            "--confcutdir=/root/vllm-kvquant/tests/quantization",
            "-q",
            "--tb=short",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    print(f"pytest exit code: {result.returncode}")
    if result.returncode != 0:
        raise RuntimeError(
            "Precompiled vLLM installed, but custom kvquant_k3 tests failed."
        )

    print("PRECOMPILED VLLM KVQUANT CHECK PASSED")


@app.local_entrypoint()
def main() -> None:
    verify_precompiled_install.remote()
