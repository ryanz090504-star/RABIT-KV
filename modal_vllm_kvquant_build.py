r"""
Build and verify the custom vLLM KVQuant fork on Modal.

Place this file in kvquant_pkg beside:
    vllm-kvquant/
    Quantization_v2/

Run:
    & "C:\Users\ryanz\Documents\GitHub\benchquant\turboquant\venv\Scripts\python.exe" -m modal run .\modal_vllm_kvquant_build.py
"""

import modal

app = modal.App("vllm-kvquant-build-check")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "git",
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
    )
    .run_commands(
        (
            "python -m pip install "
            "torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 "
            "--index-url https://download.pytorch.org/whl/cu130"
        ),
    )
    .add_local_dir(
        "vllm-kvquant",
        "/root/vllm-kvquant",
        copy=True,
        ignore=[
            ".git/**",
            "build/**",
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
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "MAX_JOBS": "8",
            # The archive has no .git metadata, so setuptools-scm needs
            # an explicit version while building the editable package.
            "SETUPTOOLS_SCM_PRETEND_VERSION": "0.10.0+kvquant",
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM": "0.10.0+kvquant",
        }
    )
    .run_commands(
        (
            "cd /root/vllm-kvquant && "
            "python use_existing_torch.py --prefix"
        ),
        (
            "cd /root/vllm-kvquant && "
            "python -m pip install -e . --no-build-isolation"
        ),
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
)
def verify_build() -> None:
    import subprocess
    import sys

    import torch
    import vllm

    print("=" * 76)
    print("Custom vLLM KVQuant Build Check")
    print("=" * 76)
    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA ready:   {torch.cuda.is_available()}")
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"vLLM:         {getattr(vllm, '__version__', 'unknown')}")
    print(f"vLLM path:    {vllm.__file__}")
    print()

    print("Running custom kvquant_k3 tests...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "/root/vllm-kvquant/tests/quantization/test_kvquant_k3.py",
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
            "The custom vLLM fork built, but kvquant_k3 tests failed."
        )

    print("CUSTOM VLLM BUILD CHECK PASSED")


@app.local_entrypoint()
def main() -> None:
    verify_build.remote()
