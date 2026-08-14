"""
Modal smoke check for the Quantization_v2 framework.

Run this file from the kvquant_pkg directory:
    python -m modal run modal_kvquant_env_check.py

This step installs only Quantization_v2. It does not build the vLLM fork yet.
"""

import modal

app = modal.App("kvquant-env-check")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.3",
        "transformers>=4.43",
        "numpy>=1.24",
        "pytest>=7",
    )
    .add_local_dir(
        "Quantization_v2",
        "/root/Quantization_v2",
        copy=True,
    )
    .run_commands(
        "python -m pip install -e /root/Quantization_v2",
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
)
def check_environment() -> None:
    import platform
    import subprocess
    import sys

    import torch
    import transformers
    import kvquant
    from kvquant.policies import list_policies

    print("=" * 72)
    print("KVQuant Modal Environment Check")
    print("=" * 72)
    print(f"Python:       {sys.version.split()[0]}")
    print(f"Platform:     {platform.platform()}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"CUDA:         {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the Modal container.")

    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"kvquant:      {kvquant.__file__}")
    print("Policies:")
    for policy in list_policies():
        print(f"  - {policy}")

    print()
    print("Running lightweight unit tests...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "/root/Quantization_v2/tests",
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
            "The Quantization_v2 environment loaded, but one or more tests failed."
        )

    print("ENVIRONMENT CHECK PASSED")


@app.local_entrypoint()
def main() -> None:
    check_environment.remote()
