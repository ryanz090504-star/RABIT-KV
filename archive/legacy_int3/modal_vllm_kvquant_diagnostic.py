r"""
Diagnostic build for the custom vLLM KVQuant fork on Modal.

Place this file in kvquant_pkg beside:
    vllm-kvquant/
    Quantization_v2/

Run:
    & "C:\Users\ryanz\Documents\GitHub\benchquant\turboquant\venv\Scripts\python.exe" -m modal run .\modal_vllm_kvquant_diagnostic.py

This version:
- limits parallel compilation to reduce RAM pressure;
- builds a wheel instead of an editable install;
- suppresses the enormous live compiler log;
- prints only the final 250 log lines if compilation fails.
"""

import modal

app = modal.App("vllm-kvquant-diagnostic-build")

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
            "MAX_JOBS": "2",
            "CMAKE_BUILD_PARALLEL_LEVEL": "2",
            "NVCC_THREADS": "1",
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
            "rm -rf build dist *.egg-info /tmp/vllm-wheelhouse "
            "/tmp/vllm_build.log && "
            "mkdir -p /tmp/vllm-wheelhouse && "
            "(python -m pip wheel . "
            "--no-build-isolation --no-deps "
            "-w /tmp/vllm-wheelhouse "
            "> /tmp/vllm_build.log 2>&1 "
            "|| (echo '===== FINAL 250 BUILD LOG LINES ====='; "
            "tail -n 250 /tmp/vllm_build.log; exit 1)) && "
            "echo '===== BUILD SUCCEEDED =====' && "
            "ls -lh /tmp/vllm-wheelhouse"
        ),
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=600,
)
def confirm_image() -> None:
    import subprocess

    print("Diagnostic wheel build completed.")
    subprocess.run(
        ["bash", "-lc", "ls -lh /tmp/vllm-wheelhouse"],
        check=True,
    )


@app.local_entrypoint()
def main() -> None:
    confirm_image.remote()
