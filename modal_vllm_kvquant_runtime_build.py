"""
Runtime diagnostic build for the custom vLLM KVQuant fork on Modal.

Place this file in the kvquant_pkg folder beside:
- vllm-kvquant
- Quantization_v2
"""

import os
import subprocess
import time
from pathlib import Path

import modal

app = modal.App("vllm-kvquant-runtime-build")

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
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "MAX_JOBS": "2",
            "CMAKE_BUILD_PARALLEL_LEVEL": "2",
            "NVCC_THREADS": "1",
            "SETUPTOOLS_SCM_PRETEND_VERSION": "0.10.0+kvquant",
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM": "0.10.0+kvquant",
        }
    )
)


def tail_lines(path: Path, count: int = 8) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-count:]
    except OSError:
        return []


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=10800,
)
def build_with_heartbeat() -> None:
    source = Path("/root/vllm-kvquant")
    log_path = Path("/tmp/vllm_build.log")
    wheel_dir = Path("/tmp/vllm-wheelhouse")

    print("=" * 78, flush=True)
    print("vLLM KVQuant runtime diagnostic build", flush=True)
    print("=" * 78, flush=True)

    subprocess.run(
        ["python", "use_existing_torch.py", "--prefix"],
        cwd=source,
        check=True,
    )

    subprocess.run(
        [
            "bash",
            "-lc",
            "rm -rf build dist *.egg-info /tmp/vllm-wheelhouse "
            "/tmp/vllm_build.log && mkdir -p /tmp/vllm-wheelhouse",
        ],
        cwd=source,
        check=True,
    )

    command = [
        "python",
        "-m",
        "pip",
        "wheel",
        ".",
        "--no-build-isolation",
        "--no-deps",
        "-w",
        str(wheel_dir),
    ]

    print("Starting wheel compilation...", flush=True)
    print("Command:", " ".join(command), flush=True)

    env = os.environ.copy()
    started = time.monotonic()

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=source,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        while process.poll() is None:
            time.sleep(30)
            elapsed = int(time.monotonic() - started)
            print(f"[heartbeat] build still running: {elapsed}s", flush=True)
            for line in tail_lines(log_path, 5):
                print(f"  {line}", flush=True)

    elapsed = int(time.monotonic() - started)
    print(f"Build process exited with code {process.returncode} after {elapsed}s.", flush=True)

    if process.returncode != 0:
        print("", flush=True)
        print("=" * 78, flush=True)
        print("FINAL 300 BUILD LOG LINES", flush=True)
        print("=" * 78, flush=True)
        for line in tail_lines(log_path, 300):
            print(line, flush=True)
        raise RuntimeError("vLLM wheel compilation failed.")

    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError("Compilation returned success, but no wheel was produced.")

    print("", flush=True)
    print("BUILD SUCCEEDED", flush=True)
    for wheel in wheels:
        size_mb = wheel.stat().st_size / (1024 * 1024)
        print(f"{wheel.name} — {size_mb:.1f} MB", flush=True)


@app.local_entrypoint()
def main() -> None:
    build_with_heartbeat.remote()
