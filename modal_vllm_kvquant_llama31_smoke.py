"""
End-to-end vLLM engine smoke test for the custom kvquant_k3 KV-cache dtype.

Place this file in kvquant_pkg beside:
- vllm-kvquant
- Quantization_v2

The test loads TinyLlama through the real vLLM V1 engine, creates a
kvquant_k3 cache, performs prefill and decode, and verifies token generation.
"""

import modal

BASE_COMMIT = "f329ce405b12623fb8b1cf1830f12e5a712523be"

app = modal.App("vllm-kvquant-llama31-smoke-v2")
hf_cache = modal.Volume.from_name(
    "vllm-kvquant-hf-cache",
    create_if_missing=True,
)

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
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_engine_smoke_test() -> None:
    import gc
    import os
    import time

    # The custom kvquant_k3 integration lives in the vLLM V1 engine.
    os.environ["VLLM_USE_V1"] = "1"

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"

    print("=" * 78)
    print("Llama 3.1 8B kvquant_k3 end-to-end vLLM engine smoke test")
    print("=" * 78)
    print(f"Model:          {model_name}")
    print(f"vLLM:           {getattr(vllm, '__version__', 'unknown')}")
    print(f"vLLM source:    {vllm.__file__}")
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA runtime:   {torch.version.cuda}")
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print("KV-cache dtype: kvquant_k3")
    print()

    print("1/2 Initializing the real vLLM V1 engine...", flush=True)
    init_start = time.perf_counter()

    llm = LLM(
        model=model_name,
        dtype="float16",
        kv_cache_dtype="kvquant_k3",
        max_model_len=4096,
        gpu_memory_utilization=0.70,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=False,
        disable_log_stats=True,
    )

    init_seconds = time.perf_counter() - init_start
    print(f"Engine initialized in {init_seconds:.2f} seconds.")

    engine = getattr(llm, "llm_engine", None)
    cache_config = getattr(engine, "cache_config", None)
    resolved_cache_dtype = getattr(cache_config, "cache_dtype", None)
    if resolved_cache_dtype is not None:
        print(f"Resolved engine cache dtype: {resolved_cache_dtype}")

    print()
    print("2/2 Running prefill and autoregressive decode...", flush=True)

    messages = [
        {
            "role": "user",
            "content": (
                "In one short sentence, explain why compressing an LLM "
                "KV cache is useful."
            ),
        }
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        min_tokens=8,
        max_tokens=32,
    )

    generation_start = time.perf_counter()
    outputs = llm.chat(
        messages,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    generation_seconds = time.perf_counter() - generation_start

    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("vLLM returned an unexpected output structure.")

    completion = outputs[0].outputs[0]
    generated_text = completion.text.strip()
    generated_tokens = len(completion.token_ids)

    print()
    print(f"User message:     {messages[0]['content']}")
    print(f"Generated text:   {generated_text!r}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Generation time:  {generation_seconds:.3f} seconds")

    if generated_tokens < 8:
        raise RuntimeError(
            f"Expected at least 8 decode tokens, but received {generated_tokens}."
        )
    if not generated_text:
        raise RuntimeError(
            "The engine decoded multiple tokens, but the visible output was empty."
        )

    print()
    print("LLAMA 3.1 8B KVQUANT_K3 ENGINE SMOKE TEST PASSED")
    print(
        "The real vLLM engine initialized, wrote the quantized KV cache, "
        "read it during attention, and completed autoregressive decoding."
    )

    del llm
    gc.collect()
    torch.cuda.empty_cache()


@app.local_entrypoint()
def main() -> None:
    run_engine_smoke_test.remote()
