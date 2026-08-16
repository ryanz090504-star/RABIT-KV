"""
Paired Llama 3.1 8B benchmark:
BF16 KV cache baseline vs custom kvquant_k3 KV cache.

Place this file in kvquant_pkg beside:
- vllm-kvquant
- Quantization_v2

The benchmark runs both cache modes in isolated subprocesses on the same H100,
with identical model, context lengths, output length, and eager-mode settings.
"""

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import modal

BASE_COMMIT = "f329ce405b12623fb8b1cf1830f12e5a712523be"

app = modal.App("vllm-kvquant-paired-benchmark")
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


WORKER_CODE = r"""
import argparse
import gc
import json
import os
import statistics
import time

os.environ["VLLM_USE_V1"] = "1"

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
CONTEXT_LENGTHS = [512, 2048, 4096]
MAX_NEW_TOKENS = 64
REPEATS = 2

parser = argparse.ArgumentParser()
parser.add_argument("--cache-dtype", required=True)
args = parser.parse_args()

cache_dtype = args.cache_dtype
label = "BF16" if cache_dtype == "auto" else "kvquant_k3"

print("=" * 88, flush=True)
print(f"BENCHMARK MODE: {label} (kv_cache_dtype={cache_dtype})", flush=True)
print("=" * 88, flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

base_text = (
    "Artificial intelligence systems process long sequences of information. "
    "The attention mechanism uses previously computed keys and values to avoid "
    "recomputing the entire context during every decoding step. "
)

def make_prompt(target_tokens: int):
    repeated = base_text * ((target_tokens // 35) + 10)
    ids = tokenizer.encode(repeated, add_special_tokens=False)[:target_tokens]
    prompt = tokenizer.decode(ids, skip_special_tokens=True)
    actual = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, actual

init_start = time.perf_counter()
llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    kv_cache_dtype=cache_dtype,
    max_model_len=8192,
    gpu_memory_utilization=0.70,
    tensor_parallel_size=1,
    enforce_eager=True,
    trust_remote_code=False,
    disable_log_stats=True,
    enable_prefix_caching=False,
)
init_seconds = time.perf_counter() - init_start

engine = getattr(llm, "llm_engine", None)
cache_config = getattr(engine, "cache_config", None)
if cache_config is None and engine is not None:
    vllm_config = getattr(engine, "vllm_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)

block_size = getattr(cache_config, "block_size", None)
num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
capacity_tokens = None
if isinstance(block_size, int) and isinstance(num_gpu_blocks, int):
    capacity_tokens = block_size * num_gpu_blocks

print(
    f"ENGINE_METADATA mode={label} init_seconds={init_seconds:.3f} "
    f"block_size={block_size} num_gpu_blocks={num_gpu_blocks} "
    f"capacity_tokens={capacity_tokens}",
    flush=True,
)

results = []
for target_context in CONTEXT_LENGTHS:
    prompt, actual_context = make_prompt(target_context)

    # Warm this exact prompt shape and the custom Triton kernel before timing.
    warmup_params = SamplingParams(
        temperature=0.0,
        min_tokens=8,
        max_tokens=8,
    )
    llm.generate([prompt], warmup_params, use_tqdm=False)

    sampling_params = SamplingParams(
        temperature=0.0,
        min_tokens=MAX_NEW_TOKENS,
        max_tokens=MAX_NEW_TOKENS,
    )

    trial_ms = []
    generated_counts = []
    for trial in range(REPEATS):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = llm.generate([prompt], sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        generated = len(output[0].outputs[0].token_ids)
        trial_ms.append(elapsed_ms)
        generated_counts.append(generated)

        print(
            f"TRIAL mode={label} context={actual_context} "
            f"trial={trial + 1} latency_ms={elapsed_ms:.3f} "
            f"generated_tokens={generated}",
            flush=True,
        )

    median_ms = statistics.median(trial_ms)
    output_tokens = min(generated_counts)
    ms_per_output_token = median_ms / output_tokens

    row = {
        "mode": label,
        "cache_dtype": cache_dtype,
        "target_context_tokens": target_context,
        "actual_context_tokens": actual_context,
        "output_tokens": output_tokens,
        "median_e2e_ms": median_ms,
        "median_ms_per_output_token": ms_per_output_token,
        "trial_ms": trial_ms,
    }
    results.append(row)

payload = {
    "mode": label,
    "cache_dtype": cache_dtype,
    "model": MODEL,
    "dtype": "bfloat16",
    "init_seconds": init_seconds,
    "block_size": block_size,
    "num_gpu_blocks": num_gpu_blocks,
    "capacity_tokens": capacity_tokens,
    "results": results,
}

print("BENCHMARK_JSON=" + json.dumps(payload), flush=True)

del llm
gc.collect()
torch.cuda.empty_cache()
"""


def run_worker(cache_dtype: str) -> tuple[dict, str]:
    worker_path = Path(f"/tmp/kvquant_benchmark_{cache_dtype}.py")
    worker_path.write_text(WORKER_CODE, encoding="utf-8")

    process = subprocess.run(
        [
            sys.executable,
            str(worker_path),
            "--cache-dtype",
            cache_dtype,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    output = process.stdout
    print(output, flush=True)

    if process.returncode != 0:
        raise RuntimeError(
            f"Benchmark worker for {cache_dtype} failed with "
            f"exit code {process.returncode}."
        )

    match = re.search(r"^BENCHMARK_JSON=(.+)$", output, re.MULTILINE)
    if not match:
        raise RuntimeError(
            f"Benchmark worker for {cache_dtype} did not emit BENCHMARK_JSON."
        )

    return json.loads(match.group(1)), output


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_paired_benchmark() -> None:
    print("\nRunning BF16 KV-cache baseline...", flush=True)
    baseline, baseline_log = run_worker("auto")

    print("\nRunning custom kvquant_k3 KV cache...", flush=True)
    quantized, quantized_log = run_worker("kvquant_k3")

    baseline_capacity_match = re.search(
        r"GPU KV cache size:\s*([0-9,]+) tokens",
        baseline_log,
    )
    quant_capacity_match = re.search(
        r"GPU KV cache size:\s*([0-9,]+) tokens",
        quantized_log,
    )

    baseline_capacity = (
        int(baseline_capacity_match.group(1).replace(",", ""))
        if baseline_capacity_match
        else baseline.get("capacity_tokens")
    )
    quantized_capacity = (
        int(quant_capacity_match.group(1).replace(",", ""))
        if quant_capacity_match
        else quantized.get("capacity_tokens")
    )

    print("\n" + "=" * 110)
    print("PAIRED BENCHMARK SUMMARY — Llama 3.1 8B on H100")
    print("=" * 110)
    print(
        f"{'Context':>10} | {'BF16 E2E ms':>13} | "
        f"{'INT3 E2E ms':>13} | {'BF16 ms/out':>13} | "
        f"{'INT3 ms/out':>13} | {'INT3 speed vs BF16':>20}"
    )
    print("-" * 110)

    for base_row, quant_row in zip(
        baseline["results"],
        quantized["results"],
    ):
        speed_ratio = (
            base_row["median_ms_per_output_token"]
            / quant_row["median_ms_per_output_token"]
        )
        print(
            f"{base_row['actual_context_tokens']:>10} | "
            f"{base_row['median_e2e_ms']:>13.2f} | "
            f"{quant_row['median_e2e_ms']:>13.2f} | "
            f"{base_row['median_ms_per_output_token']:>13.3f} | "
            f"{quant_row['median_ms_per_output_token']:>13.3f} | "
            f"{speed_ratio:>19.3f}x"
        )

    print("-" * 110)
    print(f"BF16 engine initialization:       {baseline['init_seconds']:.2f} s")
    print(f"kvquant_k3 engine initialization: {quantized['init_seconds']:.2f} s")
    print(f"BF16 GPU KV-cache capacity:       {baseline_capacity} tokens")
    print(f"kvquant_k3 GPU KV-cache capacity: {quantized_capacity} tokens")

    if baseline_capacity and quantized_capacity:
        capacity_ratio = quantized_capacity / baseline_capacity
        print(f"KV-cache capacity improvement:    {capacity_ratio:.3f}x")

    final_payload = {
        "baseline": baseline,
        "kvquant_k3": quantized,
        "baseline_capacity_tokens": baseline_capacity,
        "kvquant_k3_capacity_tokens": quantized_capacity,
    }
    print("\nPAIRED_BENCHMARK_JSON=" + json.dumps(final_payload))
    print("\nPAIRED BF16 VS KVQUANT_K3 BENCHMARK COMPLETED")


@app.local_entrypoint()
def main() -> None:
    run_paired_benchmark.remote()
