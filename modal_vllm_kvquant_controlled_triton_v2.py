"""
Controlled same-backend Llama 3.1 8B attention benchmark:

1. BF16 KV cache + explicitly forced Triton attention
2. INT3 kvquant_k3 KV cache + Triton attention

Place this file in kvquant_pkg beside:
- vllm-kvquant
- Quantization_v2
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import modal

BASE_COMMIT = "f329ce405b12623fb8b1cf1830f12e5a712523be"

app = modal.App("vllm-kvquant-controlled-triton-v2")
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

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
args = parser.parse_args()

MODES = {
    "bf16_triton": {
        "label": "BF16 + Triton",
        "cache_dtype": "auto",
        "attention_backend": "TRITON_ATTN",
    },
    "int3_triton": {
        "label": "INT3 + Triton",
        "cache_dtype": "kvquant_k3",
        "attention_backend": "TRITON_ATTN",
    },
}

if args.mode not in MODES:
    raise ValueError(f"Unknown mode: {args.mode}")

config = MODES[args.mode]
label = config["label"]
cache_dtype = config["cache_dtype"]
attention_backend = config["attention_backend"]

os.environ["VLLM_USE_V1"] = "1"

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs

engine_fields = getattr(EngineArgs, "__dataclass_fields__", {})
backend_strategy = None

# The user's custom fork is newer internally than its spoofed 0.10.0
# package version. Prefer the actual EngineArgs API when present.
if "attention_backend" in engine_fields:
    backend_strategy = "attention_backend argument"
elif "attention_config" in engine_fields:
    backend_strategy = "attention_config argument"
else:
    # Legacy vLLM V1 enum name.
    backend_strategy = "legacy environment variable"
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"

print(f"Backend control strategy: {backend_strategy}", flush=True)

MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
CONTEXT_LENGTHS = [512, 2048, 4096]
MAX_NEW_TOKENS = 64
REPEATS = 3

print("=" * 96, flush=True)
print(f"CONTROLLED MODE: {label}", flush=True)
print(f"kv_cache_dtype={cache_dtype}", flush=True)
print(f"VLLM_ATTENTION_BACKEND={attention_backend or 'AUTO'}", flush=True)
print("=" * 96, flush=True)

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
llm_kwargs = {
    "model": MODEL,
    "dtype": "bfloat16",
    "kv_cache_dtype": cache_dtype,
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.70,
    "tensor_parallel_size": 1,
    "enforce_eager": True,
    "trust_remote_code": False,
    "disable_log_stats": True,
    "enable_prefix_caching": False,
}

if backend_strategy == "attention_backend argument":
    llm_kwargs["attention_backend"] = attention_backend
elif backend_strategy == "attention_config argument":
    from vllm.config import AttentionConfig
    llm_kwargs["attention_config"] = AttentionConfig(
        backend=attention_backend,
    )

llm = LLM(**llm_kwargs)
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
    f"ENGINE_METADATA mode={args.mode} init_seconds={init_seconds:.3f} "
    f"block_size={block_size} num_gpu_blocks={num_gpu_blocks} "
    f"capacity_tokens={capacity_tokens}",
    flush=True,
)

results = []
for target_context in CONTEXT_LENGTHS:
    prompt, actual_context = make_prompt(target_context)

    # First warm-up compiles the backend/kernel for this shape.
    warmup_params = SamplingParams(
        temperature=0.0,
        min_tokens=16,
        max_tokens=16,
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
            f"TRIAL mode={args.mode} context={actual_context} "
            f"trial={trial + 1} latency_ms={elapsed_ms:.3f} "
            f"generated_tokens={generated}",
            flush=True,
        )

    median_ms = statistics.median(trial_ms)
    output_tokens = min(generated_counts)

    results.append(
        {
            "mode": args.mode,
            "label": label,
            "cache_dtype": cache_dtype,
            "attention_backend": attention_backend or "AUTO",
            "target_context_tokens": target_context,
            "actual_context_tokens": actual_context,
            "output_tokens": output_tokens,
            "median_e2e_ms": median_ms,
            "median_ms_per_output_token": median_ms / output_tokens,
            "trial_ms": trial_ms,
        }
    )

payload = {
    "mode": args.mode,
    "label": label,
    "cache_dtype": cache_dtype,
    "attention_backend": attention_backend or "AUTO",
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


def run_worker(mode: str) -> tuple[dict, str]:
    worker_path = Path(f"/tmp/controlled_{mode}.py")
    worker_path.write_text(WORKER_CODE, encoding="utf-8")

    process = subprocess.run(
        [sys.executable, str(worker_path), "--mode", mode],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    output = process.stdout
    print(output, flush=True)

    if process.returncode != 0:
        raise RuntimeError(
            f"Controlled worker {mode} failed with exit code "
            f"{process.returncode}."
        )

    match = re.search(r"^BENCHMARK_JSON=(.+)$", output, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Worker {mode} did not emit BENCHMARK_JSON.")

    return json.loads(match.group(1)), output


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_controlled_benchmark() -> None:
    modes = ["bf16_triton", "int3_triton"]
    data = {}
    logs = {}

    for mode in modes:
        print(f"\nRunning {mode}...", flush=True)
        data[mode], logs[mode] = run_worker(mode)

    backend_patterns = {
        mode: re.findall(
            r"Using ([A-Z0-9_]+) attention backend",
            logs[mode],
        )
        for mode in modes
    }

    detected = {
        mode: re.findall(
            r"Using ([A-Z0-9_]+) attention backend",
            logs[mode],
        )
        for mode in modes
    }

    # Refuse to publish a misleading table if BF16 was not actually Triton.
    for mode in modes:
        if "TRITON_ATTN" not in detected[mode]:
            raise RuntimeError(
                f"{mode} did not use TRITON_ATTN. "
                f"Detected backend entries: {detected[mode]}"
            )

    print("\n" + "=" * 104)
    print("CONTROLLED SAME-BACKEND BENCHMARK — Llama 3.1 8B on H100")
    print("=" * 104)
    print(
        f"{'Context':>8} | {'BF16 Triton ms/tok':>20} | "
        f"{'INT3 Triton ms/tok':>20} | "
        f"{'INT3 latency / BF16':>21} | {'INT3 speed vs BF16':>19}"
    )
    print("-" * 104)

    for bf16_row, int3_row in zip(
        data["bf16_triton"]["results"],
        data["int3_triton"]["results"],
    ):
        bf16 = bf16_row["median_ms_per_output_token"]
        int3 = int3_row["median_ms_per_output_token"]

        print(
            f"{bf16_row['actual_context_tokens']:>8} | "
            f"{bf16:>20.3f} | "
            f"{int3:>20.3f} | "
            f"{int3 / bf16:>20.3f}x | "
            f"{bf16 / int3:>18.3f}x"
        )

    print("-" * 104)

    bf16_capacity = data["bf16_triton"].get("capacity_tokens")
    int3_capacity = data["int3_triton"].get("capacity_tokens")

    print(f"BF16 Triton KV capacity: {bf16_capacity} tokens")
    print(f"INT3 Triton KV capacity: {int3_capacity} tokens")
    if bf16_capacity and int3_capacity:
        print(
            f"INT3 capacity improvement over BF16: "
            f"{int3_capacity / bf16_capacity:.3f}x"
        )

    print("\nVerified backend log entries:")
    for mode in modes:
        print(f"  {mode}: {detected[mode]}")

    payload = {
        "results": data,
        "detected_backends": detected,
    }
    print("\nCONTROLLED_BENCHMARK_JSON=" + json.dumps(payload))
    print("\nCONTROLLED SAME-BACKEND TRITON BENCHMARK COMPLETED")


@app.local_entrypoint()
def main() -> None:
    run_controlled_benchmark.remote()
