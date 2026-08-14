"""
Controlled TTFT/TPOT benchmark for Llama 3.1 8B:

- BF16 KV cache + Triton attention
- INT3 kvquant_k3 KV cache + Triton attention

TTFT is approximated by a one-output-token request.
TPOT is estimated from:
    (latency for 65 output tokens - latency for 1 output token) / 64

Place this file in kvquant_pkg beside vllm-kvquant.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import modal

BASE_COMMIT = "f329ce405b12623fb8b1cf1830f12e5a712523be"

app = modal.App("vllm-kvquant-ttft-tpot")
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
parser.add_argument("--mode", choices=["bf16_triton", "int3_triton"], required=True)
args = parser.parse_args()

cache_dtype = "auto" if args.mode == "bf16_triton" else "kvquant_k3"
label = "BF16 + Triton" if args.mode == "bf16_triton" else "INT3 + Triton"

os.environ["VLLM_USE_V1"] = "1"

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs

MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
CONTEXT_LENGTHS = [512, 2048, 4096]
REPEATS = 3

engine_fields = getattr(EngineArgs, "__dataclass_fields__", {})
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

if "attention_backend" in engine_fields:
    llm_kwargs["attention_backend"] = "TRITON_ATTN"
    backend_strategy = "attention_backend argument"
elif "attention_config" in engine_fields:
    from vllm.config import AttentionConfig
    llm_kwargs["attention_config"] = AttentionConfig(backend="TRITON_ATTN")
    backend_strategy = "attention_config argument"
else:
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"
    backend_strategy = "legacy environment variable"

print("=" * 90, flush=True)
print(f"MODE: {label}", flush=True)
print(f"Backend strategy: {backend_strategy}", flush=True)
print("=" * 90, flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
base_text = (
    "Artificial intelligence systems process long sequences of information. "
    "The attention mechanism stores keys and values from earlier tokens so "
    "future decoding steps can reuse them efficiently. "
)

def make_prompt(target_tokens: int):
    repeated = base_text * ((target_tokens // 30) + 12)
    ids = tokenizer.encode(repeated, add_special_tokens=False)[:target_tokens]
    prompt = tokenizer.decode(ids, skip_special_tokens=True)
    actual = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, actual

init_start = time.perf_counter()
llm = LLM(**llm_kwargs)
init_seconds = time.perf_counter() - init_start

engine = getattr(llm, "llm_engine", None)
cache_config = getattr(engine, "cache_config", None)
if cache_config is None and engine is not None:
    vllm_config = getattr(engine, "vllm_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)

block_size = getattr(cache_config, "block_size", None)
num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
capacity_tokens = (
    block_size * num_gpu_blocks
    if isinstance(block_size, int) and isinstance(num_gpu_blocks, int)
    else None
)

results = []
for target_context in CONTEXT_LENGTHS:
    prompt, actual_context = make_prompt(target_context)

    # Compile and warm both short and long decode paths before measurement.
    llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, min_tokens=4, max_tokens=4),
        use_tqdm=False,
    )
    llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, min_tokens=16, max_tokens=16),
        use_tqdm=False,
    )

    ttft_trials = []
    total65_trials = []
    tpot_trials = []

    for trial in range(REPEATS):
        one_params = SamplingParams(
            temperature=0.0,
            min_tokens=1,
            max_tokens=1,
        )
        sixty_five_params = SamplingParams(
            temperature=0.0,
            min_tokens=65,
            max_tokens=65,
        )

        torch.cuda.synchronize()
        start = time.perf_counter()
        one_output = llm.generate([prompt], one_params, use_tqdm=False)
        torch.cuda.synchronize()
        one_ms = (time.perf_counter() - start) * 1000.0

        torch.cuda.synchronize()
        start = time.perf_counter()
        long_output = llm.generate([prompt], sixty_five_params, use_tqdm=False)
        torch.cuda.synchronize()
        total65_ms = (time.perf_counter() - start) * 1000.0

        one_count = len(one_output[0].outputs[0].token_ids)
        long_count = len(long_output[0].outputs[0].token_ids)
        if one_count != 1 or long_count != 65:
            raise RuntimeError(
                f"Unexpected token counts: one={one_count}, long={long_count}"
            )

        tpot_ms = (total65_ms - one_ms) / 64.0
        ttft_trials.append(one_ms)
        total65_trials.append(total65_ms)
        tpot_trials.append(tpot_ms)

        print(
            f"TRIAL mode={args.mode} context={actual_context} "
            f"trial={trial + 1} ttft_ms={one_ms:.3f} "
            f"total65_ms={total65_ms:.3f} tpot_ms={tpot_ms:.3f}",
            flush=True,
        )

    results.append(
        {
            "mode": args.mode,
            "label": label,
            "actual_context_tokens": actual_context,
            "median_ttft_ms": statistics.median(ttft_trials),
            "median_tpot_ms": statistics.median(tpot_trials),
            "ttft_trials_ms": ttft_trials,
            "total65_trials_ms": total65_trials,
            "tpot_trials_ms": tpot_trials,
        }
    )

payload = {
    "mode": args.mode,
    "label": label,
    "cache_dtype": cache_dtype,
    "init_seconds": init_seconds,
    "capacity_tokens": capacity_tokens,
    "results": results,
}
print("BENCHMARK_JSON=" + json.dumps(payload), flush=True)

del llm
gc.collect()
torch.cuda.empty_cache()
"""


def run_worker(mode: str) -> tuple[dict, str]:
    worker = Path(f"/tmp/ttft_tpot_{mode}.py")
    worker.write_text(WORKER_CODE, encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(worker), "--mode", mode],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, flush=True)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Worker {mode} failed with exit code {completed.returncode}."
        )

    match = re.search(r"^BENCHMARK_JSON=(.+)$", completed.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Worker {mode} emitted no BENCHMARK_JSON.")

    return json.loads(match.group(1)), completed.stdout


def confirmed_triton(log: str) -> bool:
    return bool(
        re.search(r"Using AttentionBackendEnum\.TRITON_ATTN backend", log)
        or re.search(r"Using TRITON_ATTN attention backend", log)
    )


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_benchmark() -> None:
    bf16, bf16_log = run_worker("bf16_triton")
    int3, int3_log = run_worker("int3_triton")

    if not confirmed_triton(bf16_log):
        raise RuntimeError("BF16 worker did not confirm the Triton backend.")
    if not confirmed_triton(int3_log):
        raise RuntimeError("INT3 worker did not confirm the Triton backend.")

    print("\n" + "=" * 110)
    print("CONTROLLED TTFT/TPOT — Llama 3.1 8B, H100, Triton backend")
    print("=" * 110)
    print(
        f"{'Context':>8} | {'BF16 TTFT ms':>13} | {'INT3 TTFT ms':>13} | "
        f"{'BF16 TPOT ms':>13} | {'INT3 TPOT ms':>13} | "
        f"{'INT3/BF16 TPOT':>16}"
    )
    print("-" * 110)

    for b, q in zip(bf16["results"], int3["results"]):
        ratio = q["median_tpot_ms"] / b["median_tpot_ms"]
        print(
            f"{b['actual_context_tokens']:>8} | "
            f"{b['median_ttft_ms']:>13.3f} | "
            f"{q['median_ttft_ms']:>13.3f} | "
            f"{b['median_tpot_ms']:>13.3f} | "
            f"{q['median_tpot_ms']:>13.3f} | "
            f"{ratio:>15.3f}x"
        )

    print("-" * 110)
    print(f"BF16 KV capacity: {bf16['capacity_tokens']} tokens")
    print(f"INT3 KV capacity: {int3['capacity_tokens']} tokens")
    print(
        f"Capacity improvement: "
        f"{int3['capacity_tokens'] / bf16['capacity_tokens']:.3f}x"
    )

    print("\nTTFT_TPOT_JSON=" + json.dumps({"bf16": bf16, "int3": int3}))
    print("\nCONTROLLED TTFT/TPOT BENCHMARK COMPLETED")


@app.local_entrypoint()
def main() -> None:
    run_benchmark.remote()
