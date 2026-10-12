"""
MLSys 2027 Experiment 3 -- single-dtype real-engine worker (runs INSIDE the
Modal container, one fresh process per ABBA leg).

Mirrors the canonical real_engine() measurement embedded in
benchmarks/performance/benchmark_deployment.py (RUNNER_Z): same LLM(...)
engine arguments, same 2048-token prompt construction, same TTFT/TPOT/wall
definitions. The ONLY parameter is --kv-cache-dtype. Protocol differences
from the canonical run (fixed for every dtype):
  * warmup uses the exact measured request shape (2048 prompt, 32 output
    tokens) instead of 2 x 8-token requests, and is reported separately;
  * --reps measured repetitions per ABBA leg (15; 2 legs per dtype = 30)
    instead of 5;
  * the engine configuration actually in effect, the resolved KV dtype and
    the allocator capacity are printed as machine-readable lines.

The resolved KV dtype is established WITHOUT any engine RPC: requested
kv_cache_dtype, the engine config's cache_dtype, and local resolution via
get_kv_cache_torch_dtype (the runner adds a physical bytes/token cross-check).
(Attempt 1 hung on a collective_rpc("get_kv_cache_spec") whose torch.dtype
payload the engine could not serialize; that RPC has been removed.)

Machine-readable output lines (parsed by run_experiment3_deployment.py):
  EXP3_LEG=<json>
  EXP3_REQUESTED_ENGINE_KWARGS=<json>
  EXP3_RABIT_MARKERS=<json>
  EXP3_EFFECTIVE_ENGINE_CONFIG=<json>
  EXP3_KV_DTYPE=<json>
  EXP3_CAPACITY=<json>
  EXP3_WORKLOAD=<json>
  EXP3_WARMUP_BEGIN / EXP3_WARMUP {...} / EXP3_WARMUP_END
  EXP3_MEASUREMENT_BEGIN / EXP3_SAMPLE {...} / EXP3_MEASUREMENT_END
  EXP3_WORKER_COMPLETE

This file never modifies vllm-kvquant; it only imports it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

# Identical to the canonical LLM(...) call in benchmark_deployment.py's
# embedded runner, except that kv_cache_dtype is a parameter and model is the
# resolved snapshot path. run_experiment3_deployment.py verifies this dict
# against the canonical source by AST before any run.
BASE_ENGINE_KWARGS = {
    "dtype": "bfloat16",
    "block_size": 32,
    "max_model_len": 32768,
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 32,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": True,
    "gpu_memory_utilization": 0.82,
    "enforce_eager": True,
    "trust_remote_code": True,
    "disable_log_stats": False,
    "attention_config": {"backend": "TRITON_ATTN"},
}
ALLOWED_KV_CACHE_DTYPES = ("bfloat16", "rabit_kv2")
CONTEXT_TOKENS = 2048
OUTPUT_TOKENS = 32


def emit(tag: str, payload) -> None:
    print(f"{tag}={json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def enum_name(value, enum_cls) -> str:
    """Member name of `value` in `enum_cls`, independent of Enum.__str__."""
    if isinstance(value, enum_cls):
        return value.name
    try:
        return enum_cls(value).name
    except (ValueError, TypeError):
        return f"UNRECOGNIZED:{value!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-cache-dtype", required=True, choices=ALLOWED_KV_CACHE_DTYPES)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--warmups", type=int, required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--leg", required=True, help="ABBA leg label, e.g. A1/B1/B2/A2")
    args = ap.parse_args()
    if args.warmups < 1 or args.reps < 15:
        raise SystemExit("warmups must be >= 1 and reps >= 15 per leg")
    emit("EXP3_LEG", {"leg": args.leg, "kv_cache_dtype": args.kv_cache_dtype})

    from vllm import LLM, SamplingParams
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    import vllm.v1.attention.ops.rabit_kv2 as r

    # Same frozen-source dispatch assertions as the canonical real_engine().
    markers = {
        "_RABIT2_STAGE4D3_4_TRITON_TAILPREP": bool(getattr(r, "_RABIT2_STAGE4D3_4_TRITON_TAILPREP", False)),
        "_RABIT2_FINAL_FAST_DECODE_APPEND": bool(getattr(r, "_RABIT2_FINAL_FAST_DECODE_APPEND", False)),
        "stage4d3_4_dispatch_active": r._rabit2_stage4b1_exactmeta_emit_tail_partial
        is r._rabit2_stage4d3_4_emit_tail_partial,
    }
    emit("EXP3_RABIT_MARKERS", markers)
    if not all(markers.values()):
        raise RuntimeError(f"RABIT-KV frozen-source markers missing: {markers}")

    kwargs = {"model": args.model_dir, **BASE_ENGINE_KWARGS, "kv_cache_dtype": args.kv_cache_dtype}
    emit("EXP3_REQUESTED_ENGINE_KWARGS", kwargs)

    llm = LLM(**kwargs)

    cfg = llm.llm_engine.vllm_config
    mc, cc, sc = cfg.model_config, cfg.cache_config, cfg.scheduler_config
    comp = cfg.compilation_config
    effective = {
        "model": mc.model,
        "model_dtype": str(mc.dtype),
        "max_model_len": mc.max_model_len,
        "enforce_eager": mc.enforce_eager,
        "seed": mc.seed,
        "trust_remote_code": mc.trust_remote_code,
        "quantization": mc.quantization,
        "block_size": cc.block_size,
        "gpu_memory_utilization": cc.gpu_memory_utilization,
        "enable_prefix_caching": cc.enable_prefix_caching,
        "max_num_batched_tokens": sc.max_num_batched_tokens,
        "max_num_seqs": sc.max_num_seqs,
        "enable_chunked_prefill": sc.enable_chunked_prefill,
        "attention_backend": str(cfg.attention_config.backend),
        # Raw representations plus names normalized via the enum classes
        # themselves (never via Enum.__str__, which differs between IntEnum
        # and Enum: CompilationMode.NONE prints as "0").
        "compilation_mode": enum_name(comp.mode, CompilationMode),
        "compilation_mode_raw": {"repr": repr(comp.mode), "str": str(comp.mode)},
        "cudagraph_mode": enum_name(comp.cudagraph_mode, CUDAGraphMode),
        "cudagraph_mode_raw": {"repr": repr(comp.cudagraph_mode), "str": str(comp.cudagraph_mode)},
        "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size,
        "log_stats": bool(getattr(llm.llm_engine, "log_stats", False)),
    }
    emit("EXP3_EFFECTIVE_ENGINE_CONFIG", effective)

    emit(
        "EXP3_KV_DTYPE",
        {
            "requested_kv_cache_dtype": args.kv_cache_dtype,
            "engine_cache_dtype": cc.cache_dtype,
            "resolved_kv_torch_dtype": str(get_kv_cache_torch_dtype(cc.cache_dtype, mc.dtype)),
        },
    )

    capacity = cc.num_gpu_blocks * cc.block_size
    emit(
        "EXP3_CAPACITY",
        {"num_gpu_blocks": cc.num_gpu_blocks, "block_size": cc.block_size, "capacity_tokens": capacity},
    )

    tok = llm.get_tokenizer()
    bos = tok.bos_token_id
    filler = tok.encode(" the", add_special_tokens=False)[-1]
    prompt = [{"prompt_token_ids": [bos] + [filler] * (CONTEXT_TOKENS - 1)}]
    sp = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True)
    emit(
        "EXP3_WORKLOAD",
        {
            "context_tokens": CONTEXT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "warmups": args.warmups,
            "reps": args.reps,
            "temperature": sp.temperature,
            "ignore_eos": sp.ignore_eos,
            "max_tokens": sp.max_tokens,
            "prompt_token_ids_sha256": hashlib.sha256(
                json.dumps(prompt[0]["prompt_token_ids"]).encode("utf-8")
            ).hexdigest(),
        },
    )

    def one(rep: int) -> dict:
        t0 = time.perf_counter()
        out = llm.generate(prompt, sp, use_tqdm=False)[0]
        wall_ms = (time.perf_counter() - t0) * 1000.0
        m = out.metrics
        n = len(out.outputs[0].token_ids)
        return {
            "rep": rep,
            "prompt_tokens": len(out.prompt_token_ids),
            "output_tokens": n,
            "ttft_ms": m.first_token_latency * 1000.0,
            "tpot_ms": (m.last_token_ts - m.first_token_ts) / (n - 1) * 1000.0,
            "wall_ms": wall_ms,
        }

    print("EXP3_WARMUP_BEGIN", flush=True)
    for i in range(args.warmups):
        print("EXP3_WARMUP " + json.dumps(one(i), sort_keys=True), flush=True)
    print("EXP3_WARMUP_END", flush=True)

    print("EXP3_MEASUREMENT_BEGIN", flush=True)
    for i in range(args.reps):
        print("EXP3_SAMPLE " + json.dumps(one(i), sort_keys=True), flush=True)
    print("EXP3_MEASUREMENT_END", flush=True)

    print("EXP3_WORKER_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
