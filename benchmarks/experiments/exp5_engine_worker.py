"""
MLSys 2027 Experiment 5 -- single-(dtype, context) real-engine worker (runs
INSIDE the Modal container, one fresh process per cell of the context sweep).

Derived from the frozen Experiment 3/4 workers (NOT modified).
run_experiment5_context_scaling.py proves by AST before any run that this
worker shares with the Experiment 3 worker:
  * BASE_ENGINE_KWARGS and OUTPUT_TOKENS (identical values);
  * the tokenizer / BOS / filler / SamplingParams statements (identical) and
    the prompt statement (identical except CONTEXT_TOKENS -> args.prompt_tokens);
  * the timed region of one(): t0 / llm.generate / wall_ms / metrics
    (identical statements) and every Experiment 3 sample field;
  * no engine RPC (the Experiment 3 attempt-1 hang cause).
Differences (all outside the timed region):
  * --context-point (grid label) and --prompt-tokens (exact prompt length);
    only bfloat16 and rabit_kv2 are accepted;
  * the effective config and KV dtype records of the Experiment 4 worker;
  * per request, SHA-256 of the prompt token IDs the engine actually saw and
    of the generated token IDs, computed after wall time is taken;
  * GPU memory (nvidia-smi memory.used, device-wide) after engine init and
    after the measured reps. No live-KV probe: live paged-KV usage is DERIVED
    by the runner from the engine-reported capacity (see the runner);
  * --conditioning: an UNMEASURED conditioning cell (5 full-shape warmups,
    zero measured reps) run before the official sweep;
  * an exception raised by a request AFTER successful engine initialization
    (warmup or measured llm.generate) is reported as a workload-level failure
    (EXP5_WORKLOAD_FAILURE) with exit code WORKLOAD_FAILURE_EXIT. Only kinds
    request_oom / request_execution_failure are workload-level; a request the
    engine rejects (validation) or a programming error is reported as
    request_rejected_or_programming_error, which the sweep treats as a
    methodology failure. An engine initialization failure is never caught
    here and propagates as an ordinary crash.
Machine-readable lines use the EXP5_ prefix.

This file never modifies vllm-kvquant; it only imports it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time

# Identical to exp3_engine_worker.BASE_ENGINE_KWARGS (verified by AST).
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
OUTPUT_TOKENS = 32
WORKLOAD_FAILURE_EXIT = 75


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


def gpu_memory_used_mib() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         text=True, capture_output=True, check=True).stdout
    return [int(x.strip()) for x in out.strip().splitlines()]


def is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return "OutOfMemoryError" in text or "out of memory" in text.lower()


def failure_kind(exc: BaseException) -> str:
    """request_oom / request_execution_failure are workload-level failures; a
    validation or programming error (e.g. a prompt the engine rejects) is a
    methodology failure and must stop the sweep."""
    if is_oom(exc):
        return "request_oom"
    if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError, AssertionError)):
        return "request_rejected_or_programming_error"
    return "request_execution_failure"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-cache-dtype", required=True, choices=ALLOWED_KV_CACHE_DTYPES)
    ap.add_argument("--context-point", type=int, required=True, help="context grid point (label)")
    ap.add_argument("--prompt-tokens", type=int, required=True, help="exact prompt length in tokens")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--warmups", type=int, required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--leg", required=True, help="cell label, e.g. A512 / B32768 / conditioning_A512")
    ap.add_argument("--conditioning", action="store_true", help="unmeasured conditioning cell (reps must be 0)")
    args = ap.parse_args()
    if args.warmups < 1 or (args.reps != 0 if args.conditioning else args.reps < 10):
        raise SystemExit("warmups must be >= 1; reps must be 0 for a conditioning cell and >= 10 otherwise")
    if not 1 <= args.prompt_tokens <= args.context_point:
        raise SystemExit("prompt tokens must be in [1, context point]")
    role = "conditioning" if args.conditioning else "measured"
    emit("EXP5_LEG", {"leg": args.leg, "kv_cache_dtype": args.kv_cache_dtype, "role": role,
                      "context_point": args.context_point, "prompt_tokens": args.prompt_tokens})

    from vllm import LLM, SamplingParams
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode
    import vllm.v1.attention.ops.rabit_kv2 as r

    # Same frozen-source dispatch assertions as the canonical real_engine().
    markers = {
        "_RABIT2_STAGE4D3_4_TRITON_TAILPREP": bool(getattr(r, "_RABIT2_STAGE4D3_4_TRITON_TAILPREP", False)),
        "_RABIT2_FINAL_FAST_DECODE_APPEND": bool(getattr(r, "_RABIT2_FINAL_FAST_DECODE_APPEND", False)),
        "stage4d3_4_dispatch_active": r._rabit2_stage4b1_exactmeta_emit_tail_partial
        is r._rabit2_stage4d3_4_emit_tail_partial,
    }
    emit("EXP5_RABIT_MARKERS", markers)
    if not all(markers.values()):
        raise RuntimeError(f"RABIT-KV frozen-source markers missing: {markers}")

    kwargs = {"model": args.model_dir, **BASE_ENGINE_KWARGS, "kv_cache_dtype": args.kv_cache_dtype}
    emit("EXP5_REQUESTED_ENGINE_KWARGS", kwargs)

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
        "compilation_mode": enum_name(comp.mode, CompilationMode),
        "compilation_mode_raw": {"repr": repr(comp.mode), "str": str(comp.mode)},
        "cudagraph_mode": enum_name(comp.cudagraph_mode, CUDAGraphMode),
        "cudagraph_mode_raw": {"repr": repr(comp.cudagraph_mode), "str": str(comp.cudagraph_mode)},
        "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size,
        "log_stats": bool(getattr(llm.llm_engine, "log_stats", False)),
        "calculate_kv_scales": cc.calculate_kv_scales,
        "kv_cache_dtype_skip_layers": list(cc.kv_cache_dtype_skip_layers),
        "hf_quantization_config": getattr(mc.hf_config, "quantization_config", None),
    }
    emit("EXP5_EFFECTIVE_ENGINE_CONFIG", effective)

    mode = get_kv_quant_mode(cc.cache_dtype)
    emit(
        "EXP5_KV_DTYPE",
        {
            "requested_kv_cache_dtype": args.kv_cache_dtype,
            "engine_cache_dtype": cc.cache_dtype,
            "resolved_kv_torch_dtype": str(get_kv_cache_torch_dtype(cc.cache_dtype, mc.dtype)),
            "kv_quant_mode": mode.name,
            "fp8_storage_view_dtype": str(current_platform.fp8_dtype())
            if mode == KVQuantMode.FP8_PER_TENSOR
            else None,
        },
    )

    capacity = cc.num_gpu_blocks * cc.block_size
    emit(
        "EXP5_CAPACITY",
        {"num_gpu_blocks": cc.num_gpu_blocks, "block_size": cc.block_size, "capacity_tokens": capacity},
    )
    emit("EXP5_GPU_MEMORY", {"phase": "after_engine_init", "memory_used_mib": gpu_memory_used_mib()})

    tok = llm.get_tokenizer()
    bos = tok.bos_token_id
    filler = tok.encode(" the", add_special_tokens=False)[-1]
    prompt = [{"prompt_token_ids": [bos] + [filler] * (args.prompt_tokens - 1)}]
    sp = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True)
    emit(
        "EXP5_WORKLOAD",
        {
            "role": role,
            "context_point": args.context_point,
            "prompt_tokens": args.prompt_tokens,
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
            "prompt_token_ids_sha256": hashlib.sha256(
                json.dumps(list(out.prompt_token_ids)).encode("utf-8")
            ).hexdigest(),
            "output_token_ids_sha256": hashlib.sha256(
                json.dumps(list(out.outputs[0].token_ids)).encode("utf-8")
            ).hexdigest(),
        }

    phase, rep = "warmup", None
    try:
        print("EXP5_WARMUP_BEGIN", flush=True)
        for i in range(args.warmups):
            rep = i
            print("EXP5_WARMUP " + json.dumps(one(i), sort_keys=True), flush=True)
        print("EXP5_WARMUP_END", flush=True)

        phase, rep = "measurement", None
        print("EXP5_MEASUREMENT_BEGIN", flush=True)
        for i in range(args.reps):
            rep = i
            print("EXP5_SAMPLE " + json.dumps(one(i), sort_keys=True), flush=True)
        print("EXP5_MEASUREMENT_END", flush=True)
    except Exception as exc:  # noqa: BLE001  (request-level failure after a successful engine init)
        emit("EXP5_WORKLOAD_FAILURE", {
            "leg": args.leg, "phase": phase, "rep": rep,
            "kind": failure_kind(exc),
            "error": f"{type(exc).__name__}: {exc}"[:2000],
        })
        return WORKLOAD_FAILURE_EXIT

    emit("EXP5_GPU_MEMORY", {"phase": "after_measurement", "memory_used_mib": gpu_memory_used_mib()})
    print("EXP5_WORKER_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
