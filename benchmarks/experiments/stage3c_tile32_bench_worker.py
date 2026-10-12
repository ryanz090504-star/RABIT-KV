"""
Stage3C tile32 before/after BENCHMARK worker (runs INSIDE the Modal container;
one fresh engine process per series). Benchmark evidence only -- NOT
Experiment 5 evidence.

Derived from stage3c_diag_worker.py (NOT modified). The only additions:
  * --stage3c-impl {reference, tile32}: exported as VLLM_RABIT2_STAGE3C_IMPL
    BEFORE the engine starts (the EngineCore process inherits it) and echoed in
    S3C_STAGE_IMPL together with the selector value the frozen module reports;
  * the tile32 module file hash is recorded (provenance).
The engine kwargs, prompt construction, SamplingParams and the timed region of
one() are proven identical to the Experiment 3 worker by the runner (AST).

Request guard: 600 s per-request SIGALRM guard, with the process-group watchdog
and the Modal function timeout as hard process-level backstops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
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
REQUEST_TIMEOUT_EXIT = 76
REQUEST_FAILURE_EXIT = 77


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


class RequestCapExceeded(Exception):
    pass


def _on_alarm(signum, frame):
    raise RequestCapExceeded()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-cache-dtype", required=True, choices=ALLOWED_KV_CACHE_DTYPES)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--points", required=True, help="comma-separated measured prompt lengths, ascending")
    ap.add_argument("--conditioning-prompt", type=int, required=True)
    ap.add_argument("--request-cap-s", type=int, required=True)
    ap.add_argument("--series", required=True)
    ap.add_argument("--stage3c-impl", required=True, choices=("reference", "tile32"))
    args = ap.parse_args()
    os.environ["VLLM_RABIT2_STAGE3C_IMPL"] = args.stage3c_impl
    points = [int(x) for x in args.points.split(",")]
    if points != sorted(points) or len(set(points)) != len(points):
        raise SystemExit("points must be strictly ascending")
    emit("S3C_SERIES", {"series": args.series, "kv_cache_dtype": args.kv_cache_dtype, "points": points,
                        "conditioning_prompt_tokens": args.conditioning_prompt, "request_cap_s": args.request_cap_s})

    from vllm import LLM, SamplingParams
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    from vllm.v1.kv_cache_interface import get_kv_quant_mode
    import vllm.v1.attention.ops.rabit_kv2 as r
    import vllm.v1.attention.ops.rabit_kv2_stage3c_tile32 as t32

    markers = {
        "_RABIT2_STAGE4D3_4_TRITON_TAILPREP": bool(getattr(r, "_RABIT2_STAGE4D3_4_TRITON_TAILPREP", False)),
        "_RABIT2_FINAL_FAST_DECODE_APPEND": bool(getattr(r, "_RABIT2_FINAL_FAST_DECODE_APPEND", False)),
        "stage4d3_4_dispatch_active": r._rabit2_stage4b1_exactmeta_emit_tail_partial
        is r._rabit2_stage4d3_4_emit_tail_partial,
    }
    emit("S3C_RABIT_MARKERS", markers)
    if not all(markers.values()):
        raise RuntimeError(f"RABIT-KV frozen-source markers missing: {markers}")

    emit("S3C_STAGE_IMPL", {"requested": args.stage3c_impl, "selector_reports": t32.rabit2_stage3c_impl(),
                              "tile32_module_sha256": hashlib.sha256(open(t32.__file__, "rb").read()).hexdigest()})
    if t32.rabit2_stage3c_impl() != args.stage3c_impl:
        raise RuntimeError("Stage3C selector does not report the requested implementation")
    kwargs = {"model": args.model_dir, **BASE_ENGINE_KWARGS, "kv_cache_dtype": args.kv_cache_dtype}
    emit("S3C_REQUESTED_ENGINE_KWARGS", kwargs)
    emit("S3C_GPU_MEMORY", {"phase": "before_engine_init", "memory_used_mib": gpu_memory_used_mib()})

    llm = LLM(**kwargs)

    cfg = llm.llm_engine.vllm_config
    mc, cc, sc = cfg.model_config, cfg.cache_config, cfg.scheduler_config
    comp = cfg.compilation_config
    emit("S3C_EFFECTIVE_ENGINE_CONFIG", {
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
        "cudagraph_mode": enum_name(comp.cudagraph_mode, CUDAGraphMode),
        "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size,
        "calculate_kv_scales": cc.calculate_kv_scales,
        "kv_cache_dtype_skip_layers": list(cc.kv_cache_dtype_skip_layers),
        "hf_quantization_config": getattr(mc.hf_config, "quantization_config", None),
    })
    emit("S3C_KV_DTYPE", {
        "requested_kv_cache_dtype": args.kv_cache_dtype,
        "engine_cache_dtype": cc.cache_dtype,
        "resolved_kv_torch_dtype": str(get_kv_cache_torch_dtype(cc.cache_dtype, mc.dtype)),
        "kv_quant_mode": get_kv_quant_mode(cc.cache_dtype).name,
    })
    emit("S3C_CAPACITY", {"num_gpu_blocks": cc.num_gpu_blocks, "block_size": cc.block_size,
                          "capacity_tokens": cc.num_gpu_blocks * cc.block_size})
    emit("S3C_GPU_MEMORY", {"phase": "after_engine_init", "memory_used_mib": gpu_memory_used_mib()})

    tok = llm.get_tokenizer()
    bos = tok.bos_token_id
    filler = tok.encode(" the", add_special_tokens=False)[-1]
    sp = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True)
    prompt = None

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

    signal.signal(signal.SIGALRM, _on_alarm)
    plan = [("conditioning", args.conditioning_prompt)] + [("measured", p) for p in points]
    for i, (role, prompt_tokens) in enumerate(plan):
        prompt = [{"prompt_token_ids": [bos] + [filler] * (prompt_tokens - 1)}]
        planned_hash = hashlib.sha256(json.dumps(prompt[0]["prompt_token_ids"]).encode("utf-8")).hexdigest()
        emit("S3C_POINT_BEGIN", {"i": i, "role": role, "prompt_tokens": prompt_tokens,
                                 "planned_prompt_token_ids_sha256": planned_hash, "unix_time": time.time()})
        signal.alarm(args.request_cap_s)
        try:
            row = one(i)
        except RequestCapExceeded:
            emit("S3C_REQUEST_TIMEOUT", {"i": i, "role": role, "prompt_tokens": prompt_tokens,
                                         "cap_s": args.request_cap_s, "unix_time": time.time()})
            os._exit(REQUEST_TIMEOUT_EXIT)
        except Exception as exc:  # noqa: BLE001  (reported, never retried)
            signal.alarm(0)
            emit("S3C_REQUEST_FAILURE", {"i": i, "role": role, "prompt_tokens": prompt_tokens,
                                         "error": f"{type(exc).__name__}: {exc}"[:2000]})
            os._exit(REQUEST_FAILURE_EXIT)
        signal.alarm(0)
        emit("S3C_POINT", {**row, "i": i, "role": role, "planned_prompt_tokens": prompt_tokens,
                           "planned_prompt_token_ids_sha256": planned_hash, "unix_time": time.time(),
                           "gpu_memory_used_mib_after": gpu_memory_used_mib()})

    print("S3C_SERIES_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
