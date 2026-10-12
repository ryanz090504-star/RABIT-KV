"""
MLSys 2027 Experiment 14 -- single-condition real-engine worker for Model B (runs INSIDE the Modal container, one fresh
process per leg of the counterbalanced A-B-B-A session).

Derived MECHANICALLY from the accepted Experiment 13 worker (exp13_engine_worker.py, NOT modified; itself derived from
the accepted Exp4 worker). The Exp14 runner proves by AST before any run that this worker shares with the Exp13 worker:
BASE_ENGINE_KWARGS, CONTEXT_TOKENS, OUTPUT_TOKENS (identical values) and the timed region of one() with every sample
field. Differences (all outside the timed region):
  * machine-readable tags use the EXP14_ prefix;
  * --kv-cache-dtype accepts only bfloat16 and rabit_kv2 (no TurboQuant; the attention_config pin is never removed);
  * prompt construction: Model B's tokenizer defines NO BOS token (Qwen2.5: bos_token null, add_bos_token false), so
    the prompt is CONTEXT_TOKENS copies of the ' the' token id; for a tokenizer WITH a BOS token the Exp13 prompt
    ([BOS] + ' the' x 2047) is produced unchanged -- the length is CONTEXT_TOKENS either way;
  * EXP14_MODEL_GEOMETRY records the served model's attention geometry and the frozen RABIT-KV dispatch predicates
    evaluated on it (which serving path the frozen source selects; proven separately by exp14_shape_gate.py).
Engine topology: vLLM's default multiprocess engine core (identical to Exp4 / Exp13) for every condition.
This file never modifies vllm-kvquant; it only imports it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

# Identical to exp13_engine_worker.BASE_ENGINE_KWARGS (verified by AST).
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


def build_prompt_ids(bos, filler: int) -> list:
    """[BOS] + filler x (CONTEXT_TOKENS - 1) when the tokenizer has a BOS token (the Exp13 prompt), else
    filler x CONTEXT_TOKENS (Model B)."""
    if bos is None:
        return [filler] * CONTEXT_TOKENS
    return [bos] + [filler] * (CONTEXT_TOKENS - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-cache-dtype", required=True, choices=ALLOWED_KV_CACHE_DTYPES)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--warmups", type=int, required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--leg", required=True, help="leg label, e.g. A1/B1/B2/A2")
    args = ap.parse_args()
    if args.warmups < 1 or args.reps < 15:
        raise SystemExit("warmups must be >= 1 and reps >= 15 per leg")
    emit("EXP14_LEG", {"leg": args.leg, "kv_cache_dtype": args.kv_cache_dtype})

    from vllm import LLM, SamplingParams
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    from vllm.v1.kv_cache_interface import get_kv_quant_mode
    import vllm.v1.attention.ops.rabit_kv2 as r

    # Same frozen-source dispatch assertions as the canonical real_engine().
    markers = {
        "_RABIT2_STAGE4D3_4_TRITON_TAILPREP": bool(getattr(r, "_RABIT2_STAGE4D3_4_TRITON_TAILPREP", False)),
        "_RABIT2_FINAL_FAST_DECODE_APPEND": bool(getattr(r, "_RABIT2_FINAL_FAST_DECODE_APPEND", False)),
        "stage4d3_4_dispatch_active": r._rabit2_stage4b1_exactmeta_emit_tail_partial
        is r._rabit2_stage4d3_4_emit_tail_partial,
    }
    emit("EXP14_RABIT_MARKERS", markers)
    if not all(markers.values()):
        raise RuntimeError(f"RABIT-KV frozen-source markers missing: {markers}")

    kwargs = {"model": args.model_dir, **BASE_ENGINE_KWARGS, "kv_cache_dtype": args.kv_cache_dtype}
    emit("EXP14_REQUESTED_ENGINE_KWARGS", kwargs)

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
    emit("EXP14_EFFECTIVE_ENGINE_CONFIG", effective)

    q_heads = mc.get_num_attention_heads(cfg.parallel_config)
    kv_heads = mc.get_num_kv_heads(cfg.parallel_config)
    head_dim = mc.get_head_size()
    emit(
        "EXP14_MODEL_GEOMETRY",
        {
            "architectures": list(getattr(mc.hf_config, "architectures", []) or []),
            "num_hidden_layers": mc.get_num_layers(cfg.parallel_config),
            "num_attention_heads": q_heads,
            "num_kv_heads": kv_heads,
            "head_dim": head_dim,
            "sliding_window": mc.get_sliding_window(),
            "rabit_gqa4_decode_engaged": q_heads % kv_heads == 0 and (q_heads // kv_heads) % 4 == 0,
            "rabit_fast_append_engaged": kv_heads == 8 and head_dim == 128,
        },
    )

    mode = get_kv_quant_mode(cc.cache_dtype)
    emit(
        "EXP14_KV_DTYPE",
        {
            "requested_kv_cache_dtype": args.kv_cache_dtype,
            "engine_cache_dtype": cc.cache_dtype,
            "resolved_kv_torch_dtype": str(get_kv_cache_torch_dtype(cc.cache_dtype, mc.dtype)),
            "kv_quant_mode": mode.name,
        },
    )

    capacity = cc.num_gpu_blocks * cc.block_size
    emit(
        "EXP14_CAPACITY",
        {"num_gpu_blocks": cc.num_gpu_blocks, "block_size": cc.block_size, "capacity_tokens": capacity},
    )

    tok = llm.get_tokenizer()
    bos = tok.bos_token_id
    filler = tok.encode(" the", add_special_tokens=False)[-1]
    prompt = [{"prompt_token_ids": build_prompt_ids(bos, filler)}]
    sp = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True)
    emit(
        "EXP14_WORKLOAD",
        {
            "context_tokens": CONTEXT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "warmups": args.warmups,
            "reps": args.reps,
            "temperature": sp.temperature,
            "ignore_eos": sp.ignore_eos,
            "max_tokens": sp.max_tokens,
            "bos_token_id": bos,
            "filler_token_id": filler,
            "prompt_rule": "filler x 2048 (no BOS token)" if bos is None else "[BOS] + filler x 2047",
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
            "output_token_ids_sha256": hashlib.sha256(
                json.dumps(list(out.outputs[0].token_ids)).encode("utf-8")
            ).hexdigest(),
        }

    print("EXP14_WARMUP_BEGIN", flush=True)
    for i in range(args.warmups):
        print("EXP14_WARMUP " + json.dumps(one(i), sort_keys=True), flush=True)
    print("EXP14_WARMUP_END", flush=True)

    print("EXP14_MEASUREMENT_BEGIN", flush=True)
    for i in range(args.reps):
        print("EXP14_SAMPLE " + json.dumps(one(i), sort_keys=True), flush=True)
    print("EXP14_MEASUREMENT_END", flush=True)

    print("EXP14_WORKER_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
