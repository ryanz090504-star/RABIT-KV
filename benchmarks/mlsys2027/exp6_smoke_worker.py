"""
MLSys 2027 Experiment 6 -- concurrency CORRECTNESS SMOKE worker (runs INSIDE the
Modal container; one fresh engine process per dtype). Correctness only: NO
performance claim is made from this smoke test.

Protocol (fixed before any run):
  * engine: the frozen Experiment 3/5 BASE_ENGINE_KWARGS with ONE override,
    max_num_seqs = 4 (both dtypes), and kv_cache_dtype = bfloat16 | rabit_kv2;
  * rabit_kv2: VLLM_RABIT2_STAGE3C_IMPL=shared_decode and
    VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK=32 set explicitly BEFORE the engine
    starts and verified (requested == effective); bfloat16: both unset;
    profiling flags must be off;
  * workload: 4 distinct TEST prompts of exactly 2048 tokens (BOS + a fixed
    English passage repeated), 32 output tokens, greedy (temperature 0,
    ignore_eos); identical prompts and order for both dtypes;
  * phases (markers delimit them in the log):
      warmup      -- unmeasured: 4 distinct WARMUP prompts, each alone, then all 4
                     together (compiles single- and batch-4 shapes);
      single      -- each TEST prompt alone, in order 0..3 (reference outputs);
      concurrent  -- all 4 TEST prompts in ONE llm.generate call (max_num_seqs 4);
  * per request: prompt / output token-ID SHA-256, the output token IDs, and the
    engine-core timestamps scheduled_ts / first_token_ts / last_token_ts (one
    monotonic engine clock) used to PROVE overlap in the concurrent phase and
    serialization in the single phase.
Machine lines use the EXP6S_ prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

# Identical to exp3_engine_worker / exp5_engine_worker BASE_ENGINE_KWARGS (verified by AST in the runner).
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
SMOKE_OVERRIDES = {"max_num_seqs": 4}
ALLOWED_KV_CACHE_DTYPES = ("bfloat16", "rabit_kv2")
PROMPT_TOKENS = 2048
OUTPUT_TOKENS = 32
CONCURRENCY = 4
STAGE3C_ENV = ("VLLM_RABIT2_STAGE3C_IMPL", "VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK")
PROFILE_ENV = ("VLLM_RABIT2_STAGE3C_PROFILE", "VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE")
TEST_PASSAGES = (
    " The river bends twice before it reaches the old stone bridge near the mill.",
    " Seventeen violins were tuned by candlelight while the orchestra waited outside.",
    " Copper wire, glass beads and a broken compass filled the drawer of the desk.",
    " A cold wind from the northern mountains carried snow across the empty valley.",
)
WARMUP_PASSAGES = (
    " Bright lanterns hung from every balcony along the narrow harbor street.",
    " The mathematician wrote three pages of proofs on the back of a menu.",
    " Tall grass swayed around the abandoned railway station at dusk.",
    " Fresh bread cooled on the windowsill of the baker's small kitchen.",
)


def emit(tag: str, payload) -> None:
    print(f"{tag}={json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def gpu_memory_used_mib() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         text=True, capture_output=True, check=True).stdout
    return [int(x.strip()) for x in out.strip().splitlines()]


def sha(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode("utf-8")).hexdigest()


def build_prompt(tok, bos: int, passage: str) -> list[int]:
    body = tok.encode(passage, add_special_tokens=False)
    ids = [bos]
    while len(ids) < PROMPT_TOKENS:
        ids.extend(body)
    return ids[:PROMPT_TOKENS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-cache-dtype", required=True, choices=ALLOWED_KV_CACHE_DTYPES)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--stage3c-impl", choices=("shared_decode",), default=None)
    ap.add_argument("--query-block", type=int, choices=(32,), default=None)
    args = ap.parse_args()
    if any(os.environ.get(v, "0") != "0" for v in PROFILE_ENV):
        raise SystemExit("Stage3C profiling must be OFF")
    if args.kv_cache_dtype == "rabit_kv2":
        if args.stage3c_impl is None or args.query_block is None:
            raise SystemExit("rabit_kv2 requires explicit --stage3c-impl and --query-block")
        os.environ[STAGE3C_ENV[0]] = args.stage3c_impl
        os.environ[STAGE3C_ENV[1]] = str(args.query_block)
    else:
        if args.stage3c_impl is not None or args.query_block is not None:
            raise SystemExit("bfloat16 takes no Stage3C selection")
        for v in STAGE3C_ENV:
            os.environ.pop(v, None)

    from vllm import LLM, SamplingParams
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    from vllm.v1.kv_cache_interface import get_kv_quant_mode

    if args.kv_cache_dtype == "rabit_kv2":
        import vllm.v1.attention.ops.rabit_kv2_stage3c_shared_decode as sd

        s3 = {"applicable": True, "requested_impl": args.stage3c_impl, "effective_impl": sd.rabit2_stage3c_impl(),
              "requested_query_block": args.query_block,
              "effective_query_block": sd.rabit2_shared_decode_query_block(),
              "env": {v: os.environ.get(v) for v in STAGE3C_ENV},
              "profiling_env": {v: os.environ.get(v) for v in PROFILE_ENV},
              "shared_decode_module_sha256": hashlib.sha256(open(sd.__file__, "rb").read()).hexdigest()}
        emit("EXP6S_STAGE_IMPL", s3)
        if (s3["effective_impl"], s3["effective_query_block"]) != (args.stage3c_impl, args.query_block):
            raise RuntimeError(f"Stage3C selector does not report the requested configuration: {s3}")
    else:
        emit("EXP6S_STAGE_IMPL", {"applicable": False, "env": {v: os.environ.get(v) for v in STAGE3C_ENV},
                                  "profiling_env": {v: os.environ.get(v) for v in PROFILE_ENV}})

    kwargs = {"model": args.model_dir, **BASE_ENGINE_KWARGS, **SMOKE_OVERRIDES, "kv_cache_dtype": args.kv_cache_dtype}
    emit("EXP6S_REQUESTED_ENGINE_KWARGS", kwargs)
    emit("EXP6S_GPU_MEMORY", {"phase": "before_engine_init", "memory_used_mib": gpu_memory_used_mib()})
    llm = LLM(**kwargs)
    cfg = llm.llm_engine.vllm_config
    mc, cc, sc = cfg.model_config, cfg.cache_config, cfg.scheduler_config
    emit("EXP6S_EFFECTIVE_ENGINE_CONFIG", {
        "model": mc.model, "model_dtype": str(mc.dtype), "max_model_len": mc.max_model_len,
        "enforce_eager": mc.enforce_eager, "block_size": cc.block_size,
        "gpu_memory_utilization": cc.gpu_memory_utilization, "enable_prefix_caching": cc.enable_prefix_caching,
        "max_num_batched_tokens": sc.max_num_batched_tokens, "max_num_seqs": sc.max_num_seqs,
        "enable_chunked_prefill": sc.enable_chunked_prefill, "attention_backend": str(cfg.attention_config.backend),
        "log_stats": bool(getattr(llm.llm_engine, "log_stats", False))})
    emit("EXP6S_KV_DTYPE", {"requested_kv_cache_dtype": args.kv_cache_dtype, "engine_cache_dtype": cc.cache_dtype,
                            "resolved_kv_torch_dtype": str(get_kv_cache_torch_dtype(cc.cache_dtype, mc.dtype)),
                            "kv_quant_mode": get_kv_quant_mode(cc.cache_dtype).name})
    emit("EXP6S_CAPACITY", {"num_gpu_blocks": cc.num_gpu_blocks, "block_size": cc.block_size,
                            "capacity_tokens": cc.num_gpu_blocks * cc.block_size})
    emit("EXP6S_GPU_MEMORY", {"phase": "after_engine_init", "memory_used_mib": gpu_memory_used_mib()})

    tok = llm.get_tokenizer()
    bos = tok.bos_token_id
    tests = [build_prompt(tok, bos, p) for p in TEST_PASSAGES]
    warm = [build_prompt(tok, bos, p) for p in WARMUP_PASSAGES]
    emit("EXP6S_WORKLOAD", {"prompt_tokens": PROMPT_TOKENS, "output_tokens": OUTPUT_TOKENS,
                            "concurrency": CONCURRENCY, "temperature": 0.0, "ignore_eos": True,
                            "test_prompt_sha256": [sha(p) for p in tests],
                            "warmup_prompt_sha256": [sha(p) for p in warm]})
    sp = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True)

    def run(prompts: list[list[int]]) -> list[dict]:
        outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
        rows = []
        for o in outs:
            m = o.metrics
            ids = list(o.outputs[0].token_ids)
            rows.append({"prompt_tokens": len(o.prompt_token_ids), "prompt_token_ids_sha256": sha(o.prompt_token_ids),
                         "output_tokens": len(ids), "output_token_ids_sha256": sha(ids), "output_token_ids": ids,
                         "finish_reason": o.outputs[0].finish_reason,
                         "scheduled_ts": getattr(m, "scheduled_ts", None),
                         "first_token_ts": getattr(m, "first_token_ts", None),
                         "last_token_ts": getattr(m, "last_token_ts", None)})
        return rows

    print("EXP6S_WARMUP_BEGIN", flush=True)
    for p in warm:
        run([p])
    run(warm)
    print("EXP6S_WARMUP_END", flush=True)

    print("EXP6S_SINGLE_BEGIN", flush=True)
    for i, p in enumerate(tests):
        emit("EXP6S_SINGLE", {"i": i, **run([p])[0]})
    print("EXP6S_SINGLE_END", flush=True)

    print("EXP6S_CONCURRENT_BEGIN", flush=True)
    for i, row in enumerate(run(tests)):
        emit("EXP6S_CONCURRENT", {"i": i, **row})
    print("EXP6S_CONCURRENT_END", flush=True)

    emit("EXP6S_GPU_MEMORY", {"phase": "after_smoke", "memory_used_mib": gpu_memory_used_mib()})
    print("EXP6S_WORKER_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
