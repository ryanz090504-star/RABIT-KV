"""
MLSys 2027 Experiment 6 -- ONE concurrency point (runs INSIDE the Modal container;
one fresh engine process per dtype x prompt length x concurrency x trial).

Execution path (reviewed amendment): vLLM's normal multi-request LLM.generate
API, validated by the accepted concurrency correctness smoke, instead of
`vllm bench throughput` (which accepts rabit_kv2 but samples at temperature 1.0
and exposes no per-request timestamps / token IDs needed by the pre-registered
metrics and audit).

Per point:
  * engine: frozen Experiment 3/5 BASE_ENGINE_KWARGS with max_num_seqs = C (the
    target concurrency, identical for both dtypes) and kv_cache_dtype;
    rabit_kv2: VLLM_RABIT2_STAGE3C_IMPL / VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK set
    explicitly BEFORE the engine starts and verified; bfloat16: both unset;
    profiling flags off;
  * workload (exp6_workload.py): the frozen ordered set of 256 distinct prompts
    of this length (hash-checked), 32 greedy output tokens (temperature 0,
    ignore_eos);
  * phases: setup | compile-conditioning (reviewed amendment: exactly C fixed
    conditioning prompts of the sweep length, one concurrent llm.generate call
    under max_num_seqs = C, identical for both dtypes; outputs / timings are
    discarded from the results and emitted only for validity checks) | 2 warmup
    requests (discarded) | 256 measured requests in ONE llm.generate call
    (closed-loop: the scheduler keeps at most C sequences in flight and admits
    the next request as a slot frees);
  * per measured request: engine-core queued / scheduled / first-token /
    last-token timestamps (one monotonic engine clock), output token count,
    finish reason, prompt / output token-ID SHA-256; total wall time
    (perf_counter around the measured llm.generate); vllm:num_preemptions
    before / after the measured phase when the metrics API exposes it;
  * a request exception after engine init is reported as EXP6_REQUEST_FAILURE
    (kind request_oom / request_execution_failure) with exit code 75.
Machine lines use the EXP6_ prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp6_workload as wl  # noqa: E402

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
ALLOWED_KV_CACHE_DTYPES = ("bfloat16", "rabit_kv2")
REQUEST_FAILURE_EXIT = 75
STAGE3C_ENV = ("VLLM_RABIT2_STAGE3C_IMPL", "VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK")
PROFILE_ENV = ("VLLM_RABIT2_STAGE3C_PROFILE", "VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE")


def emit(tag: str, payload) -> None:
    print(f"{tag}={json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def gpu_memory_used_mib() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         text=True, capture_output=True, check=True).stdout
    return [int(x.strip()) for x in out.strip().splitlines()]


def preemptions(llm):
    """vllm:num_preemptions (summed over engines) or None when the metrics API does not expose it."""
    try:
        vals = [m for m in llm.get_metrics() if getattr(m, "name", "") in ("vllm:num_preemptions",
                                                                        "vllm:num_preemptions_total")]
    except Exception:  # noqa: BLE001
        return None
    return sum(float(getattr(m, "value", 0.0)) for m in vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-cache-dtype", required=True, choices=ALLOWED_KV_CACHE_DTYPES)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-tokens", type=int, required=True, choices=wl.PROMPT_LENGTHS)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--trial", type=int, required=True, choices=range(1, wl.TRIALS + 1))
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt-set-sha256", required=True, help="pinned ordered_set_sha256 of the measured prompts")
    ap.add_argument("--stage3c-impl", choices=("shared_decode",), default=None)
    ap.add_argument("--query-block", type=int, choices=(32,), default=None)
    args = ap.parse_args()
    if args.concurrency < 1:
        raise SystemExit("concurrency must be >= 1")
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

    measured = wl.measured_prompts(args.prompt_tokens)
    warm = wl.warmup_prompts(args.prompt_tokens)
    cond = wl.conditioning_prompts(args.prompt_tokens, args.concurrency)
    digest = wl.set_digest(measured)
    if digest["ordered_set_sha256"] != args.prompt_set_sha256:
        raise SystemExit("measured prompt set does not match the pinned hash")
    emit("EXP6_POINT", {"label": args.label, "kv_cache_dtype": args.kv_cache_dtype, "trial": args.trial,
                        "prompt_tokens": args.prompt_tokens, "target_concurrency": args.concurrency})

    from vllm import LLM, SamplingParams
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    from vllm.v1.kv_cache_interface import get_kv_quant_mode

    if args.kv_cache_dtype == "rabit_kv2":
        import vllm.v1.attention.ops.rabit_kv2_stage3c_shared_decode as sd

        s3 = {"applicable": True, "requested_impl": args.stage3c_impl, "effective_impl": sd.rabit2_stage3c_impl(),
              "requested_query_block": args.query_block, "effective_query_block": sd.rabit2_shared_decode_query_block(),
              "env": {v: os.environ.get(v) for v in STAGE3C_ENV},
              "profiling_env": {v: os.environ.get(v) for v in PROFILE_ENV},
              "shared_decode_module_sha256": hashlib.sha256(open(sd.__file__, "rb").read()).hexdigest()}
        emit("EXP6_STAGE_IMPL", s3)
        if (s3["effective_impl"], s3["effective_query_block"]) != (args.stage3c_impl, args.query_block):
            raise RuntimeError(f"Stage3C selector does not report the requested configuration: {s3}")
    else:
        emit("EXP6_STAGE_IMPL", {"applicable": False, "env": {v: os.environ.get(v) for v in STAGE3C_ENV},
                                 "profiling_env": {v: os.environ.get(v) for v in PROFILE_ENV}})

    kwargs = {"model": args.model_dir, **BASE_ENGINE_KWARGS, "max_num_seqs": args.concurrency,
              "kv_cache_dtype": args.kv_cache_dtype}
    emit("EXP6_REQUESTED_ENGINE_KWARGS", kwargs)
    emit("EXP6_GPU_MEMORY", {"phase": "before_engine_init", "memory_used_mib": gpu_memory_used_mib()})
    llm = LLM(**kwargs)
    cfg = llm.llm_engine.vllm_config
    mc, cc, sc = cfg.model_config, cfg.cache_config, cfg.scheduler_config
    emit("EXP6_EFFECTIVE_ENGINE_CONFIG", {
        "model": mc.model, "model_dtype": str(mc.dtype), "max_model_len": mc.max_model_len,
        "enforce_eager": mc.enforce_eager, "block_size": cc.block_size,
        "gpu_memory_utilization": cc.gpu_memory_utilization, "enable_prefix_caching": cc.enable_prefix_caching,
        "max_num_batched_tokens": sc.max_num_batched_tokens, "max_num_seqs": sc.max_num_seqs,
        "enable_chunked_prefill": sc.enable_chunked_prefill, "attention_backend": str(cfg.attention_config.backend),
        "log_stats": bool(getattr(llm.llm_engine, "log_stats", False))})
    emit("EXP6_KV_DTYPE", {"requested_kv_cache_dtype": args.kv_cache_dtype, "engine_cache_dtype": cc.cache_dtype,
                           "resolved_kv_torch_dtype": str(get_kv_cache_torch_dtype(cc.cache_dtype, mc.dtype)),
                           "kv_quant_mode": get_kv_quant_mode(cc.cache_dtype).name})
    emit("EXP6_CAPACITY", {"num_gpu_blocks": cc.num_gpu_blocks, "block_size": cc.block_size,
                           "capacity_tokens": cc.num_gpu_blocks * cc.block_size})
    emit("EXP6_GPU_MEMORY", {"phase": "after_engine_init", "memory_used_mib": gpu_memory_used_mib()})
    emit("EXP6_WORKLOAD", {"prompt_tokens": args.prompt_tokens, "output_tokens": wl.OUTPUT_TOKENS,
                           "measured_requests": wl.MEASURED_REQUESTS, "warmup_requests": wl.WARMUP_REQUESTS,
                           "temperature": 0.0, "ignore_eos": True, "prompt_set": digest,
                           "warmup_prompt_sha256": wl.set_digest(warm)["per_prompt_sha256"],
                           "conditioning_requests": len(cond), "conditioning_prompt_set": wl.set_digest(cond)})
    sp = SamplingParams(temperature=0.0, max_tokens=wl.OUTPUT_TOKENS, ignore_eos=True)

    phase = "conditioning"
    try:
        print("EXP6_CONDITIONING_BEGIN", flush=True)
        couts = llm.generate([{"prompt_token_ids": p} for p in cond], sp, use_tqdm=False)
        print("EXP6_CONDITIONING_END", flush=True)
        for i, o in enumerate(couts):  # validity evidence only; never used in result tables
            m = o.metrics
            ids = list(o.outputs[0].token_ids) if o.outputs else []
            emit("EXP6_CONDITIONING_REQUEST", {
                "i": i, "prompt_tokens": len(o.prompt_token_ids),
                "prompt_token_ids_sha256": wl.ids_sha256(o.prompt_token_ids), "output_tokens": len(ids),
                "finish_reason": o.outputs[0].finish_reason if o.outputs else None,
                "scheduled_ts": getattr(m, "scheduled_ts", None), "last_token_ts": getattr(m, "last_token_ts", None)})
        phase = "warmup"
        print("EXP6_WARMUP_BEGIN", flush=True)
        llm.generate([{"prompt_token_ids": p} for p in warm], sp, use_tqdm=False)
        print("EXP6_WARMUP_END", flush=True)
        phase = "measured"
        pre0 = preemptions(llm)
        print("EXP6_MEASURED_BEGIN", flush=True)
        t0 = time.perf_counter()
        outs = llm.generate([{"prompt_token_ids": p} for p in measured], sp, use_tqdm=False)
        wall_s = time.perf_counter() - t0
        print("EXP6_MEASURED_END", flush=True)
    except Exception as exc:  # noqa: BLE001  (reported, never retried)
        text = f"{type(exc).__name__}: {exc}"
        kind = "request_oom" if ("OutOfMemoryError" in text or "out of memory" in text.lower()) \
            else "request_execution_failure"
        emit("EXP6_REQUEST_FAILURE", {"phase": phase, "kind": kind, "error": text[:2000]})
        return REQUEST_FAILURE_EXIT
    pre1 = preemptions(llm)
    for i, o in enumerate(outs):
        m = o.metrics
        ids = list(o.outputs[0].token_ids) if o.outputs else []
        emit("EXP6_REQUEST", {
            "i": i, "prompt_tokens": len(o.prompt_token_ids),
            "prompt_token_ids_sha256": wl.ids_sha256(o.prompt_token_ids),
            "output_tokens": len(ids), "output_token_ids_sha256": wl.ids_sha256(ids),
            "finish_reason": o.outputs[0].finish_reason if o.outputs else None,
            "queued_ts": getattr(m, "queued_ts", None), "scheduled_ts": getattr(m, "scheduled_ts", None),
            "first_token_ts": getattr(m, "first_token_ts", None), "last_token_ts": getattr(m, "last_token_ts", None)})
    emit("EXP6_MEASURED_SUMMARY", {"wall_s": wall_s, "returned_requests": len(outs),
                                   "preemptions_before": pre0, "preemptions_after": pre1,
                                   "preemption_counter_available": pre0 is not None and pre1 is not None})
    emit("EXP6_GPU_MEMORY", {"phase": "after_measured", "memory_used_mib": gpu_memory_used_mib()})
    print("EXP6_WORKER_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
