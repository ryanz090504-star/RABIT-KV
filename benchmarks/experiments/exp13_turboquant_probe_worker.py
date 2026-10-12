"""
MLSys 2027 Experiment 13 -- TurboQuant FEASIBILITY PROBE worker (NON-EVIDENCE; runs INSIDE the Modal container).

Constructs exactly ONE vLLM engine with kv_cache_dtype=turboquant_k3v4_nc under the intended frozen Exp13 engine
settings (requested block_size 32), records the effective state the engine actually chose, and runs one short
sanity generation. No latency is measured; nothing here is Exp13 evidence. No retry, no alternate settings.

The attention backend is NOT pinned: TRITON_ATTN (used by BF16 / FP8 / RABIT) rejects TurboQuant cache dtypes, and
pinning TURBOQUANT globally would also apply to the BF16 boundary layers; vLLM's method-required selection is used.

For introspection of per-layer cache tensors and backends WITHOUT engine RPC, the engine core runs in-process
(VLLM_ENABLE_V1_MULTIPROCESSING=0). This changes process topology only, not block-size / page / allocator logic.
This file never modifies vllm-kvquant; it only imports it. Machine-readable lines use the EXP13_PROBE_ prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # before importing vllm

REQUESTED_BLOCK_SIZE = 32
PROBE_ENGINE_KWARGS = {
    "dtype": "bfloat16",
    "block_size": REQUESTED_BLOCK_SIZE,
    "max_model_len": 32768,
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 32,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": True,
    "gpu_memory_utilization": 0.82,
    "enforce_eager": True,
    "trust_remote_code": True,
    "disable_log_stats": False,
    "kv_cache_dtype": "turboquant_k3v4_nc",
}


def emit(tag: str, payload) -> None:
    print(f"EXP13_PROBE_{tag}={json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _get(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _model_runner(llm):
    candidates = ["llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner",
                  "llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner",
                  "llm_engine.model_executor.driver_worker.worker.model_runner"]
    errors = {}
    for c in candidates:
        try:
            return _get(llm, c), c, errors
        except AttributeError as e:
            errors[c] = repr(e)
    return None, None, errors


def _spec_record(spec) -> dict:
    out = {"class": type(spec).__name__}
    for k in ("block_size", "num_kv_heads", "head_size", "head_size_v", "dtype", "tq_slot_size", "page_size_padded",
              "indexes_kv_by_block_stride", "kv_quant_mode"):
        if hasattr(spec, k):
            out[k] = str(getattr(spec, k))
    for k in ("page_size_bytes", "real_page_size_bytes"):
        try:
            out[k] = int(getattr(spec, k))
        except Exception as e:  # noqa: BLE001
            out[k] = f"unavailable: {e!r}"
    return out


def _tensors_of(obj, prefix: str, torch) -> list[dict]:
    found = []
    for name, val in list(vars(obj).items()):
        if torch.is_tensor(val) and val.is_cuda:
            found.append({"attr": f"{prefix}.{name}", "dtype": str(val.dtype), "shape": list(val.shape),
                          "nbytes": val.numel() * val.element_size()})
        elif isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                if torch.is_tensor(v) and v.is_cuda:
                    found.append({"attr": f"{prefix}.{name}[{i}]", "dtype": str(v.dtype), "shape": list(v.shape),
                                  "nbytes": v.numel() * v.element_size()})
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()
    kwargs = {"model": args.model_dir, **PROBE_ENGINE_KWARGS}
    emit("LABEL", "NON-EVIDENCE FEASIBILITY PROBE -- not Exp13 evidence; no latency measured")
    emit("REQUESTED_ENGINE_KWARGS", kwargs)
    emit("PROCESS_TOPOLOGY", {"VLLM_ENABLE_V1_MULTIPROCESSING": os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"],
                              "attention_backend_pinned": None})

    import torch
    from vllm import LLM, SamplingParams

    try:
        llm = LLM(**kwargs)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        frames = traceback.extract_tb(e.__traceback__)
        emit("STARTUP", {"success": False, "exception_type": type(e).__name__, "exception": str(e),
                         "failure_location": {"file": frames[-1].filename, "line": frames[-1].lineno,
                                              "function": frames[-1].name} if frames else None})
        print("EXP13_PROBE_TRACEBACK_BEGIN\n" + tb + "EXP13_PROBE_TRACEBACK_END", flush=True)
        return 2
    emit("STARTUP", {"success": True})

    cfg = llm.llm_engine.vllm_config
    mc, cc, sc = cfg.model_config, cfg.cache_config, cfg.scheduler_config
    effective = {
        "requested_block_size": REQUESTED_BLOCK_SIZE, "effective_block_size": cc.block_size,
        "block_size_changed": cc.block_size != REQUESTED_BLOCK_SIZE,
        "num_gpu_blocks": cc.num_gpu_blocks,
        "allocatable_kv_tokens": (cc.num_gpu_blocks * cc.block_size) if cc.num_gpu_blocks else None,
        "cache_dtype": cc.cache_dtype, "kv_cache_dtype_skip_layers": list(cc.kv_cache_dtype_skip_layers),
        "skip_page_size_padded": getattr(cc, "skip_page_size_padded", None),
        "gpu_memory_utilization": cc.gpu_memory_utilization, "enable_prefix_caching": cc.enable_prefix_caching,
        "max_model_len": mc.max_model_len, "enforce_eager": mc.enforce_eager, "model_dtype": str(mc.dtype),
        "max_num_seqs": sc.max_num_seqs, "max_num_batched_tokens": sc.max_num_batched_tokens,
        "enable_chunked_prefill": sc.enable_chunked_prefill,
        "attention_config_backend": str(cfg.attention_config.backend),
        "flash_attn_version": getattr(cfg.attention_config, "flash_attn_version", None),
    }
    emit("EFFECTIVE_ENGINE_STATE", effective)

    mr, path, errors = _model_runner(llm)
    emit("MODEL_RUNNER", {"found": mr is not None, "path": path, "errors": errors})
    if mr is not None:
        kcc = getattr(mr, "kv_cache_config", None)
        if kcc is not None:
            emit("KV_CACHE_CONFIG", {
                "num_blocks": getattr(kcc, "num_blocks", None),
                "kv_cache_tensors": [{"size": t.size, "shared_by": list(t.shared_by)}
                                     for t in getattr(kcc, "kv_cache_tensors", [])],
                "kv_cache_groups": [{"layer_names": list(g.layer_names), "spec": _spec_record(g.kv_cache_spec)}
                                    for g in getattr(kcc, "kv_cache_groups", [])]})
        ctx = cfg.compilation_config.static_forward_context
        layers, total_kv = {}, 0
        for name, mod in ctx.items():
            kv = getattr(mod, "kv_cache", None)
            t = kv[0] if isinstance(kv, (list, tuple)) and kv else kv
            rec = {"module": type(mod).__name__, "kv_cache_dtype": getattr(mod, "kv_cache_dtype", None),
                   "attn_backend": getattr(getattr(mod, "attn_backend", None), "__name__", None),
                   "impl": type(getattr(mod, "impl", None)).__name__}
            if torch.is_tensor(t):
                nbytes = t.numel() * t.element_size()
                total_kv += nbytes
                rec.update({"kv_tensor_dtype": str(t.dtype), "kv_tensor_shape": list(t.shape),
                            "kv_tensor_nbytes": nbytes, "kv_tensor_data_ptr": t.data_ptr()})
            other = _tensors_of(mod, "layer", torch)
            impl = getattr(mod, "impl", None)
            if impl is not None:
                other += _tensors_of(impl, "impl", torch)
            rec["other_cuda_tensors"] = [x for x in other if not (torch.is_tensor(t) and x["nbytes"] == rec.get(
                "kv_tensor_nbytes") and x["attr"].startswith("layer.kv_cache"))]
            layers[name] = rec
        emit("LAYERS", layers)
        emit("KV_TOTAL", {"sum_layer_kv_tensor_nbytes": total_kv,
                          "torch_cuda_memory_allocated": torch.cuda.memory_allocated(),
                          "torch_cuda_memory_reserved": torch.cuda.memory_reserved()})

    tok = llm.get_tokenizer()
    ids = tok.encode("The capital of France is", add_special_tokens=True)
    out = llm.generate([{"prompt_token_ids": ids}], SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False)[0]
    emit("SANITY_GENERATION", {"prompt_tokens": len(ids), "output_tokens": len(out.outputs[0].token_ids),
                               "text": out.outputs[0].text, "finish_reason": out.outputs[0].finish_reason})
    print("EXP13_PROBE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
