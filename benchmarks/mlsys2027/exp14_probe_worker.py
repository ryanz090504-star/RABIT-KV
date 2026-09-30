"""
MLSys 2027 Experiment 14 -- NON-EVIDENCE feasibility-probe worker (runs INSIDE the Modal container, one fresh process
per KV dtype). Checks that Model B (Qwen2.5-7B-Instruct) initializes on the frozen snapshot with the Exp14 engine
settings, reports its geometry / cache dtype / allocator capacity / backend, and completes one latency-workload-shaped
generation plus one short natural-language sanity generation.

It emits NO timing of any kind (no TTFT / TPOT / wall): the probe cannot preview the measured comparison. Its engine
settings and prompt rule are imported from exp14_engine_worker.py so they cannot drift from the measured legs.
This file never modifies vllm-kvquant; it only imports it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp14_engine_worker import (  # noqa: E402
    ALLOWED_KV_CACHE_DTYPES,
    BASE_ENGINE_KWARGS,
    OUTPUT_TOKENS,
    build_prompt_ids,
)

SANITY_QUESTION = "What is the capital of France? Answer in one word."
SANITY_MAX_TOKENS = 16


def emit(tag: str, payload) -> None:
    print(f"{tag}={json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-cache-dtype", required=True, choices=ALLOWED_KV_CACHE_DTYPES)
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.v1.kv_cache_interface import get_kv_quant_mode

    kwargs = {"model": args.model_dir, **BASE_ENGINE_KWARGS, "kv_cache_dtype": args.kv_cache_dtype}
    emit("EXP14P_REQUESTED_ENGINE_KWARGS", kwargs)
    llm = LLM(**kwargs)
    cfg = llm.llm_engine.vllm_config
    mc, cc = cfg.model_config, cfg.cache_config
    q_heads = mc.get_num_attention_heads(cfg.parallel_config)
    kv_heads = mc.get_num_kv_heads(cfg.parallel_config)
    head_dim = mc.get_head_size()
    emit("EXP14P_MODEL_GEOMETRY", {
        "architectures": list(getattr(mc.hf_config, "architectures", []) or []),
        "num_hidden_layers": mc.get_num_layers(cfg.parallel_config), "num_attention_heads": q_heads,
        "num_kv_heads": kv_heads, "head_dim": head_dim, "sliding_window": mc.get_sliding_window(),
        "rabit_gqa4_decode_engaged": q_heads % kv_heads == 0 and (q_heads // kv_heads) % 4 == 0,
        "rabit_fast_append_engaged": kv_heads == 8 and head_dim == 128})
    emit("EXP14P_KV", {"engine_cache_dtype": cc.cache_dtype, "kv_quant_mode": get_kv_quant_mode(cc.cache_dtype).name,
                       "block_size": cc.block_size, "num_gpu_blocks": cc.num_gpu_blocks,
                       "capacity_tokens": cc.num_gpu_blocks * cc.block_size,
                       "attention_backend": str(cfg.attention_config.backend),
                       "kv_cache_dtype_skip_layers": list(cc.kv_cache_dtype_skip_layers)})

    tok = llm.get_tokenizer()
    filler = tok.encode(" the", add_special_tokens=False)[-1]
    ids = build_prompt_ids(tok.bos_token_id, filler)
    out = llm.generate([{"prompt_token_ids": ids}],
                       SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True), use_tqdm=False)[0]
    gen = list(out.outputs[0].token_ids)
    emit("EXP14P_WORKLOAD_GENERATION", {"prompt_tokens": len(out.prompt_token_ids), "output_tokens": len(gen),
                                        "bos_token_id": tok.bos_token_id, "filler_token_id": filler,
                                        "output_token_ids_sha256": hashlib.sha256(json.dumps(gen).encode()).hexdigest()})

    chat = tok.apply_chat_template([{"role": "user", "content": SANITY_QUESTION}], tokenize=False,
                                   add_generation_prompt=True)
    s = llm.generate([chat], SamplingParams(temperature=0.0, max_tokens=SANITY_MAX_TOKENS), use_tqdm=False)[0]
    emit("EXP14P_SANITY_GENERATION", {"question": SANITY_QUESTION, "text": s.outputs[0].text,
                                      "output_tokens": len(s.outputs[0].token_ids)})
    print("EXP14P_WORKER_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
