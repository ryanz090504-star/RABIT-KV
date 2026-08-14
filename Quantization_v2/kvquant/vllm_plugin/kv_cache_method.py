"""Scaffold for a future KVQuant KV cache method for vLLM.

This file is not a complete vLLM integration. It must grow the current vLLM
KV-cache quantization API before any result can be labeled deploy latency.

Reference: vllm/model_executor/layers/quantization/kv_cache.py
"""

from __future__ import annotations

from typing import Any

try:
    import vllm  # noqa: F401
    HAS_VLLM = True
except ImportError:  # pragma: no cover
    HAS_VLLM = False


if HAS_VLLM:
    import torch
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

    class KVQuantKVCacheMethod(BaseKVCacheMethod):
        """Incomplete vLLM KV cache method sketch.

        Future work must implement the vLLM-required setup and runtime hooks.
        """

        def apply(self, layer: torch.nn.Module) -> torch.Tensor:
            """Quantize or dequantize a cache entry.

            In KVQuant, dequantization happens inside the Triton attention
            kernel, so this method is a pass-through that marks the cache
            dtype for the backend.
            """
            return layer  # identity; actual dequant is in the attention kernel
