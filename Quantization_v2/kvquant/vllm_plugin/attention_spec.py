"""Scaffold for a future KVQuant attention spec for vLLM V1.

The dataclasses here are local descriptors, not registered vLLM ``KVCacheSpec``
subclasses.

Reference: vllm/v1/kv_cache_interface.py:AttentionSpec
"""

from __future__ import annotations

from dataclasses import dataclass

from kvquant.vllm_plugin.config import KVQuantConfig, packed_kv_page_bytes


def make_spec_from_config(
    cfg: KVQuantConfig,
    *,
    num_kv_heads_override: int | None = None,
    head_size_override: int | None = None,
) -> dict:
    """Build a local attention spec dict for future vLLM integration.

    This return value is not consumed by vLLM in the current codebase.
    """
    num_kv_heads = num_kv_heads_override or cfg.num_kv_heads
    head_size = head_size_override or cfg.head_dim
    if num_kv_heads == cfg.num_kv_heads and head_size == cfg.head_dim:
        byte_size = cfg.page_size_bytes
        packed_size = cfg.packed_page_size_bytes
        scale_size = cfg.scale_page_size_bytes
    else:
        packed_size = packed_kv_page_bytes(
            nbits=cfg.nbits,
            num_kv_heads=num_kv_heads,
            block_size=cfg.block_size,
            head_dim=head_size,
        )
        if cfg.scale_strategy == "per_head":
            scale_size = 2 * num_kv_heads * cfg.scale_dtype_bytes
        elif cfg.scale_strategy in {"per_token", "per_token_head", "per_token_per_head"}:
            scale_size = 2 * num_kv_heads * cfg.block_size * cfg.scale_dtype_bytes
        elif cfg.scale_strategy in {"none", "global"}:
            scale_size = 2 * cfg.scale_dtype_bytes
        else:
            raise ValueError(f"unsupported scale_strategy: {cfg.scale_strategy}")
        byte_size = packed_size + scale_size

    return {
        "page_size_bytes": byte_size,
        "packed_page_size_bytes": packed_size,
        "scale_page_size_bytes": scale_size,
        "block_size": cfg.block_size,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "dtype_str": cfg.kv_cache_dtype,
        "nbits": cfg.nbits,
        "policy_name": cfg.policy_name,
        "scale_strategy": cfg.scale_strategy,
    }


@dataclass
class KVQuantAttentionSpec:
    """Python-level descriptor for KVQuant custom dtypes.

    It is usable without importing vLLM internals and is meant as a stable
    handoff object for a future integration pass.
    """

    num_kv_heads: int
    head_size: int
    block_size: int = 16
    nbits: int = 3
    scale_strategy: str = "per_token_head"
    scale_dtype_bytes: int = 4

    @property
    def dtype_str(self) -> str:
        from kvquant.vllm_plugin.config import KVQUANT_NBITS_TO_DTYPE
        return KVQUANT_NBITS_TO_DTYPE[self.nbits]

    @property
    def page_size_bytes(self) -> int:
        packed = packed_kv_page_bytes(
            nbits=self.nbits,
            num_kv_heads=self.num_kv_heads,
            block_size=self.block_size,
            head_dim=self.head_size,
        )
        if self.scale_strategy == "per_head":
            scale = 2 * self.num_kv_heads * self.scale_dtype_bytes
        elif self.scale_strategy in {"per_token", "per_token_head", "per_token_per_head"}:
            scale = 2 * self.num_kv_heads * self.block_size * self.scale_dtype_bytes
        elif self.scale_strategy in {"none", "global"}:
            scale = 2 * self.scale_dtype_bytes
        else:
            raise ValueError(f"unsupported scale_strategy: {self.scale_strategy}")
        return packed + scale
