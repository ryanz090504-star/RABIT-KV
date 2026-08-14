"""KVQuant configuration for vLLM integration.

Defines the KVQuantConfig dataclass that mirrors the policy configuration
and handles memory calculations for custom kv_cache_dtype values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── KV cache dtype string constants ────────────────────────────────────

KVQUANT_DTYPE_K8 = "kvquant_k8"
KVQUANT_DTYPE_K4 = "kvquant_k4"
KVQUANT_DTYPE_K3 = "kvquant_k3"
KVQUANT_DTYPE_K2 = "kvquant_k2"
KVQUANT_DTYPE_K1 = "kvquant_k1"

ALL_KVQUANT_DTYPES = frozenset({
    KVQUANT_DTYPE_K8,
    KVQUANT_DTYPE_K4,
    KVQUANT_DTYPE_K3,
    KVQUANT_DTYPE_K2,
    KVQUANT_DTYPE_K1,
})

# Map from dtype string → policy nbits
KVQUANT_DTYPE_TO_NBITS: dict[str, int] = {
    KVQUANT_DTYPE_K8: 8,
    KVQUANT_DTYPE_K4: 4,
    KVQUANT_DTYPE_K3: 3,
    KVQUANT_DTYPE_K2: 2,
    KVQUANT_DTYPE_K1: 1,
}

# Map from policy nbits → dtype string
KVQUANT_NBITS_TO_DTYPE: dict[int, str] = {v: k for k, v in KVQUANT_DTYPE_TO_NBITS.items()}


def packed_kv_page_bytes(
    *, nbits: int, num_kv_heads: int, block_size: int, head_dim: int
) -> int:
    """Packed K+V bytes for one page using per-head byte rounding."""
    packed_head_dim = (head_dim * nbits + 7) // 8
    return 2 * num_kv_heads * block_size * packed_head_dim


@dataclass
class KVQuantConfig:
    """Configuration for a KVQuant custom kv_cache_dtype.

    This is the single configuration object that the vLLM plugin uses
    to determine the quantization policy, bit width, and memory layout.
    """

    policy_name: str = "document_naive"
    """Quantization policy name (must be registered in kvquant.policies)."""

    nbits: int = 3
    """Bit width. One of {8, 4, 3, 2, 1}."""

    scale_strategy: str = "per_token_head"
    """Scale layout for the vLLM fork path. Default: one K/V scale per token and KV head."""

    scale_dtype_bytes: int = 4
    """Bytes per scale value in the vLLM fork auxiliary buffer.

    The current vLLM per-token-head scale cache stores float32 scales inline
    after each packed head.
    """

    block_size: int = 16
    """KV cache page/block size in tokens (vLLM default)."""

    head_dim: int = 64
    """Head dimension (set at runtime from model config)."""

    num_kv_heads: int = 4
    """Number of KV heads (set at runtime from model config)."""

    metadata: dict[str, Any] = field(default_factory=dict)

    # ── derived properties ──

    @property
    def kv_cache_dtype(self) -> str:
        """The vLLM kv_cache_dtype string for this config."""
        if self.nbits not in KVQUANT_NBITS_TO_DTYPE:
            raise ValueError(f"unsupported nbits: {self.nbits}")
        return KVQUANT_NBITS_TO_DTYPE[self.nbits]

    @property
    def bytes_per_element(self) -> float:
        """Average bytes per scalar element in packed storage."""
        return max(self.nbits / 8, 0.125)

    @property
    def packed_page_size_bytes(self) -> int:
        """Packed K/V bytes for one KV cache page, excluding scale buffers.

        Each page stores K and V tensors of shape [num_kv_heads, block_size, head_dim].
        """
        return packed_kv_page_bytes(
            nbits=self.nbits,
            num_kv_heads=self.num_kv_heads,
            block_size=self.block_size,
            head_dim=self.head_dim,
        )

    @property
    def scale_page_size_bytes(self) -> int:
        """Auxiliary scale bytes for one page."""
        if self.scale_strategy in {"none", "global"}:
            return 2 * self.scale_dtype_bytes
        if self.scale_strategy == "per_head":
            return 2 * self.num_kv_heads * self.scale_dtype_bytes
        if self.scale_strategy in {"per_token", "per_token_head", "per_token_per_head"}:
            return 2 * self.num_kv_heads * self.block_size * self.scale_dtype_bytes
        raise ValueError(f"unsupported scale_strategy: {self.scale_strategy}")

    @property
    def page_size_bytes(self) -> int:
        """Total bytes for one KV cache page, including vLLM fork scale buffers."""
        return self.packed_page_size_bytes + self.scale_page_size_bytes

    def memory_layout(self) -> dict[str, Any]:
        """Return the page layout expected by the vLLM fork patch."""
        return {
            "kv_cache_dtype": self.kv_cache_dtype,
            "nbits": self.nbits,
            "packed_page_size_bytes": self.packed_page_size_bytes,
            "scale_page_size_bytes": self.scale_page_size_bytes,
            "page_size_bytes": self.page_size_bytes,
            "scale_strategy": self.scale_strategy,
            "scale_dtype_bytes": self.scale_dtype_bytes,
            "block_size": self.block_size,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }
