"""Scaffold for a future KVQuant packed attention backend for vLLM V1.

This module is not registered with vLLM yet. It only keeps the intended
call shape for future work.

Reference: vllm/v1/attention/backends/ (FlashAttn, FlashInfer, Triton, KVarN, etc.)
"""

from __future__ import annotations


# ── Backend metadata ────────────────────────────────────────────────

BACKEND_NAME = "kvquant_triton_packed"

# The kv_cache dtypes this backend can handle.
SUPPORTED_KV_CACHE_DTYPES = frozenset({
    "kvquant_k8",
    "kvquant_k4",
    "kvquant_k3",
    "kvquant_k2",
    "kvquant_k1",
})


def get_backend_name() -> str:
    return BACKEND_NAME


def supported_dtypes() -> frozenset[str]:
    return SUPPORTED_KV_CACHE_DTYPES


# ── Attention entry point ────────────────────────────────────────────

def forward(
    query: object,
    key_packed: object,
    value_packed: object,
    key_min: object,
    key_scale: object,
    value_min: object,
    value_scale: object,
    nbits: int,
    scale: float,
    *,
    block_m: int = 64,
) -> object:
    """Run packed attention using the KVQuant Triton kernel.

    This is the main entry point called by vLLM's attention layer.
    It delegates to the Triton kernel in ``kvquant.triton_kernel``.

    Parameters
    ----------
    query: [batch, num_heads, head_dim] float16
    key_packed: [bytes] uint8 bit-packed key bytes
    value_packed: [bytes] uint8 bit-packed value bytes
    key_min, key_scale, value_min, value_scale: scalar tensors
    nbits: bit width (8, 4, 3, 2, or 1)
    scale: 1/sqrt(head_dim)
    block_m: Triton tile size
    """
    from kvquant.triton_kernel.packed_attention import run_packed_attention

    return run_packed_attention(
        query=query,
        key_packed=key_packed,
        value_packed=value_packed,
        key_min=key_min,
        key_scale=key_scale,
        value_min=value_min,
        value_scale=value_scale,
        nbits=nbits,
        scale=scale,
        block_m=block_m,
    )
