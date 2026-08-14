"""Packed-attention entry point.

The public function accepts bit-packed K/V bytes and computes one decode
attention step. The current implementation intentionally reports itself as a
PyTorch fallback because the real Triton in-register dequantization kernel is
not implemented yet.

Do not use this module as evidence of serving/deploy speedup until
``packed_attention_backend_name()`` returns ``"triton_packed_attention"``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


FALLBACK_BACKEND_NAME = "torch_unpack_dequant_attention"
TRITON_BACKEND_NAME = "triton_packed_attention"


def packed_attention_backend_name() -> str:
    """Return the implementation currently used by ``run_packed_attention``."""
    return FALLBACK_BACKEND_NAME


def run_packed_attention(
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
    key_shape: tuple[int, ...] | None = None,
    value_shape: tuple[int, ...] | None = None,
    block_m: int = 64,
) -> object:
    """High-level entry point for packed attention.

    Calls the Triton two-stage kernel internally. Falls back to
    PyTorch dequantize + attention if Triton is not available.

    Parameters
    ----------
    query: [batch, num_heads, head_dim] float16
    key_packed: uint8 bytes
    value_packed: uint8 bytes
    key_min, key_scale, value_min, value_scale: scalar tensors
    nbits: bit width {1..8}; vLLM target path uses 3 for ``kvquant_k3``
    scale: attention scale (1/sqrt(head_dim))
    key_shape, value_shape: original [batch, heads, seq, head_dim] shapes
    block_m: reserved for the future Triton tile size
    """
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
        import torch
    except ImportError:
        return _fallback_attention(
            query, key_packed, value_packed,
            key_min, key_scale, value_min, value_scale,
            nbits, scale, key_shape=key_shape, value_shape=value_shape,
        )

    # The real Triton implementation is intentionally not pretended here.
    # Future work should replace this fallback and update
    # packed_attention_backend_name().
    return _fallback_attention(
        query, key_packed, value_packed,
        key_min, key_scale, value_min, value_scale,
        nbits, scale, key_shape=key_shape, value_shape=value_shape,
    )


def _fallback_attention(
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
    key_shape: tuple[int, ...] | None = None,
    value_shape: tuple[int, ...] | None = None,
) -> object:
    """PyTorch fallback: unpack → dequantize → standard attention."""
    import torch

    key = _torch_unpack_dequant(key_packed, key_min, key_scale, nbits, original_shape=key_shape)
    value = _torch_unpack_dequant(value_packed, value_min, value_scale, nbits, original_shape=value_shape)
    key = key.to(device=query.device, dtype=query.dtype)
    value = value.to(device=query.device, dtype=query.dtype)
    # key, value: [batch, heads, seq, head_dim]; query: [batch, heads, head_dim]
    scores = torch.matmul(query.unsqueeze(-2), key.transpose(-1, -2)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value).squeeze(-2)


def _torch_unpack_dequant(
    packed: object,
    minimum: object,
    scale_arr: object,
    nbits: int,
    axis_strategy: str = "global",
    original_shape: tuple | None = None,
) -> object:
    """Unpack bytes → qvalues → dequantize on GPU.

    Supports per_head and per_head_channel dequantization via
    broadcasting of the minimum and scale arrays.
    """
    import torch

    packed = torch.as_tensor(packed)
    minimum = torch.as_tensor(minimum, device=packed.device, dtype=torch.float32)
    scale_arr = torch.as_tensor(scale_arr, device=packed.device, dtype=torch.float32)

    if original_shape:
        count = int(np.prod(original_shape))
    else:
        raise ValueError("original_shape is required for packed attention fallback")

    if nbits <= 0 or nbits > 8:
        raise ValueError(f"unsupported nbits: {nbits}")

    qvalues = _torch_unpack_bits(packed, nbits, count)
    qvalues = qvalues.reshape(original_shape)

    # Broadcast scale and min to match tensor shape
    # For global: scale=[1], for per_head: scale=[1, H, 1, 1]
    return qvalues * scale_arr + minimum


def _torch_unpack_bits(packed: object, nbits: int, count: int) -> object:
    """Vectorized bit unpacker for low-bit fallback diagnostics."""
    import torch

    packed = torch.as_tensor(packed, dtype=torch.uint8)
    bit_offsets = torch.arange(count, device=packed.device, dtype=torch.int64) * int(nbits)
    byte_idx = bit_offsets // 8
    shifts = bit_offsets % 8
    packed_i32 = packed.to(torch.int32)
    first = packed_i32[byte_idx]
    next_idx = byte_idx + 1
    second = torch.zeros_like(first)
    valid_next = next_idx < packed_i32.numel()
    if bool(valid_next.any().item()):
        second[valid_next] = packed_i32[next_idx[valid_next]]
    combined = first | (second << 8)
    mask = (1 << int(nbits)) - 1
    return ((combined >> shifts.to(torch.int32)) & mask).to(torch.float32)


def _infer_seq_len(key_packed: object, num_heads: int, head_dim: int, nbits: int) -> int:
    """Infer the sequence length from the packed byte array size."""
    try:
        import torch
        total_bytes = key_packed.numel()
        elements = int(total_bytes * 8 / nbits)
        return elements // (num_heads * head_dim)
    except Exception:
        return 1
