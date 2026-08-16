# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symmetric INT3 per-token-head KV-cache helpers for the ``kvquant_k3`` research dtype.

Write path (reshape + quantize):
    ``reshape_and_cache_kvquant_k3`` — PyTorch reference, validated for paper correctness.

Read path (paged attention decode):
    ``unified_attention_kvquant_k3`` — Triton fused decode kernel that unpacks 3-bit
    codes from the packed cache at the byte level, dequantizes K/V with per-(token,
    head) float32 scales, and runs split-Q paged attention.  Reuses the epilogue +
    segment-reduction infrastructure shared with the INT4 kernel.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_alibi_to_score,
    apply_softcap,
    cdiv_fn,
    compute_kv_seq_mask,
    compute_tile_loop_bounds,
    init_softmax_M,
    load_qq_bias_tile,
    resolve_seq_and_query_len,
    softmax_step,
    store_segm_reduce_scalars,
)
from vllm.v1.attention.ops.triton_unified_attention import reduce_segments

# ---------------------------------------------------------------------------
# INT3 pack / unpack helpers (Python reference)
# ---------------------------------------------------------------------------

KVQUANT_K3_BITS = 3
KVQUANT_K3_CODE_MIN = 0
KVQUANT_K3_CODE_MAX = (1 << KVQUANT_K3_BITS) - 1  # 7
KVQUANT_K3_SIGNED_ZERO = 4      # codes 0..7 → signed −4..3
KVQUANT_K3_SIGNED_MIN = -4
KVQUANT_K3_SIGNED_MAX = 3


def _kvquant_k3_packed_dim(head_size: int) -> int:
    """Packed byte count for *head_size* INT3 values."""
    return (head_size * KVQUANT_K3_BITS + 7) // 8


def pack_int3_values(values: torch.Tensor) -> torch.Tensor:
    """Pack unsigned INT3 codes along the last dimension.

    Codes are packed little-endian by value: code ``i`` starts at bit
    ``3 * i`` of the output byte stream.  Eight INT3 codes therefore occupy
    exactly three bytes.
    """
    if values.ndim == 0:
        raise ValueError("pack_int3_values expects at least one dimension")
    if values.numel() == 0:
        output_shape = (*values.shape[:-1], 0)
        return torch.empty(output_shape, dtype=torch.uint8, device=values.device)

    codes = values.to(torch.int64)
    if bool(((codes < KVQUANT_K3_CODE_MIN) | (codes > KVQUANT_K3_CODE_MAX)).any()):
        raise ValueError("kvquant_k3 codes must be in the inclusive range [0, 7]")

    value_count = values.shape[-1]
    packed_count = _kvquant_k3_packed_dim(value_count)
    flat = codes.reshape(-1, value_count)
    packed = torch.zeros(
        flat.shape[0], packed_count, dtype=torch.int64, device=values.device
    )

    value_offsets = torch.arange(value_count, dtype=torch.int64, device=values.device)
    for bit in range(KVQUANT_K3_BITS):
        bit_positions = value_offsets * KVQUANT_K3_BITS + bit
        byte_offsets = bit_positions // 8
        bit_offsets = bit_positions % 8
        bit_values = (flat >> bit) & 1
        packed.scatter_add_(
            1,
            byte_offsets.expand(flat.shape[0], -1),
            bit_values << bit_offsets,
        )

    return packed.to(torch.uint8).reshape(*values.shape[:-1], packed_count)


def unpack_int3_values(packed: torch.Tensor, value_count: int) -> torch.Tensor:
    """Unpack unsigned INT3 codes from the last dimension."""
    if packed.ndim == 0:
        raise ValueError("unpack_int3_values expects at least one dimension")
    if value_count < 0:
        raise ValueError("value_count must be non-negative")

    expected_packed_count = _kvquant_k3_packed_dim(value_count)
    if packed.shape[-1] < expected_packed_count:
        raise ValueError(
            "packed tensor last dimension is too small for "
            f"{value_count} INT3 values: expected at least "
            f"{expected_packed_count}, got {packed.shape[-1]}"
        )
    if value_count == 0:
        output_shape = (*packed.shape[:-1], 0)
        return torch.empty(output_shape, dtype=torch.uint8, device=packed.device)

    packed_i64 = packed[..., :expected_packed_count].to(torch.int64)
    flat = packed_i64.reshape(-1, expected_packed_count)
    values = torch.zeros(
        flat.shape[0], value_count, dtype=torch.int64, device=packed.device
    )

    value_offsets = torch.arange(value_count, dtype=torch.int64, device=packed.device)
    for bit in range(KVQUANT_K3_BITS):
        bit_positions = value_offsets * KVQUANT_K3_BITS + bit
        byte_offsets = bit_positions // 8
        bit_offsets = bit_positions % 8
        source_bytes = flat.gather(1, byte_offsets.expand(flat.shape[0], -1))
        bit_values = (source_bytes >> bit_offsets) & 1
        values |= bit_values << bit

    return values.to(torch.uint8).reshape(*packed.shape[:-1], value_count)


def quantize_int3_per_token_head_ref(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference signed INT3 quantization with one scale per token/head.

    The packed cache stores unsigned codes ``0..7``.  Semantically those codes
    represent signed integers ``-4..3`` after subtracting
    ``KVQUANT_K3_SIGNED_ZERO``.  Only a scale tensor is stored, matching vLLM's
    existing per-token-head scale cache layout.
    """
    if tensor.ndim != 3:
        raise ValueError("expected tensor shape [num_tokens, num_heads, head_size]")

    absmax = tensor.float().abs().amax(dim=-1)
    scale = (absmax / float(KVQUANT_K3_SIGNED_MAX)).clamp(min=1e-6)
    signed = torch.round(tensor.float() / scale.unsqueeze(-1)).clamp(
        KVQUANT_K3_SIGNED_MIN, KVQUANT_K3_SIGNED_MAX
    )
    codes = (signed.to(torch.int16) + KVQUANT_K3_SIGNED_ZERO).to(torch.uint8)
    return pack_int3_values(codes), scale.float()


def dequantize_int3_per_token_head_ref(
    packed: torch.Tensor,
    scales: torch.Tensor,
    value_count: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference dequantization for tests and kernel bring-up."""
    codes = unpack_int3_values(packed, value_count).to(torch.int16)
    signed = codes - KVQUANT_K3_SIGNED_ZERO
    return (signed.float() * scales.float().unsqueeze(-1)).to(dtype)



# ---------------------------------------------------------------------------
# RABIT-2 correctness reference (K3/V2 + META8g64 + R4)
# ---------------------------------------------------------------------------

RABIT2_K_BITS = 3
RABIT2_V_BITS = 2
RABIT2_GROUP_SIZE = 32
RABIT2_METADATA_GROUP_SIZE = 64
RABIT2_RESIDUAL_TOKENS = 4


def _packed_dim(value_count: int, bits: int) -> int:
    if value_count < 0:
        raise ValueError("value_count must be non-negative")
    if bits not in (1, 2, 3, 4, 8):
        raise ValueError("bits must be one of 1, 2, 3, 4, 8")
    return (value_count * bits + 7) // 8


def pack_lowbit_values(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned low-bit codes along the last dimension.

    The bit stream is little-endian by value. This is the canonical Python
    reference used by the RABIT-2 correctness tests; deployment kernels must
    reproduce this byte layout exactly.
    """
    if values.ndim == 0:
        raise ValueError("pack_lowbit_values expects at least one dimension")
    if bits not in (1, 2, 3, 4, 8):
        raise ValueError("bits must be one of 1, 2, 3, 4, 8")
    if values.numel() == 0:
        return torch.empty(
            (*values.shape[:-1], 0), dtype=torch.uint8, device=values.device
        )

    codes = values.to(torch.int64)
    max_code = (1 << bits) - 1
    if bool(((codes < 0) | (codes > max_code)).any()):
        raise ValueError(f"{bits}-bit codes must be in the inclusive range [0, {max_code}]")
    if bits == 8:
        return codes.to(torch.uint8).contiguous()

    value_count = values.shape[-1]
    packed_count = _packed_dim(value_count, bits)
    flat = codes.reshape(-1, value_count)
    packed = torch.zeros(
        flat.shape[0], packed_count, dtype=torch.int64, device=values.device
    )
    value_offsets = torch.arange(value_count, dtype=torch.int64, device=values.device)
    for bit in range(bits):
        bit_positions = value_offsets * bits + bit
        byte_offsets = bit_positions // 8
        bit_offsets = bit_positions % 8
        bit_values = (flat >> bit) & 1
        packed.scatter_add_(
            1,
            byte_offsets.expand(flat.shape[0], -1),
            bit_values << bit_offsets,
        )
    return packed.to(torch.uint8).reshape(*values.shape[:-1], packed_count)


def unpack_lowbit_values(
    packed: torch.Tensor, value_count: int, bits: int
) -> torch.Tensor:
    """Inverse of :func:`pack_lowbit_values`."""
    if packed.ndim == 0:
        raise ValueError("unpack_lowbit_values expects at least one dimension")
    if value_count < 0:
        raise ValueError("value_count must be non-negative")
    if bits not in (1, 2, 3, 4, 8):
        raise ValueError("bits must be one of 1, 2, 3, 4, 8")
    expected = _packed_dim(value_count, bits)
    if packed.shape[-1] < expected:
        raise ValueError(
            f"packed last dimension is too small: expected at least {expected}, "
            f"got {packed.shape[-1]}"
        )
    if value_count == 0:
        return torch.empty(
            (*packed.shape[:-1], 0), dtype=torch.uint8, device=packed.device
        )
    if bits == 8:
        return packed[..., :value_count].to(torch.uint8).contiguous()

    flat = packed[..., :expected].to(torch.int64).reshape(-1, expected)
    values = torch.zeros(
        flat.shape[0], value_count, dtype=torch.int64, device=packed.device
    )
    value_offsets = torch.arange(value_count, dtype=torch.int64, device=packed.device)
    for bit in range(bits):
        bit_positions = value_offsets * bits + bit
        byte_offsets = bit_positions // 8
        bit_offsets = bit_positions % 8
        source = flat.gather(1, byte_offsets.expand(flat.shape[0], -1))
        values |= ((source >> bit_offsets) & 1) << bit
    return values.to(torch.uint8).reshape(*packed.shape[:-1], value_count)


def pack_int2_values(values: torch.Tensor) -> torch.Tensor:
    return pack_lowbit_values(values, 2)


def unpack_int2_values(packed: torch.Tensor, value_count: int) -> torch.Tensor:
    return unpack_lowbit_values(packed, value_count, 2)


def encode_metadata_uint8_group_ref(
    tensor: torch.Tensor,
    group_size: int = RABIT2_METADATA_GROUP_SIZE,
) -> dict[str, Any]:
    """Compress metadata exactly like the final quality reference.

    Metadata values are flattened, divided into groups of 64, quantized to
    UINT8 affine codes, and accompanied by BF16 group minima and scales.
    """
    if group_size < 8:
        raise ValueError("metadata group_size must be at least 8")
    data = tensor.detach().float().contiguous()
    original_shape = tuple(data.shape)
    flat = data.reshape(-1)
    if flat.numel() == 0:
        return {
            "meta_type": "uint8_group",
            "codes": torch.empty(0, group_size, dtype=torch.uint8, device=data.device),
            "min": torch.empty(0, 1, dtype=torch.bfloat16, device=data.device),
            "scale": torch.empty(0, 1, dtype=torch.bfloat16, device=data.device),
            "orig_shape": original_shape,
            "pad": 0,
            "group_size": int(group_size),
        }
    pad = (-flat.numel()) % group_size
    if pad:
        flat = torch.cat([flat, flat[-1:].expand(pad)], dim=0)
    grouped = flat.reshape(-1, group_size)
    meta_min = grouped.amin(dim=-1, keepdim=True)
    meta_max = grouped.amax(dim=-1, keepdim=True)
    meta_scale = (meta_max - meta_min) / 255.0
    meta_scale = torch.where(
        meta_scale.abs() < 1e-12, torch.ones_like(meta_scale), meta_scale
    )
    codes = torch.round((grouped - meta_min) / meta_scale).clamp(0, 255)
    return {
        "meta_type": "uint8_group",
        "codes": codes.to(torch.uint8).contiguous(),
        "min": meta_min.to(torch.bfloat16).contiguous(),
        "scale": meta_scale.to(torch.bfloat16).contiguous(),
        "orig_shape": original_shape,
        "pad": int(pad),
        "group_size": int(group_size),
    }


def decode_metadata_uint8_group_ref(metadata: dict[str, Any]) -> torch.Tensor:
    if metadata.get("meta_type") != "uint8_group":
        raise ValueError("expected uint8_group metadata")
    if metadata["codes"].numel() == 0:
        return torch.empty(
            metadata["orig_shape"], dtype=torch.float32, device=metadata["codes"].device
        )
    values = (
        metadata["codes"].float() * metadata["scale"].float()
        + metadata["min"].float()
    ).reshape(-1)
    pad = int(metadata.get("pad", 0))
    if pad:
        values = values[:-pad]
    return values.reshape(metadata["orig_shape"])


def metadata_storage_bytes_ref(metadata: dict[str, Any]) -> int:
    return sum(
        int(metadata[name].numel() * metadata[name].element_size())
        for name in ("codes", "min", "scale")
    )


def quantize_k3_sequence_affine_ref(
    tensor: torch.Tensor,
    seq_group_size: int = RABIT2_GROUP_SIZE,
    metadata_group_size: int = RABIT2_METADATA_GROUP_SIZE,
) -> dict[str, Any]:
    """K3 sequence-axis/channel-wise affine quantization.

    Input shape is ``[tokens, kv_heads, head_dim]``. Each metadata pair is
    shared by 32 adjacent tokens for one head and one channel, matching the
    final quality operating point.
    """
    if tensor.ndim != 3:
        raise ValueError("expected K shape [tokens, num_heads, head_dim]")
    tokens, heads, head_dim = tensor.shape
    if tokens == 0:
        raise ValueError("K tensor must contain at least one token")
    if seq_group_size <= 0:
        raise ValueError("seq_group_size must be positive")

    x = tensor.detach().float().permute(1, 0, 2).unsqueeze(0)
    pad_seq = (-tokens) % seq_group_size
    if pad_seq:
        # F.pad with a 4-D tensor pads the final two dimensions.
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_seq))
    padded_tokens = tokens + pad_seq
    grouped = x.reshape(1, heads, padded_tokens // seq_group_size, seq_group_size, head_dim)
    q_min = grouped.amin(dim=3, keepdim=True)
    q_max = grouped.amax(dim=3, keepdim=True)
    scale = (q_max - q_min) / 7.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    codes_grouped = torch.round((grouped - q_min) / scale).clamp(0, 7).to(torch.uint8)
    codes = (
        codes_grouped.reshape(1, heads, padded_tokens, head_dim)
        .squeeze(0)
        .permute(1, 0, 2)
        .contiguous()
    )
    return {
        "type": "rabit2_k3_seq_affine",
        "bits": 3,
        "packed": pack_int3_values(codes),
        "original_shape": (tokens, heads, head_dim),
        "padded_tokens": int(padded_tokens),
        "pad_seq": int(pad_seq),
        "seq_group_size": int(seq_group_size),
        "min": encode_metadata_uint8_group_ref(q_min, metadata_group_size),
        "scale": encode_metadata_uint8_group_ref(scale, metadata_group_size),
    }


def dequantize_k3_sequence_affine_ref(
    state: dict[str, Any], *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    tokens, heads, head_dim = state["original_shape"]
    padded_tokens = int(state["padded_tokens"])
    group_size = int(state["seq_group_size"])
    codes = unpack_int3_values(state["packed"], head_dim)
    codes = codes.permute(1, 0, 2).unsqueeze(0)
    grouped = codes.reshape(1, heads, padded_tokens // group_size, group_size, head_dim)
    q_min = decode_metadata_uint8_group_ref(state["min"])
    scale = decode_metadata_uint8_group_ref(state["scale"])
    values = grouped.float() * scale + q_min
    values = values.reshape(1, heads, padded_tokens, head_dim)
    values = values[:, :, :tokens, :].squeeze(0).permute(1, 0, 2).contiguous()
    return values.to(dtype)


def quantize_v2_group_affine_ref(
    tensor: torch.Tensor,
    group_size: int = RABIT2_GROUP_SIZE,
    metadata_group_size: int = RABIT2_METADATA_GROUP_SIZE,
) -> dict[str, Any]:
    """V2 token-wise, last-dimension affine quantization with G32."""
    if tensor.ndim != 3:
        raise ValueError("expected V shape [tokens, num_heads, head_dim]")
    tokens, heads, head_dim = tensor.shape
    if tokens == 0:
        raise ValueError("V tensor must contain at least one token")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    x = tensor.detach().float()
    pad_dim = (-head_dim) % group_size
    if pad_dim:
        x = torch.nn.functional.pad(x, (0, pad_dim))
    padded_dim = head_dim + pad_dim
    grouped = x.reshape(tokens, heads, padded_dim // group_size, group_size)
    q_min = grouped.amin(dim=-1, keepdim=True)
    q_max = grouped.amax(dim=-1, keepdim=True)
    scale = (q_max - q_min) / 3.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    codes = torch.round((grouped - q_min) / scale).clamp(0, 3).to(torch.uint8)
    codes = codes.reshape(tokens, heads, padded_dim)
    return {
        "type": "rabit2_v2_group_affine",
        "bits": 2,
        "packed": pack_int2_values(codes),
        "original_shape": (tokens, heads, head_dim),
        "padded_dim": int(padded_dim),
        "pad_dim": int(pad_dim),
        "group_size": int(group_size),
        "min": encode_metadata_uint8_group_ref(q_min, metadata_group_size),
        "scale": encode_metadata_uint8_group_ref(scale, metadata_group_size),
    }


def dequantize_v2_group_affine_ref(
    state: dict[str, Any], *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    tokens, heads, head_dim = state["original_shape"]
    padded_dim = int(state["padded_dim"])
    group_size = int(state["group_size"])
    codes = unpack_int2_values(state["packed"], padded_dim)
    grouped = codes.reshape(tokens, heads, padded_dim // group_size, group_size)
    q_min = decode_metadata_uint8_group_ref(state["min"])
    scale = decode_metadata_uint8_group_ref(state["scale"])
    values = (grouped.float() * scale + q_min).reshape(tokens, heads, padded_dim)
    return values[..., :head_dim].to(dtype)


def quantize_rabit2_kv_ref(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    residual_tokens: int = RABIT2_RESIDUAL_TOKENS,
    group_size: int = RABIT2_GROUP_SIZE,
    metadata_group_size: int = RABIT2_METADATA_GROUP_SIZE,
) -> dict[str, Any]:
    """Build the correctness-first final RABIT-2 state.

    The newest four tokens remain BF16. Older K uses K3 sequence-affine G32;
    older V uses V2 token/dimension-affine G32. Metadata uses UINT8 G64 with
    BF16 second-level minima/scales.
    """
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key and value must share shape [tokens, heads, head_dim]")
    if residual_tokens < 0:
        raise ValueError("residual_tokens must be non-negative")
    tokens = key.shape[0]
    recent_count = min(tokens, residual_tokens)
    old_count = tokens - recent_count
    old_k = (
        quantize_k3_sequence_affine_ref(
            key[:old_count], group_size, metadata_group_size
        )
        if old_count
        else None
    )
    old_v = (
        quantize_v2_group_affine_ref(
            value[:old_count], group_size, metadata_group_size
        )
        if old_count
        else None
    )
    return {
        "type": "rabit2_kv",
        "original_shape": tuple(key.shape),
        "residual_tokens": int(residual_tokens),
        "old_count": int(old_count),
        "old_k": old_k,
        "old_v": old_v,
        "recent_k": key[old_count:].detach().to(torch.bfloat16).contiguous(),
        "recent_v": value[old_count:].detach().to(torch.bfloat16).contiguous(),
    }


def dequantize_rabit2_kv_ref(
    state: dict[str, Any], *, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    old_count = int(state["old_count"])
    if old_count:
        old_k = dequantize_k3_sequence_affine_ref(state["old_k"], dtype=dtype)
        old_v = dequantize_v2_group_affine_ref(state["old_v"], dtype=dtype)
    else:
        shape = (0, *state["original_shape"][1:])
        device = state["recent_k"].device
        old_k = torch.empty(shape, dtype=dtype, device=device)
        old_v = torch.empty(shape, dtype=dtype, device=device)
    key = torch.cat([old_k, state["recent_k"].to(dtype)], dim=0)
    value = torch.cat([old_v, state["recent_v"].to(dtype)], dim=0)
    return key, value


def update_rabit2_kv_ref(
    state: dict[str, Any] | None,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
) -> dict[str, Any]:
    """Correctness reference for token ageing.

    This intentionally rebuilds the compact prefix after each update. It is
    the oracle for the later paged/Triton implementation, not a serving-speed
    path.
    """
    if new_key.shape != new_value.shape or new_key.ndim != 3:
        raise ValueError("new_key/new_value must share [tokens, heads, head_dim]")
    if state is None:
        full_k, full_v = new_key, new_value
        residual = RABIT2_RESIDUAL_TOKENS
    else:
        old_k, old_v = dequantize_rabit2_kv_ref(state, dtype=new_key.dtype)
        full_k = torch.cat([old_k, new_key], dim=0)
        full_v = torch.cat([old_v, new_value], dim=0)
        residual = int(state["residual_tokens"])
    return quantize_rabit2_kv_ref(full_k, full_v, residual_tokens=residual)


def _quantized_tensor_breakdown_ref(state: dict[str, Any]) -> dict[str, int]:
    return {
        "payload": int(state["packed"].numel() * state["packed"].element_size()),
        "metadata": metadata_storage_bytes_ref(state["min"])
        + metadata_storage_bytes_ref(state["scale"]),
        "residual": 0,
    }


def rabit2_state_storage_bytes_ref(state: dict[str, Any]) -> dict[str, int]:
    payload = 0
    metadata = 0
    if state["old_k"] is not None:
        k = _quantized_tensor_breakdown_ref(state["old_k"])
        v = _quantized_tensor_breakdown_ref(state["old_v"])
        payload = k["payload"] + v["payload"]
        metadata = k["metadata"] + v["metadata"]
    residual = sum(
        int(state[name].numel() * state[name].element_size())
        for name in ("recent_k", "recent_v")
    )
    total = payload + metadata + residual
    return {
        "payload": payload,
        "metadata": metadata,
        "residual": residual,
        "total": total,
    }


def attention_rabit2_ref(
    q: torch.Tensor,
    state: dict[str, Any],
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Decode-attention oracle over a RABIT-2 state.

    ``q`` has shape ``[queries, query_heads, head_dim]`` and attends to every
    cached token. GQA is supported when query_heads is a multiple of KV heads.
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [queries, query_heads, head_dim]")
    key, value = dequantize_rabit2_kv_ref(state, dtype=torch.float32)
    num_query_heads = q.shape[1]
    num_kv_heads = key.shape[1]
    if num_query_heads % num_kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    repeats = num_query_heads // num_kv_heads
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    scores = torch.einsum("qhd,thd->qht", q.float(), key) * float(scale)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("qht,thd->qhd", probabilities, value).to(q.dtype)


# ---------------------------------------------------------------------------
# Triton INT3 decode attention kernel
# ---------------------------------------------------------------------------


@triton.jit
def _attn_int3_kernel(
    # Output destinations.
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    scale,
    out_scale,
    softcap,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    qq_bias_stride_0: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    PACKED_HEAD_PADDED: tl.constexpr,  # next_pow2(ceil(head_size * 3 / 8))
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    mm_prefix_range_ptr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    IS_3D: tl.constexpr,
    # INT3-specific
    INT3_ZERO: tl.constexpr,  # 4
    INT3_MASK: tl.constexpr,  # 0x7
):
    """Split-Q paged attention over packed INT3-per-token-head KV cache.

    Symmetric INT3: unsigned codes ``0..7`` represent signed values
    ``code − INT3_ZERO`` (range ``-4..3``).  A float32 per-(token, head)
    scale is stored alongside the packed bytes; no zero-point, no RHT.
    """
    # ---- Shared prologue (same as INT4 kernel) ----------------------------
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # ---- Load Q (unpadded HEAD_SIZE — matches dequantized K dim) ----------
    offs_d = tl.arange(0, HEAD_SIZE)
    q_base = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
    )
    q_mask = query_mask_0[:, None] & query_mask_1[:, None]
    Q = tl.load(
        query_ptr + q_base + offs_d[None, :],
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)  # [BLOCK_M, HEAD_SIZE]

    block_table_offset = seq_idx * block_table_stride

    # ---- Online softmax state ---------------------------------------------
    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
    )

    # ---- INT3 byte-index vectors (loop-invariant) -------------------------
    # For dimension d, the 3-bit code starts at bit 3*d, spanning up to
    # bits 3*d+2, which may cross a byte boundary.  We load the two bytes
    # at byte_indices[d] and byte_indices[d]+1, combine into a 16-bit word,
    # then shift+mask to extract the 3-bit code.
    d_range = tl.arange(0, HEAD_SIZE)
    bit_pos = d_range * 3
    byte_idx = bit_pos // 8      # [HEAD_SIZE]
    bit_shift = bit_pos % 8      # [HEAD_SIZE]
    padded_offs = tl.arange(0, PACKED_HEAD_PADDED)

    # ---- Tile loop --------------------------------------------------------
    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        slot_in_blk = seq_offset % BLOCK_SIZE

        # --- K: load packed bytes → unpack → dequantize --------------------
        # K arranged as [HEAD_SIZE, TILE_SIZE]: each row is one head-dim
        # element loaded from the packed byte at byte_idx[dim].
        k_addr0 = (
            physical_block_idx[None, :] * stride_k_cache_0    # type: ignore[operator]
            + kv_head_idx * stride_k_cache_2
            + slot_in_blk[None, :] * stride_k_cache_1
            + byte_idx[:, None] * stride_k_cache_3
        )  # [HEAD_SIZE, TILE_SIZE]

        k_lo = tl.load(
            key_cache_ptr + k_addr0,
            mask=tile_mask[None, :],
            other=0,
        ).to(tl.int32)  # [HEAD_SIZE, TILE_SIZE]

        k_hi = tl.load(
            key_cache_ptr + k_addr0 + stride_k_cache_3,
            mask=tile_mask[None, :],
            other=0,
        ).to(tl.int32)  # [HEAD_SIZE, TILE_SIZE]

        # Combine lo/hi into a 16-bit word, shift, mask → 3-bit code
        K_word = k_lo | (k_hi << 8)            # [HEAD_SIZE, TILE_SIZE]
        K_codes = (K_word >> bit_shift[:, None]) & INT3_MASK  # [HEAD_SIZE, TILE_SIZE]
        K_signed = K_codes.to(tl.float32) - INT3_ZERO * 1.0
        # [HEAD_SIZE, TILE_SIZE]

        # --- K scales ------------------------------------------------------
        ks_idx = (
            physical_block_idx * stride_ks_blk
            + slot_in_blk * stride_ks_slot
            + kv_head_idx * stride_ks_head
        )
        k_scales = tl.load(k_scale_cache_ptr + ks_idx, mask=tile_mask, other=1.0)
        # [TILE_SIZE]

        # --- Scores --------------------------------------------------------
        # Q: [BLOCK_M, HEAD_SIZE], K_signed: [HEAD_SIZE, TILE_SIZE]
        # raw_dot: [BLOCK_M, TILE_SIZE]
        raw_dot = tl.dot(Q, K_signed)
        S = k_scales[None, :] * scale * raw_dot
        # [BLOCK_M, TILE_SIZE]

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        # Mask invalid positions
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            seq_len,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
        )
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        # Online softmax
        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]  # [BLOCK_M, HEAD_SIZE]

        # --- V: load packed bytes → unpack → dequantize --------------------
        # V arranged as [TILE_SIZE, HEAD_SIZE]: each column is one head-dim
        # element loaded from the packed byte at byte_idx[dim].
        v_addr0 = (
            physical_block_idx[:, None] * stride_v_cache_0    # type: ignore[operator]
            + kv_head_idx * stride_v_cache_2
            + slot_in_blk[:, None] * stride_v_cache_1
            + byte_idx[None, :] * stride_v_cache_3
        )  # [TILE_SIZE, HEAD_SIZE]

        v_lo = tl.load(
            value_cache_ptr + v_addr0,
            mask=tile_mask[:, None],
            other=0,
        ).to(tl.int32)  # [TILE_SIZE, HEAD_SIZE]

        v_hi = tl.load(
            value_cache_ptr + v_addr0 + stride_v_cache_3,
            mask=tile_mask[:, None],
            other=0,
        ).to(tl.int32)  # [TILE_SIZE, HEAD_SIZE]

        V_word = v_lo | (v_hi << 8)            # [TILE_SIZE, HEAD_SIZE]
        V_codes = (V_word >> bit_shift[None, :]) & INT3_MASK  # [TILE_SIZE, HEAD_SIZE]
        V_signed = V_codes.to(tl.float32) - INT3_ZERO * 1.0
        # [TILE_SIZE, HEAD_SIZE]

        # --- V scales ------------------------------------------------------
        vs_idx = (
            physical_block_idx * stride_vs_blk
            + slot_in_blk * stride_vs_slot
            + kv_head_idx * stride_vs_head
        )
        v_scales = tl.load(v_scale_cache_ptr + vs_idx, mask=tile_mask, other=1.0)
        # [TILE_SIZE]

        # Fuse V scale into attention weights
        P_v = (P * v_scales[None, :]).to(tl.float32)  # [BLOCK_M, TILE_SIZE]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            sw_mask = (context_len + qpos_lo - seq_offset) < SLIDING_WINDOW
            V_signed = tl.where(sw_mask[:, None], V_signed, 0.0)

        # acc: [BLOCK_M, HEAD_SIZE], P_v: [BLOCK_M, TILE_SIZE],
        # V_signed: [TILE_SIZE, HEAD_SIZE]
        acc += tl.dot(P_v, V_signed)

    # ---- Epilogue ---------------------------------------------------------
    # acc: [BLOCK_M, HEAD_SIZE]; output has padded stride HEAD_SIZE_PADDED.
    out_mask = query_mask_0[:, None] & query_mask_1[:, None]
    offs_do = tl.arange(0, HEAD_SIZE_PADDED)
    do_mask = tl.where(offs_do < HEAD_SIZE, 1, 0).to(tl.int1)
    if IS_3D:
        segm_base = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + segm_idx * HEAD_SIZE_PADDED
        )
        tl.store(
            segm_output_ptr + segm_base + offs_do[None, :],
            acc,
            mask=do_mask[None, :] & out_mask,
        )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        if USE_FP8:
            out_s = tl.load(out_scale)
            acc = tl.clamp(acc * out_s, float("-inf"), float("inf"))
        out_base = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
        )
        tl.store(
            output_ptr + out_base + offs_do[None, :],
            acc,
            mask=do_mask[None, :] & out_mask,
        )


# ---------------------------------------------------------------------------
# Launcher (mirrors INT4's _launch_packed_attn)
# ---------------------------------------------------------------------------


def _launch_int3_attn(
    *,
    q,
    k_cache,
    v_cache,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    softmax_scale,
    window_size,
    block_table,
    softcap,
    sinks,
    alibi_slopes,
    use_alibi_sqrt,
    qq_bias,
    output_scale,
    mm_prefix_range,
    k_scale_cache,
    v_scale_cache,
    seq_threshold_3D,
    num_par_softmax_segments,
    softmax_segm_output,
    softmax_segm_max,
    softmax_segm_expsum,
):
    """Launch ``_attn_int3_kernel`` with the same grid/dispatch as the INT4 kernel."""
    import vllm.envs as envs
    from vllm.v1.attention.ops.triton_unified_attention import _get_tile_size

    is_batch_invariant = envs.VLLM_BATCH_INVARIANT

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        assert mm_prefix_range.ndim == 3, (
            f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
        )
        use_mm_prefix = True
        max_mm_ranges = mm_prefix_range.shape[1]

    block_size = v_cache.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    TILE_SIZE_PREFILL = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=True
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )

    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )

    segm_output_ptr = softmax_segm_output if use_3d else out
    segm_max_ptr = softmax_segm_max if use_3d else out
    segm_expsum_ptr = softmax_segm_expsum if use_3d else out
    num_segments = num_par_softmax_segments if use_3d else 1

    grid: tuple[Any, ...]
    if use_3d:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        tile_size = TILE_SIZE_DECODE
    else:
        grid = (total_num_q_blocks, num_kv_heads)
        tile_size = TILE_SIZE_PREFILL

    _attn_int3_kernel[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k_cache,
        value_cache_ptr=v_cache,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        scale=softmax_scale,
        out_scale=1 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        k_scale_cache_ptr=k_scale_cache,
        v_scale_cache_ptr=v_scale_cache,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if qq_bias is not None else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        PACKED_HEAD_PADDED=triton.next_power_of_2(
            _kvquant_k3_packed_dim(head_size)
        ),
        USE_ALIBI_SLOPES=alibi_slopes is not None,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=qq_bias is not None,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=(1 + window_size[0]),
        USE_MM_PREFIX=use_mm_prefix,
        MAX_MM_RANGES=max_mm_ranges,
        mm_prefix_range_ptr=mm_prefix_range,
        stride_k_cache_0=k_cache.stride(0),
        stride_k_cache_1=k_cache.stride(1),
        stride_k_cache_2=k_cache.stride(2),
        stride_k_cache_3=k_cache.stride(3),
        stride_v_cache_0=v_cache.stride(0),
        stride_v_cache_1=v_cache.stride(1),
        stride_v_cache_2=v_cache.stride(2),
        stride_v_cache_3=v_cache.stride(3),
        stride_ks_blk=k_scale_cache.stride(0),
        stride_ks_slot=k_scale_cache.stride(1),
        stride_ks_head=k_scale_cache.stride(2),
        stride_vs_blk=v_scale_cache.stride(0),
        stride_vs_slot=v_scale_cache.stride(1),
        stride_vs_head=v_scale_cache.stride(2),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=output_scale is not None,
        IS_3D=use_3d,
        INT3_ZERO=KVQUANT_K3_SIGNED_ZERO,
        INT3_MASK=KVQUANT_K3_CODE_MAX,
    )

    if use_3d:
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _write_packed_cache(
    tensor: torch.Tensor,
    cache: torch.Tensor,
    scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    packed, scales = quantize_int3_per_token_head_ref(tensor)
    packed_dim = packed.shape[-1]
    if cache.shape[-1] < packed_dim:
        raise ValueError(
            f"kvquant_k3 cache head dimension {cache.shape[-1]} is smaller "
            f"than packed INT3 dimension {packed_dim}"
        )

    block_size = cache.shape[1]
    slots = slot_mapping.to(torch.long)
    block_indices = slots // block_size
    block_offsets = slots % block_size

    cache[block_indices, block_offsets, :, :packed_dim] = packed.to(cache.dtype)
    scale_cache[block_indices, block_offsets, :] = scales


def reshape_and_cache_kvquant_k3(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
) -> None:
    """Quantize K/V to packed INT3 and write into the paged KV cache.

    This is a PyTorch reference path used to validate layout and small smoke
    tests.  It is not the deploy-speed kernel; serving latency claims must use
    a Triton/CUDA implementation of the same layout.
    """
    _write_packed_cache(key, key_cache, k_scale_cache, slot_mapping)
    _write_packed_cache(value, value_cache, v_scale_cache, slot_mapping)


def unified_attention_kvquant_k3(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    window_size: tuple[int, int],
    block_table: torch.Tensor,
    softcap: float,
    sinks: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    use_alibi_sqrt: bool,
    qq_bias: torch.Tensor | None,
    output_scale: torch.Tensor | None,
    mm_prefix_range: torch.Tensor | None,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    seq_threshold_3D: int | None = None,
    num_par_softmax_segments: int | None = None,
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
) -> None:
    """Paged attention over the packed INT3 KV cache, writing into *out*.

    Symmetric INT3 quantization (codes 0..7 → signed −4..3) with one
    float32 scale per (token, head).  No RHT — the quantized data is used
    directly.
    """
    _launch_int3_attn(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        window_size=window_size,
        block_table=block_table,
        softcap=softcap,
        sinks=sinks,
        alibi_slopes=alibi_slopes,
        use_alibi_sqrt=use_alibi_sqrt,
        qq_bias=qq_bias,
        output_scale=output_scale,
        mm_prefix_range=mm_prefix_range,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )
