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
