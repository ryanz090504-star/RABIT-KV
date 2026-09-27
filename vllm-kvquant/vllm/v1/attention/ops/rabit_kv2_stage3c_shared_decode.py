"""RABIT-KV2 Stage3C non-initial chunked prefill: selectable ``shared_decode`` path.

``tile32`` (rabit_kv2_stage3c_tile32.py, unchanged) batches the closed-page
scan into one launch per tile, but every program still decodes its page for a
single query, so each compressed page is decoded once per query. The Stage3C
component profile showed the closed-page GPU time barely moved (-3.6%).

``shared_decode`` removes the repeated decode. One program of
``_rabit2_shared_decode_closed_page_partial_kernel`` handles one
(closed page, GQA4 query-head group, block of QUERY_BLOCK chunk queries):
  1. it decodes the page's K3 / V2 payload and META8g64 metadata for its KV
     head ONCE (verbatim reference decode);
  2. it then loops over the block's queries and, for each one, runs the
     verbatim reference per-query arithmetic (scores, max, exp, sum, p @ V
     for the four query heads) against the SAME decoded ``kval`` / ``vval``;
  3. it writes each query's own (m, l, acc) partial in the reference layout.
The per-query reduction structure is unchanged; only the decode is hoisted out
of the per-query dimension. Page decodes per tile are therefore
closed * qgroups * ceil(n / QUERY_BLOCK) instead of closed * qgroups * n.

Everything else is exactly tile32's structure: the same tiles (runs of equal
visible closed-page count, at most 32 queries, ending at every page closure),
``Rabit2CausalChunkPlan.apply_step`` + the frozen tail emitter per query in
reference order, and tile32's (verbatim reference) reduce kernel. Nothing here
changes the K3/V2/G32/R4/META8g64 representation, page layout, persistent cache
bytes, the residual window, the decode (q_len == 1) path or the initial-prefill
path.

Selection: ``VLLM_RABIT2_STAGE3C_IMPL`` = ``reference`` (default) | ``tile32`` |
``shared_decode``; ``VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK`` = 4 | 8 (default) |
16 | 32 (compile-time kernel tuning parameter).
"""

from __future__ import annotations

import os

import torch

import vllm.v1.attention.ops.rabit_kv2 as _r
import vllm.v1.attention.ops.rabit_kv2_stage3c_tile32 as _t32
from vllm.triton_utils import tl, triton

STAGE3C_IMPL_ENV = _t32.STAGE3C_IMPL_ENV
STAGE3C_IMPLS = ("reference", "tile32", "shared_decode")
QUERY_BLOCK_ENV = "VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK"
QUERY_BLOCKS = (4, 8, 16, 32)
DEFAULT_QUERY_BLOCK = 8


def rabit2_stage3c_impl() -> str:
    """Selected Stage3C implementation; ``reference`` unless explicitly set."""
    impl = os.environ.get(STAGE3C_IMPL_ENV, "reference").strip().lower() or "reference"
    if impl not in STAGE3C_IMPLS:
        raise ValueError(f"{STAGE3C_IMPL_ENV}={impl!r} is not one of {STAGE3C_IMPLS}")
    return impl


def rabit2_shared_decode_query_block() -> int:
    raw = os.environ.get(QUERY_BLOCK_ENV, "").strip()
    qb = int(raw) if raw else DEFAULT_QUERY_BLOCK
    if qb not in QUERY_BLOCKS:
        raise ValueError(f"{QUERY_BLOCK_ENV}={raw!r} is not one of {QUERY_BLOCKS}")
    return qb


# The reference kernel's @triton.jit helper, bound under the same name the
# verbatim body uses.
_rabit2_u8pair_to_bf16_f32 = _r._rabit2_u8pair_to_bf16_f32


@triton.jit
def _rabit2_shared_decode_closed_page_partial_kernel(
    q_ptr,
    cache_ptr,
    block_table_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    decode_count_ptr,
    softmax_scale,
    n_queries,
    q_tile_stride,
    partial_tile_stride,
    PAGE_BYTES: tl.constexpr,
    K_PAYLOAD_OFFSET: tl.constexpr,
    V_PAYLOAD_OFFSET: tl.constexpr,
    K_MIN_OFFSET: tl.constexpr,
    K_SCALE_OFFSET: tl.constexpr,
    V_MIN_OFFSET: tl.constexpr,
    V_SCALE_OFFSET: tl.constexpr,
    K_PRIMARY_COUNT: tl.constexpr,
    K_META_GROUPS: tl.constexpr,
    V_PRIMARY_COUNT: tl.constexpr,
    V_META_GROUPS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    K_PACKED_DIM: tl.constexpr,
    V_PACKED_DIM: tl.constexpr,
    V_GROUPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    COUNT_DECODES: tl.constexpr,
):
    # --- BEGIN VERBATIM REFERENCE DECODE (_rabit2_stage4b3_gqa4_closed_page_partial_kernel) ---
    logical_page = tl.program_id(0)
    qgroup = tl.program_id(1)

    GQA_RATIO: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    Q_PER_PROGRAM: tl.constexpr = 4
    GROUPS_PER_KV: tl.constexpr = GQA_RATIO // Q_PER_PROGRAM

    kvh = qgroup // GROUPS_PER_KV
    subgroup = qgroup - kvh * GROUPS_PER_KV
    qh_base = kvh * GQA_RATIO + subgroup * Q_PER_PROGRAM

    physical_page = tl.load(block_table_ptr + logical_page).to(tl.int64)
    page = cache_ptr + physical_page * PAGE_BYTES

    t = tl.arange(0, 32)
    d = tl.arange(0, BLOCK_D)
    dmask = d < HEAD_SIZE

    # Decode packed K3 once for this KV head.
    k_bit = d * 3
    k_byte = k_bit // 8
    k_shift = k_bit % 8
    k_row = (t[:, None] * NUM_KV_HEADS + kvh) * K_PACKED_DIM
    k_lo = tl.load(
        page + K_PAYLOAD_OFFSET + k_row + k_byte[None, :],
        mask=dmask[None, :],
        other=0,
    ).to(tl.int32)
    k_hi = tl.load(
        page + K_PAYLOAD_OFFSET + k_row + k_byte[None, :] + 1,
        mask=dmask[None, :] & ((k_byte[None, :] + 1) < K_PACKED_DIM),
        other=0,
    ).to(tl.int32)
    k_code = ((k_lo | (k_hi << 8)) >> k_shift[None, :]) & 7

    k_idx = kvh * HEAD_SIZE + d
    k_mg = k_idx // 64

    kmin_code = tl.load(
        page + K_MIN_OFFSET + k_idx, mask=dmask, other=0
    ).to(tl.float32)
    kmin_secmin_off = K_MIN_OFFSET + K_PRIMARY_COUNT + k_mg * 2
    kmin_secscale_off = (
        K_MIN_OFFSET + K_PRIMARY_COUNT + K_META_GROUPS * 2 + k_mg * 2
    )
    kmin_secmin = _rabit2_u8pair_to_bf16_f32(
        page, kmin_secmin_off, dmask
    )
    kmin_secscale = _rabit2_u8pair_to_bf16_f32(
        page, kmin_secscale_off, dmask
    )
    k_min = kmin_code * kmin_secscale + kmin_secmin

    kscale_code = tl.load(
        page + K_SCALE_OFFSET + k_idx, mask=dmask, other=0
    ).to(tl.float32)
    kscale_secmin_off = K_SCALE_OFFSET + K_PRIMARY_COUNT + k_mg * 2
    kscale_secscale_off = (
        K_SCALE_OFFSET + K_PRIMARY_COUNT + K_META_GROUPS * 2 + k_mg * 2
    )
    kscale_secmin = _rabit2_u8pair_to_bf16_f32(
        page, kscale_secmin_off, dmask
    )
    kscale_secscale = _rabit2_u8pair_to_bf16_f32(
        page, kscale_secscale_off, dmask
    )
    k_scale = kscale_code * kscale_secscale + kscale_secmin
    kval = k_code.to(tl.float32) * k_scale[None, :] + k_min[None, :]

    # Decode packed V2 once for this KV head.
    v_byte = d // 4
    v_shift = (d % 4) * 2
    v_row = (t[:, None] * NUM_KV_HEADS + kvh) * V_PACKED_DIM
    vb = tl.load(
        page + V_PAYLOAD_OFFSET + v_row + v_byte[None, :],
        mask=dmask[None, :],
        other=0,
    ).to(tl.int32)
    v_code = (vb >> v_shift[None, :]) & 3

    vg = d // 32
    v_idx = (t[:, None] * NUM_KV_HEADS + kvh) * V_GROUPS + vg[None, :]
    v_mg = v_idx // 64

    vmin_code = tl.load(
        page + V_MIN_OFFSET + v_idx,
        mask=dmask[None, :],
        other=0,
    ).to(tl.float32)
    vmin_secmin_off = V_MIN_OFFSET + V_PRIMARY_COUNT + v_mg * 2
    vmin_secscale_off = (
        V_MIN_OFFSET + V_PRIMARY_COUNT + V_META_GROUPS * 2 + v_mg * 2
    )
    vmin_secmin = _rabit2_u8pair_to_bf16_f32(
        page, vmin_secmin_off, dmask[None, :]
    )
    vmin_secscale = _rabit2_u8pair_to_bf16_f32(
        page, vmin_secscale_off, dmask[None, :]
    )
    v_min = vmin_code * vmin_secscale + vmin_secmin

    vscale_code = tl.load(
        page + V_SCALE_OFFSET + v_idx,
        mask=dmask[None, :],
        other=0,
    ).to(tl.float32)
    vscale_secmin_off = V_SCALE_OFFSET + V_PRIMARY_COUNT + v_mg * 2
    vscale_secscale_off = (
        V_SCALE_OFFSET + V_PRIMARY_COUNT + V_META_GROUPS * 2 + v_mg * 2
    )
    vscale_secmin = _rabit2_u8pair_to_bf16_f32(
        page, vscale_secmin_off, dmask[None, :]
    )
    vscale_secscale = _rabit2_u8pair_to_bf16_f32(
        page, vscale_secscale_off, dmask[None, :]
    )
    v_scale = vscale_code * vscale_secscale + vscale_secmin
    vval = v_code.to(tl.float32) * v_scale + v_min
    # --- END VERBATIM REFERENCE DECODE ---

    if COUNT_DECODES:
        # Diagnostic only (constexpr; compiled out in normal execution).
        tl.atomic_add(decode_count_ptr, 1)

    # shared_decode: apply the ONE decoded page to every query of the block.
    qblock = tl.program_id(2)
    for j in range(QUERY_BLOCK):
        tile_q = qblock * QUERY_BLOCK + j
        valid = tile_q < n_queries
        q_row = q_ptr + tile_q * q_tile_stride
        pm = partial_m_ptr + tile_q * partial_tile_stride
        pl = partial_l_ptr + tile_q * partial_tile_stride
        pacc = partial_acc_ptr + tile_q * partial_tile_stride * HEAD_SIZE
        # --- BEGIN VERBATIM REFERENCE PER-QUERY BODY (masked for the partial block) ---
        for qi in tl.static_range(0, 4):
            qh = qh_base + qi
            q = tl.load(
                q_row + qh * HEAD_SIZE + d,
                mask=dmask & valid,
                other=0.0,
            ).to(tl.float32)

            scores = tl.sum(kval * q[None, :], axis=1) * softmax_scale
            m = tl.max(scores, axis=0)
            p = tl.exp(scores - m)
            l = tl.sum(p, axis=0)
            acc = tl.sum(p[:, None] * vval, axis=0)

            seg = logical_page * NUM_Q_HEADS + qh
            tl.store(pm + seg, m, mask=valid)
            tl.store(pl + seg, l, mask=valid)
            tl.store(
                pacc + seg * HEAD_SIZE + d,
                acc,
                mask=dmask & valid,
            )
        # --- END VERBATIM REFERENCE PER-QUERY BODY ---


def rabit2_shared_decode_closed_pages(
    runtime: "_r.Rabit2SingleSequenceRuntime",
    q_tile: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    closed: int,
    softmax_scale: float,
    query_block: int,
    decode_counter: torch.Tensor | None = None,
) -> None:
    """Closed-page partials for every query of a tile (all sharing ``closed`` visible pages)."""
    n, q_heads, d = (int(x) for x in q_tile.shape)
    kv_heads = int(runtime.num_kv_heads)
    layout = runtime.layout
    k_primary = layout.num_kv_heads * layout.head_size_k
    v_groups = layout.head_size_v // _r.RABIT2_GROUP_SIZE
    v_primary = layout.block_size * layout.num_kv_heads * v_groups
    segments = closed + 1
    qgroups = kv_heads * ((q_heads // kv_heads) // 4)
    count = decode_counter is not None
    _rabit2_shared_decode_closed_page_partial_kernel[(closed, qgroups, triton.cdiv(n, query_block))](
        q_tile,
        kv_cache,
        block_table_row,
        partial_m,
        partial_l,
        partial_acc,
        decode_counter if count else partial_m,  # unused unless COUNT_DECODES
        float(softmax_scale),
        n,
        q_heads * d,
        segments * q_heads,
        PAGE_BYTES=layout.page_bytes,
        K_PAYLOAD_OFFSET=layout.k_payload_offset,
        V_PAYLOAD_OFFSET=layout.v_payload_offset,
        K_MIN_OFFSET=layout.k_min_offset,
        K_SCALE_OFFSET=layout.k_scale_offset,
        V_MIN_OFFSET=layout.v_min_offset,
        V_SCALE_OFFSET=layout.v_scale_offset,
        K_PRIMARY_COUNT=k_primary,
        K_META_GROUPS=k_primary // _r.RABIT2_METADATA_GROUP_SIZE,
        V_PRIMARY_COUNT=v_primary,
        V_META_GROUPS=v_primary // _r.RABIT2_METADATA_GROUP_SIZE,
        NUM_Q_HEADS=q_heads,
        NUM_KV_HEADS=kv_heads,
        HEAD_SIZE=d,
        K_PACKED_DIM=_r.rabit2_packed_dim(d, 3),
        V_PACKED_DIM=_r.rabit2_packed_dim(runtime.head_size_v, 2),
        V_GROUPS=v_groups,
        BLOCK_D=triton.next_power_of_2(d),
        QUERY_BLOCK=query_block,
        COUNT_DECODES=count,
        num_warps=8,
    )


def rabit2_stage3c_forward_shared_decode(
    runtime: "_r.Rabit2SingleSequenceRuntime",
    q_seq: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    out_rows: torch.Tensor,
    softmax_scale: float,
    query_block: int | None = None,
    decode_counter: torch.Tensor | None = None,
) -> bool:
    """Process one non-initial chunk (q_len > 1) with shared_decode.

    Returns False without touching any state when it does not apply (the
    caller then runs the reference path). ``decode_counter`` (int32 CUDA
    scalar) is a test/diagnostic hook counting closed-page decodes.
    """
    q_len, q_heads, d = (int(x) for x in q_seq.shape)
    kv_heads = int(runtime.num_kv_heads)
    if q_len <= 1 or not q_seq.is_cuda or not _t32.rabit2_tile32_supported(q_heads, kv_heads):
        return False
    if softmax_scale is None:
        softmax_scale = d ** -0.5
    qb = rabit2_shared_decode_query_block() if query_block is None else int(query_block)
    if qb not in QUERY_BLOCKS:
        raise ValueError(f"shared_decode query block {qb} is not one of {QUERY_BLOCKS}")

    plan = _r.Rabit2CausalChunkPlan(runtime, k_seq, v_seq, kv_cache, block_table_row)
    block_d = triton.next_power_of_2(d)
    emit_tail = _r._rabit2_stage4b1_exactmeta_emit_tail_partial  # frozen binding

    closed_after = [_t32.rabit2_chunk_closed_pages_after(plan, i) for i in range(q_len)]
    for i0, i1, closed in _t32.rabit2_tile32_tiles(closed_after):
        n = i1 - i0
        segments = closed + 1
        q_tile = q_seq[i0:i1].contiguous()
        partial_m = torch.empty((n, segments, q_heads), dtype=torch.float32, device=q_seq.device)
        partial_l = torch.empty_like(partial_m)
        partial_acc = torch.empty((n, segments, q_heads, d), dtype=torch.float32, device=q_seq.device)

        if closed:
            rabit2_shared_decode_closed_pages(runtime, q_tile, kv_cache, block_table_row, partial_m, partial_l,
                                              partial_acc, closed, float(softmax_scale), qb, decode_counter)

        for j in range(n):
            plan.apply_step(i0 + j)
            if int(runtime.closed_pages) != closed:
                raise RuntimeError(
                    "RABIT-2 shared_decode closed-page schedule diverged from the causal chunk plan"
                )
            emit_tail(q_tile[j], runtime, partial_m[j], partial_l[j], partial_acc[j],
                      closed, float(softmax_scale))

        out_tile = torch.empty((n, q_heads, d), dtype=q_seq.dtype, device=q_seq.device)
        _t32._rabit2_tile32_reduce_partials_kernel[(q_heads, n)](
            partial_m,
            partial_l,
            partial_acc,
            out_tile,
            segments,
            segments * q_heads,
            q_heads * d,
            NUM_Q_HEADS=q_heads,
            HEAD_SIZE=d,
            BLOCK_S=triton.next_power_of_2(segments),
            BLOCK_D=block_d,
            num_warps=8,
        )
        out_rows[i0:i1].copy_(out_tile)
    return True
