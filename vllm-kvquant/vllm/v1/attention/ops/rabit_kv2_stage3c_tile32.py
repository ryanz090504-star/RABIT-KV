"""RABIT-KV2 Stage3C non-initial chunked prefill: selectable ``tile32`` path.

The reference Stage3C path (``TritonAttentionImpl._forward_rabit_kv2``) evaluates
a non-initial prefill chunk token by token: for every query it launches the
full single-query closed-page scan over the entire compressed prefix, then the
per-query tail partial and the reduce. It remains the default and the semantic
oracle.

``tile32`` changes only *how many times* the closed-page scan runs. Within a run
of consecutive chunk queries whose visible closed-page count is identical (at
most 32 queries; runs end at every page closure), the closed pages are the same
bytes for every query, so one multi-query launch computes each query's own
closed-page partials. Everything that depends on the evolving per-query state
stays per query, in reference order: ``Rabit2CausalChunkPlan.apply_step`` and
the frozen tail-partial emitter (its shared scratch workspace is reused in
stream order, exactly as in the reference). The batched kernels below copy the
reference kernel bodies verbatim and only offset the per-query pointers, so
each program performs the same arithmetic as the reference program.

Nothing here changes the K3/V2/G32/R4/META8g64 representation, page layout,
persistent cache bytes (still written only by ``Rabit2CausalChunkPlan``), the
residual window, the decode (q_len == 1) path or the initial-prefill path.

Selection: ``VLLM_RABIT2_STAGE3C_IMPL`` = ``reference`` (default) | ``tile32``.
Diagnostic component timing: ``VLLM_RABIT2_STAGE3C_PROFILE=1`` (synchronizes the
device around each component; diagnostic runs only, never for latency results).
"""

from __future__ import annotations

import os
import time

import torch

import vllm.v1.attention.ops.rabit_kv2 as _r
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

STAGE3C_IMPL_ENV = "VLLM_RABIT2_STAGE3C_IMPL"
STAGE3C_PROFILE_ENV = "VLLM_RABIT2_STAGE3C_PROFILE"
STAGE3C_IMPLS = ("reference", "tile32")
TILE_MAX_QUERIES = 32


def rabit2_stage3c_impl() -> str:
    """Selected Stage3C implementation; ``reference`` unless explicitly set."""
    impl = os.environ.get(STAGE3C_IMPL_ENV, "reference").strip().lower() or "reference"
    if impl not in STAGE3C_IMPLS:
        raise ValueError(
            f"{STAGE3C_IMPL_ENV}={impl!r} is not one of {STAGE3C_IMPLS}"
        )
    return impl


def _profiling() -> bool:
    return os.environ.get(STAGE3C_PROFILE_ENV, "0") == "1"


# The reference kernel's @triton.jit helper, bound under the same name the
# verbatim body uses.
_rabit2_u8pair_to_bf16_f32 = _r._rabit2_u8pair_to_bf16_f32


@triton.jit
def _rabit2_tile32_closed_page_partial_kernel(
    q_ptr,
    cache_ptr,
    block_table_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    softmax_scale,
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
):
    # tile32: select this program's query; everything below is the verbatim
    # body of _rabit2_stage4b3_gqa4_closed_page_partial_kernel.
    tile_q = tl.program_id(2)
    q_ptr = q_ptr + tile_q * q_tile_stride
    partial_m_ptr = partial_m_ptr + tile_q * partial_tile_stride
    partial_l_ptr = partial_l_ptr + tile_q * partial_tile_stride
    partial_acc_ptr = partial_acc_ptr + tile_q * partial_tile_stride * HEAD_SIZE
    # --- BEGIN VERBATIM REFERENCE BODY ---
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

    # Reuse decoded K/V for four Q heads.
    for qi in tl.static_range(0, 4):
        qh = qh_base + qi
        q = tl.load(
            q_ptr + qh * HEAD_SIZE + d,
            mask=dmask,
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(kval * q[None, :], axis=1) * softmax_scale
        m = tl.max(scores, axis=0)
        p = tl.exp(scores - m)
        l = tl.sum(p, axis=0)
        acc = tl.sum(p[:, None] * vval, axis=0)

        seg = logical_page * NUM_Q_HEADS + qh
        tl.store(partial_m_ptr + seg, m)
        tl.store(partial_l_ptr + seg, l)
        tl.store(
            partial_acc_ptr + seg * HEAD_SIZE + d,
            acc,
            mask=dmask,
        )
    # --- END VERBATIM REFERENCE BODY ---


@triton.jit
def _rabit2_tile32_reduce_partials_kernel(
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    out_ptr,
    num_segments,
    partial_tile_stride,
    out_tile_stride,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # tile32: select this program's query; everything below is the verbatim
    # body of _rabit2_reduce_partials_kernel.
    tile_q = tl.program_id(1)
    partial_m_ptr = partial_m_ptr + tile_q * partial_tile_stride
    partial_l_ptr = partial_l_ptr + tile_q * partial_tile_stride
    partial_acc_ptr = partial_acc_ptr + tile_q * partial_tile_stride * HEAD_SIZE
    out_ptr = out_ptr + tile_q * out_tile_stride
    # --- BEGIN VERBATIM REFERENCE BODY ---
    qh = tl.program_id(0)
    s = tl.arange(0, BLOCK_S)
    d = tl.arange(0, BLOCK_D)
    smask = s < num_segments
    dmask = d < HEAD_SIZE
    seg_idx = s * NUM_Q_HEADS + qh
    ms = tl.load(partial_m_ptr + seg_idx, mask=smask, other=float("-inf"))
    ls = tl.load(partial_l_ptr + seg_idx, mask=smask, other=0.0)
    global_m = tl.max(ms, axis=0)
    w = tl.where(smask, tl.exp(ms - global_m), 0.0)
    denom = tl.sum(w * ls, axis=0)
    acc_idx = seg_idx[:, None] * HEAD_SIZE + d[None, :]
    acc = tl.load(
        partial_acc_ptr + acc_idx,
        mask=smask[:, None] & dmask[None, :],
        other=0.0,
    )
    out = tl.sum(acc * w[:, None], axis=0) / denom
    tl.store(out_ptr + qh * HEAD_SIZE + d, out, mask=dmask)
    # --- END VERBATIM REFERENCE BODY ---



def rabit2_tile32_supported(q_heads: int, kv_heads: int) -> bool:
    """Same GQA4 condition as the reference Stage4B3 dispatch."""
    return q_heads % kv_heads == 0 and (q_heads // kv_heads) % 4 == 0


def rabit2_chunk_closed_pages_after(plan: "_r.Rabit2CausalChunkPlan", step: int) -> int:
    """runtime.closed_pages after ``plan.apply_step(step)`` (same arithmetic)."""
    processed = int(step) + 1
    aged_now = max(
        0, plan.init_recent_len + processed - int(plan.runtime.residual_tokens)
    )
    open_total_now = plan.init_open_len + aged_now
    return plan.init_closed + open_total_now // int(plan.runtime.block_size)


def rabit2_tile32_tiles(closed_after: list[int]) -> list[tuple[int, int, int]]:
    """Maximal runs of equal closed-page count, capped at TILE_MAX_QUERIES:
    [(start, end, closed_pages)] covering 0..len-1 in order."""
    tiles, start = [], 0
    for i in range(1, len(closed_after) + 1):
        if (i == len(closed_after) or closed_after[i] != closed_after[start]
                or i - start == TILE_MAX_QUERIES):
            tiles.append((start, i, closed_after[start]))
            start = i
    return tiles


class _Profile:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.acc: dict[str, float] = {}
        self._t = 0.0

    def start(self) -> None:
        if self.enabled:
            torch.cuda.synchronize()
            self._t = time.perf_counter()

    def stop(self, key: str) -> None:
        if self.enabled:
            torch.cuda.synchronize()
            now = time.perf_counter()
            self.acc[key] = self.acc.get(key, 0.0) + (now - self._t) * 1000.0
            self._t = now


def rabit2_stage3c_forward_tile32(
    runtime: "_r.Rabit2SingleSequenceRuntime",
    q_seq: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    out_rows: torch.Tensor,
    softmax_scale: float,
) -> bool:
    """Process one non-initial chunk (q_len > 1) with tile32.

    Returns False without touching any state when tile32 does not apply (the
    caller then runs the reference path).
    """
    q_len, q_heads, d = (int(x) for x in q_seq.shape)
    kv_heads = int(runtime.num_kv_heads)
    if q_len <= 1 or not q_seq.is_cuda or not rabit2_tile32_supported(q_heads, kv_heads):
        return False
    if softmax_scale is None:
        softmax_scale = d ** -0.5
    prof = _Profile(_profiling())

    prof.start()
    plan = _r.Rabit2CausalChunkPlan(runtime, k_seq, v_seq, kv_cache, block_table_row)
    prof.stop("chunk_plan_ms")

    layout = runtime.layout
    k_primary = layout.num_kv_heads * layout.head_size_k
    k_meta_groups = k_primary // _r.RABIT2_METADATA_GROUP_SIZE
    v_groups = layout.head_size_v // _r.RABIT2_GROUP_SIZE
    v_primary = layout.block_size * layout.num_kv_heads * v_groups
    v_meta_groups = v_primary // _r.RABIT2_METADATA_GROUP_SIZE
    block_d = triton.next_power_of_2(d)
    qgroups = kv_heads * ((q_heads // kv_heads) // 4)
    emit_tail = _r._rabit2_stage4b1_exactmeta_emit_tail_partial  # frozen binding

    closed_after = [rabit2_chunk_closed_pages_after(plan, i) for i in range(q_len)]
    for i0, i1, closed in rabit2_tile32_tiles(closed_after):
        n = i1 - i0
        segments = closed + 1
        q_tile = q_seq[i0:i1].contiguous()
        partial_m = torch.empty((n, segments, q_heads), dtype=torch.float32, device=q_seq.device)
        partial_l = torch.empty_like(partial_m)
        partial_acc = torch.empty((n, segments, q_heads, d), dtype=torch.float32, device=q_seq.device)

        prof.start()
        if closed:
            _rabit2_tile32_closed_page_partial_kernel[(closed, qgroups, n)](
                q_tile,
                kv_cache,
                block_table_row,
                partial_m,
                partial_l,
                partial_acc,
                float(softmax_scale),
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
                K_META_GROUPS=k_meta_groups,
                V_PRIMARY_COUNT=v_primary,
                V_META_GROUPS=v_meta_groups,
                NUM_Q_HEADS=q_heads,
                NUM_KV_HEADS=kv_heads,
                HEAD_SIZE=d,
                K_PACKED_DIM=_r.rabit2_packed_dim(d, 3),
                V_PACKED_DIM=_r.rabit2_packed_dim(runtime.head_size_v, 2),
                V_GROUPS=v_groups,
                BLOCK_D=block_d,
                num_warps=8,
            )
        prof.stop("closed_page_batched_ms")

        for j in range(n):
            plan.apply_step(i0 + j)
            if int(runtime.closed_pages) != closed:
                raise RuntimeError(
                    "RABIT-2 tile32 closed-page schedule diverged from the causal chunk plan"
                )
            emit_tail(q_tile[j], runtime, partial_m[j], partial_l[j], partial_acc[j],
                      closed, float(softmax_scale))
        prof.stop("tail_partial_ms")

        out_tile = torch.empty((n, q_heads, d), dtype=q_seq.dtype, device=q_seq.device)
        _rabit2_tile32_reduce_partials_kernel[(q_heads, n)](
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
        prof.stop("reduce_ms")
        out_rows[i0:i1].copy_(out_tile)
        prof.stop("output_copy_ms")

    if prof.enabled:
        logger.info("RABIT2_STAGE3C_TILE32_PROFILE q_len=%d %s", q_len,
                    {k: round(v, 3) for k, v in prof.acc.items()})
    return True
