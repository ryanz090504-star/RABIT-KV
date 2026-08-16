# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-first physical page codec and online state for RABIT-KV2.

This module is the Stage-3A bridge between the quality oracle and the later
Triton serving kernels.  It implements the exact byte layout allocated by
``rabit2_page_layout`` and an online reference state that matches the final
quality policy:

* K: 3-bit sequence-axis affine quantization, group size 32.
* V: 2-bit last-dimension affine quantization, group size 32.
* Primary metadata: UINT8 affine, metadata group size 64, with BF16
  second-level minima/scales.
* Newest four K/V tokens: BF16 residual.

The code below is deliberately PyTorch/Python.  It is a byte-exact oracle for
Stage-3B Triton write/read kernels and is not used for latency claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.v1.attention.ops.kvquant_k3 import (
    RABIT2_GROUP_SIZE,
    RABIT2_METADATA_GROUP_SIZE,
    RABIT2_RESIDUAL_TOKENS,
    decode_metadata_uint8_group_ref,
    dequantize_k3_sequence_affine_ref,
    encode_metadata_uint8_group_ref,
    pack_int2_values,
    quantize_k3_sequence_affine_ref,
    unpack_int2_values,
)
from vllm.v1.kv_cache_interface import (
    Rabit2PageLayout,
    rabit2_page_layout,
    rabit2_packed_dim,
)


def _tensor_as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Return a contiguous 1-D uint8 view without changing device."""
    return tensor.contiguous().view(torch.uint8).reshape(-1)


def _metadata_primary_value_count(metadata: dict[str, Any]) -> int:
    n = 1
    for dim in metadata["orig_shape"]:
        n *= int(dim)
    return n


def encode_metadata_blob_ref(metadata: dict[str, Any]) -> torch.Tensor:
    """Serialize one compressed metadata tensor into its physical byte blob.

    Layout:
      1. UINT8 primary codes (including any metadata-group padding),
      2. BF16 second-level minima,
      3. BF16 second-level scales.

    For the Llama-3.1-8B RABIT layout, primary counts are multiples of 64, so
    no padded primary bytes are present.  The function retains padding support
    for reference tests and validates the blob on decode.
    """
    if metadata.get("meta_type") != "uint8_group":
        raise ValueError("expected uint8_group metadata")
    return torch.cat(
        [
            metadata["codes"].reshape(-1).to(torch.uint8),
            _tensor_as_bytes(metadata["min"]),
            _tensor_as_bytes(metadata["scale"]),
        ]
    ).contiguous()


def decode_metadata_blob_ref(
    blob: torch.Tensor,
    *,
    original_shape: tuple[int, ...],
    metadata_group_size: int = RABIT2_METADATA_GROUP_SIZE,
) -> dict[str, Any]:
    """Inverse of :func:`encode_metadata_blob_ref`."""
    if blob.dtype != torch.uint8 or blob.ndim != 1:
        raise ValueError("metadata blob must be a 1-D uint8 tensor")
    value_count = 1
    for dim in original_shape:
        value_count *= int(dim)
    groups = (value_count + metadata_group_size - 1) // metadata_group_size
    padded_count = groups * metadata_group_size
    expected = padded_count + groups * 4
    if blob.numel() != expected:
        raise ValueError(
            f"metadata blob has {blob.numel()} bytes; expected {expected}"
        )
    code_end = padded_count
    secondary_bytes = groups * 2
    min_end = code_end + secondary_bytes
    codes = blob[:code_end].reshape(groups, metadata_group_size)
    meta_min = blob[code_end:min_end].contiguous().view(torch.bfloat16).reshape(groups, 1)
    meta_scale = blob[min_end:].contiguous().view(torch.bfloat16).reshape(groups, 1)
    return {
        "meta_type": "uint8_group",
        "codes": codes,
        "min": meta_min,
        "scale": meta_scale,
        "orig_shape": tuple(original_shape),
        "pad": padded_count - value_count,
        "group_size": int(metadata_group_size),
    }


def _quantize_v2_primary_ref(
    value: torch.Tensor,
    *,
    group_size: int = RABIT2_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return V2 payload codes plus uncompressed primary min/scale.

    ``value`` is ``[tokens, heads, head_dim]``.  The returned primary metadata
    remains float32 so the online state can later reproduce META8g64 exactly
    when enough tokens are present to close the physical page.
    """
    if value.ndim != 3:
        raise ValueError("value must have shape [tokens, heads, head_dim]")
    tokens, heads, head_dim = value.shape
    padded_dim = ((head_dim + group_size - 1) // group_size) * group_size
    x = value.detach().float()
    if padded_dim != head_dim:
        x = torch.nn.functional.pad(x, (0, padded_dim - head_dim))
    grouped = x.reshape(tokens, heads, padded_dim // group_size, group_size)
    q_min = grouped.amin(dim=-1, keepdim=True)
    q_max = grouped.amax(dim=-1, keepdim=True)
    scale = (q_max - q_min) / 3.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    codes = torch.round((grouped - q_min) / scale).clamp(0, 3).to(torch.uint8)
    packed = pack_int2_values(codes.reshape(tokens, heads, padded_dim))
    return packed, q_min.contiguous(), scale.contiguous(), padded_dim


def _dequantize_v2_from_primary_ref(
    packed: torch.Tensor,
    q_min: torch.Tensor,
    scale: torch.Tensor,
    *,
    head_dim: int,
    group_size: int = RABIT2_GROUP_SIZE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tokens, heads = packed.shape[:2]
    padded_dim = q_min.shape[2] * group_size
    codes = unpack_int2_values(packed, padded_dim)
    grouped = codes.reshape(tokens, heads, padded_dim // group_size, group_size)
    values = (grouped.float() * scale.float() + q_min.float()).reshape(
        tokens, heads, padded_dim
    )
    return values[..., :head_dim].to(dtype)


def encode_rabit2_page_ref(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout: Rabit2PageLayout | None = None,
) -> torch.Tensor:
    """Encode one complete physical RABIT-KV2 page.

    Both tensors must contain exactly ``layout.block_size`` tokens.  The output
    is a 1-D uint8 tensor with ``layout.page_bytes`` elements.
    """
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key/value must share [tokens, heads, head_dim]")
    tokens, heads, head_dim = key.shape
    if layout is None:
        layout = rabit2_page_layout(
            block_size=tokens,
            num_kv_heads=heads,
            head_size_k=head_dim,
            head_size_v=value.shape[-1],
        )
    if tokens != layout.block_size:
        raise ValueError("a physical page must contain exactly block_size tokens")
    if heads != layout.num_kv_heads or head_dim != layout.head_size_k:
        raise ValueError("key shape does not match page layout")
    if value.shape[-1] != layout.head_size_v:
        raise ValueError("value shape does not match page layout")

    k_state = quantize_k3_sequence_affine_ref(key)
    v_packed, v_min_primary, v_scale_primary, _ = _quantize_v2_primary_ref(value)
    v_min = encode_metadata_uint8_group_ref(v_min_primary)
    v_scale = encode_metadata_uint8_group_ref(v_scale_primary)

    k_payload = k_state["packed"].reshape(-1).to(torch.uint8)
    v_payload = v_packed.reshape(-1).to(torch.uint8)
    k_min_blob = encode_metadata_blob_ref(k_state["min"])
    k_scale_blob = encode_metadata_blob_ref(k_state["scale"])
    v_min_blob = encode_metadata_blob_ref(v_min)
    v_scale_blob = encode_metadata_blob_ref(v_scale)

    expected_parts = {
        "k_payload": (k_payload.numel(), layout.k_payload_bytes),
        "v_payload": (v_payload.numel(), layout.v_payload_bytes),
        "k_min": (k_min_blob.numel(), layout.k_min_bytes),
        "k_scale": (k_scale_blob.numel(), layout.k_scale_bytes),
        "v_min": (v_min_blob.numel(), layout.v_min_bytes),
        "v_scale": (v_scale_blob.numel(), layout.v_scale_bytes),
    }
    for name, (actual, expected) in expected_parts.items():
        if actual != expected:
            raise ValueError(f"{name} has {actual} bytes; layout expects {expected}")

    page = torch.zeros(layout.page_bytes, dtype=torch.uint8, device=key.device)
    page[layout.k_payload_offset : layout.k_payload_offset + layout.k_payload_bytes] = k_payload
    page[layout.v_payload_offset : layout.v_payload_offset + layout.v_payload_bytes] = v_payload
    page[layout.k_min_offset : layout.k_min_offset + layout.k_min_bytes] = k_min_blob
    page[layout.k_scale_offset : layout.k_scale_offset + layout.k_scale_bytes] = k_scale_blob
    page[layout.v_min_offset : layout.v_min_offset + layout.v_min_bytes] = v_min_blob
    page[layout.v_scale_offset : layout.v_scale_offset + layout.v_scale_bytes] = v_scale_blob
    return page


def decode_rabit2_page_ref(
    page: torch.Tensor,
    *,
    layout: Rabit2PageLayout,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one opaque physical page into dequantized K/V tensors."""
    if page.dtype != torch.uint8:
        raise ValueError("page must use uint8 storage")
    page = page.reshape(-1)
    if page.numel() != layout.page_bytes:
        raise ValueError("page size does not match layout")

    k_packed_dim = rabit2_packed_dim(layout.head_size_k, 3)
    v_padded_dim = (
        (layout.head_size_v + RABIT2_GROUP_SIZE - 1) // RABIT2_GROUP_SIZE
    ) * RABIT2_GROUP_SIZE
    v_packed_dim = rabit2_packed_dim(v_padded_dim, 2)

    k_packed = page[
        layout.k_payload_offset : layout.k_payload_offset + layout.k_payload_bytes
    ].reshape(layout.block_size, layout.num_kv_heads, k_packed_dim)
    v_packed = page[
        layout.v_payload_offset : layout.v_payload_offset + layout.v_payload_bytes
    ].reshape(layout.block_size, layout.num_kv_heads, v_packed_dim)

    k_primary_shape = (
        1,
        layout.num_kv_heads,
        layout.block_size // RABIT2_GROUP_SIZE,
        1,
        layout.head_size_k,
    )
    v_primary_shape = (
        layout.block_size,
        layout.num_kv_heads,
        v_padded_dim // RABIT2_GROUP_SIZE,
        1,
    )
    k_min = decode_metadata_blob_ref(
        page[layout.k_min_offset : layout.k_min_offset + layout.k_min_bytes],
        original_shape=k_primary_shape,
    )
    k_scale = decode_metadata_blob_ref(
        page[layout.k_scale_offset : layout.k_scale_offset + layout.k_scale_bytes],
        original_shape=k_primary_shape,
    )
    v_min_meta = decode_metadata_blob_ref(
        page[layout.v_min_offset : layout.v_min_offset + layout.v_min_bytes],
        original_shape=v_primary_shape,
    )
    v_scale_meta = decode_metadata_blob_ref(
        page[layout.v_scale_offset : layout.v_scale_offset + layout.v_scale_bytes],
        original_shape=v_primary_shape,
    )

    k_state = {
        "type": "rabit2_k3_seq_affine",
        "bits": 3,
        "packed": k_packed,
        "original_shape": (
            layout.block_size,
            layout.num_kv_heads,
            layout.head_size_k,
        ),
        "padded_tokens": layout.block_size,
        "pad_seq": 0,
        "seq_group_size": RABIT2_GROUP_SIZE,
        "min": k_min,
        "scale": k_scale,
    }
    key = dequantize_k3_sequence_affine_ref(k_state, dtype=dtype)
    v_min = decode_metadata_uint8_group_ref(v_min_meta)
    v_scale = decode_metadata_uint8_group_ref(v_scale_meta)
    value = _dequantize_v2_from_primary_ref(
        v_packed,
        v_min,
        v_scale,
        head_dim=layout.head_size_v,
        dtype=dtype,
    )
    return key, value


@dataclass
class Rabit2OnlineStateRef:
    """Exact online RABIT-KV2 state for one layer and one sequence.

    Closed 32-token old-prefix groups are stored as physical page bytes.  The
    incomplete K sequence group remains BF16 so it can be requantized exactly
    as the quality oracle grows.  Its matching V values are stored as V2
    payload plus float32 primary metadata; META8g64 is recomputed on read and
    when the page closes.  The newest four K/V tokens remain BF16.
    """

    num_kv_heads: int
    head_size_k: int
    head_size_v: int
    block_size: int = RABIT2_GROUP_SIZE
    residual_tokens: int = RABIT2_RESIDUAL_TOKENS
    pages: list[torch.Tensor] = field(default_factory=list)
    open_k: torch.Tensor | None = None
    open_v_packed: torch.Tensor | None = None
    open_v_min: torch.Tensor | None = None
    open_v_scale: torch.Tensor | None = None
    recent_k: torch.Tensor | None = None
    recent_v: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.block_size != RABIT2_GROUP_SIZE:
            raise ValueError("Stage-3A online reference requires block_size=32")
        # The physical page codec is currently specialized for the selected
        # Llama-3.1-8B policy.  These alignment conditions ensure META8g64
        # groups never cross a closed-page boundary, so page-local encoding is
        # byte-identical to whole-prefix quality encoding.
        if self.head_size_k % RABIT2_METADATA_GROUP_SIZE:
            raise ValueError("Stage-3A requires K head_size to be a multiple of 64")
        v_groups = (self.head_size_v + RABIT2_GROUP_SIZE - 1) // RABIT2_GROUP_SIZE
        if (self.block_size * self.num_kv_heads * v_groups) % RABIT2_METADATA_GROUP_SIZE:
            raise ValueError("V metadata groups must align to physical page boundaries")
        self.layout = rabit2_page_layout(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size_k=self.head_size_k,
            head_size_v=self.head_size_v,
        )

    @property
    def device(self) -> torch.device:
        for tensor in (self.recent_k, self.open_k):
            if tensor is not None:
                return tensor.device
        if self.pages:
            return self.pages[0].device
        return torch.device("cpu")

    @property
    def total_tokens(self) -> int:
        return (
            len(self.pages) * self.block_size
            + (0 if self.open_k is None else self.open_k.shape[0])
            + (0 if self.recent_k is None else self.recent_k.shape[0])
        )

    def _append_open_old(self, key: torch.Tensor, value: torch.Tensor) -> None:
        packed, q_min, scale, _ = _quantize_v2_primary_ref(value)
        self.open_k = key.to(torch.bfloat16).contiguous() if self.open_k is None else torch.cat(
            [self.open_k, key.to(torch.bfloat16)], dim=0
        )
        self.open_v_packed = packed if self.open_v_packed is None else torch.cat(
            [self.open_v_packed, packed], dim=0
        )
        self.open_v_min = q_min if self.open_v_min is None else torch.cat(
            [self.open_v_min, q_min], dim=0
        )
        self.open_v_scale = scale if self.open_v_scale is None else torch.cat(
            [self.open_v_scale, scale], dim=0
        )
        if self.open_k.shape[0] == self.block_size:
            v_min_meta = encode_metadata_uint8_group_ref(self.open_v_min)
            v_scale_meta = encode_metadata_uint8_group_ref(self.open_v_scale)
            # Reconstruct V with the compressed metadata before page encoding.
            # This ensures the page bytes and readback match the quality oracle.
            v_min_dec = decode_metadata_uint8_group_ref(v_min_meta)
            v_scale_dec = decode_metadata_uint8_group_ref(v_scale_meta)
            v_for_page = _dequantize_v2_from_primary_ref(
                self.open_v_packed,
                v_min_dec,
                v_scale_dec,
                head_dim=self.head_size_v,
                dtype=torch.float32,
            )
            # encode_rabit2_page_ref requantizes V primary metadata from values.
            # To avoid a second quantization drift, write the page from primary
            # components directly.
            self.pages.append(self._encode_open_page_exact(v_min_meta, v_scale_meta))
            self.open_k = None
            self.open_v_packed = None
            self.open_v_min = None
            self.open_v_scale = None

    def _encode_open_page_exact(
        self,
        v_min_meta: dict[str, Any],
        v_scale_meta: dict[str, Any],
    ) -> torch.Tensor:
        assert self.open_k is not None
        assert self.open_v_packed is not None
        k_state = quantize_k3_sequence_affine_ref(self.open_k)
        blobs = {
            "k_payload": k_state["packed"].reshape(-1).to(torch.uint8),
            "v_payload": self.open_v_packed.reshape(-1).to(torch.uint8),
            "k_min": encode_metadata_blob_ref(k_state["min"]),
            "k_scale": encode_metadata_blob_ref(k_state["scale"]),
            "v_min": encode_metadata_blob_ref(v_min_meta),
            "v_scale": encode_metadata_blob_ref(v_scale_meta),
        }
        page = torch.zeros(
            self.layout.page_bytes, dtype=torch.uint8, device=self.open_k.device
        )
        for name, offset, size in (
            ("k_payload", self.layout.k_payload_offset, self.layout.k_payload_bytes),
            ("v_payload", self.layout.v_payload_offset, self.layout.v_payload_bytes),
            ("k_min", self.layout.k_min_offset, self.layout.k_min_bytes),
            ("k_scale", self.layout.k_scale_offset, self.layout.k_scale_bytes),
            ("v_min", self.layout.v_min_offset, self.layout.v_min_bytes),
            ("v_scale", self.layout.v_scale_offset, self.layout.v_scale_bytes),
        ):
            if blobs[name].numel() != size:
                raise ValueError(f"{name} does not match physical layout")
            page[offset : offset + size] = blobs[name]
        return page

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append one or more tokens while preserving exact grouping semantics."""
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("key/value must share [tokens, heads, head_dim]")
        if key.shape[1:] != (self.num_kv_heads, self.head_size_k):
            raise ValueError("key shape does not match online state")
        if value.shape[-1] != self.head_size_v:
            raise ValueError("value head dimension does not match online state")
        for i in range(key.shape[0]):
            k_one = key[i : i + 1]
            v_one = value[i : i + 1]
            self.recent_k = (
                k_one.to(torch.bfloat16).contiguous()
                if self.recent_k is None
                else torch.cat([self.recent_k, k_one.to(torch.bfloat16)], dim=0)
            )
            self.recent_v = (
                v_one.to(torch.bfloat16).contiguous()
                if self.recent_v is None
                else torch.cat([self.recent_v, v_one.to(torch.bfloat16)], dim=0)
            )
            if self.recent_k.shape[0] > self.residual_tokens:
                old_k = self.recent_k[:1]
                old_v = self.recent_v[:1]
                self.recent_k = self.recent_k[1:].contiguous()
                self.recent_v = self.recent_v[1:].contiguous()
                self._append_open_old(old_k, old_v)

    def _decode_open(self, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self.open_k is None:
            empty_k = torch.empty(
                (0, self.num_kv_heads, self.head_size_k),
                dtype=dtype,
                device=self.device,
            )
            empty_v = torch.empty(
                (0, self.num_kv_heads, self.head_size_v),
                dtype=dtype,
                device=self.device,
            )
            return empty_k, empty_v
        assert self.open_v_packed is not None
        assert self.open_v_min is not None
        assert self.open_v_scale is not None

        # Match the quality oracle's incomplete K group: zero-pad to 32 before
        # computing affine min/scale, then keep only the real tokens.
        k_state = quantize_k3_sequence_affine_ref(self.open_k)
        open_k = dequantize_k3_sequence_affine_ref(k_state, dtype=dtype)

        v_min_meta = encode_metadata_uint8_group_ref(self.open_v_min)
        v_scale_meta = encode_metadata_uint8_group_ref(self.open_v_scale)
        v_min = decode_metadata_uint8_group_ref(v_min_meta)
        v_scale = decode_metadata_uint8_group_ref(v_scale_meta)
        open_v = _dequantize_v2_from_primary_ref(
            self.open_v_packed,
            v_min,
            v_scale,
            head_dim=self.head_size_v,
            dtype=dtype,
        )
        return open_k, open_v

    def materialize(
        self, *, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode all physical/staging regions into logical K/V order."""
        key_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        for page in self.pages:
            k_page, v_page = decode_rabit2_page_ref(
                page, layout=self.layout, dtype=dtype
            )
            key_parts.append(k_page)
            value_parts.append(v_page)
        open_k, open_v = self._decode_open(dtype=dtype)
        if open_k.numel():
            key_parts.append(open_k)
            value_parts.append(open_v)
        if self.recent_k is not None:
            key_parts.append(self.recent_k.to(dtype))
            value_parts.append(self.recent_v.to(dtype))
        if not key_parts:
            return (
                torch.empty(
                    (0, self.num_kv_heads, self.head_size_k),
                    dtype=dtype,
                    device=self.device,
                ),
                torch.empty(
                    (0, self.num_kv_heads, self.head_size_v),
                    dtype=dtype,
                    device=self.device,
                ),
            )
        return torch.cat(key_parts, dim=0), torch.cat(value_parts, dim=0)

    def storage_breakdown(self) -> dict[str, int]:
        """Actual bytes currently held by the Stage-3A physical state."""
        pages = sum(int(page.numel()) for page in self.pages)
        open_k = 0 if self.open_k is None else self.open_k.numel() * self.open_k.element_size()
        open_v_payload = (
            0
            if self.open_v_packed is None
            else self.open_v_packed.numel() * self.open_v_packed.element_size()
        )
        open_v_primary = 0
        for tensor in (self.open_v_min, self.open_v_scale):
            if tensor is not None:
                open_v_primary += tensor.numel() * tensor.element_size()
        residual = 0
        for tensor in (self.recent_k, self.recent_v):
            if tensor is not None:
                residual += tensor.numel() * tensor.element_size()
        total = pages + open_k + open_v_payload + open_v_primary + residual
        return {
            "pages": int(pages),
            "open_k_bf16": int(open_k),
            "open_v_payload": int(open_v_payload),
            "open_v_primary_metadata": int(open_v_primary),
            "residual": int(residual),
            "total": int(total),
        }
# === RABIT2_STAGE3B1_TRITON_BEGIN ===
# Stage 3B1: Triton closed-page codec + fused packed-read attention oracle.
# FINAL 3B1: keep the fully-Triton writer experimental; deployment uses the exact CUDA reference writer while the Triton packed-read attention path is gated for correctness.
# This intentionally does NOT remove the triton_attn.py engine guard yet: the
# selected R4 policy needs an online staging sidecar before engine integration.
from vllm.triton_utils import tl, triton

_RABIT2_STAGE3B1 = True


@triton.jit
def _rabit2_round_to_even_nonnegative(x):
    """Match torch.round for the non-negative quantizer coordinates."""
    flo = tl.floor(x)
    frac = x - flo
    flo_i = flo.to(tl.int32)
    tie_up = (flo_i & 1) != 0
    return tl.where(frac > 0.5, flo + 1.0, tl.where(frac < 0.5, flo, tl.where(tie_up, flo + 1.0, flo)))


def _rabit2_meta_parts(page: torch.Tensor, offset: int, value_count: int):
    """Zero-copy views of one META8g64 blob inside an opaque page."""
    groups = (int(value_count) + RABIT2_METADATA_GROUP_SIZE - 1) // RABIT2_METADATA_GROUP_SIZE
    padded = groups * RABIT2_METADATA_GROUP_SIZE
    codes = page[offset : offset + padded]
    mins_begin = offset + padded
    mins_end = mins_begin + groups * 2
    mins = page[mins_begin:mins_end].contiguous().view(torch.bfloat16)
    scales = page[mins_end : mins_end + groups * 2].contiguous().view(torch.bfloat16)
    return codes, mins, scales


@triton.jit
def _rabit2_k_primary_kernel(
    key_ptr,
    min_ptr,
    scale_ptr,
    stride_t: tl.int64,
    stride_h: tl.int64,
    stride_d: tl.int64,
    HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    head = tl.program_id(0)
    d_block = tl.program_id(1)
    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    t = tl.arange(0, 32)
    mask_d = d < HEAD_SIZE
    ptrs = key_ptr + t[:, None] * stride_t + head * stride_h + d[None, :] * stride_d
    x = tl.load(ptrs, mask=mask_d[None, :], other=0.0).to(tl.float32)
    q_min = tl.min(x, axis=0)
    q_max = tl.max(x, axis=0)
    q_scale = (q_max - q_min) / 7.0
    q_scale = tl.where(tl.abs(q_scale) < 1.0e-8, 1.0, q_scale)
    out = head * HEAD_SIZE + d
    tl.store(min_ptr + out, q_min, mask=mask_d)
    tl.store(scale_ptr + out, q_scale, mask=mask_d)


@triton.jit
def _rabit2_k_pack_kernel(
    key_ptr,
    min_ptr,
    scale_ptr,
    payload_ptr,
    stride_t: tl.int64,
    stride_h: tl.int64,
    stride_d: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    b = tl.arange(0, BLOCK_B)
    mask_b = b < PACKED_DIM
    base_d = (b * 8) // 3
    stream_shift = (b * 8) % 3
    word = tl.zeros([BLOCK_B], dtype=tl.int32)
    for j in range(4):
        d = base_d + j
        mask = mask_b & (d < HEAD_SIZE)
        x = tl.load(
            key_ptr + token * stride_t + head * stride_h + d * stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        mn = tl.load(min_ptr + head * HEAD_SIZE + d, mask=mask, other=0.0)
        sc = tl.load(scale_ptr + head * HEAD_SIZE + d, mask=mask, other=1.0)
        q = _rabit2_round_to_even_nonnegative((x - mn) / sc).to(tl.int32)
        q = tl.maximum(0, tl.minimum(7, q))
        word = word | (q << (3 * j))
    packed = (word >> stream_shift) & 0xFF
    out = (token * NUM_HEADS + head) * PACKED_DIM + b
    tl.store(payload_ptr + out, packed.to(tl.uint8), mask=mask_b)


@triton.jit
def _rabit2_v_primary_kernel(
    value_ptr,
    min_ptr,
    scale_ptr,
    stride_t: tl.int64,
    stride_h: tl.int64,
    stride_d: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    group = tl.program_id(2)
    d = group * 32 + tl.arange(0, 32)
    mask = d < HEAD_SIZE
    x = tl.load(
        value_ptr + token * stride_t + head * stride_h + d * stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    q_min = tl.min(x, axis=0)
    q_max = tl.max(x, axis=0)
    q_scale = (q_max - q_min) / 3.0
    q_scale = tl.where(tl.abs(q_scale) < 1.0e-8, 1.0, q_scale)
    out = (token * NUM_HEADS + head) * NUM_GROUPS + group
    tl.store(min_ptr + out, q_min)
    tl.store(scale_ptr + out, q_scale)


@triton.jit
def _rabit2_v_pack_kernel(
    value_ptr,
    min_ptr,
    scale_ptr,
    payload_ptr,
    stride_t: tl.int64,
    stride_h: tl.int64,
    stride_d: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    b = tl.arange(0, BLOCK_B)
    mask_b = b < PACKED_DIM
    base_d = b * 4
    group = base_d // 32
    pidx = (token * NUM_HEADS + head) * NUM_GROUPS + group
    mn = tl.load(min_ptr + pidx, mask=mask_b, other=0.0)
    sc = tl.load(scale_ptr + pidx, mask=mask_b, other=1.0)
    byte = tl.zeros([BLOCK_B], dtype=tl.int32)
    for j in range(4):
        d = base_d + j
        valid = mask_b & (d < HEAD_SIZE)
        x = tl.load(
            value_ptr + token * stride_t + head * stride_h + d * stride_d,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        q = _rabit2_round_to_even_nonnegative((x - mn) / sc).to(tl.int32)
        q = tl.maximum(0, tl.minimum(3, q))
        byte = byte | (q << (2 * j))
    out = (token * NUM_HEADS + head) * PACKED_DIM + b
    tl.store(payload_ptr + out, byte.to(tl.uint8), mask=mask_b)


@triton.jit
def _rabit2_meta_encode_kernel(
    primary_ptr,
    code_ptr,
    secondary_min_ptr,
    secondary_scale_ptr,
    VALUE_COUNT: tl.constexpr,
):
    group = tl.program_id(0)
    offs = tl.arange(0, 64)
    idx = group * 64 + offs
    # Stage-3B1 is restricted to the selected aligned Llama layout, where
    # primary value counts are exact multiples of META8g64.
    x = tl.load(primary_ptr + idx).to(tl.float32)
    mn = tl.min(x, axis=0)
    mx = tl.max(x, axis=0)
    sc = (mx - mn) / 255.0
    sc = tl.where(tl.abs(sc) < 1.0e-12, 1.0, sc)
    q = _rabit2_round_to_even_nonnegative((x - mn) / sc).to(tl.int32)
    q = tl.maximum(0, tl.minimum(255, q))
    tl.store(code_ptr + idx, q.to(tl.uint8))
    tl.store(secondary_min_ptr + group, mn.to(tl.bfloat16))
    tl.store(secondary_scale_ptr + group, sc.to(tl.bfloat16))


@triton.jit
def _rabit2_page_decode_kernel(
    k_payload_ptr,
    v_payload_ptr,
    kmin_code_ptr,
    kmin_secmin_ptr,
    kmin_secscale_ptr,
    kscale_code_ptr,
    kscale_secmin_ptr,
    kscale_secscale_ptr,
    vmin_code_ptr,
    vmin_secmin_ptr,
    vmin_secscale_ptr,
    vscale_code_ptr,
    vscale_secmin_ptr,
    vscale_secscale_ptr,
    out_k_ptr,
    out_v_ptr,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    K_PACKED_DIM: tl.constexpr,
    V_PACKED_DIM: tl.constexpr,
    V_GROUPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    mask = d < HEAD_SIZE

    # K3 payload.
    bit_pos = d * 3
    byte_idx = bit_pos // 8
    bit_shift = bit_pos % 8
    krow = (token * NUM_HEADS + head) * K_PACKED_DIM
    lo = tl.load(k_payload_ptr + krow + byte_idx, mask=mask, other=0).to(tl.int32)
    hi_mask = mask & ((byte_idx + 1) < K_PACKED_DIM)
    hi = tl.load(k_payload_ptr + krow + byte_idx + 1, mask=hi_mask, other=0).to(tl.int32)
    k_code = ((lo | (hi << 8)) >> bit_shift) & 7

    # K primary metadata index: one G32 sequence group per physical page.
    kidx = head * HEAD_SIZE + d
    kg = kidx // 64
    kmin_primary = (
        tl.load(kmin_code_ptr + kidx, mask=mask, other=0).to(tl.float32)
        * tl.load(kmin_secscale_ptr + kg, mask=mask, other=1.0).to(tl.float32)
        + tl.load(kmin_secmin_ptr + kg, mask=mask, other=0.0).to(tl.float32)
    )
    kscale_primary = (
        tl.load(kscale_code_ptr + kidx, mask=mask, other=0).to(tl.float32)
        * tl.load(kscale_secscale_ptr + kg, mask=mask, other=1.0).to(tl.float32)
        + tl.load(kscale_secmin_ptr + kg, mask=mask, other=0.0).to(tl.float32)
    )
    kval = k_code.to(tl.float32) * kscale_primary + kmin_primary

    # V2 payload and per-token/head/G32 metadata.
    vbyte = d // 4
    vshift = (d % 4) * 2
    vrow = (token * NUM_HEADS + head) * V_PACKED_DIM
    vb = tl.load(v_payload_ptr + vrow + vbyte, mask=mask, other=0).to(tl.int32)
    v_code = (vb >> vshift) & 3
    vgdim = d // 32
    vidx = (token * NUM_HEADS + head) * V_GROUPS + vgdim
    vmg = vidx // 64
    vmin_primary = (
        tl.load(vmin_code_ptr + vidx, mask=mask, other=0).to(tl.float32)
        * tl.load(vmin_secscale_ptr + vmg, mask=mask, other=1.0).to(tl.float32)
        + tl.load(vmin_secmin_ptr + vmg, mask=mask, other=0.0).to(tl.float32)
    )
    vscale_primary = (
        tl.load(vscale_code_ptr + vidx, mask=mask, other=0).to(tl.float32)
        * tl.load(vscale_secscale_ptr + vmg, mask=mask, other=1.0).to(tl.float32)
        + tl.load(vscale_secmin_ptr + vmg, mask=mask, other=0.0).to(tl.float32)
    )
    vval = v_code.to(tl.float32) * vscale_primary + vmin_primary

    out_base = (token * NUM_HEADS + head) * HEAD_SIZE + d
    tl.store(out_k_ptr + out_base, kval, mask=mask)
    tl.store(out_v_ptr + out_base, vval, mask=mask)


@triton.jit
def _rabit2_closed_page_attention_kernel(
    q_ptr,
    k_payload_ptr,
    v_payload_ptr,
    kmin_code_ptr,
    kmin_secmin_ptr,
    kmin_secscale_ptr,
    kscale_code_ptr,
    kscale_secmin_ptr,
    kscale_secscale_ptr,
    vmin_code_ptr,
    vmin_secmin_ptr,
    vmin_secscale_ptr,
    vscale_code_ptr,
    vscale_secmin_ptr,
    vscale_secscale_ptr,
    out_ptr,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    K_PACKED_DIM: tl.constexpr,
    V_PACKED_DIM: tl.constexpr,
    V_GROUPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    qh = tl.program_id(0)
    kvh = qh // (NUM_Q_HEADS // NUM_KV_HEADS)
    t = tl.arange(0, 32)
    d = tl.arange(0, BLOCK_D)
    dmask = d < HEAD_SIZE
    q = tl.load(q_ptr + qh * HEAD_SIZE + d, mask=dmask, other=0.0).to(tl.float32)

    # Decode K [32, D] directly from packed page bytes.
    bit_pos = d * 3
    kbyte = bit_pos // 8
    kshift = bit_pos % 8
    krow = (t[:, None] * NUM_KV_HEADS + kvh) * K_PACKED_DIM
    klo = tl.load(k_payload_ptr + krow + kbyte[None, :], mask=dmask[None, :], other=0).to(tl.int32)
    khimask = dmask[None, :] & ((kbyte[None, :] + 1) < K_PACKED_DIM)
    khi = tl.load(k_payload_ptr + krow + kbyte[None, :] + 1, mask=khimask, other=0).to(tl.int32)
    kcode = ((klo | (khi << 8)) >> kshift[None, :]) & 7

    kidx = kvh * HEAD_SIZE + d
    kg = kidx // 64
    kmin = (
        tl.load(kmin_code_ptr + kidx, mask=dmask, other=0).to(tl.float32)
        * tl.load(kmin_secscale_ptr + kg, mask=dmask, other=1.0).to(tl.float32)
        + tl.load(kmin_secmin_ptr + kg, mask=dmask, other=0.0).to(tl.float32)
    )
    kscale = (
        tl.load(kscale_code_ptr + kidx, mask=dmask, other=0).to(tl.float32)
        * tl.load(kscale_secscale_ptr + kg, mask=dmask, other=1.0).to(tl.float32)
        + tl.load(kscale_secmin_ptr + kg, mask=dmask, other=0.0).to(tl.float32)
    )
    kval = kcode.to(tl.float32) * kscale[None, :] + kmin[None, :]
    scores = tl.sum(kval * q[None, :], axis=1) * softmax_scale
    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    p = p / tl.sum(p, axis=0)

    # Decode V [32, D] inline.
    vbyte = d // 4
    vshift = (d % 4) * 2
    vrow = (t[:, None] * NUM_KV_HEADS + kvh) * V_PACKED_DIM
    vb = tl.load(v_payload_ptr + vrow + vbyte[None, :], mask=dmask[None, :], other=0).to(tl.int32)
    vcode = (vb >> vshift[None, :]) & 3
    vgd = d // 32
    vidx = (t[:, None] * NUM_KV_HEADS + kvh) * V_GROUPS + vgd[None, :]
    vmg = vidx // 64
    vmin = (
        tl.load(vmin_code_ptr + vidx, mask=dmask[None, :], other=0).to(tl.float32)
        * tl.load(vmin_secscale_ptr + vmg, mask=dmask[None, :], other=1.0).to(tl.float32)
        + tl.load(vmin_secmin_ptr + vmg, mask=dmask[None, :], other=0.0).to(tl.float32)
    )
    vscale = (
        tl.load(vscale_code_ptr + vidx, mask=dmask[None, :], other=0).to(tl.float32)
        * tl.load(vscale_secscale_ptr + vmg, mask=dmask[None, :], other=1.0).to(tl.float32)
        + tl.load(vscale_secmin_ptr + vmg, mask=dmask[None, :], other=0.0).to(tl.float32)
    )
    vval = vcode.to(tl.float32) * vscale + vmin
    acc = tl.sum(p[:, None] * vval, axis=0)
    tl.store(out_ptr + qh * HEAD_SIZE + d, acc, mask=dmask)


def _rabit2_stage3b1_validate_layout(layout: Rabit2PageLayout) -> tuple[int, int, int, int]:
    if layout.block_size != RABIT2_GROUP_SIZE:
        raise ValueError("Stage-3B1 closed-page kernels currently require block_size=32")
    if layout.head_size_k != layout.head_size_v:
        raise ValueError("Stage-3B1 currently requires equal K/V head dimensions")
    if layout.head_size_k % RABIT2_GROUP_SIZE:
        raise ValueError("Stage-3B1 requires head_size to be a multiple of 32")
    k_primary = layout.num_kv_heads * layout.head_size_k
    v_groups = layout.head_size_v // RABIT2_GROUP_SIZE
    v_primary = layout.block_size * layout.num_kv_heads * v_groups
    if k_primary % RABIT2_METADATA_GROUP_SIZE or v_primary % RABIT2_METADATA_GROUP_SIZE:
        raise ValueError("Stage-3B1 requires META8g64-aligned primary metadata")
    k_packed = rabit2_packed_dim(layout.head_size_k, 3)
    v_packed = rabit2_packed_dim(layout.head_size_v, 2)
    return k_primary, v_primary, k_packed, v_packed


def _rabit2_encode_meta_triton(primary: torch.Tensor):
    values = int(primary.numel())
    if values % RABIT2_METADATA_GROUP_SIZE:
        raise ValueError("Stage-3B1 metadata encoder requires a multiple of 64 values")
    groups = values // RABIT2_METADATA_GROUP_SIZE
    codes = torch.empty(values, dtype=torch.uint8, device=primary.device)
    mins = torch.empty(groups, dtype=torch.bfloat16, device=primary.device)
    scales = torch.empty(groups, dtype=torch.bfloat16, device=primary.device)
    _rabit2_meta_encode_kernel[(groups,)](
        primary.reshape(-1), codes, mins, scales, VALUE_COUNT=values, num_warps=1
    )
    return codes, mins, scales


def encode_rabit2_page_triton_experimental(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout: Rabit2PageLayout | None = None,
) -> torch.Tensor:
    """Triton quantize+pack path for one *closed* 32-token RABIT page.

    Payload quantization/packing and META8g64 reductions execute in Triton.
    Final concatenation into the opaque page is a set of device-to-device
    slice copies; there is no CPU tensor materialization.
    """
    if not key.is_cuda or not value.is_cuda:
        raise ValueError("Stage-3B1 Triton codec requires CUDA tensors")
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key/value must share [tokens, heads, head_dim]")
    tokens, heads, head_dim = key.shape
    if layout is None:
        layout = rabit2_page_layout(
            block_size=tokens,
            num_kv_heads=heads,
            head_size_k=head_dim,
            head_size_v=value.shape[-1],
        )
    k_primary_n, v_primary_n, k_packed_dim, v_packed_dim = _rabit2_stage3b1_validate_layout(layout)
    if tuple(key.shape) != (layout.block_size, layout.num_kv_heads, layout.head_size_k):
        raise ValueError("key shape does not match layout")

    k_min_primary = torch.empty(k_primary_n, dtype=torch.float32, device=key.device)
    k_scale_primary = torch.empty_like(k_min_primary)
    k_block_d = min(128, triton.next_power_of_2(layout.head_size_k))
    _rabit2_k_primary_kernel[
        (layout.num_kv_heads, triton.cdiv(layout.head_size_k, k_block_d))
    ](
        key,
        k_min_primary,
        k_scale_primary,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        HEAD_SIZE=layout.head_size_k,
        BLOCK_D=k_block_d,
        num_warps=4,
    )

    k_payload = torch.empty(layout.k_payload_bytes, dtype=torch.uint8, device=key.device)
    k_block_b = triton.next_power_of_2(k_packed_dim)
    _rabit2_k_pack_kernel[(layout.block_size, layout.num_kv_heads)](
        key,
        k_min_primary,
        k_scale_primary,
        k_payload,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        NUM_HEADS=layout.num_kv_heads,
        HEAD_SIZE=layout.head_size_k,
        PACKED_DIM=k_packed_dim,
        BLOCK_B=k_block_b,
        num_warps=4,
    )

    v_groups = layout.head_size_v // RABIT2_GROUP_SIZE
    v_min_primary = torch.empty(v_primary_n, dtype=torch.float32, device=value.device)
    v_scale_primary = torch.empty_like(v_min_primary)
    _rabit2_v_primary_kernel[(layout.block_size, layout.num_kv_heads, v_groups)](
        value,
        v_min_primary,
        v_scale_primary,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        NUM_HEADS=layout.num_kv_heads,
        HEAD_SIZE=layout.head_size_v,
        NUM_GROUPS=v_groups,
        num_warps=1,
    )
    v_payload = torch.empty(layout.v_payload_bytes, dtype=torch.uint8, device=value.device)
    v_block_b = triton.next_power_of_2(v_packed_dim)
    _rabit2_v_pack_kernel[(layout.block_size, layout.num_kv_heads)](
        value,
        v_min_primary,
        v_scale_primary,
        v_payload,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        NUM_HEADS=layout.num_kv_heads,
        HEAD_SIZE=layout.head_size_v,
        NUM_GROUPS=v_groups,
        PACKED_DIM=v_packed_dim,
        BLOCK_B=v_block_b,
        num_warps=2,
    )

    kmin = _rabit2_encode_meta_triton(k_min_primary)
    kscale = _rabit2_encode_meta_triton(k_scale_primary)
    vmin = _rabit2_encode_meta_triton(v_min_primary)
    vscale = _rabit2_encode_meta_triton(v_scale_primary)

    page = torch.empty(layout.page_bytes, dtype=torch.uint8, device=key.device)
    page[layout.k_payload_offset : layout.k_payload_offset + layout.k_payload_bytes].copy_(k_payload)
    page[layout.v_payload_offset : layout.v_payload_offset + layout.v_payload_bytes].copy_(v_payload)
    for offset, parts in (
        (layout.k_min_offset, kmin),
        (layout.k_scale_offset, kscale),
        (layout.v_min_offset, vmin),
        (layout.v_scale_offset, vscale),
    ):
        codes, mins, scales = parts
        pos = offset
        page[pos : pos + codes.numel()].copy_(codes)
        pos += codes.numel()
        min_bytes = mins.contiguous().view(torch.uint8)
        page[pos : pos + min_bytes.numel()].copy_(min_bytes)
        pos += min_bytes.numel()
        scale_bytes = scales.contiguous().view(torch.uint8)
        page[pos : pos + scale_bytes.numel()].copy_(scale_bytes)
    return page



def encode_rabit2_page_exact_cuda(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout: Rabit2PageLayout | None = None,
) -> torch.Tensor:
    """Exact GPU writer used for Stage-3B1/3B2 integration.

    This deliberately reuses the Stage-3A CUDA/PyTorch byte-exact quantizer.
    The performance-critical packed read/dequant/attention path is Triton.
    Closed-page writes happen only once per 32 aged tokens, so keeping this
    path exact is preferable to accepting silent rounding drift before the
    end-to-end engine is working and profiled.
    """
    if not key.is_cuda or not value.is_cuda:
        raise ValueError("RABIT-2 exact CUDA writer requires CUDA tensors")
    return encode_rabit2_page_ref(key, value, layout=layout)


def _rabit2_page_views(page: torch.Tensor, layout: Rabit2PageLayout):
    k_primary_n, v_primary_n, k_packed_dim, v_packed_dim = _rabit2_stage3b1_validate_layout(layout)
    flat = page.reshape(-1)
    k_payload = flat[layout.k_payload_offset : layout.k_payload_offset + layout.k_payload_bytes]
    v_payload = flat[layout.v_payload_offset : layout.v_payload_offset + layout.v_payload_bytes]
    kmin = _rabit2_meta_parts(flat, layout.k_min_offset, k_primary_n)
    kscale = _rabit2_meta_parts(flat, layout.k_scale_offset, k_primary_n)
    vmin = _rabit2_meta_parts(flat, layout.v_min_offset, v_primary_n)
    vscale = _rabit2_meta_parts(flat, layout.v_scale_offset, v_primary_n)
    return k_payload, v_payload, kmin, kscale, vmin, vscale, k_packed_dim, v_packed_dim


def decode_rabit2_page_triton(
    page: torch.Tensor,
    *,
    layout: Rabit2PageLayout,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a closed physical page with Triton (no Python unpack/dequant)."""
    if not page.is_cuda or page.dtype != torch.uint8:
        raise ValueError("Stage-3B1 Triton decoder requires a CUDA uint8 page")
    if page.numel() != layout.page_bytes:
        raise ValueError("page size does not match layout")
    views = _rabit2_page_views(page, layout)
    k_payload, v_payload, kmin, kscale, vmin, vscale, k_packed_dim, v_packed_dim = views
    out_k = torch.empty(
        (layout.block_size, layout.num_kv_heads, layout.head_size_k),
        dtype=torch.float32,
        device=page.device,
    )
    out_v = torch.empty_like(out_k)
    block_d = triton.next_power_of_2(layout.head_size_k)
    _rabit2_page_decode_kernel[(layout.block_size, layout.num_kv_heads)](
        k_payload,
        v_payload,
        *kmin,
        *kscale,
        *vmin,
        *vscale,
        out_k,
        out_v,
        NUM_HEADS=layout.num_kv_heads,
        HEAD_SIZE=layout.head_size_k,
        K_PACKED_DIM=k_packed_dim,
        V_PACKED_DIM=v_packed_dim,
        V_GROUPS=layout.head_size_v // RABIT2_GROUP_SIZE,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out_k.to(dtype), out_v.to(dtype)


def rabit2_closed_page_attention_triton(
    query: torch.Tensor,
    page: torch.Tensor,
    *,
    layout: Rabit2PageLayout,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Fused packed-read/dequant attention over one immutable closed page.

    This kernel is the Stage-3B1 proof that K3/V2 payload and META8g64 can be
    consumed *in attention* without first materializing BF16 K/V. It handles
    GQA for one decode-query token. Online open-group/R4 state is deliberately
    excluded until the Stage-3B2 sidecar manager is installed.
    """
    if query.ndim == 3:
        if query.shape[0] != 1:
            raise ValueError("Stage-3B1 fused attention supports one decode query token")
        q = query[0]
    elif query.ndim == 2:
        q = query
    else:
        raise ValueError("query must have shape [q_heads, head_dim] or [1, q_heads, head_dim]")
    if not q.is_cuda or not page.is_cuda:
        raise ValueError("Stage-3B1 fused attention requires CUDA tensors")
    if q.shape[1] != layout.head_size_k:
        raise ValueError("query head dimension does not match page layout")
    if q.shape[0] % layout.num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    views = _rabit2_page_views(page, layout)
    k_payload, v_payload, kmin, kscale, vmin, vscale, k_packed_dim, v_packed_dim = views
    out = torch.empty_like(q)
    block_d = triton.next_power_of_2(layout.head_size_k)
    _rabit2_closed_page_attention_kernel[(q.shape[0],)](
        q,
        k_payload,
        v_payload,
        *kmin,
        *kscale,
        *vmin,
        *vscale,
        out,
        float(softmax_scale),
        NUM_Q_HEADS=q.shape[0],
        NUM_KV_HEADS=layout.num_kv_heads,
        HEAD_SIZE=layout.head_size_k,
        K_PACKED_DIM=k_packed_dim,
        V_PACKED_DIM=v_packed_dim,
        V_GROUPS=layout.head_size_v // RABIT2_GROUP_SIZE,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return out
# === RABIT2_STAGE3B1_TRITON_END ===
# === RABIT2_STAGE3B2_SINGLESEQ_BEGIN ===
# Stage 3B2: exact online sidecar lifecycle + direct raw-page Triton decode attention.
# Scope: one active sequence per batch, unchunked initial prefill, causal decoder attention.
# Closed pages live only in vLLM's opaque physical KV cache; no duplicate Python page list.


@triton.jit
def _rabit2_u8pair_to_bf16_f32(base_ptr, byte_offset, mask):
    lo = tl.load(base_ptr + byte_offset, mask=mask, other=0).to(tl.uint16)
    hi = tl.load(base_ptr + byte_offset + 1, mask=mask, other=0).to(tl.uint16)
    bits = lo | (hi << 8)
    return tl.cast(bits, tl.bfloat16, bitcast=True).to(tl.float32)


@triton.jit
def _rabit2_closed_page_partial_kernel(
    q_ptr,
    cache_ptr,
    block_table_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    softmax_scale,
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
    logical_page = tl.program_id(0)
    qh = tl.program_id(1)
    kvh = qh // (NUM_Q_HEADS // NUM_KV_HEADS)
    physical_page = tl.load(block_table_ptr + logical_page).to(tl.int64)
    page = cache_ptr + physical_page * PAGE_BYTES

    t = tl.arange(0, 32)
    d = tl.arange(0, BLOCK_D)
    dmask = d < HEAD_SIZE
    q = tl.load(q_ptr + qh * HEAD_SIZE + d, mask=dmask, other=0.0).to(tl.float32)

    # K3 payload.
    k_bit = d * 3
    k_byte = k_bit // 8
    k_shift = k_bit % 8
    k_row = (t[:, None] * NUM_KV_HEADS + kvh) * K_PACKED_DIM
    k_lo = tl.load(
        page + K_PAYLOAD_OFFSET + k_row + k_byte[None, :],
        mask=dmask[None, :], other=0,
    ).to(tl.int32)
    k_hi = tl.load(
        page + K_PAYLOAD_OFFSET + k_row + k_byte[None, :] + 1,
        mask=dmask[None, :] & ((k_byte[None, :] + 1) < K_PACKED_DIM), other=0,
    ).to(tl.int32)
    k_code = ((k_lo | (k_hi << 8)) >> k_shift[None, :]) & 7

    # K META8g64: one primary value per (kv_head, channel).
    k_idx = kvh * HEAD_SIZE + d
    k_mg = k_idx // 64

    kmin_code = tl.load(page + K_MIN_OFFSET + k_idx, mask=dmask, other=0).to(tl.float32)
    kmin_secmin_off = K_MIN_OFFSET + K_PRIMARY_COUNT + k_mg * 2
    kmin_secscale_off = K_MIN_OFFSET + K_PRIMARY_COUNT + K_META_GROUPS * 2 + k_mg * 2
    kmin_secmin = _rabit2_u8pair_to_bf16_f32(page, kmin_secmin_off, dmask)
    kmin_secscale = _rabit2_u8pair_to_bf16_f32(page, kmin_secscale_off, dmask)
    k_min = kmin_code * kmin_secscale + kmin_secmin

    kscale_code = tl.load(page + K_SCALE_OFFSET + k_idx, mask=dmask, other=0).to(tl.float32)
    kscale_secmin_off = K_SCALE_OFFSET + K_PRIMARY_COUNT + k_mg * 2
    kscale_secscale_off = K_SCALE_OFFSET + K_PRIMARY_COUNT + K_META_GROUPS * 2 + k_mg * 2
    kscale_secmin = _rabit2_u8pair_to_bf16_f32(page, kscale_secmin_off, dmask)
    kscale_secscale = _rabit2_u8pair_to_bf16_f32(page, kscale_secscale_off, dmask)
    k_scale = kscale_code * kscale_secscale + kscale_secmin

    kval = k_code.to(tl.float32) * k_scale[None, :] + k_min[None, :]
    scores = tl.sum(kval * q[None, :], axis=1) * softmax_scale
    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    l = tl.sum(p, axis=0)

    # V2 payload + META8g64.
    v_byte = d // 4
    v_shift = (d % 4) * 2
    v_row = (t[:, None] * NUM_KV_HEADS + kvh) * V_PACKED_DIM
    vb = tl.load(
        page + V_PAYLOAD_OFFSET + v_row + v_byte[None, :],
        mask=dmask[None, :], other=0,
    ).to(tl.int32)
    v_code = (vb >> v_shift[None, :]) & 3
    vg = d // 32
    v_idx = (t[:, None] * NUM_KV_HEADS + kvh) * V_GROUPS + vg[None, :]
    v_mg = v_idx // 64

    vmin_code = tl.load(page + V_MIN_OFFSET + v_idx, mask=dmask[None, :], other=0).to(tl.float32)
    vmin_secmin_off = V_MIN_OFFSET + V_PRIMARY_COUNT + v_mg * 2
    vmin_secscale_off = V_MIN_OFFSET + V_PRIMARY_COUNT + V_META_GROUPS * 2 + v_mg * 2
    vmin_secmin = _rabit2_u8pair_to_bf16_f32(page, vmin_secmin_off, dmask[None, :])
    vmin_secscale = _rabit2_u8pair_to_bf16_f32(page, vmin_secscale_off, dmask[None, :])
    v_min = vmin_code * vmin_secscale + vmin_secmin

    vscale_code = tl.load(page + V_SCALE_OFFSET + v_idx, mask=dmask[None, :], other=0).to(tl.float32)
    vscale_secmin_off = V_SCALE_OFFSET + V_PRIMARY_COUNT + v_mg * 2
    vscale_secscale_off = V_SCALE_OFFSET + V_PRIMARY_COUNT + V_META_GROUPS * 2 + v_mg * 2
    vscale_secmin = _rabit2_u8pair_to_bf16_f32(page, vscale_secmin_off, dmask[None, :])
    vscale_secscale = _rabit2_u8pair_to_bf16_f32(page, vscale_secscale_off, dmask[None, :])
    v_scale = vscale_code * vscale_secscale + vscale_secmin
    vval = v_code.to(tl.float32) * v_scale + v_min

    acc = tl.sum(p[:, None] * vval, axis=0)
    seg = logical_page * NUM_Q_HEADS + qh
    tl.store(partial_m_ptr + seg, m)
    tl.store(partial_l_ptr + seg, l)
    tl.store(partial_acc_ptr + seg * HEAD_SIZE + d, acc, mask=dmask)


@triton.jit
def _rabit2_tail_partial_kernel(
    q_ptr,
    tail_k_ptr,
    tail_v_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    tail_len,
    segment_idx,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    qh = tl.program_id(0)
    kvh = qh // (NUM_Q_HEADS // NUM_KV_HEADS)
    t = tl.arange(0, BLOCK_T)
    d = tl.arange(0, BLOCK_D)
    tmask = t < tail_len
    dmask = d < HEAD_SIZE
    q = tl.load(q_ptr + qh * HEAD_SIZE + d, mask=dmask, other=0.0).to(tl.float32)
    base = (t[:, None] * NUM_KV_HEADS + kvh) * HEAD_SIZE + d[None, :]
    kval = tl.load(tail_k_ptr + base, mask=tmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
    vval = tl.load(tail_v_ptr + base, mask=tmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
    scores = tl.sum(kval * q[None, :], axis=1) * softmax_scale
    scores = tl.where(tmask, scores, float("-inf"))
    m = tl.max(scores, axis=0)
    p = tl.where(tmask, tl.exp(scores - m), 0.0)
    l = tl.sum(p, axis=0)
    acc = tl.sum(p[:, None] * vval, axis=0)
    seg = segment_idx * NUM_Q_HEADS + qh
    tl.store(partial_m_ptr + seg, m)
    tl.store(partial_l_ptr + seg, l)
    tl.store(partial_acc_ptr + seg * HEAD_SIZE + d, acc, mask=dmask)


@triton.jit
def _rabit2_reduce_partials_kernel(
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    out_ptr,
    num_segments,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
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


class Rabit2SingleSequenceRuntime:
    """Exact bounded sidecar for one active decoder sequence.

    Closed pages are written directly to the vLLM opaque KV cache and are not
    retained here, so the sidecar memory remains O(1) in context length.
    """

    def __init__(self, num_kv_heads: int, head_size_k: int, head_size_v: int):
        self.num_kv_heads = int(num_kv_heads)
        self.head_size_k = int(head_size_k)
        self.head_size_v = int(head_size_v)
        self.block_size = int(RABIT2_GROUP_SIZE)
        self.residual_tokens = int(RABIT2_RESIDUAL_TOKENS)
        self.layout = rabit2_page_layout(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size_k=self.head_size_k,
            head_size_v=self.head_size_v,
        )
        _rabit2_stage3b1_validate_layout(self.layout)
        self.closed_pages = 0
        self.open_k = None
        self.open_v_packed = None
        self.open_v_min = None
        self.open_v_scale = None
        self.recent_k = None
        self.recent_v = None

    @property
    def total_tokens(self) -> int:
        return (
            self.closed_pages * self.block_size
            + (0 if self.open_k is None else int(self.open_k.shape[0]))
            + (0 if self.recent_k is None else int(self.recent_k.shape[0]))
        )

    def _closed_page_exact(self) -> torch.Tensor:
        assert self.open_k is not None and self.open_k.shape[0] == self.block_size
        assert self.open_v_packed is not None
        assert self.open_v_min is not None and self.open_v_scale is not None
        v_min_meta = encode_metadata_uint8_group_ref(self.open_v_min)
        v_scale_meta = encode_metadata_uint8_group_ref(self.open_v_scale)
        k_state = quantize_k3_sequence_affine_ref(self.open_k)
        blobs = {
            "k_payload": k_state["packed"].reshape(-1).to(torch.uint8),
            "v_payload": self.open_v_packed.reshape(-1).to(torch.uint8),
            "k_min": encode_metadata_blob_ref(k_state["min"]),
            "k_scale": encode_metadata_blob_ref(k_state["scale"]),
            "v_min": encode_metadata_blob_ref(v_min_meta),
            "v_scale": encode_metadata_blob_ref(v_scale_meta),
        }
        page = torch.zeros(
            self.layout.page_bytes, dtype=torch.uint8, device=self.open_k.device
        )
        for name, offset, size in (
            ("k_payload", self.layout.k_payload_offset, self.layout.k_payload_bytes),
            ("v_payload", self.layout.v_payload_offset, self.layout.v_payload_bytes),
            ("k_min", self.layout.k_min_offset, self.layout.k_min_bytes),
            ("k_scale", self.layout.k_scale_offset, self.layout.k_scale_bytes),
            ("v_min", self.layout.v_min_offset, self.layout.v_min_bytes),
            ("v_scale", self.layout.v_scale_offset, self.layout.v_scale_bytes),
        ):
            if blobs[name].numel() != size:
                raise ValueError(f"{name} does not match RABIT-2 physical layout")
            page[offset : offset + size] = blobs[name]
        return page

    def _flush_open_page(self, kv_cache: torch.Tensor, block_table_row: torch.Tensor) -> None:
        page = self._closed_page_exact()
        if self.closed_pages >= int(block_table_row.shape[0]):
            raise RuntimeError("RABIT-2 block table is too short for a closed page")
        cache2d = kv_cache.reshape(kv_cache.shape[0], -1)
        if cache2d.shape[1] != self.layout.page_bytes:
            raise ValueError("opaque RABIT-2 KV page size does not match runtime layout")
        physical_id = block_table_row[self.closed_pages : self.closed_pages + 1].to(torch.long)
        cache2d.index_copy_(0, physical_id, page.reshape(1, -1))
        self.closed_pages += 1
        self.open_k = None
        self.open_v_packed = None
        self.open_v_min = None
        self.open_v_scale = None

    def _append_aged(self, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor, block_table_row: torch.Tensor) -> None:
        packed, q_min, scale, _ = _quantize_v2_primary_ref(value)
        k_bf16 = key.to(torch.bfloat16).contiguous()
        self.open_k = k_bf16 if self.open_k is None else torch.cat([self.open_k, k_bf16], dim=0)
        self.open_v_packed = packed if self.open_v_packed is None else torch.cat([self.open_v_packed, packed], dim=0)
        self.open_v_min = q_min if self.open_v_min is None else torch.cat([self.open_v_min, q_min], dim=0)
        self.open_v_scale = scale if self.open_v_scale is None else torch.cat([self.open_v_scale, scale], dim=0)
        if self.open_k.shape[0] == self.block_size:
            self._flush_open_page(kv_cache, block_table_row)

    def append(self, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor, block_table_row: torch.Tensor) -> None:
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("key/value must share [tokens, kv_heads, head_dim]")
        if key.shape[1:] != (self.num_kv_heads, self.head_size_k):
            raise ValueError("key shape does not match RABIT-2 runtime")
        if value.shape[-1] != self.head_size_v:
            raise ValueError("value head dimension does not match RABIT-2 runtime")
        for i in range(key.shape[0]):
            k_one = key[i : i + 1].to(torch.bfloat16).contiguous()
            v_one = value[i : i + 1].to(torch.bfloat16).contiguous()
            self.recent_k = k_one if self.recent_k is None else torch.cat([self.recent_k, k_one], dim=0)
            self.recent_v = v_one if self.recent_v is None else torch.cat([self.recent_v, v_one], dim=0)
            if self.recent_k.shape[0] > self.residual_tokens:
                old_k = self.recent_k[:1]
                old_v = self.recent_v[:1]
                self.recent_k = self.recent_k[1:].contiguous()
                self.recent_v = self.recent_v[1:].contiguous()
                self._append_aged(old_k, old_v, kv_cache, block_table_row)

    def tail_materialize(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        k_parts = []
        v_parts = []
        if self.open_k is not None:
            assert self.open_v_packed is not None
            assert self.open_v_min is not None and self.open_v_scale is not None
            k_state = quantize_k3_sequence_affine_ref(self.open_k)
            k_parts.append(dequantize_k3_sequence_affine_ref(k_state, dtype=dtype))
            v_min_meta = encode_metadata_uint8_group_ref(self.open_v_min)
            v_scale_meta = encode_metadata_uint8_group_ref(self.open_v_scale)
            v_min = decode_metadata_uint8_group_ref(v_min_meta)
            v_scale = decode_metadata_uint8_group_ref(v_scale_meta)
            v_parts.append(
                _dequantize_v2_from_primary_ref(
                    self.open_v_packed,
                    v_min,
                    v_scale,
                    head_dim=self.head_size_v,
                    dtype=dtype,
                )
            )
        if self.recent_k is not None:
            k_parts.append(self.recent_k.to(dtype))
            v_parts.append(self.recent_v.to(dtype))
        if not k_parts:
            device = kv_cache.device if False else torch.device("cpu")
            return (
                torch.empty((0, self.num_kv_heads, self.head_size_k), dtype=dtype, device=device),
                torch.empty((0, self.num_kv_heads, self.head_size_v), dtype=dtype, device=device),
            )
        return torch.cat(k_parts, dim=0).contiguous(), torch.cat(v_parts, dim=0).contiguous()

    def sidecar_bytes(self) -> int:
        total = 0
        for x in (self.open_k, self.open_v_packed, self.open_v_min, self.open_v_scale, self.recent_k, self.recent_v):
            if x is not None:
                total += x.numel() * x.element_size()
        return int(total)


def rabit2_online_decode_attention_triton(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    runtime: Rabit2SingleSequenceRuntime,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Decode one token from closed packed pages + exact bounded sidecar."""
    if query.ndim != 3 or query.shape[0] != 1:
        raise ValueError("Stage-3B2 online attention requires one decode query token")
    if not query.is_cuda or not kv_cache.is_cuda:
        raise ValueError("Stage-3B2 online attention requires CUDA tensors")
    q = query[0].contiguous()
    if q.shape[1] != runtime.head_size_k or q.shape[0] % runtime.num_kv_heads:
        raise ValueError("query shape is incompatible with RABIT-2 GQA layout")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    tail_k, tail_v = runtime.tail_materialize(dtype=query.dtype)
    if tail_k.device != query.device:
        tail_k = tail_k.to(query.device)
        tail_v = tail_v.to(query.device)
    tail_len = int(tail_k.shape[0])
    if tail_len <= 0 or tail_len > (RABIT2_GROUP_SIZE + RABIT2_RESIDUAL_TOKENS):
        raise RuntimeError(f"invalid RABIT-2 online tail length: {tail_len}")

    closed = int(runtime.closed_pages)
    segments = closed + 1
    q_heads = int(q.shape[0])
    d = int(q.shape[1])
    partial_m = torch.empty((segments, q_heads), dtype=torch.float32, device=q.device)
    partial_l = torch.empty_like(partial_m)
    partial_acc = torch.empty((segments, q_heads, d), dtype=torch.float32, device=q.device)
    layout = runtime.layout
    k_primary = layout.num_kv_heads * layout.head_size_k
    k_meta_groups = k_primary // RABIT2_METADATA_GROUP_SIZE
    v_groups = layout.head_size_v // RABIT2_GROUP_SIZE
    v_primary = layout.block_size * layout.num_kv_heads * v_groups
    v_meta_groups = v_primary // RABIT2_METADATA_GROUP_SIZE
    block_d = triton.next_power_of_2(d)

    if closed:
        _rabit2_closed_page_partial_kernel[(closed, q_heads)](
            q,
            kv_cache,
            block_table_row,
            partial_m,
            partial_l,
            partial_acc,
            float(softmax_scale),
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
            NUM_KV_HEADS=layout.num_kv_heads,
            HEAD_SIZE=d,
            K_PACKED_DIM=rabit2_packed_dim(d, 3),
            V_PACKED_DIM=rabit2_packed_dim(d, 2),
            V_GROUPS=v_groups,
            BLOCK_D=block_d,
            num_warps=8,
        )

    _rabit2_tail_partial_kernel[(q_heads,)](
        q,
        tail_k,
        tail_v,
        partial_m,
        partial_l,
        partial_acc,
        tail_len,
        closed,
        float(softmax_scale),
        NUM_Q_HEADS=q_heads,
        NUM_KV_HEADS=layout.num_kv_heads,
        HEAD_SIZE=d,
        BLOCK_T=64,
        BLOCK_D=block_d,
        num_warps=8,
    )

    out = torch.empty_like(q)
    block_s = triton.next_power_of_2(segments)
    _rabit2_reduce_partials_kernel[(q_heads,)](
        partial_m,
        partial_l,
        partial_acc,
        out,
        segments,
        NUM_Q_HEADS=q_heads,
        HEAD_SIZE=d,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return out.unsqueeze(0)

# === RABIT2_STAGE3B2_SINGLESEQ_END ===
