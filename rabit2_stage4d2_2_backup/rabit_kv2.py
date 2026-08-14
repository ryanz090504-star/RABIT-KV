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

# === RABIT2_STAGE3C_LIFECYCLE_BEGIN ===
# Stage 3C scheduler/attention bridge. The GPU model runner publishes the final
# post-reorder request order and CPU-known query/context lengths once per engine
# step, so every attention layer can bind sidecars without GPU->CPU metadata syncs.
import weakref as _rabit2_weakref

_RABIT2_ACTIVE_BATCH: tuple[tuple[str, int, int], ...] = ()
_RABIT2_ATTENTION_IMPLS = _rabit2_weakref.WeakSet()


def rabit2_register_attention_impl(impl) -> None:
    _RABIT2_ATTENTION_IMPLS.add(impl)


def rabit2_set_active_batch(req_ids, query_lens, context_lens) -> None:
    global _RABIT2_ACTIVE_BATCH
    ids = tuple(str(x) for x in req_ids)
    qlens = tuple(int(x) for x in query_lens)
    clens = tuple(int(x) for x in context_lens)
    if not (len(ids) == len(qlens) == len(clens)):
        raise ValueError("RABIT-2 active batch fields must have equal length")
    if any(q <= 0 for q in qlens):
        raise ValueError("RABIT-2 active query lengths must be positive")
    if any(c < 0 for c in clens):
        raise ValueError("RABIT-2 active context lengths must be non-negative")
    _RABIT2_ACTIVE_BATCH = tuple(zip(ids, qlens, clens))
    # Synthetic warmup runtimes must never alias real requests.
    if ids:
        for impl in list(_RABIT2_ATTENTION_IMPLS):
            table = getattr(impl, "_rabit2_runtimes", None)
            if table is not None:
                for key in tuple(table):
                    if str(key).startswith("__rabit_profile_"):
                        table.pop(key, None)


def rabit2_get_active_batch() -> tuple[tuple[str, int, int], ...]:
    return _RABIT2_ACTIVE_BATCH


def rabit2_get_active_request_ids() -> tuple[str, ...]:
    return tuple(x[0] for x in _RABIT2_ACTIVE_BATCH)


def rabit2_finish_requests(req_ids) -> int:
    finished = tuple(str(x) for x in req_ids)
    if not finished:
        return 0
    removed = 0
    for impl in list(_RABIT2_ATTENTION_IMPLS):
        table = getattr(impl, "_rabit2_runtimes", None)
        if table is None:
            continue
        for req_id in finished:
            if table.pop(req_id, None) is not None:
                removed += 1
    return removed


def rabit2_total_live_sidecars() -> int:
    total = 0
    for impl in list(_RABIT2_ATTENTION_IMPLS):
        table = getattr(impl, "_rabit2_runtimes", None)
        if table is not None:
            total += len(table)
    return int(total)

# === RABIT2_STAGE3C_LIFECYCLE_END ===

# === RABIT2_STAGE4B1_EXACTMETA_BEGIN ===
_RABIT2_STAGE4B1_EXACTMETA = True
_rabit2_online_decode_attention_triton_stage3c_exact = rabit2_online_decode_attention_triton


@triton.jit
def _rabit2_stage4b1_exactmeta_tail_partial_kernel(
    q_ptr,
    k_packed_ptr,
    open_v_packed_ptr,
    recent_k_ptr,
    recent_v_ptr,
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
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    open_len,
    recent_len,
    segment_idx,
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
    t = tl.arange(0, 64)
    d = tl.arange(0, BLOCK_D)
    dmask = d < HEAD_SIZE
    open_mask = t < open_len
    ridx = t - open_len
    recent_mask = (t >= open_len) & (ridx < recent_len)
    valid = open_mask | recent_mask

    q = tl.load(q_ptr + qh * HEAD_SIZE + d, mask=dmask, other=0.0).to(tl.float32)

    k_byte = (d * 3) // 8
    k_shift = (d * 3) % 8
    k_row = (t[:, None] * NUM_KV_HEADS + kvh) * K_PACKED_DIM
    klo = tl.load(
        k_packed_ptr + k_row + k_byte[None, :],
        mask=open_mask[:, None] & dmask[None, :],
        other=0,
    ).to(tl.int32)
    khi = tl.load(
        k_packed_ptr + k_row + k_byte[None, :] + 1,
        mask=open_mask[:, None] & dmask[None, :]
        & ((k_byte[None, :] + 1) < K_PACKED_DIM),
        other=0,
    ).to(tl.int32)
    kcode = ((klo | (khi << 8)) >> k_shift[None, :]) & 7

    kidx = kvh * HEAD_SIZE + d
    kmg = kidx // 64
    kmin_code = tl.load(kmin_code_ptr + kidx, mask=dmask, other=0).to(tl.float32)
    kmin_secmin = tl.load(kmin_secmin_ptr + kmg, mask=dmask, other=0.0).to(tl.float32)
    kmin_secscale = tl.load(kmin_secscale_ptr + kmg, mask=dmask, other=1.0).to(tl.float32)
    kmin = kmin_code * kmin_secscale + kmin_secmin

    kscale_code = tl.load(kscale_code_ptr + kidx, mask=dmask, other=0).to(tl.float32)
    kscale_secmin = tl.load(kscale_secmin_ptr + kmg, mask=dmask, other=0.0).to(tl.float32)
    kscale_secscale = tl.load(kscale_secscale_ptr + kmg, mask=dmask, other=1.0).to(tl.float32)
    kscale = kscale_code * kscale_secscale + kscale_secmin
    kopen = (kcode.to(tl.float32) * kscale[None, :] + kmin[None, :]).to(tl.bfloat16).to(tl.float32)

    v_byte = d // 4
    v_shift = (d % 4) * 2
    v_row = (t[:, None] * NUM_KV_HEADS + kvh) * V_PACKED_DIM
    vb = tl.load(
        open_v_packed_ptr + v_row + v_byte[None, :],
        mask=open_mask[:, None] & dmask[None, :],
        other=0,
    ).to(tl.int32)
    vcode = (vb >> v_shift[None, :]) & 3
    vg = d // 32
    vidx = (t[:, None] * NUM_KV_HEADS + kvh) * V_GROUPS + vg[None, :]
    vmg = vidx // 64

    vmin_code = tl.load(vmin_code_ptr + vidx, mask=open_mask[:, None] & dmask[None, :], other=0).to(tl.float32)
    vmin_secmin = tl.load(vmin_secmin_ptr + vmg, mask=open_mask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
    vmin_secscale = tl.load(vmin_secscale_ptr + vmg, mask=open_mask[:, None] & dmask[None, :], other=1.0).to(tl.float32)
    vmin = vmin_code * vmin_secscale + vmin_secmin

    vscale_code = tl.load(vscale_code_ptr + vidx, mask=open_mask[:, None] & dmask[None, :], other=0).to(tl.float32)
    vscale_secmin = tl.load(vscale_secmin_ptr + vmg, mask=open_mask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
    vscale_secscale = tl.load(vscale_secscale_ptr + vmg, mask=open_mask[:, None] & dmask[None, :], other=1.0).to(tl.float32)
    vscale = vscale_code * vscale_secscale + vscale_secmin
    vopen = (vcode.to(tl.float32) * vscale + vmin).to(tl.bfloat16).to(tl.float32)

    rbase = (ridx[:, None] * NUM_KV_HEADS + kvh) * HEAD_SIZE + d[None, :]
    rk = tl.load(recent_k_ptr + rbase, mask=recent_mask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
    rv = tl.load(recent_v_ptr + rbase, mask=recent_mask[:, None] & dmask[None, :], other=0.0).to(tl.float32)

    kval = tl.where(open_mask[:, None], kopen, rk)
    vval = tl.where(open_mask[:, None], vopen, rv)
    scores = tl.sum(kval * q[None, :], axis=1) * softmax_scale
    scores = tl.where(valid, scores, float("-inf"))
    m = tl.max(scores, axis=0)
    p = tl.where(valid, tl.exp(scores - m), 0.0)
    l = tl.sum(p, axis=0)
    acc = tl.sum(p[:, None] * vval, axis=0)

    seg = segment_idx * NUM_Q_HEADS + qh
    tl.store(partial_m_ptr + seg, m)
    tl.store(partial_l_ptr + seg, l)
    tl.store(partial_acc_ptr + seg * HEAD_SIZE + d, acc, mask=dmask)


def _rabit2_stage4b1_exactmeta_emit_tail_partial(
    q, runtime, partial_m, partial_l, partial_acc, segment_idx, softmax_scale
):
    recent_k = runtime.recent_k
    recent_v = runtime.recent_v
    if recent_k is None or recent_v is None:
        raise RuntimeError("RABIT-2 Stage4B1 exactmeta requires recent tokens")

    q_heads = int(q.shape[0])
    d = int(q.shape[1])
    block_d = triton.next_power_of_2(d)
    recent_len = int(recent_k.shape[0])

    if runtime.open_k is None:
        _rabit2_tail_partial_kernel[(q_heads,)](
            q, recent_k, recent_v, partial_m, partial_l, partial_acc,
            recent_len, int(segment_idx), float(softmax_scale),
            NUM_Q_HEADS=q_heads,
            NUM_KV_HEADS=runtime.num_kv_heads,
            HEAD_SIZE=d,
            BLOCK_T=4,
            BLOCK_D=block_d,
            num_warps=4,
        )
        return

    assert runtime.open_v_packed is not None
    assert runtime.open_v_min is not None
    assert runtime.open_v_scale is not None

    k_state = quantize_k3_sequence_affine_ref(runtime.open_k)
    vmin_meta = encode_metadata_uint8_group_ref(runtime.open_v_min)
    vscale_meta = encode_metadata_uint8_group_ref(runtime.open_v_scale)
    kmin_meta = k_state["min"]
    kscale_meta = k_state["scale"]
    v_groups = runtime.head_size_v // RABIT2_GROUP_SIZE

    _rabit2_stage4b1_exactmeta_tail_partial_kernel[(q_heads,)](
        q,
        k_state["packed"].contiguous(),
        runtime.open_v_packed,
        recent_k,
        recent_v,
        kmin_meta["codes"].reshape(-1),
        kmin_meta["min"].reshape(-1),
        kmin_meta["scale"].reshape(-1),
        kscale_meta["codes"].reshape(-1),
        kscale_meta["min"].reshape(-1),
        kscale_meta["scale"].reshape(-1),
        vmin_meta["codes"].reshape(-1),
        vmin_meta["min"].reshape(-1),
        vmin_meta["scale"].reshape(-1),
        vscale_meta["codes"].reshape(-1),
        vscale_meta["min"].reshape(-1),
        vscale_meta["scale"].reshape(-1),
        partial_m, partial_l, partial_acc,
        int(runtime.open_k.shape[0]),
        recent_len,
        int(segment_idx),
        float(softmax_scale),
        NUM_Q_HEADS=q_heads,
        NUM_KV_HEADS=runtime.num_kv_heads,
        HEAD_SIZE=d,
        K_PACKED_DIM=rabit2_packed_dim(d, 3),
        V_PACKED_DIM=rabit2_packed_dim(runtime.head_size_v, 2),
        V_GROUPS=v_groups,
        BLOCK_D=block_d,
        num_warps=8,
    )


def rabit2_online_decode_attention_triton_stage4b1_exactmeta(
    query, kv_cache, block_table_row, runtime, *, softmax_scale=None
):
    if query.ndim != 3 or query.shape[0] != 1:
        raise ValueError("Stage4B1 exactmeta online attention requires one decode query token")
    if not query.is_cuda or not kv_cache.is_cuda:
        raise ValueError("Stage4B1 exactmeta online attention requires CUDA tensors")

    q = query[0].contiguous()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

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
            q, kv_cache, block_table_row,
            partial_m, partial_l, partial_acc,
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

    _rabit2_stage4b1_exactmeta_emit_tail_partial(
        q, runtime, partial_m, partial_l, partial_acc,
        closed, float(softmax_scale)
    )

    out = torch.empty_like(q)
    _rabit2_reduce_partials_kernel[(q_heads,)](
        partial_m, partial_l, partial_acc, out, segments,
        NUM_Q_HEADS=q_heads,
        HEAD_SIZE=d,
        BLOCK_S=triton.next_power_of_2(segments),
        BLOCK_D=block_d,
        num_warps=8,
    )
    return out.unsqueeze(0)


rabit2_online_decode_attention_triton = rabit2_online_decode_attention_triton_stage4b1_exactmeta
# === RABIT2_STAGE4B1_EXACTMETA_END ===

# === RABIT2_STAGE4B2_EXACT_V2_FIXED_BEGIN ===
# Stage 4B2 exact V2 fast path.
#
# IMPORTANT: torch.compile(mode="reduce-overhead") uses CUDA graphs. Returned
# tensors can point into graph-managed output buffers and be overwritten by the
# next invocation. RABIT persists packed/min/scale tensors in its sidecar, so we
# clone graph outputs before they escape this function.
_RABIT2_STAGE4B2_EXACT_V2_FIXED = True
_quantize_v2_primary_ref_stage4b1_exact = _quantize_v2_primary_ref


def _rabit2_stage4b2_stats_8x128(value: torch.Tensor):
    x = value.float()
    grouped = x.reshape(1, 8, 4, 32)
    q_min = grouped.amin(dim=-1, keepdim=True)
    q_max = grouped.amax(dim=-1, keepdim=True)
    scale = (q_max - q_min) / 3.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    return x, q_min.contiguous(), scale.contiguous()


def _rabit2_stage4b2_pack_8x128(codes: torch.Tensor):
    c = codes.reshape(1, 8, 32, 4)
    return (
        c[..., 0]
        | (c[..., 1] << 2)
        | (c[..., 2] << 4)
        | (c[..., 3] << 6)
    ).to(torch.uint8).contiguous()


_rabit2_stage4b2_stats_compiled = torch.compile(
    _rabit2_stage4b2_stats_8x128,
    fullgraph=True,
    mode="reduce-overhead",
)
_rabit2_stage4b2_pack_compiled = torch.compile(
    _rabit2_stage4b2_pack_8x128,
    fullgraph=True,
    mode="reduce-overhead",
)


def _quantize_v2_primary_ref(
    value: torch.Tensor,
    group_size: int = RABIT2_GROUP_SIZE,
):
    if (
        value.is_cuda
        and value.dtype == torch.bfloat16
        and tuple(value.shape) == (1, 8, 128)
        and int(group_size) == 32
    ):
        x, q_min_graph, scale_graph = _rabit2_stage4b2_stats_compiled(value)

        # q_min/scale are persistent sidecar state. Detach them from the
        # CUDA-graph-managed output pool BEFORE invoking another compiled graph.
        q_min = q_min_graph.clone()
        scale = scale_graph.clone()

        # Keep torch.round eager: fully compiling it caused exact packed-byte
        # drift on tie cases.
        grouped = x.reshape(1, 8, 4, 32)
        codes = torch.round((grouped - q_min_graph) / scale_graph).clamp(0, 3).to(torch.uint8)

        # Packed bytes are also persisted by the runtime, so clone them out of
        # the pack graph's reusable output buffer.
        packed = _rabit2_stage4b2_pack_compiled(
            codes.reshape(1, 8, 128)
        ).clone()

        return packed, q_min, scale, 128

    return _quantize_v2_primary_ref_stage4b1_exact(
        value, group_size=group_size
    )
# === RABIT2_STAGE4B2_EXACT_V2_FIXED_END ===

# === RABIT2_STAGE4B3_GQA4_BEGIN ===
# Stage 4B3 final latency path:
# Reuse one packed K3/V2 decode across four query heads that share a KV head.
# For geometries whose GQA ratio is not divisible by 4, fall back to the
# Stage4B1/Stage4B2 exact path.
_RABIT2_STAGE4B3_GQA4 = True
_rabit2_online_decode_attention_triton_stage4b2_exact = (
    rabit2_online_decode_attention_triton
)


@triton.jit
def _rabit2_stage4b3_gqa4_closed_page_partial_kernel(
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


def rabit2_online_decode_attention_triton_stage4b3_gqa4(
    query,
    kv_cache,
    block_table_row,
    runtime,
    *,
    softmax_scale=None,
):
    """Stage4B1 exact tail + Stage4B3 GQA4 closed-page reuse."""
    if query.ndim != 3 or query.shape[0] != 1:
        raise ValueError("Stage4B3 GQA4 online attention requires one decode query token")
    if not query.is_cuda or not kv_cache.is_cuda:
        raise ValueError("Stage4B3 GQA4 online attention requires CUDA tensors")

    q = query[0].contiguous()
    q_heads = int(q.shape[0])
    d = int(q.shape[1])
    kv_heads = int(runtime.num_kv_heads)

    # Generic safety fallback.
    if q_heads % kv_heads or (q_heads // kv_heads) % 4:
        return _rabit2_online_decode_attention_triton_stage4b2_exact(
            query,
            kv_cache,
            block_table_row,
            runtime,
            softmax_scale=softmax_scale,
        )

    if softmax_scale is None:
        softmax_scale = d ** -0.5

    closed = int(runtime.closed_pages)
    segments = closed + 1
    partial_m = torch.empty(
        (segments, q_heads), dtype=torch.float32, device=q.device
    )
    partial_l = torch.empty_like(partial_m)
    partial_acc = torch.empty(
        (segments, q_heads, d), dtype=torch.float32, device=q.device
    )

    layout = runtime.layout
    k_primary = layout.num_kv_heads * layout.head_size_k
    k_meta_groups = k_primary // RABIT2_METADATA_GROUP_SIZE
    v_groups = layout.head_size_v // RABIT2_GROUP_SIZE
    v_primary = layout.block_size * layout.num_kv_heads * v_groups
    v_meta_groups = v_primary // RABIT2_METADATA_GROUP_SIZE
    block_d = triton.next_power_of_2(d)

    if closed:
        groups_per_kv = (q_heads // kv_heads) // 4
        qgroups = kv_heads * groups_per_kv
        _rabit2_stage4b3_gqa4_closed_page_partial_kernel[(closed, qgroups)](
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
            NUM_KV_HEADS=kv_heads,
            HEAD_SIZE=d,
            K_PACKED_DIM=rabit2_packed_dim(d, 3),
            V_PACKED_DIM=rabit2_packed_dim(runtime.head_size_v, 2),
            V_GROUPS=v_groups,
            BLOCK_D=block_d,
            num_warps=8,
        )

    _rabit2_stage4b1_exactmeta_emit_tail_partial(
        q,
        runtime,
        partial_m,
        partial_l,
        partial_acc,
        closed,
        float(softmax_scale),
    )

    out = torch.empty_like(q)
    _rabit2_reduce_partials_kernel[(q_heads,)](
        partial_m,
        partial_l,
        partial_acc,
        out,
        segments,
        NUM_Q_HEADS=q_heads,
        HEAD_SIZE=d,
        BLOCK_S=triton.next_power_of_2(segments),
        BLOCK_D=block_d,
        num_warps=8,
    )
    return out.unsqueeze(0)


rabit2_online_decode_attention_triton = (
    rabit2_online_decode_attention_triton_stage4b3_gqa4
)
# === RABIT2_STAGE4B3_GQA4_END ===

# === RABIT2_STAGE4B4_CAUSAL_PREFILL_BEGIN ===
# Exact bulk/causal-prefill sidecar update.
#
# Initial prefill can install the chunk's final sidecar state immediately because
# dense causal attention consumes the original k_seq/v_seq tensors directly.
#
# Non-initial chunked prefill must preserve the exact state after EVERY token.
# Rabit2CausalChunkPlan precomputes the chunk's future exact representation once,
# writes future closed pages into scheduler-owned physical blocks, then exposes
# only the causally legal prefix via apply_step().
_RABIT2_STAGE4B4_CAUSAL_PREFILL = True


def _rabit2_stage4b4_encode_page_from_primary(
    runtime: Rabit2SingleSequenceRuntime,
    key_page: torch.Tensor,
    v_packed: torch.Tensor,
    v_min: torch.Tensor,
    v_scale: torch.Tensor,
) -> torch.Tensor:
    """Encode one physical page exactly from already-quantized V2 primaries."""
    v_min_meta = encode_metadata_uint8_group_ref(v_min)
    v_scale_meta = encode_metadata_uint8_group_ref(v_scale)
    k_state = quantize_k3_sequence_affine_ref(key_page)

    blobs = {
        "k_payload": k_state["packed"].reshape(-1).to(torch.uint8),
        "v_payload": v_packed.reshape(-1).to(torch.uint8),
        "k_min": encode_metadata_blob_ref(k_state["min"]),
        "k_scale": encode_metadata_blob_ref(k_state["scale"]),
        "v_min": encode_metadata_blob_ref(v_min_meta),
        "v_scale": encode_metadata_blob_ref(v_scale_meta),
    }

    page = torch.zeros(
        runtime.layout.page_bytes,
        dtype=torch.uint8,
        device=key_page.device,
    )
    for name, offset, size in (
        ("k_payload", runtime.layout.k_payload_offset, runtime.layout.k_payload_bytes),
        ("v_payload", runtime.layout.v_payload_offset, runtime.layout.v_payload_bytes),
        ("k_min", runtime.layout.k_min_offset, runtime.layout.k_min_bytes),
        ("k_scale", runtime.layout.k_scale_offset, runtime.layout.k_scale_bytes),
        ("v_min", runtime.layout.v_min_offset, runtime.layout.v_min_bytes),
        ("v_scale", runtime.layout.v_scale_offset, runtime.layout.v_scale_bytes),
    ):
        blob = blobs[name]
        if blob.numel() != size:
            raise RuntimeError(f"RABIT-2 Stage4B4 {name} size mismatch")
        page[offset : offset + size].copy_(blob)
    return page


def _rabit2_stage4b4_exact_v2_batch(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Exact multi-token V2 primary quantization.

    Stage4B2's decode-specialized fast path is intentionally [1,8,128].
    Prefill chunks use the preserved Stage4B1 reference implementation, whose
    token-wise grouping semantics are identical when evaluated as one batch.
    """
    ref = globals().get("_quantize_v2_primary_ref_stage4b1_exact")
    if ref is None:
        ref = _quantize_v2_primary_ref
    return ref(value)


def rabit2_bulk_append_exact(
    runtime: Rabit2SingleSequenceRuntime,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
) -> None:
    """Install the exact FINAL sidecar state of a whole chunk.

    Safe for initial prefill where attention uses the dense input K/V directly.
    Do not use this helper by itself for non-initial causal chunked prefill;
    use Rabit2CausalChunkPlan there.
    """
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key/value must share [tokens, kv_heads, head_dim]")
    if key.shape[1:] != (runtime.num_kv_heads, runtime.head_size_k):
        raise ValueError("key shape does not match RABIT-2 runtime")
    if value.shape[-1] != runtime.head_size_v:
        raise ValueError("value head dimension does not match RABIT-2 runtime")
    if key.shape[0] == 0:
        return

    key = key.to(torch.bfloat16).contiguous()
    value = value.to(torch.bfloat16).contiguous()

    if runtime.recent_k is None:
        combined_recent_k = key
        combined_recent_v = value
    else:
        combined_recent_k = torch.cat((runtime.recent_k, key), dim=0)
        combined_recent_v = torch.cat((runtime.recent_v, value), dim=0)

    aged_n = max(
        0,
        int(combined_recent_k.shape[0]) - int(runtime.residual_tokens),
    )

    if aged_n:
        aged_k = combined_recent_k[:aged_n].contiguous()
        aged_v = combined_recent_v[:aged_n].contiguous()
        vp, vmn, vsc, _ = _rabit2_stage4b4_exact_v2_batch(aged_v)

        if runtime.open_k is None:
            all_k, all_vp, all_vmn, all_vsc = aged_k, vp, vmn, vsc
        else:
            all_k = torch.cat((runtime.open_k, aged_k), dim=0)
            all_vp = torch.cat((runtime.open_v_packed, vp), dim=0)
            all_vmn = torch.cat((runtime.open_v_min, vmn), dim=0)
            all_vsc = torch.cat((runtime.open_v_scale, vsc), dim=0)

        block_size = int(runtime.block_size)
        total = int(all_k.shape[0])
        full_pages = total // block_size
        leftover = total % block_size

        if full_pages:
            pages = []
            for page_idx in range(full_pages):
                s = page_idx * block_size
                e = s + block_size
                pages.append(
                    _rabit2_stage4b4_encode_page_from_primary(
                        runtime,
                        all_k[s:e],
                        all_vp[s:e],
                        all_vmn[s:e],
                        all_vsc[s:e],
                    )
                )
            pages_tensor = torch.stack(pages, dim=0)
            start = int(runtime.closed_pages)
            end = start + full_pages
            if end > int(block_table_row.shape[0]):
                raise RuntimeError(
                    "RABIT-2 block table is too short for Stage4B4 closed pages"
                )
            physical_ids = block_table_row[start:end].to(torch.long)
            cache2d = kv_cache.reshape(kv_cache.shape[0], -1)
            cache2d.index_copy_(0, physical_ids, pages_tensor)
            runtime.closed_pages = end

        if leftover:
            s = full_pages * block_size
            runtime.open_k = all_k[s:].contiguous()
            runtime.open_v_packed = all_vp[s:].contiguous()
            runtime.open_v_min = all_vmn[s:].contiguous()
            runtime.open_v_scale = all_vsc[s:].contiguous()
        else:
            runtime.open_k = None
            runtime.open_v_packed = None
            runtime.open_v_min = None
            runtime.open_v_scale = None

        runtime.recent_k = combined_recent_k[aged_n:].contiguous()
        runtime.recent_v = combined_recent_v[aged_n:].contiguous()
    else:
        runtime.recent_k = combined_recent_k.contiguous()
        runtime.recent_v = combined_recent_v.contiguous()


class Rabit2CausalChunkPlan:
    """Precompute exact future chunk state, expose only its causal prefix."""

    def __init__(
        self,
        runtime: Rabit2SingleSequenceRuntime,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table_row: torch.Tensor,
    ) -> None:
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("key/value must share [tokens, kv_heads, head_dim]")
        if key.shape[1:] != (runtime.num_kv_heads, runtime.head_size_k):
            raise ValueError("key shape does not match RABIT-2 runtime")
        if value.shape[-1] != runtime.head_size_v:
            raise ValueError("value head dimension does not match RABIT-2 runtime")
        if key.shape[0] <= 0:
            raise ValueError("RABIT-2 causal chunk plan requires at least one token")

        self.runtime = runtime
        self.kv_cache = kv_cache
        self.block_table_row = block_table_row
        self.num_tokens = int(key.shape[0])

        key = key.to(torch.bfloat16).contiguous()
        value = value.to(torch.bfloat16).contiguous()

        self.init_closed = int(runtime.closed_pages)
        self.init_open_len = (
            0 if runtime.open_k is None else int(runtime.open_k.shape[0])
        )
        self.init_recent_len = (
            0 if runtime.recent_k is None else int(runtime.recent_k.shape[0])
        )

        empty_k = torch.empty(
            (0, runtime.num_kv_heads, runtime.head_size_k),
            dtype=torch.bfloat16,
            device=key.device,
        )
        empty_v = torch.empty(
            (0, runtime.num_kv_heads, runtime.head_size_v),
            dtype=torch.bfloat16,
            device=value.device,
        )

        old_recent_k = empty_k if runtime.recent_k is None else runtime.recent_k
        old_recent_v = empty_v if runtime.recent_v is None else runtime.recent_v
        self.combined_recent_k = torch.cat((old_recent_k, key), dim=0)
        self.combined_recent_v = torch.cat((old_recent_v, value), dim=0)

        aged_total = max(
            0,
            int(self.combined_recent_k.shape[0]) - int(runtime.residual_tokens),
        )

        if aged_total:
            aged_k = self.combined_recent_k[:aged_total].contiguous()
            aged_v = self.combined_recent_v[:aged_total].contiguous()
            vp, vmn, vsc, _ = _rabit2_stage4b4_exact_v2_batch(aged_v)
        else:
            groups = (
                runtime.head_size_v + RABIT2_GROUP_SIZE - 1
            ) // RABIT2_GROUP_SIZE
            aged_k = empty_k
            vp = torch.empty(
                (
                    0,
                    runtime.num_kv_heads,
                    rabit2_packed_dim(runtime.head_size_v, 2),
                ),
                dtype=torch.uint8,
                device=value.device,
            )
            vmn = torch.empty(
                (0, runtime.num_kv_heads, groups, 1),
                dtype=torch.float32,
                device=value.device,
            )
            vsc = torch.empty_like(vmn)

        if runtime.open_k is None:
            self.all_k, self.all_vp = aged_k, vp
            self.all_vmn, self.all_vsc = vmn, vsc
        else:
            self.all_k = torch.cat((runtime.open_k, aged_k), dim=0)
            self.all_vp = torch.cat((runtime.open_v_packed, vp), dim=0)
            self.all_vmn = torch.cat((runtime.open_v_min, vmn), dim=0)
            self.all_vsc = torch.cat((runtime.open_v_scale, vsc), dim=0)

        block_size = int(runtime.block_size)
        full_pages = int(self.all_k.shape[0]) // block_size
        if full_pages:
            pages = []
            for page_idx in range(full_pages):
                s = page_idx * block_size
                e = s + block_size
                pages.append(
                    _rabit2_stage4b4_encode_page_from_primary(
                        runtime,
                        self.all_k[s:e],
                        self.all_vp[s:e],
                        self.all_vmn[s:e],
                        self.all_vsc[s:e],
                    )
                )
            pages_tensor = torch.stack(pages, dim=0)
            start = self.init_closed
            end = start + full_pages
            if end > int(block_table_row.shape[0]):
                raise RuntimeError(
                    "RABIT-2 block table is too short for Stage4B4 chunk plan"
                )
            physical_ids = block_table_row[start:end].to(torch.long)
            kv_cache.reshape(kv_cache.shape[0], -1).index_copy_(
                0, physical_ids, pages_tensor
            )

        self._last_step = -1

    def apply_step(self, step: int) -> None:
        """Expose exact runtime state AFTER consuming chunk token `step`."""
        step = int(step)
        if step != self._last_step + 1:
            raise RuntimeError(
                "RABIT-2 causal chunk plan must be applied sequentially"
            )
        if step < 0 or step >= self.num_tokens:
            raise IndexError("RABIT-2 causal chunk step out of range")

        processed = step + 1
        aged_now = max(
            0,
            self.init_recent_len
            + processed
            - int(self.runtime.residual_tokens),
        )

        open_total_now = self.init_open_len + aged_now
        block_size = int(self.runtime.block_size)
        closed_new = open_total_now // block_size
        open_start = closed_new * block_size
        open_end = open_total_now

        self.runtime.closed_pages = self.init_closed + closed_new

        if open_end > open_start:
            self.runtime.open_k = self.all_k[open_start:open_end]
            self.runtime.open_v_packed = self.all_vp[open_start:open_end]
            self.runtime.open_v_min = self.all_vmn[open_start:open_end]
            self.runtime.open_v_scale = self.all_vsc[open_start:open_end]
        else:
            self.runtime.open_k = None
            self.runtime.open_v_packed = None
            self.runtime.open_v_min = None
            self.runtime.open_v_scale = None

        recent_start = aged_now
        recent_end = self.init_recent_len + processed
        if recent_end > recent_start:
            self.runtime.recent_k = self.combined_recent_k[
                recent_start:recent_end
            ]
            self.runtime.recent_v = self.combined_recent_v[
                recent_start:recent_end
            ]
        else:
            self.runtime.recent_k = None
            self.runtime.recent_v = None

        self._last_step = step
# === RABIT2_STAGE4B4_CAUSAL_PREFILL_END ===

# === RABIT2_STAGE4D2_VECTORIZED_WRITER_BEGIN ===
# Exact vectorized physical-page writer.
#
# Stage4D1 showed that the Stage4B4 initial-prefill writer spent essentially all
# of its time in a Python loop that encoded one 32-token page at a time.
# Stage4D2 preserves the exact RABIT-2 physical byte layout while vectorizing
# K3 quantization, INT3 packing, META8g64 encoding, and page assembly across all
# closed pages in a chunk.
_RABIT2_STAGE4D2_VECTORIZED_WRITER = True

_rabit2_stage4d2_old_bulk_append_exact = rabit2_bulk_append_exact
_rabit2_stage4d2_old_chunkplan_init = Rabit2CausalChunkPlan.__init__


def _rabit2_stage4d2_meta_blob_pages_exact(
    data: torch.Tensor,
) -> torch.Tensor:
    """Vectorized page-local META8g64 physical blobs.

    The first dimension is pages. Every page is flattened and second-level
    grouped independently, exactly matching encode_metadata_uint8_group_ref()
    followed by encode_metadata_blob_ref() on each page.
    """
    if data.ndim < 2:
        raise ValueError("Stage4D2 metadata input must have a page dimension")
    pages = int(data.shape[0])
    flat = data.float().reshape(pages, -1)
    count = int(flat.shape[1])
    pad = (-count) % int(RABIT2_METADATA_GROUP_SIZE)
    if pad:
        flat = torch.cat(
            (flat, flat[:, -1:].expand(pages, pad)),
            dim=1,
        )
    grouped = flat.reshape(
        pages, -1, int(RABIT2_METADATA_GROUP_SIZE)
    )
    meta_min = grouped.amin(dim=-1, keepdim=True)
    meta_max = grouped.amax(dim=-1, keepdim=True)
    meta_scale = (meta_max - meta_min) / 255.0
    meta_scale = torch.where(
        meta_scale.abs() < 1e-12,
        torch.ones_like(meta_scale),
        meta_scale,
    )
    codes = torch.round(
        (grouped - meta_min) / meta_scale
    ).clamp(0, 255).to(torch.uint8)

    min_bytes = (
        meta_min.to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .reshape(pages, -1)
    )
    scale_bytes = (
        meta_scale.to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .reshape(pages, -1)
    )
    return torch.cat(
        (codes.reshape(pages, -1), min_bytes, scale_bytes),
        dim=1,
    ).contiguous()


def _rabit2_stage4d2_pack_int3_direct_exact(
    codes: torch.Tensor,
) -> torch.Tensor:
    """Exact little-endian INT3 packing without scatter_add_.

    Eight 3-bit values map to three bytes. This is byte-identical to the
    reference pack_int3_values() for head dimensions divisible by eight.
    """
    value_count = int(codes.shape[-1])
    if value_count % 8 != 0:
        return pack_int3_values(codes)

    c = codes.to(torch.uint8).reshape(
        *codes.shape[:-1], value_count // 8, 8
    )
    c0 = c[..., 0]
    c1 = c[..., 1]
    c2 = c[..., 2]
    c3 = c[..., 3]
    c4 = c[..., 4]
    c5 = c[..., 5]
    c6 = c[..., 6]
    c7 = c[..., 7]

    b0 = c0 | (c1 << 3) | ((c2 & 0x03) << 6)
    b1 = (
        (c2 >> 2)
        | (c3 << 1)
        | (c4 << 4)
        | ((c5 & 0x01) << 7)
    )
    b2 = (c5 >> 1) | (c6 << 2) | (c7 << 5)

    return torch.stack((b0, b1, b2), dim=-1).reshape(
        *codes.shape[:-1], value_count * 3 // 8
    ).contiguous()


def _rabit2_stage4d2_encode_pages_vectorized_exact(
    runtime: Rabit2SingleSequenceRuntime,
    key_pages: torch.Tensor,
    v_packed_pages: torch.Tensor,
    v_min_pages: torch.Tensor,
    v_scale_pages: torch.Tensor,
) -> torch.Tensor:
    """Encode [pages, block, heads, dim] into exact opaque physical pages."""
    if key_pages.ndim != 4:
        raise ValueError("Stage4D2 key_pages must be [pages, block, heads, dim]")

    pages, block, heads, head_dim = map(int, key_pages.shape)
    if block != int(runtime.block_size):
        raise ValueError("Stage4D2 page block size does not match runtime")
    if heads != int(runtime.num_kv_heads):
        raise ValueError("Stage4D2 page head count does not match runtime")
    if head_dim != int(runtime.head_size_k):
        raise ValueError("Stage4D2 K head dimension does not match runtime")
    if block != int(RABIT2_GROUP_SIZE):
        raise ValueError(
            "Stage4D2 vectorized writer requires physical block == K G32"
        )

    # K3 sequence-axis affine G32. A 32-token physical page is one exact
    # sequence group, so pages can be quantized independently in one tensor.
    x = key_pages.float()
    k_min = x.amin(dim=1)
    k_max = x.amax(dim=1)
    k_scale = (k_max - k_min) / 7.0
    k_scale = torch.where(
        k_scale.abs() < 1e-8,
        torch.ones_like(k_scale),
        k_scale,
    )
    k_codes = torch.round(
        (x - k_min[:, None, :, :])
        / k_scale[:, None, :, :]
    ).clamp(0, 7).to(torch.uint8)
    k_payload = _rabit2_stage4d2_pack_int3_direct_exact(
        k_codes
    ).reshape(pages, -1)

    v_payload = v_packed_pages.reshape(pages, -1).to(torch.uint8)
    k_min_blob = _rabit2_stage4d2_meta_blob_pages_exact(k_min)
    k_scale_blob = _rabit2_stage4d2_meta_blob_pages_exact(k_scale)
    v_min_blob = _rabit2_stage4d2_meta_blob_pages_exact(v_min_pages)
    v_scale_blob = _rabit2_stage4d2_meta_blob_pages_exact(v_scale_pages)

    out = torch.zeros(
        (pages, runtime.layout.page_bytes),
        dtype=torch.uint8,
        device=key_pages.device,
    )

    for blob, offset, size, name in (
        (
            k_payload,
            runtime.layout.k_payload_offset,
            runtime.layout.k_payload_bytes,
            "k_payload",
        ),
        (
            v_payload,
            runtime.layout.v_payload_offset,
            runtime.layout.v_payload_bytes,
            "v_payload",
        ),
        (
            k_min_blob,
            runtime.layout.k_min_offset,
            runtime.layout.k_min_bytes,
            "k_min",
        ),
        (
            k_scale_blob,
            runtime.layout.k_scale_offset,
            runtime.layout.k_scale_bytes,
            "k_scale",
        ),
        (
            v_min_blob,
            runtime.layout.v_min_offset,
            runtime.layout.v_min_bytes,
            "v_min",
        ),
        (
            v_scale_blob,
            runtime.layout.v_scale_offset,
            runtime.layout.v_scale_bytes,
            "v_scale",
        ),
    ):
        if int(blob.shape[1]) != int(size):
            raise RuntimeError(
                f"RABIT-2 Stage4D2 {name} size mismatch: "
                f"{blob.shape[1]} != {size}"
            )
        out[:, offset : offset + size].copy_(blob)

    return out


def _rabit2_stage4d2_write_full_pages(
    runtime: Rabit2SingleSequenceRuntime,
    all_k: torch.Tensor,
    all_vp: torch.Tensor,
    all_vmn: torch.Tensor,
    all_vsc: torch.Tensor,
    full_pages: int,
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    *,
    start_closed: int,
) -> int:
    """Vector-encode and scatter `full_pages` to scheduler-owned blocks."""
    full_pages = int(full_pages)
    if full_pages <= 0:
        return int(start_closed)

    block = int(runtime.block_size)
    closed_n = full_pages * block

    pages_tensor = _rabit2_stage4d2_encode_pages_vectorized_exact(
        runtime,
        all_k[:closed_n].reshape(
            full_pages,
            block,
            runtime.num_kv_heads,
            runtime.head_size_k,
        ),
        all_vp[:closed_n].reshape(
            full_pages, block, *all_vp.shape[1:]
        ),
        all_vmn[:closed_n].reshape(
            full_pages, block, *all_vmn.shape[1:]
        ),
        all_vsc[:closed_n].reshape(
            full_pages, block, *all_vsc.shape[1:]
        ),
    )

    start = int(start_closed)
    end = start + full_pages
    if end > int(block_table_row.shape[0]):
        raise RuntimeError(
            "RABIT-2 block table is too short for Stage4D2 closed pages"
        )
    physical_ids = block_table_row[start:end].to(torch.long)
    kv_cache.reshape(kv_cache.shape[0], -1).index_copy_(
        0, physical_ids, pages_tensor
    )
    return end


def rabit2_bulk_append_exact(
    runtime: Rabit2SingleSequenceRuntime,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
) -> None:
    """Stage4D2 exact bulk append with vectorized physical-page writing."""
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key/value must share [tokens, kv_heads, head_dim]")
    if key.shape[1:] != (runtime.num_kv_heads, runtime.head_size_k):
        raise ValueError("key shape does not match RABIT-2 runtime")
    if value.shape[-1] != runtime.head_size_v:
        raise ValueError("value head dimension does not match RABIT-2 runtime")
    if key.shape[0] == 0:
        return

    # Keep the old exact path as a compatibility fallback for unexpected
    # layouts. The production Llama-3.1 RABIT layout is block32/G32.
    if int(runtime.block_size) != int(RABIT2_GROUP_SIZE):
        return _rabit2_stage4d2_old_bulk_append_exact(
            runtime, key, value, kv_cache, block_table_row
        )

    key = key.to(torch.bfloat16).contiguous()
    value = value.to(torch.bfloat16).contiguous()

    if runtime.recent_k is None:
        combined_recent_k = key
        combined_recent_v = value
    else:
        combined_recent_k = torch.cat((runtime.recent_k, key), dim=0)
        combined_recent_v = torch.cat((runtime.recent_v, value), dim=0)

    aged_n = max(
        0,
        int(combined_recent_k.shape[0]) - int(runtime.residual_tokens),
    )

    if aged_n:
        aged_k = combined_recent_k[:aged_n].contiguous()
        aged_v = combined_recent_v[:aged_n].contiguous()
        vp, vmn, vsc, _ = _rabit2_stage4b4_exact_v2_batch(aged_v)

        if runtime.open_k is None:
            all_k, all_vp, all_vmn, all_vsc = aged_k, vp, vmn, vsc
        else:
            all_k = torch.cat((runtime.open_k, aged_k), dim=0)
            all_vp = torch.cat((runtime.open_v_packed, vp), dim=0)
            all_vmn = torch.cat((runtime.open_v_min, vmn), dim=0)
            all_vsc = torch.cat((runtime.open_v_scale, vsc), dim=0)

        block = int(runtime.block_size)
        total = int(all_k.shape[0])
        full_pages = total // block
        leftover = total % block

        if full_pages:
            runtime.closed_pages = _rabit2_stage4d2_write_full_pages(
                runtime,
                all_k,
                all_vp,
                all_vmn,
                all_vsc,
                full_pages,
                kv_cache,
                block_table_row,
                start_closed=int(runtime.closed_pages),
            )

        if leftover:
            s = full_pages * block
            runtime.open_k = all_k[s:].contiguous()
            runtime.open_v_packed = all_vp[s:].contiguous()
            runtime.open_v_min = all_vmn[s:].contiguous()
            runtime.open_v_scale = all_vsc[s:].contiguous()
        else:
            runtime.open_k = None
            runtime.open_v_packed = None
            runtime.open_v_min = None
            runtime.open_v_scale = None

        runtime.recent_k = combined_recent_k[aged_n:].contiguous()
        runtime.recent_v = combined_recent_v[aged_n:].contiguous()
    else:
        runtime.recent_k = combined_recent_k.contiguous()
        runtime.recent_v = combined_recent_v.contiguous()


def _rabit2_stage4d2_chunkplan_init(
    self,
    runtime: Rabit2SingleSequenceRuntime,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
) -> None:
    """Stage4B4 causal plan with Stage4D2 vectorized future-page encoding."""
    if int(runtime.block_size) != int(RABIT2_GROUP_SIZE):
        return _rabit2_stage4d2_old_chunkplan_init(
            self, runtime, key, value, kv_cache, block_table_row
        )

    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key/value must share [tokens, kv_heads, head_dim]")
    if key.shape[1:] != (runtime.num_kv_heads, runtime.head_size_k):
        raise ValueError("key shape does not match RABIT-2 runtime")
    if value.shape[-1] != runtime.head_size_v:
        raise ValueError("value head dimension does not match RABIT-2 runtime")
    if key.shape[0] <= 0:
        raise ValueError("RABIT-2 causal chunk plan requires at least one token")

    self.runtime = runtime
    self.kv_cache = kv_cache
    self.block_table_row = block_table_row
    self.num_tokens = int(key.shape[0])

    key = key.to(torch.bfloat16).contiguous()
    value = value.to(torch.bfloat16).contiguous()

    self.init_closed = int(runtime.closed_pages)
    self.init_open_len = (
        0 if runtime.open_k is None else int(runtime.open_k.shape[0])
    )
    self.init_recent_len = (
        0 if runtime.recent_k is None else int(runtime.recent_k.shape[0])
    )

    empty_k = torch.empty(
        (0, runtime.num_kv_heads, runtime.head_size_k),
        dtype=torch.bfloat16,
        device=key.device,
    )
    empty_v = torch.empty(
        (0, runtime.num_kv_heads, runtime.head_size_v),
        dtype=torch.bfloat16,
        device=value.device,
    )

    old_recent_k = empty_k if runtime.recent_k is None else runtime.recent_k
    old_recent_v = empty_v if runtime.recent_v is None else runtime.recent_v
    self.combined_recent_k = torch.cat((old_recent_k, key), dim=0)
    self.combined_recent_v = torch.cat((old_recent_v, value), dim=0)

    aged_total = max(
        0,
        int(self.combined_recent_k.shape[0]) - int(runtime.residual_tokens),
    )

    if aged_total:
        aged_k = self.combined_recent_k[:aged_total].contiguous()
        aged_v = self.combined_recent_v[:aged_total].contiguous()
        vp, vmn, vsc, _ = _rabit2_stage4b4_exact_v2_batch(aged_v)
    else:
        groups = (
            runtime.head_size_v + RABIT2_GROUP_SIZE - 1
        ) // RABIT2_GROUP_SIZE
        aged_k = empty_k
        vp = torch.empty(
            (
                0,
                runtime.num_kv_heads,
                rabit2_packed_dim(runtime.head_size_v, 2),
            ),
            dtype=torch.uint8,
            device=value.device,
        )
        vmn = torch.empty(
            (0, runtime.num_kv_heads, groups, 1),
            dtype=torch.float32,
            device=value.device,
        )
        vsc = torch.empty_like(vmn)

    if runtime.open_k is None:
        self.all_k, self.all_vp = aged_k, vp
        self.all_vmn, self.all_vsc = vmn, vsc
    else:
        self.all_k = torch.cat((runtime.open_k, aged_k), dim=0)
        self.all_vp = torch.cat((runtime.open_v_packed, vp), dim=0)
        self.all_vmn = torch.cat((runtime.open_v_min, vmn), dim=0)
        self.all_vsc = torch.cat((runtime.open_v_scale, vsc), dim=0)

    block = int(runtime.block_size)
    full_pages = int(self.all_k.shape[0]) // block
    if full_pages:
        _rabit2_stage4d2_write_full_pages(
            runtime,
            self.all_k,
            self.all_vp,
            self.all_vmn,
            self.all_vsc,
            full_pages,
            kv_cache,
            block_table_row,
            start_closed=self.init_closed,
        )

    self._last_step = -1


Rabit2CausalChunkPlan.__init__ = _rabit2_stage4d2_chunkplan_init
# === RABIT2_STAGE4D2_VECTORIZED_WRITER_END ===
