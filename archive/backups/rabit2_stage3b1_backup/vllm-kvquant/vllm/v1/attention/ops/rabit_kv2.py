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
