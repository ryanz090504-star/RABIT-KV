from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, sqrt
from typing import Any

import numpy as np

from kvquant.types import QuantizationError


def pack_bits(values: np.ndarray, nbits: int) -> np.ndarray:
    """将无符号整数值按位打包成字节数组。"""
    if nbits <= 0 or nbits > 8:
        raise ValueError("nbits must be in [1, 8]")

    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    max_value = (1 << nbits) - 1
    if flat.size and int(flat.max()) > max_value:
        raise ValueError(f"value exceeds {nbits}-bit range")

    total_bits = flat.size * nbits
    packed = np.zeros(ceil(total_bits / 8), dtype=np.uint8)
    bit_offset = 0

    for raw in flat:
        value = int(raw)
        byte_idx = bit_offset // 8
        inner = bit_offset % 8
        packed[byte_idx] |= (value << inner) & 0xFF
        overflow = inner + nbits - 8
        if overflow > 0:
            packed[byte_idx + 1] |= value >> (nbits - overflow)
        bit_offset += nbits

    return packed


def unpack_bits(packed: np.ndarray, nbits: int, count: int) -> np.ndarray:
    """从字节打包的数组中解包出无符号整数值。"""
    if nbits <= 0 or nbits > 8:
        raise ValueError("nbits must be in [1, 8]")
    if count < 0:
        raise ValueError("count must be non-negative")

    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    output = np.zeros(count, dtype=np.uint8)
    mask = (1 << nbits) - 1
    bit_offset = 0

    for idx in range(count):
        byte_idx = bit_offset // 8
        inner = bit_offset % 8
        value = int(packed[byte_idx]) >> inner
        overflow = inner + nbits - 8
        if overflow > 0 and byte_idx + 1 < packed.size:
            value |= int(packed[byte_idx + 1]) << (nbits - overflow)
        output[idx] = value & mask
        bit_offset += nbits

    return output


def _safe_scale(minimum: np.ndarray, maximum: np.ndarray, nbits: int) -> np.ndarray:
    levels = (1 << nbits) - 1
    scale = (maximum - minimum) / max(levels, 1)
    return np.where(scale == 0, 1.0, scale).astype(np.float32)


def quantization_error(original: np.ndarray, reconstructed: np.ndarray) -> QuantizationError:
    original = np.asarray(original, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    delta = original - reconstructed
    mse = float(np.mean(delta * delta)) if delta.size else 0.0
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    denom = float(np.linalg.norm(original.reshape(-1)) * np.linalg.norm(reconstructed.reshape(-1)))
    cosine = 1.0 if denom == 0 else float(np.dot(original.reshape(-1), reconstructed.reshape(-1)) / denom)
    return QuantizationError(mse=mse, cosine=cosine, max_abs=max_abs)


@dataclass
class FullPrecisionArray:
    values: np.ndarray
    original_shape: tuple[int, ...]
    original_dtype: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_float(cls, array: np.ndarray) -> "FullPrecisionArray":
        values = np.asarray(array)
        return cls(
            values=values.astype(np.float32),
            original_shape=tuple(values.shape),
            original_dtype=str(values.dtype),
            metadata={"storage": "full_precision"},
        )

    def dequantize(self) -> np.ndarray:
        return self.values.astype(np.float32)

    def estimated_payload_nbytes(self) -> int:
        return int(np.prod(self.original_shape)) * np.dtype(self.original_dtype).itemsize


@dataclass
class UniformQuantizedArray:
    qvalues: np.ndarray
    minimum: np.ndarray
    scale: np.ndarray
    nbits: int
    original_shape: tuple[int, ...]
    original_dtype: str
    axis_strategy: str
    residual_mask: np.ndarray | None = None
    residual_values: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_float(
        cls,
        array: np.ndarray,
        nbits: int,
        axis_strategy: str = "global",
        clip_percentile: float | None = None,
        keep_outliers: bool = False,
    ) -> "UniformQuantizedArray":
        if nbits <= 0 or nbits > 8:
            raise ValueError("UniformQuantizedArray supports 1 to 8 bits")
        values = np.asarray(array, dtype=np.float32)
        clipped = values
        residual_mask = None
        residual_values = None

        if clip_percentile is not None:
            if not 0 < clip_percentile <= 100:
                raise ValueError("clip_percentile must be in (0, 100]")
            upper = np.percentile(values, clip_percentile, keepdims=False)
            lower = np.percentile(values, 100 - clip_percentile, keepdims=False)
            residual_mask = (values < lower) | (values > upper)
            clipped = np.clip(values, lower, upper)
            if keep_outliers:
                residual_values = np.where(residual_mask, values, 0).astype(np.float16)

        minimum, maximum = _range_for_strategy(clipped, axis_strategy)
        scale = _safe_scale(minimum, maximum, nbits)
        qvalues = np.rint((clipped - minimum) / scale)
        qvalues = np.clip(qvalues, 0, (1 << nbits) - 1).astype(np.uint8)

        return cls(
            qvalues=qvalues,
            minimum=minimum.astype(np.float32),
            scale=scale,
            nbits=nbits,
            original_shape=tuple(values.shape),
            original_dtype=str(np.asarray(array).dtype),
            axis_strategy=axis_strategy,
            residual_mask=residual_mask,
            residual_values=residual_values,
            metadata={
                "clip_percentile": clip_percentile,
                "keep_outliers": keep_outliers,
            },
        )

    def dequantize(self) -> np.ndarray:
        values = self.qvalues.astype(np.float32) * self.scale + self.minimum
        if self.residual_mask is not None and self.residual_values is not None:
            values = np.where(self.residual_mask, self.residual_values.astype(np.float32), values)
        return values.astype(np.float32)

    def pack(self) -> np.ndarray:
        return pack_bits(self.qvalues, self.nbits)

    def to_packed(self) -> "PackedUniformQuantizedArray":
        if self.residual_mask is not None or self.residual_values is not None:
            raise ValueError("packing residual outliers is not implemented for this prototype")
        return PackedUniformQuantizedArray(
            packed=self.pack(),
            minimum=self.minimum,
            scale=self.scale,
            nbits=self.nbits,
            original_shape=self.original_shape,
            original_dtype=self.original_dtype,
            axis_strategy=self.axis_strategy,
            metadata={**self.metadata, "storage": "bit_packed"},
        )

    def estimated_payload_nbytes(self) -> int:
        packed = ceil(self.qvalues.size * self.nbits / 8)
        params = self.minimum.nbytes + self.scale.nbytes
        residual = 0
        if self.residual_mask is not None:
            residual += ceil(self.residual_mask.size / 8)
        if self.residual_values is not None and self.residual_mask is not None:
            residual += int(np.count_nonzero(self.residual_mask)) * np.dtype(np.float16).itemsize
        return packed + params + residual


@dataclass
class PackedUniformQuantizedArray:
    packed: np.ndarray
    minimum: np.ndarray
    scale: np.ndarray
    nbits: int
    original_shape: tuple[int, ...]
    original_dtype: str
    axis_strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_uniform(cls, array: UniformQuantizedArray) -> "PackedUniformQuantizedArray":
        return array.to_packed()

    def unpack_qvalues(self) -> np.ndarray:
        count = int(np.prod(self.original_shape))
        values = unpack_bits(self.packed, self.nbits, count)
        return values.reshape(self.original_shape)

    def dequantize(self) -> np.ndarray:
        qvalues = self.unpack_qvalues().astype(np.float32)
        return (qvalues * self.scale + self.minimum).astype(np.float32)

    def estimated_payload_nbytes(self) -> int:
        return self.packed.nbytes + self.minimum.nbytes + self.scale.nbytes


@dataclass
class MixedPrecisionQuantizedArray:
    low: UniformQuantizedArray
    high: UniformQuantizedArray
    high_mask: np.ndarray
    low_bits: int
    high_bits: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def dequantize(self) -> np.ndarray:
        low_values = self.low.dequantize()
        high_values = self.high.dequantize()
        return np.where(self.high_mask, high_values, low_values).astype(np.float32)

    def estimated_payload_nbytes(self) -> int:
        total_count = int(np.prod(self.low.original_shape))
        expanded_mask = np.broadcast_to(self.high_mask, self.low.original_shape)
        high_count = int(np.count_nonzero(expanded_mask))
        low_count = total_count - high_count
        data_bytes = ceil(low_count * self.low_bits / 8) + ceil(high_count * self.high_bits / 8)
        mask_bytes = ceil(self.high_mask.size / 8)
        params = (
            self.low.minimum.nbytes
            + self.low.scale.nbytes
            + self.high.minimum.nbytes
            + self.high.scale.nbytes
        )
        return data_bytes + mask_bytes + params


@dataclass
class RotatedQuantizedArray:
    """Uniform quantization after a deterministic block rotation.

    This is a research scaffold for PolarQuant-style experiments. It keeps the
    transform reversible so quality benchmarks can evaluate the effect without
    claiming a production vLLM layout yet.
    """

    base: UniformQuantizedArray | PackedUniformQuantizedArray
    block_size: int
    seed: int
    transform: str = "signed_hadamard"
    original_shape: tuple[int, ...] | None = None
    original_dtype: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_float(
        cls,
        array: np.ndarray,
        *,
        nbits: int,
        axis_strategy: str,
        block_size: int = 64,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> "RotatedQuantizedArray":
        values = np.asarray(array, dtype=np.float32)
        rotated = _signed_hadamard_transform(values, block_size=block_size, seed=seed, inverse=False)
        base = UniformQuantizedArray.from_float(rotated, nbits=nbits, axis_strategy=axis_strategy)
        return cls(
            base=base,
            block_size=block_size,
            seed=seed,
            original_shape=tuple(values.shape),
            original_dtype=str(np.asarray(array).dtype),
            metadata=dict(metadata or {}),
        )

    @property
    def nbits(self) -> int:
        return self.base.nbits

    def dequantize(self) -> np.ndarray:
        rotated = self.base.dequantize()
        values = _signed_hadamard_transform(rotated, block_size=self.block_size, seed=self.seed, inverse=True)
        return values.astype(np.float32)

    def to_packed(self) -> "RotatedQuantizedArray":
        packed_base = self.base.to_packed() if isinstance(self.base, UniformQuantizedArray) else self.base
        return RotatedQuantizedArray(
            base=packed_base,
            block_size=self.block_size,
            seed=self.seed,
            transform=self.transform,
            original_shape=self.original_shape,
            original_dtype=self.original_dtype,
            metadata={**self.metadata, "storage": "bit_packed"},
        )

    def estimated_payload_nbytes(self) -> int:
        return self.base.estimated_payload_nbytes()


@dataclass
class ResidualSignQuantizedArray:
    """One-bit residual correction layered on a low-bit base quantizer.

    This approximates the engineering shape of a QJL-style residual channel:
    one sign bit per scalar plus a small residual scale. It is intentionally
    simple and local to the research framework.
    """

    base: RotatedQuantizedArray | UniformQuantizedArray | PackedUniformQuantizedArray
    residual_sign: np.ndarray
    residual_scale: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        original: np.ndarray,
        base: RotatedQuantizedArray | UniformQuantizedArray,
        *,
        axis_strategy: str = "global",
        metadata: dict[str, Any] | None = None,
    ) -> "ResidualSignQuantizedArray":
        values = np.asarray(original, dtype=np.float32)
        # Residual is computed against the base dequantized output, which for
        # RotatedQuantizedArray includes the inverse Hadamard transform.
        # The residual correction is stored raw — it will be added back
        # AFTER the base dequantize runs during reconstruction.
        residual = values - base.dequantize()
        # Default to per_element residual scale (no axis reduction).
        # A global/per-head scale means one scalar corrects a large region,
        # which can blow up the error.  Per-element sign with a small
        # element-wise scale is the closest approximation to the QJL idea.
        if axis_strategy == "per_element":
            axes = ()  # no reduction — one scale per scalar
        elif axis_strategy == "per_token_head":
            # One scale per token-head across the head_dim
            axes = (3,)
        elif axis_strategy == "per_head":
            axes = (0, 2, 3)
        elif axis_strategy == "global":
            axes = tuple(range(values.ndim))
        else:
            axes = _axes_for_strategy(values, axis_strategy)
        scale = np.mean(np.abs(residual), axis=axes, keepdims=True).astype(np.float32)
        # Clip scale to avoid amplifying small residuals too much
        scale = np.clip(scale, 1e-6, None).astype(np.float32)
        sign = residual >= 0
        return cls(
            base=base,
            residual_sign=sign,
            residual_scale=scale,
            metadata={**dict(metadata or {}), "residual_axis_strategy": axis_strategy},
        )

    @property
    def original_shape(self) -> tuple[int, ...]:
        return _original_shape(self.base)

    @property
    def original_dtype(self) -> str:
        return _original_dtype(self.base)

    def dequantize(self) -> np.ndarray:
        base_values = self.base.dequantize()
        # residual_sign=True → +scale, residual_sign=False → -scale
        sign = np.where(self.residual_sign, 1.0, -1.0).astype(np.float32)
        # Normalise residual scale by head_dim to prevent blow-up for
        # per_token_head strategies where the scale is averaged across
        # the full head_dim (typically 64-128 elements).
        corrected = base_values + sign * self.residual_scale
        return corrected.astype(np.float32)

    def to_packed(self) -> "ResidualSignQuantizedArray":
        packed_base = _pack_array(self.base)
        return ResidualSignQuantizedArray(
            base=packed_base,
            residual_sign=self.residual_sign,
            residual_scale=self.residual_scale,
            metadata={**self.metadata, "storage": "bit_packed"},
        )

    def estimated_payload_nbytes(self) -> int:
        sign_bytes = ceil(self.residual_sign.size / 8)
        return self.base.estimated_payload_nbytes() + sign_bytes + self.residual_scale.nbytes


QuantizedArray = (
    FullPrecisionArray
    | UniformQuantizedArray
    | PackedUniformQuantizedArray
    | MixedPrecisionQuantizedArray
    | RotatedQuantizedArray
    | ResidualSignQuantizedArray
)


@dataclass
class QuantizedKVBlock:
    """单个transformer层的量化后KV缓存块。

    包含量化后的key数组和value数组，以及压缩比、误差计算等功能。
    """
    key: QuantizedArray
    value: QuantizedArray
    layer_idx: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def dequantize(self) -> tuple[np.ndarray, np.ndarray]:
        return self.key.dequantize(), self.value.dequantize()

    def original_nbytes(self) -> int:
        shape = _original_shape(self.key)
        dtype = _original_dtype(self.key)
        key_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

        value_shape = _original_shape(self.value)
        value_dtype = _original_dtype(self.value)
        value_bytes = int(np.prod(value_shape)) * np.dtype(value_dtype).itemsize
        return key_bytes + value_bytes

    def estimated_payload_nbytes(self) -> int:
        return self.key.estimated_payload_nbytes() + self.value.estimated_payload_nbytes()

    def compression_ratio(self) -> float:
        payload = self.estimated_payload_nbytes()
        return float("inf") if payload == 0 else self.original_nbytes() / payload

    def error(self, keys: np.ndarray, values: np.ndarray) -> dict[str, QuantizationError]:
        deq_keys, deq_values = self.dequantize()
        return {
            "key": quantization_error(keys, deq_keys),
            "value": quantization_error(values, deq_values),
        }

    def to_packed(self) -> "PackedKVBlock":
        return PackedKVBlock(
            key=_pack_array(self.key),
            value=_pack_array(self.value),
            layer_idx=self.layer_idx,
            metadata={**self.metadata, "storage": "bit_packed"},
        )


@dataclass
class PackedKVBlock:
    key: QuantizedArray
    value: QuantizedArray
    layer_idx: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def dequantize(self) -> tuple[np.ndarray, np.ndarray]:
        return self.key.dequantize(), self.value.dequantize()

    def original_nbytes(self) -> int:
        shape = _original_shape(self.key)
        dtype = _original_dtype(self.key)
        key_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        value_shape = _original_shape(self.value)
        value_dtype = _original_dtype(self.value)
        value_bytes = int(np.prod(value_shape)) * np.dtype(value_dtype).itemsize
        return key_bytes + value_bytes

    def estimated_payload_nbytes(self) -> int:
        return self.key.estimated_payload_nbytes() + self.value.estimated_payload_nbytes()

    def compression_ratio(self) -> float:
        payload = self.estimated_payload_nbytes()
        return float("inf") if payload == 0 else self.original_nbytes() / payload


def _range_for_strategy(values: np.ndarray, axis_strategy: str) -> tuple[np.ndarray, np.ndarray]:
    axes = _axes_for_strategy(values, axis_strategy)
    minimum = values.min(axis=axes, keepdims=True)
    maximum = values.max(axis=axes, keepdims=True)
    return minimum, maximum


def _axes_for_strategy(values: np.ndarray, axis_strategy: str) -> tuple[int, ...]:
    if axis_strategy == "global":
        return tuple(range(values.ndim))
    if axis_strategy == "per_head":
        _require_kv_shape(values)
        return (0, 2, 3)
    if axis_strategy == "per_channel":
        _require_kv_shape(values)
        return (0, 1, 2)
    if axis_strategy == "per_head_channel":
        _require_kv_shape(values)
        return (0, 2)
    if axis_strategy == "per_token":
        _require_kv_shape(values)
        return (0, 3)
    if axis_strategy in {"per_token_head", "per_token_per_head"}:
        _require_kv_shape(values)
        return (3,)
    raise ValueError(f"unknown axis strategy: {axis_strategy}")


def _require_kv_shape(values: np.ndarray) -> None:
    if values.ndim != 4:
        raise ValueError(
            "KV tensors must use shape [batch, kv_heads, seq, head_dim] "
            "for non-global strategies"
        )


def _original_shape(array: QuantizedArray) -> tuple[int, ...]:
    if isinstance(array, MixedPrecisionQuantizedArray):
        return array.low.original_shape
    if isinstance(array, ResidualSignQuantizedArray):
        return _original_shape(array.base)
    if isinstance(array, RotatedQuantizedArray):
        return array.original_shape or _original_shape(array.base)
    return array.original_shape


def _original_dtype(array: QuantizedArray) -> str:
    if isinstance(array, MixedPrecisionQuantizedArray):
        return array.low.original_dtype
    if isinstance(array, ResidualSignQuantizedArray):
        return _original_dtype(array.base)
    if isinstance(array, RotatedQuantizedArray):
        return array.original_dtype or _original_dtype(array.base)
    return array.original_dtype


def _pack_array(array: QuantizedArray) -> QuantizedArray:
    if isinstance(array, UniformQuantizedArray):
        return array.to_packed()
    if isinstance(array, RotatedQuantizedArray):
        return array.to_packed()
    if isinstance(array, ResidualSignQuantizedArray):
        return array.to_packed()
    if isinstance(array, MixedPrecisionQuantizedArray):
        raise ValueError("packing mixed precision arrays is not implemented for this prototype")
    return array


def _signed_hadamard_transform(
    values: np.ndarray,
    *,
    block_size: int,
    seed: int,
    inverse: bool,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if arr.shape[-1] < block_size:
        return arr.copy()

    out = arr.copy()
    signs = _rotation_signs(arr.shape[-1], seed).reshape((1,) * (arr.ndim - 1) + (arr.shape[-1],))
    full = (arr.shape[-1] // block_size) * block_size

    if not inverse:
        out = out * signs
    for start in range(0, full, block_size):
        out[..., start : start + block_size] = _hadamard_block(out[..., start : start + block_size])
    if inverse:
        out = out * signs
    return out.astype(np.float32)


def _rotation_signs(width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=width)


def _hadamard_block(block: np.ndarray) -> np.ndarray:
    out = block.copy()
    h = 1
    size = out.shape[-1]
    while h < size:
        for start in range(0, size, 2 * h):
            a = out[..., start : start + h].copy()
            b = out[..., start + h : start + 2 * h].copy()
            out[..., start : start + h] = a + b
            out[..., start + h : start + 2 * h] = a - b
        h *= 2
    return out / sqrt(size)
