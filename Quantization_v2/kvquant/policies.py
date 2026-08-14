from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from typing import Callable

import numpy as np

from kvquant.packing import (
    FullPrecisionArray,
    MixedPrecisionQuantizedArray,
    QuantizedKVBlock,
    ResidualSignQuantizedArray,
    RotatedQuantizedArray,
    UniformQuantizedArray,
)
from kvquant.types import QuantizationContext


class KVQuantPolicy(ABC):
    """量化算法实验的基类。

    每个policy拥有算法决策权：比特位宽、轴粒度、异常值处理、attention感知精度分配。
    后端只负责如何使用生成的块进行质量或延迟测量。
    """

    name: str

    @abstractmethod
    def quantize(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        context: QuantizationContext,
    ) -> QuantizedKVBlock:
        raise NotImplementedError

    @property
    def requires_attention(self) -> bool:
        return False


@dataclass
class NoQuantPolicy(KVQuantPolicy):
    name: str = "no_quant"

    def quantize(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        context: QuantizationContext,
    ) -> QuantizedKVBlock:
        key = FullPrecisionArray.from_float(keys)
        value = FullPrecisionArray.from_float(values)
        return QuantizedKVBlock(key=key, value=value, layer_idx=context.layer_idx, metadata={"policy": self.name})


@dataclass
class NaiveMinMaxPolicy(KVQuantPolicy):
    nbits: int = 4
    quantize_keys: bool = True
    quantize_values: bool = True
    name: str = "naive"

    def quantize(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        context: QuantizationContext,
    ) -> QuantizedKVBlock:
        return _uniform_kv_block(
            keys,
            values,
            context,
            self.nbits,
            axis_strategy="global",
            quantize_keys=self.quantize_keys,
            quantize_values=self.quantize_values,
            policy_name=self.name,
        )


@dataclass
class DocumentNaiveMinMaxPolicy(NaiveMinMaxPolicy):
    """PDF基线：每个K张量一个全局uniform min-max scale，每个V张量一个。"""

    name: str = "document_naive"


@dataclass
class PerHeadPolicy(KVQuantPolicy):
    nbits: int = 4
    quantize_keys: bool = True
    quantize_values: bool = True
    name: str = "per_head"

    def quantize(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        context: QuantizationContext,
    ) -> QuantizedKVBlock:
        return _uniform_kv_block(
            keys,
            values,
            context,
            self.nbits,
            axis_strategy="per_head",
            quantize_keys=self.quantize_keys,
            quantize_values=self.quantize_values,
            policy_name=self.name,
        )


@dataclass
class PerChannelPolicy(KVQuantPolicy):
    nbits: int = 4
    per_head: bool = True
    quantize_keys: bool = True
    quantize_values: bool = True
    name: str = "per_channel"

    def quantize(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        context: QuantizationContext,
    ) -> QuantizedKVBlock:
        axis = "per_head_channel" if self.per_head else "per_channel"
        return _uniform_kv_block(
            keys,
            values,
            context,
            self.nbits,
            axis_strategy=axis,
            quantize_keys=self.quantize_keys,
            quantize_values=self.quantize_values,
            policy_name=self.name,
        )


@dataclass
class GroupQuantPolicy(KVQuantPolicy):
    nbits: int = 4
    group_size: int = 64
    axis_strategy: str = "per_head"
    name: str = "group"

    def quantize(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        context: QuantizationContext,
    ) -> QuantizedKVBlock:
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        key = _quantize_grouped(keys, self.nbits, self.group_size, self.axis_strategy)
        value = _quantize_grouped(values, self.nbits, self.group_size, self.axis_strategy)
        return QuantizedKVBlock(
            key=key,
            value=value,
            layer_idx=context.layer_idx,
            metadata={"policy": self.name, "group_size": self.group_size},
        )


@dataclass
class OutlierResidualPolicy(KVQuantPolicy):
    nbits: int = 4
    axis_strategy: str = "per_head"
    clip_percentile: float = 99.9
    name: str = "outlier_residual"

    def quantize(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        context: QuantizationContext,
    ) -> QuantizedKVBlock:
        key = UniformQuantizedArray.from_float(
            keys,
            self.nbits,
            axis_strategy=self.axis_strategy,
            clip_percentile=self.clip_percentile,
            keep_outliers=True,
        )
        value = UniformQuantizedArray.from_float(
            values,
            self.nbits,
            axis_strategy=self.axis_strategy,
            clip_percentile=self.clip_percentile,
            keep_outliers=True,
        )
        return QuantizedKVBlock(
            key=key,
            value=value,
            layer_idx=context.layer_idx,
            metadata={
                "policy": self.name,
                "clip_percentile": self.clip_percentile,
                "axis_strategy": self.axis_strategy,
            },
        )


@dataclass
class AttentionAwareMixedPrecisionPolicy(KVQuantPolicy):
    low_bits: int = 4
    high_bits: int = 8
    keep_ratio: float = 0.05
    axis_strategy: str = "per_head"
    fallback_recent_tokens: int = 16
    name: str = "attention_mixed"

    @property
    def requires_attention(self) -> bool:
        return True

    def quantize(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        context: QuantizationContext,
    ) -> QuantizedKVBlock:
        high_mask = self._importance_mask(keys, context)
        key = MixedPrecisionQuantizedArray(
            low=UniformQuantizedArray.from_float(keys, self.low_bits, self.axis_strategy),
            high=UniformQuantizedArray.from_float(keys, self.high_bits, self.axis_strategy),
            high_mask=high_mask,
            low_bits=self.low_bits,
            high_bits=self.high_bits,
            metadata={"keep_ratio": self.keep_ratio},
        )
        value = MixedPrecisionQuantizedArray(
            low=UniformQuantizedArray.from_float(values, self.low_bits, self.axis_strategy),
            high=UniformQuantizedArray.from_float(values, self.high_bits, self.axis_strategy),
            high_mask=high_mask,
            low_bits=self.low_bits,
            high_bits=self.high_bits,
            metadata={"keep_ratio": self.keep_ratio},
        )
        return QuantizedKVBlock(
            key=key,
            value=value,
            layer_idx=context.layer_idx,
            metadata={
                "policy": self.name,
                "low_bits": self.low_bits,
                "high_bits": self.high_bits,
                "keep_ratio": self.keep_ratio,
                "high_token_fraction": float(np.mean(high_mask)) if high_mask.size else 0.0,
            },
        )

    def _importance_mask(self, keys: np.ndarray, context: QuantizationContext) -> np.ndarray:
        if keys.ndim != 4:
            raise ValueError("attention-aware policy expects [batch, heads, seq, head_dim]")
        batch, heads, seq, _ = keys.shape
        scores = context.attention_scores

        if scores is None:
            importance = np.zeros((batch, heads, seq), dtype=np.float32)
            recent = min(self.fallback_recent_tokens, seq)
            importance[:, :, seq - recent :] = 1.0
        else:
            importance = _attention_to_importance(scores, batch, heads, seq)

        keep = max(1, int(round(seq * self.keep_ratio)))
        mask = np.zeros((batch, heads, seq), dtype=bool)
        for b in range(batch):
            for h in range(heads):
                order = np.argsort(importance[b, h])
                mask[b, h, order[-keep:]] = True
        return mask[:, :, :, None]


@dataclass
class KVQuantInt3Policy(KVQuantPolicy):
    """First deploy-oriented INT3 baseline for the vLLM fork path.

    The default scale granularity is per token and KV head, matching the first
    vLLM integration target. It is still a uniform quantizer, not TurboQuant.
    """

    nbits: int = 3
    axis_strategy: str = "per_token_head"
    quantize_keys: bool = True
    quantize_values: bool = True
    name: str = "kvquant_int3"

    def quantize(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        context: QuantizationContext,
    ) -> QuantizedKVBlock:
        return _uniform_kv_block(
            keys,
            values,
            context,
            self.nbits,
            axis_strategy=self.axis_strategy,
            quantize_keys=self.quantize_keys,
            quantize_values=self.quantize_values,
            policy_name=self.name,
        )


@dataclass
class PolarInt3Policy(KVQuantPolicy):
    """PolarQuant-style INT3 research scaffold.

    This applies a deterministic signed Hadamard rotation before uniform INT3
    quantization and reverses it during dequantization. It is a safe local
    approximation for algorithm studies, not a complete TurboQuant replica.
    """

    nbits: int = 3
    axis_strategy: str = "per_token_head"
    block_size: int = 64
    seed: int = 0
    name: str = "polar_int3"

    def quantize(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        context: QuantizationContext,
    ) -> QuantizedKVBlock:
        metadata = {
            "policy": self.name,
            "nbits": self.nbits,
            "axis_strategy": self.axis_strategy,
            "block_size": self.block_size,
            "seed": self.seed,
            "algorithm_stage": "polar_int3_research_scaffold",
        }
        key = RotatedQuantizedArray.from_float(
            keys,
            nbits=self.nbits,
            axis_strategy=self.axis_strategy,
            block_size=self.block_size,
            seed=self.seed,
            metadata=metadata,
        )
        value = RotatedQuantizedArray.from_float(
            values,
            nbits=self.nbits,
            axis_strategy=self.axis_strategy,
            block_size=self.block_size,
            seed=self.seed + 1,
            metadata=metadata,
        )
        return QuantizedKVBlock(key=key, value=value, layer_idx=context.layer_idx, metadata=metadata)


@dataclass
class TurboInt3Policy(KVQuantPolicy):
    """TurboQuant-like INT3 scaffold: rotated INT3 plus 1-bit residual signs.

    The residual is computed per-element (no axis reduction in the sign channel)
    so that each scalar gets its own +/- correction.  This matches the QJL
    engineering sketch more closely than a per-token-head residual scale.
    """

    nbits: int = 3
    axis_strategy: str = "per_token_head"
    residual_axis_strategy: str = "per_element"
    block_size: int = 64
    seed: int = 0
    name: str = "turbo_int3"

    def quantize(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        context: QuantizationContext,
    ) -> QuantizedKVBlock:
        metadata = {
            "policy": self.name,
            "nbits": self.nbits,
            "axis_strategy": self.axis_strategy,
            "residual_axis_strategy": self.residual_axis_strategy,
            "block_size": self.block_size,
            "seed": self.seed,
            "algorithm_stage": "turbo_int3_qjl_residual_scaffold",
            "claim_boundary": "not_full_turboquant_reproduction",
        }
        key_base = RotatedQuantizedArray.from_float(
            keys,
            nbits=self.nbits,
            axis_strategy=self.axis_strategy,
            block_size=self.block_size,
            seed=self.seed,
            metadata=metadata,
        )
        value_base = RotatedQuantizedArray.from_float(
            values,
            nbits=self.nbits,
            axis_strategy=self.axis_strategy,
            block_size=self.block_size,
            seed=self.seed + 1,
            metadata=metadata,
        )
        key = ResidualSignQuantizedArray.from_base(
            keys,
            key_base,
            axis_strategy=self.residual_axis_strategy,
            metadata={"qjl_residual": True},
        )
        value = ResidualSignQuantizedArray.from_base(
            values,
            value_base,
            axis_strategy=self.residual_axis_strategy,
            metadata={"qjl_residual": True},
        )
        return QuantizedKVBlock(key=key, value=value, layer_idx=context.layer_idx, metadata=metadata)


def _uniform_kv_block(
    keys: np.ndarray,
    values: np.ndarray,
    context: QuantizationContext,
    nbits: int,
    axis_strategy: str,
    quantize_keys: bool,
    quantize_values: bool,
    policy_name: str,
) -> QuantizedKVBlock:
    key = (
        UniformQuantizedArray.from_float(keys, nbits, axis_strategy)
        if quantize_keys
        else NoQuantPolicy().quantize(keys, keys, context).key
    )
    value = (
        UniformQuantizedArray.from_float(values, nbits, axis_strategy)
        if quantize_values
        else NoQuantPolicy().quantize(values, values, context).value
    )
    return QuantizedKVBlock(
        key=key,
        value=value,
        layer_idx=context.layer_idx,
        metadata={
            "policy": policy_name,
            "nbits": nbits,
            "axis_strategy": axis_strategy,
            "quantize_keys": quantize_keys,
            "quantize_values": quantize_values,
        },
    )


def _quantize_grouped(
    array: np.ndarray,
    nbits: int,
    group_size: int,
    axis_strategy: str,
) -> UniformQuantizedArray:
    values = np.asarray(array)
    if values.ndim != 4:
        raise ValueError("group quantization expects [batch, heads, seq, head_dim]")

    qvalues = np.zeros(values.shape, dtype=np.uint8)
    minimum = np.zeros(values.shape[:2] + (values.shape[2], 1), dtype=np.float32)
    scale = np.ones_like(minimum, dtype=np.float32)

    for start in range(0, values.shape[2], group_size):
        end = min(start + group_size, values.shape[2])
        group = UniformQuantizedArray.from_float(values[:, :, start:end, :], nbits, axis_strategy)
        qvalues[:, :, start:end, :] = group.qvalues
        minimum[:, :, start:end, :] = group.minimum
        scale[:, :, start:end, :] = group.scale

    return UniformQuantizedArray(
        qvalues=qvalues,
        minimum=minimum,
        scale=scale,
        nbits=nbits,
        original_shape=tuple(values.shape),
        original_dtype=str(values.dtype),
        axis_strategy=f"group_{axis_strategy}",
        metadata={"group_size": group_size},
    )


def _attention_to_importance(
    scores: np.ndarray,
    batch: int,
    heads: int,
    seq: int,
) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.ndim == 4:
        # [batch, heads, query, key]
        arr = arr.mean(axis=2)
    elif arr.ndim == 3:
        # [batch, heads, key]
        pass
    elif arr.ndim == 2:
        # [batch, key]
        arr = np.repeat(arr[:, None, :], heads, axis=1)
    elif arr.ndim == 1:
        arr = np.broadcast_to(arr[None, None, :], (batch, heads, arr.shape[0]))
    else:
        raise ValueError("unsupported attention score shape")

    if arr.shape[-1] != seq:
        raise ValueError(f"attention key dimension {arr.shape[-1]} does not match seq {seq}")
    if arr.shape[0] != batch:
        arr = np.broadcast_to(arr, (batch, arr.shape[1], seq))
    if arr.shape[1] != heads:
        arr = np.broadcast_to(arr, (batch, heads, seq))
    return arr


PolicyFactory = Callable[..., KVQuantPolicy]


_POLICIES: dict[str, PolicyFactory] = {
    "document_naive": DocumentNaiveMinMaxPolicy,
    "no_quant": NoQuantPolicy,
    "naive": NaiveMinMaxPolicy,
    "per_head": PerHeadPolicy,
    "per_channel": PerChannelPolicy,
    "group": GroupQuantPolicy,
    "kvquant_int3": KVQuantInt3Policy,
    "outlier_residual": OutlierResidualPolicy,
    "polar_int3": PolarInt3Policy,
    "attention_mixed": AttentionAwareMixedPrecisionPolicy,
    "turbo_int3": TurboInt3Policy,
}


def list_policies() -> list[str]:
    return sorted(_POLICIES)


def build_policy(name: str, **kwargs: object) -> KVQuantPolicy:
    try:
        factory = _POLICIES[name]
    except KeyError as exc:
        known = ", ".join(list_policies())
        raise ValueError(f"unknown policy {name!r}; known policies: {known}") from exc
    return factory(**kwargs)


def policy_spec(policy: KVQuantPolicy) -> dict[str, object]:
    """返回一个policy配置的稳定、JSON友好的描述。"""
    params = asdict(policy) if is_dataclass(policy) else {"name": policy.name}
    return {
        "class": type(policy).__name__,
        "name": policy.name,
        "parameters": _jsonable(params),
    }


def policy_nbits(policy: KVQuantPolicy) -> int | None:
    """Return the primary bit width for tabular experiment metadata."""
    params = policy_spec(policy)["parameters"]
    if not isinstance(params, dict):
        return None
    nbits = params.get("nbits")
    return int(nbits) if isinstance(nbits, int) else None


def policy_quantization_scope(policy: KVQuantPolicy) -> str:
    """Return whether a policy quantizes K, V, both, or neither."""
    if policy.name == "no_quant":
        return "none"
    params = policy_spec(policy)["parameters"]
    if not isinstance(params, dict):
        return "K+V"
    quantize_keys = bool(params.get("quantize_keys", True))
    quantize_values = bool(params.get("quantize_values", True))
    if quantize_keys and quantize_values:
        return "K+V"
    if quantize_keys:
        return "K-only"
    if quantize_values:
        return "V-only"
    return "none"


def policy_signature(policy: KVQuantPolicy) -> str:
    payload = json.dumps(policy_spec(policy), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
