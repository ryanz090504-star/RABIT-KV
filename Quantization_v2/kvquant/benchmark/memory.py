"""Memory analysis: KV cache storage estimates.

Analytical calculations of KV cache memory footprint under
different quantization policies and bit widths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any


@dataclass
class MemoryResult:
    """KV cache memory estimate for a given configuration."""

    model: str
    policy_name: str
    policy_signature: str
    nbits: int | None

    layers: int
    kv_heads: int
    head_dim: int
    batch_size: int
    seq_len: int
    dtype_bytes: int = 2  # fp16/bf16
    kv_cache_dtype: str | None = None
    kv_source: str = "analytical_estimate"
    hardware: str | None = "analytical"
    vllm_commit: str | None = None
    quant_kernel: str | None = None

    baseline_cache_gb: float = 0.0
    estimated_cache_gb: float = 0.0
    memory_reduction: float = 0.0
    compression_ratio: float = 0.0

    max_batch_under_budget: int | None = None
    max_seq_len_under_budget: int | None = None
    memory_budget_gb: float | None = None
    memory_breakdown: dict[str, int] = field(default_factory=dict)

    effective_bits_per_element: float | None = None
    """Effective bit width after accounting for all overhead (scales, codebook, etc.).
    Computed as 8 * total_estimated_bytes / total_elements."""

    evidence_label: str = "memory_estimate_not_runtime_peak"
    metadata: dict[str, Any] = field(default_factory=dict)


def estimate_memory(
    model: str,
    policy,
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    batch_size: int = 1,
    seq_len: int = 4096,
    dtype_bytes: int = 2,
    memory_budget_gb: float | None = None,
    attention_layers: int | None = None,
    block_size: int = 16,
    page_overhead_bytes: int = 16,
) -> MemoryResult:
    """Estimate KV cache memory for a given configuration.

    Parameters
    ----------
    model: model name
    policy: KVQuantPolicy instance
    layers: number of transformer layers
    kv_heads: number of KV heads (GQA)
    head_dim: head dimension
    batch_size: batch size
    seq_len: sequence length (tokens)
    dtype_bytes: bytes per element in fp16/bf16 (2)
    memory_budget_gb: optional GPU memory budget
    attention_layers: actual attention layers (for hybrid models)
    """
    effective_layers = attention_layers or layers
    from kvquant.policies import policy_signature, policy_spec

    # Baseline: 2 * layers * kv_heads * seq_len * head_dim * dtype_bytes
    baseline_bytes = batch_size * effective_layers * 2 * kv_heads * seq_len * head_dim * dtype_bytes
    baseline_gb = baseline_bytes / (1024**3)

    params = policy_spec(policy).get("parameters", {})
    params = params if isinstance(params, dict) else {}
    nbits = _policy_bits(policy, params)
    bits_per_elem = nbits if nbits < 16 else 16
    breakdown = _memory_breakdown_bytes(
        policy_name=policy.name,
        params=params,
        nbits=nbits,
        batch_size=batch_size,
        layers=effective_layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        dtype_bytes=dtype_bytes,
        block_size=block_size,
        page_overhead_bytes=page_overhead_bytes,
        baseline_bytes=baseline_bytes,
    )
    estimated_bytes = breakdown["total_estimated_bytes"]
    estimated_gb = estimated_bytes / (1024**3)

    memory_reduction = (baseline_gb - estimated_gb) / baseline_gb if baseline_gb > 0 else 0.0
    compression_ratio = baseline_gb / estimated_gb if estimated_gb > 0 else float("inf")

    # Effective bits: total storage bits divided by number of float16 elements
    total_elements = batch_size * effective_layers * 2 * kv_heads * seq_len * head_dim
    effective_bits = (estimated_bytes * 8) / max(total_elements, 1)

    max_batch = None
    max_seq = None
    if memory_budget_gb:
        budget_bytes = memory_budget_gb * 1024**3
        per_batch_bytes = estimated_bytes / max(batch_size, 1)
        per_token_bytes = estimated_bytes / max(batch_size * seq_len, 1)
        max_batch = int(budget_bytes / per_batch_bytes) if per_batch_bytes > 0 else None
        max_seq = int(budget_bytes / (per_token_bytes * batch_size)) if per_token_bytes > 0 else None

    return MemoryResult(
        model=model,
        policy_name=policy.name,
        policy_signature=policy_signature(policy),
        nbits=nbits,
        kv_cache_dtype=_kv_cache_dtype(nbits),
        quant_kernel=_quant_kernel(policy.name),
        layers=effective_layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        dtype_bytes=dtype_bytes,
        baseline_cache_gb=baseline_gb,
        estimated_cache_gb=estimated_gb,
        memory_reduction=memory_reduction,
        compression_ratio=compression_ratio,
        max_batch_under_budget=max_batch,
        max_seq_len_under_budget=max_seq,
        memory_budget_gb=memory_budget_gb,
        memory_breakdown=breakdown,
        effective_bits_per_element=effective_bits,
    )


def _policy_bits(policy: object, params: dict[str, Any]) -> int:
    if getattr(policy, "name", "") == "no_quant":
        return 16
    for key in ("nbits", "low_bits"):
        value = params.get(key, getattr(policy, key, None))
        if isinstance(value, int):
            return int(value)
    return 16


def _memory_breakdown_bytes(
    *,
    policy_name: str,
    params: dict[str, Any],
    nbits: int,
    batch_size: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int,
    block_size: int,
    page_overhead_bytes: int,
    baseline_bytes: int,
) -> dict[str, int]:
    elements = batch_size * layers * 2 * kv_heads * seq_len * head_dim
    if nbits >= 16 or policy_name == "no_quant":
        return {
            "baseline_bytes": int(baseline_bytes),
            "packed_kv_bytes": int(baseline_bytes),
            "scale_bytes": 0,
            "codebook_bytes": 0,
            "qjl_residual_bytes": 0,
            "mixed_precision_mask_bytes": 0,
            "allocator_page_overhead_bytes": 0,
            "total_estimated_bytes": int(baseline_bytes),
        }

    if policy_name == "attention_mixed":
        low_bits = int(params.get("low_bits", nbits) or nbits)
        high_bits = int(params.get("high_bits", 8) or 8)
        keep_ratio = float(params.get("keep_ratio", 0.05) or 0.05)
        high_elements = int(elements * keep_ratio)
        low_elements = elements - high_elements
        packed_kv_bytes = ceil(low_elements * low_bits / 8) + ceil(high_elements * high_bits / 8)
        mixed_mask_bytes = ceil(elements / max(head_dim, 1) / 8)
    else:
        packed_kv_bytes = ceil(elements * nbits / 8)
        mixed_mask_bytes = 0
    scale_bytes = _scale_bytes(policy_name, params, batch_size, layers, kv_heads, head_dim, seq_len, dtype_bytes)
    codebook_bytes = _codebook_bytes(policy_name, nbits, layers, kv_heads, dtype_bytes)
    qjl_bytes = _qjl_residual_bytes(policy_name, elements, scale_bytes)
    pages = batch_size * layers * 2 * kv_heads * ceil(seq_len / max(block_size, 1))
    overhead = pages * page_overhead_bytes
    total = packed_kv_bytes + scale_bytes + codebook_bytes + qjl_bytes + overhead + mixed_mask_bytes
    return {
        "baseline_bytes": int(baseline_bytes),
        "packed_kv_bytes": int(packed_kv_bytes),
        "scale_bytes": int(scale_bytes),
        "codebook_bytes": int(codebook_bytes),
        "qjl_residual_bytes": int(qjl_bytes),
        "mixed_precision_mask_bytes": int(mixed_mask_bytes),
        "allocator_page_overhead_bytes": int(overhead),
        "total_estimated_bytes": int(total),
    }


def _scale_bytes(
    policy_name: str,
    params: dict[str, Any],
    batch_size: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int,
) -> int:
    strategy = str(params.get("axis_strategy", "global"))
    if policy_name in {"kvquant_int3", "polar_int3", "turbo_int3"}:
        strategy = str(params.get("axis_strategy", "per_token_head"))
    if strategy == "global":
        count = layers * 2
    elif strategy == "per_head":
        count = layers * 2 * kv_heads
    elif strategy == "per_channel":
        count = layers * 2 * head_dim
    elif strategy == "per_head_channel":
        count = layers * 2 * kv_heads * head_dim
    elif strategy in {"per_token", "per_token_head", "per_token_per_head"}:
        count = batch_size * layers * 2 * kv_heads * seq_len
    elif strategy.startswith("group"):
        group_size = int(params.get("group_size", 64) or 64)
        count = batch_size * layers * 2 * kv_heads * ceil(seq_len / max(group_size, 1))
    else:
        count = layers * 2
    return int(count * dtype_bytes * 2)  # minimum + scale


def _codebook_bytes(policy_name: str, nbits: int, layers: int, kv_heads: int, dtype_bytes: int) -> int:
    if policy_name not in {"polar_int3", "turbo_int3"}:
        return 0
    return int(layers * 2 * kv_heads * (1 << nbits) * dtype_bytes)


def _qjl_residual_bytes(policy_name: str, elements: int, scale_bytes: int) -> int:
    if policy_name != "turbo_int3":
        return 0
    return int(ceil(elements / 8) + scale_bytes)


def _kv_cache_dtype(nbits: int) -> str | None:
    if nbits >= 16:
        return "fp16"
    return f"kvquant_k{nbits}"


def _quant_kernel(policy_name: str) -> str | None:
    if policy_name in {"kvquant_int3", "polar_int3", "turbo_int3"}:
        return "kvquant_int3_packed_decode"
    return None
