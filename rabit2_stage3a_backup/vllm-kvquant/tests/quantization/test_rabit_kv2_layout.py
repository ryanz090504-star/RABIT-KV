# SPDX-License-Identifier: Apache-2.0
"""Stage-2 allocator tests for the exact RABIT-KV K3/V2 page format."""

import math

import pytest
import torch

from vllm.config.cache import CacheConfig
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    is_quantized_kv_cache as torch_utils_is_quantized_kv_cache,
    kv_cache_uses_per_token_head_scales as torch_utils_uses_pth_scales,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVQuantMode,
    RABIT2_GROUP_SIZE,
    get_kv_quant_mode,
    is_quantized_kv_cache,
    kv_cache_uses_per_token_head_scales,
    rabit2_effective_compression_with_staging,
    rabit2_encoded_metadata_bytes,
    rabit2_exact_online_staging_bytes_per_sequence,
    rabit2_page_layout,
    rabit2_packed_dim,
    rabit2_residual_bytes_per_sequence,
)


def test_rabit_kv2_dtype_and_mode_mapping() -> None:
    assert STR_DTYPE_TO_TORCH_DTYPE["rabit_kv2"] is torch.uint8
    assert torch_utils_is_quantized_kv_cache("rabit_kv2")
    assert is_quantized_kv_cache("rabit_kv2")
    assert not torch_utils_uses_pth_scales("rabit_kv2")
    assert not kv_cache_uses_per_token_head_scales("rabit_kv2")
    assert get_kv_quant_mode("rabit_kv2") == KVQuantMode.RABIT_KV2
    assert get_kv_quant_mode("rabit_kv2").is_rabit_kv2


def test_lowbit_and_metadata_byte_formulas() -> None:
    assert rabit2_packed_dim(128, 3) == 48
    assert rabit2_packed_dim(128, 2) == 32
    assert rabit2_packed_dim(80, 2) == 20
    # 1024 UINT8 primary values + 16 BF16 min/scale pairs.
    assert rabit2_encoded_metadata_bytes(1024) == 1024 + 16 * 4


def test_llama31_8b_page_layout_exact_offsets_and_bytes() -> None:
    layout = rabit2_page_layout(
        block_size=32,
        num_kv_heads=8,
        head_size_k=128,
        head_size_v=128,
    )

    assert layout.k_payload_offset == 0
    assert layout.k_payload_bytes == 32 * 8 * 48
    assert layout.v_payload_offset == layout.k_payload_bytes
    assert layout.v_payload_bytes == 32 * 8 * 32

    # K has one 32-token sequence group per page. K min/scale each contain
    # 8 heads * 128 channels = 1024 primary metadata values.
    assert layout.k_min_bytes == 1088
    assert layout.k_scale_bytes == 1088
    # V has 32 tokens * 8 heads * 4 dimension groups = 1024 values.
    assert layout.v_min_bytes == 1088
    assert layout.v_scale_bytes == 1088

    assert layout.payload_bytes == 20480
    assert layout.metadata_bytes == 4352
    assert layout.page_bytes == 24832
    assert layout.v_scale_offset + layout.v_scale_bytes == layout.page_bytes


def test_llama31_8b_physical_compression_and_exact_online_staging() -> None:
    layout = rabit2_page_layout(
        block_size=32,
        num_kv_heads=8,
        head_size_k=128,
        head_size_v=128,
    )
    bf16_page_bytes = 2 * 32 * 8 * 128 * 2
    assert bf16_page_bytes == 131072
    assert math.isclose(bf16_page_bytes / layout.page_bytes, 5.278, rel_tol=1e-3)

    residual_bytes = rabit2_residual_bytes_per_sequence(
        num_kv_heads=8,
        head_size_k=128,
        head_size_v=128,
    )
    assert residual_bytes == 16384

    # Exact online K sequence-group updates need original BF16 K for up to
    # 31 open-group tokens in addition to R4 K/V.
    staging_bytes = rabit2_exact_online_staging_bytes_per_sequence(
        num_kv_heads=8,
        head_size_k=128,
        head_size_v=128,
    )
    assert staging_bytes == 79872

    assert math.isclose(
        rabit2_effective_compression_with_staging(
            context_tokens=1024, block_size=32, num_kv_heads=8,
            head_size_k=128, head_size_v=128,
        ),
        4.7963,
        rel_tol=1e-3,
    )
    assert math.isclose(
        rabit2_effective_compression_with_staging(
            context_tokens=16384, block_size=32, num_kv_heads=8,
            head_size_k=128, head_size_v=128,
        ),
        5.2456,
        rel_tol=1e-3,
    )


def test_attention_specs_budget_exact_rabit_page_bytes() -> None:
    expected = rabit2_page_layout(
        block_size=32,
        num_kv_heads=8,
        head_size_k=128,
        head_size_v=128,
    ).page_bytes
    spec = AttentionSpec(
        block_size=32,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.RABIT_KV2,
    )
    full_spec = FullAttentionSpec(
        block_size=32,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.RABIT_KV2,
    )
    assert spec.real_page_size_bytes == expected
    assert spec.page_size_bytes == expected
    assert full_spec.real_page_size_bytes == expected
    assert full_spec.page_size_bytes == expected


def test_triton_backend_uses_opaque_exact_page_shape() -> None:
    pytest.importorskip("cbor2", reason="Triton backend import needs vLLM deps")
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    layout = rabit2_page_layout(
        block_size=32,
        num_kv_heads=8,
        head_size_k=128,
        head_size_v=128,
    )
    shape = TritonAttentionBackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=32,
        num_kv_heads=8,
        head_size=128,
        cache_dtype_str="rabit_kv2",
    )
    assert "rabit_kv2" in TritonAttentionBackend.supported_kv_cache_dtypes
    assert shape == (3, 1, 1, 1, layout.page_bytes)
    assert math.prod(shape[1:]) == layout.page_bytes


def test_rabit_kv2_rejects_block_size_16() -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        rabit2_page_layout(
            block_size=16,
            num_kv_heads=8,
            head_size_k=128,
            head_size_v=128,
        )
    with pytest.raises(ValueError, match="multiple of 32"):
        CacheConfig(cache_dtype="rabit_kv2", block_size=16)


def test_rabit_kv2_accepts_block_size_32() -> None:
    config = CacheConfig(cache_dtype="rabit_kv2", block_size=RABIT2_GROUP_SIZE)
    assert config.block_size == 32
