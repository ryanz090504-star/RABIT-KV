# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    is_quantized_kv_cache as torch_utils_is_quantized_kv_cache,
    kv_cache_uses_per_token_head_scales as torch_utils_uses_pth_scales,
)
from vllm.v1.attention.ops.kvquant_k3 import (
    dequantize_int3_per_token_head_ref,
    pack_int3_values,
    quantize_int3_per_token_head_ref,
    reshape_and_cache_kvquant_k3,
    unified_attention_kvquant_k3,
    unpack_int3_values,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
    is_quantized_kv_cache,
    kv_cache_uses_per_token_head_scales,
    kvquant_k3_packed_dim,
)


def test_kvquant_k3_dtype_mapping() -> None:
    assert STR_DTYPE_TO_TORCH_DTYPE["kvquant_k3"] is torch.uint8
    assert torch_utils_is_quantized_kv_cache("kvquant_k3")
    assert is_quantized_kv_cache("kvquant_k3")
    assert torch_utils_uses_pth_scales("kvquant_k3")
    assert kv_cache_uses_per_token_head_scales("kvquant_k3")
    assert get_kv_quant_mode("kvquant_k3") == KVQuantMode.INT3_PER_TOKEN_HEAD


@pytest.mark.parametrize("value_count", [1, 2, 3, 7, 8, 9, 16, 17])
def test_pack_unpack_int3_roundtrip(value_count: int) -> None:
    values = (torch.arange(2 * value_count, dtype=torch.uint8) % 8).reshape(
        2, value_count
    )

    packed = pack_int3_values(values)
    unpacked = unpack_int3_values(packed, value_count)

    assert packed.dtype == torch.uint8
    assert packed.shape == (2, math.ceil(value_count * 3 / 8))
    assert torch.equal(unpacked, values)


def test_pack_int3_rejects_out_of_range_codes() -> None:
    with pytest.raises(ValueError, match=r"\[0, 7\]"):
        pack_int3_values(torch.tensor([0, 8], dtype=torch.uint8))


def test_kvquant_k3_page_size_bytes() -> None:
    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.INT3_PER_TOKEN_HEAD,
    )

    packed_dim = kvquant_k3_packed_dim(64)
    real_bytes = 2 * 16 * 4 * packed_dim
    scale_bytes = 2 * 16 * 4 * 4

    assert packed_dim == 24
    assert spec.real_page_size_bytes == real_bytes
    assert spec.page_size_bytes == real_bytes + scale_bytes


def test_kvquant_k3_full_attention_page_size_bytes() -> None:
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=64,
        head_size_v=80,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.INT3_PER_TOKEN_HEAD,
    )

    real_bytes = 16 * 4 * (kvquant_k3_packed_dim(64) + kvquant_k3_packed_dim(80))
    scale_bytes = 2 * 16 * 4 * 4

    assert spec.real_page_size_bytes == real_bytes
    assert spec.page_size_bytes == real_bytes + scale_bytes


def test_triton_backend_kvquant_k3_shape() -> None:
    pytest.importorskip("cbor2", reason="Triton backend import needs vLLM deps")

    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    shape = TritonAttentionBackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=16,
        num_kv_heads=4,
        head_size=64,
        cache_dtype_str="kvquant_k3",
    )

    assert "kvquant_k3" in TritonAttentionBackend.supported_kv_cache_dtypes
    assert shape == (3, 2, 16, 4, kvquant_k3_packed_dim(64) + 4)


def test_reference_cache_write_uses_packed_area_and_scales() -> None:
    key = torch.tensor(
        [
            [[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, 2.0, -2.0]],
            [[0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 0.0, 0.5, -0.5]],
        ],
        dtype=torch.float16,
    )
    value = key * 0.5
    block_size = 16
    packed_dim = kvquant_k3_packed_dim(key.shape[-1])
    padded_dim = packed_dim + 4

    key_cache = torch.zeros(1, block_size, 1, padded_dim, dtype=torch.uint8)
    value_cache = torch.zeros_like(key_cache)
    k_scale_cache = torch.ones(1, block_size, 1, dtype=torch.float32)
    v_scale_cache = torch.ones_like(k_scale_cache)
    slot_mapping = torch.tensor([3, 7], dtype=torch.long)

    reshape_and_cache_kvquant_k3(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
    )

    ref_k_packed, ref_k_scales = quantize_int3_per_token_head_ref(key)
    ref_v_packed, ref_v_scales = quantize_int3_per_token_head_ref(value)

    for token_idx, slot in enumerate(slot_mapping.tolist()):
        assert torch.equal(key_cache[0, slot, :, :packed_dim], ref_k_packed[token_idx])
        assert torch.equal(
            value_cache[0, slot, :, :packed_dim], ref_v_packed[token_idx]
        )
        assert torch.equal(
            key_cache[0, slot, :, packed_dim:],
            torch.zeros(1, 4, dtype=torch.uint8),
        )
        assert torch.allclose(k_scale_cache[0, slot], ref_k_scales[token_idx])
        assert torch.allclose(v_scale_cache[0, slot], ref_v_scales[token_idx])

    dequant = dequantize_int3_per_token_head_ref(
        key_cache[0, 3, :, :packed_dim],
        k_scale_cache[0, 3],
        key.shape[-1],
        dtype=key.dtype,
    )
    assert dequant.shape == key[0].shape


def test_attention_read_path_kernel_launches() -> None:
    """Smoke test: the INT3 attention kernel computes output without crashing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    pytest.importorskip("cbor2", reason="Triton backend import needs vLLM deps")

    device = torch.device("cuda")
    num_blocks = 4
    block_size = 16
    num_kv_heads = 2
    num_query_heads = 4
    head_size = 64
    num_tokens = 2
    packed_dim = kvquant_k3_packed_dim(head_size)
    padded_dim = packed_dim + 4

    k_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        padded_dim,
        dtype=torch.uint8,
        device=device,
    )
    v_cache = torch.zeros_like(k_cache)
    k_scale_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, dtype=torch.float32, device=device
    )
    v_scale_cache = torch.zeros_like(k_scale_cache)

    # Write two tokens with known float values.
    key = torch.randn(
        num_tokens, num_kv_heads, head_size, dtype=torch.float16, device=device
    )
    value = torch.randn(
        num_tokens, num_kv_heads, head_size, dtype=torch.float16, device=device
    )
    slot_mapping = torch.tensor([0, 1], dtype=torch.long, device=device)

    reshape_and_cache_kvquant_k3(
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
    )

    q = torch.randn(1, num_query_heads, head_size, dtype=torch.float16, device=device)
    out = torch.zeros(1, num_query_heads, head_size, dtype=torch.float16, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([2], dtype=torch.int32, device=device)
    block_table = torch.tensor([[0]], dtype=torch.int32, device=device)

    # Sanity: kernel runs without error and produces non-NaN output.
    unified_attention_kvquant_k3(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=2,
        softmax_scale=1.0 / (head_size**0.5),
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        sinks=None,
        alibi_slopes=None,
        use_alibi_sqrt=False,
        qq_bias=None,
        output_scale=None,
        mm_prefix_range=None,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
    )
    assert not torch.isnan(out).any(), "INT3 attention output contains NaN"
    assert not torch.isinf(out).any(), "INT3 attention output contains Inf"
